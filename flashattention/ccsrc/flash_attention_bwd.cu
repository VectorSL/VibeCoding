#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <vector>

// ===== FlashAttention v2 Backward Kernel =====

// Configuration for kernel launch - must match forward kernel
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_D = 64;
constexpr int THREADS = 256;
constexpr float NEG_INF = -1e10f;
//
// Backward pass computes gradients for Q, K, V using the same
// tiling strategy as forward. Key insight: we can reuse the
// forward computation with dO to compute gradients.

// Forward declaration
struct FlashAttentionFwdParams;

// Configuration for backward
struct FlashAttentionBwdParams {
    const half* Q;      // [B, H, N, D]
    const half* K;      // [B, H, M, D]
    const half* V;      // [B, H, M, D]
    const half* O;      // [B, H, N, D]
    const float* L;     // [B, H, N] - logsumexp from forward
    const half* dO;     // [B, H, N, D] - gradient of output
    half* dQ;           // [B, H, N, D]
    half* dK;           // [B, H, M, D]
    half* dV;           // [B, H, M, D]
    int B;              // batch size
    int H;              // num heads
    int N;              // sequence length Q
    int M;              // sequence length KV
    int D;              // head dimension
    float softmax_scale;
};

// Helper: maximum
__device__ __forceinline__ float max_f(float a, float b) {
    return a > b ? a : b;
}

// Backward kernel - simplified version for correctness
// Full optimization: tile-based with shared memory
__global__ void flash_attention_bwd_kernel(
    FlashAttentionBwdParams params
) {
    extern __shared__ half sram[];

    const int B = params.B;
    const int H = params.H;
    const int N = params.N;
    const int M = params.M;
    const int D = params.D;
    const float scale = params.softmax_scale;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int kv_block_idx = blockIdx.x;  // Process K,V blocks

    const int tid = threadIdx.x;
    const int num_threads = THREADS;

    // Shared memory layout
    // K: BLOCK_N * BLOCK_D
    // V: BLOCK_N * BLOCK_D
    // dK: BLOCK_N * BLOCK_D
    // dV: BLOCK_N * BLOCK_D
    half* sram_k = sram;
    half* sram_v = sram + BLOCK_N * BLOCK_D;
    half* sram_dk = sram + 2 * BLOCK_N * BLOCK_D;
    half* sram_dv = sram + 3 * BLOCK_N * BLOCK_D;

    // Initialize dK and dV for this block
    const int kv_start = kv_block_idx * BLOCK_N;
    const int kv_len = min(BLOCK_N, M - kv_start);

    const int k_batch_stride = H * M * D;
    const int k_base = batch_idx * k_batch_stride + head_idx * M * D;

    // Zero initialize dK and dV for this block
    for (int j = tid; j < kv_len * D; j += num_threads) {
        int kj = j / D;
        int dk = j % D;
        sram_dk[kj * D + dk] = __float2half(0.0f);
        sram_dv[kj * D + dk] = __float2half(0.0f);
    }
    __syncthreads();

    // Load K and V for this block
    for (int j = tid; j < kv_len * D; j += num_threads) {
        int kj = j / D;
        int dk = j % D;
        int kk = kv_start + kj;
        if (kk < M && dk < D) {
            sram_k[kj * D + dk] = params.K[k_base + kk * D + dk];
            sram_v[kj * D + dk] = params.V[k_base + kk * D + dk];
        }
    }
    __syncthreads();

    // Process all Q rows to accumulate gradients for this KV block
    const int num_q_blocks = (N + BLOCK_M - 1) / BLOCK_M;

    const int q_batch_stride = H * N * D;
    const int q_base = batch_idx * q_batch_stride + head_idx * N * D;

    for (int q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
        const int q_start = q_block_idx * BLOCK_M;
        const int q_len = min(BLOCK_M, N - q_start);

        // For each Q row in block
        for (int qi = 0; qi < q_len; qi++) {
            int q_row = q_start + qi;
            int q_offset = q_base + q_row * D;
            int l_offset = batch_idx * H * N + head_idx * N + q_row;

            // Load Q row
            half q_row_reg[BLOCK_D];
            for (int d = tid; d < D; d += num_threads) {
                q_row_reg[d] = params.Q[q_offset + d];
            }

            // Load L (logsumexp)
            float l_val;
            if (tid == 0) {
                l_val = params.L[l_offset];
            }
            l_val = __shfl_sync(0xffffffff, l_val, 0);

            // Compute attention weights: P = softmax(QK^T)
            // For this Q row against all K in current block
            float weights[BLOCK_N];
            float max_val = NEG_INF;

            // Compute QK^T scores
            for (int kj = 0; kj < kv_len; kj++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += __half2float(q_row_reg[d]) * __half2float(sram_k[kj * D + d]);
                }
                weights[kj] = dot * scale;
            }

            // Compute softmax: max, sum, normalized
            for (int kj = 0; kj < kv_len; kj++) {
                max_val = max_f(max_val, weights[kj]);
            }

            // Warp reduce max
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                max_val = max_f(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
            }
            max_val = __shfl_sync(0xffffffff, max_val, 0);

            // Compute normalized softmax
            float sum_exp = 0.0f;
            for (int kj = 0; kj < kv_len; kj++) {
                weights[kj] = __expf(weights[kj] - max_val);
                sum_exp += weights[kj];
            }

            // Normalize by l_val (logsumexp from forward)
            float l_correction = __expf(max_val - l_val);

            #pragma unroll
            for (int kj = 0; kj < kv_len; kj++) {
                weights[kj] = weights[kj] * l_correction / sum_exp;
            }

            // Load dO row
            half dO_row[BLOCK_D];
            for (int d = tid; d < D; d += num_threads) {
                dO_row[d] = params.dO[q_offset + d];
            }

            // Compute dV: dV += p * dO
            // p is the attention weight, dO is gradient flowing back
            #pragma unroll
            for (int kj = 0; kj < BLOCK_N; kj++) {
                if (kj < kv_len) {
                    float w = weights[kj];
                    #pragma unroll
                    for (int d = 0; d < BLOCK_D; d++) {
                        int dv = tid + d * num_threads;
                        if (dv < D) {
                            float v = __half2float(sram_dv[kj * D + dv]);
                            v += w * __half2float(dO_row[dv]);
                            sram_dv[kj * D + dv] = __float2half(v);
                        }
                    }
                }
            }
            __syncthreads();

            // Compute dQ_local: accumulate for this Q row
            // This will be written back after processing all KV blocks
        }
    }

    // Write dK and dV back to global memory
    for (int j = tid; j < kv_len * D; j += num_threads) {
        int kj = j / D;
        int dk = j % D;
        int kk = kv_start + kj;
        if (kk < M && dk < D) {
            params.dK[k_base + kk * D + dk] = sram_dk[kj * D + dk];
            params.dV[k_base + kk * D + dk] = sram_dv[kj * D + dk];
        }
    }
    __syncthreads();

    // Now process Q gradients - iterate over all Q blocks
    // and accumulate dQ from each KV block
    const int num_kv_blocks = (M + BLOCK_N - 1) / BLOCK_N;

    // Zero dQ
    for (int q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
        const int q_start = q_block_idx * BLOCK_M;
        const int q_len = min(BLOCK_M, N - q_start);

        for (int qi = tid; qi < q_len * D; qi += num_threads) {
            int q_row = qi / D;
            int dk = qi % D;
            int q_idx = q_start + q_row;
            if (q_idx < N) {
                params.dQ[q_base + q_idx * D + dk] = __float2half(0.0f);
            }
        }
    }
    __syncthreads();

    // Accumulate dQ from all KV blocks
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        // Reload K block
        const int k_start = kv_block_idx * BLOCK_N;
        const int k_len = min(BLOCK_N, M - k_start);

        for (int j = tid; j < k_len * D; j += num_threads) {
            int kj = j / D;
            int dk = j % D;
            sram_k[kj * D + dk] = params.K[k_base + (k_start + kj) * D + dk];
        }
        __syncthreads();

        // For each Q row
        for (int q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {
            const int q_start = q_block_idx * BLOCK_M;
            const int q_len = min(BLOCK_M, N - q_start);

            for (int qi = 0; qi < q_len; qi++) {
                int q_row = q_start + qi;
                int q_offset = q_base + q_row * D;
                int l_offset = batch_idx * H * N + head_idx * N + q_row;

                // Load Q row
                half q_row_reg[BLOCK_D];
                #pragma unroll
                for (int d = 0; d < BLOCK_D; d++) {
                    int dv = tid + d * num_threads;
                    if (dv < D) {
                        q_row_reg[d] = params.Q[q_offset + dv];
                    }
                }

                // Load dO row
                half dO_row[BLOCK_D];
                #pragma unroll
                for (int d = 0; d < BLOCK_D; d++) {
                    int dv = tid + d * num_threads;
                    if (dv < D) {
                        dO_row[d] = params.dO[q_offset + dv];
                    }
                }

                // Compute weights (softmax)
                float weights[BLOCK_N];
                float max_val = NEG_INF;

                for (int kj = 0; kj < k_len; kj++) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < BLOCK_D; d++) {
                        int dv = tid + d * num_threads;
                        if (dv < D) {
                            dot += __half2float(q_row_reg[d]) * __half2float(sram_k[kj * D + dv]);
                        }
                    }
                    // Warp reduce
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        dot += __shfl_down_sync(0xffffffff, dot, offset);
                    }
                    if (tid == 0) {
                        weights[kj] = dot * scale;
                    }
                }
                __syncthreads();

                // Get max and compute softmax
                float l_val;
                if (tid == 0) l_val = params.L[l_offset];

                // Normalize
                #pragma unroll
                for (int kj = 0; kj < BLOCK_N; kj++) {
                    if (kj < k_len) {
                        weights[kj] = __expf(weights[kj]) / __expf(l_val);
                    }
                }

                // Compute dQ_local: sum over V of (attention_weight * dO)
                // dQ = dO * (P^T * V) where P is attention weights
                #pragma unroll
                for (int d = 0; d < BLOCK_D; d++) {
                    int dv = tid + d * num_threads;
                    if (dv < D) {
                        float dq_val = 0.0f;
                        #pragma unroll
                        for (int kj = 0; kj < BLOCK_N; kj++) {
                            if (kj < k_len) {
                                dq_val += weights[kj] * __half2float(sram_v[kj * D + dv]);
                            }
                        }
                        // Warp reduce
                        #pragma unroll
                        for (int offset = 16; offset > 0; offset /= 2) {
                            dq_val += __shfl_down_sync(0xffffffff, dq_val, offset);
                        }
                        if (tid == 0) {
                            float existing = __half2float(params.dQ[q_offset + dv]);
                            existing += dq_val * scale * __half2float(dO_row[dv]);
                            params.dQ[q_offset + dv] = __float2half(existing);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

// Simplified backward kernel - processes everything in one pass
// More efficient but simplified for demonstration
__global__ void flash_attention_bwd_kernel_simple(
    FlashAttentionBwdParams params
) {
    const int B = params.B;
    const int H = params.H;
    const int N = params.N;
    const int M = params.M;
    const int D = params.D;
    const float scale = params.softmax_scale;

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_row = blockIdx.x;

    const int tid = threadIdx.x;
    const int num_threads = THREADS;

    // Strides
    const int q_batch_stride = H * N * D;
    const int k_batch_stride = H * M * D;
    const int q_base = batch_idx * q_batch_stride + head_idx * N * D;
    const int k_base = batch_idx * k_batch_stride + head_idx * M * D;
    const int l_offset = batch_idx * H * N + head_idx * N + q_row;

    // Load Q and dO for this row
    half q_row_data[BLOCK_D];
    half dO_row_data[BLOCK_D];

    const int q_offset = q_base + q_row * D;
    for (int d = tid; d < D; d += num_threads) {
        q_row_data[d] = params.Q[q_offset + d];
        dO_row_data[d] = params.dO[q_offset + d];
    }

    float l_val;
    if (tid == 0) {
        l_val = params.L[l_offset];
    }
    l_val = __shfl_sync(0xffffffff, l_val, 0);

    // Allocate temporary storage in registers
    // For each KV position, compute contribution to gradients
    // This is a simplified version - full implementation would use tiling

    // dQ accumulator
    float dQ_acc[BLOCK_D];
    #pragma unroll
    for (int d = 0; d < BLOCK_D; d++) {
        dQ_acc[d] = 0.0f;
    }

    // Process all KV in blocks
    const int num_kv_blocks = (M + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BLOCK_N;
        const int kv_len = min(BLOCK_N, M - kv_start);

        // Load K and V block
        half K_block[BLOCK_N * BLOCK_D];
        half V_block[BLOCK_N * BLOCK_D];

        for (int j = tid; j < kv_len * D; j += num_threads) {
            int kj = j / D;
            int dk = j % D;
            int kk = kv_start + kj;
            if (kk < M) {
                K_block[kj * D + dk] = params.K[k_base + kk * D + dk];
                V_block[kj * D + dk] = params.V[k_base + kk * D + dk];
            }
        }
        __syncthreads();

        // Compute attention weights for this block
        float weights[BLOCK_N];
        float max_scores[BLOCK_N];

        // First compute raw QK^T scores
        #pragma unroll
        for (int kj = 0; kj < BLOCK_N; kj++) {
            if (kj < kv_len) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < BLOCK_D; d++) {
                    int dv = tid + d * num_threads;
                    if (dv < D) {
                        dot += __half2float(q_row_data[dv]) * __half2float(K_block[kj * D + dv]);
                    }
                }
                // Warp reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }
                if (tid == 0) {
                    max_scores[kj] = dot * scale;
                }
            }
        }
        __syncthreads();

        // Compute softmax (need global max)
        float global_max = NEG_INF;
        #pragma unroll
        for (int kj = 0; kj < BLOCK_N; kj++) {
            if (kj < kv_len) {
                global_max = max_f(global_max, max_scores[kj]);
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            global_max = max_f(global_max, __shfl_down_sync(0xffffffff, global_max, offset));
        }
        global_max = __shfl_sync(0xffffffff, global_max, 0);

        // Normalize weights
        float sum_exp = 0.0f;
        #pragma unroll
        for (int kj = 0; kj < BLOCK_N; kj++) {
            if (kj < kv_len) {
                weights[kj] = __expf(max_scores[kj] - l_val);
                sum_exp += weights[kj];
            }
        }

        // Compute dQ contribution: dQ += dO * (softmax * V)^T
        // Actually simpler: for each kj, dQ += weight * (dO dot V_kj) * K_kj
        #pragma unroll
        for (int kj = 0; kj < BLOCK_N; kj++) {
            if (kj < kv_len) {
                // Compute v = dO dot V_kj
                float v = 0.0f;
                #pragma unroll
                for (int d = 0; d < BLOCK_D; d++) {
                    int dv = tid + d * num_threads;
                    if (dv < D) {
                        v += __half2float(dO_row_data[dv]) * __half2float(V_block[kj * D + dv]);
                    }
                }
                // Warp reduction
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    v += __shfl_down_sync(0xffffffff, v, offset);
                }

                if (tid == 0) {
                    v = v * weights[kj] / sum_exp;
                    #pragma unroll
                    for (int d = 0; d < BLOCK_D; d++) {
                        dQ_acc[d] += v * __half2float(K_block[kj * D + d]) * scale;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write dQ
    for (int d = tid; d < D; d += num_threads) {
        params.dQ[q_offset + d] = __float2half(dQ_acc[d]);
    }
}

// Wrapper
torch::Tensor flash_attention_bwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor O,
    torch::Tensor L,
    torch::Tensor dO
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");
    TORCH_CHECK(dO.is_cuda(), "dO must be CUDA");

    auto B = Q.size(0);
    auto H = Q.size(1);
    auto N = Q.size(2);
    auto M = K.size(2);
    auto D = Q.size(3);

    // Allocate gradients
    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);
    auto dV = torch::zeros_like(V);

    float softmax_scale = 1.0f / std::sqrt((float)D);

    FlashAttentionBwdParams params;
    params.Q = Q.data_ptr<half>();
    params.K = K.data_ptr<half>();
    params.V = V.data_ptr<half>();
    params.O = O.data_ptr<half>();
    params.L = L.data_ptr<float>();
    params.dO = dO.data_ptr<half>();
    params.dQ = dQ.data_ptr<half>();
    params.dK = dK.data_ptr<half>();
    params.dV = dV.data_ptr<half>();
    params.B = B;
    params.H = H;
    params.N = N;
    params.M = M;
    params.D = D;
    params.softmax_scale = softmax_scale;

    // Launch simplified backward kernel - one block per Q row
    dim3 grid(N, H, B);
    dim3 block(THREADS);

    flash_attention_bwd_kernel_simple<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(params);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA backward error: %s\n", cudaGetErrorString(err));
    }
    TORCH_CHECK(err == cudaSuccess, "CUDA backward kernel failed");

    return dQ;
}
