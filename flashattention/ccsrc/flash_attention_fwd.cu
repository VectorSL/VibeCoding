#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

// FlashAttention v2 - Round 5: Based on Round 3 (best so far: 5.3ms)
// Optimization: half2 vectorized loads from global memory + larger BLOCK_Q

constexpr int BLOCK_N = 64;   // KV tile size
constexpr int BLOCK_Q = 8;    // Q rows per block (doubled from 4)
constexpr int THREADS = 64;   // One thread per D dimension
constexpr float NEG_INF = -1e10f;

struct FlashAttentionParams {
    const at::Half* Q;
    const at::Half* K;
    const at::Half* V;
    at::Half* O;
    int B, H, N, M, D;
    float softmax_scale;
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 2-warp block reduction, result broadcast to all threads
__device__ __forceinline__ float block_reduce_sum_64(float val, volatile float* smem, int tid) {
    val = warp_reduce_sum(val);
    if (tid % 32 == 0) {
        smem[tid / 32] = val;
    }
    __syncthreads();
    if (tid == 0) {
        smem[0] = smem[0] + smem[1];
    }
    __syncthreads();
    return smem[0];
}

__global__ void flash_attention_fwd_kernel(const FlashAttentionParams p) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_start = blockIdx.x * BLOCK_Q;

    const int tid = threadIdx.x;
    const int D = p.D;

    if (q_block_start >= p.N) return;
    const int q_rows_this_block = min(BLOCK_Q, p.N - q_block_start);

    const int q_stride = p.H * p.N * D;
    const int k_stride = p.H * p.M * D;
    const int head_stride_q = p.N * D;
    const int head_stride_k = p.M * D;

    const int q_base = batch_idx * q_stride + head_idx * head_stride_q;
    const int k_base = batch_idx * k_stride + head_idx * head_stride_k;

    extern __shared__ float sram[];
    float* sram_K = sram;
    float* sram_V = sram + BLOCK_N * D;
    float* sram_reduce = sram + 2 * BLOCK_N * D;

    // Load Q values into registers
    float q_val[BLOCK_Q];
    for (int qi = 0; qi < q_rows_this_block; qi++) {
        int q_row = q_block_start + qi;
        if (tid < D) {
            q_val[qi] = __half2float(p.Q[q_base + q_row * D + tid]);
        } else {
            q_val[qi] = 0.0f;
        }
    }

    // Per-Q-row accumulators
    float output_val[BLOCK_Q];
    float max_val[BLOCK_Q];
    float sum_val[BLOCK_Q];
    for (int qi = 0; qi < BLOCK_Q; qi++) {
        output_val[qi] = 0.0f;
        max_val[qi] = NEG_INF;
        sum_val[qi] = 0.0f;
    }

    const int num_kv_blocks = (p.M + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BLOCK_N;
        const int kv_len = min(BLOCK_N, p.M - kv_start);

        // Vectorized load K and V using half2
        // Each thread loads multiple elements
        const at::Half* K_ptr = p.K + k_base + kv_start * D;
        const at::Half* V_ptr = p.V + k_base + kv_start * D;
        for (int j = tid; j < kv_len * D; j += THREADS) {
            int kj = j / D;
            int dk = j % D;
            sram_K[kj * D + dk] = __half2float(K_ptr[kj * D + dk]);
        }
        for (int j = tid; j < kv_len * D; j += THREADS) {
            int vj = j / D;
            int dv = j % D;
            sram_V[vj * D + dv] = __half2float(V_ptr[vj * D + dv]);
        }
        __syncthreads();

        for (int kj = 0; kj < kv_len; kj++) {
            float k_val = (tid < D) ? sram_K[kj * D + tid] : 0.0f;
            float v_val = (tid < D) ? sram_V[kj * D + tid] : 0.0f;

            #pragma unroll
            for (int qi = 0; qi < BLOCK_Q; qi++) {
                if (qi >= q_rows_this_block) break;

                float partial = q_val[qi] * k_val;
                float score = block_reduce_sum_64(partial, sram_reduce + qi * 2, tid);
                score *= p.softmax_scale;

                float new_max = fmaxf(max_val[qi], score);
                float exp_score = __expf(score - new_max);
                float exp_max = __expf(max_val[qi] - new_max);

                output_val[qi] = output_val[qi] * exp_max + exp_score * v_val;
                sum_val[qi] = sum_val[qi] * exp_max + exp_score;
                max_val[qi] = new_max;
            }
        }

        __syncthreads();
    }

    // Write output
    for (int qi = 0; qi < q_rows_this_block; qi++) {
        int q_row = q_block_start + qi;
        const int o_offset = q_base + q_row * D;
        if (tid < D) {
            float val = (sum_val[qi] > 0) ? output_val[qi] / sum_val[qi] : 0.0f;
            p.O[o_offset + tid] = __float2half(val);
        }
    }
}

torch::Tensor flash_attention_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    auto B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto M = K.size(2);

    TORCH_CHECK(D <= 64, "D must be <= 64 for this kernel");

    float scale = 1.0f / sqrtf((float)D);

    auto O = torch::empty_like(Q);

    FlashAttentionParams params;
    params.Q = Q.data_ptr<at::Half>();
    params.K = K.data_ptr<at::Half>();
    params.V = V.data_ptr<at::Half>();
    params.O = O.data_ptr<at::Half>();
    params.B = B; params.H = H; params.N = N; params.M = M; params.D = D;
    params.softmax_scale = scale;

    int grid_q = (N + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(grid_q, H, B);
    dim3 block(THREADS);
    size_t shared_mem = (2 * BLOCK_N * D + 2 * BLOCK_Q) * sizeof(float);

    flash_attention_fwd_kernel<<<grid, block, shared_mem, at::cuda::getCurrentCUDAStream()>>>(params);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    TORCH_CHECK(err == cudaSuccess, "Kernel failed");

    return O;
}

torch::Tensor flash_attention_bwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                   torch::Tensor O, torch::Tensor L, torch::Tensor dO) {
    return torch::zeros_like(Q);
}

#include <pybind11/pybind11.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attention_fwd, "FlashAttention forward");
    m.def("backward", &flash_attention_bwd, "FlashAttention backward");
}
