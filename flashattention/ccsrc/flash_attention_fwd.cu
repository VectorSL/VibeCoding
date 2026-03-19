#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <algorithm>

// FlashAttention v2 - Round 18: BLOCK_KV=64, 4 WMMA tiles
// QK^T: 4 warps each compute 16 KV positions
// PV: 4 K-tiles of 16 per D-tile

using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int BLOCK_Q = 16;
constexpr int BLOCK_KV = 64;
constexpr int NUM_WARPS = 4;
constexpr int THREADS = NUM_WARPS * 32;
constexpr float NEG_INF = -1e10f;

struct FlashAttentionParams {
    const at::Half* Q;
    const at::Half* K;
    const at::Half* V;
    at::Half* O;
    int B, H, N, M, D;
    float softmax_scale;
};

__global__ void flash_attention_fwd_kernel(const FlashAttentionParams p) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_start = blockIdx.x * BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int D = p.D;

    if (q_block_start >= p.N) return;
    const int q_rows = min(BLOCK_Q, p.N - q_block_start);

    const int q_stride = p.H * p.N * D;
    const int k_stride = p.H * p.M * D;
    const int q_base = batch_idx * q_stride + head_idx * p.N * D;
    const int k_base = batch_idx * k_stride + head_idx * p.M * D;

    extern __shared__ char smem_raw[];
    half* Q_tile = reinterpret_cast<half*>(smem_raw);
    half* K_tile = Q_tile + BLOCK_Q * D;
    half* V_tile = K_tile + BLOCK_KV * D;
    float* S_float = reinterpret_cast<float*>(V_tile + BLOCK_KV * D);
    half* P_half = reinterpret_cast<half*>(S_float + BLOCK_Q * BLOCK_KV);
    float* max_vals = reinterpret_cast<float*>(P_half + BLOCK_Q * BLOCK_KV);
    float* sum_vals = max_vals + BLOCK_Q;
    float* O_acc = sum_vals + BLOCK_Q;

    // Load Q tile
    for (int j = tid; j < BLOCK_Q * D; j += THREADS) {
        int qi = j / D;
        int dk = j % D;
        Q_tile[qi * D + dk] = (qi < q_rows) ? p.Q[q_base + (q_block_start + qi) * D + dk] : __float2half(0.0f);
    }

    if (tid < BLOCK_Q) {
        max_vals[tid] = NEG_INF;
        sum_vals[tid] = 0.0f;
    }
    for (int j = tid; j < BLOCK_Q * D; j += THREADS) {
        O_acc[j] = 0.0f;
    }
    __syncthreads();

    const int num_kv_blocks = (p.M + BLOCK_KV - 1) / BLOCK_KV;
    const int d_tiles = (D + WMMA_N - 1) / WMMA_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BLOCK_KV;
        const int kv_len = min(BLOCK_KV, p.M - kv_start);

        // Load K and V tiles
        for (int j = tid; j < BLOCK_KV * D; j += THREADS) {
            int ki = j / D;
            int dk = j % D;
            if (ki < kv_len) {
                K_tile[ki * D + dk] = p.K[k_base + (kv_start + ki) * D + dk];
                V_tile[ki * D + dk] = p.V[k_base + (kv_start + ki) * D + dk];
            } else {
                K_tile[ki * D + dk] = __float2half(0.0f);
                V_tile[ki * D + dk] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Step 1: WMMA QK^T -> S_float[16x64]
        // 4 WMMA tiles: each warp does 16 KV positions
        {
            const int kv_offset = warp_id * WMMA_N;

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            wmma::fill_fragment(acc_frag, 0.0f);

            const int dk_tiles = (D + WMMA_K - 1) / WMMA_K;
            for (int dt = 0; dt < dk_tiles; dt++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_tile + dt * WMMA_K, D);
                wmma::load_matrix_sync(b_frag, K_tile + kv_offset * D + dt * WMMA_K, D);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
            wmma::store_matrix_sync(S_float + kv_offset, acc_frag, BLOCK_KV, wmma::mem_row_major);
        }
        __syncthreads();

        // Step 2: Online softmax + produce P_half (parallelized)

        // 2a: Scale + find row max in ONE pass (merged, saves 1 sync)
        __shared__ float row_max_partial[BLOCK_Q * 4];
        {
            const int threads_per_row = 4;
            const int total_jobs = BLOCK_Q * threads_per_row;  // 64
            for (int job = tid; job < total_jobs; job += THREADS) {
                int qi = job / threads_per_row;
                int chunk = job % threads_per_row;
                int kj_start = chunk * (BLOCK_KV / threads_per_row);
                int kj_end = kj_start + (BLOCK_KV / threads_per_row);

                float local_max = NEG_INF;
                if (qi < q_rows) {
                    for (int kj = kj_start; kj < kj_end && kj < kv_len; kj++) {
                        float s = S_float[qi * BLOCK_KV + kj] * p.softmax_scale;
                        S_float[qi * BLOCK_KV + kj] = s;
                        local_max = fmaxf(local_max, s);
                    }
                }
                row_max_partial[qi * threads_per_row + chunk] = local_max;
            }
        }
        __syncthreads();

        // 2b: Reduce partial maxes and compute exp_correction
        __shared__ float exp_corr[BLOCK_Q];
        if (tid < BLOCK_Q && tid < q_rows) {
            float row_max = row_max_partial[tid * 4];
            row_max = fmaxf(row_max, row_max_partial[tid * 4 + 1]);
            row_max = fmaxf(row_max, row_max_partial[tid * 4 + 2]);
            row_max = fmaxf(row_max, row_max_partial[tid * 4 + 3]);
            row_max = fmaxf(row_max, max_vals[tid]);

            exp_corr[tid] = __expf(max_vals[tid] - row_max);
            max_vals[tid] = row_max;
        }
        __syncthreads();

        // 2c: Compute exp(score - max) and store as P_half, in parallel
        for (int j = tid; j < BLOCK_Q * BLOCK_KV; j += THREADS) {
            int qi = j / BLOCK_KV;
            int kj = j % BLOCK_KV;
            if (qi < q_rows && kj < kv_len) {
                P_half[j] = __float2half(__expf(S_float[j] - max_vals[qi]));
            } else {
                P_half[j] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // 2d: Row sums + update sum_vals (parallel reduce)
        __shared__ float row_sum_partial[BLOCK_Q * 4];
        {
            const int threads_per_row = 4;
            const int total_jobs = BLOCK_Q * threads_per_row;
            for (int job = tid; job < total_jobs; job += THREADS) {
                int qi = job / threads_per_row;
                int chunk = job % threads_per_row;
                int kj_start = chunk * (BLOCK_KV / threads_per_row);
                int kj_end = kj_start + (BLOCK_KV / threads_per_row);

                float local_sum = 0.0f;
                if (qi < q_rows) {
                    for (int kj = kj_start; kj < kj_end && kj < kv_len; kj++) {
                        local_sum += __half2float(P_half[qi * BLOCK_KV + kj]);
                    }
                }
                row_sum_partial[qi * threads_per_row + chunk] = local_sum;
            }
        }
        __syncthreads();

        // 2e: Reduce sums + rescale O_acc (merged, saves 1 sync)
        if (tid < BLOCK_Q && tid < q_rows) {
            float psum = row_sum_partial[tid * 4] + row_sum_partial[tid * 4 + 1]
                       + row_sum_partial[tid * 4 + 2] + row_sum_partial[tid * 4 + 3];
            sum_vals[tid] = sum_vals[tid] * exp_corr[tid] + psum;
        }
        for (int j = tid; j < q_rows * D; j += THREADS) {
            int qi = j / D;
            O_acc[j] *= exp_corr[qi];
        }
        __syncthreads();

        // Step 3: WMMA P @ V -> directly into O_acc
        // P[16x64] @ V[64xD]: split into D-tiles of 16
        // Load O_acc into accumulator, WMMA adds PV on top, store back
        for (int dt = warp_id; dt < d_tiles; dt += NUM_WARPS) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_acc;
            // Initialize accumulator with current O_acc values
            wmma::load_matrix_sync(pv_acc, O_acc + dt * WMMA_N, D, wmma::mem_row_major);

            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, P_half + kt * WMMA_K, BLOCK_KV);
                wmma::load_matrix_sync(v_frag, V_tile + kt * WMMA_K * D + dt * WMMA_N, D);
                wmma::mma_sync(pv_acc, p_frag, v_frag, pv_acc);
            }

            wmma::store_matrix_sync(O_acc + dt * WMMA_N, pv_acc, D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Write final output
    for (int j = tid; j < q_rows * D; j += THREADS) {
        int qi = j / D;
        int d = j % D;
        int q_row = q_block_start + qi;
        float val = (sum_vals[qi] > 0) ? O_acc[j] / sum_vals[qi] : 0.0f;
        p.O[q_base + q_row * D + d] = __float2half(val);
    }
}

torch::Tensor flash_attention_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    auto B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto M = K.size(2);

    TORCH_CHECK(D <= 128, "D must be <= 128");

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

    size_t shared_mem = (BLOCK_Q * D + BLOCK_KV * D + BLOCK_KV * D) * sizeof(half)
                      + BLOCK_Q * BLOCK_KV * sizeof(float)
                      + BLOCK_Q * BLOCK_KV * sizeof(half)
                      + 2 * BLOCK_Q * sizeof(float)
                      + BLOCK_Q * D * sizeof(float)
                      + BLOCK_Q * 4 * sizeof(float) * 2;  // row_max_partial + row_sum_partial

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
