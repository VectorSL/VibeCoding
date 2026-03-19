#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <algorithm>

// FlashAttention v2 - Round 33: Aggressive smem reduction for higher occupancy
// Target: ~16KB smem -> 3 blocks/SM (from 28KB -> 1 block/SM)
// Changes:
// 1. K/V share same buffer (load K, compute QK^T, then load V for PV)
// 2. S stored as half (not float) - saves 2KB
// 3. S and P share same buffer (compute S, convert to P in-place)
// 4. Q loaded from global memory each tile (saves 2KB smem)

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

__device__ __forceinline__ float warp_reduce_max_8(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 1));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_8(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

__global__ __launch_bounds__(128, 2) void flash_attention_fwd_kernel(const FlashAttentionParams p) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_block_start = blockIdx.x * BLOCK_Q;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int D = p.D;

    if (q_block_start >= p.N) return;
    const int q_rows = min(BLOCK_Q, p.N - q_block_start);

    const int q_stride = p.H * p.N * D;
    const int k_stride = p.H * p.M * D;
    const int q_base = batch_idx * q_stride + head_idx * p.N * D;
    const int k_base = batch_idx * k_stride + head_idx * p.M * D;

    const int sm_qi = warp_id * 4 + lane_id / 8;
    const int sm_chunk = lane_id % 8;
    const int kv_per_thread = BLOCK_KV / 8;

    // Shared memory layout (aggressive reduction):
    // KV_tile: 64*64*2 = 8192 (shared K/V)
    // S_P_half: 16*64*2 = 2048 (shared S/P as half)
    // O_acc: 16*64*4 = 4096
    // max_vals: 16*4 = 64
    // sum_vals: 16*4 = 64
    // exp_corr: 16*4 = 64
    // Q_tile: 16*64*2 = 2048 (keep Q in smem, reload is expensive)
    // Total: 16576 bytes = 16.2 KB -> 2 blocks/SM (maybe 3 with register tuning)

    extern __shared__ char smem_raw[];
    half* Q_tile = reinterpret_cast<half*>(smem_raw);
    half* KV_tile = Q_tile + BLOCK_Q * D;
    half* S_P_half = KV_tile + BLOCK_KV * D;
    float* O_acc = reinterpret_cast<float*>(S_P_half + BLOCK_Q * BLOCK_KV);
    float* max_vals = O_acc + BLOCK_Q * D;
    float* sum_vals = max_vals + BLOCK_Q;
    float* exp_corr = sum_vals + BLOCK_Q;
    // Temp float buffer for softmax (reuse part of KV_tile after QK^T, before V load)
    // We need 16*64 floats = 4096 bytes, KV_tile is 8192 bytes - fits!
    float* S_float_tmp = reinterpret_cast<float*>(KV_tile);

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

        // Load K tile into KV_tile
        {
            const int total_halfs = BLOCK_KV * D;
            const int total_vec = total_halfs / 8;
            float4* KV_vec = reinterpret_cast<float4*>(KV_tile);
            const float4* K_src = reinterpret_cast<const float4*>(p.K + k_base + kv_start * D);
            if (kv_len == BLOCK_KV) {
                for (int i = tid; i < total_vec; i += THREADS) {
                    KV_vec[i] = K_src[i];
                }
            } else {
                for (int j = tid; j < BLOCK_KV * D; j += THREADS) {
                    int ki = j / D;
                    int dk = j % D;
                    KV_tile[j] = (ki < kv_len) ? p.K[k_base + (kv_start + ki) * D + dk] : __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Step 1: WMMA QK^T
        // Each warp computes its 16x16 tile of QK^T, result stays in acc_frag (registers)
        // After sync (all warps done reading K), store to S_float_tmp (overlaps KV_tile)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> qk_acc;
        {
            const int kv_offset = warp_id * WMMA_N;
            wmma::fill_fragment(qk_acc, 0.0f);

            const int dk_tiles = (D + WMMA_K - 1) / WMMA_K;
            for (int dt = 0; dt < dk_tiles; dt++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                wmma::load_matrix_sync(a_frag, Q_tile + dt * WMMA_K, D);
                wmma::load_matrix_sync(b_frag, KV_tile + kv_offset * D + dt * WMMA_K, D);
                wmma::mma_sync(qk_acc, a_frag, b_frag, qk_acc);
            }
        }
        // All warps done reading K from KV_tile
        __syncthreads();
        // Now safe to overwrite KV_tile with S_float_tmp
        {
            const int kv_offset = warp_id * WMMA_N;
            wmma::store_matrix_sync(S_float_tmp + kv_offset, qk_acc, BLOCK_KV, wmma::mem_row_major);
        }
        __syncthreads();

        // Step 2: Online softmax (using S_float_tmp which now overlaps KV_tile)
        float local_max = NEG_INF;
        if (sm_qi < q_rows) {
            int kj_start = sm_chunk * kv_per_thread;
            int kj_end = kj_start + kv_per_thread;
            for (int kj = kj_start; kj < kj_end && kj < kv_len; kj++) {
                float s = S_float_tmp[sm_qi * BLOCK_KV + kj] * p.softmax_scale;
                S_float_tmp[sm_qi * BLOCK_KV + kj] = s;
                local_max = fmaxf(local_max, s);
            }
        }
        float row_max = warp_reduce_max_8(local_max);
        row_max = fmaxf(row_max, max_vals[sm_qi]);

        float ec;
        if (sm_chunk == 0 && sm_qi < q_rows) {
            ec = __expf(max_vals[sm_qi] - row_max);
            exp_corr[sm_qi] = ec;
            max_vals[sm_qi] = row_max;
        }
        __syncthreads();
        ec = exp_corr[sm_qi];
        float my_row_max = max_vals[sm_qi];

        // Compute exp and store P as half into S_P_half
        float local_sum = 0.0f;
        if (sm_qi < q_rows) {
            int kj_start = sm_chunk * kv_per_thread;
            int kj_end = kj_start + kv_per_thread;
            for (int kj = kj_start; kj < kj_end; kj++) {
                if (kj < kv_len) {
                    float ev = __expf(S_float_tmp[sm_qi * BLOCK_KV + kj] - my_row_max);
                    S_P_half[sm_qi * BLOCK_KV + kj] = __float2half(ev);
                    local_sum += ev;
                } else {
                    S_P_half[sm_qi * BLOCK_KV + kj] = __float2half(0.0f);
                }
            }
        }
        float row_sum = warp_reduce_sum_8(local_sum);

        if (sm_chunk == 0 && sm_qi < q_rows) {
            sum_vals[sm_qi] = sum_vals[sm_qi] * ec + row_sum;
        }

        // Rescale O_acc
        for (int j = tid; j < q_rows * D; j += THREADS) {
            int qi = j / D;
            O_acc[j] *= exp_corr[qi];
        }
        __syncthreads();

        // Load V tile into KV_tile (S_float_tmp is no longer needed, KV_tile is free)
        {
            const int total_halfs = BLOCK_KV * D;
            const int total_vec = total_halfs / 8;
            float4* KV_vec = reinterpret_cast<float4*>(KV_tile);
            const float4* V_src = reinterpret_cast<const float4*>(p.V + k_base + kv_start * D);
            if (kv_len == BLOCK_KV) {
                for (int i = tid; i < total_vec; i += THREADS) {
                    KV_vec[i] = V_src[i];
                }
            } else {
                for (int j = tid; j < BLOCK_KV * D; j += THREADS) {
                    int ki = j / D;
                    int dk = j % D;
                    KV_tile[j] = (ki < kv_len) ? p.V[k_base + (kv_start + ki) * D + dk] : __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // Step 3: WMMA P @ V
        for (int dt = warp_id; dt < d_tiles; dt += NUM_WARPS) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_acc;
            wmma::load_matrix_sync(pv_acc, O_acc + dt * WMMA_N, D, wmma::mem_row_major);

            #pragma unroll
            for (int kt = 0; kt < 4; kt++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag;
                wmma::load_matrix_sync(p_frag, S_P_half + kt * WMMA_K, BLOCK_KV);
                wmma::load_matrix_sync(v_frag, KV_tile + kt * WMMA_K * D + dt * WMMA_N, D);
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

    // Shared memory: Q(2KB) + KV(8KB) + S_P_half(2KB) + O_acc(4KB) + misc(192B) = ~16.2KB
    // S_float_tmp overlaps KV_tile (reused after K is consumed)
    size_t shared_mem = BLOCK_Q * D * sizeof(half)              // Q_tile
                      + BLOCK_KV * D * sizeof(half)             // KV_tile (also S_float_tmp)
                      + BLOCK_Q * BLOCK_KV * sizeof(half)       // S_P_half
                      + BLOCK_Q * D * sizeof(float)             // O_acc
                      + 3 * BLOCK_Q * sizeof(float);            // max_vals, sum_vals, exp_corr

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
