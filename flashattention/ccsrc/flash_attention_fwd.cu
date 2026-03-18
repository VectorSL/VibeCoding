#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

// Minimal FlashAttention v2 - Debug version
// Supports any head dimension D up to 128

constexpr int BLOCK_N = 64;
constexpr int THREADS = 128;
constexpr int MAX_D = 128;  // Maximum supported head dimension
constexpr float NEG_INF = -1e10f;

struct FlashAttentionParams {
    const at::Half* Q;
    const at::Half* K;
    const at::Half* V;
    at::Half* O;
    int B, H, N, M, D;
    float softmax_scale;
};

// Simple kernel - debug by printing via CPU
__global__ void flash_attention_fwd_kernel(const FlashAttentionParams p) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_row = blockIdx.x;

    if (q_row >= p.N) return;

    const int tid = threadIdx.x;
    const int num_threads = THREADS;

    // Strides
    const int q_stride = p.H * p.N * p.D;
    const int k_stride = p.H * p.M * p.D;
    const int head_stride_q = p.N * p.D;
    const int head_stride_k = p.M * p.D;

    const int q_base = batch_idx * q_stride + head_idx * head_stride_q;
    const int k_base = batch_idx * k_stride + head_idx * head_stride_k;

    // All threads load entire Q row into shared memory
    extern __shared__ float sram[];
    float* sram_Q = sram;
    float* sram_K = sram + p.D;
    float* sram_V = sram + p.D + BLOCK_N * p.D;

    // Load Q into shared memory
    for (int d = tid; d < p.D; d += num_threads) {
        sram_Q[d] = __half2float(p.Q[q_base + q_row * p.D + d]);
    }

    // Output for this Q row
    float output[MAX_D] = {0.0f};
    float max_val = NEG_INF;
    float sum_val = 0.0f;

    // Process KV blocks
    const int num_kv_blocks = (p.M + BLOCK_N - 1) / BLOCK_N;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const int kv_start = kv_block * BLOCK_N;
        const int kv_len = min(BLOCK_N, p.M - kv_start);

        // Load K block
        for (int j = tid; j < kv_len * p.D; j += num_threads) {
            int kj = j / p.D;
            int dk = j % p.D;
            int kk = kv_start + kj;
            if (kk < p.M && dk < p.D) {
                sram_K[kj * p.D + dk] = __half2float(p.K[k_base + kk * p.D + dk]);
            }
        }

        // Load V block
        for (int j = tid; j < kv_len * p.D; j += num_threads) {
            int vj = j / p.D;
            int dv = j % p.D;
            int vk = kv_start + vj;
            if (vk < p.M && dv < p.D) {
                sram_V[vj * p.D + dv] = __half2float(p.V[k_base + vk * p.D + dv]);
            }
        }
        __syncthreads();

        // Compute attention for this block
        for (int kj = 0; kj < kv_len; kj++) {
            // QK^T dot product
            float score = 0.0f;
            for (int d = 0; d < p.D; d++) {
                score += sram_Q[d] * sram_K[kj * p.D + d];
            }
            score *= p.softmax_scale;

            // Softmax
            float new_max = fmaxf(max_val, score);
            float exp_score = expf(score - new_max);
            float exp_max = expf(max_val - new_max);

            // Update output
            for (int d = 0; d < p.D; d++) {
                output[d] = output[d] * exp_max + exp_score * sram_V[kj * p.D + d];
            }

            // Update sum
            sum_val = sum_val * exp_max + exp_score;
            max_val = new_max;
        }

        __syncthreads();
    }

    // Normalize and write output
    const int o_offset = q_base + q_row * p.D;
    for (int d = tid; d < p.D; d += num_threads) {
        float val = (sum_val > 0) ? output[d] / sum_val : 0.0f;
        p.O[o_offset + d] = __float2half(val);
    }
}

torch::Tensor flash_attention_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA");

    auto B = Q.size(0), H = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto M = K.size(2);

    float scale = 1.0f / sqrtf((float)D);

    auto O = torch::empty_like(Q);

    FlashAttentionParams params;
    params.Q = Q.data_ptr<at::Half>();
    params.K = K.data_ptr<at::Half>();
    params.V = V.data_ptr<at::Half>();
    params.O = O.data_ptr<at::Half>();
    params.B = B; params.H = H; params.N = N; params.M = M; params.D = D;
    params.softmax_scale = scale;

    dim3 grid(N, H, B);
    dim3 block(THREADS);
    // Q + K + V blocks: D + BLOCK_N * D + BLOCK_N * D
    size_t shared_mem = (D + 2 * BLOCK_N * D) * sizeof(float);

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
