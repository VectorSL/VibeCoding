"""Get kernel attributes and occupancy analysis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
import flash_attention_cuda
import numpy as np

B, H, N, M, D = 1, 8, 512, 512, 64
device = torch.device("cuda")

Q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
K = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
V = torch.randn(B, H, M, D, device=device, dtype=torch.float16)

# Torch profiler with CPU+CUDA
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    for _ in range(10):
        O = flash_attention_cuda.forward(Q, K, V)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# CUDA events timing
torch.cuda.synchronize()
events = []
for _ in range(50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    O = flash_attention_cuda.forward(Q, K, V)
    end.record()
    torch.cuda.synchronize()
    events.append(start.elapsed_time(end))

print(f"\nCUDA Event timing (50 runs): {np.mean(events):.3f} +/- {np.std(events):.3f} ms")
print(f"Min: {np.min(events):.3f} ms, Max: {np.max(events):.3f} ms")

# GPU info
props = torch.cuda.get_device_properties(0)
print(f"\nGPU: {props.name}")
print(f"SMs: {props.multi_processor_count}")
print(f"Max shared memory per block: {props.sharedMemPerBlock} bytes" if hasattr(props, 'sharedMemPerBlock') else "")
print(f"Max threads per SM: {props.max_threads_per_multi_processor}")
print(f"Warp size: {props.warp_size}")
print(f"Total global memory: {props.total_memory / 1024**3:.1f} GB")

# RTX 4060 Laptop: 48KB shared memory per SM
SMEM_PER_SM = 48 * 1024

# Kernel config analysis
BLOCK_Q, BLOCK_KV, D_val = 16, 64, 64
THREADS = 128
smem = BLOCK_Q * D_val * 2 \
     + BLOCK_KV * D_val * 2 \
     + BLOCK_Q * BLOCK_KV * 2 \
     + BLOCK_Q * D_val * 4 \
     + 3 * BLOCK_Q * 4
grid_blocks = ((N + BLOCK_Q - 1) // BLOCK_Q) * H * B

print(f"\n{'='*60}")
print(f"Kernel Configuration Analysis")
print(f"{'='*60}")
print(f"Threads per block: {THREADS}")
print(f"Warps per block: {THREADS // 32}")
print(f"Shared memory per block: {smem} bytes ({smem/1024:.1f} KB)")
print(f"Grid: {(N + BLOCK_Q - 1) // BLOCK_Q} x {H} x {B} = {grid_blocks} blocks")
print(f"Blocks per SM (smem limited, 48KB): {SMEM_PER_SM // smem}")
print(f"Blocks per SM (thread limited, {props.max_threads_per_multi_processor}): {props.max_threads_per_multi_processor // THREADS}")

# Theoretical occupancy
max_blocks_smem = SMEM_PER_SM // smem
max_blocks_threads = props.max_threads_per_multi_processor // THREADS
max_blocks = min(max_blocks_smem, max_blocks_threads)
active_warps = max_blocks * (THREADS // 32)
max_warps_per_sm = props.max_threads_per_multi_processor // 32
occupancy = active_warps / max_warps_per_sm * 100

print(f"\nOccupancy estimate (ignoring registers):")
print(f"  Max blocks per SM: {max_blocks}")
print(f"  Active warps per SM: {active_warps}")
print(f"  Max warps per SM: {max_warps_per_sm}")
print(f"  Theoretical occupancy: {occupancy:.1f}%")

# Wave analysis
sm_count = props.multi_processor_count
waves = grid_blocks / (sm_count * max_blocks)
print(f"\nWave analysis:")
print(f"  Total blocks: {grid_blocks}")
print(f"  SM count: {sm_count}")
print(f"  Blocks per SM: {max_blocks}")
print(f"  Waves: {waves:.2f}")
print(f"  Tail effect: {'Yes' if waves != int(waves) else 'No'} ({grid_blocks % (sm_count * max_blocks)} leftover blocks)")

# Compute intensity analysis
print(f"\n{'='*60}")
print(f"Compute vs Memory Analysis (per KV tile iteration)")
print(f"{'='*60}")
# Per tile: load K(64x64x2=8KB) + V(64x64x2=8KB) = 16KB
# Compute: QK^T = 16x64x64 = 65536 FMA, PV = 16x64x64 = 65536 FMA
# Total: 131072 FMA = 262144 FLOP
mem_per_tile = (BLOCK_KV * D_val * 2 + BLOCK_KV * D_val * 2)  # K + V in bytes
flop_per_tile = 2 * BLOCK_Q * D_val * BLOCK_KV + 2 * BLOCK_Q * BLOCK_KV * D_val  # QK^T + PV
softmax_ops = BLOCK_Q * BLOCK_KV * 5  # scale, max, exp, sum, div approx
print(f"  Memory load per tile: {mem_per_tile} bytes ({mem_per_tile/1024:.1f} KB)")
print(f"  Tensor Core FLOP per tile: {flop_per_tile}")
print(f"  Softmax scalar ops per tile: ~{softmax_ops}")
print(f"  Arithmetic intensity: {flop_per_tile / mem_per_tile:.1f} FLOP/byte")
print(f"  KV tiles per block: {(M + BLOCK_KV - 1) // BLOCK_KV}")
