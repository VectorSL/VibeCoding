"""Generate optimization iteration charts and save to logs/."""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os

save_dir = os.path.join(os.path.dirname(__file__), "logs")

# ============================================================
# Chart 1: CUDA kernel optimization journey (all rounds)
# ============================================================
rounds = [0, 2, 3, 5, 6, 7, 12, 13, 16, 17, 18, 19, 20, 24, 25, 26, 33]
labels = [
    'R0\nBaseline', 'R2\nParallel\nDot', 'R3\nMulti-Q\nTiling',
    'R5\nBLOCK_Q=8', 'R6\nWarp/Row', 'R7\n8 Warps',
    'R12\nsm_89 Fix', 'R13\nWMMA QK',
    'R16\nWMMA QK+PV', 'R17\nBKV=32', 'R18\nBKV=64',
    'R19\nPar Softmax', 'R20\nDirect Acc',
    'R24\nfloat4 Load', 'R25\n8T/Row SM',
    'R26\nWarp Shuffle', 'R33\nSmem 16KB'
]
times_ms = [87, 15.5, 5.3, 4.9, 2.66, 2.20, 2.14, 1.32, 0.76, 0.65, 0.60, 0.41, 0.31, 0.26, 0.22, 0.219, 0.212]
speedups = [87 / t for t in times_ms]

fig, ax1 = plt.subplots(figsize=(18, 7))

color1 = '#2196F3'
color2 = '#FF5722'

bars = ax1.bar(range(len(rounds)), times_ms, color=color1, alpha=0.7, label='Latency (ms)')
ax1.set_ylabel('Latency (ms)', color=color1, fontsize=13)
ax1.set_yscale('log')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.1, 200)

# Add value labels on bars
for i, (t, bar) in enumerate(zip(times_ms, bars)):
    if t >= 1:
        ax1.text(bar.get_x() + bar.get_width()/2, t * 1.15, f'{t:.1f}ms',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax1.text(bar.get_x() + bar.get_width()/2, t * 1.15, f'{t:.2f}ms',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2 = ax1.twinx()
ax2.plot(range(len(rounds)), speedups, color=color2, marker='o', linewidth=2.5, markersize=7, label='Speedup vs Baseline')
ax2.set_ylabel('Speedup (x)', color=color2, fontsize=13)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 450)

# Add speedup labels
for i, s in enumerate(speedups):
    if s > 10:
        ax2.annotate(f'{s:.0f}x', (i, s), textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, color=color2, fontweight='bold')

ax1.set_xticks(range(len(rounds)))
ax1.set_xticklabels(labels, fontsize=7.5, rotation=0)
ax1.set_xlabel('Optimization Round', fontsize=13)

# Phase annotations
ax1.axvspan(-0.5, 5.5, alpha=0.05, color='blue', label='Phase 1: Basic')
ax1.axvspan(5.5, 11.5, alpha=0.05, color='green', label='Phase 2: Tensor Core')
ax1.axvspan(11.5, 14.5, alpha=0.05, color='orange', label='Phase 3: Fine-tuning')
ax1.axvspan(14.5, 16.5, alpha=0.05, color='red', label='Phase 5: Profiling')

# PyTorch reference line
ax1.axhline(y=0.057, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
ax1.text(len(rounds)-1, 0.057 * 0.75, 'PyTorch SDPA (0.057ms)', fontsize=9, color='green', ha='right')

plt.title('FlashAttention CUDA Kernel Optimization Journey\n87ms → 0.21ms (414x Speedup)', fontsize=15, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'cuda_optimization_journey.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved cuda_optimization_journey.png")

# ============================================================
# Chart 2: Three-way comparison across configs
# ============================================================
configs = ['1,8,512\nD=64', '1,8,1024\nD=64', '1,16,2048\nD=64', '2,16,2048\nD=64', '4,16,4096\nD=64']
pytorch_times = [0.057, 0.130, 0.747, 1.551, 13.059]
triton_times = [0.067, 0.148, 0.681, 1.306, 10.104]
cuda_times = [0.212, 0.576, 4.356, 8.522, 66.567]

x = np.arange(len(configs))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width, pytorch_times, width, label='PyTorch SDPA', color='#4CAF50', alpha=0.85)
bars2 = ax.bar(x, triton_times, width, label='Triton', color='#2196F3', alpha=0.85)
bars3 = ax.bar(x + width, cuda_times, width, label='Custom CUDA', color='#FF5722', alpha=0.85)

ax.set_ylabel('Latency (ms)', fontsize=13)
ax.set_xlabel('Configuration (B, H, N, D)', fontsize=13)
ax.set_title('Performance Comparison: PyTorch SDPA vs Triton vs Custom CUDA', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.legend(fontsize=11)
ax.set_yscale('log')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        label = f'{h:.2f}' if h < 1 else f'{h:.1f}'
        ax.text(bar.get_x() + bar.get_width()/2, h * 1.08, label,
                ha='center', va='bottom', fontsize=7.5, fontweight='bold')

fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'three_way_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved three_way_comparison.png")

# ============================================================
# Chart 3: Triton vs PyTorch speedup ratio
# ============================================================
triton_vs_pytorch = [p / t for p, t in zip(pytorch_times, triton_times)]
cuda_vs_pytorch = [c / p for c, p in zip(cuda_times, pytorch_times)]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(configs))
width = 0.35

bars1 = ax.bar(x - width/2, triton_vs_pytorch, width, label='Triton / PyTorch', color='#2196F3', alpha=0.85)
bars2 = ax.bar(x + width/2, cuda_vs_pytorch, width, label='Custom CUDA / PyTorch', color='#FF5722', alpha=0.85)

ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylabel('Ratio vs PyTorch SDPA', fontsize=13)
ax.set_xlabel('Configuration', fontsize=13)
ax.set_title('Performance Ratio vs PyTorch SDPA\n(< 1.0 = faster than PyTorch)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=10)
ax.legend(fontsize=11)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, f'{h:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'performance_ratio.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved performance_ratio.png")

# ============================================================
# Chart 4: Occupancy improvement (Round 26 vs Round 33)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Occupancy pie charts
for ax, title, occ, smem, blocks in [
    (axes[0], 'Round 26 (28KB smem)', 8.3, 28.2, 1),
    (axes[1], 'Round 33 (16KB smem)', 16.7, 16.2, 2),
]:
    colors = ['#2196F3', '#E0E0E0']
    ax.pie([occ, 100-occ], labels=[f'Active\n{occ}%', f'Idle\n{100-occ:.1f}%'],
           colors=colors, autopct='', startangle=90, textprops={'fontsize': 12})
    ax.set_title(f'{title}\n{blocks} block/SM, {smem}KB/block', fontsize=12, fontweight='bold')

fig.suptitle('SM Occupancy: Before vs After Profiling-Driven Optimization', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'occupancy_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved occupancy_comparison.png")

# ============================================================
# Chart 5: Development effort comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Lines of code
categories = ['Custom CUDA', 'Triton']
loc = [303, 130]
axes[0].barh(categories, loc, color=['#FF5722', '#2196F3'], alpha=0.85)
axes[0].set_xlabel('Lines of Code')
axes[0].set_title('Code Size', fontweight='bold')
for i, v in enumerate(loc):
    axes[0].text(v + 5, i, str(v), va='center', fontweight='bold')

# Iterations
iters = [33, 3]
axes[1].barh(categories, iters, color=['#FF5722', '#2196F3'], alpha=0.85)
axes[1].set_xlabel('Optimization Rounds')
axes[1].set_title('Development Iterations', fontweight='bold')
for i, v in enumerate(iters):
    axes[1].text(v + 0.5, i, str(v), va='center', fontweight='bold')

# Performance (small config)
perf = [0.212, 0.067]
axes[2].barh(categories, perf, color=['#FF5722', '#2196F3'], alpha=0.85)
axes[2].set_xlabel('Latency (ms)')
axes[2].set_title('Performance (B=1,H=8,N=512)', fontweight='bold')
for i, v in enumerate(perf):
    axes[2].text(v + 0.005, i, f'{v:.3f}ms', va='center', fontweight='bold')

fig.suptitle('Development Effort: Custom CUDA vs Triton', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, 'dev_effort_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved dev_effort_comparison.png")

print("\nAll charts saved to logs/")
