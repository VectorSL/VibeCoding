"""
FlashAttention v2 性能对比模拟结果

由于当前环境没有CUDA，本脚本模拟展示预期的性能对比结果。
实际使用时，请在有CUDA的环境中运行 run.py 获取真实数据。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# ===== 模拟性能数据 =====
# 这些是典型的FlashAttention性能数据（基于A100 GPU）
# 实际运行时会由 run.py 生成真实数据

# 配置: (B, H, N, M, D)
configs = [
    (1, 8, 512, 512, 64),
    (1, 8, 1024, 1024, 64),
    (1, 16, 2048, 2048, 64),
    (2, 16, 2048, 2048, 64),
    (4, 16, 4096, 4096, 64),
]

# 模拟性能数据 (ms) - 基于典型A100性能
# Custom CUDA: 初始实现，可能比PyTorch慢一些
# PyTorch SDPA: 优化良好的FlashAttention后端
# xformers: Triton实现，性能优秀

performance_data = {
    'Custom CUDA': {
        (1, 8, 512, 512, 64): 2.5,
        (1, 8, 1024, 1024, 64): 8.2,
        (1, 16, 2048, 2048, 64): 28.5,
        (2, 16, 2048, 2048, 64): 55.0,
        (4, 16, 4096, 4096, 64): 180.0,
    },
    'PyTorch SDPA': {
        (1, 8, 512, 512, 64): 1.8,
        (1, 8, 1024, 1024, 64): 6.5,
        (1, 16, 2048, 2048, 64): 22.0,
        (2, 16, 2048, 2048, 64): 42.0,
        (4, 16, 4096, 4096, 64): 145.0,
    },
    'xformers': {
        (1, 8, 512, 512, 64): 1.5,
        (1, 8, 1024, 1024, 64): 5.8,
        (1, 16, 2048, 2048, 64): 20.0,
        (2, 16, 2048, 2048, 64): 38.0,
        (4, 16, 4096, 4096, 64): 130.0,
    },
}

# 精度对比 (max abs difference)
accuracy_data = {
    'Custom vs PyTorch': {
        (1, 8, 512, 512, 64): 1.2e-4,
        (1, 8, 1024, 1024, 64): 2.5e-4,
        (1, 16, 2048, 2048, 64): 5.0e-4,
        (2, 16, 2048, 2048, 64): 6.5e-4,
        (4, 16, 4096, 4096, 64): 1.2e-3,
    },
    'xformers vs PyTorch': {
        (1, 8, 512, 512, 64): 8.0e-5,
        (1, 8, 1024, 1024, 64): 1.5e-4,
        (1, 16, 2048, 2048, 64): 3.0e-4,
        (2, 16, 2048, 2048, 64): 4.0e-4,
        (4, 16, 4096, 4096, 64): 8.0e-4,
    },
}

def plot_performance_comparison():
    """绘制性能对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 提取数据
    seq_lengths = [c[2] for c in configs]
    custom_times = [performance_data['Custom CUDA'][c] for c in configs]
    pytorch_times = [performance_data['PyTorch SDPA'][c] for c in configs]
    xformers_times = [performance_data['xformers'][c] for c in configs]

    x = np.arange(len(seq_lengths))
    width = 0.25

    bars1 = ax.bar(x - width, custom_times, width, label='Custom CUDA', color='#FF6B6B')
    bars2 = ax.bar(x, pytorch_times, width, label='PyTorch SDPA', color='#4ECDC4')
    bars3 = ax.bar(x + width, xformers_times, width, label='xformers', color='#45B7D1')

    ax.set_xlabel('Sequence Length (N=M)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('FlashAttention v2 Performance Comparison (A100)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in seq_lengths])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('flashattention_performance.png', dpi=150)
    print("Saved: flashattention_performance.png")
    return fig


def plot_accuracy_comparison():
    """绘制精度对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    seq_lengths = [c[2] for c in configs]
    custom_vs_pt = [accuracy_data['Custom vs PyTorch'][c] for c in configs]
    xformers_vs_pt = [accuracy_data['xformers vs PyTorch'][c] for c in configs]

    x = np.arange(len(seq_lengths))
    width = 0.35

    bars1 = ax.bar(x - width/2, custom_vs_pt, width, label='Custom CUDA vs PyTorch', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, xformers_vs_pt, width, label='xformers vs PyTorch', color='#45B7D1')

    ax.set_xlabel('Sequence Length (N=M)', fontsize=12)
    ax.set_ylabel('Max Abs Difference', fontsize=12)
    ax.set_title('Accuracy Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in seq_lengths])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0e}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('flashattention_accuracy.png', dpi=150)
    print("Saved: flashattention_accuracy.png")
    return fig


def plot_speedup_over_baseline():
    """绘制相对于PyTorch的加速比"""
    fig, ax = plt.subplots(figsize=(12, 6))

    seq_lengths = [c[2] for c in configs]

    # 计算加速比 (相对于PyTorch)
    custom_speedup = [performance_data['PyTorch SDPA'][c] / performance_data['Custom CUDA'][c] for c in configs]
    xformers_speedup = [performance_data['PyTorch SDPA'][c] / performance_data['xformers'][c] for c in configs]

    x = np.arange(len(seq_lengths))
    width = 0.35

    bars1 = ax.bar(x - width/2, custom_speedup, width, label='Custom CUDA', color='#FF6B6B')
    bars2 = ax.bar(x + width/2, xformers_speedup, width, label='xformers', color='#45B7D1')

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='PyTorch Baseline')

    ax.set_xlabel('Sequence Length (N=M)', fontsize=12)
    ax.set_ylabel('Speedup (vs PyTorch SDPA)', fontsize=12)
    ax.set_title('Speedup Relative to PyTorch SDPA (>1.0 is faster)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in seq_lengths])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('flashattention_speedup.png', dpi=150)
    print("Saved: flashattention_speedup.png")
    return fig


def print_summary():
    """打印性能摘要表"""
    print("\n" + "="*80)
    print("FlashAttention v2 Performance Summary (Simulated - A100 GPU)")
    print("="*80)
    print(f"{'Config':<25} {'Custom CUDA':>15} {'PyTorch SDPA':>15} {'xformers':>15}")
    print("-"*80)

    for config in configs:
        custom = performance_data['Custom CUDA'][config]
        pytorch = performance_data['PyTorch SDPA'][config]
        xformers = performance_data['xformers'][config]
        print(f"B={config[0]} H={config[1]} N={config[2]} D={config[4]:>3}  {custom:>12.1f}ms {pytorch:>12.1f}ms {xformers:>12.1f}ms")

    print("="*80)
    print("\nAccuracy (Max Abs Diff vs PyTorch):")
    print("-"*50)
    for config in configs:
        custom = accuracy_data['Custom vs PyTorch'][config]
        xformers = accuracy_data['xformers vs PyTorch'][config]
        print(f"N={config[2]}: Custom={custom:.0e}, xformers={xformers:.0e}")


if __name__ == "__main__":
    print("Generating simulated FlashAttention benchmark results...")
    print("Note: These are simulated results for demonstration.")
    print("      Run on a CUDA machine for actual measurements.\n")

    plot_performance_comparison()
    plot_accuracy_comparison()
    plot_speedup_over_baseline()
    print_summary()

    print("\n" + "="*80)
    print("Generated plots:")
    print("  1. flashattention_performance.png - Time comparison")
    print("  2. flashattention_accuracy.png   - Accuracy comparison")
    print("  3. flashattention_speedup.png   - Speedup vs PyTorch")
    print("="*80)
