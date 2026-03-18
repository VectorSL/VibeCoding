#!/usr/bin/env python3
"""
Visualize FlashAttention test results with charts.

Usage:
    python visualize_results.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
import time
import numpy as np
from attention import (
    custom_flash_attention,
    pytorch_flash_attention,
    compare_outputs,
    HAS_CUSTOM_CUDA,
)


def run_correctness_tests():
    """Run all correctness tests and collect results."""
    print("Running correctness tests...")

    configs = [
        (1, 4, 128, 128, 32, "D=32"),
        (1, 8, 256, 256, 64, "256x256"),
        (1, 8, 512, 512, 64, "512x512"),
        (2, 8, 512, 512, 64, "B=2"),
        (1, 16, 1024, 1024, 64, "1024x1024"),
        (1, 4, 256, 128, 64, "Asymmetric"),
    ]

    results = []

    for B, H, N, M, D, name in configs:
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)

        custom_out = custom_flash_attention(Q, K, V)
        pytorch_out = pytorch_flash_attention(Q, K, V)

        comp = compare_outputs(custom_out, pytorch_out)
        results.append({
            'name': name,
            'B': B, 'H': H, 'N': N, 'M': M, 'D': D,
            'max_diff': comp['max_abs_diff'],
            'mean_diff': comp['mean_abs_diff'],
            'rel_error': comp['max_rel_error'],
            'passed': comp['max_abs_diff'] < 1e-2
        })

    return results


def run_performance_tests():
    """Run performance tests and collect results."""
    print("Running performance tests...")

    configs = [
        (1, 4, 128, 128, 64),
        (1, 8, 256, 256, 64),
        (1, 8, 512, 512, 64),
        (1, 8, 1024, 1024, 64),
        (1, 16, 1024, 1024, 64),
        (1, 16, 2048, 2048, 64),
        (2, 8, 512, 512, 64),
    ]

    results = []
    num_warmup = 5
    num_runs = 20

    for B, H, N, M, D in configs:
        Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        K = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)
        V = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(num_warmup):
            _ = pytorch_flash_attention(Q, K, V)
            _ = custom_flash_attention(Q, K, V)
        torch.cuda.synchronize()

        # Benchmark PyTorch
        pytorch_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = pytorch_flash_attention(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            pytorch_times.append((end - start) * 1000)

        # Benchmark Custom CUDA
        custom_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = custom_flash_attention(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            custom_times.append((end - start) * 1000)

        results.append({
            'name': f'{N}x{N}',
            'N': N,
            'pytorch_mean': np.mean(pytorch_times),
            'pytorch_std': np.std(pytorch_times),
            'custom_mean': np.mean(custom_times),
            'custom_std': np.std(custom_times),
            'speedup': np.mean(pytorch_times) / np.mean(custom_times)
        })

    return results


def print_table_results(correctness_results, performance_results):
    """Print results in table format."""
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST RESULTS")
    print("=" * 80)
    print(f"{'Config':<25} {'B':>3} {'H':>3} {'N':>5} {'M':>5} {'D':>3} {'Max Diff':>10} {'Mean Diff':>10} {'Status':>8}")
    print("-" * 80)
    for r in correctness_results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{r['name']:<25} {r['B']:>3} {r['H']:>3} {r['N']:>5} {r['M']:>5} {r['D']:>3} {r['max_diff']:>10.6f} {r['mean_diff']:>10.6f} {status:>8}")

    print("\n" + "=" * 80)
    print("PERFORMANCE TEST RESULTS (ms)")
    print("=" * 80)
    print(f"{'Config':<15} {'N':>6} {'PyTorch':>12} {'Custom CUDA':>12} {'Speedup':>10}")
    print("-" * 80)
    for r in performance_results:
        print(f"{r['name']:<15} {r['N']:>6} {r['pytorch_mean']:>10.3f}±{r['pytorch_std']:.2f} {r['custom_mean']:>10.3f}±{r['custom_std']:.2f} {r['speedup']:>10.4f}x")


def generate_ascii_charts(correctness_results, performance_results):
    """Generate ASCII bar charts."""
    print("\n" + "=" * 80)
    print("CORRECTNESS - Max Absolute Difference")
    print("=" * 80)

    max_diff_max = max(r['max_diff'] for r in correctness_results)
    bar_width = 40

    for r in correctness_results:
        bar_len = int((r['max_diff'] / max_diff_max) * bar_width) if max_diff_max > 0 else 0
        bar = '*' * bar_len
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{r['name']:<20} |{bar:<40}| {r['max_diff']:.6f} {status}")

    print("\n" + "=" * 80)
    print("PERFORMANCE - Execution Time (ms)")
    print("=" * 80)

    time_max = max(max(r['pytorch_mean'], r['custom_mean']) for r in performance_results)
    bar_width = 40

    print(f"{'Config':<15} {'PyTorch':<25} {'Custom CUDA':<25}")
    for r in performance_results:
        pytorch_len = int((r['pytorch_mean'] / time_max) * bar_width)
        custom_len = int((r['custom_mean'] / time_max) * bar_width)
        pytorch_bar = '#' * pytorch_len
        custom_bar = '-' * custom_len
        print(f"{r['name']:<15} |{pytorch_bar:<40}| {r['pytorch_mean']:.3f}ms")
        print(f"{'':<15} |{custom_bar:<40}| {r['custom_mean']:.3f}ms")

    print("\n" + "=" * 80)
    print("PERFORMANCE - Speedup (PyTorch / Custom CUDA)")
    print("=" * 80)

    speedup_max = max(r['speedup'] for r in performance_results)
    for r in performance_results:
        bar_len = int((r['speedup'] / speedup_max) * bar_width) if speedup_max > 0 else 0
        bar = '#' * bar_len
        color = "slow" if r['speedup'] < 1 else "fast"
        print(f"{r['name']:<15} |{bar:<40}| {r['speedup']:.4f}x ({color})")


def save_results_to_file(correctness_results, performance_results):
    """Save results to CSV files."""
    import csv

    # Save correctness results
    with open('test_results_correctness.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'B', 'H', 'N', 'M', 'D', 'max_diff', 'mean_diff', 'rel_error', 'passed'])
        writer.writeheader()
        for r in correctness_results:
            writer.writerow(r)

    # Save performance results
    with open('test_results_performance.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'N', 'pytorch_mean', 'pytorch_std', 'custom_mean', 'custom_std', 'speedup'])
        writer.writeheader()
        for r in performance_results:
            writer.writerow(r)

    print("\nResults saved to test_results_correctness.csv and test_results_performance.csv")


def main():
    print("=" * 80)
    print("FlashAttention v2 - Test Results Visualization")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Custom CUDA: {HAS_CUSTOM_CUDA}")

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    # Run tests
    correctness_results = run_correctness_tests()
    performance_results = run_performance_tests()

    # Print table results
    print_table_results(correctness_results, performance_results)

    # Generate ASCII charts
    generate_ascii_charts(correctness_results, performance_results)

    # Save to CSV
    save_results_to_file(correctness_results, performance_results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    passed = sum(1 for r in correctness_results if r['passed'])
    total = len(correctness_results)
    print(f"Correctness: {passed}/{total} tests passed")

    avg_speedup = np.mean([r['speedup'] for r in performance_results])
    print(f"Average speedup: {avg_speedup:.4f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
