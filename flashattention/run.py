#!/usr/bin/env python3
"""
FlashAttention v2 Benchmark Runner

Usage:
    python run.py                    # Run all tests
    python run.py --correctness      # Only correctness tests
    python run.py --performance      # Only performance tests
    python run.py --build            # Build the CUDA extension
"""

import argparse
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
from attention import (
    run_correctness_test,
    run_performance_test,
    benchmark_attention,
    HAS_CUSTOM_CUDA,
    HAS_XFORMERS,
    HAS_TRITON,
)


def build_extension():
    """Build the CUDA extension."""
    print("Building CUDA extension...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=os.path.dirname(__file__),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Build failed:\n{result.stderr}")
        return False
    print("Build successful!")
    return True


def main():
    parser = argparse.ArgumentParser(description="FlashAttention v2 Benchmark")
    parser.add_argument("--build", action="store_true", help="Build CUDA extension first")
    parser.add_argument("--correctness", action="store_true", help="Run correctness tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--config", nargs="+", type=int,
                        help="Config: B H N M D (e.g., 2 8 512 512 64)")

    args = parser.parse_args()

    # Check CUDA availability
    print("="*60)
    print("FlashAttention v2 Benchmark")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
    print(f"Custom CUDA available: {HAS_CUSTOM_CUDA}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"xformers available: {HAS_XFORMERS}")
    print("="*60)

    # Build if requested
    if args.build:
        if not build_extension():
            print("Failed to build extension")
            return 1

    # Determine what to run
    run_all = args.all or (not args.correctness and not args.performance)

    # Run correctness tests
    if run_all or args.correctness:
        print("\n" + "="*60)
        print("CORRECTNESS TESTS")
        print("="*60)

        # Test different configurations
        configs = [
            (1, 4, 128, 128, 32),
            (1, 8, 256, 256, 64),
            (1, 8, 512, 512, 64),
            (2, 8, 512, 512, 64),
            (1, 16, 1024, 1024, 64),
        ]

        for B, H, N, M, D in configs:
            passed = run_correctness_test(B=B, H=H, N=N, M=M, D=D, tolerance=1e-2)
            if not passed:
                print(f"FAILED at config B={B}, H={H}, N={N}, M={M}, D={D}")
                if not args.performance:
                    return 1

    # Run performance tests
    if run_all or args.performance:
        print("\n" + "="*60)
        print("PERFORMANCE TESTS")
        print("="*60)

        if args.config and len(args.config) == 5:
            B, H, N, M, D = args.config
            print(f"\nSingle config: B={B}, H={H}, N={N}, M={M}, D={D}")
            run_performance_test([(B, H, N, M, D)])
        else:
            # Default performance test configs
            configs = [
                (1, 8, 512, 512, 64),
                (1, 8, 1024, 1024, 64),
                (1, 16, 2048, 2048, 64),
                (2, 16, 2048, 2048, 64),
                (4, 16, 4096, 4096, 64),
            ]
            run_performance_test(configs)

    print("\n" + "="*60)
    print("Done!")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
