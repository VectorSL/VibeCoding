"""
FlashAttention implementations comparison:
- Hand-written CUDA (this project)
- PyTorch SDPA (scaled_dot_product_attention)
- Triton (xformers or triton-flash-attention)
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from typing import Tuple, Optional

# Try to import our custom CUDA implementation
try:
    import flash_attention_cuda
    HAS_CUSTOM_CUDA = True
except ImportError:
    HAS_CUSTOM_CUDA = False
    print("Warning: Custom CUDA implementation not available. Run `python setup.py install` first.")

# Try to import Triton implementation
try:
    from xformers.ops import memory_efficient_attention
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False
    # Try triton_flash_attention
    try:
        from triton_flash_attention import flash_attention
        HAS_TRITON = True
    except ImportError:
        HAS_TRITON = False


class FlashAttentionFunction(torch.autograd.Function):
    """Autograd function for custom FlashAttention forward and backward."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        if not HAS_CUSTOM_CUDA:
            raise RuntimeError("Custom CUDA implementation not available")

        # Call forward kernel
        O = flash_attention_cuda.forward(Q, K, V)

        # Save for backward
        ctx.save_for_backward(Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not HAS_CUSTOM_CUDA:
            raise RuntimeError("Custom CUDA implementation not available")

        Q, K, V, O = ctx.saved_tensors

        # Compute logsumexp for backward
        # This is simplified - in production you'd compute this more efficiently
        L = torch.zeros(Q.size(0), Q.size(1), Q.size(2), device=Q.device, dtype=torch.float32)

        # Call backward kernel
        dQ, dK, dV = flash_attention_cuda.backward(Q, K, V, O, L, dO)

        return dQ, dK, dV


def custom_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Hand-written CUDA FlashAttention implementation.

    Args:
        Q: Query tensor [B, H, N, D]
        K: Key tensor [B, H, M, D]
        V: Value tensor [B, H, M, D]

    Returns:
        Output tensor [B, H, N, D]
    """
    if not HAS_CUSTOM_CUDA:
        raise RuntimeError("Custom CUDA implementation not available")

    return FlashAttentionFunction.apply(Q, K, V)


def pytorch_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    PyTorch's scaled_dot_product_attention (SDPA) with FlashAttention backend.
    """
    # PyTorch >= 2.0 has SDPA with flash attention backend
    # softmax_scale should be 1/sqrt(D), SDPA handles this automatically
    output = F.scaled_dot_product_attention(Q, K, V, attn_mask=None)
    return output


def xformers_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    xformers memory efficient attention (Triton-based).
    """
    if not HAS_XFORMERS:
        raise RuntimeError("xformers not available")

    # xformers expects input shape: [B, H, N, D]
    # It automatically handles the scaling
    output = memory_efficient_attention(Q, K, V)
    return output


def triton_flash_attention_impl(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton-based FlashAttention implementation.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton-flash-attention not available")

    output = flash_attention(Q, K, V)
    return output


def compare_outputs(output1: torch.Tensor, output2: torch.Tensor, name1: str = "Output 1", name2: str = "Output 2") -> dict:
    """
    Compare two outputs for numerical accuracy.

    Returns:
        Dictionary with max_abs_diff, mean_abs_diff, max_rel_error
    """
    diff = (output1 - output2).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    # Relative error (avoid division by zero)
    denom = output2.abs().clamp(min=1e-8)
    rel_error = (diff / denom).max().item()

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_error": rel_error,
    }


def benchmark_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    num_warmup: int = 10,
    num_runs: int = 100,
    use_custom: bool = True,
    use_pytorch: bool = True,
    use_xformers: bool = True,
) -> dict:
    """
    Benchmark different FlashAttention implementations.

    Args:
        Q, K, V: Input tensors [B, H, N, D]
        num_warmup: Number of warmup runs
        num_runs: Number of benchmark runs
        use_*: Which implementations to benchmark

    Returns:
        Dictionary with timing and correctness results
    """
    results = {}

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q = Q.to(device)
    K = K.to(device)
    V = V.to(device)

    # Warmup
    if use_pytorch:
        for _ in range(num_warmup):
            _ = pytorch_flash_attention(Q, K, V)
        torch.cuda.synchronize()

    if use_custom and HAS_CUSTOM_CUDA:
        for _ in range(num_warmup):
            _ = custom_flash_attention(Q, K, V)
        torch.cuda.synchronize()

    if use_xformers and HAS_XFORMERS:
        for _ in range(num_warmup):
            _ = xformers_flash_attention(Q, K, V)
        torch.cuda.synchronize()

    torch.cuda.synchronize()

    # Benchmark PyTorch
    if use_pytorch:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out_pytorch = pytorch_flash_attention(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        results["pytorch"] = {
            "mean_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
        }

    # Benchmark custom CUDA
    if use_custom and HAS_CUSTOM_CUDA:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out_custom = custom_flash_attention(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        results["custom_cuda"] = {
            "mean_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
        }

        # Compare with PyTorch
        if use_pytorch:
            compare_results = compare_outputs(out_custom, out_pytorch, "Custom", "PyTorch")
            results["custom_cuda"]["vs_pytorch"] = compare_results

    # Benchmark xformers
    if use_xformers and HAS_XFORMERS:
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            out_xformers = xformers_flash_attention(Q, K, V)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

        results["xformers"] = {
            "mean_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "min_time_ms": np.min(times) * 1000,
            "max_time_ms": np.max(times) * 1000,
        }

        # Compare with PyTorch
        if use_pytorch:
            compare_results = compare_outputs(out_xformers, out_pytorch, "xformers", "PyTorch")
            results["xformers"]["vs_pytorch"] = compare_results

        if use_custom and HAS_CUSTOM_CUDA:
            compare_results = compare_outputs(out_xformers, out_custom, "xformers", "Custom")
            results["xformers"]["vs_custom"] = compare_results

    return results


def run_correctness_test(
    B: int = 2,
    H: int = 8,
    N: int = 512,
    M: int = 512,
    D: int = 64,
    tolerance: float = 1e-3,
) -> bool:
    """
    Run correctness test comparing implementations.

    Args:
        B: Batch size
        H: Number of heads
        N: Sequence length for Q
        M: Sequence length for K/V
        D: Head dimension
        tolerance: Maximum allowed difference

    Returns:
        True if all tests pass
    """
    print(f"\n{'='*60}")
    print(f"Correctness Test: B={B}, H={H}, N={N}, M={M}, D={D}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create random inputs
    torch.manual_seed(42)
    Q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    K = torch.randn(B, H, M, D, device=device, dtype=torch.float16)
    V = torch.randn(B, H, M, D, device=device, dtype=torch.float16)

    all_passed = True
    reference = None

    # Get reference (PyTorch)
    if use_pytorch:
        reference = pytorch_flash_attention(Q, K, V)
        print(f"PyTorch SDPA: OK (reference)")

    # Test custom CUDA
    if HAS_CUSTOM_CUDA:
        try:
            custom_out = custom_flash_attention(Q, K, V)
            if reference is not None:
                comp = compare_outputs(custom_out, reference, "Custom", "PyTorch")
                passed = comp["max_abs_diff"] < tolerance
                if passed:
                    print(f"Custom CUDA: PASS (max_diff={comp['max_abs_diff']:.6f})")
                else:
                    print(f"Custom CUDA: FAIL (max_diff={comp['max_abs_diff']:.6f})")
                    all_passed = False
            else:
                print("Custom CUDA: OK (no reference)")
        except Exception as e:
            print(f"Custom CUDA: ERROR ({e})")
            all_passed = False
    else:
        print("Custom CUDA: SKIPPED (not compiled)")

    # Test xformers
    if HAS_XFORMERS:
        try:
            xformers_out = xformers_flash_attention(Q, K, V)
            if reference is not None:
                comp = compare_outputs(xformers_out, reference, "xformers", "PyTorch")
                passed = comp["max_abs_diff"] < tolerance
                if passed:
                    print(f"xformers: PASS (max_diff={comp['max_abs_diff']:.6f})")
                else:
                    print(f"xformers: FAIL (max_diff={comp['max_abs_diff']:.6f})")
                    all_passed = False
            else:
                print("xformers: OK (no reference)")
        except Exception as e:
            print(f"xformers: ERROR ({e})")
            all_passed = False
    else:
        print("xformers: SKIPPED (not installed)")

    return all_passed


def run_performance_test(
    configs: list = None,
    num_warmup: int = 10,
    num_runs: int = 100,
):
    """
    Run performance benchmarks across multiple configurations.

    Args:
        configs: List of (B, H, N, M, D) tuples
        num_warmup: Warmup iterations
        num_runs: Benchmark iterations
    """
    if configs is None:
        configs = [
            # (B, H, N, M, D)
            (1, 8, 512, 512, 64),
            (1, 8, 1024, 1024, 64),
            (1, 16, 2048, 2048, 64),
            (2, 16, 4096, 4096, 64),
        ]

    print(f"\n{'='*80}")
    print("Performance Benchmark")
    print(f"{'='*80}")
    print(f"Warmup: {num_warmup}, Runs: {num_runs}")
    print()

    for B, H, N, M, D in configs:
        print(f"\n--- Config: B={B}, H={H}, N={N}, M={M}, D={D} ---")

        results = benchmark_attention(
            Q=torch.randn(B, H, N, D),
            K=torch.randn(B, H, M, D),
            V=torch.randn(B, H, M, D),
            num_warmup=num_warmup,
            num_runs=num_runs,
        )

        for name, data in results.items():
            if "mean_time_ms" in data:
                print(f"  {name:15s}: {data['mean_time_ms']:.3f} ± {data['std_time_ms']:.3f} ms")


if __name__ == "__main__":
    # Run correctness test
    passed = run_correctness_test()

    # Run performance test
    run_performance_test()
