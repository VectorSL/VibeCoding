#!/usr/bin/env python3
"""
Unit tests for FlashAttention CUDA implementation.

Usage:
    python -m pytest test_attention.py -v
    python test_attention.py  # Run directly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
from attention import (
    custom_flash_attention,
    pytorch_flash_attention,
    compare_outputs,
    HAS_CUSTOM_CUDA,
)


class TestFlashAttention:
    """Test suite for FlashAttention implementation."""

    @classmethod
    def setup_class(cls):
        """Setup test class."""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

    def test_small_sequence(self):
        """Test with small sequence length."""
        B, H, N, M, D = 1, 4, 32, 32, 32
        Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
        K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
        V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

        custom_out = custom_flash_attention(Q, K, V)
        pytorch_out = pytorch_flash_attention(Q, K, V)

        comp = compare_outputs(custom_out, pytorch_out)
        assert comp["max_abs_diff"] < 1e-2, f"Max diff {comp['max_abs_diff']} too large"

    def test_medium_sequence(self):
        """Test with medium sequence length."""
        B, H, N, M, D = 1, 8, 256, 256, 64
        Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
        K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
        V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

        custom_out = custom_flash_attention(Q, K, V)
        pytorch_out = pytorch_flash_attention(Q, K, V)

        comp = compare_outputs(custom_out, pytorch_out)
        assert comp["max_abs_diff"] < 1e-2, f"Max diff {comp['max_abs_diff']} too large"

    def test_large_sequence(self):
        """Test with large sequence length."""
        B, H, N, M, D = 1, 8, 512, 512, 64
        Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
        K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
        V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

        custom_out = custom_flash_attention(Q, K, V)
        pytorch_out = pytorch_flash_attention(Q, K, V)

        comp = compare_outputs(custom_out, pytorch_out)
        assert comp["max_abs_diff"] < 1e-2, f"Max diff {comp['max_abs_diff']} too large"

    def test_different_head_dimensions(self):
        """Test with different head dimensions."""
        for D in [32, 64]:
            B, H, N, M = 1, 4, 128, 128
            Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
            K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
            V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

            custom_out = custom_flash_attention(Q, K, V)
            pytorch_out = pytorch_flash_attention(Q, K, V)

            comp = compare_outputs(custom_out, pytorch_out)
            assert comp["max_abs_diff"] < 1e-2, f"D={D}: Max diff {comp['max_abs_diff']} too large"

    def test_batch_size(self):
        """Test with different batch sizes."""
        for B in [1, 2, 4]:
            H, N, M, D = 4, 128, 128, 64
            Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
            K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
            V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

            custom_out = custom_flash_attention(Q, K, V)
            pytorch_out = pytorch_flash_attention(Q, K, V)

            comp = compare_outputs(custom_out, pytorch_out)
            assert comp["max_abs_diff"] < 1e-2, f"B={B}: Max diff {comp['max_abs_diff']} too large"

    def test_multiple_heads(self):
        """Test with different number of heads."""
        for H in [1, 4, 8, 16]:
            B, N, M, D = 1, 128, 128, 64
            Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
            K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
            V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

            custom_out = custom_flash_attention(Q, K, V)
            pytorch_out = pytorch_flash_attention(Q, K, V)

            comp = compare_outputs(custom_out, pytorch_out)
            assert comp["max_abs_diff"] < 1e-2, f"H={H}: Max diff {comp['max_abs_diff']} too large"

    def test_asymmetric_sequence_lengths(self):
        """Test with different Q and KV sequence lengths."""
        B, H, N, M, D = 1, 4, 256, 128, 64
        Q = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
        K = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
        V = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

        custom_out = custom_flash_attention(Q, K, V)
        pytorch_out = pytorch_flash_attention(Q, K, V)

        comp = compare_outputs(custom_out, pytorch_out)
        assert comp["max_abs_diff"] < 1e-2, f"Max diff {comp['max_abs_diff']} too large"

    def test_deterministic(self):
        """Test that results are deterministic with fixed seed."""
        B, H, N, M, D = 1, 4, 128, 128, 64

        torch.manual_seed(42)
        Q1 = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
        K1 = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
        V1 = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

        torch.manual_seed(42)
        Q2 = torch.randn(B, H, N, D, device=self.device, dtype=torch.float16)
        K2 = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)
        V2 = torch.randn(B, H, M, D, device=self.device, dtype=torch.float16)

        out1 = custom_flash_attention(Q1, K1, V1)
        out2 = custom_flash_attention(Q2, K2, V2)

        assert torch.allclose(out1, out2, atol=1e-6), "Results are not deterministic"


def run_tests():
    """Run tests directly without pytest."""
    print("="*60)
    print("FlashAttention CUDA Unit Tests")
    print("="*60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Custom CUDA available: {HAS_CUSTOM_CUDA}")
    print("="*60)

    if not torch.cuda.is_available():
        print("SKIPPED: CUDA not available")
        return

    if not HAS_CUSTOM_CUDA:
        print("SKIPPED: Custom CUDA not compiled")
        return

    test = TestFlashAttention()
    test.setup_class()

    tests = [
        ("Small Sequence", test.test_small_sequence),
        ("Medium Sequence", test.test_medium_sequence),
        ("Large Sequence", test.test_large_sequence),
        ("Different Head Dimensions", test.test_different_head_dimensions),
        ("Batch Size", test.test_batch_size),
        ("Multiple Heads", test.test_multiple_heads),
        ("Asymmetric Sequence Lengths", test.test_asymmetric_sequence_lengths),
        ("Deterministic", test.test_deterministic),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name} - {e}")
            failed += 1

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
