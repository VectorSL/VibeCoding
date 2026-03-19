# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hand-written CUDA implementation of FlashAttention v2, benchmarked against PyTorch SDPA and xformers. The goal is to iteratively optimize the CUDA kernels to close the performance gap with production implementations.

## Build & Run

All commands run from the `flashattention/` directory:

```bash
# Build the CUDA extension (requires CUDA toolkit + PyTorch with CUDA)
python setup.py build_ext --inplace

# Run all tests (correctness + performance)
python run.py

# Run only correctness or performance tests
python run.py --correctness
python run.py --performance

# Run with a specific config: B H N M D
python run.py --config 2 8 512 512 64

# Unit tests via pytest
python -m pytest test_attention.py -v

# Visualize results (requires matplotlib, needs CUDA)
python visualize_results.py
```

## Architecture

```
flashattention/
  ccsrc/
    flash_attention_fwd.cu   # Forward kernel - the main optimization target
    flash_attention_bwd.cu   # Backward kernel (stub in fwd.cu, full impl here but not wired up)
  python/
    attention.py             # Python bindings, benchmark harness, correctness comparison
  setup.py                   # PyTorch CUDAExtension build config
  run.py                     # CLI entry point
  test_attention.py          # pytest suite
  notes/
    optimization_rules.md    # Optimization workflow rules (read this on startup)
    perf_log.md              # Performance log tracking each optimization round
```

Key details:
- Tensors are `[B, H, N, D]` layout, float16. N = query seq len, M = KV seq len, D = head dim.
- Forward kernel: one CUDA block per query row, tiles over KV in blocks of `BLOCK_N=64`. Uses online softmax (running max + sum).
- `setup.py` only compiles `flash_attention_fwd.cu`. The backward file exists but isn't linked.
- `python/attention.py` wraps the CUDA extension via `FlashAttentionFunction` (autograd Function) and provides `benchmark_attention()` / `compare_outputs()` utilities.
- Correctness tolerance is `1e-2` (max abs diff vs PyTorch SDPA).

## Optimization Workflow

Read `flashattention/notes/optimization_rules.md` at the start of each session. Key rules:

1. One change at a time — isolate what caused the performance delta
2. Log every attempt in `notes/perf_log.md` with timing results
3. Only modify files under `ccsrc/` unless the benchmark harness itself is buggy
4. Run correctness tests before performance tests after every change
5. Commit after each effective optimization round
6. After 3 consecutive rounds with no progress, search the web for new optimization ideas

## Current Status

Baseline forward kernel: ~87ms for B=1, H=8, N=512, D=64. PyTorch SDPA: ~0.05ms (1700x gap). Attempted optimizations (all ineffective so far): increased threads, constant memory, larger blocks, loop unrolling, double buffering, warp-level parallelism, vectorized loads. The main bottleneck is lack of tensor core usage and kernel fusion.
