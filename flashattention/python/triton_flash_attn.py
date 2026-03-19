"""
Triton Flash Attention v2 implementation.
Matches the same interface as the custom CUDA kernel: Q, K, V -> O
Input shape: [B, H, N, D], dtype=float16
"""

import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_KV': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_KV': 128}, num_warps=8, num_stages=2),
    ],
    key=['N', 'M', 'D'],
)
@triton.jit
def _flash_attention_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_km, stride_kd,
    stride_vb, stride_vh, stride_vm, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, M, D: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr, BLOCK_D: tl.constexpr,
    EVEN_M: tl.constexpr, EVEN_D: tl.constexpr,
    softmax_scale,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    num_heads = stride_qb // stride_qh
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads

    q_base = pid_b * stride_qb + pid_h * stride_qh
    k_base = pid_b * stride_kb + pid_h * stride_kh
    v_base = pid_b * stride_vb + pid_h * stride_vh
    o_base = pid_b * stride_ob + pid_h * stride_oh

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q tile and pre-scale
    q_ptrs = Q_ptr + q_base + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    if EVEN_D:
        q = tl.load(q_ptrs, mask=offs_q[:, None] < N, other=0.0)
    else:
        q = tl.load(q_ptrs, mask=(offs_q[:, None] < N) & (offs_d[None, :] < D), other=0.0)
    q = (q * softmax_scale).to(q.dtype)

    # Initialize accumulators
    m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Iterate over KV blocks
    num_kv_blocks = tl.cdiv(M, BLOCK_KV)
    for kv_block in range(num_kv_blocks):
        kv_start = kv_block * BLOCK_KV
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)

        # Load K tile
        k_ptrs = K_ptr + k_base + offs_kv[:, None] * stride_km + offs_d[None, :] * stride_kd
        if EVEN_M and EVEN_D:
            k = tl.load(k_ptrs)
        elif EVEN_D:
            k = tl.load(k_ptrs, mask=offs_kv[:, None] < M, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_kv[:, None] < M) & (offs_d[None, :] < D), other=0.0)

        # QK^T (Q already scaled)
        s = tl.dot(q, tl.trans(k))

        # Mask out-of-bounds KV
        if not EVEN_M:
            s = tl.where(offs_kv[None, :] < M, s, float('-inf'))

        # Online softmax
        m_ij = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        o_acc = o_acc * alpha[:, None]

        # Load V tile
        v_ptrs = V_ptr + v_base + offs_kv[:, None] * stride_vm + offs_d[None, :] * stride_vd
        if EVEN_M and EVEN_D:
            v = tl.load(v_ptrs)
        elif EVEN_D:
            v = tl.load(v_ptrs, mask=offs_kv[:, None] < M, other=0.0)
        else:
            v = tl.load(v_ptrs, mask=(offs_kv[:, None] < M) & (offs_d[None, :] < D), other=0.0)

        o_acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # Final normalization
    o_acc = o_acc / l_i[:, None]

    # Store output
    o_ptrs = O_ptr + o_base + offs_q[:, None] * stride_on + offs_d[None, :] * stride_od
    if EVEN_D:
        tl.store(o_ptrs, o_acc.to(tl.float16), mask=offs_q[:, None] < N)
    else:
        tl.store(o_ptrs, o_acc.to(tl.float16), mask=(offs_q[:, None] < N) & (offs_d[None, :] < D))


def triton_flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Triton Flash Attention forward pass with autotune.

    Args:
        Q: [B, H, N, D] float16
        K: [B, H, M, D] float16
        V: [B, H, M, D] float16

    Returns:
        O: [B, H, N, D] float16
    """
    B, H, N, D = Q.shape
    M = K.shape[2]

    assert Q.dtype == torch.float16
    assert K.dtype == torch.float16
    assert V.dtype == torch.float16

    O = torch.empty_like(Q)

    softmax_scale = 1.0 / math.sqrt(D)

    BLOCK_D = triton.next_power_of_2(D)
    EVEN_D = (D == BLOCK_D)

    # EVEN_M: M is divisible by all possible BLOCK_KV values (64, 128)
    EVEN_M = (M % 128 == 0)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_Q']), B * H)

    _flash_attention_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, M, D,
        BLOCK_D=BLOCK_D,
        EVEN_M=EVEN_M,
        EVEN_D=EVEN_D,
        softmax_scale=softmax_scale,
    )

    return O
