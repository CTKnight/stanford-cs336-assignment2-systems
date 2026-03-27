from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, autograd
import triton
import triton.language as tl

def cdiv(a: int, b: int) -> int:
    return -(-a // b)


class FlashAttentionPytorch(autograd.Function):
    @staticmethod
    def flash_attention_forward(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = True,
        b_q: int = 16,
        b_k: int = 16,
    ) -> tuple[Tensor, Tensor]:
        s, d_k = q.shape[-2:]
        t, d_v = v.shape[-2:]
        scale = sqrt(d_k)
        o = q.new_zeros((*q.shape[:-1], d_v))
        lse = q.new_empty(q.shape[:-1])

        for i in range(cdiv(s, b_q)):
            i_tile_start = i * b_q
            i_tile_len = min((i + 1) * b_q, s) - i_tile_start
            q_i = q[..., i_tile_start : i_tile_start + i_tile_len, :]
            o_i = q.new_zeros((*q_i.shape[:-1], d_v))
            l_i = q.new_zeros(q_i.shape[:-1])
            m_i = q.new_full(q_i.shape[:-1], float("-inf"))

            for j in range(cdiv(t, b_k)):
                j_tile_start = j * b_k
                j_tile_len = min((j + 1) * b_k, t) - j_tile_start
                if is_causal and j_tile_start > i_tile_start + i_tile_len - 1:
                    continue

                k_j = k[..., j_tile_start : j_tile_start + j_tile_len, :]
                v_j = v[..., j_tile_start : j_tile_start + j_tile_len, :]
                s_ij = q_i @ k_j.transpose(-1, -2) / scale

                if is_causal:
                    q_idx = torch.arange(i_tile_start, i_tile_start + i_tile_len, device=q.device)
                    k_idx = torch.arange(j_tile_start, j_tile_start + j_tile_len, device=q.device)
                    causal_mask = q_idx[:, None] < k_idx[None, :]
                    s_ij = s_ij.masked_fill(causal_mask, float("-inf"))

                old_m_i = m_i
                m_i = torch.maximum(old_m_i, s_ij.amax(dim=-1))
                p_ij = torch.exp(s_ij - m_i.unsqueeze(-1))
                l_i = l_i * torch.exp(old_m_i - m_i) + p_ij.sum(dim=-1)
                o_i = o_i * torch.exp(old_m_i - m_i).unsqueeze(-1) + p_ij @ v_j

            o_i = o_i / l_i.unsqueeze(-1)
            o[..., i_tile_start : i_tile_start + i_tile_len, :] = o_i
            lse[..., i_tile_start : i_tile_start + i_tile_len] = m_i + torch.log(l_i)

        return o, lse

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = True) -> Tensor:
        o, lse = FlashAttentionPytorch.flash_attention_forward(q, k, v, is_causal=is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, lse)
        return o

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError()


class FlashAttentionTriton(autograd.Function):
    @staticmethod
    def flash_attention_forward(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = True,
        q_tile_size: int = 32,
        k_tile_size: int = 32,
    ) -> tuple[Tensor, Tensor]:
        if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
            raise ValueError("flash_attention_forward_triton expects CUDA tensors.")
        if q.ndim < 2 or k.ndim < 2 or v.ndim < 2:
            raise ValueError(f"Expected tensors shaped [..., seq, dim], got {q.shape=}, {k.shape=}, {v.shape=}.")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError(f"Expected matching leading dimensions, got {q.shape=}, {k.shape=}, {v.shape=}.")
        if q.shape[-1] != k.shape[-1]:
            raise ValueError(f"Expected q and k to have the same head dimension, got {q.shape[-1]} and {k.shape[-1]}.")
        if v.shape[-1] != q.shape[-1]:
            raise ValueError(f"Expected v.shape[-1] == q.shape[-1], got {v.shape[-1]} and {q.shape[-1]}.")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise ValueError(f"Expected matching dtypes, got {q.dtype=}, {k.dtype=}, {v.dtype=}.")

        leading_shape = q.shape[:-2]
        batch_size = 1
        for dim in leading_shape:
            batch_size *= dim

        q = q.contiguous().view(batch_size, q.shape[-2], q.shape[-1])
        k = k.contiguous().view(batch_size, k.shape[-2], k.shape[-1])
        v = v.contiguous().view(batch_size, v.shape[-2], v.shape[-1])

        _, n_queries, d = q.shape
        _, n_keys, _ = k.shape
        o = torch.empty_like(q)
        lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
        scale = sqrt(d)

        grid = (cdiv(n_queries, q_tile_size), batch_size)
        flash_attention_kernel[grid](
            q,
            k,
            v,
            o,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            lse.stride(0),
            lse.stride(1),
            n_queries,
            n_keys,
            scale,
            D=d,
            Q_TILE_SIZE=q_tile_size,
            K_TILE_SIZE=k_tile_size,
            is_causal=is_causal,
        )
        return o.view(*leading_shape, n_queries, d), lse.view(*leading_shape, n_queries)

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = True) -> Tensor:
        o, lse = FlashAttentionTriton.flash_attention_forward(q, k, v, is_causal=is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, lse)
        return o

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError()

@triton.jit
def flash_attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # i
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb, 
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb, 
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb, 
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    o_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE, ), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE, ), dtype=tl.float32)
    q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        S_ij = tl.dot(q_block, k_block.trans()) / scale
        k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        valid_mask = k_idx[None, :] < N_KEYS
        if is_causal:
            valid_mask = valid_mask & (q_idx[:, None] >= k_idx[None, :])
        S_ij = tl.where(valid_mask, S_ij, float("-inf"))
        old_m_i = m_i
        m_i = tl.maximum(old_m_i, tl.max(S_ij, axis=-1))
        p_ij = tl.exp(S_ij - m_i[:, None])
        l_i = l_i * tl.exp(old_m_i - m_i) + tl.sum(p_ij, axis=-1)
        alpha = tl.exp(old_m_i - m_i)
        o_i = o_i * alpha[:, None]
        o_i = tl.dot(p_ij.to(v_block.dtype), v_block, acc=o_i)
        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))
    o_i /= l_i[:, None]
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob, 
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    tl.store(O_block_ptr, o_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb, 
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)
    )
    tl.store(L_block_ptr, m_i + tl.log(l_i), boundary_check=(0))
