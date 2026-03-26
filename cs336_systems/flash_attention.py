from __future__ import annotations

from math import sqrt

import torch
from torch import Tensor, autograd


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def flash_attention_forward_pytorch(
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


class FlashAttentionPytorch(autograd.Function):
    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = True) -> Tensor:
        o, lse = flash_attention_forward_pytorch(q, k, v, is_causal=is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, lse)
        return o

    @staticmethod
    def backward(ctx, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        raise NotImplementedError()
