from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from typing import Callable

import einops
import torch

from cs336_basics.model import attention as attention_module
from cs336_basics.model import norm as norm_module
from cs336_basics.model import swiglu as swiglu_module
from cs336_basics.model.loss import cross_entropy
from cs336_basics.optimizer.adamw import AdamW

from cs336_systems.benchmarking import (
    MODEL_SIZES,
    ROPE_THETA,
    VOCAB_SIZE,
    ModelConfig,
    build_model,
    make_random_batch,
    resolve_device,
    resolve_dtype,
    set_seed,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single NVTX-annotated step for Nsight Systems.")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="small")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--rope-theta", type=float, default=ROPE_THETA)
    parser.add_argument("--profile-mode", choices=["forward", "forward-backward", "train-step"], default="train-step")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


@contextmanager
def nvtx_range(message: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


class NvtxInstrumentor:
    def __init__(self) -> None:
        self._restorers: list[Callable[[], None]] = []

    def patch_attr(self, obj: object, attr_name: str, wrapper: Callable) -> None:
        original = getattr(obj, attr_name)
        setattr(obj, attr_name, wrapper(original))
        self._restorers.append(lambda: setattr(obj, attr_name, original))

    def restore(self) -> None:
        for restore in reversed(self._restorers):
            restore()
        self._restorers.clear()

    def __enter__(self) -> "NvtxInstrumentor":
        def wrap_scaled_dot_product_attention(original):
            def wrapped(Q, K, V, mask=None):
                with nvtx_range("attention/qk_matmul"):
                    qk = einops.einsum(Q, K, "... q d_k, ... k d_k -> ... q k")
                with nvtx_range("attention/mask"):
                    masked = qk if mask is None else qk.masked_fill(~mask, float("-inf"))
                with nvtx_range("attention/softmax"):
                    a = attention_module.softmax(x=masked / (Q.shape[-1] ** 0.5), dim=-1)
                with nvtx_range("attention/av_matmul"):
                    return einops.einsum(a, V, "... q k, ... k d_v -> ... q d_v")

            return wrapped

        def wrap_attention_forward(original):
            def wrapped(self, x, token_positions=None):
                with nvtx_range("attention/qkv_proj"):
                    qkv = einops.einsum(self.qkv_proj, x, "d_out d_in, ... s d_in -> ... s d_out")
                q, k, v = qkv.chunk(3, dim=-1)
                with nvtx_range("attention/rearrange_qkv"):
                    q = einops.rearrange(q, "... s (h d) -> ... h s d", h=self.num_heads)
                    k = einops.rearrange(k, "... s (h d) -> ... h s d", h=self.num_heads)
                    v = einops.rearrange(v, "... s (h d) -> ... h s d", h=self.num_heads)
                s = x.shape[-2]
                if s > self.max_seq_len:
                    raise ValueError(f"sequence length {s} exceeds max_seq_len {self.max_seq_len}")
                if token_positions is None:
                    pos = torch.arange(s, device=x.device)
                    token_positions = pos.view(*([1] * (q.ndim - 2)), s).expand(*q.shape[:-2], s)
                elif token_positions.shape == (*x.shape[:-2], s):
                    token_positions = token_positions.unsqueeze(-2).expand(*x.shape[:-2], self.num_heads, s)
                with nvtx_range("attention/rope"):
                    q_roped = self.rope(q, token_positions)
                    k_roped = self.rope(k, token_positions)
                mask = self._causal_mask[:s, :s]
                with nvtx_range("attention/sdpa"):
                    attn = attention_module.scaled_dot_product_attention(q_roped, k_roped, v, mask)
                with nvtx_range("attention/concat_heads"):
                    attn = einops.rearrange(attn, "... h s d -> ... s (h d)")
                with nvtx_range("attention/o_proj"):
                    return einops.einsum(self.o_proj, attn, "d_out d_in, ... d_in -> ... d_out")

            return wrapped

        def wrap_rmsnorm_forward(original):
            def wrapped(self, x):
                with nvtx_range("norm/rmsnorm"):
                    return original(self, x)

            return wrapped

        def wrap_swiglu_forward(original):
            def wrapped(self, x):
                with nvtx_range("ffn/w1_proj"):
                    gate_linear = einops.einsum(self._w1, x, "d_ff d_model, ... d_model -> ... d_ff")
                with nvtx_range("ffn/silu"):
                    gate = swiglu_module.SwiGLU.silu(gate_linear)
                with nvtx_range("ffn/w3_proj"):
                    value = einops.einsum(self._w3, x, "d_ff d_model, ... d_model -> ... d_ff")
                with nvtx_range("ffn/gate_mul"):
                    glu = gate * value
                with nvtx_range("ffn/w2_proj"):
                    return einops.einsum(self._w2, glu, "d_model d_ff, ... d_ff -> ... d_model")

            return wrapped

        self.patch_attr(attention_module, "scaled_dot_product_attention", wrap_scaled_dot_product_attention)
        self.patch_attr(attention_module.MultiHeadSelfAttention, "forward", wrap_attention_forward)
        self.patch_attr(norm_module.RMSNorm, "forward", wrap_rmsnorm_forward)
        self.patch_attr(swiglu_module.SwiGLU, "forward", wrap_swiglu_forward)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()


def compute_loss(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with nvtx_range("forward/model"):
        logits = model(inputs)
    vocab_size = logits.shape[-1]
    with nvtx_range("forward/loss"):
        return cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def run_profiled_iteration(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    profile_mode: str,
    device: torch.device,
) -> None:
    if profile_mode in {"forward-backward", "train-step"}:
        model.zero_grad(set_to_none=True)

    synchronize(device)
    if profile_mode == "forward":
        with nvtx_range("forward_pass"):
            with torch.no_grad():
                _ = compute_loss(model, inputs, targets)
    else:
        with nvtx_range("forward_pass"):
            loss = compute_loss(model, inputs, targets)
        with nvtx_range("backward_pass"):
            loss.backward()
        if profile_mode == "train-step":
            if optimizer is None:
                raise ValueError("optimizer is required for train-step profiling")
            with nvtx_range("optimizer_step"):
                optimizer.step()
    synchronize(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    model_config: ModelConfig = MODEL_SIZES[args.model_size]

    if device.type != "cuda":
        raise ValueError("Nsight Systems profiling in this script expects --device cuda.")
    if args.dtype == "float16" and device.type == "cpu":
        raise ValueError("float16 profiling on CPU is not supported.")
    if model_config.d_model % model_config.num_heads != 0:
        raise ValueError("num_heads must divide d_model")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = build_model(
        config=model_config,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        rope_theta=args.rope_theta,
        device=device,
        dtype=dtype,
        compile_model=False,
        compile_mode="default",
    )
    optimizer = None
    if args.profile_mode == "train-step":
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
    inputs, targets = make_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )

    print(
        " ".join(
            [
                f"profile_mode={args.profile_mode}",
                f"model_size={args.model_size}",
                f"context_length={args.context_length}",
                f"batch_size={args.batch_size}",
                f"dtype={args.dtype}",
                f"device={device}",
                *[f"{key}={value}" for key, value in asdict(model_config).items()],
            ]
        ),
        flush=True,
    )

    with NvtxInstrumentor():
        for _ in range(args.warmup_steps):
            with nvtx_range("warmup"):
                run_profiled_iteration(model, optimizer, inputs, targets, args.profile_mode, device)
        run_profiled_iteration(model, optimizer, inputs, targets, args.profile_mode, device)


if __name__ == "__main__":
    main()
