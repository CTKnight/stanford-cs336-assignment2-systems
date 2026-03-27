from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton.testing

from cs336_basics.model.attention import scaled_dot_product_attention
from cs336_systems.flash_attention import FlashAttentionTriton


DEFAULT_SEQUENCE_LENGTHS = [2**i for i in range(7, 17)]
DEFAULT_DIMS = [2**i for i in range(4, 8)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton FlashAttention forward against regular PyTorch attention."
    )
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--dtypes", nargs="+", choices=["bfloat16", "float32"], default=["bfloat16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for full benchmark results.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=25, help="triton.testing.do_bench warmup in ms.")
    parser.add_argument("--rep", type=int, default=100, help="triton.testing.do_bench repetition window in ms.")
    parser.add_argument("--q-tile-size", type=int, default=None)
    parser.add_argument("--k-tile-size", type=int, default=None)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def regular_attention_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
    if not is_causal:
        raise ValueError("This benchmark expects causal attention.")
    q_len = q.shape[-2]
    k_len = k.shape[-2]
    causal_mask = torch.arange(q_len, device=q.device)[:, None] >= torch.arange(k_len, device=q.device)[None, :]
    return scaled_dot_product_attention(q, k, v, causal_mask)


def choose_tile_sizes(sequence_length: int, d_model: int) -> tuple[int, int]:
    if d_model >= 128:
        return 32, 32
    if sequence_length >= 8192:
        return 32, 32
    if d_model >= 64:
        return 32, 64
    return 64, 64


def benchmark_once(fn, *, warmup: int, rep: int) -> float:
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def assert_outputs_close(triton_out: torch.Tensor, torch_out: torch.Tensor) -> None:
    if triton_out.dtype == torch.bfloat16:
        torch.testing.assert_close(triton_out, torch_out, rtol=2e-2, atol=2e-2)
        return
    torch.testing.assert_close(triton_out, torch_out, rtol=1e-2, atol=1e-2)


def format_table(results: list[dict[str, object]]) -> str:
    lines = [
        "| dtype | seq_len | d_model | q_tile | k_tile | torch_ms | triton_ms | speedup_x | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            "| {dtype} | {sequence_length} | {d_model} | {q_tile_size} | {k_tile_size} | "
            "{torch_ms} | {triton_ms} | {speedup_x} | {status} |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark expects a CUDA device.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    results: list[dict[str, object]] = []
    for dtype_name in args.dtypes:
        dtype = resolve_dtype(dtype_name)
        for sequence_length in args.sequence_lengths:
            for d_model in args.dims:
                q_tile_size, k_tile_size = choose_tile_sizes(sequence_length, d_model)
                if args.q_tile_size is not None:
                    q_tile_size = args.q_tile_size
                if args.k_tile_size is not None:
                    k_tile_size = args.k_tile_size

                row: dict[str, object] = {
                    "dtype": dtype_name,
                    "sequence_length": sequence_length,
                    "d_model": d_model,
                    "q_tile_size": q_tile_size,
                    "k_tile_size": k_tile_size,
                    "torch_ms": "oom",
                    "triton_ms": "oom",
                    "speedup_x": "n/a",
                    "status": "oom",
                }

                try:
                    q = torch.randn(args.batch_size, sequence_length, d_model, device=device, dtype=dtype)
                    k = torch.randn(args.batch_size, sequence_length, d_model, device=device, dtype=dtype)
                    v = torch.randn(args.batch_size, sequence_length, d_model, device=device, dtype=dtype)

                    with torch.no_grad():
                        torch_out = regular_attention_forward(q, k, v, is_causal=True)
                        triton_out, _ = FlashAttentionTriton.flash_attention_forward(
                            q,
                            k,
                            v,
                            is_causal=True,
                            q_tile_size=q_tile_size,
                            k_tile_size=k_tile_size,
                        )
                    assert_outputs_close(triton_out, torch_out)

                    torch_ms = benchmark_once(
                        lambda: regular_attention_forward(q, k, v, is_causal=True), warmup=args.warmup, rep=args.rep
                    )
                    triton_ms = benchmark_once(
                        lambda: FlashAttentionTriton.flash_attention_forward(
                            q,
                            k,
                            v,
                            is_causal=True,
                            q_tile_size=q_tile_size,
                            k_tile_size=k_tile_size,
                        )[0],
                        warmup=args.warmup,
                        rep=args.rep,
                    )
                    row["torch_ms"] = f"{torch_ms:.3f}"
                    row["triton_ms"] = f"{triton_ms:.3f}"
                    row["speedup_x"] = f"{(torch_ms / triton_ms):.2f}" if triton_ms > 0 else "inf"
                    row["status"] = "ok"
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                except Exception as exc:
                    if "out of resource" in str(exc).lower() or "shared memory" in str(exc).lower():
                        row["status"] = "unsupported"
                    else:
                        raise
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                results.append(row)

    table = format_table(results)
    print(table)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
