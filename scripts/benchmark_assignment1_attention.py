from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import timeit
from pathlib import Path

import torch

from cs336_basics.model.attention import scaled_dot_product_attention


DEFAULT_D_MODELS = [16, 32, 64, 128]
DEFAULT_SEQUENCE_LENGTHS = [256, 1024, 4096, 8192, 16384]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Assignment 1 attention implementation across sequence lengths and embedding sizes."
    )
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-models", nargs="+", type=int, default=DEFAULT_D_MODELS)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the attention function.")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON file for the full sweep output.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--d-model", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def make_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, sequence_length, d_model, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, sequence_length, d_model, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, sequence_length, d_model, device=device, dtype=dtype, requires_grad=True)
    mask = torch.tril(torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool))
    return q, k, v, mask


def run_forward_pass(
    attention_impl,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return attention_impl(q, k, v, mask)


def benchmark_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    attention_impl,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = run_forward_pass(attention_impl, q, k, v, mask)
        synchronize(device)

    timings: list[float] = []
    for _ in range(timed_steps):
        synchronize(device)
        start = timeit.default_timer()
        with torch.no_grad():
            _ = run_forward_pass(attention_impl, q, k, v, mask)
        synchronize(device)
        timings.append(timeit.default_timer() - start)
    return timings


def benchmark_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    attention_impl,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
) -> tuple[list[float], int | None, int | None]:
    for _ in range(warmup_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = run_forward_pass(attention_impl, q, k, v, mask)
        loss = out.sum()
        synchronize(device)
        loss.backward()
        synchronize(device)

    q.grad = None
    k.grad = None
    v.grad = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = run_forward_pass(attention_impl, q, k, v, mask)
    synchronize(device)
    memory_allocated_before_backward = None
    memory_reserved_before_backward = None
    if device.type == "cuda":
        memory_allocated_before_backward = torch.cuda.memory_allocated(device)
        memory_reserved_before_backward = torch.cuda.memory_reserved(device)
    del out
    q.grad = None
    k.grad = None
    v.grad = None

    timings: list[float] = []
    for _ in range(timed_steps):
        q.grad = None
        k.grad = None
        v.grad = None
        out = run_forward_pass(attention_impl, q, k, v, mask)
        loss = out.sum()
        synchronize(device)
        start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        timings.append(timeit.default_timer() - start)
    return timings, memory_allocated_before_backward, memory_reserved_before_backward


def summarize_timings(timings: list[float]) -> dict[str, float | list[float]]:
    if len(timings) == 1:
        std_seconds = 0.0
    else:
        std_seconds = statistics.stdev(timings)
    return {
        "mean_seconds": statistics.mean(timings),
        "std_seconds": std_seconds,
        "timings_seconds": timings,
    }


def benchmark_one_configuration(args: argparse.Namespace) -> dict[str, object]:
    if args.d_model is None or args.sequence_length is None:
        raise ValueError("Worker mode requires --d-model and --sequence-length.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.compile and (not hasattr(torch, "compile") or device.type == "mps"):
        raise ValueError("torch.compile is not available for this configuration.")

    attention_impl = scaled_dot_product_attention
    if args.compile:
        attention_impl = torch.compile(attention_impl, mode=args.compile_mode, backend=args.compile_backend)

    try:
        q, k, v, mask = make_inputs(
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            d_model=args.d_model,
            device=device,
            dtype=dtype,
        )
        forward_timings = benchmark_forward(
            q=q,
            k=k,
            v=v,
            mask=mask,
            attention_impl=attention_impl,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            device=device,
        )
        backward_timings, memory_allocated, memory_reserved = benchmark_backward(
            q=q,
            k=k,
            v=v,
            mask=mask,
            attention_impl=attention_impl,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            device=device,
        )
    except RuntimeError as exc:
        if not is_oom_error(exc):
            raise
        return {
            "status": "oom",
            "device": str(device),
            "dtype": args.dtype,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "sequence_length": args.sequence_length,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "device": str(device),
        "dtype": args.dtype,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "compile_backend": args.compile_backend,
        "batch_size": args.batch_size,
        "d_model": args.d_model,
        "sequence_length": args.sequence_length,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "forward": summarize_timings(forward_timings),
        "backward": summarize_timings(backward_timings),
        "memory_allocated_before_backward_bytes": memory_allocated,
        "memory_reserved_before_backward_bytes": memory_reserved,
    }


def run_worker_subprocess(
    *,
    script_path: Path,
    args: argparse.Namespace,
    d_model: int,
    sequence_length: int,
) -> dict[str, object]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--device",
        str(resolve_device(args.device)),
        "--dtype",
        args.dtype,
        "--batch-size",
        str(args.batch_size),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--seed",
        str(args.seed),
        "--compile-mode",
        args.compile_mode,
        "--compile-backend",
        args.compile_backend,
        "--d-model",
        str(d_model),
        "--sequence-length",
        str(sequence_length),
    ]
    if args.compile:
        cmd.append("--compile")
    completed = subprocess.run(cmd, capture_output=True, text=True)
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return {
            "status": "error",
            "device": str(resolve_device(args.device)),
            "dtype": args.dtype,
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "compile_backend": args.compile_backend,
            "batch_size": args.batch_size,
            "d_model": d_model,
            "sequence_length": sequence_length,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": completed.stderr.strip(),
        }
    return json.loads(stdout)


def print_result_summary(result: dict[str, object]) -> None:
    header = (
        f"d_model={result['d_model']} "
        f"sequence_length={result['sequence_length']} "
        f"compile={result['compile']} "
        f"compile_backend={result['compile_backend']} "
        f"status={result['status']}"
    )
    print(header, flush=True)
    if result["status"] != "ok":
        return

    forward = result["forward"]
    backward = result["backward"]
    print(
        " ".join(
            [
                f"forward_mean_ms={forward['mean_seconds'] * 1e3:.3f}",
                f"forward_std_ms={forward['std_seconds'] * 1e3:.3f}",
                f"backward_mean_ms={backward['mean_seconds'] * 1e3:.3f}",
                f"backward_std_ms={backward['std_seconds'] * 1e3:.3f}",
                f"memory_allocated_before_backward_bytes={result['memory_allocated_before_backward_bytes']}",
                f"memory_reserved_before_backward_bytes={result['memory_reserved_before_backward_bytes']}",
            ]
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()

    if args.worker:
        result = benchmark_one_configuration(args)
        print(json.dumps(result), flush=True)
        return

    script_path = Path(__file__).resolve()
    results: list[dict[str, object]] = []
    for d_model in args.d_models:
        for sequence_length in args.sequence_lengths:
            result = run_worker_subprocess(
                script_path=script_path,
                args=args,
                d_model=d_model,
                sequence_length=sequence_length,
            )
            results.append(result)
            print_result_summary(result)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
