from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


MODEL_SIZES = ["small", "medium", "large", "xl", "2.7B"]
DEFAULT_OUTPUT = Path("data/benchmark_results.json")


@dataclass
class BenchmarkSummary:
    mode: str
    model_size: str
    warmup_steps: int
    mean_ms: float
    std_ms: float
    timings_ms: list[float]
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated benchmark sweeps via cs336_systems.benchmarking.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, nargs="+", default=[5, 0, 1, 2])
    parser.add_argument("--model-sizes", nargs="+", default=MODEL_SIZES, choices=MODEL_SIZES)
    parser.add_argument("--modes", nargs="+", default=["forward", "forward-backward"], choices=["forward", "forward-backward"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def run_one(
    model_size: str,
    mode: str,
    warmup_steps: int,
    args: argparse.Namespace,
) -> BenchmarkSummary:
    cmd = [
        sys.executable,
        "-m",
        "cs336_systems.benchmarking",
        "--model-size",
        model_size,
        "--mode",
        mode,
        "--warmup-steps",
        str(warmup_steps),
        "--steps",
        str(args.steps),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--context-length",
        str(args.context_length),
        "--batch-size",
        str(args.batch_size),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = completed.stdout.strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Unexpected benchmark output:\n{raw}")

    header = dict(field.split("=", 1) for field in lines[0].split())
    metrics = dict(field.split("=", 1) for field in lines[1].split(maxsplit=2)[:2])
    timings_prefix = "timings_ms="
    timings_start = lines[1].find(timings_prefix)
    if timings_start == -1:
        raise ValueError(f"timings_ms missing from benchmark output:\n{raw}")
    timings_ms = ast.literal_eval(lines[1][timings_start + len(timings_prefix):])

    return BenchmarkSummary(
        mode=header["mode"],
        model_size=header["model_size"],
        warmup_steps=int(header["warmup_steps"]),
        mean_ms=float(metrics["mean_ms"]),
        std_ms=float(metrics["std_ms"]),
        timings_ms=timings_ms,
        raw=raw,
    )


def main() -> None:
    args = parse_args()
    results: list[dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for warmup_steps in args.warmup_steps:
        for model_size in args.model_sizes:
            for mode in args.modes:
                try:
                    summary = run_one(model_size=model_size, mode=mode, warmup_steps=warmup_steps, args=args)
                    result = {
                        "status": "ok",
                        "mode": summary.mode,
                        "model_size": summary.model_size,
                        "warmup_steps": summary.warmup_steps,
                        "mean_ms": summary.mean_ms,
                        "std_ms": summary.std_ms,
                        "timings_ms": summary.timings_ms,
                    }
                    print(json.dumps(result), flush=True)
                    results.append(result)
                    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                except subprocess.CalledProcessError as exc:
                    result = {
                        "status": "error",
                        "mode": mode,
                        "model_size": model_size,
                        "warmup_steps": warmup_steps,
                        "returncode": exc.returncode,
                        "stderr": exc.stderr.strip(),
                        "stdout": exc.stdout.strip(),
                    }
                    print(json.dumps(result), flush=True)
                    results.append(result)
                    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                    if args.stop_on_error:
                        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                        raise

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
