from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


MODEL_SIZES = ["2.7B"]
CONTEXT_LENGTHS = [128, 256, 512]
PROFILE_MODES = ["forward", "train-step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CUDA memory-profile sweeps via cs336_systems.benchmarking.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--mixed-precision", default="none", choices=["none", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--model-sizes", nargs="+", default=MODEL_SIZES)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=CONTEXT_LENGTHS)
    parser.add_argument("--profile-modes", nargs="+", default=PROFILE_MODES, choices=["forward", "forward-backward", "train-step"])
    parser.add_argument("--memory-history-max-entries", type=int, default=1_000_000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/memory_profiles"))
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def sanitize_filename_part(value: str) -> str:
    return value.replace(".", "p").replace("/", "_")


def parse_benchmark_output(raw: str) -> dict[str, object]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Unexpected benchmark output:\n{raw}")

    header = dict(field.split("=", 1) for field in lines[0].split())
    metrics_tokens = lines[1].split()
    metrics: dict[str, str] = {}
    for token in metrics_tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metrics[key] = value

    timings_prefix = "timings_ms="
    timings_start = lines[1].find(timings_prefix)
    if timings_start == -1:
        raise ValueError(f"timings_ms missing from benchmark output:\n{raw}")

    parsed: dict[str, object] = {
        "mode": header["mode"],
        "model_size": header["model_size"],
        "device": header["device"],
        "dtype": header["dtype"],
        "mixed_precision": header["mixed_precision"],
        "batch_size": int(header["batch_size"]),
        "context_length": int(header["context_length"]),
        "warmup_steps": int(header["warmup_steps"]),
        "steps": int(header["steps"]),
        "mean_ms": float(metrics["mean_ms"]),
        "std_ms": float(metrics["std_ms"]),
        "timings_ms": ast.literal_eval(lines[1][timings_start + len(timings_prefix):]),
    }
    if "peak_memory_allocated_bytes" in metrics:
        parsed["peak_memory_allocated_bytes"] = int(metrics["peak_memory_allocated_bytes"])
    if "peak_memory_reserved_bytes" in metrics:
        parsed["peak_memory_reserved_bytes"] = int(metrics["peak_memory_reserved_bytes"])
    if "memory_snapshot_path" in metrics:
        parsed["memory_snapshot_path"] = metrics["memory_snapshot_path"]
    return parsed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []

    for model_size in args.model_sizes:
        for context_length in args.context_lengths:
            for profile_mode in args.profile_modes:
                stem = (
                    f"{sanitize_filename_part(model_size)}"
                    f"_ctx{context_length}"
                    f"_{sanitize_filename_part(profile_mode)}"
                    f"_{args.dtype}"
                )
                if args.mixed_precision != "none":
                    stem += f"_{sanitize_filename_part(args.mixed_precision)}"
                snapshot_path = args.output_dir / f"{stem}.pickle"
                cmd = [
                    sys.executable,
                    "-m",
                    "cs336_systems.benchmarking",
                    "--model-size",
                    model_size,
                    "--context-length",
                    str(context_length),
                    "--batch-size",
                    str(args.batch_size),
                    "--mode",
                    profile_mode,
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--steps",
                    str(args.steps),
                    "--dtype",
                    args.dtype,
                    "--mixed-precision",
                    args.mixed_precision,
                    "--device",
                    args.device,
                    "--memory-profile",
                    "--memory-snapshot-path",
                    str(snapshot_path),
                    "--memory-history-max-entries",
                    str(args.memory_history_max_entries),
                ]
                try:
                    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    raw = completed.stdout.strip()
                    parsed = parse_benchmark_output(raw)
                    result = {
                        "status": "ok",
                        **parsed,
                        "stdout": raw,
                    }
                except (subprocess.CalledProcessError, ValueError) as exc:
                    if isinstance(exc, subprocess.CalledProcessError):
                        stdout = exc.stdout.strip()
                        stderr = exc.stderr.strip()
                        returncode = exc.returncode
                    else:
                        stdout = ""
                        stderr = str(exc)
                        returncode = None
                    result = {
                        "status": "error",
                        "model_size": model_size,
                        "context_length": context_length,
                        "profile_mode": profile_mode,
                        "dtype": args.dtype,
                        "mixed_precision": args.mixed_precision,
                        "memory_snapshot_path": str(snapshot_path),
                        "returncode": returncode,
                        "stdout": stdout,
                        "stderr": stderr,
                    }
                    manifest.append(result)
                    print(json.dumps(result), flush=True)
                    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    if args.stop_on_error:
                        raise
                    continue

                manifest.append(result)
                print(json.dumps(result), flush=True)
                (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
