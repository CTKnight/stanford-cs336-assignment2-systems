from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MODEL_SIZES = ["small", "medium", "large", "xl", "2.7B"]
CONTEXT_LENGTHS = [128, 256, 512, 1024]
PROFILE_MODES = ["forward", "forward-backward", "train-step"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nsight Systems profiles for model/context sweeps.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--model-sizes", nargs="+", default=MODEL_SIZES, choices=MODEL_SIZES)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=CONTEXT_LENGTHS)
    parser.add_argument("--profile-modes", nargs="+", default=PROFILE_MODES, choices=PROFILE_MODES)
    parser.add_argument("--output-dir", type=Path, default=Path("data/nsys"))
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def sanitize_filename_part(value: str) -> str:
    return value.replace(".", "p").replace("/", "_")


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
                output_prefix = args.output_dir / stem
                cmd = [
                    "nsys",
                    "profile",
                    "--force-overwrite",
                    "true",
                    "--sample=none",
                    "--trace=cuda,nvtx,osrt",
                    "--output",
                    str(output_prefix),
                    sys.executable,
                    "-m",
                    "cs336_systems.nsys_profile",
                    "--model-size",
                    model_size,
                    "--context-length",
                    str(context_length),
                    "--batch-size",
                    str(args.batch_size),
                    "--dtype",
                    args.dtype,
                    "--device",
                    args.device,
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--profile-mode",
                    profile_mode,
                ]
                try:
                    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    result = {
                        "status": "ok",
                        "model_size": model_size,
                        "context_length": context_length,
                        "profile_mode": profile_mode,
                        "dtype": args.dtype,
                        "report_prefix": str(output_prefix),
                        "stdout": completed.stdout.strip(),
                    }
                except subprocess.CalledProcessError as exc:
                    result = {
                        "status": "error",
                        "model_size": model_size,
                        "context_length": context_length,
                        "profile_mode": profile_mode,
                        "dtype": args.dtype,
                        "report_prefix": str(output_prefix),
                        "returncode": exc.returncode,
                        "stdout": exc.stdout.strip(),
                        "stderr": exc.stderr.strip(),
                    }
                    if args.stop_on_error:
                        manifest.append(result)
                        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                        raise

                manifest.append(result)
                print(json.dumps(result), flush=True)
                (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
