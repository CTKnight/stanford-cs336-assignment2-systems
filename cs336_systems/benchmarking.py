from __future__ import annotations

import argparse
import statistics
import timeit
from dataclasses import asdict, dataclass

import torch

from cs336_basics.model.lm import TransformerLM
from cs336_basics.model.loss import cross_entropy


VOCAB_SIZE = 10_000
ROPE_THETA = 10_000.0


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    model_size: str
    device: str
    dtype: str
    batch_size: int
    context_length: int
    warmup_steps: int
    steps: int
    mean_seconds: float
    std_seconds: float
    timings_seconds: list[float]


MODEL_SIZES: dict[str, ModelConfig] = {
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7B": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end Transformer forward/backward passes.")
    parser.add_argument("--model-size", choices=MODEL_SIZES, default="small")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--rope-theta", type=float, default=ROPE_THETA)
    parser.add_argument("--mode", choices=["forward", "forward-backward"], default="forward-backward")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default=None, help="Defaults to cuda, then mps, then cpu.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the model.")
    parser.add_argument("--compile-mode", default="default")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: ModelConfig,
    vocab_size: int,
    context_length: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    compile_mode: str,
) -> torch.nn.Module:
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )
    if compile_model and hasattr(torch, "compile") and device.type != "mps":
        model = torch.compile(model, mode=compile_mode)
    return model


def resolve_model_config(args: argparse.Namespace) -> tuple[str, ModelConfig]:
    base_name = args.model_size
    base_config = MODEL_SIZES[base_name]
    override_fields = {
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
    }
    if all(value is None for value in override_fields.values()):
        return base_name, base_config

    config_dict = asdict(base_config)
    for key, value in override_fields.items():
        if value is not None:
            config_dict[key] = value

    custom_config = ModelConfig(**config_dict)
    return "custom", custom_config


def make_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def run_forward(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    vocab_size = logits.shape[-1]
    return cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def benchmark_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> float:
    if mode == "forward-backward":
        model.zero_grad(set_to_none=True)

    synchronize(device)
    start = timeit.default_timer()
    if mode == "forward":
        with torch.no_grad():
            _ = run_forward(model, inputs, targets)
    else:
        loss = run_forward(model, inputs, targets)
        loss.backward()
    synchronize(device)
    end = timeit.default_timer()

    return end - start


def benchmark(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    warmup_steps: int,
    steps: int,
    device: torch.device,
) -> list[float]:
    model.train(mode == "forward-backward")

    if warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0")
    if steps <= 0:
        raise ValueError("--steps must be > 0")

    for _ in range(warmup_steps):
        benchmark_step(model, inputs, targets, mode, device)

    return [benchmark_step(model, inputs, targets, mode, device) for _ in range(steps)]


def format_result(result: BenchmarkResult, model_config: ModelConfig) -> str:
    timings_ms = [timing * 1000.0 for timing in result.timings_seconds]
    details = {
        "mode": result.mode,
        "model_size": result.model_size,
        **asdict(model_config),
        "device": result.device,
        "dtype": result.dtype,
        "batch_size": result.batch_size,
        "context_length": result.context_length,
        "warmup_steps": result.warmup_steps,
        "steps": result.steps,
    }
    header = " ".join(f"{key}={value}" for key, value in details.items())
    return (
        f"{header}\n"
        f"mean_ms={result.mean_seconds * 1000.0:.3f} "
        f"std_ms={result.std_seconds * 1000.0:.3f} "
        f"timings_ms={[round(timing, 3) for timing in timings_ms]}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    model_size_name, model_config = resolve_model_config(args)

    if model_config.num_heads <= 0 or model_config.d_model % model_config.num_heads != 0:
        raise ValueError("num_heads must be > 0 and divide d_model")

    if args.dtype == "float16" and device.type == "cpu":
        raise ValueError("float16 benchmarking on CPU is not supported.")

    if args.dtype == "bfloat16" and device.type == "mps":
        raise ValueError("bfloat16 benchmarking on MPS is not supported by this script.")

    if device.type == "cuda":
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
        compile_model=args.compile,
        compile_mode=args.compile_mode,
    )
    inputs, targets = make_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )
    timings = benchmark(
        model=model,
        inputs=inputs,
        targets=targets,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        device=device,
    )
    result = BenchmarkResult(
        mode=args.mode,
        model_size=model_size_name,
        device=str(device),
        dtype=args.dtype,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        mean_seconds=statistics.mean(timings),
        std_seconds=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        timings_seconds=timings,
    )
    print(format_result(result, model_config))


if __name__ == "__main__":
    main()
