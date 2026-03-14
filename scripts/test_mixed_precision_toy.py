from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


@dataclass
class ToyRunResult:
    parameter_dtype: torch.dtype
    fc1_output_dtype: torch.dtype
    layernorm_output_dtype: torch.dtype
    logits_dtype: torch.dtype
    loss_dtype: torch.dtype
    grad_dtype: torch.dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect mixed-precision dtypes on a toy model.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--autocast-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--in-features", type=int, default=16)
    parser.add_argument("--out-features", type=int, default=8)
    return parser.parse_args()


def resolve_autocast_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def inspect_model_dtypes(
    model: ToyModel,
    x: torch.Tensor,
    targets: torch.Tensor,
    autocast_dtype: torch.dtype,
) -> ToyRunResult:
    activations: dict[str, torch.dtype] = {}

    def capture_dtype(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
            activations[name] = output.dtype

        return hook

    handles = [
        model.fc1.register_forward_hook(capture_dtype("fc1")),
        model.ln.register_forward_hook(capture_dtype("ln")),
        model.fc2.register_forward_hook(capture_dtype("fc2")),
    ]

    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=x.device.type, dtype=autocast_dtype):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
    loss.backward()

    for handle in handles:
        handle.remove()

    first_grad = next(parameter.grad for parameter in model.parameters() if parameter.grad is not None)
    return ToyRunResult(
        parameter_dtype=next(model.parameters()).dtype,
        fc1_output_dtype=activations["fc1"],
        layernorm_output_dtype=activations["ln"],
        logits_dtype=logits.dtype,
        loss_dtype=loss.dtype,
        grad_dtype=first_grad.dtype,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    autocast_dtype = resolve_autocast_dtype(args.autocast_dtype)

    if device.type != "cuda":
        raise ValueError("This toy mixed-precision script expects --device cuda.")
    if args.autocast_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not report BF16 support.")

    model = ToyModel(args.in_features, args.out_features).to(device=device, dtype=torch.float32)
    x = torch.randn(args.batch_size, args.in_features, device=device, dtype=torch.float32)
    targets = torch.randint(0, args.out_features, (args.batch_size,), device=device, dtype=torch.long)

    result = inspect_model_dtypes(model, x, targets, autocast_dtype)

    print(f"autocast_dtype={autocast_dtype}")
    print(f"parameter_dtype={result.parameter_dtype}")
    print(f"fc1_output_dtype={result.fc1_output_dtype}")
    print(f"layernorm_output_dtype={result.layernorm_output_dtype}")
    print(f"logits_dtype={result.logits_dtype}")
    print(f"loss_dtype={result.loss_dtype}")
    print(f"grad_dtype={result.grad_dtype}")


if __name__ == "__main__":
    main()
