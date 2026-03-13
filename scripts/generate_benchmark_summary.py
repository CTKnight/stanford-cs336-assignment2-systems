from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown summary from benchmark JSON outputs.")
    parser.add_argument("--part-b", type=Path, required=True, help="JSON results for the 5-warmup sweep.")
    parser.add_argument("--part-c", type=Path, required=True, help="JSON results for the 0/1/2-warmup sweep.")
    parser.add_argument("--output", type=Path, default=Path("data/benchmark_summary.md"))
    return parser.parse_args()


def load_results(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data if row.get("status") == "ok"]


def index_results(rows: list[dict[str, object]]) -> dict[tuple[int, str, str], dict[str, object]]:
    indexed: dict[tuple[int, str, str], dict[str, object]] = {}
    for row in rows:
        key = (int(row["warmup_steps"]), str(row["model_size"]), str(row["mode"]))
        indexed[key] = row
    return indexed


def format_ms(value: object) -> str:
    return f"{float(value):.3f}"


def build_part_b_table(rows: list[dict[str, object]]) -> str:
    ordered_sizes = ["small", "medium", "large", "xl", "2.7B"]
    indexed = index_results(rows)
    lines = [
        "| Model size | Forward mean (ms) | Forward std (ms) | Forward+backward mean (ms) | Forward+backward std (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in ordered_sizes:
        forward = indexed[(5, size, "forward")]
        backward = indexed[(5, size, "forward-backward")]
        lines.append(
            f"| {size} | {format_ms(forward['mean_ms'])} | {format_ms(forward['std_ms'])} | "
            f"{format_ms(backward['mean_ms'])} | {format_ms(backward['std_ms'])} |"
        )
    return "\n".join(lines)


def build_part_c_table(rows: list[dict[str, object]]) -> str:
    ordered_sizes = ["small", "medium", "large", "xl", "2.7B"]
    indexed = index_results(rows)
    lines = [
        "| Model size | Mode | Warmup 0 mean±std (ms) | Warmup 1 mean±std (ms) | Warmup 2 mean±std (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for size in ordered_sizes:
        for mode in ["forward", "forward-backward"]:
            values = []
            for warmup in [0, 1, 2]:
                row = indexed[(warmup, size, mode)]
                values.append(f"{format_ms(row['mean_ms'])} ± {format_ms(row['std_ms'])}")
            lines.append(f"| {size} | {mode} | {values[0]} | {values[1]} | {values[2]} |")
    return "\n".join(lines)


def build_summary_text(part_b_rows: list[dict[str, object]], part_c_rows: list[dict[str, object]]) -> str:
    indexed_b = index_results(part_b_rows)
    indexed_c = index_results(part_c_rows)

    part_b_sentence = (
        "With 5 warmup steps on an RTX 4090 under WSL2 (`float16`, batch size 4, context length 256), "
        f"forward latency ranged from {format_ms(indexed_b[(5, 'small', 'forward')]['mean_ms'])} ms (`small`) "
        f"to {format_ms(indexed_b[(5, '2.7B', 'forward')]['mean_ms'])} ms (`2.7B`), while forward+backward ranged "
        f"from {format_ms(indexed_b[(5, 'small', 'forward-backward')]['mean_ms'])} ms to "
        f"{format_ms(indexed_b[(5, '2.7B', 'forward-backward')]['mean_ms'])} ms. "
        "The standard deviations were generally small relative to the means, though the larger models showed somewhat more variability."
    )

    part_c_sentence = (
        "Without warmup, the first measured step was dominated by CUDA startup and allocator/kernel initialization, "
        "which inflated both the averages and the standard deviations dramatically. "
        f"For example, `small` forward changed from {format_ms(indexed_b[(5, 'small', 'forward')]['mean_ms'])} ± "
        f"{format_ms(indexed_b[(5, 'small', 'forward')]['std_ms'])} ms at 5 warmups to "
        f"{format_ms(indexed_c[(0, 'small', 'forward')]['mean_ms'])} ± {format_ms(indexed_c[(0, 'small', 'forward')]['std_ms'])} ms with 0 warmups, "
        f"and `2.7B` forward+backward changed from {format_ms(indexed_b[(5, '2.7B', 'forward-backward')]['mean_ms'])} ± "
        f"{format_ms(indexed_b[(5, '2.7B', 'forward-backward')]['std_ms'])} ms to "
        f"{format_ms(indexed_c[(0, '2.7B', 'forward-backward')]['mean_ms'])} ± "
        f"{format_ms(indexed_c[(0, '2.7B', 'forward-backward')]['std_ms'])} ms. "
        "One or two warmup steps remove most of the startup penalty, but they can still differ from 5 warmups because some kernels, caches, and memory behavior continue settling over the first few iterations."
    )

    return "\n\n".join([part_b_sentence, part_c_sentence])


def main() -> None:
    args = parse_args()
    part_b_rows = load_results(args.part_b)
    part_c_rows = load_results(args.part_c)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "# Benchmark Summary",
            "",
            "## Setup",
            "",
            "- Device: NVIDIA GeForce RTX 4090 (WSL2)",
            "- Dtype: `float16`",
            "- Batch size: `4`",
            "- Context length: `256`",
            "- Measurement steps: `10`",
            "",
            "## Part (b)",
            "",
            build_part_b_table(part_b_rows),
            "",
            build_summary_text(part_b_rows, part_c_rows).split("\n\n")[0],
            "",
            "## Part (c)",
            "",
            build_part_c_table(part_c_rows),
            "",
            build_summary_text(part_b_rows, part_c_rows).split("\n\n")[1],
            "",
        ]
    )
    args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
