# Benchmark Summary

## Setup

- Device: NVIDIA GeForce RTX 4090 (WSL2)
- Dtype: `float16`
- Batch size: `4`
- Context length: `256`
- Measurement steps: `10`

## Part (b)

| Model size | Forward mean (ms) | Forward std (ms) | Forward+backward mean (ms) | Forward+backward std (ms) |
|---|---:|---:|---:|---:|
| small | 14.887 | 2.164 | 39.305 | 1.911 |
| medium | 29.713 | 1.986 | 72.818 | 2.682 |
| large | 48.695 | 2.552 | 118.572 | 6.981 |
| xl | 80.742 | 7.581 | 205.624 | 14.978 |
| 2.7B | 87.467 | 4.453 | 226.874 | 5.476 |

With 5 warmup steps on an RTX 4090 under WSL2 (`float16`, batch size 4, context length 256), forward latency ranged from 14.887 ms (`small`) to 87.467 ms (`2.7B`), while forward+backward ranged from 39.305 ms to 226.874 ms. The standard deviations were generally small relative to the means, though the larger models showed somewhat more variability.

## Part (c)

| Model size | Mode | Warmup 0 mean±std (ms) | Warmup 1 mean±std (ms) | Warmup 2 mean±std (ms) |
|---|---|---:|---:|---:|
| small | forward | 87.995 ± 228.239 | 19.125 ± 3.661 | 14.692 ± 1.024 |
| small | forward-backward | 126.564 ± 268.767 | 46.269 ± 6.347 | 37.159 ± 1.412 |
| medium | forward | 90.954 ± 197.287 | 31.903 ± 1.790 | 27.746 ± 1.192 |
| medium | forward-backward | 144.445 ± 224.231 | 81.974 ± 11.639 | 73.441 ± 1.355 |
| large | forward | 109.565 ± 201.943 | 47.195 ± 2.118 | 46.891 ± 1.931 |
| large | forward-backward | 200.070 ± 265.164 | 124.164 ± 5.236 | 112.194 ± 1.942 |
| xl | forward | 139.950 ± 213.295 | 74.663 ± 2.667 | 69.255 ± 0.629 |
| xl | forward-backward | 281.307 ± 288.572 | 188.558 ± 2.168 | 189.571 ± 1.300 |
| 2.7B | forward | 143.519 ± 197.674 | 73.481 ± 0.858 | 78.869 ± 3.160 |
| 2.7B | forward-backward | 306.812 ± 254.124 | 222.238 ± 12.887 | 224.608 ± 11.955 |

Without warmup, the first measured step was dominated by CUDA startup and allocator/kernel initialization, which inflated both the averages and the standard deviations dramatically. For example, `small` forward changed from 14.887 ± 2.164 ms at 5 warmups to 87.995 ± 228.239 ms with 0 warmups, and `2.7B` forward+backward changed from 226.874 ± 5.476 ms to 306.812 ± 254.124 ms. One or two warmup steps remove most of the startup penalty, but they can still differ from 5 warmups because some kernels, caches, and memory behavior continue settling over the first few iterations.
