# Assignment 1 Attention Compiled vs Eager

## Setup

- Device: `cuda`
- Dtype: `float32`
- Batch size: `8`
- Warmup steps: `10`
- Timed steps: `100`
- Compiled backend: `inductor`
- Compared configurations: `sequence_length in {256, 1024, 4096, 8192}` for `d_model in {16, 32, 64, 128}`
- Note: `16384` is excluded here because the compiled full sweep was interrupted before a clean artifact was written.

## Results

| d_model | Seq len | Eager fwd (ms) | Compiled fwd (ms) | Fwd speedup | Eager bwd (ms) | Compiled bwd (ms) | Bwd speedup | Eager mem (GiB) | Compiled mem (GiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 0.261 | 0.077 | 3.38x | 0.518 | 0.314 | 1.65x | 0.02 | 0.02 |
| 16 | 1024 | 0.400 | 0.106 | 3.79x | 1.315 | 0.367 | 3.58x | 0.11 | 0.08 |
| 16 | 4096 | 10.291 | 2.765 | 3.72x | 25.326 | 5.403 | 4.69x | 1.56 | 1.04 |
| 16 | 8192 | 41.206 | 10.782 | 3.82x | 100.088 | 20.677 | 4.84x | 6.16 | 4.09 |
| 32 | 256 | 0.249 | 0.081 | 3.08x | 0.595 | 0.335 | 1.78x | 0.02 | 0.02 |
| 32 | 1024 | 0.411 | 0.109 | 3.76x | 1.328 | 0.377 | 3.52x | 0.12 | 0.08 |
| 32 | 4096 | 10.314 | 2.752 | 3.75x | 25.422 | 5.352 | 4.75x | 1.56 | 1.05 |
| 32 | 8192 | 40.710 | 10.805 | 3.77x | 359.097 | 20.844 | 17.23x | 6.17 | 4.11 |
| 64 | 256 | 0.197 | 0.129 | 1.52x | 0.553 | 0.341 | 1.62x | 0.02 | 0.02 |
| 64 | 1024 | 0.417 | 0.122 | 3.43x | 1.371 | 0.385 | 3.56x | 0.12 | 0.09 |
| 64 | 4096 | 10.353 | 2.816 | 3.68x | 25.351 | 5.605 | 4.52x | 1.58 | 1.06 |
| 64 | 8192 | 40.884 | 10.941 | 3.74x | 173.976 | 20.939 | 8.31x | 6.20 | 4.14 |
| 128 | 256 | 0.222 | 0.084 | 2.65x | 0.600 | 0.319 | 1.88x | 0.03 | 0.02 |
| 128 | 1024 | 0.430 | 0.155 | 2.77x | 1.363 | 0.468 | 2.91x | 0.13 | 0.10 |
| 128 | 4096 | 10.619 | 2.902 | 3.66x | 25.884 | 5.664 | 4.57x | 1.61 | 1.09 |
| 128 | 8192 | 41.999 | 11.167 | 3.76x | 250.735 | 21.515 | 11.65x | 6.27 | 4.20 |

## Takeaways

Compiled attention consistently outperformed eager attention on every tested configuration through sequence length `8192`. Forward speedups ranged from 1.52x to 3.82x, while backward speedups ranged from 1.62x to 17.23x.

The largest forward speedup in this sweep was at `d_model=16, sequence_length=8192` (3.82x), and the largest backward speedup was at `d_model=32, sequence_length=8192` (17.23x). At the long-sequence points (`8192`), compiled backward time dropped from `100.088-359.097 ms` eager to `20.677-21.515 ms` compiled.

Memory before backward also dropped across the board. At `sequence_length=8192`, eager memory was about `6.16-6.27 GiB`, while compiled memory was about `4.09-4.20 GiB`, a reduction of roughly `33.3%`.