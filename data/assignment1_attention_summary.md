# Assignment 1 Attention Benchmark Summary

## Setup

- Device: `cuda`
- Dtype: `float32`
- Batch size: `8`
- Attention implementation: `cs336_basics.model.attention.scaled_dot_product_attention`
- Warmup steps: `10`
- Timed steps: `100`
- Sweep:
  - `d_model in {16, 32, 64, 128}`
  - `sequence_length in {256, 1024, 4096, 8192, 16384}`

## Results

| d_model | Sequence length | Status | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Memory before backward (GiB) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 16 | 256 | ok | 0.261 | 0.102 | 0.518 | 0.101 | 0.02 |
| 16 | 1024 | ok | 0.400 | 0.072 | 1.315 | 0.208 | 0.11 |
| 16 | 4096 | ok | 10.291 | 0.421 | 25.326 | 0.393 | 1.56 |
| 16 | 8192 | ok | 41.206 | 0.611 | 100.088 | 0.302 | 6.16 |
| 16 | 16384 | oom | - | - | - | - | - |
| 32 | 256 | ok | 0.249 | 0.093 | 0.595 | 0.136 | 0.02 |
| 32 | 1024 | ok | 0.411 | 0.098 | 1.328 | 0.196 | 0.12 |
| 32 | 4096 | ok | 10.314 | 0.453 | 25.422 | 0.401 | 1.56 |
| 32 | 8192 | ok | 40.710 | 0.613 | 359.097 | 126.151 | 6.17 |
| 32 | 16384 | oom | - | - | - | - | - |
| 64 | 256 | ok | 0.197 | 0.056 | 0.553 | 0.160 | 0.02 |
| 64 | 1024 | ok | 0.417 | 0.091 | 1.371 | 0.247 | 0.12 |
| 64 | 4096 | ok | 10.353 | 0.446 | 25.351 | 0.865 | 1.58 |
| 64 | 8192 | ok | 40.884 | 0.583 | 173.976 | 6.164 | 6.20 |
| 64 | 16384 | oom | - | - | - | - | - |
| 128 | 256 | ok | 0.222 | 0.090 | 0.600 | 0.193 | 0.03 |
| 128 | 1024 | ok | 0.430 | 0.090 | 1.363 | 0.208 | 0.13 |
| 128 | 4096 | ok | 10.619 | 0.448 | 25.884 | 0.426 | 1.61 |
| 128 | 8192 | ok | 41.999 | 0.503 | 250.735 | 14.527 | 6.27 |
| 128 | 16384 | oom | - | - | - | - | - |

## OOM Boundary

All four `sequence_length=16384` configurations ran out of memory. All four `sequence_length=8192` configurations completed successfully.

## Takeaways

Memory before backward grows sharply with sequence length and is dominated by the quadratic attention score/probability tensors rather than by `d_model`. Across `d_model=16` through `128`, the memory at sequence length `8192` stays in a narrow band around `6.2 GiB`, while `4096` is around `1.6 GiB`, which is roughly the expected 4x jump when doubling sequence length in an `O(n^2)` attention implementation.

Forward time also tracks sequence length much more strongly than `d_model`: the mean is about `0.2-0.4 ms` at `256`, about `10.3-10.6 ms` at `4096`, and about `40.7-42.0 ms` at `8192`. The largest-context backward measurements show more variance than the smaller runs, but the overall cutoff is clear: on this GPU, `8192` fits and `16384` does not for the tested `d_model` values.
