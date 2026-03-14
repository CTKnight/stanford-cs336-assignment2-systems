# Nsight Systems Runbook

Use the sweep launcher to generate `.nsys-rep` files under this directory:

```bash
uv run python scripts/run_nsys_profile_sweep.py \
  --device cuda \
  --dtype float16 \
  --batch-size 4 \
  --warmup-steps 2 \
  --output-dir data/nsys
```

For a smaller subset while iterating:

```bash
uv run python scripts/run_nsys_profile_sweep.py \
  --model-sizes small medium \
  --context-lengths 128 256 \
  --profile-modes forward forward-backward train-step \
  --output-dir data/nsys
```

The profiled target is `cs336_systems.nsys_profile`, which inserts NVTX ranges for:

- `forward_pass`
- `backward_pass`
- `optimizer_step`
- `forward/model`
- `forward/loss`
- `attention/qkv_proj`
- `attention/qk_matmul`
- `attention/softmax`
- `attention/av_matmul`
- `attention/o_proj`
- `norm/rmsnorm`
- `ffn/w1_proj`
- `ffn/silu`
- `ffn/w3_proj`
- `ffn/gate_mul`
- `ffn/w2_proj`

Useful GUI views in Nsight Systems:

- `Stats System View` -> `CUDA GPU Kernel Summary`
- `Stats System View` -> `NVTX PushPop Summary`
- Timeline filtered by NVTX ranges such as `forward_pass`, `backward_pass`, or `optimizer_step`

Useful CLI exports:

```bash
nsys stats --report nvtx_sum,cuda_gpu_kern_sum --format csv data/nsys/<report>.nsys-rep
```

Question mapping:

- Part (a): compare total `forward_pass` time against the Python benchmark numbers.
- Part (b): in `CUDA GPU Kernel Summary`, identify the kernel with the largest cumulative GPU time; use NVTX filters like `attention/qkv_proj` or `ffn/w1_proj` to see which region is responsible.
- Part (c): besides GEMMs, inspect ranges such as `attention/softmax`, `norm/rmsnorm`, `attention/mask`, and elementwise FFN kernels.
- Part (d): compare `forward` versus `train-step` traces; the optimizer and backward pass should reduce the fraction spent in GEMMs.
- Part (e): inside self-attention, compare `attention/softmax` against the matmul-heavy ranges `attention/qk_matmul` and `attention/av_matmul`.
