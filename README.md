# CUDA decode kernels

I want to make one small part of LLM inference faster: the feed-forward layers when a model is generating one token at a time. This repository contains the CUDA kernels, the cuBLAS comparison, and a runner that plugs the kernels into a real pretrained model.

There is **no measured GPU win in this repository yet**. The code was written on a Mac without an NVIDIA GPU or `nvcc`, and the attempted Modal run could not connect. GitHub CI was blocked before execution by an account billing lock. I would rather leave the benchmark cells empty than pretend a speedup happened. The [validation log](results/STATUS.md) records what actually ran.

## What is implemented

The workload is a SwiGLU feed-forward layer:

```text
gate = W_gate @ x + b_gate
up   = W_up   @ x + b_up
h    = SiLU(gate) * up
y    = W_down @ h + b_down + x
```

The standalone benchmark compares a raw projection and the complete layer. It uses FP16 weights and inputs, FP32 accumulation, an FP16 intermediate, and FP32 final output. The cuBLAS path uses three `cublasGemmEx` calls and two pointwise kernels. The custom layer uses two kernels.

The model runner loads **TinyLlama-1.1B-Chat-v1.0**, replaces the decode MLP in every layer, and runs three prompts through the full model. Attention, normalization, the KV cache, the output head, and multi-token prefill use the existing PyTorch implementation. The model path preserves FP16 rounding after projections, SiLU, multiplication, and the down projection. Residual addition stays with the transformer layer.

## Where the speedup could come from

Single-token decoding turns the projection into a matrix-vector product. Most of the work is reading weights. A general matrix-multiply kernel has less opportunity to reuse those weights than it does for a large batch.

The custom kernels assign one, two, four, or eight warps to each output row. There are ten registered configurations, including aligned 128-bit PTX loads with four independent FP32 accumulation chains and variable rows per block. Misaligned or non-multiple-of-eight inputs use a guarded half2 or scalar path. One configuration list generates both the host metadata and CUDA dispatch, so their block geometries stay consistent. Warp shuffles reduce the dot products, with a small shared-memory reduction when several warps share a row.

The gate kernel reads the input for both projections and writes the activated product directly. It avoids materializing the two projection outputs. The down kernel can add bias and the residual before writing the result. This saves launches and intermediate traffic. Whether that beats the library on a particular GPU is a measurement, not a guarantee.

No tensor-core tiling, quantization, attention replacement, or claim about TensorRT-LLM is hidden behind this comparison. This is a deliberately narrow batch-one implementation. Batched inference needs a different design.

## Benchmarks

Current results:

| Comparison | Workload | GPU latency | Speedup |
|---|---|---:|---:|
| Custom vs cuBLAS GemmEx | 2048 → 8192 projection and SwiGLU layer | unmeasured | unmeasured |
| Custom vs cuBLAS GemmEx | 4096 → 11008 projection and SwiGLU layer | unmeasured | unmeasured |
| Custom vs cuBLAS GemmEx | 4096 → 14336 projection and SwiGLU layer | unmeasured | unmeasured |
| Custom vs stock PyTorch | TinyLlama, batch one, 32 decode tokens, three prompts | unmeasured | unmeasured |

The shape suite uses seeded synthetic tensors. Those rows exercise model-sized operations; they are not pretrained-model results. The separate TinyLlama runner uses downloaded pretrained weights and actual token decoding, records the resolved model revision, and checks the entire model output.

Kernel timing uses warmed CUDA Graphs and CUDA events for both implementations. It tunes ten projection configurations, 100 independent gate/down pairs for a complete layer, and two cuBLAS algorithm choices, then compares the selected candidates using fresh, interleaved samples. Correctness is checked against an independent CPU reference using double dot-product accumulation. Odd dimensions, tiny inputs, and partial output blocks are covered by the GPU correctness suite.

The model benchmark checks every decode MLP took the custom path, compares logits, and requires identical greedy tokens before reporting timings. It measures GPU event time and wall time per token, excluding prefill and setup. Wall time includes Python dispatch and cache handling. Both paths use SDPA attention and the same sequence of growing cache lengths. The kernels are tuned on an independent random first-layer input, not on the timed prompts. Decode layers cache detached weights and reuse per-layer scratch and output tensors through an allocation-free binding. These workspaces belong to this single-stream runner; concurrent requests need separate workspaces. Output buffers are overwritten by the next call, so callers must copy outputs they need to retain.

A report calls a result a **separated win** only when the median improves by at least 5% and the slowest custom sample is faster than the fastest baseline sample. This is a conservative noise check, not a statistical confidence interval. Raw samples and numerical errors are retained. Slower and inconclusive results stay in the report.

The cuBLAS baseline is GemmEx with default and tensor-op algorithms. A win there would not establish a win against tuned cuBLASLt, cuDNN, or TensorRT-LLM. A faster MLP also does not necessarily make the full model faster. Both comparisons matter.

## Run on an NVIDIA machine

You need a C++17 compiler, CMake 3.24+, and CUDA 12+. The host-only checks also run on macOS:

```sh
make host-test
```

Build and run the CUDA correctness suite and comparison:

```sh
make bench
# Raw data: results/kernels.json
# Readable comparison: results/kernels.md
```

Set architectures explicitly if you are building for another machine:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

For the actual model, use Python 3.11 and a CUDA 12.6 toolkit. Install the matching CUDA PyTorch wheel first:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-model.txt
make model
# Raw data: results/model.json
# Readable comparison: results/model.md
```

The first model run downloads roughly 2.2 GB of weights and compiles the PyTorch extension. It uses `safetensors` and does not enable remote model code. The runner supports bias-free Llama-style SiLU MLPs and single-token, batch-one decoding. Training is not supported. Full-model validation currently covers the three included prompts, not general model quality or perplexity.

To reproduce a model run, pass the commit from `resolved_revision` in the saved JSON:

```sh
python tools/model_bench.py --revision MODEL_COMMIT --tokens 32 --samples 7
```

For a quick GPU smoke check:

```sh
./build/decode_bench --suite smoke --repeats 20 --samples 3
```

## Run on Modal

```sh
pip install modal==1.5.0
modal token new                    # only if your account is not configured
modal run modal_run.py
# Or: modal run modal_run.py --kernels-only
```

This creates one L4 worker with four CPUs, 8 GiB of memory, no retries, and a 15-minute function timeout. It builds with CUDA 12.6.3, runs correctness checks, benchmarks the kernels, then benchmarks TinyLlama. It returns raw results, reports, source hashes, and logs to `results/`. It creates no persistent volume or deployed service. Cloud compute and image builds can incur charges; the function timeout is not an account billing cap.

## Use your own captured layer

The standalone runner can load real weights and a captured post-normalization token input. Export a local FP16/FP32 safetensors layer plus a single-token NumPy input:

```sh
python tools/export_ffn.py \
  --weights model.safetensors \
  --prefix model.layers.0.mlp \
  --input hidden_state.npy --output fixture
```

The exporter prints the dimensions. Use them in the benchmark:

```sh
./build/decode_bench --fixture fixture --hidden H --intermediate I \
  --output results/captured.json
python tools/report.py results/captured.json
```

All tensors for the chosen layer must be in the supplied shard. Convert BF16 weights before using the NumPy exporter. Missing biases become zero. The fixture manifest records SHA-256 hashes; benchmark inputs are length-checked and rejected if they contain non-finite FP16 values.

## Reading the code

- [`include/kernels.cuh`](include/kernels.cuh): vectorized projections and fused epilogues.
- [`src/benchmark.cu`](src/benchmark.cu): cuBLAS path, CPU reference, tuning, graph timing, and JSON results.
- [`src/torch_extension.cu`](src/torch_extension.cu): checked PyTorch binding on the current CUDA stream.
- [`tools/model_bench.py`](tools/model_bench.py): full-model substitution, correctness gates, and latency measurements.
- [`tools/report.py`](tools/report.py) and [`tools/model_report.py`](tools/model_report.py): evidence validation and readable reports.

For a broader sweep and device validation:

```sh
python tools/gpu_validate.py
python tools/sweep.py --model
python tools/report.py results/kernels.json --require-all-wins
compute-sanitizer --tool racecheck ./build/decode_bench --suite correctness --repeats 5 --samples 3
compute-sanitizer --tool synccheck ./build/decode_bench --suite correctness --repeats 5 --samples 3
```

The all-wins gate fails on any slower or inconclusive case. It is a goal check; no current artifact passes it because GPU execution remains unavailable. Tail latency in the kernel report is p95 of replay-batch averages, not a per-request SLA. Fifteen new correctness shapes cover scalar, half2, vector8, warp, loop, and row-group boundaries. The optimization series is listed in [COMMIT_SERIES.md](COMMIT_SERIES.md); individual commit bodies explain the change and validation limits.

Public CI checks host contracts and CUDA compilation. It has no GPU, so a green compile job is not a GPU correctness or performance result. On a CUDA machine, run `compute-sanitizer --tool memcheck ./build/decode_bench --suite correctness --repeats 5 --samples 3` before relying on the kernels in an application.

The implementation follows the [NVIDIA CUDA programming guide](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html) and [cuBLAS API](https://docs.nvidia.com/cuda/cublas/). The model integration follows the [Llama MLP implementation](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/llama/modeling_llama.py) in Transformers 4.51.3. Model weights keep their own license; this repository's code is MIT licensed.
