# Benchmark protocol

This repository separates three kinds of artifacts:

| `result_class` | Who produces it | May quote as 5060 performance? |
|---|---|---|
| `measured` | `decode_bench` / `model_bench.py` on a real NVIDIA GPU | Only if the recorded `gpu` name is that 5060 |
| `environment-probe` | `tools/run_bench.py --probe-only` | No |
| `analytical-lower-bound-not-a-benchmark` | `tools/intensity.py` | No. Lower bound, not a latency. |

Test-only timings inside `tests/test_report.py` are fixtures for the reporter. They are not GPU results.

## Kernel runner (`decode_bench`)

Build (CUDA 12.0+; 12.8+ and `CMAKE_CUDA_ARCHITECTURES=120` for native RTX 50-series cubin):

```sh
make build                          # default architectures for this machine
make blackwell                      # sm_120, CUDA 12.8+
python3 tools/run_bench.py --arch 120
```

Correctness happens before any holdout timing. Device outputs are compared to an independent CPU reference (`matvec` + SiLU in double) with atol 0.003 and rtol 0.01. Failure raises; those cases never become a speedup row.

Timing, kernel-only (default JSON `custom` / `baseline`):

1. Warm the chosen implementation, including cuBLAS workspaces.
2. Capture a CUDA Graph of the device work. Weights stay on device.
3. Replay 20 times, then `cudaStreamSynchronize`.
4. For each sample, record CUDA events around `repeats` graph launches and divide by `repeats`.
5. Autotune uses 3 such samples per candidate. Holdout uses `--samples` (default 9) interleaved custom/baseline measurements. Tuning samples are not the reported medians.
6. `tools/report.py` prints median and nearest-rank p95 of those holdout samples. A **separated win** needs median speedup >= 1.05 and every custom sample faster than every baseline sample.

Transfer-inclusive (`--include-transfers`):

- Same warmed graph, still with resident weights.
- Each repeat is pinned H2D of the token input, graph replay, D2H of the output vector.
- Stored under `transfer_inclusive` so it cannot be confused with kernel-only rows.

Suites:

| `--suite` | Shapes |
|---|---|
| `correctness` | Edge sizes for scalar, half2, vector8, warp, and tail paths |
| `smoke` | 512 x 1024 |
| `tinyllama` | 2048 x 5632 (TinyLlama-1.1B MLP; not in the default decode suite) |
| `decode` | 2048 x 8192, 4096 x 11008, 4096 x 14336 |

`decode-1b-shape` is **not** TinyLlama. TinyLlama's intermediate size is 5632. Use `--suite tinyllama` or `tools/model_bench.py` for the real model.

Metadata recorded in the JSON (schema_version 1): GPU name, compute capability, L2, total memory, SM count, clocks, memory bus width, driver/runtime/cuBLAS versions, suite, seed, repeats, samples, timing protocol, cache policy, transfer protocol, PCI bus id. On the host, `tools/run_bench.py` also stores an `nvidia-smi` snapshot in `results/environment.json` (power, persistence, throttle reasons, clocks) when the binary exists.

Power and clocks are **recorded**, not set. Do not change persistence mode, clocks, or driver settings from this repo.

## Full-model runner (`tools/model_bench.py`)

Requires a CUDA PyTorch wheel, TinyLlama weights (~2.2 GB download), and a GPU. Gates:

- Every decode MLP of every timed token must take the custom binding.
- Greedy tokens must match the stock model exactly.
- Logits must pass the runner's atol/rtol before timing counts.

Prefill is excluded. Wall time includes Python and KV-cache handling; GPU events are also stored. Both paths use SDPA attention.

## Commands

```sh
make host-test                      # no GPU required
python3 tools/run_bench.py --probe-only
python3 tools/intensity.py          # analytical tables, not timings
make bench                          # needs nvcc + GPU; kernel-only JSON
./build/decode_bench --suite decode --include-transfers --output results/kernels.json
python3 tools/report.py results/kernels.json --output results/kernels.md
./build/decode_bench --suite tinyllama --include-transfers --output results/tinyllama.json
python3 tools/sweep.py              # three seeds, correctness + decode
```

`compute-sanitizer` remains the device race/sync/memcheck path. It is not a performance result.

## Reading results without lying

- Empty cells in the README table stay empty until a measured JSON exists.
- Never copy analytical microseconds into the "GPU latency" column.
- Never compare a 5060 estimate to a number that was not measured.
- A win against `cublasGemmEx` default/tensor-op is not a win against cuBLASLt, cuDNN, or TensorRT-LLM.
