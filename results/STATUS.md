# Observed validation

Recorded on 2026-10-12 in `/Users/shubh/Documents/cuda-kernel` on an Apple Silicon Mac.

| Gate | Result |
|---|---|
| C++ host build | Passed with Apple Clang 21 |
| Independent reference, dimensions, non-finite rejection, timing statistics, JSON escaping | Passed |
| Python report regression suite | 9 tests passed |
| Python syntax compilation | Passed |
| Modal app definition | Imported with Modal 1.5.0 |
| CUDA configure | Blocked: `Failed to find nvcc` |
| Modal GPU attempt | Failed before execution: `Could not connect to the Modal server` |
| GPU correctness and sanitizer | Unverified |
| cuBLAS latency comparisons | Unmeasured |
| Full pretrained-model correctness and latency | Unmeasured |

One intermediate CTest invocation was killed by the host; the direct executable and the subsequent CTest rerun passed. The Python report tests were rerun after the final report gates were added.

No GPU speedup is claimed. Test-only timing fixtures in the report unit tests are artificial data used to test the reporting logic. They are not benchmark results.
