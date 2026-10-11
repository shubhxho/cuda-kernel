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
| Public repository | Published at https://github.com/shubhxho/cuda-kernel |
| Remote source verification | All 21 published files match local Git blob hashes at `d52102b3e1d0e73bde8a6ec89786810c0f4c22be` |
| GitHub host workflow | Failed before steps: account locked due to billing issue |
| GitHub CUDA compile workflow | Failed before steps: account locked due to billing issue |

Two CTest invocations were killed by the host; the direct executable passed, and a CTest rerun also passed. The final direct host-test invocation and all nine Python report tests passed. This intermittent CTest behavior remains an environment limitation, not a consistently clean CTest gate.

The [host run](https://github.com/shubhxho/cuda-kernel/actions/runs/38179607726) and [CUDA compile run](https://github.com/shubhxho/cuda-kernel/actions/runs/38179607611) both have no executed steps. Their check annotations state: `The job was not started because your account is locked due to a billing issue.` No remote compilation or test execution is claimed.

No GPU speedup is claimed. Test-only timing fixtures in the report unit tests are artificial data used to test the reporting logic. They are not benchmark results.

## Optimization series

The follow-up adds 128-bit PTX load variants, independent gate/down autotuning, stabilized activation, allocation-free model decode buffers, alias guards, wider correctness coverage, a multi-seed sweep, and a complete-suite win gate.

Every commit was checked for non-empty changes, Python syntax, and whitespace errors. Host contract changes were compiled and executed. Python report regressions were executed. These checks do not compile or execute CUDA. The series does not establish lower GPU latency or a library win; CUDA compilation, GPU correctness, sanitizers, and pretrained-model execution still require an accessible NVIDIA environment.

Final-workspace validation after the series: both CTests passed, all 14 Python report tests passed, and the portable mapping audit passed 1,520 cases. The Modal 1.5.0 app definition imported successfully. A fresh GPU run again failed before execution with `Could not connect to the Modal server`; the active profile was also reported as `Unknown (profile misconfigured)`.

Three follow-up commits unify the dispatch registry, add the mapping audit, and retain successful partial artifacts when a later GPU gate fails. A fourth follow-up records these observations. The 100-commit series and these four follow-ups total 104 new commits.
