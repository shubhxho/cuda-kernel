# Optimization commit series

This series contains 100 non-empty commits. Each has a problem/change explanation and its validation scope. Performance labels describe intended mechanisms, not measured GPU gains.

| # | Commit | Change |
|---:|---|---|
| 1 | `d6c0b10c0a23` | feat(kernel): define portable launch configuration registry |
| 2 | `ab2308c1fe1e` | fix(kernel): reject unknown dispatch configurations |
| 3 | `3e0001282625` | refactor(kernel): include the shared dispatch registry |
| 4 | `d5e06d2fb639` | fix(kernel): validate required launch buffers |
| 5 | `e1a2ace24e9d` | fix(numerics): stabilize device SiLU for negative inputs |
| 6 | `bdd7de73ed2a` | perf(kernel): share the stabilized activation implementation |
| 7 | `60f368188ae4` | fix(numerics): use stable activation in library epilogue |
| 8 | `ad33cb34d171` | fix(kernel): guard half2 loads for sliced input pointers |
| 9 | `d4d2d8cc67ee` | perf(kernel): parameterize row grouping and vector width |
| 10 | `1fda41b493b0` | perf(kernel): express compile-time block resource bounds |
| 11 | `888e557a9c0c` | refactor(kernel): eliminate the fixed row-group constant |
| 12 | `f3ccf74fa7bf` | perf(kernel): add aligned 128-bit PTX load primitive |
| 13 | `b458e55a91ed` | perf(kernel): pipeline four independent FP32 accumulation chains |
| 14 | `bbfa234befe0` | perf(kernel): add vector8 configuration 16 with 1 warps and 4 rows |
| 15 | `56a59f8fc48f` | perf(kernel): add vector8 configuration 17 with 2 warps and 4 rows |
| 16 | `09876f8a7a2f` | perf(kernel): add vector8 configuration 18 with 4 warps and 2 rows |
| 17 | `77ea447b45d5` | perf(kernel): add vector8 configuration 19 with 1 warps and 8 rows |
| 18 | `0d0cd257fdb3` | perf(kernel): add vector8 configuration 20 with 2 warps and 2 rows |
| 19 | `c8dc3ecc6e76` | perf(kernel): add vector8 configuration 21 with 4 warps and 1 rows |
| 20 | `1fc165f103cc` | perf(kernel): reduce multi-warp partials cooperatively |
| 21 | `913ab44b1c99` | perf(kernel): shrink inactive reduction storage |
| 22 | `39b45d010024` | fix(reference): stabilize CPU SiLU for extreme negatives |
| 23 | `921d097da518` | fix(reference): reject invalid numeric tolerances |
| 24 | `5ec8ab22835e` | perf(bench): expose separate down-projection dispatch |
| 25 | `6761940a5a7a` | perf(bench): apply independently selected down configuration |
| 26 | `128836382f51` | perf(bench): search the complete launch configuration pairs |
| 27 | `aed2ce217127` | fix(bench): validate every captured candidate before tuning |
| 28 | `f6781c368ce8` | perf(bench): pre-upload executable graphs before timing |
| 29 | `2fb86d7008f9` | fix(bench): close graph capture and release handles on failure |
| 30 | `1ab56205dbe8` | perf(bench): reserve raw event sample storage |
| 31 | `d1e0f70c764d` | fix(bench): prohibit copied CUDA contexts |
| 32 | `bff914003aa9` | fix(bench): prohibit copied CUDA events |
| 33 | `4ed8196c27f0` | fix(bench): bound sample and repetition budgets |
| 34 | `d8c1645208a8` | perf(torch): add allocation-free SwiGLU output binding |
| 35 | `4beb0247d584` | fix(torch): reject overlapping scratch and output storage |
| 36 | `58cff6f90ad0` | fix(torch): reject configuration IDs before narrowing |
| 37 | `06371348ee10` | perf(model): cache detached projection weights per layer |
| 38 | `2c3c9ec6e909` | perf(model): reuse cached weights in the decode closure |
| 39 | `b5b216c6b4b4` | perf(model): allocate per-layer decode workspaces once |
| 40 | `5ebcbea628f8` | perf(model): route decoding through the output binding |
| 41 | `517b1c43815b` | perf(model): discover the kernel configuration search space |
| 42 | `43eb76be6341` | perf(model): tune independent gate and down geometry |
| 43 | `ee1eb21c2236` | perf(model): apply independent down configuration at inference |
| 44 | `35343efb9121` | perf(model): evaluate all registered gate and down pairs |
| 45 | `0186c4001f59` | perf(model): capture the allocation-free tuning operation |
| 46 | `9100a2bfcdcf` | feat(model): retain selected down configuration in evidence |
| 47 | `a94f0f6698eb` | perf(model): select a measured configuration pair |
| 48 | `76b52a0018ce` | feat(model): record the complete selected launch pair |
| 49 | `567d35582750` | fix(model): hash the configuration registry with the CUDA sources |
| 50 | `5e6190ead513` | fix(model): explicitly warm both complete decode paths |
| 51 | `2abcae51efaa` | fix(model): validate timed custom routing on every sample |
| 52 | `216aaa2222b5` | fix(model): constrain decoding to the model context window |
| 53 | `8228174ba702` | feat(model): report peak CUDA allocation and reserved memory |
| 54 | `f5add265d912` | feat(model): persist peak device memory counters |
| 55 | `21addf1b6fe6` | fix(report): require integer sampling counts |
| 56 | `de5c406002fa` | fix(report): reject boolean correctness error values |
| 57 | `000e70bbd7c4` | fix(report): require measured positive workload dimensions |
| 58 | `43d0f63dfaa4` | fix(report): reject duplicate workload case identities |
| 59 | `fdeae8eb5ebc` | fix(report): restrict accepted operation names |
| 60 | `258c9483e441` | fix(report): validate data provenance labels |
| 61 | `12d12ceff6cf` | feat(report): include tail latency for both GPU candidates |
| 62 | `072b3214fa92` | feat(report): add a gate requiring wins on every measured case |
| 63 | `0c3b747a060f` | fix(report): enforce the complete-suite win gate |
| 64 | `ed3e740f9642` | fix(model-report): require positive model workload counts |
| 65 | `af80397c4c5e` | fix(model): atomically replace the final evidence artifact |
| 66 | `83c818881ac6` | test(cuda): cover one-to-tail projection and FFN shapes |
| 67 | `77c36151442b` | test(cuda): cover half2-min projection and FFN shapes |
| 68 | `685ff28bf127` | test(cuda): cover vector8-min projection and FFN shapes |
| 69 | `99277de75e35` | test(cuda): cover below-vector8 projection and FFN shapes |
| 70 | `0fb7af8422f6` | test(cuda): cover above-vector8 projection and FFN shapes |
| 71 | `63a53981d0f6` | test(cuda): cover below-warp projection and FFN shapes |
| 72 | `370724df9ac0` | test(cuda): cover exact-warp projection and FFN shapes |
| 73 | `e0ea30bd2c2c` | test(cuda): cover above-warp projection and FFN shapes |
| 74 | `381644bfc464` | test(cuda): cover half2-warp projection and FFN shapes |
| 75 | `e0dfce8bac7c` | test(cuda): cover half2-tail projection and FFN shapes |
| 76 | `2b691ccbf848` | test(cuda): cover vector8-warp projection and FFN shapes |
| 77 | `1f6409121f1a` | test(cuda): cover vector8-loop projection and FFN shapes |
| 78 | `850f9050ace3` | test(cuda): cover multiwarp-boundary projection and FFN shapes |
| 79 | `33b1ccc75433` | test(cuda): cover long-odd projection and FFN shapes |
| 80 | `d73dc0469d0c` | test(cuda): cover long-even-tail projection and FFN shapes |
| 81 | `4662e3d801a9` | test(reference): cover extreme negative SiLU |
| 82 | `950c1ecb3dd4` | test(reference): cover saturated positive SiLU |
| 83 | `7a7175b8d4be` | test(reference): reject infinite numeric comparisons |
| 84 | `0ae48d48b8c6` | test(reference): reject negative relative tolerances |
| 85 | `2748e2b712c1` | test(reference): reject zero absolute tolerance |
| 86 | `1d0aa54f2fac` | test(reference): reject NaN tolerance |
| 87 | `47250d9cbd08` | test(reference): cover cancellation in the independent dot product |
| 88 | `2cb60d647842` | test(stats): reject non-finite event samples |
| 89 | `62392768c8e4` | test(stats): cover a single positive timing sample |
| 90 | `634393e3bc60` | test(dispatch): audit registered block geometries and IDs |
| 91 | `59b443d08c66` | test(report): reject boolean correctness metrics |
| 92 | `5015f7836922` | test(report): reject duplicate case coverage |
| 93 | `c00b776704ce` | test(report): reject missing workload dimensions |
| 94 | `fa39583266a4` | test(report): reject fractional sampling counts |
| 95 | `ecd6ae3fb5d7` | test(report): retain slower candidates in the report |
| 96 | `21f930526a90` | test(cuda): exercise the model binding, alignment, and workspace contract |
| 97 | `f5fd81f56a38` | feat(bench): add reproducible multi-seed regression sweep |
| 98 | `dcb60189796f` | docs(perf): explain vector loads, buffer ownership, and the expanded benchmark gates |
| 99 | `3ec7cdda37be` | docs(validation): record optimization evidence without inventing GPU results |
| 100 | this commit | Audit the final source tree and record the complete optimization series |

GPU execution remains required before making a speedup claim. No empty commits or repeated copies of the same change were used.
