# Testing Specs

This layer defines required tests and evidence for Weft-RV MLIR.

## Pre-Development Checklist

- [ ] Does MLIR syntax, parsing, printing, verification, or pass behavior have lit/FileCheck coverage?
- [ ] Are C++ tests added for registry, capability helper, or non-textual compiler APIs when lit/FileCheck is insufficient?
- [ ] Does CMake configure/build include the relevant compiler libraries, dialects, passes, tools, and tests?
- [ ] Does any RVV runtime/correctness/performance claim include `ssh rvv` probe or run output?
- [ ] If local MLIR tools are unavailable, is the missing toolchain reported explicitly?
- [ ] Does RVV testing avoid positive legacy `i32m1` route-table artifacts?
- [ ] Does RVV testing fail-close source-front-door/source-artifact positive routes? (见 core-invariants I7)
- [ ] Are tests attached to production-path compiler changes rather than dashboards or report-only surfaces?
- [ ] Does every evidence cell land as a measurement lattice or structural-proof lattice with a state ∈ `{measured|stale|board-pending|open|n_a}`? (见 mlir-testing-contract 格 schema 二分)
- [ ] Is any effect/Δ claim gated on a T-N noise floor (`|Δ|>2×地板` 且 bootstrap CI 不含 0)?
- [ ] Does any vs-framework performance cell carry an adversary resolution probe artifact? (无探针工件 = INVALID)
- [ ] Does the correctness gate declare byte-exact for integer paths and a ULP upper bound for float paths ([K-5])?
- [ ] Are beat claims withheld until [PERF-1] 八门 全绿 (byte-exact + VLEN-flip lit + 双板 objdump + micro∧e2e + selector-attribution)?

## Guidelines Index

| Spec | Description |
|---|---|
| [MLIR Testing Contract](./mlir-testing-contract.md) | lit/FileCheck, C++ tests, CMake checks, RVV evidence, 格 schema 二分 + 状态枚举, T-N 噪声地板, 对手解析探针, byte-exact/ULP 门, [PERF-1] 八门 |

## Quality Check

- Dialect syntax, parser/printer, verifier, pass rewrite, and diagnostics need lit/FileCheck tests.
- C++ tests are appropriate for compiler APIs that are not naturally visible in textual MLIR.
- Python tests may validate tooling scripts, but they do not replace MLIR behavior tests.
- Positive generated artifact tests are allowed only for corrected generic typed `weft_rvv` routes, not old `RVVI32M1*` / `rvv-i32m1-*` paths.
