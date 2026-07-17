# Bounded runtime element-count ABI for RVV scalar dispatch

## Goal

Make the bounded RVV+scalar i32-vadd dispatch target artifact path carry an explicit runtime element-count ABI/control input through the generated callable header/source/object/caller path. The existing descriptor-local `tcrv_rvv.element_count` remains only bounded sample/export metadata and must not become runtime `n`, tensor shape, AVL, VL, correctness coverage, or performance evidence.

## Requirements

* Keep the implementation owner vertical bounded to C++ runtime ABI metadata, RVV/scalar i32-vadd target exporters, RVV+scalar dispatch wrapper/header/source generation, and minimal lit/script/spec updates.
* Use existing C++ runtime ABI support first; add only the smallest typed role if required to represent runtime element count.
* Model runtime element count as an explicit ABI/control input distinct from `runtime_guard`, `rvv_available`, capability facts, selected-plan capacity metadata, and descriptor-local sample metadata.
* Fail closed in C++ diagnostics when required runtime element-count ABI metadata is malformed or missing; do not fall back to descriptor-local `element_count` for real callable runtime behavior.
* Update generated RVV microkernel, scalar fallback, dispatch wrapper/header/source/caller path so loop bounds use the runtime ABI input for the bounded i32-vadd path.
* Keep Python changes limited to runner/evidence orchestration if needed.
* Preserve project invariants: `tcrv.exec` remains compute-free, extension details remain plugin-local or target-export-local, and no broad rewrite of variant selection, registry, dialects, ODS, or runtime ABI outside this slice.

## Acceptance Criteria

* [x] Focused lit/C++ checks prove descriptor-local `tcrv_rvv.element_count` remains sample/export metadata.
* [x] Focused checks prove runtime element count is explicit callable ABI metadata/parameter.
* [x] Focused checks prove `runtime_guard`/`rvv_available` remains a separate ABI/control value.
* [x] Generated dispatch artifacts no longer hard-code descriptor-local `element_count` as callable loop bound.
* [x] Local checks pass: `git diff --check`, configure under `artifacts/tmp/tianchenrv-build`, and `check-tianchenrv`.
* [x] Real `ssh rvv` evidence is collected for the bounded explicit runtime element-count i32-vadd dispatch external caller if the callable ABI changes, unless hardware/credentials block it.
* [x] One coherent commit is created and the worktree is clean.

## Definition of Done

* Active C++ compiler/export/runtime-ABI owner code changed unless inspection proves the path already exists end-to-end.
* Focused tests and any necessary spec/README/task updates included.
* Trellis task archived before final report.
* No build directories, `artifacts/tmp` outputs, logs, editor files, credentials, or unrelated generated files committed.

## Out of Scope

* Python implementation of compiler internals, ABI semantics, target export semantics, lowering, selection, plugin registry, capability decisions, or runtime ABI shape.
* New high-level tensor/tile compute semantics in `tcrv.exec`.
* IME/AME/Sophgo/offload hardware widening.
* Performance, throughput, latency, speedup, or general RVV backend claims.
* Fake RVV evidence through x86, emulation-only output, canned logs, or unaudited success strings.

## Technical Notes

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Expected starting HEAD from user audit: `8263603 feat: derive rvv microkernel sample size from capacity`.
* Current compiler-system gap: dispatch bundle has prior real ssh rvv correctness evidence, but exported callable path risks being a fixed descriptor sample rather than an explicit runtime element-count ABI/control path.
* Required inspection includes runtime ABI support, RVV/scalar plugins, microkernel exporters, dispatch wrapper, target artifact bundle/export/manifest code, transforms, script runner, and focused tests under `test/Target` and `test/Scripts`.
