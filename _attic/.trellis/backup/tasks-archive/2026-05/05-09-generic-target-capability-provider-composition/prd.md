# Generic Target Capability Provider Composition

## Goal

Make target/capability provider composition a generic compiler-visible contract instead of relying on the linalg-specific `tcrv_frontend_capability_providers` bridge as the normal path. A marked bounded linalg i32-vadd input should be able to name a target profile whose composition declares the providers planning must see, then reuse the existing RVV/scalar plugin proposal, legality, dispatch, lowering-boundary, emission-plan, and bundle export path.

## Requirements

* Add or complete a generic `tcrv.exec.target` / capability composition surface in C++/MLIR/TableGen.
* Validate provider references generically: refs must resolve to module-level `tcrv.exec.capability` or capability-provider `tcrv.exec.target` symbols, carry non-empty capability identity, avoid duplicate symbols/ids, reject self references, and reject obvious cycles if nested target providers are allowed.
* Refactor bounded linalg i32-vadd lowering so provider imports are resolved from the selected target profile composition first.
* Keep `tcrv_frontend_capability_providers`, if retained, as an explicit transitional override/supplement with validation, not the only durable path.
* Preserve plugin locality: no RVV/scalar route IDs, march strings, runtime guard names, target-family semantics, or compute behavior in core/frontend provider resolution.
* Keep the linalg frontend bounded to the existing marked i32-vadd shape and existing ABI boundary.
* Update focused lit/FileCheck coverage for target-level provider composition feeding RVV+scalar dispatch/bundle behavior.
* Add at most one focused negative test for a new generic composition failure mode.

## Acceptance Criteria

* [ ] Existing linalg-fed dispatch bundle fixture declares planning providers through target/capability composition, not only `tcrv_frontend_capability_providers`.
* [ ] Planning still materializes RVV dispatch case, scalar fallback, dispatch guard, selected lowering boundaries, supported emission plans, and RVV+scalar bundle metadata.
* [ ] Malformed generic provider composition fails with a focused diagnostic.
* [ ] `git diff --check` passes.
* [ ] `cmake --build build --target tcrv-opt` passes.
* [ ] `cmake --build build --target tcrv-translate` passes.
* [ ] `cmake --build build --target check-tianchenrv` passes.
* [ ] A coherent commit is created and the worktree is clean.

## Definition of Done

* Core IR remains execution/capability/variant/dispatch/fallback focused and compute-free.
* Primary implementation remains C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
* Python is used only by existing runner/probe/artifact scripts, not for compiler semantics.
* Extension behavior remains plugin-local behind generic capability and registry interfaces.
* RVV runtime/correctness/performance claims are made only if real `ssh rvv` evidence is rerun.
* Trellis task is validated and archived before final commit.

## Technical Approach

Use a target-profile attribute such as `capability_providers = [@provider, ...]` on `tcrv.exec.target` as the generic composition contract unless existing code already exposes an equivalent. Implement reusable C++ support helpers that resolve the selected kernel target profile and recursively collect provider symbols into the effective capability scope. The helper should be consumed by both `TargetCapabilitySet::buildFromKernelChecked` and `LowerLinalgI32VAddToExec`, so frontend lowering and backend planning see the same generic provider composition.

`tcrv_frontend_capability_providers` remains a validated supplemental import path for transition/testing only. It must share the same provider validation logic and duplicate-id/symbol checks with target-level composition.

## Out of Scope

* Generic linalg lowering beyond the existing marked i32-vadd wrapper.
* New compute ops in `tcrv.exec`.
* RVV/scalar special cases in core/frontend/provider-resolution code.
* New Python implementation of capability resolution or planning.
* Performance or RVV runtime correctness claims unless the targeted `ssh rvv` path is rerun.

## Technical Notes

* Required inspection confirmed repo root `/home/kingdom/phdworks/TianchenRV`, clean initial worktree, and HEAD `3c2ada6 feat: feed linalg vadd dispatch bundle planning`.
* Relevant specs: `.trellis/spec/index.md`, `core-dialect/tcrv-exec-contract.md`, `capability-model/capability-contract.md`, `plugin-protocol/*`, `variant-pipeline/generation-selection-tuning.md`, `lowering-runtime/emission-runtime-contract.md`, `testing/mlir-testing-contract.md`.
* Existing bridge is in `lib/Transforms/LowerLinalgI32VAddToExec.cpp` via `tcrv_frontend_capability_providers`.
* Existing capability construction is in `include/TianChenRV/Support/CapabilityModel.h` and `lib/Support/CapabilityModel.cpp`.
