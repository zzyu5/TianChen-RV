# Stage2 RVV Gearbox resource-aware selected-body realization foundation

## Goal

Move one Stage 2 RVV Gearbox/resource-aware decision from mirror metadata into production selected-body realization behavior. The bounded owner is the RVV plugin-local contraction realization path for typed low-precision product-reduction-dequantization: consume typed `tcrv_rvv` body facts plus pass/provider resource facts into realized `tcrv_rvv` structure before provider route construction.

## What I already know

* The repository has no previous `.trellis/.current-task`; this task was created from the Hermes direction brief supplied in the current worker prompt.
* The requested steering file `artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md` is not present in the current worktree, so the live direction source is the user-provided brief plus current specs and code.
* Relevant specs require the RVV route chain to remain `tcrv.exec` envelope -> selected typed `tcrv_rvv` body -> RVV plugin legality/selected-body realization/route provider -> common EmitC route.
* `lib/Plugin/RVV/RVVGearboxSchedules.cpp` already materializes Gearbox/resource attrs and fail-closes stale facts.
* `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp` already realizes low-precision pre-realized bodies into typed `setvl/with_vl/load/widening_product/standalone_reduce/dequantize/store` structure and copies low-precision resource attrs to the realized `with_vl`.
* `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp` and `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already validate low-precision resource facts before route construction.

## Requirements

* Implement one production C++ selected-body realization decision in the RVV plugin, not in common EmitC/export.
* The decision must be derived from typed low-precision `tcrv_rvv` body facts and pass/provider resource facts.
* Resource facts must affect realized `tcrv_rvv` structure, not only diagnostics or target artifact metadata.
* Stale or unsupported resource facts used by the decision must fail closed with targeted diagnostics.
* Existing route/provider validation must stay faithful to realized facts and must not infer compute, dtype, schedule, or route support from route IDs, artifact names, ABI strings, or test names.

## Acceptance Criteria

* [x] Production C++ selected-body realization or provider code changes in the RVV Gearbox/resource-aware path.
* [x] Positive lit/FileCheck coverage shows typed body plus resource facts produce realized structure affected by the resource decision.
* [x] Negative lit/FileCheck coverage shows stale or unsupported resource facts fail closed before route construction.
* [x] Focused checks pass for the changed module/tests.
* [x] Bounded old-authority scan over touched files and added diff lines shows no new positive legacy i32 route authority, descriptor-driven compute, direct-C/source-export authority, or source-front-door positive route.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are reported.

## Out of Scope

* No new segment2 ABI closeout.
* No fixture-only or ssh-evidence-only milestone.
* No source-front-door positive route, dashboard, broad smoke matrix, or high-level Linalg/Vector/StableHLO frontend work.
* No common EmitC invention of RVV semantics.
* No dtype/LMUL clone batch and no new dtype-prefixed helper op family.

## Technical Notes

* Relevant specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/variant-pipeline/index.md`, `.trellis/spec/implementation-stack/index.md`, and `.trellis/spec/guides/*.md`.
* Current bounded implementation target: `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp` with tests under `test/Target/RVV/` and/or `test/Transforms/RVV/`.
* The structural resource decision for this round is low-precision `vsetvl_region_count` consumption for product-reduction-dequantization selected-body realization.

## Completion Notes

* Structural resource decision made: low-precision product-reduction-dequantization selected-body realization now consumes the pass-produced `unroll_factor = 1`, `vsetvl_region_count = 2`, and `peak_live_vector_groups = 4` facts before materializing the realized `with_vl`; it emits realization-owned facts on the realized `with_vl`.
* Provider behavior changed: the low-precision direct-contraction route-family plan now requires the realization-owned producer, decision, realized unroll, realized vsetvl region count, and realized peak-live-vector facts before route acceptance.
* Self-repair performed: the first focused lit run failed because the dialect verifier did not yet allow the new RVV-owned realization resource facts on `tcrv_rvv.with_vl`; fixed by adding the new facts to `isRVVLowPrecisionResourceAttrName`. A later header-artifact run used a stale `tcrv-translate`; fixed by rebuilding `tcrv-translate`.
* Checks passed: `cmake --build build --target tcrv-opt -j 8`, `cmake --build build --target tcrv-translate -j 8`, filtered lit for `Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`, filtered lit for `Transforms/RVV/rvv-gearbox-widening-product-reduce-dequantize-f32.mlir`, `./build/bin/tianchenrv-rvv-extension-plugin-test`, `git diff --check`, bounded old-authority scan, and `git diff --cached --check`.
