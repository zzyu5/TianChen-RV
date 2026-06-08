# Stage2 RVV runtime-scalar-cmp masked segment2 store artifact ABI boundary

## Goal

Harden the existing RVV runtime scalar compare + computed-mask + masked
segment2 store selected-body route at the executable artifact ABI boundary. The
route must line up typed `tcrv_rvv` masked segment2 store body facts,
`rhs_scalar` runtime ABI binding and splat, compare-produced mask facts,
field0/field1 payload roles, interleaved destination role, inactive-lane
preservation, dtype/SEW/LMUL/policy, runtime AVL/VL, provider-owned segment2
route-family facts, `TCRVEmitCLowerableRoute`, target artifact export,
generated bundle ABI, and `ssh rvv` correctness evidence.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `944867fe`.
* Commit `944867fe` closed the paired runtime-scalar segment2 load owner gap by
  classifying `RuntimeScalarComputedMaskSegment2LoadUnitStore` as computed-mask
  segment2 memory instead of standalone computed-mask memory.
* The previous archived load PRD records refreshed explicit and pre-realized
  load `ssh rvv` evidence and warns that repeated artifact evidence alone is
  not enough if a production owner gap remains.
* Workspace journal session 556 records earlier explicit and pre-realized
  runtime-scalar segment2 store generated bundles passing on `ssh rvv`, so this
  round must either harden a current under-validated production seam or prove
  exact no-source-change executable evidence for the store path.
* Current source inspection shows
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp` already classifies
  `RuntimeScalarComputedMaskSegment2StoreUnitLoad` as a computed-mask segment2
  operation, store-only route, and `ComputedMaskUnitLoadSegment2Store` memory
  form.
* Current source inspection also shows
  `RVVEmitCSegment2RouteFamilyPlanOwners.cpp` has a runtime-scalar segment2
  store planning owner, ABI binding plan
  `rvv-route-operand-binding:runtime_scalar_cmp_masked_segment2_store_unit_load.v1`,
  runtime ABI order `lhs,rhs_scalar,src0,src1,dst,n`, and required
  `rhs_scalar` ABI/splat facts.
* The immediate under-validation found before implementation is C++ coverage:
  the provider statement-plan test covers computed-mask segment2 store/update
  and registry names, but does not yet analyze a positive
  `RuntimeScalarComputedMaskSegment2StoreUnitLoad` route through runtime-scalar
  provider-plan flags, `rhs_scalar` binding, statement-plan construction, and
  provider-built route consumption.

## Requirements

* Stay on `runtime_scalar_cmp_masked_segment2_store_unit_load`; use load,
  computed-mask segment2 store/update, and plain segment2 routes only as
  bounded references.
* Keep route authority in typed or realized `tcrv_rvv` body facts and
  RVV-provider-owned route-family facts. Common EmitC remains neutral and must
  not infer RVV semantics.
* Do not add descriptor/direct-C/source-front-door or legacy `i32m1` route
  authority.
* If source inspection finds store production owner/source is already complete,
  make a focused test/evidence hardening rather than inventing unrelated
  production changes.
* The runtime-scalar store path must bind ABI/header parameters in order
  `lhs,rhs_scalar,src0,src1,dst,n`.
* `rhs_scalar` must be scalar ABI `int32_t` with role `rhs-scalar-value`,
  realized through `tcrv_rvv.splat`, and mirrored as
  `runtime-scalar-splat-compare-rhs`.
* Store route facts must carry field0/field1 payload source roles, tuple
  creation, masked segment-store, interleaved destination, inactive-lane
  destination preservation, runtime AVL/VL, and provider support mirror.
* Target artifact validation must fail closed on stale executable-boundary
  facts such as vector compare producer or stale field binding.
* Runtime correctness claims in this round require real non-dry-run `ssh rvv`
  generated-bundle evidence for both explicit and pre-realized selected-body
  store fixtures.

## Acceptance Criteria

* [x] Current production source either changes a real store seam or is
      explicitly justified as already closed by source inspection.
* [x] Focused C++ plugin coverage exercises
      `RuntimeScalarComputedMaskSegment2StoreUnitLoad` through analysis,
      computed-mask memory provider-plan classification, segment2 route-family
      provider plan, statement-plan construction, runtime scalar ABI/splat
      facts, and provider-built `TCRVEmitCLowerableRoute`.
* [x] Existing explicit and pre-realized store target fixtures still expose
      ABI order, operand binding, runtime-scalar mask producer, inactive-lane
      contract, field roles, header/prototype, and stale producer/binding
      fail-closed checks.
* [x] Explicit and pre-realized runtime-scalar segment2 store generated bundles
      run non-dry-run on `ssh rvv` when runtime correctness is claimed.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit dry-run/fail-closed tests for
      `runtime-scalar-cmp-masked-segment2-store` pass.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
      the harness script is touched. The harness script was not touched in
      this round, so no self-test rerun was required.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or
      mirror-only route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly reported.

## Definition Of Done

* The runtime-scalar segment2 store artifact/ABI seam is either source-hardened
  or precisely proven as production-complete with focused C++ coverage.
* Explicit and pre-realized generated bundles prove executable behavior on the
  real RVV target when runtime correctness is claimed.
* Trellis task context, results, workspace journal, archive status, and one
  coherent commit are completed if the task is finished.

## Out Of Scope

* No broad segment2 matrix.
* No computed-masked segment2 update expansion.
* No segment2 interleave/deinterleave expansion.
* No dtype/LMUL clone batch.
* No MAcc/reduction/compare-select rework except as reference.
* No source-front-door positive route.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No dashboard/index/report-only work.
* No Common EmitC invention of RVV semantics.
* No unrelated memory or mask route rewrites outside this runtime-scalar
  segment2 store artifact seam.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/index.md`.
* Previous relevant archive:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-load-artifact-abi-boundary/`.
* Bounded code inspected:
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  and the explicit/pre-realized target and script fixtures for
  `runtime_scalar_cmp_masked_segment2_store_unit_load`.

## Results

* Production source justification:
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp` already includes
  `RuntimeScalarComputedMaskSegment2StoreUnitLoad` in the computed-mask
  segment2 operation classifier, store-only route classifier, and
  `ComputedMaskUnitLoadSegment2Store` memory-form mapping. No production source
  change was needed for this seam.
* Focused C++ hardening:
  `test/Plugin/RVVExtensionPluginTest.cpp` now adds a direct provider-route
  fixture for `runtime_scalar_cmp_masked_segment2_store_unit_load` and checks
  analysis, computed-mask memory route-family facts, segment2 route-family
  provider-plan flags, `rhs_scalar` ABI/splat facts, segment2 statement-plan
  calls, and provider-built `TCRVEmitCLowerableRoute` consumption.
* Existing target artifact fixtures remain the fail-closed boundary for stale
  `runtime-scalar-splat-compare-rhs` and stale field binding mirrors in both
  explicit and pre-realized store artifacts.
* Non-dry-run `ssh rvv` evidence was refreshed:
  explicit selected body passed with `n=0,1,16,17,257`, patterns `0,1`, and
  `rhs_scalar=-37,91` at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rtseg-store-explicit-ssh-provider-coverage/evidence.json`.
  The remote PASS line was
  `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1`.
* Pre-realized selected body passed with `n=0,1,7,16,23,257`, patterns `0,1`,
  and `rhs_scalar=-37,91` at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rtseg-store-prerealized-ssh-provider-coverage/evidence.json`.
  The remote PASS line was
  `PASS op=runtime_scalar_cmp_masked_segment2_store_unit_load counts=0,1,7,16,23,257 patterns=0,1`.
* Both remote runs reported active and inactive lanes, inactive interleaved
  destination preservation, field-distinguishing lanes, source preservation,
  and tail preservation.
* Checks run:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `cmake --build build --target tianchenrv-target-artifact-export-test`,
  `build/bin/tianchenrv-target-artifact-export-test`,
  lit filter
  `runtime-scalar-cmp-masked-segment2-store|pre-realized-runtime-scalar-cmp-segment2-store|explicit-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store`,
  the two non-dry-run generated-bundle evidence commands above, context
  validation, bounded old-authority scans, and `git diff --check`.
* Self-repair:
  the first explicit evidence run was blocked before remote execution because
  `llvm-readobj` was not on `PATH`; rerunning with
  `/usr/lib/llvm-20/bin/llvm-readobj` produced passing explicit and
  pre-realized `ssh rvv` evidence.
* Spec update judgment:
  no `.trellis/spec/` change was needed. The existing RVV plugin and EmitC
  route specs already describe the runtime-scalar computed-mask segment2 store
  owner, statement-plan, target export, and evidence requirements; this round
  implemented missing focused coverage and evidence against those existing
  contracts.
