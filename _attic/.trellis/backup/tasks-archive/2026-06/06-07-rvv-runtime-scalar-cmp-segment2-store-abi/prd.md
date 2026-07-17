# Stage2 RVV runtime-scalar-cmp masked segment2 store executable artifact ABI boundary

## Goal

Make one bounded RVV Stage 2 memory-movement seam truthful for a runtime-scalar comparison that produces a mask consumed by a masked segment2 store. The selected or pre-realized typed `tcrv_rvv` body facts must line up across runtime scalar ABI binding, scalar splat/compare mask construction, mask-tail policy, segment field value roles, RVV provider route validation, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence when executable behavior is claimed. If the production route is absent or any executable-boundary fact is missing, stale, or mirror-derived, the path must fail closed before artifact acceptance.

## What I Already Know

* The previous archived task `06-07-rvv-computed-masked-segment2-store-artifact-abi` hardened the vector/vector computed-mask segment2 store seam with generated-bundle evidence, mask-tail provider/candidate fail-closed coverage, and explicit/pre-realized `ssh rvv` runs.
* Repo inspection found existing runtime-scalar computed-mask memory routes such as `runtime_scalar_cmp_masked_store`, `runtime_scalar_cmp_masked_load_store`, `runtime_scalar_cmp_masked_macc_add`, and runtime-scalar standalone reductions.
* Repo inspection found existing segment2 memory routes such as `computed_masked_segment2_load_unit_store`, `computed_masked_segment2_store_unit_load`, `computed_masked_segment2_update_unit_load`, `segment2_deinterleave_unit_store`, and `segment2_interleave_unit_load`.
* A bounded search for `runtime_scalar*segment2`, `segment2*runtime_scalar`, and related spellings found no existing runtime-scalar-cmp segment2 store route family. This task is therefore a production seam addition or a precise fail-closed proof, not merely a FileCheck wording cleanup.
* Existing computed-mask segment2 store uses vector compare operands with ABI order `cmp_lhs, cmp_rhs, src0, src1, dst, n`. The target route must replace the RHS compare buffer with a runtime scalar ABI value while preserving typed body/config/memory facts and segment field store behavior.
* Existing runtime-scalar masked memory routes provide the reference boundary for `rhs_scalar` ABI binding, scalar splat producer source, compare predicate, runtime scalar coverage, and fail-closed target validation.
* The relevant long-term rules are in `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, and `.trellis/spec/lowering-runtime/emitc-route.md`: RVV provider owns route semantics; Common EmitC/export remains neutral and must not infer dtype, mask-tail policy, segment lane roles, ABI roles, or intrinsic spelling from metadata, route ids, helper names, or test names.

## Requirements

* Scope the implementation to one route family: runtime-scalar-cmp masked segment2 store with unit-stride field loads and interleaved segment2 destination store.
* Preserve the production route chain:
  typed `tcrv_rvv` selected/pre-realized body -> RVV plugin-owned route/fact validation and selected-body realization if needed -> segment2 statement plan -> `TCRVEmitCLowerableRoute` -> common EmitC materialization -> target artifact export -> generated bundle ABI -> `ssh rvv` evidence when runtime correctness is claimed.
* Runtime ABI order for the new seam must be explicit and provider-derived, expected as `lhs, rhs_scalar, src0, src1, dst, n` unless live code evidence during implementation proves a more locally consistent naming contract. Every exported ABI value must carry provider binding and header/prototype participation facts.
* The typed body/config must carry element type, SEW, LMUL, policy, runtime AVL/VL, compare predicate, mask source/role/form, mask-tail policy owner/plan, source field roles, destination segment memory form, inactive-lane preservation behavior, and segment count.
* The route provider must validate or derive runtime scalar splat/compare facts and segment2 store facts from typed body/config/runtime facts. It must fail closed if runtime scalar binding, scalar C type, splat producer source, compare predicate, mask-tail plan/owner, field role, ABI order, header/type mapping, runtime AVL/VL, or segment store statement facts are missing or stale.
* Common EmitC/export must only materialize provider-built route payloads. Do not add RVV semantic inference to common code.
* Generated-bundle evidence must cover explicit selected body. If the full pre-realized route is too large for one round, finish explicit selected-body end to end and leave the exact pre-realized continuation point.
* Negative evidence must cover at least one stale or missing executable-boundary fact for the new seam, preferably runtime scalar binding or route operand/header binding, plus any existing mask-tail/segment2 stale checks needed to keep the artifact boundary fail-closed.

## Acceptance Criteria

* [ ] The repository has a concrete runtime-scalar-cmp masked segment2 store route surface, or a documented fail-closed production reason if implementation proves the route is not yet legally supportable.
* [ ] Positive explicit selected-body evidence reaches emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness when executable behavior is claimed.
* [ ] If implemented, pre-realized selected-body evidence is added or the PRD records the exact missing owner/realization continuation point.
* [ ] Generated evidence exposes runtime scalar boundary facts including `rhs_scalar` ABI/header participation, scalar splat/compare producer source, compare predicate, and runtime scalar coverage.
* [ ] Generated evidence exposes segment2 store boundary facts including field0/field1 source roles, interleaved destination role, mask-tail plan/owner, inactive old-destination preservation, segment count, and runtime AVL/VL.
* [ ] Target artifact validation rejects at least one stale provider or candidate mirror for the runtime scalar boundary of this route.
* [ ] Target artifact validation still rejects stale mask-tail, segment field, ABI/header, type/header, and segment-store statement facts relevant to this route.
* [ ] `scripts/rvv_generated_bundle_abi_e2e.py` remains syntactically valid and its focused self-test passes if touched.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test` pass after any C++ changes.
* [ ] Relevant lit filters for runtime-scalar-cmp and segment2 store fixtures pass.
* [ ] Bounded old-authority scan over touched files and added diff lines finds no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, or source-front-door authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final `git status --short` are checked during finish/commit.

## Definition of Done

* The runtime-scalar-cmp masked segment2 store executable artifact/ABI seam is either implemented with truthful provider-owned generated-artifact evidence and fail-closed validation, or explicitly left open with a precise production blocker and continuation point.
* Trellis task context and workspace journal are updated truthfully.
* One coherent commit records the implementation and evidence if the task reaches a complete bounded module. If unfinished, leave the task open and do not create a misleading completion commit.

## Out of Scope

* No broad memory-route matrix.
* No indexed gather/scatter expansion.
* No segment2 load rework except as reference.
* No LMUL clone batch.
* No MAcc, dot, product, dequant, reduction, compare/select, or conversion expansion outside the route facts needed for this store seam.
* No high-level Linalg/Vector/StableHLO frontend work.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or report-only work.
* No source-front-door positive route.
* No Common EmitC invention of RVV semantics.
* No descriptor-driven or direct-C route authority.

## Technical Notes

Read and used before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-computed-masked-segment2-store-artifact-abi/prd.md`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-store-dry-run.test`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-store.mlir`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`

Initial repo findings:

* `rg` found no existing runtime-scalar segment2 route spelling.
* `scripts/rvv_generated_bundle_abi_e2e.py` already has runtime-scalar masked memory expectations and computed-masked segment2 store expectations, but no combined expectation.
* `test/Target/TargetArtifactExportTest.cpp` builds computed-mask segment2 store fixtures with vector compare operands and runtime-scalar masked memory fixtures separately.
* The likely implementation path is to combine the existing runtime-scalar computed-mask memory fact surface with the existing computed-mask segment2 store route-family surface, without routing through Common EmitC semantic inference.
