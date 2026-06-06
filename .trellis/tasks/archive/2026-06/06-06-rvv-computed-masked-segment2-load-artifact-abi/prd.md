# Stage2 RVV computed-masked segment2 load executable artifact ABI boundary

## Goal

Make the existing RVV computed-masked segment2 load selected-body route family truthful at the executable artifact ABI boundary. The selected or pre-realized typed `tcrv_rvv` segment2 tuple-load body facts must line up with RVV plugin-owned route validation, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence when executable behavior is claimed. If any executable-boundary fact is missing, stale, or derived from mirror metadata instead of typed body/config/runtime structure, the path must fail closed with a targeted diagnostic.

## What I Already Know

* The current direction starts after commit `99986725 rvv: resource computed masked strided dot route`.
* The previous owner converged a computed-masked strided-input widening dot-reduce-add resource/artifact ABI seam.
* This task switches to a memory-movement owner: computed-masked segment2 load only.
* Relevant facts include tuple field0/field1 memory roles, computed mask or predicate facts, active/inactive lane policy, unit-store result binding, dtype, SEW, LMUL, config, runtime AVL/VL, per-operand ABI/header bindings, RVV plugin segment2 validation, EmitC materialization, artifact export, generated bundle ABI, and `ssh rvv` evidence.
* Route ids, helper names, test names, descriptors, artifact metadata, emission-plan mirrors, and common EmitC code are not allowed to become semantic authority.

## Requirements

* Scope the implementation to the computed-masked segment2 load route family.
* Inspect and harden the production seam from selected/pre-realized `tcrv_rvv` segment2 load body facts through RVV route planning/provider validation, common EmitC materialization, target artifact validation/export, generated bundle ABI, and runtime execution evidence.
* Preserve the architecture boundary where `tcrv_rvv` typed body/config/runtime facts and RVV plugin-owned validation drive RVV semantics.
* Ensure tuple field roles, mask binding, inactive-lane policy, memory form, store result binding, dtype/config, ABI/header binding, runtime AVL/VL, and statement facts are validated before an executable route is claimed.
* If the existing path is dry-run-only or under-validated, convert the focused load seam into a real executable artifact path or fail it closed at the precise missing boundary.
* Add or update focused positive evidence for materialized selected boundary, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness only when runtime behavior is actually executable.
* Add or update focused fail-closed evidence for at least one stale or missing executable-boundary fact.
* Keep common EmitC/export mechanics semantically neutral.

## Acceptance Criteria

* [x] Computed-masked segment2 load route construction consumes typed body/config/capability/runtime facts for tuple fields, mask, inactive-lane policy, memory form, result store binding, dtype/config, ABI/header binding, and AVL/VL.
* [x] The generated RVV bundle ABI for computed-masked segment2 load has a truthful prototype/header and argument order matching the selected body/runtime boundary.
* [x] Target artifact route-family validation rejects at least one stale or missing computed-masked segment2 executable-boundary fact with a targeted diagnostic.
* [x] Positive generated-bundle evidence covers computed-masked segment2 load through artifact export and compile.
* [x] If runtime correctness is claimed, the computed-masked segment2 load evidence includes a real `ssh rvv` correctness result.
* [x] Focused plugin/target tests pass, including `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test`.
* [x] Relevant generated-bundle dry-run tests pass.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if the script changes.
* [x] A bounded old-authority scan over touched files and added diff lines finds no new route-id, helper-name, descriptor, metadata, source-front-door, or common-EmitC semantic authority.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are clean after commit.

## Definition of Done

* The production computed-masked segment2 load artifact/ABI seam is either executable with truthful evidence or fail-closed at the exact unsupported boundary.
* Trellis task context, implementation/check context, and workspace journal are updated truthfully.
* The task is finished or archived according to repository convention.
* One coherent commit records the implementation and evidence.

## Out of Scope

* No broad segment2 route matrix.
* No segment2 store/update/interleave/deinterleave batch unless the load seam is already complete and the same coherent boundary fits this round.
* No indexed gather/scatter expansion.
* No additional dot, MAcc, or low-precision resource rework except as reference.
* No dtype or LMUL clone batch.
* No high-level Linalg, Vector, or StableHLO frontend work.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or report-only owner.
* No source-front-door positive route.
* No descriptor-driven authority.
* No common EmitC invention of RVV semantics.
* No mass rewrite of unrelated memory, reduction, compare/select, contraction, widening conversion, or mask routes.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-computed-masked-strided-dot-resource-abi/`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-segment2-load-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-load-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-segment2-load-dry-run.test`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-load.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir`

Findings after repo inspection:

* The production provider and target validation seam already has a provider-owned `RVVComputedMaskSegment2MemoryRouteFacts` surface and `RVVSegment2MemoryRouteValidationContract` for computed-mask segment2 load/store/update.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already compares segment2 load route descriptions against the provider-owned validation contract, including runtime ABI order, header/type mapping, mask facts, field roles, segment layout, statement counts, and selected typed RVV source provenance.
* `test/Target/TargetArtifactExportTest.cpp` already covers computed-mask segment2 load positive provider/candidate validation and many fail-closed mutations, including stale ABI order, field role, provider mirror, target leaf, headers, C type mapping, computed-mask plan, stale plain segment2 plan, mask source/role/form, segment count/layout, destination memory form, inactive-lane contract, passthrough layout, field destination, binding mirrors, segment tuple, segment store residue, and route statement mutations.
* The bounded gap is evidence hardening: generated-bundle script expectations and dry-run FileCheck coverage do not force computed-mask segment2 load to expose `tcrv_rvv.mask_tail_policy_route_family_plan` and `tcrv_rvv.mask_tail_policy_owner`, even though provider/target production facts carry them. The C++ route validation contract match also does not explicitly assert those two fields for the positive computed-mask segment2 load contract.
* The implementation should therefore strengthen the executable artifact ABI evidence and C++ fail-closed coverage for these mask/tail plan-owner facts without changing common EmitC semantics or inventing route authority in scripts/tests.

## Technical Approach

* Add mask/tail route-family plan and owner to the computed-mask segment2 load generated-bundle expected metadata.
* Add explicit script-level self-test expectations so missing/stale mask-tail metadata is treated as an evidence failure.
* Extend explicit and pre-realized generated-bundle dry-run FileCheck tests to expose the mask-tail plan and owner for computed-mask segment2 load.
* Extend C++ target artifact tests so the segment2 route validation contract positive match includes mask-tail plan and owner, and stale provider/candidate mask-tail facts fail closed for the computed-mask segment2 load route.
* Run focused script, C++ target, lit/dry-run, and `ssh rvv` checks. Runtime correctness will only be claimed if non-dry-run `ssh rvv` runs pass.

## Implementation Outcome

The bounded production seam already carried provider-owned computed-mask segment2 route facts and a target validation contract. This round hardened the executable artifact/ABI evidence and fail-closed tests for the missing mask/tail route-family plan-owner facts:

* Generated-bundle expectations now require `tcrv_rvv.mask_tail_policy_route_family_plan` and `tcrv_rvv.mask_tail_policy_owner` for computed-masked segment2 load.
* Script verification now summarizes the computed-mask segment2 load mask/tail boundary from selected/materialized body evidence, emitted RVV C++ facts, route metadata, artifact ABI, and runtime counts.
* Explicit and pre-realized dry-run FileCheck tests now assert that the mask/tail plan and owner are mirror-only-after-provider-route facts, not script or artifact authority.
* Target artifact C++ tests now compare mask-tail plan-owner fields in the positive provider contract and reject stale provider/candidate mask-tail mirrors before artifact acceptance.
* Explicit and pre-realized non-dry-run generated bundles passed on `ssh rvv` with active/inactive mask patterns, field0/field1 distinguishing lanes, old-field inactive-lane preservation, source preservation, and tail preservation.

## Current Phase

Finish. Implementation and focused checks passed; the remaining closeout is Trellis archive and one coherent commit.
