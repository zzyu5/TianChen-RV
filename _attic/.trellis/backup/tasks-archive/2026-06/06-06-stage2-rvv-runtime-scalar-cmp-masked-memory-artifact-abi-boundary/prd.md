# Stage2 RVV Runtime-Scalar Compare-Masked Memory Artifact ABI Boundary

## Goal

Make the pre-realized runtime-scalar compare-masked RVV load-store/store selected-body route truthful at the executable generated-artifact ABI boundary. The owner is the memory seam from typed `tcrv_rvv` body facts through RVV plugin-owned validation, `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact export, generated bundle ABI, and focused `ssh rvv` correctness evidence when executable behavior is claimed.

## Why Now

The preceding runtime-scalar masked standalone reduction task established executable `ssh rvv` evidence for a scalar-result boundary, but it was Trellis/journal-only. This task must avoid another evidence-only round unless repository audit proves the memory path is already complete; if the path is already executable, the PRD narrows to the exact under-validated production memory-boundary fact discovered during audit.

## What I Already Know

* The current real mainline is selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned realization/route validation -> common EmitC materializer -> generated target artifact -> `ssh rvv` evidence.
* `tcrv.exec` binds ABI/runtime roles and selected variants; it does not own RVV compute, dtype, memory, mask, policy, or intrinsic semantics.
* The RVV plugin must derive or validate memory form, load/store roles, runtime scalar compare-mask provenance, mask/tail policy, runtime AVL/VL, type mapping, header/prototype bindings, and per-operand ABI order from typed body/config/capability/runtime facts.
* Emission-plan diagnostics, route ids, helper names, generated artifact names, manifests, and common EmitC/export code are mirrors/mechanics only; they must not become semantic authority.
* Existing read-first files for this task include RVV runtime-scalar memory realization owners, computed-mask memory route-family plan owners, runtime-scalar compare-masked memory construction in `RVVEmitCRouteProvider.cpp`, target artifact route-family validation, generated bundle ABI script tests, and bounded i64/LMUL m2 references.

## Requirements

* Audit the current runtime-scalar compare-masked memory path before changing source files.
* If the current path is dry-run-only, stale, or under-validated, harden the production memory seam in the smallest owner-local module that actually controls the missing fact.
* Preserve separation of responsibility:
  * selected/pre-realized `tcrv_rvv` body carries the memory, runtime scalar, compare-mask, policy, and VL facts;
  * RVV provider/plan owner validates and derives RVV route facts;
  * common EmitC materialization remains neutral and does not invent RVV semantics;
  * target artifact export validates executable ABI/header/prototype/statements rather than trusting route ids or metadata.
* Cover both runtime-scalar compare-masked load-store and store-only boundaries as appropriate for the discovered gap.
* Add or update focused fail-closed evidence for at least one stale or missing executable-boundary fact, such as wrong runtime scalar ABI order, missing compare-mask provenance, stale mask/tail policy, swapped load/store roles, missing source preservation, missing header/prototype binding, wrong generated C type, or unsupported executable route claim.
* Add positive generated-bundle evidence through materialized selected boundary, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness if runtime correctness is claimed.

## Audit Findings And Resolution

Repository audit found that the production runtime-scalar compare-masked memory seam is already owner-local and fail-closed at the required boundary. No compiler source change was justified for this round.

* `RVVRuntimeScalarMemorySelectedBodyRealizationOwner` materializes selected pre-realized store and load-store bodies with explicit runtime ABI order `lhs,rhs_scalar,src,dst,n`, runtime `setvl`, compare-mask construction, source load, destination role, and inactive-lane preservation semantics.
* `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners` verifies runtime-scalar producer facts, memory form, operand-binding facts, typed config, materialization leaves, runtime ABI roles and order, statement-plan flags, route-control facts, and mask/tail policy before a route can be built.
* `RVVEmitCRouteProvider` routes runtime-scalar computed-mask memory through provider-owned materialization facts, operand-binding facts, and computed-mask memory statement plans; common EmitC only materializes provider route facts.
* `RVVTargetArtifactRouteFamilyValidation` validates `RuntimeScalarComputedMaskStore` and `RuntimeScalarComputedMaskLoadStore` against the unit-stride masked memory contract, including header/prototype binding, runtime ABI parameters, scalar splat, compare, masked memory statements, runtime AVL/VL contract, provider mirrors, and stale fact rejection.
* Existing focused C++ tests already reject stale runtime-scalar ABI/fact boundaries, stale base-memory plans, stale runtime-scalar splat intrinsics, stale runtime AVL/VL contracts, stale mask/tail and typed-compute mirrors, and the retired direct pre-realized route-entry shortcut.

The exact remaining boundary was executable evidence, not another production source gap. This round therefore repaired the PRD to record a precise no-source-change justification and produced non-dry-run generated-bundle `ssh rvv` evidence for both runtime-scalar compare-masked memory operations:

* `runtime_scalar_cmp_masked_load_store`
* `runtime_scalar_cmp_masked_store`

The generated bundle evidence covered counts `0,1,16,23,257` and rhs scalars `-37,91`; both operations reported `source_preserved` and `tail_preserved`, with mixed mask cases and inactive-lane preservation on non-trivial counts.

## Acceptance Criteria

* [x] The PRD is refined with concrete audit findings before implementation.
* [x] Production code either changes the runtime-scalar compare-masked memory seam or records a precise no-source-change justification tied to a hardened production validation gap.
* [x] Positive evidence exists for the relevant pre-realized runtime-scalar compare-masked load-store/store path through generated artifact export.
* [x] At least one focused fail-closed test covers a missing/stale executable-boundary fact in this memory seam.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests pass.
* [x] `ssh rvv` correctness evidence is recorded if executable runtime behavior is claimed.
* [x] Bounded old-authority scan over touched files and added diff lines is clean for this seam.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are clean.

## Out of Scope

* No new memory route family.
* No dtype/LMUL clone batch.
* No high-level Linalg/Vector/StableHLO frontend work.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or index/report-only task.
* No source-front-door positive route.
* No common EmitC invention of RVV semantics.
* No mass rewrite of product/dequant, contraction, standalone reduction, compare/select, conversion, or unrelated provider-fact ownership.

## Technical Notes

* This PRD starts from the Hermes Direction Brief in the user prompt on 2026-06-06.
* Relevant specs to read before implementation:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
* Relevant prior task:
  * `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-runtime-scalar-masked-standalone-reduction-executable-artifact-abi-boundary/`
* Initial production targets to inspect:
  * `include/TianChenRV/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h`
  * `lib/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`
  * `include/TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  * `scripts/rvv_generated_bundle_abi_e2e.py`
  * relevant `test/Scripts/*runtime-scalar-cmp-masked*` and `test/Target/RVV/*runtime-scalar-cmp-masked*` fixtures.

## Current Phase

Finish. The production seam audit, focused fail-closed checks, generated-bundle dry-runs, target fixture lit checks, and non-dry-run `ssh rvv` evidence are complete. The task is ready to archive with a no-source-change production justification and executable memory artifact evidence.
