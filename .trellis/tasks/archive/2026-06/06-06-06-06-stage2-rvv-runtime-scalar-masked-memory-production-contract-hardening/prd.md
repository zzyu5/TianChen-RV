# Stage2 RVV Runtime-Scalar Masked Memory Production Contract Hardening

## Goal

Harden one active production validation contract in the runtime-scalar
compare-masked RVV memory seam before another executable artifact claim is
accepted. The bounded owner is the selected-body realization, RVV provider
route-family/statement-plan facts, target artifact validation, and generated
bundle ABI for `runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store`.

## Why Now

Commit `a7cdc21f` proved the current runtime-scalar compare-masked memory
generated bundles execute on `ssh rvv`, but that round closed without a
production source change. This task must not repeat a Trellis-only or
evidence-only closure. The next bottleneck is enforcing the same facts in
compiler/target code so stale or incomplete runtime scalar, compare-mask,
memory-role, inactive-lane, VL/control, operand-binding, header, or ABI facts
fail closed before executable artifact acceptance.

## What I Already Know

* TianChen-RV's current RVV authority chain is selected `tcrv.exec` envelope ->
  typed low-level `tcrv_rvv` body -> RVV plugin-owned legality/realization/
  route provider -> common EmitC materializer -> target artifact -> `ssh rvv`
  evidence when runtime behavior is claimed.
* `tcrv.exec` binds ABI/runtime roles; it does not own RVV compute, dtype,
  memory form, mask/tail policy, VL placement, intrinsic spelling, or route
  support.
* Common EmitC carries provider-built route facts and mirrors. It must not
  invent RVV semantics, infer support from artifact metadata, or repair stale
  provider facts.
* The runtime-scalar compare-masked memory path must preserve source vs
  destination memory roles, compare-mask provenance, inactive-lane behavior,
  runtime AVL/VL facts, runtime scalar ABI order, per-operand `abi|hdr`
  binding, header/type summaries, and provider-built statement plans.
* The previous archived memory ABI task concluded with a no-source-change
  justification plus non-dry-run `ssh rvv` evidence. This task is explicitly a
  follow-up production hardening task and must choose a concrete under-enforced
  contract after auditing the live implementation.

## Requirements

* Audit the current runtime-scalar compare-masked memory production seam before
  changing compiler source.
* Pick one concrete under-enforced production contract inside this seam. Good
  targets include swapped `src`/`dst` roles, missing/compressed per-operand
  header binding, stale compare-mask provenance, missing inactive-lane
  preservation facts, wrong runtime scalar ABI order, stale mask/tail policy,
  stale runtime AVL/VL facts, or target artifact validation accepting an
  executable claim without provider-owned memory facts.
* Implement the hardening in production C++/MLIR target code, not only in
  Trellis notes, scripts, metadata, or broad smoke tests.
* Preserve owner boundaries:
  * selected/pre-realized `tcrv_rvv` body carries runtime scalar, compare-mask,
    memory-role, inactive-lane, VL/control, and policy facts;
  * RVV provider/plan owners validate and derive provider route facts;
  * common EmitC remains neutral;
  * target artifact validation consumes rebuilt provider facts and rejects
    stale mirrors before bundle/header/object acceptance.
* Cover both runtime-scalar compare-masked store and load-store when the chosen
  contract is shared; otherwise explain why the selected sub-family is the
  precise owner.
* Add focused positive and fail-closed evidence for the chosen contract.
* Do not add a new route family, dtype/LMUL clone batch, frontend path,
  descriptor-driven route, dashboard/report-only artifact, or common EmitC RVV
  semantic branch.

## Acceptance Criteria

* [x] PRD records the concrete production contract chosen after code audit.
* [x] At least one active production source change hardens that contract in the
      runtime-scalar compare-masked memory validation seam.
* [x] Positive evidence proves existing runtime-scalar compare-masked memory
      artifacts still pass through provider route construction, target artifact
      validation, and generated-bundle dry-run coverage.
* [x] At least one negative fixture or script/C++ check proves the stale or
      missing fact now fails closed before executable artifact acceptance.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests pass.
* [x] `ssh rvv` evidence is run only if this round makes a runtime correctness
      claim after the production change.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new legacy `i32m1`, descriptor, source-front-door, artifact-name, or
      exact-intrinsic-as-authority route drift.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean.

## Out of Scope

* No new route family.
* No mass rewrite of product/dequant, contraction, standalone reduction,
  compare/select, conversion, segment2, or unrelated memory ownership.
* No high-level Linalg/Vector/StableHLO frontend work.
* No per-Linalg route authority.
* No dtype/LMUL clone batch.
* No performance tuning database, dashboard, or report-only closure.
* No source-front-door positive route.
* No descriptor-driven computation or direct C exporter as route authority.
* No common EmitC invention of RVV semantics.

## Technical Notes

* Direction source: Hermes/user brief on 2026-06-06.
* Relevant specs:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
* Prior context:
  * `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-runtime-scalar-cmp-masked-memory-artifact-abi-boundary/`
* Initial production targets:
  * `include/TianChenRV/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h`
  * `lib/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  * `scripts/rvv_generated_bundle_abi_e2e.py`
  * relevant `test/Scripts/*runtime-scalar-cmp-masked*` tests
  * relevant `test/Target/RVV/*runtime-scalar-cmp-masked*` fixtures

## Audit Finding And Selected Contract

The live audit found that runtime-scalar compare-masked store/load-store already
have provider-owned unit-stride masked memory route validation contracts,
including runtime AVL/VL selected-boundary facts, `abi|hdr` operand bindings,
mask producer facts, inactive-lane contracts, and candidate metadata mirrors.
The remaining production weakness is narrower: in
`validateRVVCompareSelectMaskRoutePayloadFacts`, unit-stride masked memory
routes first validate the description against
`RVVUnitStrideMaskedMemoryRouteValidationContract`, but route header, type
mapping, and ABI mapping checks still go through the generic compare/select
helpers using the mutable route description. Indexed and strided computed-mask
memory routes already validate those rebuilt route payload facts directly
against their provider-owned contracts.

This task hardens the unit-stride masked memory target artifact path so
runtime-scalar computed-mask store/load-store header, type, and ABI route
payload validation consume `RVVUnitStrideMaskedMemoryRouteValidationContract`
directly. Focused negative tests will mutate rebuilt route payload headers,
type mappings, and ABI value mappings for runtime-scalar masked memory and
prove the contract rejects the stale facts before statement-plan or executable
artifact acceptance.

## Evidence

* `cmake --build build --target tianchenrv-target-artifact-export-test
  tianchenrv-rvv-extension-plugin-test` completed successfully.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'runtime-scalar-cmp-masked-(load-store|store)(\\.mlir|-dry-run\\.test|.*fail-closed\\.test)'`
  from `build/test` passed 8 selected tests.
* New target C++ negative checks prove unit-stride runtime-scalar masked memory
  route payload validation rejects a missing `riscv_vector.h` route header, a
  stale `!tcrv_rvv.vl -> uint64_t` type mapping, and a stale
  `rhs_scalar` ABI value mapping before statement-plan acceptance.
* No new `ssh rvv` run was needed because this round makes a production
  validation-contract claim, not a new runtime correctness or performance
  claim.
* `git diff --check` and `git diff --cached --check` passed.
* Bounded added-line old-authority scan over touched production/test files found
  no new legacy route authority drift.

## Spec Update Decision

No `.trellis/spec/` change is needed. This round implements the existing
contract-first target artifact rule already captured in
`.trellis/spec/extension-plugins/rvv-plugin.md` and
`.trellis/spec/lowering-runtime/emitc-route.md`: target artifact validation
must consume provider-owned route validation contracts and treat candidate
metadata as mirrors only. The implementation extends the already documented
indexed/strided memory pattern to unit-stride runtime-scalar masked memory
without adding a new API, route family, or architectural convention.

## Current Phase

Finish. The production hardening and focused verification are complete.
