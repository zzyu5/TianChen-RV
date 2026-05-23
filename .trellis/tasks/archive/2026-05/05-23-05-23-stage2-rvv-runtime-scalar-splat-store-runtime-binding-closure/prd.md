# Stage2 RVV runtime scalar splat-store runtime and binding closure

## Goal

Close the RVV plugin-local runtime scalar splat-store provider boundary for the
active `RuntimeI32SplatStore` route. Route materialization must depend on typed
`tcrv_rvv` runtime-scalar-to-vector-store body facts, RVV-owned runtime
AVL/VL control validation, runtime scalar splat-store family-plan validation,
and `RouteOperandBindingPlan` closure before producing a
`TCRVEmitCLowerableRoute`.

## What I already know

* The task source is the Hermes Direction Brief supplied for this session.
* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repository state: worktree clean; HEAD `5cc6cac4 rvv: close scalar-broadcast provider binding`.
* No `.trellis/.current-task` existed before this task was created.
* The adjacent scalar-broadcast task is archived at `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-scalar-broadcast-runtime-binding-closure/`.
* Existing runtime scalar splat-store paths are bounded to explicit and pre-realized `runtime_i32_splat_store` artifact fixtures.
* Current planning already derives a `RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan`, but provider validation is weaker than the repaired scalar-broadcast family: it checks the family plan, mirrors, runtime ABI parameters, and binding plan id, but does not yet close the full runtime AVL/VL mirror set or route operand binding closure before materialization.

## Requirements

* Inventory active runtime scalar splat-store consumers and isolate non-splat-store routes.
* Require `RuntimeI32SplatStore` consumers to carry the runtime scalar splat-store family plan before provider materialization.
* Reject stale runtime scalar splat-store plans on non-consumers.
* Validate family plan operation and `runtime-scalar-splat-store` memory form.
* Validate runtime control and all mirrored runtime AVL/VL facts before provider materialization: SEW, LMUL, policy, runtime control plan id, config contract id, runtime VL contract id, runtime AVL source, VL def/scope/use, EmitC loop names, remaining AVL metadata, pointer advance metadata, bounded slice, and multi-VL marker.
* Validate runtime ABI order and runtime ABI parameter mirrors for `rhs_scalar,out,n`.
* Validate target/header/type mirrors: target leaf profile, provider-supported mirror, header declaration mirror, C type mapping, VL C type, vector type, and vector C type.
* Validate setvl, runtime scalar splat, store intrinsic mirrors, and route result name.
* Validate `RouteOperandBindingPlan` closure, including plan id, logical operands, runtime ABI order, materialized uses, runtime ABI parameter mirrors, and route operand binding summary mirror.
* Preserve common EmitC/export neutrality: runtime scalar splat-store semantics must remain in RVV planning/provider/realization/target support, not common materialization, artifact names, route ids, helper strings, descriptors, or mirror fields.

## Acceptance Criteria

* [x] Active explicit and pre-realized runtime scalar splat-store consumers are inventoried, with adjacent scalar-broadcast elementwise and vector-store routes explicitly excluded or fail-closed.
* [x] Missing runtime scalar splat-store family plans fail closed for `RuntimeI32SplatStore` consumers.
* [x] Stale runtime scalar splat-store family plans fail closed on non-consumers.
* [x] Runtime control mismatch failures are covered.
* [x] Runtime scalar ABI or operand-binding mismatch failures are covered.
* [x] Intrinsic, type, target/header, and result-name mirror mismatch failures are covered.
* [x] Selected-body realization still produces expected runtime scalar splat-store family facts for explicit and pre-realized paths.
* [x] Generated-bundle dry-runs cover representative explicit and pre-realized `runtime_i32_splat_store` at counts 7, 16, and 23 with at least two scalar values.
* [x] Real `ssh rvv` evidence covers representative scalar values, signed behavior, tail preservation, and runtime `n` variation.
* [x] Active-authority scan over touched RVV/plugin/export/script/test paths shows no new `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor, direct-C/source-export, or exact intrinsic spelling as route authority.
* [x] `check-tianchenrv` passes.
* [x] `git diff --check` passes.
* [x] Final `git status --short` is clean after commit.

## Definition of Done

* Focused implementation changes are made in RVV planning/provider/realization/target support only as needed.
* Focused lit/FileCheck or C++ coverage proves positive runtime splat-store provider validation and fail-closed cases.
* Generated bundle and real `ssh rvv` evidence are collected for representative explicit and pre-realized routes.
* Trellis task status, context, and workspace journal are kept truthful.
* The task is finished or archived using the repo's Trellis convention.
* One coherent commit records the completed task unless a blocker remains and is documented.

## Out of Scope

* New splat-store operations.
* Dtype or LMUL clone expansion.
* High-level frontend, Linalg, Vector, StableHLO, source-front-door, or source-artifact routes.
* Global tuning, dashboards, readiness state machines, or standalone evidence packaging.
* Reworking scalar-broadcast elementwise, plain compare-select, widening conversion, contraction, computed-mask select, computed-mask MAcc, standalone reduction, vector elementwise arithmetic, base memory movement, segment2, or other adjacent route families except for isolation checks.
* Moving runtime scalar splat-store semantics into common EmitC/export, artifact names, route ids, helper strings, descriptors, or mirror metadata.
* Reintroducing legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact authority, or exact intrinsic spelling as provider authority.

## Technical Approach

Follow the just-closed scalar-broadcast provider validation pattern, but keep
the implementation bounded to runtime scalar splat-store:

1. Compare `verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans`
   with `verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans`.
2. Add missing runtime AVL/VL mirror checks to the runtime splat-store provider
   verifier.
3. Add `verifyRVVRouteOperandBindingClosure` before provider materialization.
4. Add focused tests that mutate or construct stale/mismatched runtime
   splat-store plans and assert fail-closed diagnostics.
5. Re-run generated-bundle and `ssh rvv` evidence for explicit and
   pre-realized runtime splat-store fixtures only.

## Decision (ADR-lite)

**Context**: Runtime scalar splat-store is an active Stage2 RVV route family.
Its route-family plan already exists, but provider materialization still lacks
the full runtime and binding closure now required by adjacent repaired route
families.

**Decision**: Close runtime scalar splat-store at the RVV plugin-local provider
boundary by requiring validated typed body/config/runtime facts, route-family
mirrors, runtime ABI mirrors, and `RouteOperandBindingPlan` closure before route
construction.

**Consequences**: This strengthens the active `runtime_i32_splat_store` path
without adding coverage or moving semantics into common EmitC/export. Existing
fixtures that relied on stale or partial mirrors must become negative tests or
be repaired to carry the validated plan.

## Technical Notes

Initial read targets from the brief:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-scalar-broadcast-runtime-binding-closure/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/*runtime-i32-splat-store*.mlir`

## Completion Evidence

### Provider validation fields now checked

`verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyProviderPlans` now
requires the selected runtime scalar splat-store provider route description to
match the validated family plan for:

* operation and `runtime-scalar-splat-store` memory form;
* SEW, LMUL, tail policy, mask policy, runtime control plan id, config contract
  id, runtime VL contract id, runtime AVL source, VL def/scope/use, EmitC loop
  names, remaining AVL metadata, pointer advance metadata, bounded slice, and
  multi-VL marker;
* runtime ABI order and runtime ABI parameter mirrors for `rhs_scalar,out,n`;
* target leaf profile, provider-supported mirror, required header declaration
  mirror, C type mapping, VL C type, vector type, and vector C type;
* setvl, runtime scalar splat, and store intrinsic mirrors;
* result name;
* `RouteOperandBindingPlan` id, runtime ABI order, logical operand roles,
  materialized uses, runtime ABI parameter mirrors, and binding summary mirror.

### Active routes covered

* Explicit selected-body runtime scalar splat-store:
  `runtime_i32_splat_store`.
* Pre-realized selected-body runtime scalar splat-store:
  `runtime_i32_splat_store`.

### Adjacent route-family boundaries

Included only as isolation checks:

* scalar-broadcast add/sub/mul remain owned by
  `scalarBroadcastElementwiseRouteFamilyPlan`;
* `runtime_i32_splat_store` remains outside scalar-broadcast elementwise;
* scalar-broadcast routes reject stale runtime scalar splat-store plans.

Excluded from implementation by design:

* vector-vector elementwise arithmetic, plain compare-select, computed-mask
  select, widening conversion, contraction, standalone reduction,
  accumulation, base memory movement, segment2, source-front-door, and
  descriptor paths.

### Selected-body realization evidence

Focused lit coverage passed for:

* `test/Target/RVV/explicit-selected-body-artifact-runtime-i32-splat-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-i32-splat-store.mlir`
* `test/Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-materialization.mlir`
* `test/Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`
* `test/Transforms/LoweringBoundary/rvv-pre-realized-runtime-scalar-splat-store-negative.mlir`

### Runtime and binding closure evidence

`test/Plugin/RVVExtensionPluginTest.cpp` now covers:

* missing runtime scalar splat-store family plan rejection;
* stale runtime scalar splat-store plan rejection on non-consumers;
* runtime AVL source mirror mismatch rejection;
* runtime scalar splat intrinsic mirror mismatch rejection;
* result-name mirror mismatch rejection;
* runtime ABI parameter mirror mismatch rejection;
* runtime scalar binding role mismatch rejection;
* route operand binding summary mismatch rejection;
* route operand binding plan mirror mismatch rejection.

### Commands and results

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2 && ./build/bin/tianchenrv-rvv-extension-plugin-test`
  * Result: `RVV extension plugin smoke test passed`.
* Focused lit:
  * Result: 5/5 passed.
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  * Result: `rvv_generated_bundle_abi_e2e self-test passed`.
* Explicit generated-bundle dry-run:
  * Command: `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2_runtime_scalar_splat_store_runtime_binding_closure --run-id explicit-dry-run --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --op-kind runtime_i32_splat_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar 11 --rhs-scalar=-5`
  * Result: `dry_run_success`.
* Pre-realized generated-bundle dry-run:
  * Command: same as above with `--pre-realized-selected-body --run-id pre-realized-dry-run`.
  * Result: `dry_run_success`.
* Explicit real `ssh rvv`:
  * Command: same as explicit dry-run without `--dry-run`, `--run-id explicit-ssh`.
  * Result: `PASS op=runtime_i32_splat_store counts=7,16,23 rhs_scalars=11,-5`, with each case reporting `tail_preserved`.
* Pre-realized real `ssh rvv`:
  * Command: same as pre-realized dry-run without `--dry-run`, `--run-id pre-realized-ssh`.
  * Result: `PASS op=runtime_i32_splat_store counts=7,16,23 rhs_scalars=11,-5`, with each case reporting `tail_preserved`.
* Diff-only active-authority scan:
  * No new `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
    `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor,
    direct-C, or source-export authority in added lines.
  * New exact intrinsic spelling additions are confined to C++ tests asserting
    provider-derived mirrors and stale-mirror rejection, not route authority.
* `git diff --check`
  * Result: passed.
* `cmake --build build --target check-tianchenrv -j2`
  * Result: 361/361 passed.

### Spec update judgment

No `.trellis/spec/` update is needed. This task applies the already-recorded
RVV plugin-owned runtime/binding closure rule to the adjacent runtime scalar
splat-store family; it does not create a new command/API signature, cross-layer
contract, route authority rule, or testing convention.
