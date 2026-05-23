# Stage2 RVV scalar-broadcast elementwise runtime binding closure

## Goal

Close the RVV plugin-local scalar-broadcast elementwise provider boundary for the active ScalarBroadcastAdd, ScalarBroadcastSub, and ScalarBroadcastMul routes. Route materialization must depend on typed vector-plus-runtime-scalar `tcrv_rvv` body facts, RVV-owned runtime control validation, scalar-broadcast family-plan validation, and `RouteOperandBindingPlan` closure before producing a `TCRVEmitCLowerableRoute`.

## What I already know

* The current HEAD is `703baa80 rvv: close plain compare-select provider binding`.
* The worktree was clean before this task was created.
* The task source is the Hermes Direction Brief supplied in this session.
* Existing scalar-broadcast elementwise fixtures and generated-bundle paths are expected to exist for add, sub, and mul.
* Adjacent route families such as compare-select, widening conversion, contraction, select, reduction, accumulation, and computed-mask MAcc already have stronger runtime and binding closure than scalar-broadcast elementwise.
* This task is bounded to existing explicit and pre-realized scalar-broadcast artifact paths. It must not add new operations, dtype/LMUL clones, frontend routes, global tuning, dashboards, or evidence packaging.

## Requirements

* Inventory active scalar-broadcast consumers and isolate non-broadcast routes before changing provider behavior.
* Require scalar-broadcast consumers to carry a scalar-broadcast elementwise family plan.
* Reject stale scalar-broadcast family plans on non-consumers.
* Validate scalar-broadcast family plan operation selection for add, sub, and mul.
* Validate memory form, runtime control, VL/AVL facts, runtime ABI order and parameters, target/header/type mirrors, vector/VL C types, intrinsic mirrors, result naming, and mirror-only metadata semantics before provider materialization.
* Validate setvl, vector load, scalar RHS splat or equivalent scalar-broadcast representation, arithmetic, and store intrinsic mirrors.
* Validate `RouteOperandBindingPlan` closure, including vector input/output/runtime count operands and scalar RHS ABI/binding consistency.
* Preserve common EmitC/export neutrality: scalar-broadcast semantics must remain in the RVV plugin/provider/realization boundary, not common materialization, artifact names, route ids, helper strings, descriptors, or mirror fields.
* Keep add/sub/mul isolated from runtime splat-store and vector-vector elementwise routes.

## Acceptance Criteria

* [x] Active scalar-broadcast add/sub/mul consumers are inventoried, with adjacent non-broadcast route families explicitly excluded or fail-closed.
* [x] Missing scalar-broadcast family plans fail closed for scalar-broadcast consumers.
* [x] Stale scalar-broadcast family plans fail closed on non-consumers.
* [x] Runtime control mismatch failures are covered.
* [x] Scalar RHS ABI or operand-binding mismatch failures are covered.
* [x] Intrinsic, type, target/header, and result-name mirror mismatch failures are covered.
* [x] Add, sub, and mul operation-kind selection is validated and tested without cross-family leakage.
* [x] Selected-body realization still produces the expected scalar-broadcast family facts for explicit and pre-realized scalar-broadcast paths.
* [x] Generated-bundle dry-runs cover representative explicit and pre-realized scalar_broadcast_add, scalar_broadcast_sub, and scalar_broadcast_mul at counts 7, 16, and 23 with two scalar RHS values.
* [x] Real `ssh rvv` evidence covers representative scalar-broadcast arithmetic values, signed RHS behavior, tail preservation, and runtime `n` variation.
* [x] Active-authority scan over touched RVV/plugin/export/script/test paths shows no `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*` authority, descriptor residue, source-front-door positive route, or exact intrinsic spelling as route authority.
* [x] `check-tianchenrv` passes.
* [x] `git diff --check` passes.
* [x] Final `git status --short` is clean after commit.

## Definition of Done

* Focused implementation changes are made only in RVV planning/provider/realization/target support areas as needed.
* Focused lit/FileCheck or C++ tests cover positive scalar-broadcast provider validation and fail-closed cases.
* Generated bundle and real `ssh rvv` evidence are collected for the representative routes.
* Task status, context, and workspace journal are kept truthful.
* The task is finished or archived using the repo's Trellis convention.
* One coherent commit records the completed task, unless a blocker remains and is documented.

## Out of Scope

* New scalar-broadcast operations.
* Dtype or LMUL clone expansion.
* High-level frontend, Linalg, Vector, StableHLO, or source-front-door routes.
* Global tuning, dashboards, readiness state machines, or standalone evidence packaging.
* Reworking plain compare-select, widening conversion, contraction, computed-mask select, computed-mask MAcc, standalone reduction, elementwise vector arithmetic, base memory movement, runtime i32 splat-store, segment2, or other adjacent route families except for isolation checks.
* Moving scalar-broadcast semantics into common EmitC/export, artifact names, route ids, helper strings, descriptors, or mirror metadata.
* Reintroducing legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact authority, or exact intrinsic spelling as provider authority.

## Technical Approach

Follow the adjacent closed provider families as patterns, especially plain compare-select from commit `703baa80`, but keep the scope limited to scalar-broadcast elementwise add/sub/mul. Strengthen scalar-broadcast family-plan verification so the RVV provider only materializes a route after validating typed body/config/runtime facts, ABI and operand binding closure, and expected mirrors. Update realization or target support only if inspection shows the scalar-broadcast family plan cannot carry the required facts.

## Decision (ADR-lite)

**Context**: Scalar-broadcast elementwise routes are active and adjacent to provider families that now validate runtime control and operand binding closure before route materialization.

**Decision**: Close scalar-broadcast add/sub/mul at the RVV plugin-local provider boundary, using typed `tcrv_rvv` body facts and fail-closed family-plan validation rather than adding new operations or widening route coverage.

**Consequences**: This should improve Stage2 route-supported/executable confidence for existing scalar-broadcast paths while keeping common EmitC neutral. It may require updating stale fixtures that previously depended on incomplete mirror or binding behavior.

## Technical Notes

Initial read targets from the brief:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-plain-compare-select-runtime-binding-closure/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/*scalar-broadcast*.mlir`

## Completion Evidence

### Provider validation fields now checked

`verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans`
now requires the selected scalar-broadcast provider route description to match
the validated family plan for:

* operation and `rhs-scalar-broadcast` memory form;
* SEW, LMUL, tail policy, mask policy, runtime control plan id, config
  contract id, runtime VL contract id, runtime AVL source, VL def/scope/use,
  EmitC loop names, remaining AVL metadata, pointer advance metadata, bounded
  slice, and multi-VL marker;
* runtime ABI order and runtime ABI parameter mirrors;
* target leaf profile, provider-supported mirror, required header declaration
  mirror, C type mapping, VL C type, vector type/C type;
* setvl, load, RHS scalar splat, arithmetic, and store intrinsic mirrors;
* result name;
* `RouteOperandBindingPlan` id, runtime ABI order, logical operand roles,
  materialized uses, runtime ABI parameter mirrors, and binding summary mirror.

### Active routes covered

* Explicit selected-body scalar-broadcast routes:
  `scalar_broadcast_add`, `scalar_broadcast_sub`, and
  `scalar_broadcast_mul`.
* Pre-realized selected-body scalar-broadcast routes:
  `scalar_broadcast_add`, `scalar_broadcast_sub`, and
  `scalar_broadcast_mul`.

### Adjacent route-family boundaries

Included only as isolation checks:

* runtime scalar splat-store remains owned by
  `runtimeScalarSplatStoreRouteFamilyPlan`;
* vector-vector elementwise arithmetic remains outside the
  scalar-broadcast family;
* non-scalar-broadcast consumers reject stale scalar-broadcast plans.

Excluded from implementation by design:

* plain compare-select, computed-mask select, widening conversion,
  contraction, standalone reduction, accumulation, base memory movement,
  segment2, runtime splat-store implementation, source-front-door, and
  descriptor paths.

### Checks run

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused FileCheck for all six scalar-broadcast target fixtures:
  explicit add/sub/mul `PLAN` and `HEADER`; pre-realized add/sub/mul
  `REALIZED`, `PLAN`, and `HEADER`.
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Generated-bundle dry-run, explicit selected body:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2_scalar_broadcast_runtime_binding_closure --run-id explicit --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar=5 --rhs-scalar=-3`
* Generated-bundle dry-run, pre-realized selected body:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_scalar_broadcast_runtime_binding_closure --run-id pre-realized --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar=5 --rhs-scalar=-3`
* Real `ssh rvv`, explicit selected body:
  same command without `--dry-run`, `--run-id explicit-ssh`.
  Result: PASS for add/sub/mul, counts `7,16,23`, RHS scalars `5,-3`,
  and `tail_preserved`.
* Real `ssh rvv`, pre-realized selected body:
  same command with `--pre-realized-selected-body`, `--run-id pre-realized-ssh`.
  Result: PASS for add/sub/mul, counts `7,16,23`, RHS scalars `5,-3`,
  and `tail_preserved`.
* Diff-only active-authority scan over touched RVV/plugin/script/test paths:
  no new `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
  descriptor/direct-C/source-export authority. New exact intrinsic spelling
  additions are confined to C++ tests that assert provider-derived mirrors and
  stale-mirror rejection, not route authority.
* `git diff --check`
* `cmake --build build --target check-tianchenrv -j2`: 361/361 passed.

### Spec update judgment

Updated `.trellis/spec/testing/mlir-testing-contract.md` to preserve the
durable evidence rule discovered in this round: RVV generated-bundle runtime
evidence over runtime `n` must check active lanes and guard/tail sentinels for
memory-writing routes. This is evidence quality guidance only, not route
authority.
