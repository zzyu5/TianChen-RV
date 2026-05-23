# Stage2 RVV vector elementwise arithmetic runtime and binding closure

## Goal

Close the RVV plugin-local elementwise arithmetic provider boundary for the
existing route-supported vector-vector add/sub/mul family. Provider
materialization must depend on typed `tcrv_rvv` body facts, RVV-owned runtime
AVL/VL control validation, elementwise arithmetic family-plan validation, and
`RouteOperandBindingPlan` closure before producing a
`TCRVEmitCLowerableRoute`.

## What I already know

* The task source is the Hermes Direction Brief supplied for this session.
* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repository state: worktree clean; HEAD
  `0846a4d0 rvv: close base memory provider binding`.
* No `.trellis/.current-task` existed before this task was created.
* The adjacent archived task
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-base-memory-runtime-binding-closure/`
  closed the same provider-boundary class for base memory movement by
  validating full runtime AVL/VL mirrors and `RouteOperandBindingPlan` closure.
* Existing elementwise arithmetic fixtures cover explicit and pre-realized
  add/sub/mul, masked add/sub/mul, and strided add where already supported by
  the route-family/harness.
* Bounded inspection confirms the current elementwise arithmetic provider
  verifier already requires the family plan, rejects stale plans, validates
  operation, memory form, runtime ABI parameters, route operand binding plan id,
  type/header/intrinsic/mask/stride mirrors, and runtime control plan id.
* The same verifier does not yet compare the full runtime AVL/VL mirror set
  against `plan.runtimeControlPlan`, and it returns success without calling
  `verifyRVVRouteOperandBindingClosure`.

## Requirements

* Inventory active elementwise arithmetic consumers and isolate
  scalar-broadcast, reduction, contraction, memory movement, splat-store, and
  adjacent computed-mask/runtime-scalar route families.
* Require active plain vector-vector add/sub/mul, masked add/sub/mul, masked
  mul, and strided add consumers to carry the elementwise arithmetic
  route-family plan before provider materialization.
* Reject stale elementwise arithmetic route-family plans on non-consumers.
* Validate plan operation and memory form before materialization.
* Validate runtime AVL/VL facts before materialization: SEW, LMUL, tail/mask
  policy, runtime control plan id, config contract id, runtime VL contract id,
  runtime AVL source, VL def/scope/use, EmitC loop names, remaining AVL
  metadata, pointer advance metadata, bounded slice, and multi-VL marker.
* Validate runtime ABI order and runtime ABI parameter mirrors.
* Validate target/header/type mirrors, including target leaf profile,
  provider-supported mirror, header declaration mirror, C type mapping, VL C
  type, vector/mask type names, and vector/mask C types.
* Validate setvl, load, strided-load, arithmetic, compare, merge, store, and
  strided-store intrinsic mirrors as applicable to the route form.
* Validate mask role/source/memory form, inactive-lane and passthrough layout
  facts, stride source/layout facts, and source/destination memory facts.
* Validate `RouteOperandBindingPlan` closure, including plan id, logical
  operands, runtime ABI order, materialized uses, runtime ABI parameter mirrors,
  and route operand binding summary mirror.
* Preserve common EmitC/export neutrality: elementwise arithmetic semantics
  remain in RVV planning/provider/realization/target support, not common
  materialization, artifact names, route ids, helper strings, descriptors, or
  mirror fields.

## Acceptance Criteria

* [x] Active elementwise arithmetic consumers are inventoried, with
  scalar-broadcast, reduction, contraction, memory, splat-store, computed-mask,
  and other adjacent routes explicitly excluded or fail-closed.
* [x] Missing elementwise arithmetic route-family plans fail closed for
  elementwise arithmetic consumers.
* [x] Stale elementwise arithmetic route-family plans fail closed on
  non-consumers.
* [x] Runtime control mirror mismatch failures are covered.
* [x] Runtime ABI or operand-binding mismatch failures are covered.
* [x] Mask, stride, type, target/header, result-name, and intrinsic mirror
  mismatch failures are covered.
* [x] Selected-body realization still produces expected elementwise arithmetic
  family facts for representative explicit and pre-realized paths.
* [x] Generated-bundle dry-runs cover representative explicit and pre-realized
  add/sub/mul at counts 7, 16, and 23, plus masked or strided variants where
  the existing harness supports them.
* [x] Real `ssh rvv` evidence covers representative explicit and pre-realized
  add/sub/mul at counts 7, 16, and 23, plus masked or strided variants where
  the existing harness supports them.
* [x] Unsupported fixture/harness combinations are documented with an exact
  continuation point instead of broadening this task.
* [x] Active-authority scan over touched RVV/plugin/export/script/test paths
  shows no new `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor,
  direct-C/source-export, or exact intrinsic spelling as route authority.
* [x] `check-tianchenrv` passes.
* [x] `git diff --check` passes.
* [x] Final `git status --short` is clean after commit.

## Definition of Done

* Focused implementation changes are made in RVV planning/provider/realization
  and target support only as needed.
* Focused C++ or lit/FileCheck coverage proves positive elementwise arithmetic
  provider validation and fail-closed cases.
* Generated bundle and real `ssh rvv` evidence are collected for
  representative supported routes.
* Trellis task status, context, and workspace journal are kept truthful.
* The task is finished or archived using the repo's Trellis convention.
* One coherent commit records the completed task unless a blocker remains and
  is documented.

## Out of Scope

* New arithmetic operations.
* Dtype or LMUL clone expansion.
* High-level frontend, Linalg, Vector, StableHLO, source-front-door, or
  source-artifact routes.
* Global tuning, dashboards, readiness state machines, or standalone evidence
  packaging.
* Reworking base memory movement, runtime scalar splat-store, scalar-broadcast
  elementwise, plain compare-select, widening conversion, contraction,
  computed-mask select, computed-mask MAcc, standalone reduction, segment2, or
  runtime-scalar compare-mask routes except for isolation checks.
* Moving arithmetic semantics into common EmitC/export, artifact names, route
  ids, helper strings, descriptors, or mirror metadata.
* Reintroducing legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact authority, or exact
  intrinsic spelling as provider authority.

## Technical Approach

Follow the just-closed base memory movement provider validation pattern, but
keep the implementation bounded to elementwise arithmetic:

1. Compare `verifyRVVSelectedBodyElementwiseArithmeticRouteFamilyProviderPlans`
   with the repaired base memory, scalar-broadcast, and runtime scalar
   splat-store provider validators.
2. Inventory active elementwise arithmetic route forms and existing
   fixture/harness support.
3. Add missing full runtime AVL/VL mirror checks to the elementwise arithmetic
   provider verifier.
4. Add `verifyRVVRouteOperandBindingClosure` before provider materialization.
5. Add focused positive and negative tests for missing/stale plans, runtime
   control mismatches, ABI/binding mismatches, and route-form-specific mirror
   mismatches.
6. Re-run generated-bundle and `ssh rvv` evidence for supported representative
   explicit and pre-realized elementwise arithmetic fixtures.

## Decision (ADR-lite)

**Context**: Elementwise arithmetic is an active Stage2 RVV route family. Its
route-family plan exists and is consumed by provider construction, but its
provider validation is weaker than adjacent repaired families.

**Decision**: Close elementwise arithmetic at the RVV plugin-local provider
boundary by requiring validated typed body/config/runtime facts,
route-family mirrors, runtime ABI mirrors, route-form-specific arithmetic
facts, and `RouteOperandBindingPlan` closure before route construction.

**Consequences**: This strengthens existing add/sub/mul paths without adding
route coverage or moving semantics into common EmitC/export. Existing fixtures
that relied on stale or partial mirrors must become negative tests or be
repaired to carry the validated plan.

## Technical Notes

Initial read targets from the brief:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-base-memory-runtime-binding-closure/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
* `test/Target/RVV/*artifact-sub*.mlir`
* `test/Target/RVV/*artifact-mul*.mlir`
* `test/Target/RVV/*masked-add*.mlir`
* `test/Target/RVV/*strided-add*.mlir`

## Completion Evidence

### Provider validation fields now checked

`verifyRVVSelectedBodyElementwiseArithmeticRouteFamilyProviderPlans` now
requires the selected elementwise arithmetic provider route description to
match the validated family plan for:

* operation and memory form;
* SEW, LMUL, tail policy, mask policy, runtime control plan id, config
  contract id, runtime VL contract id, runtime AVL source, VL def/scope/use,
  EmitC loop kind, induction name, full-chunk VL name, loop VL name, remaining
  AVL metadata, pointer advance metadata, bounded slice, and multi-VL marker;
* runtime ABI order and runtime ABI parameter mirrors;
* target leaf profile, provider-supported mirror, required header declaration
  mirror, C type mapping, VL C type, vector/mask type mirrors, and vector/mask
  C type mirrors;
* setvl, unit load, strided load, arithmetic, compare, masked merge, unit
  store, and strided store intrinsic mirrors;
* result name, mask role/source/memory form, inactive-lane contract, masked
  passthrough layout, strided layout, stride source mirrors, and
  source/destination memory-form mirrors;
* `RouteOperandBindingPlan` id, runtime ABI order, logical operand roles,
  materialized uses, runtime ABI parameter mirrors, and binding summary mirror.

### Active routes covered

Explicit selected-body routes covered by provider/unit tests, generated
dry-run, and real `ssh rvv` evidence:

* `add`
* `sub`
* `mul`
* `masked_add`
* `masked_sub`
* `masked_mul`
* `strided_add`

Pre-realized selected-body routes covered by existing realization fixtures,
generated dry-run, and real `ssh rvv` evidence:

* `add`
* `sub`
* `mul`
* `masked_add`
* `masked_sub`
* `masked_mul`
* `strided_add`

### Adjacent route-family boundaries

Included only as isolation checks:

* scalar-broadcast add stays owned by scalar-broadcast elementwise;
* strided load/unit store stays owned by base memory movement;
* computed-mask select, widening conversion, standalone reduction, and MAcc
  stay outside elementwise arithmetic classification.

Excluded from implementation by design:

* base memory movement, runtime scalar splat-store, scalar-broadcast
  elementwise, plain compare-select, widening conversion, contraction,
  computed-mask select, computed-mask MAcc, standalone reduction, segment2,
  runtime-scalar compare-mask, source-front-door, and descriptor paths.

### Selected-body realization and runtime evidence

Focused target fixtures already cover explicit and pre-realized selected-body
elementwise arithmetic realization/materialization:

* `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-masked-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-masked-sub.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-masked-mul.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-strided-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-sub.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-mul.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-masked-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-masked-sub.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-masked-mul.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-strided-add.mlir`

Generated-bundle dry-run evidence:

* explicit add/sub/mul/masked_add/masked_sub/masked_mul/strided_add:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --artifact-root artifacts/tmp/stage2-rvv-elementwise-arithmetic-runtime-binding-closure
  --run-id explicit-elementwise --overwrite --runtime-count 7
  --runtime-count 16 --runtime-count 23 --op-kind add --op-kind sub
  --op-kind mul --op-kind masked_add --op-kind masked_sub --op-kind masked_mul
  --op-kind strided_add` -> `dry_run_success`.
* pre-realized add/sub/mul/masked_add/masked_sub/masked_mul/strided_add:
  same command with `--pre-realized-selected-body` and run id
  `pre-realized-elementwise` -> `dry_run_success`.

Real `ssh rvv` evidence:

* explicit add/sub/mul/masked_add/masked_sub/masked_mul/strided_add at counts
  7, 16, and 23 -> all `PASS`; masked routes reported passthrough-preserved
  false lanes.
* pre-realized add/sub/mul/masked_add/masked_sub/masked_mul/strided_add at
  counts 7, 16, and 23 -> all `PASS`; masked routes reported
  passthrough-preserved false lanes.

### Checks

* [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2 && ./build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [OK] explicit generated-bundle dry-run for add/sub/mul, masked add/sub/mul,
  masked mul, and strided add at counts 7, 16, and 23
* [OK] pre-realized generated-bundle dry-run for add/sub/mul, masked
  add/sub/mul, masked mul, and strided add at counts 7, 16, and 23
* [OK] explicit real `ssh rvv` evidence for add/sub/mul, masked add/sub/mul,
  masked mul, and strided add at counts 7, 16, and 23
* [OK] pre-realized real `ssh rvv` evidence for add/sub/mul, masked
  add/sub/mul, masked mul, and strided add at counts 7, 16, and 23
* [OK] `git diff --check`
* [OK] `cmake --build build --target check-tianchenrv -j2` (361/361)

### Active-authority scan

The touched source/test diff adds no `RVVI32M1`, `rvv-i32m1`, finite
`tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
descriptor, direct-C/source-export, or exact intrinsic spelling as route
authority. Full touched-path scans still report pre-existing provider-derived
intrinsic constants, legacy fail-closed tests, script self-test negative
strings, and PRD non-goal wording; those are not new positive route authority
from this task.

### Spec update judgment

No `.trellis/spec/**` update is needed. The existing RVV plugin, unified EmitC
route, and testing specs already require typed `tcrv_rvv` body authority,
plugin-owned route providers, mirror-only metadata, runtime evidence, and
common EmitC neutrality. This task applies that existing contract to the
elementwise arithmetic family rather than changing the long-term rule.
