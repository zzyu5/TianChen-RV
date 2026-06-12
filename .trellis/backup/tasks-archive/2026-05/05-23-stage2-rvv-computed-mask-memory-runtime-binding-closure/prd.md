# Stage2 RVV computed-mask memory runtime and binding closure

## Goal

Close the RVV plugin-local provider boundary for the existing
route-supported computed-mask memory family. Materialization must require a
validated computed-mask memory route-family plan, validated runtime AVL/VL and
ABI facts, typed body mask/memory facts, and `RouteOperandBindingPlan` closure
before producing a `TCRVEmitCLowerableRoute`.

## What I already know

* The task source is the Hermes Direction Brief supplied for this session.
* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repository state: worktree clean; HEAD
  `cce29794 rvv: close elementwise arithmetic provider binding`.
* No `.trellis/.current-task` existed before this task was created.
* Existing specs require the RVV-first path to start from a typed
  `tcrv_rvv` body, keep RVV semantics in the RVV plugin, and keep common
  EmitC/export neutral.
* The previous archived task closed this provider-boundary class for
  elementwise arithmetic by validating runtime control mirrors, route-family
  mirrors, runtime ABI mirrors, and `RouteOperandBindingPlan` closure.
* Current code already derives and applies
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` for computed-mask memory
  routes, and validates the plan internally.
* Current provider entry only requires the computed-mask memory plan to exist
  for computed-mask memory consumers before materialization. It does not yet
  reject stale computed-mask memory plans on non-consumers or compare the
  selected route description against the validated computed-mask memory plan.
* Active computed-mask memory consumers are:
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`,
  `computed_masked_unit_load_store`,
  `computed_masked_strided_store`,
  `computed_masked_strided_load_unit_store`,
  `computed_masked_indexed_gather_load_unit_store`,
  `computed_masked_indexed_scatter_store_unit_load`,
  `computed_masked_segment2_load_unit_store`, and
  `computed_masked_segment2_store_unit_load`.

## Requirements

* Inventory active computed-mask memory consumers and isolate base memory,
  plain segment2, elementwise arithmetic, compare/select, reduction,
  contraction, and accumulation routes.
* Require computed-mask memory consumers to carry the computed-mask memory
  route-family plan before provider materialization.
* Reject stale computed-mask memory route-family plans on non-consumers.
* Validate plan operation, memory form, and computed-mask producer source
  before materialization.
* Validate runtime AVL/VL facts before materialization: SEW, LMUL, tail/mask
  policy, runtime control plan id, config contract id, runtime VL contract id,
  runtime AVL source, VL def/scope/use, EmitC loop kind, induction name,
  full-chunk VL name, loop VL name, remaining AVL metadata, pointer advance
  metadata, bounded slice, and multi-VL marker.
* Validate runtime ABI order and runtime ABI parameter mirrors.
* Validate target/header/type mirrors: target leaf profile,
  provider-supported mirror, required header declarations, C type mapping, VL
  C type, vector/index/mask type names, and vector/index/mask C types.
* Validate provider-owned intrinsic mirrors for setvl, unit load, index load,
  index scale, scalar splat, compare, masked load, masked store, strided
  store, indexed store, segment2 load/store/field extraction, and result
  naming where applicable to the selected memory form.
* Validate mask role/source/memory form, inactive-lane contract,
  passthrough layout, masked/strided/indexed/segment memory layout, source and
  destination memory forms, stride source, index facts, and segment2 field
  roles/names/source/destination forms.
* Validate `RouteOperandBindingPlan` closure, including selected plan id,
  logical operands, runtime ABI order, materialized uses, runtime ABI
  parameter mirrors, mirror plan id, and binding summary mirror.
* Keep common EmitC/export neutral. Computed-mask memory route semantics must
  remain in RVV planning/provider/realization/target support, not in common
  materialization, artifact names, route ids, helper strings, descriptors, or
  mirror-only metadata.

## Acceptance Criteria

* [x] Active computed-mask memory consumers are inventoried, with base memory,
  plain segment2, elementwise arithmetic, compare/select, reduction,
  contraction, and accumulation routes explicitly excluded or owned by their
  adjacent family verifiers.
* [x] Missing computed-mask memory route-family plans fail closed for
  computed-mask memory consumers.
* [x] Stale computed-mask memory route-family plans fail closed on
  non-computed-mask-memory consumers.
* [x] Runtime control mirror mismatch failures are covered.
* [x] Runtime ABI or route-operand binding mismatch failures are covered.
* [x] Mask producer, mask, stride, index, segment, type/header, result-name,
  and intrinsic mirror mismatch failures are covered.
* [x] Selected-body realization still produces expected computed-mask memory
  family facts for representative explicit and pre-realized paths.
* [x] Generated-bundle dry-runs cover representative supported explicit and
  pre-realized computed-mask memory routes at counts 7, 16, and 23, including
  at least one non-segment memory route and one computed-mask segment2 route
  if supported by the existing harness.
* [x] Real `ssh rvv` evidence covers representative supported explicit and
  pre-realized computed-mask memory routes at counts 7, 16, and 23, including
  at least one non-segment memory route and one computed-mask segment2 route
  if supported by the existing harness.
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
* Focused C++ or lit/FileCheck coverage proves positive computed-mask memory
  provider validation and fail-closed cases.
* Generated bundle and real `ssh rvv` evidence are collected for
  representative supported routes.
* Trellis task status, context, and workspace journal are kept truthful.
* The task is finished or archived using the repo's Trellis convention.
* One coherent commit records the completed task unless a blocker remains and
  is documented.

## Out of Scope

* New computed-mask memory operations or new memory-route coverage.
* Dtype or LMUL clone expansion.
* High-level frontend, Linalg, Vector, StableHLO, source-front-door, or
  source-artifact routes.
* Global tuning, dashboards, readiness state machines, or standalone evidence
  packaging.
* Reworking vector elementwise arithmetic, base memory movement, runtime
  scalar splat-store, scalar-broadcast elementwise, plain compare-select,
  widening conversion, contraction, computed-mask select, computed-mask MAcc,
  standalone reduction, runtime-scalar standalone reduction, or plain segment2
  routes except for isolation checks.
* Moving computed-mask memory semantics into common EmitC/export, artifact
  names, route ids, helper strings, descriptors, or mirror metadata.
* Reintroducing legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact authority, or exact
  intrinsic spelling as provider authority.

## Technical Approach

1. Add a dedicated computed-mask memory route-family provider verifier beside
   the existing family verifiers in RVV route planning.
2. Call that verifier from the RVV EmitC route provider before materialization.
3. Keep the existing aggregate memory-family verifier as the broad family gate
   for memory umbrella missing-plan checks.
4. Add focused C++ coverage in `RVVExtensionPluginTest.cpp` for consumer
   classification, missing/stale plan failures, positive provider validation,
   route-family mirror mismatch failures, runtime ABI mismatch, and
   `RouteOperandBindingPlan` closure failures.
5. Use existing target fixtures and `rvv_generated_bundle_abi_e2e.py` support
   for generated-bundle and `ssh rvv` evidence. Do not add new route kinds.

## Decision (ADR-lite)

**Context**: Computed-mask memory routes are active Stage2 RVV memory-family
routes. Their route-family plan is derived from typed body/config/runtime facts
and consumed by provider construction, but the provider boundary is weaker than
recently closed adjacent families.

**Decision**: Close computed-mask memory at the RVV plugin-local provider
boundary by requiring the selected route description and materialization
inputs to match the validated computed-mask memory family plan and the
`RouteOperandBindingPlan`.

**Consequences**: This strengthens existing computed-mask memory paths without
adding route coverage or moving semantics into common EmitC/export. Fixtures
or tests that rely on stale mirrors must fail closed or be repaired to carry
validated plan facts.

## Technical Notes

Read targets for this round:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-elementwise-arithmetic-runtime-binding-closure/prd.md`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* `test/Target/RVV/*computed-masked*load*store*.mlir`
* `test/Target/RVV/*computed-masked*segment2*.mlir`

## Completion Evidence

### Provider validation fields now checked

`verifyRVVSelectedBodyComputedMaskMemoryRouteFamilyProviderPlans` now requires
the selected computed-mask memory provider route description to match the
validated family plan for:

* operation and memory form;
* computed-mask route-family plan id and mask producer source;
* SEW, LMUL, tail policy, mask policy, runtime control plan id, config
  contract id, runtime VL contract id, runtime AVL source, VL def/scope/use,
  EmitC loop kind, induction name, full-chunk VL name, loop VL name, remaining
  AVL metadata, pointer advance metadata, bounded slice, and multi-VL marker;
* runtime ABI order and runtime ABI parameter mirrors;
* target leaf profile, provider-supported mirror, required header
  declarations, C type mapping, VL C type, vector/index/mask type mirrors,
  and vector/index/mask C type mirrors;
* setvl, unit load, index load, index scale, scalar splat, compare, masked
  load, masked/store, strided store, indexed store, segment load/store/field
  extraction, and primary route intrinsic mirrors;
* result name, mask name, mask role/source/memory form, inactive-lane
  contract, passthrough layout, masked/strided/indexed/segment memory layout,
  source/destination memory forms, stride source, index facts, and segment2
  field roles/names/source/destination forms;
* `RouteOperandBindingPlan` id, runtime ABI order, logical operand roles,
  materialized uses, runtime ABI parameter mirrors, route description mirror
  plan id, and binding summary mirror;
* runtime-scalar vs vector-compare mask-producer classification and
  store-only/load-merge classification.

### Active routes covered

The C++ provider test inventories and isolates these active computed-mask
memory consumers:

* `runtime_scalar_cmp_masked_store`
* `runtime_scalar_cmp_masked_load_store`
* `computed_masked_unit_load_store`
* `computed_masked_strided_store`
* `computed_masked_strided_load_unit_store`
* `computed_masked_indexed_gather_load_unit_store`
* `computed_masked_indexed_scatter_store_unit_load`
* `computed_masked_segment2_load_unit_store`
* `computed_masked_segment2_store_unit_load`

Adjacent route families are explicitly excluded or delegated to their existing
family verifiers:

* base memory movement: `strided_load_unit_store` and related base memory
  forms;
* plain segment2 memory: `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load`;
* computed-mask select: `computed_mask_select` and runtime-scalar
  compare-select forms;
* computed-mask accumulation: computed-mask MAcc and computed-mask standalone
  reduction forms;
* elementwise arithmetic, plain compare-select, widening conversion,
  contraction, standalone reduction, scalar-broadcast elementwise, and
  runtime scalar splat-store remain outside this provider verifier.

### Selected-body realization and runtime evidence

Focused provider/unit coverage now exercises:

* runtime-scalar computed-mask store;
* vector-compare computed-mask strided load;
* vector-compare computed-mask indexed gather;
* vector-compare computed-mask segment2 load;
* missing computed-mask memory plan failure;
* stale computed-mask memory plan on non-consumer failure;
* operation, mask-producer, runtime control, runtime ABI, route binding,
  mask/stride/index/segment/intrinsic mirror mismatch failures.

Generated-bundle dry-run evidence:

* explicit supported computed-mask memory subset at counts 7, 16, and 23:
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`,
  `computed_masked_strided_load_unit_store`,
  `computed_masked_indexed_gather_load_unit_store`,
  `computed_masked_indexed_scatter_store_unit_load`,
  `computed_masked_segment2_load_unit_store`, and
  `computed_masked_segment2_store_unit_load` -> `dry_run_success`.
* pre-realized computed-mask memory full supported set at counts 7, 16, and
  23: all nine active computed-mask memory op kinds listed above ->
  `dry_run_success`.

The explicit harness does not currently support
`computed_masked_unit_load_store` or `computed_masked_strided_store`; the
script fails closed with:

```text
--op-kind values ['computed_masked_unit_load_store', 'computed_masked_strided_store'] are not supported in explicit-selected-body mode
```

The exact continuation point for those two explicit fixtures is to add
explicit selected-body fixture/harness support in `rvv_generated_bundle_abi_e2e.py`
and corresponding `test/Target/RVV/explicit-selected-body-artifact-*` files,
not to broaden this provider-boundary task.

Real `ssh rvv` evidence:

* explicit supported computed-mask memory subset at counts 7, 16, and 23:
  all seven supported op kinds passed, including runtime-scalar store/load,
  strided load, indexed gather/scatter, and segment2 load/store. Outputs
  reported active lanes, inactive-lane preservation, source/tail preservation,
  stride/index/segment distinguishing checks as applicable.
* pre-realized computed-mask memory full supported set at counts 7, 16, and
  23: all nine active op kinds passed, including
  `computed_masked_unit_load_store` and `computed_masked_strided_store`.

### Checks

* [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [OK] explicit generated-bundle dry-run for supported computed-mask memory
  subset at counts 7, 16, and 23
* [OK] pre-realized generated-bundle dry-run for all nine active
  computed-mask memory op kinds at counts 7, 16, and 23
* [OK] explicit real `ssh rvv` evidence for supported computed-mask memory
  subset at counts 7, 16, and 23
* [OK] pre-realized real `ssh rvv` evidence for all nine active computed-mask
  memory op kinds at counts 7, 16, and 23
* [OK] `git diff --check`
* [OK] `cmake --build build --target check-tianchenrv -j2` (361/361)

### Active-authority scan

The touched source/test diff adds no new `RVVI32M1`, `rvv-i32m1`, finite
`tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
descriptor, direct-C/source-export, or exact intrinsic spelling as route
authority. Full touched-path scans still report PRD non-goal wording,
pre-existing fail-closed/source-front-door tests, provider-derived intrinsic
constants, and C++ mirror/mismatch assertions. The new computed-mask memory
test mentions exact intrinsic spellings only as provider-derived mirrors or
stale-mirror negative cases, not as route authority.

### Spec update judgment

No `.trellis/spec/**` update is needed. Existing RVV plugin, unified EmitC
route, and testing specs already require typed `tcrv_rvv` body authority,
plugin-owned provider mapping, mirror-only metadata, runtime evidence, and
common EmitC neutrality. This task applies that existing contract to the
computed-mask memory family rather than changing the long-term rule.
