# Stage2 RVV plain segment2 memory runtime and binding closure

## Goal

Close the RVV plugin-local provider boundary for the existing
route-supported plain segment2 memory family. Materialization must require a
validated plain segment2 memory route-family plan, validated runtime AVL/VL
and ABI facts, segment field/layout facts, and `RouteOperandBindingPlan`
closure before producing a `TCRVEmitCLowerableRoute`.

## What I already know

* The task source is the Hermes Direction Brief supplied for this session.
* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repository state: worktree clean; HEAD
  `f0002e21 rvv: close computed-mask memory provider binding`.
* No `.trellis/.current-task` existed before this task was created.
* Specs require the RVV-first path to start from typed `tcrv_rvv` body facts,
  keep RVV route semantics in the RVV plugin, and keep common EmitC/export
  neutral.
* The previous archived computed-mask memory task closed this provider-boundary
  class for computed-mask memory by validating runtime control mirrors, route
  family mirrors, runtime ABI mirrors, and `RouteOperandBindingPlan` closure.
* Current code already derives, validates, and applies
  `RVVSelectedBodySegment2MemoryRouteFamilyPlan` for the two plain segment2
  memory routes:
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
* Current aggregate memory-family verifier requires the plain segment2 plan to
  exist for plain segment2 consumers, but there is no dedicated plain segment2
  provider verifier that rejects stale plans on non-consumers or compares the
  provider route description against the validated plain segment2 plan.
* Dedicated provider verifiers already exist for computed-mask memory and base
  memory movement; this task should follow that bounded pattern rather than
  adding route coverage.
* Active target fixtures and generated-bundle support already exist for
  pre-realized plain segment2 deinterleave/interleave routes. Computed-mask
  segment2 fixtures are adjacent computed-mask memory routes and must remain
  isolated from this plain segment2 closure.

## Requirements

* Inventory active plain segment2 memory consumers and isolate computed-mask
  segment2, computed-mask memory, base memory, elementwise arithmetic,
  compare/select, reduction, contraction, and accumulation routes.
* Require plain segment2 memory consumers to carry the segment2 memory
  route-family plan before provider materialization.
* Reject stale segment2 memory route-family plans on non-plain-segment2
  consumers.
* Validate plan operation, memory form, and deinterleave/interleave direction
  booleans before provider materialization.
* Validate runtime AVL/VL facts before materialization: SEW, LMUL, tail/mask
  policy, runtime control plan id, config contract id, runtime VL contract id,
  runtime AVL source, VL def/scope/use, EmitC loop kind, induction name,
  full-chunk VL name, loop VL name, remaining AVL metadata, pointer advance
  metadata, bounded slice, and multi-VL marker.
* Validate runtime ABI order and runtime ABI parameter mirrors for
  `src,out0,out1,n` deinterleave and `src0,src1,dst,n` interleave.
* Validate target/header/type mirrors: target leaf profile,
  provider-supported mirror, required header declarations, C type mapping,
  VL C type, vector type name, vector C type, and segment tuple C type.
* Validate provider-owned intrinsic mirrors for setvl, vector load, unit
  store, segment2 load, segment2 store, tuple create, and field extraction as
  applicable to deinterleave/interleave.
* Validate source/destination/segment memory layout, source and destination
  memory forms, segment count, field roles, field names, field source memory
  forms, and field destination memory forms.
* Validate `RouteOperandBindingPlan` closure, including selected plan id,
  logical operands, runtime ABI order, materialized uses, runtime ABI
  parameter mirrors, route description mirror plan id, and binding summary
  mirror.
* Keep common EmitC/export neutral. Plain segment2 memory route semantics must
  remain in RVV planning/provider/realization/target support, not in common
  materialization, artifact names, route ids, helper strings, descriptors, or
  mirror-only metadata.

## Acceptance Criteria

* [x] Active plain segment2 consumers are inventoried, with computed-mask
  segment2, computed-mask memory, base memory movement, elementwise
  arithmetic, compare/select, reduction, contraction, and accumulation routes
  explicitly excluded or owned by adjacent family verifiers.
* [x] Missing plain segment2 memory route-family plans fail closed for
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
* [x] Stale plain segment2 memory route-family plans fail closed on
  non-plain-segment2 consumers.
* [x] Runtime control mirror mismatch failures are covered.
* [x] Runtime ABI or route-operand binding mismatch failures are covered.
* [x] Operation, memory form, direction, segment count, field role/name,
  layout/type/header, and intrinsic mirror mismatch failures are covered.
* [x] Selected-body realization still produces expected plain segment2 memory
  family facts for representative pre-realized deinterleave and interleave
  paths.
* [x] Generated-bundle dry-runs cover representative route-supported plain
  segment2 interleave/deinterleave routes at counts 7, 16, and 23 where the
  existing harness supports them.
* [x] Real `ssh rvv` evidence covers representative route-supported plain
  segment2 interleave/deinterleave routes at counts 7, 16, and 23 where the
  existing harness supports them.
* [x] Unsupported explicit selected-body segment2 fixtures or harness modes,
  if still unsupported, are documented with exact boundaries and continuation
  points instead of broadening this task.
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
* Focused C++ or lit/FileCheck coverage proves positive plain segment2 provider
  validation and fail-closed cases.
* Generated bundle and real `ssh rvv` evidence are collected for
  representative supported plain segment2 routes.
* Trellis task status, context, and workspace journal are kept truthful.
* The task is finished or archived using the repo's Trellis convention.
* One coherent commit records the completed task unless a blocker remains and
  is documented.

## Out of Scope

* New segment operations or new segment route coverage.
* Dtype or LMUL clone expansion.
* High-level frontend, Linalg, Vector, StableHLO, source-front-door, or
  source-artifact routes.
* Global tuning, dashboards, readiness state machines, or standalone evidence
  packaging.
* Reworking computed-mask memory, computed-mask segment2, base memory
  movement, vector elementwise arithmetic, runtime scalar splat-store,
  scalar-broadcast elementwise, plain compare-select, widening conversion,
  contraction, computed-mask select, computed-mask MAcc, standalone reduction,
  or runtime-scalar standalone reduction except for isolation checks.
* Moving segment2 memory semantics into common EmitC/export, artifact names,
  route ids, helper strings, descriptors, or mirror metadata.
* Reintroducing legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact authority, or exact
  intrinsic spelling as provider authority.

## Technical Approach

1. Add a dedicated plain segment2 memory route-family provider verifier beside
   the existing base memory and computed-mask memory family verifiers in RVV
   route planning.
2. Call that verifier from the RVV EmitC route provider before
   materialization.
3. Keep the existing aggregate memory-family verifier as the broad family gate
   for memory umbrella missing-plan checks.
4. Add focused C++ coverage in `RVVExtensionPluginTest.cpp` for consumer
   classification, missing/stale plan failures, positive provider validation,
   route-family mirror mismatch failures, runtime ABI mismatch, and
   `RouteOperandBindingPlan` closure failures.
5. Use existing target fixtures and `rvv_generated_bundle_abi_e2e.py` support
   for generated-bundle and `ssh rvv` evidence. Do not add new route kinds.

## Decision (ADR-lite)

**Context**: Plain segment2 deinterleave/interleave routes are active Stage2
RVV memory-family routes. Their route-family plan is derived from typed body,
config, runtime, and segment field facts, but provider materialization only has
the broad memory-family plan-presence gate.

**Decision**: Close plain segment2 memory at the RVV plugin-local provider
boundary by requiring the selected route description and materialization inputs
to match the validated segment2 memory family plan and the
`RouteOperandBindingPlan`.

**Consequences**: This strengthens existing plain segment2 paths without adding
route coverage or moving semantics into common EmitC/export. Fixtures or tests
that rely on stale mirrors must fail closed or be repaired to carry validated
plan facts.

## Technical Notes

Read targets for this round:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-memory-runtime-binding-closure/prd.md`
* `.trellis/workspace/codex/journal-14.md`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* `test/Target/RVV/*segment2-interleave*.mlir`
* `test/Target/RVV/*segment2-deinterleave*.mlir`
* `test/Target/RVV/*computed*segment2*.mlir`

## Completion Evidence

### Provider validation fields now checked

`verifyRVVSelectedBodySegment2MemoryRouteFamilyProviderPlans` now requires the
selected plain segment2 provider route description to match the validated
segment2 memory family plan for:

* operation and plain segment2 consumer classification;
* stale plan rejection on non-plain-segment2 routes;
* family plan id and selected route operation;
* SEW, LMUL, tail policy, mask policy, runtime control plan id, config
  contract id, runtime VL contract id, runtime AVL source, VL def/scope/use,
  EmitC loop kind, induction name, full-chunk VL name, loop VL name, remaining
  AVL metadata, pointer advance metadata, bounded slice, and multi-VL marker;
* runtime ABI order and runtime ABI parameters;
* target leaf profile, provider-supported mirror, required header
  declarations, C type mapping, VL C type, vector type name, vector C type,
  and segment tuple C type;
* setvl, vector load, unit store, segment2 load, segment2 store, tuple create,
  and field extract intrinsic mirrors where applicable;
* result name, source/destination memory forms, segment memory layout, segment
  count, field roles, field names, field source memory forms, and field
  destination memory forms;
* `RouteOperandBindingPlan` id, logical operand roles, materialized uses,
  runtime ABI parameter mirrors, route description mirror plan id, and binding
  summary mirror;
* deinterleave/interleave direction classification.

### Active routes covered

The C++ provider test inventories and isolates these active plain segment2
memory consumers:

* `segment2_deinterleave_unit_store`
* `segment2_interleave_unit_load`

Adjacent route families are explicitly excluded or delegated to their existing
family verifiers:

* computed-mask segment2 memory:
  `computed_masked_segment2_load_unit_store` and
  `computed_masked_segment2_store_unit_load` remain computed-mask memory;
* base memory movement: `strided_load_unit_store` remains base memory;
* computed-mask select: `computed_mask_select` remains computed-mask select;
* computed-mask accumulation: `computed_masked_macc_add` remains
  computed-mask accumulation;
* elementwise arithmetic, plain compare/select, standalone reduction,
  contraction, conversion, scalar-broadcast, and scalar splat-store remain
  outside this verifier.

### Focused provider and selected-body evidence

Focused C++ coverage now exercises:

* positive provider validation for explicit realized
  `segment2_deinterleave_unit_store`;
* positive provider validation for explicit realized
  `segment2_interleave_unit_load`;
* missing plain segment2 memory plan failure;
* stale plain segment2 memory plan on non-consumer failure;
* direction boolean mismatch failure;
* runtime AVL mirror mismatch failure;
* runtime ABI mismatch failure;
* route operand binding role mismatch failure;
* route operand binding summary mismatch failure;
* segment load/store intrinsic mirror mismatch failures;
* field role/source-form and segment-count mirror mismatch failures.

Selected-body realization/export checks:

* `./build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
* `./build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
* corresponding `tcrv-translate --tcrv-export-target-header-artifact`
  pipelines for both fixtures.

These commands produced the expected plain segment2 route-family plan mirrors,
runtime ABI order, binding plan mirrors, target leaf profile mirrors, and
provider-supported mirrors.

### Generated-bundle and `ssh rvv` evidence

Explicit selected-body generated-bundle mode remains intentionally unsupported
for the two plain segment2 op kinds and fails closed with:

```text
--op-kind values ['segment2_deinterleave_unit_store', 'segment2_interleave_unit_load'] are not supported in explicit-selected-body mode
```

The exact continuation point, if later needed, is to add explicit selected-body
plain segment2 fixtures/harness support. This task does not broaden route
coverage.

Pre-realized generated-bundle dry-run evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run \
  --pre-realized-selected-body \
  --artifact-root /tmp/tianchenrv-rvv-segment2-closure \
  --run-id pre-realized-plain-segment2 --overwrite \
  --op-kind segment2_deinterleave_unit_store \
  --op-kind segment2_interleave_unit_load \
  --runtime-count 7 --runtime-count 16 --runtime-count 23
```

Result: `dry_run_success`.

Real `ssh rvv` evidence:

```text
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tianchenrv-rvv-segment2-closure \
  --run-id pre-realized-plain-segment2-ssh --overwrite \
  --op-kind segment2_deinterleave_unit_store \
  --op-kind segment2_interleave_unit_load \
  --runtime-count 7 --runtime-count 16 --runtime-count 23
```

Result:

```text
PASS op=segment2_deinterleave_unit_store counts=7,16,23
PASS op=segment2_interleave_unit_load counts=7,16,23
```

The run reported field-order distinguishing lanes and tail preservation for
counts 7, 16, and 23 on both routes.

### Checks

* [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [OK] pre-realized generated-bundle dry-run for plain segment2
  deinterleave/interleave at counts 7, 16, and 23
* [OK] pre-realized real `ssh rvv` evidence for plain segment2
  deinterleave/interleave at counts 7, 16, and 23
* [OK] explicit generated-bundle mode fail-closed for both plain segment2 op
  kinds with exact unsupported-mode diagnostic
* [OK] `git diff --check`
* [OK] `cmake --build build --target check-tianchenrv -j2` (361/361)

### Active-authority scan

The touched source/test diff adds no new `RVVI32M1`, `rvv-i32m1`, finite
`tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
descriptor, direct-C/source-export, or exact intrinsic spelling as route
authority. Added exact `__riscv_...` strings appear only in C++ provider tests
as provider-derived segment intrinsic mirrors or stale-mirror negative cases,
not as route authority.

### Spec update judgment

No `.trellis/spec/**` update is needed. Existing RVV plugin, unified EmitC
route, and testing specs already require typed `tcrv_rvv` body authority,
plugin-owned provider mapping, mirror-only metadata, runtime evidence, and
common EmitC neutrality. This task applies that existing contract to the plain
segment2 memory family rather than changing the long-term rule.
