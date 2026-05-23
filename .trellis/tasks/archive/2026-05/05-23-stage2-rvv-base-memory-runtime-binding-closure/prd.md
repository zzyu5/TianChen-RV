# Stage2 RVV base memory movement runtime and binding closure

## Goal

Close the RVV plugin-local base memory movement provider boundary for the
existing unit, masked, strided, indexed-gather, and indexed-scatter memory
movement routes. Route materialization must depend on typed `tcrv_rvv` memory
body facts, RVV-owned runtime AVL/VL control validation, base memory movement
family-plan validation, and `RouteOperandBindingPlan` closure before producing
a `TCRVEmitCLowerableRoute`.

## What I already know

* The task source is the Hermes Direction Brief supplied for this session.
* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repository state: worktree clean; HEAD
  `ee57b672 rvv: close runtime splat-store provider binding`.
* No `.trellis/.current-task` existed before this task was created.
* The adjacent runtime scalar splat-store task is archived at
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-runtime-scalar-splat-store-runtime-binding-closure/`.
* Existing active base memory fixtures cover explicit and/or pre-realized unit
  load/store, masked unit load/store, strided memory, indexed gather, and
  indexed scatter routes.
* Bounded inspection from the brief says the current base memory verifier
  validates family-plan mirrors, ABI parameters, and binding plan id, but does
  not yet close the full runtime AVL/VL mirror set or route operand binding
  closure now present in adjacent repaired route families.

## Requirements

* Inventory active base memory consumers and isolate non-memory routes.
* Require base memory consumers to carry the base memory movement family plan
  before provider materialization.
* Reject stale base memory movement plans on non-consumers.
* Validate family plan operation and memory form for unit, masked, strided,
  indexed-gather, and indexed-scatter routes.
* Validate runtime control and all mirrored runtime AVL/VL facts before provider
  materialization: SEW, LMUL, policy, runtime control plan id, config contract
  id, runtime VL contract id, runtime AVL source, VL def/scope/use, EmitC loop
  names, remaining AVL metadata, pointer advance metadata, bounded slice, and
  multi-VL marker.
* Validate runtime ABI order and runtime ABI parameter mirrors for each active
  base memory route form.
* Validate target/header/type mirrors, including target leaf profile,
  provider-supported mirror, header declaration mirror, C type mapping, VL C
  type, vector/index/mask/VL C types, and vector/index/mask type mirrors.
* Validate setvl, load, store, masked, strided, indexed-gather, and
  indexed-scatter intrinsic mirrors as applicable to the route form.
* Validate mask role/source/memory form, inactive-lane policy, passthrough
  layout facts, stride layout facts, index source/destination layout facts, and
  source/destination memory layout facts.
* Validate `RouteOperandBindingPlan` closure, including plan id, logical
  operands, runtime ABI order, materialized uses, runtime ABI parameter mirrors,
  and route operand binding summary mirror.
* Preserve common EmitC/export neutrality: memory movement semantics must remain
  in RVV planning/provider/realization/target support, not common
  materialization, artifact names, route ids, helper strings, descriptors, or
  mirror fields.

## Acceptance Criteria

* [x] Active explicit and pre-realized base memory consumers are inventoried,
  with elementwise, scalar-broadcast, splat-store, segment2, and other
  non-memory routes explicitly excluded or fail-closed.
* [x] Missing base memory movement family plans fail closed for base memory
  consumers.
* [x] Stale base memory movement family plans fail closed on non-consumers.
* [x] Runtime control mismatch failures are covered.
* [x] Runtime ABI or operand-binding mismatch failures are covered.
* [x] Mask, stride, index, type, target/header, result-name, and intrinsic
  mirror mismatch failures are covered.
* [x] Selected-body realization still produces expected base memory movement
  family facts for representative explicit and pre-realized paths.
* [x] Generated-bundle dry-runs cover representative explicit and pre-realized
  unit, masked, strided, indexed-gather, and indexed-scatter routes at counts
  7, 16, and 23 where the existing harness supports them.
* [x] Real `ssh rvv` evidence covers representative explicit and pre-realized
  unit, masked, strided, indexed-gather, and indexed-scatter routes at counts
  7, 16, and 23 where the existing harness supports them.
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
* Focused lit/FileCheck or C++ coverage proves positive base memory provider
  validation and fail-closed cases.
* Generated bundle and real `ssh rvv` evidence are collected for representative
  supported routes.
* Trellis task status, context, and workspace journal are kept truthful.
* The task is finished or archived using the repo's Trellis convention.
* One coherent commit records the completed task unless a blocker remains and is
  documented.

## Out of Scope

* New memory movement operations.
* Dtype or LMUL clone expansion.
* High-level frontend, Linalg, Vector, StableHLO, source-front-door, or
  source-artifact routes.
* Global tuning, dashboards, readiness state machines, or standalone evidence
  packaging.
* Reworking runtime scalar splat-store, scalar-broadcast elementwise, plain
  compare-select, widening conversion, contraction, computed-mask select,
  computed-mask MAcc, standalone reduction, vector elementwise arithmetic,
  accumulation, segment2, or other adjacent route families except for isolation
  checks.
* Moving memory movement semantics into common EmitC/export, artifact names,
  route ids, helper strings, descriptors, or mirror metadata.
* Reintroducing legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact authority, or exact
  intrinsic spelling as provider authority.

## Technical Approach

Follow the just-closed runtime scalar splat-store and scalar-broadcast provider
validation pattern, but keep the implementation bounded to base memory movement:

1. Compare
   `verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans` with the
   repaired adjacent provider validators.
2. Inventory active base memory route forms and existing fixture/harness support.
3. Add missing runtime AVL/VL mirror checks to the base memory provider
   verifier.
4. Add full `verifyRVVRouteOperandBindingClosure` before provider
   materialization.
5. Add focused positive and negative tests for missing/stale plans, runtime
   control mismatches, ABI/binding mismatches, and route-form-specific mirror
   mismatches.
6. Re-run generated-bundle and `ssh rvv` evidence for supported representative
   explicit and pre-realized base memory fixtures.

## Decision (ADR-lite)

**Context**: Base memory movement is an active Stage2 RVV route family. Its
route-family plan exists, but provider materialization is weaker than the
adjacent repaired runtime/binding closure families.

**Decision**: Close base memory movement at the RVV plugin-local provider
boundary by requiring validated typed body/config/runtime facts,
route-family mirrors, runtime ABI mirrors, route-form-specific memory facts,
and `RouteOperandBindingPlan` closure before route construction.

**Consequences**: This strengthens existing memory movement routes without
adding route coverage or moving semantics into common EmitC/export. Existing
fixtures that relied on stale or partial mirrors must become negative tests or
be repaired to carry the validated plan.

## Technical Notes

Initial read targets from the brief:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-runtime-scalar-splat-store-runtime-binding-closure/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/*unit-load*store*.mlir`
* `test/Target/RVV/*strided*.mlir`
* `test/Target/RVV/*masked-unit-load-store*.mlir`
* `test/Target/RVV/*indexed-gather*.mlir`
* `test/Target/RVV/*indexed-scatter*.mlir`

## Completion Evidence

### Provider validation fields now checked

`verifyRVVSelectedBodyBaseMemoryMovementRouteFamilyProviderPlans` now requires
the selected base memory movement provider route description to match the
validated family plan for:

* operation and memory form;
* SEW, LMUL, tail policy, mask policy, runtime control plan id, config contract
  id, runtime VL contract id, runtime AVL source, VL def/scope/use, EmitC loop
  names, remaining AVL metadata, pointer advance metadata, bounded slice, and
  multi-VL marker;
* runtime ABI order and route-form-specific runtime ABI parameter mirrors;
* target leaf profile, provider-supported mirror, required header declaration
  mirror, C type mapping, VL C type, vector/index/mask type mirrors, and
  vector/index/mask C type mirrors;
* setvl, unit load/store, strided load/store, indexed gather/scatter, masked
  load/store intrinsic mirrors;
* result, mask role/source/memory form, inactive-lane, passthrough, strided
  layout, indexed layout, source/destination memory form, stride source,
  index EEW, offset unit, index source, index uniqueness, indexed data memory
  form, and indexed destination memory form mirrors;
* `RouteOperandBindingPlan` id, runtime ABI order, logical operand roles,
  materialized uses, runtime ABI parameter mirrors, and binding summary mirror.

### Active routes covered

Explicit selected-body routes covered by provider/unit tests, target fixtures,
generated dry-run, and real `ssh rvv` evidence:

* `strided_load_unit_store`
* `unit_load_strided_store`
* `indexed_gather_unit_store`
* `indexed_scatter_unit_load`
* `masked_unit_load_store`

Pre-realized selected-body routes covered by existing realization fixtures,
generated dry-run, and real `ssh rvv` evidence:

* `strided_load_unit_store`
* `unit_load_strided_store`
* `indexed_gather_unit_store`
* `indexed_scatter_unit_load`
* `masked_unit_load_store`
* `masked_unit_store`

The current generated-bundle harness has an explicit selected-body fixture for
`masked_unit_load_store`; standalone `masked_unit_store` is represented through
the pre-realized selected-body path. No new fixture was added for that gap in
this bounded provider-closure task.

### Adjacent route-family boundaries

Included only as isolation checks:

* elementwise `add` stays outside the base memory movement route family;
* computed-mask unit load/store stays owned by the computed-mask memory family;
* segment2 deinterleave stays owned by the plain segment2 memory family;
* scalar-broadcast, runtime scalar splat-store, elementwise arithmetic, and
  other adjacent families are not changed by this task.

Excluded from implementation by design:

* runtime scalar splat-store, scalar-broadcast elementwise, plain
  compare-select, widening conversion, contraction, computed-mask select,
  computed-mask MAcc, standalone reduction, vector elementwise arithmetic,
  accumulation, segment2, source-front-door, and descriptor paths.

### Selected-body realization evidence

Focused target fixtures already cover explicit and pre-realized selected-body
memory movement realization/materialization:

* `test/Target/RVV/explicit-selected-body-artifact-strided-load-unit-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-strided-load-unit-store.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-unit-load-strided-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-unit-load-strided-store.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-indexed-gather-unit-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-indexed-gather-unit-store.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-indexed-scatter-unit-load.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-indexed-scatter-unit-load.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-masked-unit-load-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-masked-unit-load-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-masked-unit-store.mlir`

These are exercised by `check-tianchenrv` and by the generated-bundle commands
below.

### Runtime and binding closure evidence

`test/Plugin/RVVExtensionPluginTest.cpp` now covers:

* all six active base memory consumers:
  `strided_load_unit_store`, `unit_load_strided_store`,
  `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
  `masked_unit_load_store`, and `masked_unit_store`;
* non-consumer isolation for elementwise `add`, computed-mask memory, and
  segment2 memory;
* missing base memory plan rejection;
* stale base memory plan rejection on a non-consumer;
* runtime AVL source mirror mismatch rejection;
* strided intrinsic mirror mismatch rejection;
* indexed offset and intrinsic mirror mismatch rejection;
* mask role and masked-load intrinsic mirror mismatch rejection;
* runtime ABI parameter and route operand binding mismatch rejection;
* route operand binding summary and mirror plan id mismatch rejection.

### Commands and results

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2 && ./build/bin/tianchenrv-rvv-extension-plugin-test`
  * Result: `RVV extension plugin smoke test passed`.
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  * Result: passed.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  * Result: `rvv_generated_bundle_abi_e2e self-test passed`.
* Explicit generated-bundle dry-run:
  * Command: `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2_base_memory_runtime_binding_closure --run-id explicit-dry-run --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --op-kind strided_load_unit_store --op-kind unit_load_strided_store --op-kind indexed_gather_unit_store --op-kind indexed_scatter_unit_load --op-kind masked_unit_load_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --stride-bytes 4 --stride-bytes 8 --stride-bytes 12`
  * Result: `dry_run_success`.
* Pre-realized generated-bundle dry-run:
  * Command: same as above with `--pre-realized-selected-body` and
    `--op-kind masked_unit_store`.
  * Result: `dry_run_success`.
* Explicit real `ssh rvv`:
  * Command: same as explicit dry-run without `--dry-run`, run id
    `explicit-ssh`.
  * Result: PASS for `strided_load_unit_store`,
    `unit_load_strided_store`, `indexed_gather_unit_store`,
    `indexed_scatter_unit_load`, and `masked_unit_load_store` at counts
    7, 16, and 23. Strided cases covered stride bytes 4, 8, and 12; writing
    routes reported sentinel/tail preservation where applicable.
* Pre-realized real `ssh rvv`:
  * Command: same as pre-realized dry-run without `--dry-run`, run id
    `pre-realized-ssh`.
  * Result: PASS for `strided_load_unit_store`,
    `unit_load_strided_store`, `indexed_gather_unit_store`,
    `indexed_scatter_unit_load`, `masked_unit_load_store`, and
    `masked_unit_store` at counts 7, 16, and 23. Strided cases covered stride
    bytes 4, 8, and 12; writing routes reported sentinel/tail preservation
    where applicable.
* Diff-only active-authority scan:
  * No new production `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
    `!tcrv_rvv.i32m*`, source-front-door/source-artifact, descriptor,
    direct-C, or source-export authority in added lines.
  * New exact intrinsic spelling additions are confined to C++ tests asserting
    provider-derived mirrors and stale-mirror rejection, not production route
    authority.
* `git diff --check`
  * Result: passed.
* `cmake --build build --target check-tianchenrv -j2`
  * Result: 361/361 passed.

### Self-repair

The initial explicit `masked_unit_store` C++ fixture used source-load before
mask-load, which violated the existing RVV construction-order contract. The
fixture was repaired to match the realized selected-body order:
`mask_load -> source load -> masked_store`.

`clang-format` is unavailable in this environment (`command not found`), so the
final whitespace gate is `git diff --check`.

### Spec update judgment

No `.trellis/spec/` update is needed. This task applies the already-recorded
RVV plugin-owned runtime/binding closure rule to the adjacent base memory
movement family; it does not create a new command/API signature, cross-layer
contract, route authority rule, or testing convention.
