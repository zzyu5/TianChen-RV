# Stage2 RVV plain segment2 memory route-family extraction

## Goal

Migrate the existing unmasked/plain segment2 memory route consumers onto a
plugin-local RVV segment2 memory route-family interface. The active repository
routes are:

- `segment2_deinterleave_unit_store`
- `segment2_interleave_unit_load`

These correspond to:

- `RVVSelectedBodyOperationKind::Segment2DeinterleaveUnitStore`
- `RVVSelectedBodyOperationKind::Segment2InterleaveUnitLoad`
- `RVVSelectedBodyMemoryForm::Segment2LoadUnitStore`
- `RVVSelectedBodyMemoryForm::UnitLoadSegment2Store`

This is a migration of existing plain segment2 route authority out of residual
one-off segmented field population and into typed body/config/runtime-derived
RVV plugin planning/provider logic. It is not a new segment count, computed-mask
extension, frontend lowering phase, or broad all-memory rewrite.

## Direction Source

- Direction title: `Stage2 RVV plain segment2 memory route-family extraction`.
- Module owner: RVV plugin-local plain segment2 memory route-family support for
  the existing unmasked segment2 routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e9698c23 rvv: share computed-mask segmented memory route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

- Commit `e9698c23` completed computed-mask segmented-memory migration for:
  - `computed_masked_segment2_load_unit_store`
  - `computed_masked_segment2_store_unit_load`
- The shared computed-mask memory route-family plan already carries segment2
  facts for masked segment2 load/store consumers: segment count/layout, tuple
  C type, segment load/store leaves, field roles/names, field memory forms,
  runtime ABI order, provider-supported mirrors, target leaf/header facts, C
  type mapping, inactive-lane behavior, and `RouteOperandBindingPlan` closure.
- The remaining active plain segment2 route consumers are:
  - `Segment2DeinterleaveUnitStore`: segment2 interleaved source load into
    field0/field1 vectors, then two unit-stride field stores.
  - `Segment2InterleaveUnitLoad`: two unit-stride field loads, then one
    segment2 interleaved destination store.
- Current planning/provider code still has plain segment2-specific one-off
  description/provider population and validation blocks. This task extracts the
  shared plain segment2 facts into a plugin-owned family plan and makes both
  consumers require that plan before route materialization.
- Existing pre-realized target/script tests cover the two positive routes, and
  existing explicit-context negative tests cover stale/incomplete authority
  surfaces. This task must repair the active positive fixtures so they assert
  family-derived segment2 facts, binding closure, provider mirrors, and
  fail-closed behavior rather than relying on route-id/helper-string or
  mirror-only authority.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-computed-mask-segmented-memory-route-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-indexed-memory-route-family/prd.md` only as needed for prior family-plan mechanics.
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-strided-load-route-family-interface/prd.md` only as needed for prior memory-family mechanics.
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md` only as needed for binding-closure expectations.

## Scope

1. Introduce or repair a plugin-local plain segment2 memory route-family plan,
   facet, or equivalent planning object.
2. Make `Segment2DeinterleaveUnitStore` consume the family plan for:
   - operation and memory form;
   - runtime AVL/VL plan and ABI order;
   - segment count exactly 2;
   - segment memory layout;
   - tuple C type;
   - segment2 load leaf and field extraction facts;
   - interleaved source memory form;
   - field0/field1 output roles, names, and destination memory forms;
   - target leaf profile, provider-supported mirror, required headers, C type
     mapping, and route operand binding closure.
3. Make `Segment2InterleaveUnitLoad` consume the family plan for:
   - operation and memory form;
   - runtime AVL/VL plan and ABI order;
   - segment count exactly 2;
   - segment memory layout;
   - tuple C type;
   - segment2 store leaf and tuple field facts;
   - field0/field1 input roles, names, and source memory forms;
   - interleaved destination memory form;
   - target leaf profile, provider-supported mirror, required headers, C type
     mapping, and route operand binding closure.
4. Demote or remove the old duplicated plain segmented one-off field population
   so active plain segment2 routes cannot bypass the family plan.
5. Keep common EmitC/export neutral. Common materialization may consume
   provider output only; it must not infer segment layout, field roles,
   dtype/config, runtime roles, route support, or acceptance state from route
   ids, helper strings, artifact names, manifests, mirrors, ABI names, or C
   parameter names.

## Requirements

1. The family plan must carry or validate:
   - family plan id;
   - active operation kind and memory form;
   - route direction: deinterleave load/store or interleave load/store;
   - runtime AVL/VL control and runtime ABI order;
   - segment count and layout;
   - tuple C type;
   - segment load/store/field extraction or tuple-field facts;
   - field0/field1 roles and generated value names;
   - per-field source/destination memory forms;
   - route-level source and destination memory forms;
   - required headers, target leaf profile, provider-supported mirror, C type
     mapping, and typed SEW32 LMUL m1 vector facts.
2. `segment2_deinterleave_unit_store` must require one `segment2_load`, two
   stores, runtime `n/AVL`, SEW32 LMUL m1, interleaved source memory form,
   field0/field1 output roles, and route operand binding closure.
3. `segment2_interleave_unit_load` must require two field loads, one
   `segment2_store`, runtime `n/AVL`, SEW32 LMUL m1, field0/field1 input roles,
   interleaved destination memory form, and route operand binding closure.
4. Provider materialization must require the shared plain segment2 family plan
   before emitting either active plain segment2 route.
5. Computed-mask segmented routes must remain on their already migrated
   computed-mask memory family path and must not regress.
6. No new positive legacy `RVVI32M1`, `rvv-i32m1`, finite
   `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door, route-id,
   helper-string, artifact-name, mirror-only, or common/export semantic
   authority may be introduced.

## Acceptance Criteria

- [x] A plugin-local plain segment2 memory route-family plan/facet is actively
      consumed by both `Segment2DeinterleaveUnitStore` and
      `Segment2InterleaveUnitLoad`.
- [x] The plan exposes family-derived segment layout/count, tuple type,
      segment load/store leaves, field roles, field names, per-field memory
      forms, target leaf/header facts, provider-supported mirrors, and C type
      mapping.
- [x] `segment2_deinterleave_unit_store` derives route support from typed
      body/config/runtime facts plus the family plan and `RouteOperandBindingPlan`
      closure.
- [x] `segment2_interleave_unit_load` derives route support from typed
      body/config/runtime facts plus the family plan and `RouteOperandBindingPlan`
      closure.
- [x] The old duplicated plain segmented one-off field population is removed,
      demoted behind the family plan, or restricted with a code-level reason for
      routes intentionally out of scope.
- [x] Existing computed-mask segmented, indexed, strided, unit, and runtime
      computed-mask memory routes remain supported.
- [x] Focused pre-realized FileCheck coverage proves family-derived
      segment layout/count, field roles, field memory forms, materialized
      operands, provider mirrors, headers, and binding closure for both plain
      segment2 routes. No active explicit positive plain segment2 generated
      bundle fixture exists in this repository; existing explicit-context
      negative tests remain covered.
- [x] Negative/fail-closed coverage covers missing field role, missing `n/AVL`,
      wrong memory form, bad segment count/layout, stale route id,
      helper-string fallback, mirror-only authority, and common/export semantic
      inference where current test surfaces can express them.
- [x] Generated-bundle dry-runs pass for pre-realized
      `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`,
      counts `7,16,23`. No active explicit positive plain segment2 generated
      bundle fixture exists in this repository.
- [x] Real `ssh rvv` evidence passes for the migrated plain segment2 load/store
      routes with counts `7,16,23`, checking field correctness,
      source/destination preservation, sentinel/tail preservation, and runtime
      `n` variation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Completion Evidence

- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- Focused positive lit passed for the two migrated plain segment2 target/script
  fixtures: 4/4 selected tests.
- Focused negative/dataflow lit passed for segment2 dataflow, operand binding,
  and incomplete typed-body fail-closed surfaces: 5/5 selected tests.
- Generated-bundle dry-runs passed for pre-realized
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load` with
  counts `7,16,23`.
- Real `ssh rvv` generated-bundle runs passed for both migrated routes with
  counts `7,16,23`; each run reported field-order distinguishing lanes and
  `tail_preserved`.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Active-authority diff scan found no new positive legacy route authority.
  Full touched-file scan only hit existing negative checks/guardrails.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 349/349.

## Non-Goals

- No expansion to all memory routes.
- No new segment counts beyond existing segment2.
- No computed-mask, indexed, strided, select, accumulation, reduction,
  contraction, mask algebra, dtype/LMUL clone, frontend lowering, source
  front-door positive route, dashboard, report-only, or broad metadata work.
- No routing through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor, direct-C, source-front-door, helper-string,
  route-id, artifact-name, or mirror-only authority.
- No common EmitC/export ownership of segment layout, field roles, route
  support, dtype/config, runtime roles, result roles, target leaf facts, or
  acceptance state.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck coverage for pre-realized plain segment2
   target/header fixtures plus directly relevant explicit-context negative,
   segment2 dataflow, and fail-closed fixtures.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run generated-bundle dry-runs for the active pre-realized plain segment2
   load/store routes with counts `7,16,23`.
6. Run real `ssh rvv` generated-bundle evidence for both migrated plain
   segment2 routes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Initial Implementation Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` only if realization must
  surface missing plain segment2 facts.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and
  `lib/Dialect/RVV/IR/RVVDialect.cpp` only if verifier diagnostics need a
  narrow repair.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py` only if generated-bundle metadata
  expectations require update.
- Focused tests under `test/Target/RVV`, `test/Conversion/EmitC`,
  `test/Dialect/RVV`, `test/Plugin`, and `test/Scripts` as directly relevant.

## Definition Of Done

Both active plain segment2 memory production routes consume a plugin-local
plain segment2 memory route-family interface for segment facts,
typed body/config/runtime-derived route support, provider mirrors, and
operand-binding closure. Computed-mask segmented and other memory-family routes
remain supported; common EmitC/export remains neutral; focused tests,
generated-bundle dry-runs, real `ssh rvv` evidence, authority scan,
`check-tianchenrv`, task archive, clean status, and one coherent commit are
completed.
