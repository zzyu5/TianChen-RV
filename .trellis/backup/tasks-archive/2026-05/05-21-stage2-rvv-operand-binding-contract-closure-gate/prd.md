# Stage2 RVV operand binding contract closure gate

## Goal

Close the RVV plugin-owned `RVVRouteOperandBindingPlan` contract across the
current active route-supported/executable RVV surface. This task is a closure
gate, not new Stage2 coverage: active RVV route construction, provider
materialization, emission-plan mirrors, target artifact/header metadata, and
generated-bundle evidence must all be derived from a matching operand binding
plan or fail closed.

## Direction Brief

- Direction title: `Stage2 RVV operand binding contract closure gate`.
- Module owner: RVV plugin-owned closure invariant for
  `RVVRouteOperandBindingPlan` across current active route-supported/executable
  RVV routes.
- Initial repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `79a1ab82 rvv: adopt conversion operand binding plans`.
- There was no `.trellis/.current-task`; this task was created from the Hermes
  direction brief.

## Current Active Inventory

Inventory source is the current RVV route planning/provider code plus
`scripts/rvv_generated_bundle_abi_e2e.py` generated-bundle support. The active
surface for this closure task is:

- Ordinary arithmetic: `add`, `sub`, `mul`; `i64_add` and `lmul_m2_add` are
  typed/config variants of the same generic `Add` operation kind, not separate
  operation families.
- Compare/select: `cmp_select`, `computed_mask_select`.
- Reduction/accumulation: `reduce_add`, `standalone_reduce_add`, `macc_add`.
- Arithmetic memory forms: `strided_add`, scalar `scalar_broadcast_add`, RHS
  broadcast add/sub/mul fixtures.
- Movement: `strided_load_unit_store`, `unit_load_strided_store`,
  `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
  `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`.
- Masked movement: `masked_unit_load_store`, `masked_unit_store`,
  `computed_masked_unit_load_store`, `computed_masked_strided_store`.
- Conversion and contraction: `widen_i32_to_i64`, `widen_i16_to_i32`,
  `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`,
  `computed_masked_strided_input_widening_dot_reduce_add`.

Initial code inspection found that most converted families already carry
binding plan IDs and summaries, but generated-bundle-supported
`masked_unit_load_store` and `computed_masked_unit_load_store` still lack a
plan mirror. The provider also initializes bound operands from direct slice ABI
fields before optional plan lookup, so an active route without a plan can still
materialize through direct slice state. This task must remove that bypass for
the active surface.

## Requirements

1. Establish a single RVV plugin-owned closure invariant for active route kinds:
   every active route description must have the expected
   `RVVRouteOperandBindingPlan` ID, logical operands, runtime ABI role/C
   parameter mapping, materialized-use mapping, and non-empty mirror summary.
2. Enforce the invariant before `TCRVEmitCLowerableRoute` construction can add
   headers, ABI mappings, call operands, or target metadata.
3. Add missing binding plans for active masked movement routes that are already
   route-supported/executable, especially `masked_unit_load_store` and
   `computed_masked_unit_load_store`.
4. Keep typed/config variants such as `i64_add`, `lmul_m2_add`, and RHS
   broadcast add/sub/mul under the generic operation plan where the provider
   route semantics are the same generic operation kind.
5. Tighten target artifact mirror validation so candidate metadata mirrors the
   provider-built route operand binding plan/summary when the description has
   one; metadata must not authorize a route by itself.
6. Update generated-bundle expectations so representative active routes prove
   route operand binding plan and operand summary mirrors are present.
7. Do not promote parser-only, inactive, source-front-door, descriptor,
   direct-C, or legacy i32m1 surfaces.

## Acceptance Criteria

- [ ] Current active route inventory is documented in this PRD and bounded to
      existing route planning/provider/generated-bundle support.
- [ ] `RVVRouteOperandBindingPlan` has explicit expected plan IDs/roles for all
      active route kinds in scope.
- [ ] Provider route construction fails closed when an active route has a
      missing plan ID, wrong plan ID, missing logical operand, missing
      materialized-use token, runtime ABI order mismatch, runtime ABI mirror
      mismatch, or stale plan summary.
- [ ] `masked_unit_load_store` and `computed_masked_unit_load_store` no longer
      materialize operands or metadata outside the binding plan.
- [ ] Target artifact validation rejects stale/missing route operand binding
      plan or operand summary mirrors for provider-built active RVV routes.
- [ ] Positive structural tests cover representative active families that must
      pass through the contract: ordinary/config variant, masked movement,
      conversion, contraction, and movement.
- [ ] Negative fail-closed tests cover missing/wrong plan ID, missing logical
      operand or materialized use, stale mirror/header metadata, and direct
      slice fallback for active routes.
- [ ] Focused generated-bundle dry-runs cover representative affected families.
- [ ] Real `ssh rvv` is run for a small representative set if provider
      materialization changes can affect runtime operands.
- [ ] Active-authority scan confirms no new positive `RVVI32M1`, `rvv-i32m1`,
      finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export/source-front-door, public exact
      intrinsic, artifact-name authority, or common/export RVV semantic
      authority.
- [ ] `git diff --check` and `check-tianchenrv` pass.
- [ ] Task status/journal/archive and one coherent commit are completed if the
      closure submodule is finished.

## Non-Goals

- No new Stage2 operation family, dtype, LMUL, conversion variant, reduction
  kind, mask form, frontend lowering, Linalg/Vector frontend, or source
  front-door positive route.
- No helper-only/report-only/dashboard work as the main result.
- No common EmitC/export RVV semantic inference.
- No compatibility wrapper around legacy route authority.
- No rewrite of inactive routes without a production generated-bundle or
  provider consumer.

## Validation Plan

1. Build focused compiler/test binaries if needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
2. Run the focused C++ plugin test:
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck for touched RVV target/EmitC tests and generated
   script dry-run tests.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
5. Run generated-bundle dry-runs for representative closure routes at small
   counts such as `7,16,23`.
6. Run real `ssh rvv` for representative affected routes if runtime operand
   materialization changes are made.
7. Run active-authority scans, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant archived tasks read: route operand ABI binding, operand binding
  coverage closure, arithmetic/select, indexed and segmented movement,
  widening, contraction dot-reduction, and conversion operand-binding tasks.
- Relevant source surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Definition Of Done

- Production route planning/provider and target validation enforce the closure
  invariant for active RVV routes.
- Focused tests prove both positive plan-driven acceptance and negative
  fail-closed bypass rejection.
- Generated-bundle and, if needed, `ssh rvv` evidence are recorded.
- Trellis task is truthful, archived when complete, and committed as one
  coherent change.
