# Stage2 RVV closure-gated compare and select coverage expansion

## Goal

Expand one bounded Stage2 compare/select subcluster on the corrected generic
typed `tcrv_rvv` surface: add signed less-or-equal (`sle`) predicate support
for explicit/pre-realized compare-select and computed-mask vector select
routes. The new route support must remain RVV plugin-owned, typed-body-derived,
selected-body-realized, and gated by `RVVRouteOperandBindingPlan` before
EmitC/header/artifact output.

## Direction Brief

- Direction title: `Stage2 RVV closure-gated compare and select coverage expansion`.
- Module owner: RVV plugin-owned route-supported compare predicate and
  mask/select route expansion.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `db342494 rvv: expand closure-gated broadcast and masked binary routes`.
- There was no `.trellis/.current-task`; this task was created from the
  provided Direction Brief before source edits.

## Current Active Inventory

Inventory source is current code and fixtures, not status reports:

- Generic `tcrv_rvv.compare` currently accepts `eq` and `slt`.
- Explicit/pre-realized `cmp_select` currently has positive route support for
  `eq`, with `RVVRouteOperandBindingPlan` ID
  `rvv-route-operand-binding:cmp_select.v1`.
- `computed_mask_select` currently has positive route support for `slt`, with
  explicit compare lhs/rhs, true/false vector operands, output, runtime `n`,
  mask role/source/form mirrors, and binding plan ID
  `rvv-route-operand-binding:computed_mask_select.v1`.
- RVV route planning currently chooses the compare leaf from operation-level
  defaults (`cmp_select -> eq`, `computed_mask_select -> slt`) rather than a
  durable compare-predicate mirror carried in the route description.
- Existing generated-bundle harnesses prove `cmp_select` `eq` and
  `computed_mask_select` `slt`; they do not yet exercise a second signed
  predicate with equality-distinguishing data.

## Requirements

1. Add signed `sle` as a generic typed compare predicate for the bounded
   compare/select route surface, without changing dtype/LMUL scope.
2. Allow `TypedCompareSelectPreRealizedBodyOp` to realize `eq`, `slt`, and
   `sle` into explicit `tcrv_rvv.compare` plus `tcrv_rvv.select` structure.
3. Allow `TypedComputedMaskSelectPreRealizedBodyOp` to realize `slt` and `sle`
   while keeping computed-mask memory/dot routes scoped to their existing
   `slt` predicate unless separately expanded later.
4. Make RVV route planning derive compare intrinsic/header/mirrors from the
   structural `tcrv_rvv.compare {kind = ...}` fact, not from route ID,
   artifact name, ABI string, or common EmitC/export inference.
5. Preserve existing `RVVRouteOperandBindingPlan` closure for `cmp_select` and
   `computed_mask_select`: lhs/rhs/out/n and cmp_lhs/cmp_rhs/true/false/out/n
   materialized operands must still match the binding plan and header mirrors.
6. Add positive explicit/pre-realized target fixtures for `sle` compare-select
   and computed-mask select with value-distinguishing predicate cases.
7. Add negative fail-closed tests for unsupported compare predicate and ensure
   old `eq`/`slt` cases remain supported where intended.
8. Extend generated-bundle dry-run and `ssh rvv` evidence for the new `sle`
   computed-mask select and, if practical, `sle` compare-select.

## Acceptance Criteria

- [x] PRD records current compare/select route inventory from current code.
- [x] `tcrv_rvv.compare` accepts `eq`, `slt`, and signed `sle` only for this
      bounded Stage2 predicate surface.
- [x] Pre-realized `cmp_select` accepts and realizes `sle`; explicit
      `cmp_select` using `tcrv_rvv.compare {kind = "sle"}` routes through the
      existing closure-gated binding plan.
- [x] Pre-realized `computed_mask_select` accepts and realizes `sle`; explicit
      typed bodies with four loads, compare, select, and store route through the
      existing closure-gated binding plan.
- [x] Route descriptions and target metadata include an explicit compare
      predicate mirror derived from typed body facts; compare intrinsic
      selection follows that mirror.
- [x] Unsupported predicates, missing compare/select roles, mask/vector
      mismatches, true/false role swaps, stale/wrong binding summaries,
      route-id/helper/source-front-door/descriptor fallback, and common/export
      semantic inference remain fail-closed where expressible by current tests.
- [x] Generated-bundle dry-runs pass for representative `sle` compare/select
      route(s), with equality-distinguishing signed values.
- [x] Real `ssh rvv` PASS evidence exists for representative new `sle`
      route(s), proving predicate correctness, select correctness, runtime
      `n`/AVL handling, and tail/sentinel preservation where the harness checks
      it.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/
      source-export/source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, and `check-tianchenrv` pass before
      finish if the environment allows.
- [x] Task status/journal/archive and one coherent commit are completed if the
      module is finished.

## Completion Evidence

- Implemented signed `sle` compare predicate support for explicit and
  pre-realized `cmp_select` and `computed_mask_select` selected-body routes.
- Added route description `comparePredicateKind`, predicate-derived compare
  intrinsic selection, route metadata mirror, and target header mirror
  validation.
- Added four target fixtures:
  `explicit-selected-body-artifact-cmp-select-sle.mlir`,
  `pre-realized-selected-body-artifact-cmp-select-sle.mlir`,
  `explicit-selected-body-artifact-computed-mask-select-sle.mlir`, and
  `pre-realized-selected-body-artifact-computed-mask-select-sle.mlir`.
- Extended generated-bundle ABI evidence for explicit/pre-realized
  `cmp_select_sle` and `computed_mask_select_sle`.
- Real `ssh rvv` PASS evidence:
  `artifacts/tmp/rvv_stage2_compare_select_sle/ssh-explicit-cmp/20260521T153601Z`,
  `artifacts/tmp/rvv_stage2_compare_select_sle/ssh-explicit-computed/20260521T153601Z`,
  `artifacts/tmp/rvv_stage2_compare_select_sle/ssh-pre-cmp/20260521T153601Z`,
  and
  `artifacts/tmp/rvv_stage2_compare_select_sle/ssh-pre-computed/20260521T153601Z`.
- Final full check: `cmake --build build --target check-tianchenrv -j2`,
  279/279 passed.

## Non-Goals

- No reductions, contractions, conversions, segmented/indexed movement, masked
  memory movement, dtype/LMUL clone batch, high-level frontend lowering,
  source-front-door positive route, dashboards, report-only work, or helper-only
  cleanup.
- No broad predicate matrix beyond the bounded signed `sle` submodule.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor,
  direct-C, source-export, or artifact-name route authority.
- No ad-hoc provider fallback around the operand-binding closure gate.
- No common EmitC/export semantic inference for RVV predicate, dtype/config,
  mask policy, select layout, or intrinsic choices.

## Validation Plan

1. Run focused dialect/target lit checks for changed compare/select fixtures.
2. Run focused RVV plugin C++ tests covering route description, binding plan,
   and fail-closed predicate behavior.
3. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
4. Run generated-bundle dry-runs for new `sle` compare/select route(s).
5. Run real `ssh rvv` generated-bundle checks for representative `sle`
   route(s) when dry-runs pass.
6. Run active-authority scan over touched RVV/plugin/export/test/script paths.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant archived tasks read:
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-binary-broadcast-masked-elementwise/`,
  `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-computed-mask-vector-select-executable-slice/`,
  and `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-arithmetic-select-operand-binding/`.
- Primary implementation surfaces:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Dialect/RVV/`, `test/Target/RVV/`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Definition Of Done

- The production RVV provider route path supports signed `sle`
  compare/select routes through typed body facts and closure-gated operand
  binding.
- Positive and negative tests prove the new route behavior and fail-closed
  boundaries.
- Generated-bundle and `ssh rvv` evidence are recorded for representative new
  route(s).
- Trellis task is truthful, archived when complete, and committed as one
  coherent change.
