# Stage2 RVV Widening MAcc Selected-Body Realization Migration

## Goal

Move the `widening_macc_add` generated artifact path behind the public
selected lowering-boundary realization producer. The executable path must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv widening_macc_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / i16 lhs load / i16 rhs load /
     i32 accumulator load / widening_macc / i32 store body
  -> existing widening MAcc contraction family provider facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> direct contraction provider plan and statement owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for `widening_macc_add` must be
demoted or deleted. A generated artifact for this operation must not be
accepted from a direct route-entry shortcut, pre-realized fixture authority,
route id, artifact name, script option, ABI string, exact intrinsic spelling,
common EmitC behavior, descriptor residue, source-front-door metadata, or
legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening MAcc selected-body
  realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e73aa1a7 rvv: demote non widening macc route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The archived
  `05-27-05-27-stage2-rvv-non-widening-macc-selected-realization-closure`
  task demoted `macc_add` and `scalar_broadcast_macc_add` direct pre-realized
  route-entry support while keeping their selected-boundary producer paths
  positive.
- The current RVV selected-body owner registry still registers the
  `contraction` owner with `isPreRealizedRVVContractionOwnerOp` as both the
  selected-body realization predicate and the direct route-entry predicate.
  That route-entry predicate includes
  `tcrv_rvv.typed_widening_macc_pre_realized_body`.
- `scripts/rvv_generated_bundle_abi_e2e.py` still reports
  `widening_macc_add` as `supports_direct_pre_realized_route_entry`.
- The existing selected-boundary fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-macc-add.mlir`
  already represents the desired public selected-boundary producer path.
- Existing planning/provider code already exposes widening MAcc contraction
  family plans, route-control provider plans, math operand-binding facts,
  direct contraction provider plans, statement plans, target ABI mirrors, and
  generated-bundle evidence fields. This task should reuse those paths rather
  than add a new widening MAcc route table or common EmitC semantic branch.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep the selected-boundary producer path positive for `widening_macc_add`.
   It must realize the typed pre-realized body into `setvl`, `with_vl`, two
   i16 source loads, an i32 accumulator load, `widening_macc`, and i32 store
   before provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `widening_macc_add`. Direct shortcut requests must fail closed with
   targeted route-entry diagnostics before bundle generation.
4. Preserve typed facts for operation kind, i16 lhs/rhs source binding, i32
   accumulator/output binding, source/result SEW and LMUL relation, widening
   relation, accumulator/result layout, memory roles, runtime `n`/AVL/VL
   values, setvl placement, loop relation, selected capability, provider route
   facts, and artifact ABI order.
5. Do not demote or rewrite unrelated direct pre-realized route-entry support
   for bounded widening dot-reduction contraction cases unless required by the
   `widening_macc_add` demotion. Those cases are non-goals for this round.
6. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where the current hooks expose it:
   missing runtime_param, missing mem_window/runtime ABI value, wrong source
   dtype, wrong accumulator/result dtype, wrong widening relation, wrong
   accumulator binding, wrong lhs/rhs binding, wrong memory binding,
   dtype/config/policy mismatch, wrong AVL/VL relation, wrong setvl placement,
   missing capability, stale route id or mirror metadata,
   direct-route-entry-only authority, artifact-name/script-derived authority,
   exact-intrinsic-as-authority, and common-EmitC semantic invention.
7. Do not start widening dot-reduce follow-ons, strided-input widening dot,
   computed-mask widening dot, computed-mask strided-input widening dot,
   `widen_i16_to_i32`, segment2, compare/select, strided memory, standalone
   reduction, new dtype/LMUL clone batches, high-level Linalg/frontend
   lowering, one-intrinsic wrapper dialects, selected-body framework rewrites,
   dashboard/report work, broad smoke matrices, or proof-only tests for
   completed adjacent routes as the main achievement.

## Acceptance Criteria

- [ ] Production code no longer treats `widening_macc_add` as direct
      pre-realized route-entry eligible, while the `contraction` selected-body
      realization owner still realizes it through the public selected
      lowering-boundary producer path.
- [ ] C++ tests prove a typed pre-realized `widening_macc_add` body belongs to
      the `contraction` realization owner, is not a direct route-entry
      consumer, and fails closed when `realizePreRealizedRVVRouteEntrySelectedBody`
      is used as a direct route-entry shortcut.
- [ ] C++ or lit tests prove the selected-boundary `widening_macc_add` path
      consumes realized typed body facts, including i16 source loads, i32
      accumulator/result layout, widening relation, route-control provider
      plan, math operand-binding facts, direct contraction provider plan, ABI
      order, and provider-supported mirror.
- [ ] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind widening_macc_add` before
      route-entry materialization or bundle generation.
- [ ] Generated-bundle dry-run for `widening_macc_add` passes through
      `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization = false`, and records no direct route-entry
      materializer.
- [ ] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`, exact,
      tail, and stress cases with signed i16 lhs/rhs patterns, nonzero i32
      accumulator patterns, and tail preservation.
- [ ] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including non-widening MAcc, computed-mask MAcc,
      runtime-scalar MAcc, scalar_broadcast_add, runtime_i32_splat_store,
      computed_mask_select, mask/tail policy, VL/control runtime-AVL,
      conversion dtype-policy, compare/select, computed-mask memory,
      standalone reduction/accumulation, segment2, contraction, and base
      memory paths as practical focused filters or exact blocker.
- [ ] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis status, journal/archive, and final commit are truthful.

## Validation Plan

1. Validate Trellis task context.
2. Narrow the RVV selected-body realization owner registry so the
   `contraction` owner remains selected-boundary capable, but
   `widening_macc_add` is not direct route-entry eligible.
3. Update generated-bundle tooling and focused script coverage so
   `--direct-pre-realized-route-entry --op-kind widening_macc_add` fails
   closed.
4. Keep or strengthen selected-boundary lit/script coverage for
   `pre-realized-selected-body-artifact-widening-macc-add.mlir`.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-runs for `widening_macc_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   using the selected lowering-boundary producer path.
8. Run focused non-regression for selected-body producer paths listed in the
   acceptance criteria.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Widening dot-reduce selected-body migration, strided-input widening dot,
  computed-mask widening dot, computed-mask strided-input widening dot,
  `widen_i16_to_i32`, segment2 follow-ons, compare/select, strided memory,
  standalone reduction, new dtype/LMUL clone sets, high-level
  Linalg/Vector/StableHLO frontend work, source-front-door construction,
  descriptor-driven compute, direct C/source-export paths, and legacy
  `RVVI32M1` / `rvv-i32m1` compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant archived tasks read:
  `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-macc-executable-slice/`,
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-macc-route-family-owner/`,
  and
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-non-widening-macc-selected-realization-closure/`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Implementation Result

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` now splits the
  `contraction` selected-body owner predicate from direct route-entry
  eligibility. `TypedWideningMAccPreRealizedBodyOp` remains owned by the
  contraction selected-body realization producer, but is no longer accepted by
  the direct pre-realized route-entry bridge.
- `scripts/rvv_generated_bundle_abi_e2e.py` no longer marks
  `widening_macc_add` as direct pre-realized route-entry supported, and its
  user-facing direct-entry diagnostics now describe only the still-supported
  widening dot-reduction contraction fixtures.
- `test/Plugin/RVVExtensionPluginTest.cpp` now proves that
  `widening_macc_add` is selected-body-realization capable, not direct
  route-entry eligible, fails closed through
  `realizePreRealizedRVVRouteEntrySelectedBody`, and still reaches realized
  typed RVV facts, route analysis, math operand-binding facts, and direct
  contraction provider facts after selected-boundary realization.
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widening-macc-add-fail-closed.test`
  covers the script-level fail-closed path for
  `--direct-pre-realized-route-entry --op-kind widening_macc_add`.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-macc-add-dry-run.test`
  now asserts `route_entry_realization: false` and rejects the route-entry
  materializer marker in the selected-boundary generated-bundle evidence.

## Validation Result

- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Lit filter `widening-macc-add`: 5/5 passed.
- [x] Direct route-entry fail-closed probe for
      `--direct-pre-realized-route-entry --op-kind widening_macc_add`.
- [x] Selected-boundary dry-run for `widening_macc_add`, counts
      `0,1,16,23,257`, with evidence showing
      `materializer: tcrv-materialize-selected-lowering-boundaries`,
      `route_entry_realization: false`, selected-body producer ownership, and
      consumed pre-realized body.
- [x] Real `ssh rvv` generated-bundle run for `widening_macc_add`, counts
      `0,1,16,23,257`, with signed i16 products, i32 accumulation, and tail
      preservation.
- [x] Bounded touched-file authority scan: no new positive executable or route
      claim from direct-route-entry-only, pre-realized-fixture-only, route-id,
      artifact-name, script-derived, metadata-derived, descriptor-derived,
      source-front-door-derived, ABI-string-derived, common-EmitC-derived,
      exact-intrinsic-derived, central ad hoc, or legacy-i32 authority.
- [x] `git diff --check`
- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-27-05-27-stage2-rvv-widening-macc-selected-realization-migration`
- [x] `cmake --build build --target check-tianchenrv -j2`: 393/393 passed.
- [x] Spec update judgment: no `.trellis/spec/**` edit needed; existing specs
      already encode the selected-body realization, route-entry demotion,
      provider-built route, mirror metadata, neutral EmitC, and real RVV
      evidence contracts used here.

## Open Questions

None blocking. The direction brief and repository facts point to a bounded
implementation: remove `widening_macc_add` from direct pre-realized
route-entry eligibility while preserving the existing selected-boundary
realization and widening MAcc direct contraction provider path.
