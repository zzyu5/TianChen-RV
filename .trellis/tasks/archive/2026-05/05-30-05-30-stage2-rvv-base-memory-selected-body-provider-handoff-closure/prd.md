# Stage2 RVV Base-Memory Selected-Body Provider Handoff Closure

## Goal

Close the selected-body/provider handoff for the RVV base-memory movement
family. Pre-realized validation authority for strided load/store, indexed
gather/scatter, and static-mask load/store selected bodies must live in
`RVVEmitCBaseMemoryRouteFamilyPlanOwners` or an owner-local companion under
that route-family boundary. `RVVSelectedBodyRealization` must remain the
selected-body owner dispatcher, realization-plan construction site, and realized
IR materializer for `setvl` / `with_vl` / load / store structure only.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is clean on `main` at `a3732415`
  (`rvv: close contraction selected-body handoff`).
- There was no active `.trellis/.current-task`; this task was created from the
  supplied Hermes Direction Brief.
- The previous completed contraction handoff moved analogous widening
  contraction pre-realized selected-body validation into
  `RVVEmitCContractionRouteFamilyPlanOwners` and left a nearby implementation
  pattern.
- Current bounded inspection shows `RVVSelectedBodyRealization.cpp` still owns
  base-memory pre-realized validation helpers and constants for
  `typed_strided_memory_pre_realized_body`,
  `typed_strided_store_memory_pre_realized_body`,
  `typed_indexed_gather_memory_pre_realized_body`,
  `typed_indexed_scatter_memory_pre_realized_body`, and
  `typed_masked_memory_pre_realized_body`.
- `RVVEmitCBaseMemoryRouteFamilyPlanOwners` already owns base-memory
  route-family/provider facts, provider mirrors, operand-binding plan ids,
  runtime ABI order, target leaf profiles, type/header/intrinsic mirrors, and
  provider-plan validation.
- The stable authority chain remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> neutral EmitC materialization -> target artifact`.

## Requirements

1. Move or factor base-memory pre-realized selected-body validation for strided
   load/store, indexed gather/scatter, and static-mask load/store into the
   base-memory owner-local boundary.
2. Keep `RVVSelectedBodyRealization.cpp` limited to selected-body owner
   dispatch, construction of realized body ops, and actual materialization of
   `setvl`, `with_vl`, loads, stores, mask loads, masked loads/stores, and
   moves.
3. Preserve validation of operation kind, memory form, runtime ABI role/order,
   mem_window/runtime_param imported values, stride/index/mask facts, source
   and result element type/config, VL/AVL/setvl relation, tail/mask policy,
   inactive-lane and passthrough policy, target leaf facts, provider-plan
   mirrors, operand-binding facts, and artifact ABI facts.
4. Unsupported or inconsistent facts must fail closed with targeted diagnostics
   before provider facts, statement plans, common EmitC, or target artifact
   export.
5. Keep current executable behavior and emitted route outputs stable unless a
   concrete bug is found. This is an ownership handoff, not new RVV coverage.
6. Preserve computed-mask memory, segment2 memory, MAcc, contraction, widening
   conversion, reductions, elementwise/select, high-level frontend, source
   front-door, direct-route-entry, and runtime/performance scope as non-goals.
7. Do not add descriptor-derived, name-derived, route-id-derived,
   ABI-string-derived, artifact-name-derived, exact-intrinsic-derived,
   common-EmitC-derived, direct-route-entry-only, source-front-door-derived, or
   legacy-i32 authority.

## Acceptance Criteria

- [x] Base-memory pre-realized selected-body validation declarations and
      definitions live in `RVVEmitCBaseMemoryRouteFamilyPlanOwners` or an
      owner-local companion, not as central base-memory legality in
      `RVVSelectedBodyRealization.cpp`.
- [x] `RVVSelectedBodyRealization.cpp` dispatches base-memory validation
      through owner-local APIs and retains only family owner dispatch plus
      realized IR materialization.
- [x] Strided load/store, indexed gather/scatter, and static-mask load/store
      selected bodies keep exact positive behavior and fail closed on wrong
      op kind, memory form, stride/index/mask binding, dtype/config/policy,
      runtime ABI role/order, VL/AVL/setvl relation, missing selected variant
      requires metadata, and mixed pre-realized/realized body state.
- [x] Base-memory route-family plans, provider plans, route descriptions, and
      mirrors still consume owner-derived base-memory facts and reject stale
      plan/mirror metadata.
- [x] Focused C++ tests or existing focused coverage prove owner-local positive
      validation for representative strided, indexed, and static-mask
      base-memory bodies.
- [x] Focused fail-closed coverage proves wrong memory form, missing
      mem_window/runtime_param or runtime ABI role, wrong stride/index/mask
      binding, wrong dtype/config/policy, stale provider mirror, stale operand
      binding, and non-owner route claims.
- [x] Generated-bundle dry-runs pass for representative affected base-memory
      selected-boundary variants, or exact blockers are recorded.
- [x] Contraction selected-body/provider handoff focused coverage remains green
      as non-regression.
- [x] Bounded authority scans show central selected-body no longer owns
      base-memory legality constants or validation authority and touched
      planning/provider/test files have no legacy i32/source-front-door/
      descriptor/direct-C/source-export or mirror-only authority drift.
- [x] `git diff --check`, focused RVV plugin build/test, and `check-tianchenrv`
      pass, or exact blockers are recorded.
- [x] Task is finished/archived and one coherent commit is created with clean
      final source status.

## Completion Evidence

- Production handoff moved the five base-memory pre-realized selected-body
  validation APIs into `RVVEmitCBaseMemoryRouteFamilyPlanOwners`:
  strided load/unit store, unit load/strided store, indexed gather/unit store,
  indexed scatter/unit load, and static-mask unit load/store.
- `RVVSelectedBodyRealization.cpp` now calls the owner-local APIs and no longer
  defines the base-memory validation helpers or base-memory legality constants;
  it keeps selected-body dispatch and realized IR materialization.
- Focused C++ coverage in `tianchenrv-rvv-extension-plugin-test` proves the
  selected-boundary production route can call owner-local base-memory
  validation directly.
- Generated-bundle dry-run passed for:
  `strided_load_unit_store`, `unit_load_strided_store`,
  `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
  `masked_unit_load_store`, and `masked_unit_store` under
  `artifacts/tmp/stage2_rvv_base_memory_handoff/codex-base-memory-handoff-all`.
- Direct route-entry negative evidence failed closed with the expected
  retired-shortcut diagnostic under
  `artifacts/tmp/stage2_rvv_base_memory_handoff/codex-base-memory-handoff-direct-route-entry-negative`.
- Contraction non-regression generated-bundle dry-run passed for
  `computed_masked_strided_input_widening_dot_reduce_add` under
  `artifacts/tmp/stage2_rvv_base_memory_handoff/codex-contraction-handoff-nonregression`.
- Authority scans found no base-memory validation definitions or forbidden
  legacy/source-front-door/descriptor/direct-route authority in central
  `RVVSelectedBodyRealization.cpp`. Exact intrinsic strings remain only in
  owner/provider/test-derived facts and mirrors.
- Checks passed:
  `git diff --check`;
  `ninja -C build tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `ninja -C build check-tianchenrv` with 464/464 tests passed.
- No `ssh rvv` run was required because this round made an ownership handoff
  with stable emitted behavior and no new runtime correctness/performance
  claim.

## Out Of Scope

- No computed-mask memory, segment2 memory, compare/select, elementwise
  arithmetic, MAcc, contraction, widening conversion, standalone reduction,
  high-level frontend lowering, dtype/LMUL clone batches, one-intrinsic wrapper
  dialects, report/dashboard work, broad smoke matrices, or evidence-only
  tasks.
- No direct-route-entry shortcut, route-id authority, artifact-name authority,
  descriptor-derived semantics, exact-intrinsic authority, source-front-door
  authority, or common-EmitC semantic choices.
- No RVV runtime correctness or performance claim unless this round changes
  executable artifact/runtime behavior and `ssh rvv` evidence is collected.

## Technical Notes

- Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Nearest archived pattern:
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-contraction-selected-body-provider-handoff-closure/prd.md`.
- Production files to inspect/modify first:
  `include/TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  and `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Provider/planning consumption should stay in the existing base-memory route
  family owner, memory operand-binding boundary, route-control provider plan,
  migrated statement-plan owner, and neutral `RVVEmitCRouteProvider`
  consumption path.

## Validation Plan

1. Validate Trellis context and start this task.
2. Implement the bounded base-memory owner-local validation handoff.
3. Update focused C++ tests only if current coverage does not prove the new
   owner-local API boundary.
4. Run focused build/test target for `tianchenrv-rvv-extension-plugin-test`.
5. Run representative generated-bundle dry-runs for the migrated base-memory
   selected-boundary paths.
6. Run contraction non-regression focused evidence.
7. Run bounded authority scans and `git diff --check`.
8. Run `ninja -C build check-tianchenrv` or record the exact blocker.
9. Finish/archive the task and create one coherent commit.
