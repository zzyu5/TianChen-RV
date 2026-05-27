# Stage2 RVV Non-Widening MAcc Selected-Body Realization Closure

## Goal

Close the remaining non-widening MAcc direct-route-entry authority for
`macc_add` and `scalar_broadcast_macc_add`. The generated artifact paths for
both consumers must come from the selected lowering-boundary producer path:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv macc_add / scalar_broadcast_macc_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / lhs load / rhs load or scalar
     RHS splat / accumulator load / macc / store body
  -> existing plain or scalar-broadcast MAcc route-family provider facts
  -> route materialization facts
  -> math operand-binding facts
  -> plain MAcc statement-plan boundary, with route-control provider plan for
     scalar_broadcast_macc_add
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Support for these two operation kinds must not be claimed from direct
pre-realized route-entry acceptance, pre-realized fixture names, route ids,
artifact names, script options, ABI strings, exact intrinsic spellings,
common EmitC behavior, descriptor residue, source-front-door metadata, or
legacy i32 helper authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV non-widening MAcc selected-body
  realization closure`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0d286005 rvv: demote runtime scalar macc route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The archived `05-27-stage2-rvv-macc-selected-realization` task demoted the
  vector-compare `computed_masked_macc_add` direct route-entry shortcut and
  kept its selected-boundary producer path positive.
- The archived
  `05-27-05-27-stage2-rvv-runtime-scalar-macc-selected-realization` task
  demoted the adjacent `runtime_scalar_cmp_masked_macc_add` direct
  route-entry shortcut and left a clean worktree with `check-tianchenrv`
  passing 391/391.
- `RVVSelectedBodyRealization.cpp` still registers the `MAcc` selected-body
  owner with `isPreRealizedRVVMAccRouteEntryOp` as its route-entry predicate.
  That predicate accepts `macc_add` with `vector-rhs-load` and
  `scalar_broadcast_macc_add` with `rhs-scalar-broadcast-macc`.
- `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --direct-pre-realized-route-entry` still advertises and accepts
  `macc_add/scalar_broadcast_macc_add`.
- Selected-boundary fixtures already exist for
  `test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir` and
  `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-macc-add.mlir`.
  They should remain positive through
  `--tcrv-materialize-selected-lowering-boundaries`.
- Existing planning/provider code already exposes plain MAcc and
  scalar-broadcast MAcc family plans, route-control provider plans, math
  operand-binding facts, and the
  `RVVSelectedBodyPlainMAccRouteStatementPlan` boundary. This round should
  reuse those production paths rather than introduce a new MAcc route table.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep selected-boundary producer paths positive for both `macc_add` and
   `scalar_broadcast_macc_add`. They must realize typed pre-realized bodies
   into `setvl`, `with_vl`, loads, RHS scalar splat where needed, accumulator
   load, `macc`, and store before provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for both
   consumers. Direct shortcut requests must fail closed with targeted
   diagnostics before route-entry materialization or bundle generation.
4. Preserve typed facts for operation kind, accumulator layout, scalar RHS
   binding, vector lhs/rhs binding, memory roles, runtime `n`/AVL/VL values,
   dtype/config/policy, setvl placement, loop relation, selected capability,
   provider route facts, and artifact ABI order.
5. Preserve the already migrated selected-boundary paths and fail-closed
   direct route-entry behavior for `computed_masked_macc_add` and
   `runtime_scalar_cmp_masked_macc_add`.
6. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where the current hooks expose it:
   missing runtime param, missing mem_window/runtime ABI value, wrong scalar
   RHS binding, wrong accumulator binding, wrong lhs/rhs binding, dtype/config/
   policy mismatch, wrong AVL/VL relation, wrong setvl placement, missing
   capability, stale route id or mirror metadata, direct-route-entry-only
   authority, artifact-name/script-derived authority, exact-intrinsic-as-
   authority, and common-EmitC semantic invention.
7. Do not start widening MAcc, widening dot/reduction, segment2,
   compare/select, strided memory, conversion, new dtype/LMUL clone sets,
   high-level Linalg/frontend lowering, selected-body framework rewrites,
   dashboard/report work, broad smoke matrices, or proof-only tests for
   already migrated routes as the main achievement.

## Acceptance Criteria

- [x] Production code demotes `macc_add` and `scalar_broadcast_macc_add`
      direct pre-realized route-entry eligibility while keeping the
      selected-boundary producer paths positive.
- [x] C++ tests prove the `MAcc` selected-body owner remains a realization
      owner but no longer exposes a direct route-entry consumer for
      `macc_add` or `scalar_broadcast_macc_add`.
- [x] C++ or lit tests prove selected-boundary provider route facts for both
      operations consume realized typed body facts, including plain/scalar-
      broadcast MAcc family plan, route-control plan where applicable, math
      operand-binding facts, statement plan, ABI order, accumulator layout,
      scalar RHS or vector RHS binding, and provider-supported mirror.
- [x] The script direct pre-realized route-entry path for both operations fails
      closed or is removed from positive lit coverage.
- [x] Generated-bundle dry-runs for both operations pass through the selected
      lowering-boundary path and record no direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`, exact
      VL, tail, and stress cases for both operations, with at least two
      accumulator/input patterns and at least two RHS scalar patterns for
      `scalar_broadcast_macc_add`.
- [x] Focused non-regression covers completed selected-body realization paths
      changed or risked by this demotion, including
      `computed_masked_macc_add`, `runtime_scalar_cmp_masked_macc_add`,
      `scalar_broadcast_add`, `runtime_i32_splat_store`,
      `computed_mask_select`, mask/tail policy, VL/control runtime-AVL,
      conversion dtype-policy, compare/select, computed-mask memory,
      standalone reduction/accumulation, segment2, contraction, and base
      memory paths.
- [x] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis status, journal/archive, and final commit are truthful.

## Validation Plan

1. Validate Trellis task context.
2. Apply the RVV selected-body realization owner change so the `MAcc` family
   still realizes through the owner registry but has no active direct
   route-entry consumer.
3. Update generated-bundle tooling and focused lit/script tests so
   `--direct-pre-realized-route-entry --op-kind macc_add` and
   `--op-kind scalar_broadcast_macc_add` no longer produce artifacts.
4. Keep or strengthen selected-boundary lit coverage for both pre-realized
   selected-body MAcc artifacts.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-runs for `macc_add` and
   `scalar_broadcast_macc_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and scalar RHS values such as `-37,91` for scalar-broadcast MAcc.
8. Run focused non-regression for the selected-body producer paths listed in
   the acceptance criteria.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Stage2 coverage expansion beyond the bounded non-widening MAcc
  selected-body realization closure for `macc_add` and
  `scalar_broadcast_macc_add`.
- Widening MAcc, widening dot/reduction, segment2 follow-on work, conversion
  clone batches, dtype/LMUL expansion, high-level Linalg/Vector/StableHLO
  frontend work, source-front-door construction, descriptor-driven compute,
  direct C/source-export paths, and legacy `RVVI32M1` / `rvv-i32m1`
  compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant archived tasks read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-macc-selected-realization/`,
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-runtime-scalar-macc-selected-realization/`,
  and
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-macc-route-family-owner/`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Open Questions

None blocking. The direction brief and repository facts point to a bounded
implementation: remove the remaining direct route-entry eligibility and
script allowlist for plain/scalar-broadcast MAcc while preserving the existing
selected-boundary realization and provider path.

## Implementation Result

- `RVVSelectedBodyRealization.cpp` removed the MAcc direct route-entry
  predicate and registers the `MAcc` owner with `isRouteEntryConsumer =
  nullptr`. The owner-local selected-body realization hook remains active.
- `rvv_generated_bundle_abi_e2e.py` no longer treats `macc_add` or
  `scalar_broadcast_macc_add` as direct pre-realized route-entry-supported
  operations. Its help/error text now keeps those consumers on the selected
  lowering-boundary producer path.
- `RVVExtensionPluginTest.cpp` now proves that plain and scalar-broadcast MAcc
  provider plan IDs are selected-boundary producer cases, realized bodies
  contain the expected load/splat/macc/store structure, the `MAcc` owner has
  no direct route-entry predicate, and direct route-entry requests for both
  operation kinds fail closed with route-entry family diagnostics.
- The old positive
  `rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-macc-add-dry-run.test`
  was deleted. New fail-closed lit tests cover both direct `macc_add` and
  direct `scalar_broadcast_macc_add` requests.
- The direct route-entry materialization lit coverage no longer contains a
  positive scalar-broadcast MAcc block.

## Validation Result

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: pass.
- `git diff --check`: pass.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`: pass.
- `./build/bin/tianchenrv-rvv-extension-plugin-test`: pass.
- Direct-route fail-closed probes:
  `--direct-pre-realized-route-entry --op-kind macc_add` and
  `--direct-pre-realized-route-entry --op-kind scalar_broadcast_macc_add`
  both failed as expected before route-entry materialization, with bounded
  direct-route allowlist diagnostics.
- Selected-boundary dry-runs:
  `macc_add` and `scalar_broadcast_macc_add` passed for counts
  `0,1,16,23,257`; scalar-broadcast also covered RHS scalars `-37,91`.
  Evidence records `materializer = tcrv-materialize-selected-lowering-boundaries`
  and `route_entry_realization = false`.
- Real `ssh rvv` runs:
  `macc_add` passed for counts `0,1,16,23,257`; `scalar_broadcast_macc_add`
  passed for counts `0,1,16,23,257` and RHS scalars `-37,91`. Both reported
  explicit accumulator signed-products correctness with tail preservation.
- Authority scan:
  no active MAcc route-entry predicate remains in the production owner
  registry; remaining direct route-entry mentions for MAcc are fail-closed
  tests or negative assertions; selected-boundary evidence uses
  provider-derived typed `tcrv_rvv` multiply-accumulate facts and mirror-only
  artifact metadata.
- `check-tianchenrv`:
  the first post-fix run after deleting the stale scalar-broadcast MAcc direct
  lit block exposed that original stale positive path, which was removed. A
  later accidental double launch caused unrelated shared `build/*/Output`
  artifact collisions. After both duplicate lit processes exited, a single
  clean rerun passed `392/392`.
