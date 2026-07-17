# Stage2 RVV widen_i16_to_i32 Selected-Body Realization Migration

## Goal

Move `widen_i16_to_i32` generated artifact execution behind the RVV
plugin-local selected-body realization producer and remove its active direct
pre-realized route-entry authority.

The production path for this operation must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv widening conversion body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving i16 source, i32 result,
     sign-extension widening conversion kind, source/result SEW and LMUL
     relation, memory roles, runtime n/AVL/VL, setvl placement, policy,
     required capabilities, and artifact ABI order
  -> widening conversion route-family facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> widening conversion statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for `widen_i16_to_i32` must be demoted
or deleted. A generated artifact for this operation must not be accepted from a
direct route-entry shortcut, pre-realized fixture authority, route id, artifact
name, script option, ABI string, exact intrinsic spelling, common EmitC
behavior, descriptor residue, source-front-door metadata, or legacy i32 helper
authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV widen_i16_to_i32 conversion
  selected-body realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `2a501a8b rvv: demote computed mask strided dot route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` currently keeps
  `TypedWideningConversionPreRealizedBodyOp` in the `widening conversion`
  selected-body owner and also marks bounded widening conversion bodies as
  direct route-entry eligible through
  `isPreRealizedRVVWideningConversionRouteEntryOp`.
- That route-entry predicate accepts the `sign_extend_widen_vf2` body used by
  `widen_i16_to_i32`, with source SEW16 LMUL mf2, destination SEW32 LMUL m1,
  and relation `signed-i16mf2-to-i32m1`.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently includes
  `widen_i16_to_i32` in `supports_direct_pre_realized_route_entry`, and its
  direct-mode allowlist diagnostic names `widen_i16_to_i32`.
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widen-i16-to-i32-dry-run.test`
  is a positive direct pre-realized route-entry fixture and currently checks
  `route_entry_realization: true`.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widen-i16-to-i32-dry-run.test`
  already exercises the selected lowering-boundary producer path, but it needs
  to become the positive generated-bundle authority for this op and explicitly
  prove `route_entry_realization: false`.
- `test/Plugin/RVVExtensionPluginTest.cpp` currently treats
  `rvv_pre_route_widen_i16_to_i32` as route-entry eligible in the production
  route-path exercise. It also has provider-plan, route-control,
  statement-plan, math binding, and target artifact coverage that should remain
  positive for the realized typed widening conversion path.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-computed-mask-strided-widening-dot-reduce-selected-realization`
  completed the same migration pattern for
  `computed_masked_strided_input_widening_dot_reduce_add`: direct route-entry
  fails closed, selected-boundary dry-run and `ssh rvv` correctness passed,
  `check-tianchenrv` passed 397/397, and the task was archived.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, probes,
   harnesses, and script guardrails.
2. Keep `widen_i16_to_i32` in the RVV `widening conversion` selected-body
   realization owner. The selected-boundary producer must realize the typed
   pre-realized body before provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `widen_i16_to_i32`. Direct shortcut requests must fail closed before bundle
   generation.
4. Preserve typed facts for operation kind `sign_extend_widen_vf2`, i16 source
   binding, i32 result binding, signed widening relation
   `signed-i16mf2-to-i32m1`, source SEW16 LMUL mf2, destination SEW32 LMUL m1,
   unit-stride conversion memory form, runtime `n`/AVL/VL values, setvl
   placement, tail/mask policy, required capabilities, provider route facts,
   statement-plan facts, and artifact ABI order `lhs,out,n`.
5. Fail closed for unsupported or inconsistent selected-boundary input,
   including missing runtime_param, missing mem_window/runtime ABI value, wrong
   source dtype, wrong result dtype, wrong widening relation, wrong memory
   role/form, dtype/config/policy mismatch, wrong AVL/VL relation, wrong setvl
   placement, missing capability, stale route id or mirror metadata,
   direct-route-entry-only authority, artifact-name/script-derived authority,
   exact-intrinsic-as-authority, and common-EmitC semantic invention.
6. Reuse existing selected-body realization, widening conversion route-family,
   route-control provider, math operand-binding, statement-plan, target
   artifact, and generated-bundle boundaries where they already express the
   required facts. Do not introduce a new central route table or common EmitC
   semantic branch.
7. Do not start segment2, computed_masked_segment2, compare/select, strided
   load/store, standalone reduction cleanup, new dtype/LMUL clone batches,
   high-level Linalg/frontend lowering, one-intrinsic wrapper dialects,
   selected-body realization framework rewrites, dashboard/report work, broad
   smoke matrices, or evidence-only tasks.
8. Do not add proof-only tests for completed widening dot or MAcc
   selected-body paths except as bounded non-regression for touched files.

## Acceptance Criteria

- [x] Production code no longer treats `widen_i16_to_i32` as direct
      pre-realized route-entry eligible, while the `widening conversion`
      selected-body realization owner still realizes it through the public
      selected lowering-boundary producer path.
- [x] C++ tests prove a typed pre-realized `widen_i16_to_i32` body belongs to
      the `widening conversion` realization owner, is not a direct route-entry
      consumer, and fails closed when
      `realizePreRealizedRVVRouteEntrySelectedBody` is used as a direct
      route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary path consumes realized typed
      facts, including i16 source load, i32 widening conversion, i32 store,
      source/result SEW and LMUL relation, conversion relation, route-control
      provider plan, math operand-binding facts, widening conversion statement
      plan, ABI order, and provider-supported mirrors.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind widen_i16_to_i32` before
      route-entry materialization or bundle generation.
- [x] Generated-bundle dry-run for `widen_i16_to_i32` passes through
      `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization: false`, records selected-body producer
      evidence, records realized dtype/conversion/provider facts, and records
      no direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`, exact,
      tail, and stress cases with at least two signed i16 input patterns and
      expected i32 sign-extended results.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum the RVV extension plugin
      smoke test and focused lit for selected-boundary `widen_i16_to_i32`.
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
2. Inspect current selected-body realization, route planning/provider,
   target-support bundle, generated-bundle script, focused plugin tests, and
   target/script tests for the exact active direct route-entry authority.
3. Narrow RVV selected-body route-entry eligibility so the `widening conversion`
   owner remains selected-boundary capable, but `widen_i16_to_i32` is not
   direct route-entry eligible.
4. Update generated-bundle tooling and focused script coverage so direct
   pre-realized route-entry mode fails closed for `widen_i16_to_i32`.
5. Strengthen selected-boundary lit/script/C++ coverage so evidence explicitly
   checks `route_entry_realization: false`, i16 source/i32 result dtype
   relation, provider facts, and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-run for `widen_i16_to_i32`.
8. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and at least two signed i16 input patterns through the selected
   lowering-boundary producer path.
9. Run focused adjacent non-regression.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Completion Result

- Demoted `widen_i16_to_i32` from active direct pre-realized route-entry
  eligibility in the RVV selected-body route-entry predicate while keeping it
  under the `widening conversion` selected-body realization owner.
- Removed `widen_i16_to_i32` from generated-bundle direct route-entry support,
  deleted the positive direct shortcut lit fixture, and added a fail-closed
  direct shortcut lit fixture.
- Strengthened the selected-boundary generated-bundle path to record
  `route_entry_realization: false`,
  `rvv-plugin-local-selected-body-realization-owner-registry`, typed
  `sign_extend_widen_vf2` conversion facts, i16 source and i32 result policy,
  SEW16/LMUL mf2 to SEW32/LMUL m1 relation, provider route facts, artifact ABI
  order `lhs,out,n`, and neutral EmitC materialization.
- Expanded the `widen_i16_to_i32` harness to execute two signed i16 input
  patterns across counts `0,1,16,23,257` and check expected i32 sign-extension
  plus output tail sentinel preservation.
- Real `ssh rvv` evidence passed for the selected-boundary generated artifact:
  each count ran `pattern=0` and `pattern=1`, ending with
  `PASS op=widen_i16_to_i32 counts=0,1,16,23,257`.

## Validation Result

- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] selected-boundary generated-bundle dry-run for `widen_i16_to_i32`
- [x] direct route-entry generated-bundle request fails closed for
      `widen_i16_to_i32`
- [x] focused lit for selected-boundary and direct-fail `widen_i16_to_i32`
      generated-bundle tests: 2/2
- [x] real `ssh rvv` generated-bundle execution for counts `0,1,16,23,257`
      and two signed i16 input patterns
- [x] bounded touched-file authority scan
- [x] `git diff --check`
- [x] Trellis context validation
- [x] `cmake --build build --target check-tianchenrv -j2`: 397/397 passed

## Out Of Scope

- Segment2 follow-ons, computed-mask segment2, compare/select, strided
  load/store, standalone reduction cleanup, new dtype/LMUL clone sets,
  high-level Linalg/Vector/StableHLO frontend work, source-front-door
  construction, descriptor-driven compute, direct C/source-export paths,
  one-intrinsic wrapper dialects, selected-body framework rewrite, dashboards,
  prompt-only work, and legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Additional proof-only work for completed widening dot, widening MAcc,
  computed-mask widening dot, strided widening dot, or computed-mask strided
  widening dot paths except as bounded non-regression.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-computed-mask-strided-widening-dot-reduce-selected-realization/`.
- Relevant workspace journal entries read:
  `.trellis/workspace/codex/journal-16.md`, Sessions 238, 253, 257, 263,
  264, and 265.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related generated
  bundle / target tests.

## Open Questions

- None blocking at task start. The user brief, specs, previous archived task,
  journal, and current repository state agree on the module owner and non-goals.
