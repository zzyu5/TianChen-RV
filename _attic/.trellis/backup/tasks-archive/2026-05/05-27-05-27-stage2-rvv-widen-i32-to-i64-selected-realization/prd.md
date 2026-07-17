# Stage2 RVV widen_i32_to_i64 Selected-Body Realization Migration

## Goal

Move `widen_i32_to_i64` generated artifact execution behind the RVV
plugin-local selected-body realization producer and remove its active direct
pre-realized route-entry authority.

The production path for this operation must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv widening conversion body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving i32 source, i64 result,
     signed widening conversion kind, source/result SEW and LMUL relation,
     memory roles, runtime n/AVL/VL, setvl placement, policy,
     required capabilities, and artifact ABI order
  -> widening conversion route-family facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> widening conversion migrated statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for `widen_i32_to_i64` must be demoted
or deleted. A generated artifact for this operation must not be accepted from a
direct route-entry shortcut, pre-realized fixture authority, route id, artifact
name, script option, ABI string, exact intrinsic spelling, common EmitC
behavior, descriptor residue, source-front-door metadata, or legacy i32 helper
authority.

## Direction Source

- Direction title: `Continue: Stage2 RVV widen_i32_to_i64 selected-body
  realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `bf686735 rvv: demote widen i16 route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` keeps
  `TypedWideningConversionPreRealizedBodyOp` in the `widening conversion`
  selected-body owner and still marks `widen_i32_to_i64` as direct route-entry
  eligible through `isPreRealizedRVVWideningConversionRouteEntryOp`.
- That direct predicate currently accepts source SEW32 LMUL m1, destination
  SEW64 LMUL m2, relation `signed-i32m1-to-i64m2`, unit-stride conversion, and
  agnostic policy.
- The same owner no longer accepts `widen_i16_to_i32` direct route-entry; the
  archived predecessor task demoted that operation while keeping selected-body
  realization positive.
- `scripts/rvv_generated_bundle_abi_e2e.py` no longer includes
  `widen_i32_to_i64` in `supports_direct_pre_realized_route_entry`, so direct
  script mode already fails closed for this op but lacks a focused lit fixture.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widen-i32-to-i64-dry-run.test`
  already exercises the selected lowering-boundary producer path and checks
  `route_entry_realization: false`, selected-body producer evidence, provider
  mirrors, ABI order `lhs,out,n`, and conversion policy evidence.
- `test/Plugin/RVVExtensionPluginTest.cpp` currently has selected-boundary
  producer coverage for `widen_i16_to_i32`, but not the symmetric producer-only
  direct-route rejection coverage for `widen_i32_to_i64` in the production
  route-path exercise.
- Archived task
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-widen-i16-to-i32-selected-realization/`
  completed the same migration pattern for `widen_i16_to_i32`: direct
  route-entry fails closed, selected-boundary dry-run passed, real `ssh rvv`
  counts `0,1,16,23,257` with two signed input patterns passed, and
  `check-tianchenrv` passed 397/397.
- Earlier archived task
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-widen-i32-to-i64-dtype-runtime-artifact-boundary/`
  already established the representative `widen_i32_to_i64` typed/provider/
  artifact/ssh boundary, including real `ssh rvv` counts `0,3,16,17,257`.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, probes,
   harnesses, and script guardrails.
2. Keep `widen_i32_to_i64` in the RVV `widening conversion` selected-body
   realization owner. The selected-boundary producer must realize the typed
   pre-realized body before provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `widen_i32_to_i64`. Direct shortcut requests must fail closed before bundle
   generation or provider route construction.
4. Preserve typed facts for operation kind `widen_i32_to_i64`, i32 source
   binding, i64 result binding, signed widening relation
   `signed-i32m1-to-i64m2`, source SEW32 LMUL m1, destination SEW64 LMUL m2,
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
   conversion dtype-policy owner, route-control provider, math
   operand-binding, migrated statement-plan, target artifact, and
   generated-bundle boundaries where they already express the required facts.
   Do not introduce a new central route table or common EmitC semantic branch.
7. Do not start compare/select, strided load/store, standalone reduction,
   segment2, additional dtype or LMUL clone batches, high-level Linalg/frontend
   lowering, one-intrinsic wrapper dialects, selected-body realization
   framework rewrites, dashboard/report work, broad smoke matrices, or
   evidence-only tasks.
8. Do not add proof-only tests for completed `widen_i16_to_i32` beyond bounded
   non-regression needed by touched files.

## Acceptance Criteria

- [x] Production code no longer treats `widen_i32_to_i64` as direct
      pre-realized route-entry eligible, while the `widening conversion`
      selected-body realization owner still realizes it through the public
      selected lowering-boundary producer path.
- [x] C++ tests prove a typed pre-realized `widen_i32_to_i64` body belongs to
      the `widening conversion` realization owner, is not a direct route-entry
      consumer, and fails closed when
      `realizePreRealizedRVVRouteEntrySelectedBody` is used as a direct
      route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary path consumes realized typed
      facts, including i32 source load, i64 widening conversion, i64 store,
      source/result SEW and LMUL relation, conversion relation, route-control
      provider plan, math operand-binding facts, widening conversion statement
      plan, ABI order, and provider-supported mirrors.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind widen_i32_to_i64` before
      route-entry materialization or bundle generation.
- [x] Generated-bundle dry-run for `widen_i32_to_i64` passes through
      `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization: false`, records selected-body producer
      evidence, records realized dtype/conversion/provider facts, and records
      no direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`, exact,
      tail, and stress cases with at least two signed i32 input patterns and
      expected i64 sign-extended outputs.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum the RVV extension plugin
      smoke test and focused lit for selected-boundary `widen_i32_to_i64`.
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
   owner remains selected-boundary capable, but `widen_i32_to_i64` is not
   direct route-entry eligible.
4. Add or update generated-bundle tooling/lit coverage so direct
   pre-realized route-entry mode fails closed for `widen_i32_to_i64`.
5. Strengthen selected-boundary C++/lit/script coverage so evidence explicitly
   checks `route_entry_realization: false`, i32 source/i64 result dtype
   relation, provider facts, and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-run for `widen_i32_to_i64`.
8. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and at least two signed i32 input patterns through the selected
   lowering-boundary producer path.
9. Run focused adjacent non-regression for `widen_i16_to_i32`.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Completion Result

- Demoted the RVV `widening conversion` selected-body realization owner from
  direct pre-realized route-entry eligibility by removing the widening
  conversion route-entry predicate from
  `RVVSelectedBodyRealization.cpp`, while keeping
  `TypedWideningConversionPreRealizedBodyOp` under the plugin-local selected
  lowering-boundary realization owner.
- Kept `widen_i32_to_i64` generated artifacts on the selected-boundary path:
  the script records `route_entry_realization: false`,
  `rvv-plugin-local-selected-body-realization-owner-registry`, typed i32 source
  and i64 result facts, source SEW32/LMUL m1 to destination SEW64/LMUL m2,
  `signed-i32m1-to-i64m2`, provider route facts, ABI order `lhs,out,n`, and
  neutral EmitC materialization.
- Added direct pre-realized generated-bundle fail-closed coverage for
  `widen_i32_to_i64`; the direct shortcut now rejects this op before
  route-entry materialization or bundle generation.
- Extended the `widen_i32_to_i64` generated harness to execute two signed i32
  input patterns and check i64 sign-extension, wide i32 magnitudes outside the
  i16 range, and tail sentinel preservation across the selected runtime counts.
- Real `ssh rvv` evidence passed for the selected-boundary generated artifact:
  counts `0,1,16,23,257` each ran `pattern=0` and `pattern=1`, ending with
  `PASS op=widen_i32_to_i64 counts=0,1,16,23,257`.

## Validation Result

- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `git diff --check`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] selected-boundary generated-bundle dry-run for `widen_i32_to_i64`
- [x] direct route-entry generated-bundle request fails closed for
      `widen_i32_to_i64`
- [x] selected-boundary generated-bundle dry-run non-regression for
      `widen_i16_to_i32`
- [x] explicit selected-body generated-bundle dry-run for `widen_i32_to_i64`
- [x] real `ssh rvv` generated-bundle execution for counts `0,1,16,23,257`
      and two signed i32 input patterns
- [x] bounded touched-file authority scan
- [x] Trellis context validation
- [x] `cmake --build build --target check-tianchenrv -j2`: 398/398 passed

## Spec Update Result

No `.trellis/spec/` update was needed. The existing RVV plugin, testing,
variant-pipeline, plugin-protocol, and EmitC route specs already capture this
owner-vs-route-entry narrowing pattern: `widening conversion` remains a
selected-body realization owner, while direct route-entry eligibility must be
explicitly narrower and fail closed unless an owner task adds matching
provider facts, diagnostics, generated-bundle evidence, and real RVV evidence.

## Out Of Scope

- Compare/select, strided load/store, standalone reduction, segment2,
  additional dtype or LMUL clone batches, high-level Linalg/Vector/StableHLO
  frontend work, source-front-door construction, descriptor-driven compute,
  direct C/source-export paths, one-intrinsic wrapper dialects, selected-body
  framework rewrite, dashboards, prompt-only work, and legacy `RVVI32M1` /
  `rvv-i32m1` compatibility.
- Additional proof-only work for completed `widen_i16_to_i32`, widening dot,
  widening MAcc, computed-mask widening dot, strided widening dot, or
  computed-mask strided widening dot paths except as bounded non-regression.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Relevant archived tasks read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-widen-i16-to-i32-selected-realization/`
  and
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-widen-i32-to-i64-dtype-runtime-artifact-boundary/`.
- Relevant workspace journal entries read:
  `.trellis/workspace/codex/journal-16.md`, Sessions 253, 254, and 266.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related generated
  bundle / target tests.

## Open Questions

- None blocking at task start. The user brief, specs, previous archived task,
  current production predicate, script allowlist, and focused tests identify a
  bounded module owner.
