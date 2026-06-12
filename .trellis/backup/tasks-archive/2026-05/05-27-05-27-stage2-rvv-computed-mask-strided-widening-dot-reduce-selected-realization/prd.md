# Stage2 RVV Computed-Mask Strided-Input Widening Dot-Reduce Selected-Body Realization Migration

## Goal

Move `computed_masked_strided_input_widening_dot_reduce_add` generated artifact
execution behind the RVV plugin-local selected-body realization producer and
remove its direct pre-realized route-entry authority.

The production path for this operation must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv computed-mask strided-input widening dot-reduce body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving compare-produced mask, strided input,
     stride runtime binding, i16 dot sources, i32 accumulator/result, widening
     dtype relation, reduction shape, runtime n/AVL/VL, setvl placement, memory
     roles, mask/tail and inactive-lane policy
  -> contraction route-family facts
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

Direct pre-realized route-entry support for this operation must be demoted or
deleted. A generated artifact for this operation must not be accepted from a
direct route-entry shortcut, pre-realized fixture authority, route id, artifact
name, script option, ABI string, exact intrinsic spelling, common EmitC
behavior, descriptor residue, source-front-door metadata, or legacy i32 helper
authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV computed-mask strided-input widening
  dot-reduce selected-body realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b2d5de81 rvv: demote computed mask widening dot route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The archived
  `05-27-05-27-stage2-rvv-computed-mask-widening-dot-reduce-selected-realization-migration`
  task demoted direct pre-realized route-entry support for
  `computed_masked_widening_dot_reduce_add`, kept the selected-boundary
  producer path positive, and explicitly left
  `computed_masked_strided_input_widening_dot_reduce_add` as the remaining
  direct route-entry positive case.
- The current RVV spec requires selected pre-realized bodies to dispatch
  through the RVV plugin-local selected-body realization owner registry before
  route-family analysis, route-control provider planning, and
  `TCRVEmitCLowerableRoute` construction.
- Route-entry eligibility is narrower than selected-body realization support.
  The contraction owner may keep this body as a selected-boundary realization
  consumer while rejecting direct pre-realized route-entry attempts.
- Direct contraction provider plans must consume verified contraction
  route-family facts, materialization facts, math operand-binding facts,
  route-control provider plans, runtime ABI bindings, stride bindings, mask
  facts, dtype/config facts, and materialized leaves before common EmitC.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` are ABI/runtime envelope
  roles only. They must not define RVV compute, dtype, route support, or
  intrinsic spelling.
- Generated-bundle/runtime evidence is a consumer of provider-built route facts
  and artifact mirrors; it is not route authority.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, probes,
   harnesses, and script guardrails.
2. Keep the selected-boundary producer path positive for
   `computed_masked_strided_input_widening_dot_reduce_add`. It must realize the
   typed pre-realized body into structural `tcrv_rvv` operations before provider
   route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `computed_masked_strided_input_widening_dot_reduce_add`. Direct shortcut
   requests must fail closed with targeted route-entry diagnostics before bundle
   generation.
4. Preserve typed facts for operation kind, computed mask producer and mask use,
   inactive/pass-through behavior, which operand is strided, stride
   `runtime_param` binding, i16 lhs/rhs source binding, i32
   accumulation/result binding, widening dtype relation, reduction shape,
   source/result SEW and LMUL relation, memory roles, runtime `n`/AVL/VL
   values, setvl placement, loop relation, required capabilities, provider
   route facts, and artifact ABI order.
5. Fail closed for unsupported or inconsistent selected-boundary input,
   including missing runtime_param, missing mem_window/runtime ABI value,
   missing or stale mask binding, wrong strided operand, missing or stale stride
   binding, wrong source dtype, wrong accumulator/result dtype, wrong widening
   relation, wrong reduction shape, wrong operand binding, changed mask/tail
   policy, wrong AVL/VL relation, wrong setvl placement, missing capability,
   stale route id or mirror metadata, direct-route-entry-only authority,
   artifact-name/script-derived authority, exact-intrinsic-as-authority, and
   common-EmitC semantic invention.
6. Reuse existing selected-body realization, contraction route-family,
   route-control provider, direct contraction statement owner, target artifact,
   and generated-bundle boundaries where they already express the required
   facts. Do not introduce a new central route table or common EmitC semantic
   branch.
7. Do not start widen_i16_to_i32 conversion, segment2, compare/select,
   strided load/store, standalone reduction cleanup, new dtype/LMUL clone
   batches, high-level Linalg/frontend lowering, one-intrinsic wrapper
   dialects, selected-body realization framework rewrites, dashboard/report
   work, broad smoke matrices, or evidence-only tasks.
8. Do not add proof-only tests for completed adjacent paths as the main
   achievement. Any adjacent coverage must be bounded non-regression for files
   risked by this migration.

## Acceptance Criteria

- [x] Production code no longer treats
      `computed_masked_strided_input_widening_dot_reduce_add` as direct
      pre-realized route-entry eligible, while the `contraction` selected-body
      realization owner still realizes it through the public selected
      lowering-boundary producer path.
- [x] C++ tests prove a typed pre-realized
      `computed_masked_strided_input_widening_dot_reduce_add` body belongs to
      the `contraction` realization owner, is not a direct route-entry
      consumer, and fails closed when
      `realizePreRealizedRVVRouteEntrySelectedBody` is used as a direct
      route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary path consumes realized typed
      facts, including compare-produced mask, mask use, inactive/pass-through
      behavior, strided input memory role, stride runtime binding, i16 dot
      source loads, i32 scalar seed/result layout, widening dot-reduction
      relation, route-control provider plan, math operand-binding facts,
      direct contraction provider plan, ABI order, and provider-supported
      mirrors.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind
      computed_masked_strided_input_widening_dot_reduce_add` before route-entry
      materialization or bundle generation.
- [x] Generated-bundle dry-run for
      `computed_masked_strided_input_widening_dot_reduce_add` passes through
      `--tcrv-materialize-selected-lowering-boundaries`, records
      `route_entry_realization: false`, records selected-body producer
      evidence, records realized strided/mask/dtype/provider facts, and records
      no direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`,
      exact-vector, tail, and stress cases with at least two signed i16 input
      patterns, computed-mask patterns, stride patterns, inactive-lane skip
      checks, and expected i32 reduction results.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum
      `computed_masked_widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
      `widening_dot_reduce_add`, `widening_macc_add`, and one remaining direct
      route-entry fixture if any remains after this migration.
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
3. Narrow RVV selected-body route-entry eligibility so the `contraction` owner
   remains selected-boundary capable, but
   `computed_masked_strided_input_widening_dot_reduce_add` is not direct
   route-entry eligible.
4. Update generated-bundle tooling and focused script coverage so direct
   pre-realized route-entry mode fails closed for this op kind.
5. Strengthen selected-boundary lit/script/C++ coverage so evidence explicitly
   checks `route_entry_realization: false`, strided input/stride binding,
   compare-produced mask, i16 source/i32 result dtype relation, provider facts,
   and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-run for
   `computed_masked_strided_input_widening_dot_reduce_add`.
8. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and bounded pattern/stride variations through the selected lowering-boundary
   producer path.
9. Run focused adjacent non-regression.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Implementation Result

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` keeps
  `TypedComputedMaskStridedInputWideningDotReducePreRealizedBodyOp` in the
  `contraction` selected-body owner set, but removes it from direct
  pre-realized route-entry eligibility by fail-closing
  `isPreRealizedRVVContractionRouteEntryOp`.
- `test/Plugin/RVVExtensionPluginTest.cpp` now proves the body is a
  contraction selected-body consumer, not a route-entry consumer; direct
  `realizePreRealizedRVVRouteEntrySelectedBody` use fails closed, then the
  selected-boundary realization path produces realized compare, strided loads,
  masked widening dot-reduce, store, provider-route, materialization, math, and
  direct contraction plan facts.
- `scripts/rvv_generated_bundle_abi_e2e.py` removes
  `computed_masked_strided_input_widening_dot_reduce_add` from
  `supports_direct_pre_realized_route_entry`, removes it from the direct-mode
  allowlist diagnostic, and records it as selected-boundary-only in the script
  header.
- The generated-bundle harness for this op now runs the same artifact with two
  runtime stride pairs, two computed-mask patterns, and two signed i16 input
  patterns: `2:3/pattern 0` and `3:2/pattern 1`.
- The pre-realized selected-boundary lit test checks
  `route_entry_realization: false`, selected lowering-boundary materialization,
  mask/stride/dtype/provider facts, and the expanded harness evidence.
- A new direct pre-realized route-entry lit test proves the op fails closed
  before route-entry materialization or bundle generation.
- The explicit selected-body consumer test was updated for the expanded shared
  harness evidence because the harness generator is shared by explicit and
  pre-realized selected-body artifact paths.

## Validation Result

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: pass.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`:
  pass.
- `./build/bin/tianchenrv-rvv-extension-plugin-test`: pass,
  `RVV extension plugin smoke test passed`.
- Focused lit:
  `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='explicit-computed-masked-strided-input-widening-dot-reduce-add|pre-realized-computed-masked-strided-input-widening-dot-reduce-add|direct-pre-realized-computed-masked-strided-input-widening-dot-reduce-add'`:
  pass, `3/3`.
- Selected-boundary dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 ...`:
  pass, `route_entry_realization: false`, `materializer:
  tcrv-materialize-selected-lowering-boundaries`.
- Direct route-entry dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind computed_masked_strided_input_widening_dot_reduce_add ...`:
  fails closed as expected with the direct-mode bounded allowlist diagnostic.
- `ssh rvv` generated-bundle execution:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --ssh-target rvv --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 ...`:
  pass, with counts `0,1,16,23,257`, stride pairs `2:3,3:2`,
  `mask_patterns=2`, and `input_patterns=2`.
- Authority scan:
  `supports_direct_pre_realized_route_entry` no longer contains this op; the
  C++ route-entry predicate returns false for this typed body; remaining direct
  mentions are task text, the new fail-closed test, or unrelated still-bounded
  direct fixtures.
- `git diff --check`: pass.
- `cmake --build build --target check-tianchenrv -j2`: pass,
  `397/397`.

## Out Of Scope

- `widen_i16_to_i32`, segment2 follow-ons, compare/select, strided load/store,
  standalone reduction cleanup, new dtype/LMUL clone sets, high-level
  Linalg/Vector/StableHLO frontend work, source-front-door construction,
  descriptor-driven compute, direct C/source-export paths, one-intrinsic
  wrapper dialects, selected-body framework rewrite, dashboards, prompt-only
  work, and legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Additional proof-only work for completed
  `computed_masked_widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`, `widening_dot_reduce_add`,
  `widening_macc_add`, or non-widening MAcc paths except as bounded
  non-regression.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-computed-mask-widening-dot-reduce-selected-realization-migration/`.
- Relevant workspace journal entry read:
  `.trellis/workspace/codex/journal-16.md`, Session 264.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related generated
  bundle / target tests.

## Open Questions

- None blocking at task start. The user brief, specs, previous archived task,
  and current repository state agree on the module owner and non-goals.
