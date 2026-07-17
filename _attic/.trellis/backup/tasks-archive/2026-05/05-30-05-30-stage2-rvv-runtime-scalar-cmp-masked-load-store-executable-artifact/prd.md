# Stage2 RVV runtime-scalar compare masked load-store executable artifact closure

## Goal

Close the executable artifact evidence path for the bounded
`runtime_scalar_cmp_masked_load_store` RVV Stage2 slice.

The production path under test is:

```text
selected tcrv.exec RVV variant
  -> typed runtime-scalar computed-mask load-store tcrv_rvv pre-realized body
  -> RVV plugin-owned selected-body realization
  -> RVV provider route facts and computed-mask memory statement plan
  -> TCRVEmitCLowerableRoute
  -> target artifact route-family validation/export
  -> generated RVV C/object/header bundle
  -> ssh rvv correctness evidence
```

This task must prove that a runtime scalar compare-produced mask reaches masked
load/store memory semantics through the selected-boundary artifact/runtime path.
Executable behavior is: true mask lanes load from `src` and store into `dst`,
false mask lanes preserve old `dst` through passthrough/merge behavior, the
source buffer is not mutated, and tail lanes preserve sentinels.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime scalar cmp masked load-store
  executable artifact closure`.
- Module owner: selected-boundary generated-bundle executable artifact path for
  `runtime_scalar_cmp_masked_load_store`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e4607888 rvv: close runtime scalar cmp masked store artifact`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-runtime-scalar-cmp-masked-store-executable-artifact`
  closed the adjacent `runtime_scalar_cmp_masked_store` generated-bundle
  executable artifact path with focused selected-boundary dry-run evidence,
  direct-route-entry fail-closed coverage, `ssh rvv` correctness, and
  `check-tianchenrv` 462/462.
- `scripts/rvv_generated_bundle_abi_e2e.py` already lists
  `runtime_scalar_cmp_masked_load_store`,
  `runtime_scalar_cmp_masked_load_store_i64`, and
  `runtime_scalar_cmp_masked_load_store_lmul_m2` as known pre-realized
  selected-body operation kinds.
- The existing combined dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-memory-dry-run.test`
  includes `runtime_scalar_cmp_masked_load_store` and checks its evidence JSON
  and generated harness boundary, but it is a shared masked-memory smoke rather
  than this task's focused executable closure.
- The existing direct pre-realized route-entry fail-closed test
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-memory-fail-closed.test`
  already covers the load-store family together with store/i64/LMUL siblings.
- Target fixtures already exist for explicit and pre-realized selected-body
  `runtime_scalar_cmp_masked_load_store`, including i32, i64, and LMUL m2
  variants. This task should use the bounded i32 path as the main executable
  claim unless live evidence shows a production owner gap.
- The relevant spec boundary is non-segment computed-mask memory: selected-body
  realization must feed verified computed-mask memory family facts,
  materialization facts, memory operand-binding facts, the shared route-control
  provider plan, and the RVV-owned computed-mask memory statement plan before
  provider route construction.
- Target artifact export must rebuild the provider route and compare mirrors
  against provider facts. Artifact metadata, route ids, ABI strings, generated
  file names, direct route entries, scripts, common EmitC, descriptors, and
  exact intrinsic spellings are not route authority.

## Production Consumer Being Closed

The production consumer for this task is the generated-bundle exporter path in
`scripts/rvv_generated_bundle_abi_e2e.py` when invoked as:

```text
--pre-realized-selected-body
--op-kind runtime_scalar_cmp_masked_load_store
```

This path must consume the real selected-boundary producer and target artifact
bundle exporter:

```text
tcrv-opt <pre-realized fixture>
  --tcrv-materialize-selected-lowering-boundaries
  --tcrv-materialize-emission-plans
  -> tcrv-translate --tcrv-rvv-emitc-to-cpp
  -> tcrv-translate --tcrv-export-target-artifact-bundle
```

If this path already exists but lacks focused evidence or checks, this task
must add the missing production consumer validation and real hardware evidence.
If route/provider or target artifact facts are missing or stale, repair that
production owner directly. Do not paper over missing facts with script-only
assumptions.

## Requirements

1. Preserve selected-boundary-only ownership for
   `runtime_scalar_cmp_masked_load_store`; the executable generated-bundle path
   must record `route_entry_realization: false`.
2. The generated artifact must come from selected-boundary/provider route facts
   and target artifact export/validation. It must not be authorized by route
   id, artifact name, script-derived metadata, exact intrinsic spelling, callee
   presence, descriptor residue, source-front-door metadata, common EmitC
   semantic invention, or legacy i32 helper names.
3. The generated-bundle evidence must expose runtime ABI order:
   `lhs,rhs_scalar,src,dst,n`.
4. The body, provider facts, target metadata, generated header, and harness
   must carry exact cName/C type/role/ownership for `lhs`, `rhs_scalar`,
   `src`, `dst`, and `n`.
5. The realized typed body and emitted RVV C must prove:
   - typed compare LHS load;
   - runtime scalar RHS splat;
   - compare-produced predicate mask;
   - active-lane masked load from `src`;
   - passthrough/old-destination load for inactive lanes;
   - masked store to `dst`;
   - inactive-lane preserve-output semantics;
   - source buffer preservation;
   - element dtype, SEW, LMUL, tail/mask policy, and memory form;
   - runtime `n`/AVL/VL loop and setvl placement;
   - tail sentinel preservation.
6. The generated harness must test counts including 0, 1, exact VL, tail, and
   stress cases, with signed scalar threshold cases that exercise true lanes,
   false lanes, and boundary-equal lanes.
7. Correctness checks must prove true mask lanes copy loaded source values into
   `dst`, false mask lanes preserve old `dst`, `src` remains unchanged, and
   tail sentinels remain unchanged.
8. Focused non-regression must preserve the just-completed
   `runtime_scalar_cmp_masked_store` generated-bundle path.
9. Typed i64 or LMUL m2 sibling coverage may remain in shared tests, but this
   task must not expand into a dtype/LMUL clone batch.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module, existing production path, specs, non-goals, and validation plan.
- [x] Focused production diff improves the generated-bundle executable artifact
      closure for `runtime_scalar_cmp_masked_load_store`, or documents that no
      source diff was needed because the existing production path already
      satisfies the closure under focused evidence.
- [x] Focused generated-bundle dry-run for
      `--pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store`
      succeeds and records `route_entry_realization: false`.
- [x] Generated-bundle evidence shows selected-boundary/provider route facts,
      target artifact export, `provider_supported_mirror`, candidate mirrors,
      runtime ABI order, typed body source, computed-mask memory statement-plan
      emitted RVV C, compare mask construction, masked-load/passthrough/masked
      store operand order, inactive-lane preserve-output semantics, source
      preservation checks, runtime `n`/AVL/VL, and tail sentinel harness checks.
- [x] `ssh rvv` generated-bundle run passes for counts 0, 1, 16, 23, and 257
      with at least two signed runtime scalar thresholds.
- [x] Focused fail-closed/direct-route coverage remains present for selected
      pre-realized runtime-scalar computed-mask memory, including
      `runtime_scalar_cmp_masked_load_store`.
- [x] Focused non-regression for `runtime_scalar_cmp_masked_store` remains
      green.
- [x] Bounded authority scan confirms no executable claim in this path depends
      on central ad hoc, name-derived, metadata-only, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-only, direct-route-entry-only, pre-realized-fixture-only,
      callee-presence-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Completion Evidence

- Added focused generated-bundle dry-run lit coverage:
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-load-store-dry-run.test`.
  The test checks `route_entry_realization: false`, selected-boundary
  materialization, runtime ABI order `lhs,rhs_scalar,src,dst,n`, provider
  supported mirror, computed-mask memory route-family plan, route operand
  binding facts, emitted compare/splat/old-destination passthrough masked-load/
  store boundary, inactive-lane preserve-output semantics, source-preservation
  harness checks, counts `0,1,16,23,257`, and scalar thresholds `-37,91`.
- Added focused direct-route-entry fail-closed lit coverage:
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-load-store-fail-closed.test`.
- Production owner files were verified already sufficient; no RVV provider,
  selected-body materializer, target exporter, or generated-bundle script source
  needed changes. The focused module diff closes the evidence/test coverage
  blocker.
- Focused generated-bundle dry-run command passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --artifact-root
  artifacts/tmp/rvv_generated_bundle_cmp_masked_load_store --run-id
  dry-run-runtime-scalar-cmp-masked-load-store --overwrite --op-kind
  runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1
  --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37
  --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate
  build/bin/tcrv-translate`.
- Real `ssh rvv` generated-bundle command passed with the same operation,
  counts, and scalar thresholds. Runtime output included
  `active_lanes`, `inactive_lanes`, `inactive_preserved_lanes`,
  `payload_distinguishing_lanes`, `source_preserved`, `tail_preserved`, and
  final marker `PASS op=runtime_scalar_cmp_masked_load_store
  counts=0,1,16,23,257 rhs_scalars=-37,91`.
- Focused lit filter for load-store passed 6/6, including the two new script
  tests and the existing explicit/pre-realized target fixtures.
- Focused non-regression lit filter for adjacent `runtime_scalar_cmp_masked_store`
  passed 6/6.
- Bounded authority scan over touched task/test files found only negative guard
  wording, `implicit-check-not` assertions, and mirror/status evidence fields;
  no new executable claim depends on central ad hoc, name-derived,
  metadata-only, descriptor-derived, ABI-string-derived, script-derived,
  artifact-name-derived, common-EmitC-derived, source-front-door-derived,
  route-id-derived, exact-intrinsic-only, direct-route-entry-only,
  pre-realized-fixture-only, callee-presence-only, or legacy-i32-derived
  authority.
- `cmake --build build --target check-tianchenrv -j2` passed 464/464 tests.

## Out of Scope

- Do not redo `runtime_scalar_cmp_masked_store` beyond narrow non-regression.
- Do not start runtime-scalar dual compare/select, standalone reductions,
  segment2, widening dot/MAcc, conversion, compare/select expansion, dtype/LMUL
  clone batches, high-level Linalg/frontend lowering, selected-body framework
  rewrites, report/dashboard work, broad smoke matrices, or evidence-only churn
  beyond this one executable artifact.
- Do not resurrect direct pre-realized route-entry authority for this
  selected-boundary-only family.
- Do not make common EmitC/export choose RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry, capability
  model, lowering, or emission as Python data structures.

## Validation Plan

1. Focused generated-bundle dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
   --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_load_store
   ...`
2. Focused generated-bundle `ssh rvv` run for the same op with counts
   `0,1,16,23,257` and at least two signed threshold patterns.
3. Focused generated-bundle non-regression for
   `runtime_scalar_cmp_masked_store`.
4. Focused direct-route fail-closed check for the runtime-scalar computed-mask
   memory family if the existing shared test is still sufficient; otherwise add
   a narrower load-store-specific guard.
5. Focused lit/C++ tests only if implementation changes touch target
   validators, provider facts, selected-body realization, or script test
   contracts.
6. Bounded authority scan over touched files and this runtime-scalar masked
   load-store path.
7. `git diff --check`.
8. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The current specs
already define the selected RVV variant -> typed `tcrv_rvv` body -> RVV
selected-body realization -> provider-built route -> common EmitC -> target
artifact -> `ssh rvv` evidence chain, plus computed-mask memory statement-plan
ownership, mirror-only artifact metadata, common EmitC neutrality, runtime ABI
binding, and runtime evidence requirements. This task applies those contracts
to one generated-bundle and real hardware closure.
