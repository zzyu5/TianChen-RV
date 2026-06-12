# Stage2 RVV runtime-scalar compare masked-store executable artifact closure

## Goal

Close the executable artifact evidence path for the bounded
`runtime_scalar_cmp_masked_store` RVV Stage2 slice.

The production path under test is:

```text
selected tcrv.exec RVV variant
  -> typed runtime-scalar computed-mask store tcrv_rvv pre-realized body
  -> RVV plugin-owned selected-body realization
  -> RVV provider route facts and statement plan
  -> TCRVEmitCLowerableRoute
  -> target artifact route-family validation/export
  -> generated RVV C/object/header bundle
  -> ssh rvv correctness evidence
```

This task must prove that a runtime scalar compare-produced mask reaches
masked-store memory semantics through the selected-boundary artifact/runtime
path. The executable behavior is true mask lanes copy `src` to `dst`, false
mask lanes preserve old `dst`, and tail lanes preserve sentinels.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime-scalar compare masked-store
  executable artifact closure`.
- Module owner: generated-bundle selected-boundary artifact/runtime path for
  `runtime_scalar_cmp_masked_store`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `10b93d77 rvv: close runtime scalar dual compare select
  artifact`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-runtime-scalar-dual-cmp-select-executable-artifact`
  closed the generated-bundle executable artifact path for
  `runtime_scalar_dual_cmp_mask_and_select`.
- `scripts/rvv_generated_bundle_abi_e2e.py` already lists
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_store_i64`, and
  `runtime_scalar_cmp_masked_store_lmul_m2` as known operation kinds, with
  explicit and pre-realized selected-body expectations, ABI order
  `lhs,rhs_scalar,src,dst,n`, provider mirror
  `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-store-plan-validated`,
  route operand binding plan
  `rvv-route-operand-binding:runtime_scalar_cmp_masked_store.v1`, generated
  harness initializers, and an EmitC boundary extractor for compare-load,
  runtime scalar splat, compare mask, payload load, and masked store.
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-store.mlir`
  starts from a pre-realized selected RVV body and checks selected-boundary
  materialization, provider emission-plan facts, route operand binding facts,
  target header mirrors, runtime ABI order, and callable C prototype.
- Typed i64 and LMUL m2 sibling fixtures exist for the same pre-realized
  selected-body family and may be used only if they remain low-risk and focused.
- The relevant spec boundary is the non-segment computed-mask memory statement
  plan: runtime-scalar computed-mask store must consume verified typed
  body/config/runtime facts, route materialization facts, memory operand-binding
  facts, and route-control provider facts before provider route construction.
- Target artifact export must rebuild the provider route and compare mirrors
  against provider facts. Artifact metadata, route ids, ABI strings, generated
  file names, direct route entries, scripts, common EmitC, descriptors, and
  exact intrinsic spellings are not route authority.

## Production Consumer Being Closed

The production consumer for this task is the generated-bundle exporter path in
`scripts/rvv_generated_bundle_abi_e2e.py` when invoked as:

```text
--pre-realized-selected-body
--op-kind runtime_scalar_cmp_masked_store
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
owner directly. Do not paper over missing facts with script-only assumptions.

## Requirements

1. Preserve selected-boundary-only ownership for
   `runtime_scalar_cmp_masked_store`; the executable generated-bundle path must
   record `route_entry_realization: false`.
2. The generated artifact must come from selected-boundary/provider route facts
   and target artifact export/validation. It must not be authorized by route id,
   artifact name, script-derived metadata, exact intrinsic spelling, callee
   presence, descriptor residue, source-front-door metadata, common EmitC
   semantic invention, or legacy i32 helper names.
3. The generated-bundle evidence must expose runtime ABI order:
   `lhs,rhs_scalar,src,dst,n`.
4. The body, provider facts, target metadata, generated header, and harness must
   carry exact cName/C type/role/ownership for `lhs`, `rhs_scalar`, `src`,
   `dst`, and `n`.
5. The realized typed body and emitted RVV C must prove:
   - typed compare LHS load;
   - runtime scalar RHS splat;
   - compare-produced predicate mask;
   - payload load from `src`;
   - masked store to `dst`;
   - inactive-lane preserve-output semantics;
   - element dtype, SEW, LMUL, tail/mask policy, and memory form;
   - runtime `n`/AVL/VL loop and setvl placement;
   - tail sentinel preservation.
6. The generated harness must test counts including 0, 1, exact VL, tail, and
   stress cases, with signed scalar threshold cases that exercise true lanes,
   false lanes, and boundary-equal lanes.
7. Correctness checks must prove true mask lanes copy `src`, false mask lanes
   preserve old `dst`, and tail sentinels remain unchanged.
8. Focused non-regression must preserve the just-completed
   `runtime_scalar_dual_cmp_mask_and_select` generated-bundle path.
9. If low-risk, typed i64 or LMUL m2 sibling coverage may be added or run as
   bounded evidence, but it must not expand this task into a dtype/LMUL clone
   batch.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module, existing production path, specs, non-goals, and validation plan.
- [x] Focused production diff improves the generated-bundle executable artifact
      closure for `runtime_scalar_cmp_masked_store`, or documents that no
      source diff was needed because the existing production path already
      satisfies the closure under focused evidence.
- [x] Generated-bundle dry-run for
      `--pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_store`
      succeeds and records `route_entry_realization: false`.
- [x] Generated-bundle evidence shows selected-boundary/provider route facts,
      target artifact export, `provider_supported_mirror`, candidate mirrors,
      runtime ABI order, typed body source, statement-plan-derived emitted RVV C,
      compare mask construction, masked-store operand order, inactive-lane
      preserve-output semantics, runtime `n`/AVL/VL, and tail sentinel harness
      checks.
- [x] `ssh rvv` generated-bundle run passes for counts covering 0, 1, exact VL,
      tail, and stress cases with signed threshold patterns.
- [x] Focused fail-closed/non-regression coverage remains present for stale
      provider/candidate mirrors, stale route operand binding facts, stale
      runtime ABI order, missing statement-plan facts, unsupported direct route
      entry where applicable, and the completed dual compare/select path.
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
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-store-dry-run.test`.
  The test checks `route_entry_realization: false`, selected-boundary
  materialization, runtime ABI order `lhs,rhs_scalar,src,dst,n`, provider
  supported mirror, computed-mask memory route-family plan, route operand
  binding facts, emitted RVV compare/splat/masked-store boundary, inactive-lane
  preserve-output contract, counts `0,1,16,23,257`, and scalar thresholds
  `-37,91`.
- Added focused direct-route-entry fail-closed lit coverage:
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-store-fail-closed.test`.
- Generated-bundle dry-run command passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --artifact-root
  artifacts/tmp/rvv_generated_bundle_cmp_masked_store --run-id
  dry-run-runtime-scalar-cmp-masked-store --overwrite --op-kind
  runtime_scalar_cmp_masked_store --runtime-count 0 --runtime-count 1
  --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37
  --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate
  build/bin/tcrv-translate`.
- Real `ssh rvv` generated-bundle command passed with the same operation,
  counts, and scalar thresholds. Runtime output included true-lane
  `payload_distinguishing_lanes`, false-lane `inactive_preserved_lanes`,
  `source_preserved`, `tail_preserved`, and final marker
  `PASS op=runtime_scalar_cmp_masked_store counts=0,1,16,23,257
  rhs_scalars=-37,91`.
- Focused non-regression lit run passed for the prior
  `runtime_scalar_dual_cmp_mask_and_select` generated-bundle dry-run,
  typed dry-run, and direct-route-entry fail-closed tests.
- `cmake --build build --target check-tianchenrv -j2` passed 462/462 tests.
- `git diff --check` passed.
- Bounded authority scan found only negative/guard wording in new tests/PRD and
  existing fail-closed descriptor/source-export guards in target/script code.
  The new executable claim is selected-boundary/provider/target/ssh evidence
  based, not central ad hoc, metadata-only, descriptor-derived,
  source-front-door-derived, direct-route-entry-derived, script-derived, or
  legacy-i32-derived.

## Out of Scope

- Do not start `runtime_scalar_cmp_masked_load_store`, standalone reductions,
  MAcc, widening dot, segment2, compare/select expansion, new dtype/LMUL clone
  batches, high-level frontend lowering, one-intrinsic wrapper dialects, broad
  selected-body framework rewrites, dashboards, reports, or broad smoke
  matrices.
- Do not rework the completed `runtime_scalar_dual_cmp_mask_and_select` path
  except for focused non-regression.
- Do not resurrect direct pre-realized route-entry authority for this selected
  computed-mask memory family.
- Do not make common EmitC/export choose RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.

## Validation Plan

1. Focused generated-bundle dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
   --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_store ...`
2. Focused generated-bundle `ssh rvv` run for the same op with bounded runtime
   counts and signed threshold patterns.
3. Focused generated-bundle non-regression for
   `runtime_scalar_dual_cmp_mask_and_select`.
4. Focused lit/C++ tests only if implementation changes touch target
   validators, provider facts, selected-body realization, or script test
   contracts.
5. Bounded authority scan over touched files and this runtime-scalar masked
   store path.
6. `git diff --check`.
7. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The current specs
already define the selected RVV variant -> typed `tcrv_rvv` body -> RVV
selected-body realization -> provider-built route -> common EmitC -> target
artifact -> `ssh rvv` evidence chain, plus computed-mask memory statement-plan
ownership, mirror-only artifact metadata, common EmitC neutrality, runtime ABI
binding, and runtime evidence requirements. This task applies those contracts
to one generated-bundle and real hardware closure.
