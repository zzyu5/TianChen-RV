# Stage2 RVV runtime-scalar dual compare/select executable artifact closure

## Goal

Close the executable artifact evidence path for the bounded
`runtime_scalar_dual_cmp_mask_and_select` RVV Stage2 slice.

The production path under test is:

```text
selected tcrv.exec RVV variant
  -> typed runtime-scalar dual compare/mask/select tcrv_rvv body
  -> RVV plugin-owned selected-body realization
  -> RVV provider route facts and statement plan
  -> TCRVEmitCLowerableRoute
  -> target artifact route-family validator
  -> generated RVV C/object/header bundle
  -> ssh rvv correctness evidence
```

This task must not become another target-validation-only round. The previous
archived task closed the compare/select mask target artifact statement-plan
validator. This round must prove that the validated runtime-scalar dual
compare/mask/select path is executable through the generated artifact bundle and
real RVV runtime evidence, while preserving fail-closed coverage for stale
authority inputs.

## Direction Source

- Direction title: `Expand: Stage2 RVV runtime-scalar dual compare/select
  executable artifact closure`.
- Module owner: `runtime_scalar_dual_cmp_mask_and_select` from selected
  `tcrv.exec` RVV variant through generated RVV artifact and `ssh rvv`
  correctness evidence.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1e27493d rvv: close compare select mask ABI statement
  validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-compare-select-mask-artifact-abi-statement-plan-validation-closure`
  closed the compare/select mask target artifact consumer so acceptance depends
  on exact provider-built runtime ABI and statement-plan facts, not callee
  presence.
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
  already starts from a pre-realized selected RVV body and checks public
  selected-boundary materialization, provider emission-plan facts, target header
  mirrors, runtime ABI order, mask composition, and callable C prototype.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has a
  `runtime_scalar_dual_cmp_mask_and_select` generated-bundle path that starts
  from the pre-realized selected-body fixture, runs
  `--tcrv-materialize-selected-lowering-boundaries`, emits RVV C, exports the
  target artifact bundle, records `route_entry_realization: false`, and builds
  a harness checking dual runtime scalar comparisons, mask-and composition,
  true/false select payload, output ABI, runtime counts, and tail sentinels.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`
  already covers local dry-run generated-bundle evidence for the i32/m1 case.
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed.test`
  already asserts the direct pre-realized route-entry shortcut is unsupported
  for this selected-boundary-only compare/select family.
- `test/Target/TargetArtifactExportTest.cpp` already includes compare/select
  mask route-family validation coverage for the runtime-scalar dual
  compare/select case, including stale ABI order, scalar cName/type/role/
  ownership, stale provider/candidate mirrors, stale selected typed provenance,
  stale setvl/AVL, compare/secondary-compare/mask-composition operands, select
  result, output store, and memory/layout facts.

## Production Consumer Being Closed

The production consumer for this task is the generated-bundle exporter path in
`scripts/rvv_generated_bundle_abi_e2e.py` when invoked as:

```text
--pre-realized-selected-body
--op-kind runtime_scalar_dual_cmp_mask_and_select
```

This path consumes the real selected-boundary producer and target artifact
bundle exporter:

```text
tcrv-opt <pre-realized fixture>
  --tcrv-materialize-selected-lowering-boundaries
  --tcrv-materialize-emission-plans
  -> tcrv-translate --tcrv-rvv-emitc-to-cpp
  -> tcrv-translate --tcrv-export-target-artifact-bundle
```

If the path already passes, this task still must record focused generated-bundle
and `ssh rvv` evidence, and any code/test changes must improve production
evidence or fail-closed coverage rather than adding reports, dashboards, or
manual fixture-only acceptance.

## Requirements

1. Preserve selected-boundary-only ownership for
   `runtime_scalar_dual_cmp_mask_and_select`; `route_entry_realization` must be
   false for the executable path and the direct route-entry shortcut must fail
   closed.
2. The generated artifact must come from selected-boundary/provider route facts
   and the target artifact validator. It must not be authorized by route id,
   artifact name, script-derived metadata, exact intrinsic spelling, callee
   presence, descriptor residue, source-front-door metadata, common EmitC
   semantic invention, or legacy i32 helper names.
3. The generated-bundle evidence must expose the dual runtime ABI order:
   `cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n`.
4. The path must carry exact cName/C type/role/ownership for both runtime
   scalars, both compare LHS buffers, true/false value buffers, output buffer,
   and runtime element count.
5. The realized typed body and emitted RVV C must prove:
   - typed compare operands;
   - two runtime scalar RHS splats;
   - two compare masks;
   - `mask_and` composition;
   - select true/false operand order;
   - output store;
   - element dtype/SEW/LMUL and policy;
   - runtime `n`/AVL/VL loop and setvl placement;
   - tail preservation.
6. Focused fail-closed coverage must remain present for stale ABI order,
   scalar cName/type/role/ownership, stale mask composition, stale select
   operands/results, stale candidate/provider mirrors, stale statement-plan
   facts, and direct-route-entry-only authority.
7. `ssh rvv` evidence must run counts including 0, 1, exact/small-VL,
   tail, and stress cases with at least two scalar threshold values. The harness
   must prove both masks, composed mask true/false lanes, single-mask-only lanes,
   selected true/false payloads, output ABI, and tail sentinel preservation.

## Acceptance Criteria

- [ ] PRD, implement context, and check context accurately describe the bounded
      module, existing production path, specs, non-goals, and validation plan.
- [ ] Focused production or test diff, if needed, improves the generated-bundle
      executable artifact closure for `runtime_scalar_dual_cmp_mask_and_select`
      rather than adding a report-only or target-validation-only change.
- [ ] Generated-bundle dry-run for
      `--pre-realized-selected-body --op-kind runtime_scalar_dual_cmp_mask_and_select`
      succeeds and records `route_entry_realization: false`.
- [ ] Generated-bundle evidence shows selected-boundary/provider route facts,
      target artifact export, provider-supported mirror, candidate mirrors,
      runtime ABI order, typed body source, statement-plan-derived emitted RVV C,
      mask-and composition, select operand order, output store, `n`/AVL/VL, and
      tail sentinel harness checks.
- [ ] Focused fail-closed tests cover stale ABI order, scalar cName/type/role/
      ownership, mask composition/producer, select operands/results, provider
      mirror, candidate mirror, statement-plan facts, and unsupported direct
      route-entry.
- [ ] `ssh rvv` generated-bundle run passes for counts covering 0, 1, exact/small
      VL, tail, and stress cases with at least two scalar pairs and signed input
      patterns.
- [ ] Non-regression checks for compare/select mask validator and generated
      script tests pass.
- [ ] Bounded authority scan confirms no executable claim in this path depends
      on central ad hoc, name-derived, metadata-only, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-only, direct-route-entry-only, pre-realized-fixture-only,
      callee-presence-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Out of Scope

- Do not add another compare/select mask target-validation-only batch.
- Do not add manual fixture matrices, dashboards, reports, artifact indexes, or
  metadata mirrors as the main achievement.
- Do not start indexed gather/scatter, strided store/load, segment2,
  conversion, reduction, MAcc, high-level Linalg/frontend lowering,
  one-intrinsic wrapper dialects, new dtype/LMUL clone batches, or Stage3
  tuning.
- Do not resurrect direct pre-realized route-entry authority for this selected
  compare/select family.
- Do not make common EmitC/export choose RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry, capability
  model, lowering, or emission as Python data structures.

## Validation Plan

1. Focused generated-bundle dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
   --pre-realized-selected-body --op-kind runtime_scalar_dual_cmp_mask_and_select
   ...`
2. Focused script/lit regressions for runtime-scalar dual compare/select dry-run
   and direct-route-entry fail-closed behavior.
3. Focused C++ target artifact export test if target validator or candidate
   mirror coverage changes.
4. `ssh rvv` generated-bundle run for the same op with bounded runtime counts
   and multiple scalar values.
5. Bounded authority scan over touched files and the dual compare/select path.
6. `git diff --check`.
7. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The current specs
already define the selected RVV variant -> typed `tcrv_rvv` body -> RVV
selected-body realization -> provider-built route -> common EmitC -> target
artifact -> `ssh rvv` evidence chain, plus compare/select selected-boundary
ownership, mirror-only artifact metadata, common EmitC neutrality, and runtime
evidence requirements. This task applies those contracts to a generated-bundle
and real hardware closure.
