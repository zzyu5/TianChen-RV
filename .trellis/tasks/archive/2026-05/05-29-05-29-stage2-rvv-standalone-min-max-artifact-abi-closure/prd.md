# Stage2 RVV standalone min/max artifact ABI closure

## Goal

Close the named target artifact ABI validation boundary for signed
`standalone_reduce_min`, `standalone_reduce_max`,
`computed_mask_standalone_reduce_min`, and
`computed_mask_standalone_reduce_max`. The artifact path must be validated from
the selected typed RVV body and rebuilt provider route facts, through
mirror-only candidate metadata, neutral EmitC materialization, generated RVV C
artifacts, and real `ssh rvv` evidence when correctness is claimed.

This task is complete only if the production RVV provider-derived route facts
or target artifact route-family validator moves for min/max. Tests, generated
bundle scripts, artifact names, route ids, and reports are evidence consumers
only; they are not route authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV standalone min/max artifact ABI closure`.
- Module owner: RVV provider-derived route facts and RVV target artifact ABI
  validator for signed standalone min/max reductions.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6e07acae rvv: close base standalone reduce add ABI validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned or
  parallel agent workflow.

## Current Repository Facts

- The archived previous task
  `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-standalone-reduce-add-abi-closure`
  closed add artifact ABI validation and added standalone reduction leaf mirror
  metadata after provider route construction.
- Current provider planning already classifies plain and computed-mask
  standalone min/max operation kinds and emits provider description mirrors for
  operation kind, typed compute op, route binding, source/scalar-result channel,
  mask facts, accumulator/result layout, reduction leaf, compare leaf, merge
  leaf, and scalar-result store.
- Current target artifact validator already checks many min/max provider and
  candidate facts. The remaining production gap for this round is that the
  standalone reduction statement-plan validator mostly checks that required
  leaf callees are present, instead of consuming the rebuilt provider route's
  concrete statement operands/results that prove min/max scalar seed, lane-0
  scalar-result layout, store VL, computed-mask compare/merge channel, and
  inactive-lane neutral behavior.
- Existing target tests include min/max positives and several stale-fact
  negatives, but the task requires named closure with production validator
  movement, not tests-only coverage.

## Requirements

1. Keep provider-built route facts as authority. Candidate metadata remains
   mirror-only and must be compared to rebuilt provider route descriptions.
2. Extend production validation for plain standalone min/max reductions so the
   target artifact validator consumes rebuilt route statement facts for:
   - pre-loop full-chunk `setvl`;
   - scalar seed splat from `acc[0]`;
   - initial scalar-result store to `out` with store VL `1`;
   - loop `setvl` from remaining runtime AVL;
   - source vector load from `lhs + induction`;
   - scalar seed reload from `out[0]`;
   - signed min/max reduction leaf;
   - scalar-result store to `out` with store VL `1`.
3. Extend production validation for vector computed-mask standalone min/max
   reductions so the target artifact validator consumes rebuilt route statement
   facts for:
   - compare lhs/rhs loads from `cmp_lhs + induction` and
     `cmp_rhs + induction`;
   - payload source load from `src + induction`;
   - compare predicate leaf;
   - operation-specific inactive neutral source splat (`INT32_MAX` for min,
     `INT32_MIN` for max);
   - masked merge into the reduction input;
   - scalar seed reload from `out[0]`;
   - signed min/max reduction leaf;
   - scalar-result store to `out` with store VL `1`.
4. Preserve add standalone reduction behavior as non-regression while adding
   min/max closure. Do not reopen runtime-scalar computed-mask standalone
   reduce add/min/max except as focused non-regression evidence.
5. Fail closed with targeted diagnostics for stale add-vs-min/max facts, wrong
   min/max leaf, wrong inactive neutral literal, wrong scalar-result store VL,
   wrong scalar-result channel, wrong mask/compare/merge channel, wrong ABI
   order/roles, stale candidate mirror, direct route-entry-only authority, and
   mirror/name/script/route-id/exact-intrinsic-as-authority attempts.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module and non-goals.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` consumes rebuilt provider
      route statements for standalone reduce_min/max scalar seed, lane-0
      scalar-result, store VL, source load, min/max reduction leaf, and
      scalar-result store.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` consumes rebuilt provider
      route statements for computed-mask standalone reduce_min/max compare,
      source load, operation-specific inactive neutral literal, merge, scalar
      seed, min/max reduction leaf, and scalar-result store.
- [x] Focused C++ target/export coverage proves positive provider and candidate
      acceptance for standalone and computed-mask standalone reduce_min/max,
      including LMUL m1 and m2 where supported.
- [x] Focused negative C++ mutations fail closed for wrong min/max intrinsic,
      stale add/min/max operation facts, wrong inactive neutral literal, wrong
      scalar-result store VL, wrong scalar-result channel, wrong mask/compare/
      merge channel, wrong ABI order/role, stale candidate mirrors, and
      direct-route-entry-only authority.
- [x] Selected-boundary generated-bundle dry-run passes for min/max standalone
      and computed-mask standalone variants with `route_entry_realization:
      false`.
- [x] Direct pre-realized route-entry remains fail-closed or verified as
      selected-boundary-only non-regression for these routes.
- [x] Real `ssh rvv` generated-bundle correctness passes for min/max counts
      including `0`, `1`, exact-VL, tail, and stress cases with signed values,
      duplicate extrema, seed interaction, active/inactive mask lanes, and
      all-inactive-mask cases.
- [x] Completed add standalone reductions and runtime-scalar standalone
      reductions remain non-regressed.
- [x] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      executable authority.
- [x] `git diff --check` passes.
- [x] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is
      recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Add small target-artifact validator helpers for standalone reduction route
   statement inspection, using the rebuilt `TCRVEmitCLowerableRoute` rather
   than candidate metadata.
2. Strengthen standalone reduction statement-plan validation to check the exact
   provider-built operand/result structure for plain and computed-mask min/max.
3. Add C++ fail-closed mutations that corrupt route statement facts after route
   rebuild so the new production validation is exercised directly.
4. Run focused target artifact tests, generated-bundle dry-runs, direct
   route-entry fail-closed/non-regression, `ssh rvv` correctness, add/runtime
   scalar non-regression, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Out of Scope

- Do not start runtime-scalar computed-mask standalone reduce add/min/max as a
  new owner.
- Do not start widen/contraction, segment2, compare/select, memory movement,
  scalar frontend, high-level Linalg/Vector lowering, one-intrinsic wrapper
  dialects, dashboards, broad smoke matrices, report-only work, or
  evidence-only cleanup.
- Do not loosen provider legality, target artifact validation, or mirror-only
  metadata rules.
- Do not add common EmitC RVV semantic choices.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Generated-bundle dry-run for:
   - `standalone_reduce_min`
   - `standalone_reduce_max`
   - `standalone_reduce_min_lmul_m2`
   - `standalone_reduce_max_lmul_m2`
   - `computed_mask_standalone_reduce_min`
   - `computed_mask_standalone_reduce_max`
   - `computed_mask_standalone_reduce_min_lmul_m2`
   - `computed_mask_standalone_reduce_max_lmul_m2`
4. Direct pre-realized route-entry fail-closed/non-regression for the same
   min/max standalone variants.
5. Real `ssh rvv` generated-bundle correctness for standalone and
   computed-mask standalone reduce_min/max over counts including `0`, `1`, an
   exact-VL case, a tail case, and a stress case.
6. Bounded non-regression dry-runs for completed standalone add and
   runtime-scalar computed-mask standalone reduce_add/min/max.
7. Bounded touched-file authority scan.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/index.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-standalone-reduce-add-abi-closure/task.json`
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-standalone-reduce-add-abi-closure/prd.md`
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-standalone-reduce-add-abi-closure/implement.jsonl`
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-standalone-reduce-add-abi-closure/check.jsonl`
- Primary inspection points:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`

## Completion Evidence

- Production owner moved in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`: standalone
  reduction artifact validation now consumes rebuilt provider route statement
  operands/results for pre-loop setvl, scalar seed splat, scalar-result store,
  loop source load, computed-mask compare/source loads, inactive neutral
  splat, masked merge, reduction leaf, and lane-0 scalar-result store.
- Focused target/export C++ coverage moved in
  `test/Target/TargetArtifactExportTest.cpp`: cloned provider-built routes are
  mutated after route rebuild to prove fail-closed diagnostics for stale scalar
  seed source, stale scalar-result store VL, wrong inactive neutral literal,
  wrong merge mask operand, and min/max leaf confusion, while retaining prior
  provider/candidate mirror negatives.
- Self-repair performed: the first focused C++ run exposed that LMUL m2 vector
  computed-mask inactive neutral splats must use the provider-derived source
  vector splat, not the scalar-result m1 seed splat. The validator now keeps
  inactive neutral source-vector splat validation separate from scalar seed
  splat validation.
- Selected-boundary dry-run passed:
  `artifacts/tmp/stage2_rvv_standalone_min_max_abi_closure/pre-realized-standalone-min-max-dry`
  for standalone and computed-mask standalone reduce_min/reduce_max, LMUL m1
  and m2.
- Direct pre-realized route-entry probe failed closed as selected-boundary-only:
  `artifacts/tmp/stage2_rvv_standalone_min_max_abi_closure/direct-pre-realized-standalone-min-max-fail`.
- Real `ssh rvv` generated-bundle compile/run/correctness passed:
  `artifacts/tmp/stage2_rvv_standalone_min_max_abi_closure/pre-realized-standalone-min-max-ssh`
  for standalone and computed-mask standalone reduce_min/reduce_max, LMUL m1
  and m2, counts `0`, `1`, `16`, `23`, `257`, signed seeds `-11` and `17`,
  duplicate extrema, active/inactive mask lanes, all-inactive-mask cases, and
  tail preservation.
- Non-regression dry-run passed:
  `artifacts/tmp/stage2_rvv_standalone_min_max_abi_closure/standalone-reduction-nonregression-dry`
  for completed standalone reduce_add, computed-mask standalone reduce_add,
  and runtime-scalar computed-mask standalone reduce_add/min/max.
- Checks passed:
  `cmake --build build --target tianchenrv-target-artifact-export-test -j2`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  `git diff --check`;
  `cmake --build build --target check-tianchenrv -j2` with `459/459` tests.
- Bounded added-line authority scan over the changed production/test files
  found no new `metadata-derived`, `route-id`, `descriptor`, `__riscv_`,
  source-front-door, common-EmitC, direct-route-entry, pre-realized-fixture, or
  legacy-i32 executable authority. Existing exact intrinsic strings remain
  provider-derived leaf facts consumed by the artifact validator, not route
  authority.
- Formatting note: `clang-format` was not available in this environment, so no
  clang-format rewrite was performed; `git diff --check` and the focused/full
  builds passed.
