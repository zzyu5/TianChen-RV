# Stage2 RVV computed-mask standalone reduce_min/reduce_max runtime ABI closure

## Goal

Close `computed_mask_standalone_reduce_min` and
`computed_mask_standalone_reduce_max` as one bounded Stage2 RVV runtime-ABI and
target-artifact evidence module. The executable claim must be proven from a
selected `tcrv.exec` RVV variant through a typed
`tcrv_rvv.masked_standalone_reduce` body, RVV provider-derived route facts,
target artifact route-family validation, generated bundle evidence, and real
`ssh rvv` runtime correctness evidence.

## Direction Source

- Direction title: `Switch: Stage2 RVV computed-mask standalone reduce_min/reduce_max runtime ABI closure`.
- Module owner: the RVV target artifact route-family validator and generated-bundle evidence boundary for `computed_mask_standalone_reduce_min` and `computed_mask_standalone_reduce_max`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `a1ff0b12 rvv: close standalone min max runtime abi evidence`.
- `.trellis/.current-task` was absent, so this task was created from the supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The archived plain standalone min/max task closed provider-derived target validation and runtime evidence for `standalone_reduce_min` and `standalone_reduce_max`.
- Current specs require the RVV-first chain: `tcrv.exec` envelope, selected typed `tcrv_rvv` body, RVV plugin-owned legality/realization/provider, common EmitC materializer, target artifact, and `ssh rvv` evidence when runtime/correctness is claimed.
- `tcrv.exec` owns ABI/runtime roles and selected variant organization only. It does not own dtype, operation kind, mask/tail policy, reduction semantics, intrinsic spelling, route support, or executable evidence.
- Current `scripts/rvv_generated_bundle_abi_e2e.py` already has computed-mask standalone min/max op kinds, route operand binding strings, neutral inactive-lane text, and harness checks for active/inactive/all-inactive masks.
- Current generated-bundle evidence only attaches `reduction_accumulation_boundary` to plain standalone reductions, computed-mask standalone add, and runtime-scalar computed-mask standalone reductions. This excludes computed-mask standalone min/max and must be fixed.
- Current `RVVTargetArtifactRouteFamilyValidation.cpp` has a standalone reduction/accumulation consumer, but computed-mask standalone validation is mostly generic non-empty/mirror validation. This round must harden min/max-specific typed/provider facts and fail closed on stale masks, ABI order, scalar seed/result channels, reduction kind, neutral inactive-lane requirement, and route operand binding.
- Current `test/Target/TargetArtifactExportTest.cpp` has focused stale-fact tests for plain standalone min/max, but no computed-mask standalone min/max C++ target-artifact negatives.
- Current generated-bundle dry-run test for computed-mask standalone min/max uses counts `7,16,23`; the brief requires counts including `0`, `1`, exact, tail, and stress cases.

## Requirements

- `computed_mask_standalone_reduce_min` and `computed_mask_standalone_reduce_max` must remain selected-boundary/provider routes. Direct pre-realized route-entry must stay fail-closed.
- Target artifact validation must consume provider-derived typed facts for `tcrv_rvv.masked_standalone_reduce`:
  - operation kind `ComputedMaskStandaloneReduceMin` or `ComputedMaskStandaloneReduceMax`;
  - signed i32 dtype/config with SEW 32 and allowed LMUL relation;
  - typed compute op `tcrv_rvv.masked_standalone_reduce`;
  - memory form `ComputedMaskUnitStrideStandaloneReduction`;
  - compare mask inputs `cmp_lhs` / `cmp_rhs` with roles `lhs-input-buffer` / `rhs-input-buffer`;
  - source vector channel `src` with role `source-input-buffer`;
  - scalar seed/accumulator channel `acc[0]` with role `accumulator-input-buffer`;
  - scalar output channel `out[0]` with role `output-buffer`;
  - runtime count `n`, AVL/VL control, and runtime ABI order `cmp_lhs,cmp_rhs,src,acc,out,n`;
  - agnostic mask/tail policy facts where provided by the selected route;
  - mask role/source/form for compare-produced masks;
  - neutral inactive-lane reduction contract for min/max;
  - operation-specific route operand binding plan and exact binding summary;
  - operation-specific signed min/max reduction intrinsic relation, seed splat, inactive neutral merge, compare, load, and scalar store facts as provider payload facts;
  - provider mirror `provider_supported_mirror:rvv-computed-mask-standalone-reduction-plan-validated`.
- Candidate metadata mirrors may be checked for consistency, but mirror fields, route ids, artifact names, intrinsic spellings, scripts, source-front-door state, descriptors, or legacy i32 names must not become route authority.
- Generated-bundle evidence must expose `reduction_accumulation_boundary` for computed-mask standalone min/max, including computed-mask compare inputs, source/seed/result channels, route operand binding, neutral inactive-lane contract, `provider_supported_mirror`, and runtime count coverage.
- Runtime harnesses must prove scalar seed accumulation and neutral inactive lanes for signed min/max using counts `0`, `1`, exact, tail, and stress cases with at least two signed source/seed and mask patterns.

## Acceptance Criteria

- [ ] Target artifact provider validation accepts valid provider-built computed-mask standalone min/max selected-body routes.
- [ ] Target artifact provider validation fails closed for missing or stale typed compute op, memory form, operation kind, signed min/max reduction intrinsic relation, neutral inactive-lane requirement, mask role/source/form, compare predicate/intrinsic, inactive neutral merge, source vector channel, scalar seed/result layout, runtime n/AVL/VL, runtime ABI order, ABI parameter order, route operand binding plan/summary, and provider mirror.
- [ ] Target artifact candidate mirror validation rejects stale computed-mask standalone min/max mirrors for route operand binding, runtime ABI order, provider mirror, mask role/source/form, neutral inactive-lane requirement, scalar-result type, and accumulation/scalar-carry facts.
- [ ] Generated-bundle explicit selected-boundary dry-runs pass for computed-mask standalone min/max over counts `0`, `1`, exact, tail, and stress.
- [ ] Generated-bundle pre-realized selected-boundary dry-runs pass for computed-mask standalone min/max over the same counts and prove `pre_realized_body_consumed: true`.
- [ ] Generated-bundle evidence for computed-mask standalone min/max includes `reduction_accumulation_boundary` with the computed-mask min/max accumulation boundary, route operand binding, `provider_supported_mirror`, and neutral inactive-lane contract.
- [ ] Direct pre-realized route-entry remains fail-closed for computed-mask standalone min/max.
- [ ] Real `ssh rvv` runs pass for explicit and pre-realized computed-mask standalone min/max over counts including `0`, `1`, exact, tail, and stress cases.
- [ ] Plain standalone min/max closure has bounded non-regression coverage.
- [ ] Bounded touched-file authority scan finds no name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] Trellis task status, check notes, archive state, journal, and commit are truthful at the end of the round.

## Non-Goals

- Do not start runtime-scalar computed-mask standalone min/max.
- Do not reopen standalone reduce_add cleanup except as bounded non-regression.
- Do not reopen segment2, widening dot, MAcc, compare/select expansion, dtype/LMUL clone batches, source-front-door routes, high-level Linalg/frontend work, one-intrinsic wrapper dialects, broad smoke matrices, reports, or dashboards.
- Do not make common EmitC/export choose RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry, capability model, lowering, or emission as Python data structures.

## Technical Approach

1. Add computed-mask standalone min/max target-artifact validation gates next to the existing standalone reduction/accumulation consumer.
2. Add or extend C++ target-artifact fixtures/tests so computed-mask standalone min/max valid provider facts pass and stale provider/mirror facts fail closed.
3. Extend generated-bundle evidence so computed-mask standalone min/max emit `reduction_accumulation_boundary`, not only computed-mask add.
4. Expand the computed-mask standalone min/max dry-run lit test to cover `0`, `1`, exact, tail, and stress counts for explicit and pre-realized selected-body flows.
5. Run focused C++ and script checks, then `ssh rvv` explicit/pre-realized runtime checks.
6. Run bounded non-regression, authority scan, `git diff --check`, and `check-tianchenrv` or record an exact blocker.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. `llvm-lit -v test/Scripts/rvv-generated-bundle-abi-e2e-computed-mask-standalone-reduce-minmax-dry-run.test`
4. Generated-bundle explicit dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind computed_mask_standalone_reduce_min --op-kind computed_mask_standalone_reduce_max --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
5. Generated-bundle pre-realized dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind computed_mask_standalone_reduce_min --op-kind computed_mask_standalone_reduce_max --pre-realized-selected-body --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
6. Direct route-entry fail-closed dry-runs for computed-mask standalone min/max.
7. `ssh rvv` generated-bundle runs for explicit and pre-realized computed-mask standalone min/max over the same counts.
8. Bounded plain standalone min/max non-regression.
9. Bounded authority scan over touched RVV target/script/test files.
10. `git diff --check`
11. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Archived context read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-standalone-reduce-min-max-runtime-abi-closure/task.json`
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-standalone-reduce-min-max-runtime-abi-closure/prd.md`
- Current inspection points:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-computed-mask-standalone-reduce-minmax-dry-run.test`

