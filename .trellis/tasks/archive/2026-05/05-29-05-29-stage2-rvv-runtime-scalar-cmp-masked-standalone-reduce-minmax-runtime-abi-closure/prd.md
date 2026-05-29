# Stage2 RVV runtime-scalar computed-mask standalone reduce_min/reduce_max runtime ABI closure

## Goal

Close `runtime_scalar_cmp_masked_standalone_reduce_min` and
`runtime_scalar_cmp_masked_standalone_reduce_max` as one bounded Stage2 RVV
runtime-ABI and target-artifact validation module, including existing supported
SEW32 LMUL m1/m2 runtime-scalar min/max witnesses.

The executable claim must flow through:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv runtime-scalar computed-mask standalone reduction body
  -> RVV plugin-local selected-body realization
  -> provider-derived route facts
  -> target artifact route-family validator
  -> generated bundle ABI evidence
  -> generated RVV C artifact/harness
  -> ssh rvv correctness evidence
```

## Direction Source

- Direction title: `Expand: Stage2 RVV runtime-scalar computed-mask standalone reduce_min/reduce_max runtime ABI closure`.
- Module owner: RVV target artifact route-family validator, generated-bundle ABI evidence, and directly consumed provider facts for `runtime_scalar_cmp_masked_standalone_reduce_min` and `runtime_scalar_cmp_masked_standalone_reduce_max`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `543f9dd6 rvv: close computed-mask standalone min max runtime abi evidence`.
- `.trellis/.current-task` was absent, so this task was created from the supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The previous completed task closed vector `computed_mask_standalone_reduce_min/max` runtime ABI target validation and `ssh rvv` evidence.
- Current `scripts/rvv_generated_bundle_abi_e2e.py` already lists runtime-scalar standalone min/max op kinds, including LMUL m1/m2 variants.
- Current `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-{min,max}{,-lmul-m2}.mlir` fixtures already expose provider route operand binding, ABI name, operation kind, and selected-body facts.
- Current `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-dry-run.test` and direct fail-closed script test already cover pre-realized runtime-scalar min/max generated-bundle evidence shape.
- Current target artifact C++ tests cover plain standalone min/max and vector computed-mask standalone min/max, but do not add runtime-scalar standalone min/max target artifact positive/fail-closed coverage.
- Current `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` validates shared standalone reduction facts and requires a RHS scalar splat for runtime-scalar computed-mask standalone reductions, but the operation-specific detailed computed-mask validator returns early for non-vector computed-mask operations. This leaves runtime-scalar min/max without the same production validation for typed op, runtime scalar ABI channel, runtime ABI order, operation-specific route operand binding, provider mirror, min/max intrinsic, mask/neutral-inactive contract, source/scalar result type relation, and scalar-carry facts.

## Requirements

1. Treat runtime-scalar standalone `reduce_min` and `reduce_max` as selected-boundary/provider routes. Direct pre-realized route-entry must remain fail-closed.
2. Target artifact validation must consume provider-derived facts for both LMUL m1 and LMUL m2 where the existing typed/provider surface supports them.
3. Validator acceptance must require:
   - operation kind `RuntimeScalarComputedMaskStandaloneReduceMin` or `RuntimeScalarComputedMaskStandaloneReduceMax`;
   - typed compute op `tcrv_rvv.masked_standalone_reduce`;
   - memory form `RuntimeScalarComputedMaskUnitStrideStandaloneReduction`;
   - signed i32 element type, SEW32, LMUL m1/m2 source/work channel;
   - scalar accumulator/result vector channel `!tcrv_rvv.vector<i32, "m1">` / `vint32m1_t`;
   - compare-produced mask type and C type derived from SEW/LMUL;
   - runtime ABI order `cmp_lhs,rhs_scalar,src,acc,out,n`;
   - ABI parameter roles and C types for `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and `n`;
   - runtime scalar compare operand and role `rhs_scalar=rhs-scalar-value`;
   - vector source role `src=source-input-buffer`;
   - accumulator/output/n roles and order;
   - predicate `sle`, compare-produced mask role/source/form, and agnostic mask/tail policy facts where surfaced by provider;
   - neutral inactive-lane min/max contract, seed splat, RHS scalar splat, compare, inactive neutral merge, reduction, and scalar store intrinsic plan;
   - runtime-scalar producer source `runtime-scalar-splat-compare-rhs`;
   - accumulation/scalar-carry contracts;
   - provider mirror `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated`;
   - operation-specific route operand binding plan and binding summary for min/max.
4. Candidate metadata may be validated only as mirrors after rebuilding the provider route. Missing, stale, or inconsistent mirrors must fail closed.
5. Script/generated-bundle evidence must remain evidence, not authority. Do not infer support from op names, route IDs, artifact names, ABI strings, test names, exact intrinsic spellings, descriptors, source-front-door state, direct route-entry, or common EmitC output.

## Acceptance Criteria

- [ ] PRD, implement context, and check context accurately describe the bounded module and non-goals.
- [ ] Target artifact provider validation accepts valid runtime-scalar computed-mask standalone reduce_min and reduce_max selected-body routes for LMUL m1 and LMUL m2.
- [ ] Target artifact provider validation fails closed for missing/stale typed compute op, memory form, operation kind, signed i32 SEW32 LMUL source facts, scalar-result m1 channel, mask type, runtime ABI order, ABI parameter role/order, RHS scalar role, source role, accumulator/output/n roles, provider mirror, route operand binding plan/summary, predicate, mask role/source/form, neutral inactive-lane contract, RHS scalar splat, min/max reduction intrinsic, seed splat, inactive merge, scalar store, accumulation contracts, and runtime-scalar producer source.
- [ ] Target artifact candidate mirror validation rejects stale runtime-scalar min/max mirrors for runtime ABI order, provider mirror, route operand binding, source/scalar result type split, mask role/source/form, neutral inactive-lane requirement, runtime-scalar producer source, and scalar-carry facts.
- [ ] Generated-bundle dry-run coverage for runtime-scalar standalone min/max LMUL m1/m2 passes with counts `0`, `1`, exact, tail, and stress, two signed runtime scalar patterns, provider mirror, route operand binding, and non-empty mask/reduction evidence.
- [ ] Direct pre-realized route-entry remains fail-closed for runtime-scalar standalone min/max.
- [ ] Real `ssh rvv` generated-bundle runs pass for runtime-scalar standalone min/max over counts including `0`, `1`, exact, tail, and stress, with at least two signed seed/scalar patterns, mixed mask and all-inactive cases, and expected min/max results.
- [ ] Non-regression covers `runtime_scalar_cmp_masked_standalone_reduce_add` and vector computed-mask standalone min/max.
- [ ] Bounded touched-file authority scan finds no central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] If complete, the task is finished, archived, journaled, and committed coherently.

## Non-Goals

- Do not add unrelated reduction op kinds.
- Do not add conversion, dtype, or LMUL clone batches beyond existing runtime-scalar min/max m1/m2 surfaces.
- Do not add high-level Linalg/frontend lowering, source-front-door positive routes, one-intrinsic wrappers, direct route-entry positive shortcuts, dashboards, broad smoke matrices, or report-only work.
- Do not reopen already closed vector computed-mask standalone min/max except as bounded non-regression.
- Do not change common EmitC/export code to infer RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry, capability model, lowering, or emission as Python data structures.

## Technical Approach

1. Add runtime-scalar computed-mask standalone reduction expected binding helpers and detailed provider-fact validation in `RVVTargetArtifactRouteFamilyValidation.cpp`, parallel to the vector computed-mask standalone min/max validator but with runtime scalar RHS ABI and `runtime-scalar-splat-compare-rhs` producer facts.
2. Extend `TargetArtifactExportTest.cpp` with runtime-scalar standalone min/max fixtures, LMUL m2 validation, positive registry/export coverage, and targeted provider/candidate mirror fail-closed mutations.
3. Reuse existing generated-bundle script support when sufficient; repair only if dry-run/ssh evidence shows missing runtime-scalar min/max boundary evidence.
4. Run focused build/test/script checks, `ssh rvv` generated-bundle runs, bounded non-regressions, authority scan, `git diff --check`, and `check-tianchenrv`.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. `llvm-lit -v test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-dry-run.test`
4. `llvm-lit -v test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-fail-closed.test`
5. Generated-bundle pre-realized dry-run for `runtime_scalar_cmp_masked_standalone_reduce_min`, `runtime_scalar_cmp_masked_standalone_reduce_min_lmul_m2`, `runtime_scalar_cmp_masked_standalone_reduce_max`, and `runtime_scalar_cmp_masked_standalone_reduce_max_lmul_m2` with counts `0,1,16,23,257` and RHS scalars `-37,91`.
6. Real `ssh rvv` run for the same four runtime-scalar min/max op kinds with the same counts and RHS scalar patterns.
7. Bounded non-regression dry-runs for `runtime_scalar_cmp_masked_standalone_reduce_add`, `runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2`, `computed_mask_standalone_reduce_min`, and `computed_mask_standalone_reduce_max`.
8. Bounded authority scan over touched RVV target/script/test files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-computed-mask-standalone-reduce-min-max-runtime-abi-closure/prd.md`
  - `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-boundary/prd.md`
  - `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-computed-mask-standalone-reduce-add-lmul-m2/prd.md`
  - `.trellis/workspace/codex/journal-18.md`
- Current inspection points:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-{min,max}{,-lmul-m2}.mlir`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-fail-closed.test`
