# Stage2 RVV base standalone reduce-add artifact ABI closure

## Goal

Close the target artifact ABI validation gap for the non-runtime-scalar
`standalone_reduce_add` and `computed_mask_standalone_reduce_add` route
families. The route claim must continue to flow through the selected typed
RVV body, provider-derived route facts, target artifact route-family
validation, neutral artifact export, generated RVV C bundle, and real `ssh rvv`
evidence when runtime correctness is claimed.

This round is complete only when the production RVV target artifact
route-family validator and the directly affected target/export tests prove
that add is accepted or rejected from rebuilt provider facts and candidate
mirrors. Tests and generated-bundle scripts are evidence consumers only; they
must not become route authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV base standalone reduce-add target
  artifact ABI closure`.
- Module owner: RVV target artifact route-family validator for
  `standalone_reduce_add` and `computed_mask_standalone_reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `22175c7e rvv: close runtime scalar reduce add ABI validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned or
  parallel agent workflow.

## Current Repository Facts

- The archived previous task closed
  `runtime_scalar_cmp_masked_standalone_reduce_add` target artifact ABI
  validation for i32 LMUL m1, i32 LMUL m2, and i64, including provider facts,
  candidate mirrors, generated-bundle dry-run, `ssh rvv` evidence, and
  `check-tianchenrv` coverage.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already contains standalone
  reduction family helpers that recognize add/min/max operations and validate
  rebuilt provider route facts such as typed op, memory form, source/scalar
  result vectors, provider mirror, route operand binding, runtime ABI,
  accumulator/result layout, header/type mappings, ABI mappings, and provider
  statement plan.
- Current focused target tests cover standalone and computed-mask standalone
  min/max in detail. They do not yet give add its own positive and stale-fact
  coverage for the base standalone and vector computed-mask standalone
  validators.
- Candidate mirrors currently cover route binding, provider support, family
  plan, source/scalar-result types, runtime boundary, memory form, runtime
  control, ABI order, headers, C type mapping, accumulator/result layout,
  computed-mask accumulation facts, and mask facts. They do not explicitly
  mirror the selected operation kind, typed compute op, vector load leaf,
  scalar seed splat leaf, reduction leaf, scalar-result store leaf, or
  computed-mask compare/merge leaves for standalone reduction artifacts.
- Direct pre-realized route-entry support for `standalone_reduce_add` remains
  selected-boundary-only by spec and must stay fail-closed.

## Requirements

1. Extend the standalone reduction target artifact candidate mirror contract so
   base standalone and computed-mask standalone reduction artifacts explicitly
   mirror provider-derived operation kind, typed compute op, and leaf intrinsic
   facts after route construction.
2. Keep provider facts as authority. Candidate metadata may only be checked as
   mirrors of the rebuilt provider description and lowerable route.
3. Validate add provider facts for:
   - operation kind `StandaloneReduceAdd` or `ComputedMaskStandaloneReduceAdd`;
   - typed op `tcrv_rvv.standalone_reduce` or
     `tcrv_rvv.masked_standalone_reduce`;
   - memory form `UnitStrideStandaloneReduction` or
     `ComputedMaskUnitStrideStandaloneReduction`;
   - signed i32 element type, SEW 32, LMUL m1/m2;
   - source vector and scalar-result vector channel, including m2 source to m1
     scalar-result reduction channel;
   - mask channel for computed-mask standalone add;
   - runtime ABI order and roles: `lhs,acc,out,n` for base standalone and
     `cmp_lhs,cmp_rhs,src,acc,out,n` for computed-mask standalone;
   - route operand binding plan and summary;
   - reduction accumulator layout, result layout, scalar-result store VL, and
     scalar-result runtime boundary;
   - provider mirror, route-family plan, headers, C type mapping, ABI mappings,
     vector load, seed splat, reduction, store, compare, merge, inactive-lane
     zeroing, accumulation plan, producer source, accumulator/result contracts,
     and scalar carry contract where applicable.
4. Add focused C++ target/export positive coverage for:
   - `standalone_reduce_add` i32 LMUL m1;
   - `standalone_reduce_add` i32 LMUL m2;
   - `computed_mask_standalone_reduce_add` i32 LMUL m1;
   - `computed_mask_standalone_reduce_add` i32 LMUL m2.
5. Add focused fail-closed mutations for stale or missing add facts, including
   wrong operation kind, typed op, memory form, runtime ABI order, parameter
   role, source/scalar-result channel, provider mirror, route binding plan or
   summary, accumulator/result layout, inactive-lane policy, mask channel,
   vector load/seed/reduction/store leaves, computed-mask compare/merge leaves,
   accumulation producer/contract facts, candidate operation/typed-op/leaf
   mirrors, artifact-name/script-derived claim, and direct-route-entry-only
   authority.
6. Preserve non-regression for runtime-scalar computed-mask standalone
   reduce_add/min/max and existing standalone/computed-mask min/max validation.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module and non-goals.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` consumes rebuilt provider
      facts and candidate mirrors for base standalone and computed-mask
      standalone reduce_add, including operation kind, typed op, memory form,
      ABI order/roles, route binding, scalar-result layout, inactive policy,
      mask facts, provider mirror, type/header facts, and leaf intrinsic facts.
- [x] Candidate metadata for standalone reduction artifacts includes and
      validates provider-derived operation, typed-op, vector-load,
      scalar-seed-splat, reduction, scalar-result-store, and computed-mask
      compare/merge mirrors after route construction.
- [x] `TargetArtifactExportTest.cpp` includes positive provider and candidate
      mirror coverage for standalone and computed-mask standalone reduce_add
      i32 m1/m2.
- [x] Focused negative C++ mutations fail closed for stale operation kind,
      typed op, memory form, ABI order, parameter role, scalar-result channel,
      provider mirror, route binding, accumulator/result layout,
      inactive-lane policy, mask channel, leaf intrinsic mirrors, accumulation
      producer/contract facts, artifact/script/name-only authority, and direct
      route-entry-only authority.
- [x] Generated-bundle dry-run passes for `standalone_reduce_add` and
      `computed_mask_standalone_reduce_add` through the selected/artifact
      export path with `route_entry_realization` false.
- [x] Direct pre-realized route-entry remains fail-closed for
      `standalone_reduce_add` and `computed_mask_standalone_reduce_add`.
- [x] Real `ssh rvv` generated-bundle correctness passes for counts including
      `0`, `1`, exact-VL, tail, and stress cases with signed i32 reduction
      inputs.
- [x] Runtime-scalar computed-mask standalone reduce_add/min/max and existing
      standalone/computed-mask min/max non-regressions pass.
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

## Non-Goals

- Do not redo `runtime_scalar_cmp_masked_standalone_reduce_add`.
- Do not add or broaden runtime-scalar min/max, widening dot, MAcc,
  compare/select, segment2, strided memory, widening conversion, high-level
  Linalg/frontend lowering, one-intrinsic wrapper dialects, broad selected-body
  realization rewrites, dtype/LMUL clone batches, dashboards, reports, broad
  smoke matrices, or evidence-only tasks.
- Do not loosen provider legality or accept mirror metadata as authority.
- Do not add common EmitC RVV semantic choices.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.

## Technical Approach

1. Extend provider-produced artifact metadata for standalone reduction routes
   with mirror-only operation, typed-op, and leaf intrinsic facts.
2. Extend `RVVTargetArtifactRouteFamilyValidation.cpp` candidate mirror checks
   to consume those new mirrors for standalone reduction artifacts and reject
   stale computed-mask/runtime-scalar-only leaf mirrors on plain routes.
3. Add focused target/export C++ fixtures for base standalone and
   computed-mask standalone reduce_add m1/m2, including provider fact and
   candidate mirror positives.
4. Add targeted negative mutations for stale add provider facts and candidate
   mirrors, reusing existing min/max fixtures only as non-regression context.
5. Run focused C++ build/test, generated-bundle dry-runs, direct route-entry
   fail-closed probes, `ssh rvv` generated-bundle correctness, non-regression,
   authority scan, `git diff --check`, and `check-tianchenrv`.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Generated-bundle dry-run for:
   - `standalone_reduce_add`
   - `standalone_reduce_add_lmul_m2` or the script's current LMUL m2 spelling
   - `computed_mask_standalone_reduce_add`
   - `computed_mask_standalone_reduce_add_lmul_m2` or the script's current
     LMUL m2 spelling
4. Direct pre-realized route-entry fail-closed probe for the same add variants.
5. Real `ssh rvv` generated-bundle correctness for standalone and
   computed-mask standalone reduce_add over counts including `0`, `1`, an
   exact-VL case, a tail case, and a stress case.
6. Bounded non-regression dry-runs for runtime-scalar computed-mask standalone
   reduce_add/min/max and existing standalone/computed-mask min/max.
7. Bounded touched-file authority scan.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Completion Evidence

- Production movement:
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` now emits mirror-only
    standalone reduction leaf metadata after provider route construction:
    vector load, scalar seed splat, reduction, scalar-result store, and
    computed-mask compare/merge leaves.
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` now consumes
    provider-derived vector-load leaves and candidate mirrors for base
    standalone, computed-mask standalone, and runtime-scalar computed-mask
    standalone reduction artifact validation.
  - `test/Target/TargetArtifactExportTest.cpp` now gives standalone and
    computed-mask standalone `reduce_add` i32 m1/m2 positive provider/candidate
    coverage plus stale-fact fail-closed mutations.
- Self-repair:
  - LMUL m2 standalone reduction fixture scalar-result channels were corrected
    to the required m1 scalar-result vector shape instead of reusing the source
    m2 vector type.
  - `tcrv-opt` and `tcrv-translate` were rebuilt after provider metadata
    changes; generated-bundle evidence was rerun after that rebuild.
- Focused C++ validation:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
    passed.
  - `./build/bin/tianchenrv-target-artifact-export-test` passed.
- Generated-bundle validation:
  - Selected-boundary dry-run passed for `standalone_reduce_add`,
    `standalone_reduce_add_lmul_m2`,
    `computed_mask_standalone_reduce_add`, and
    `computed_mask_standalone_reduce_add_lmul_m2`:
    `artifacts/tmp/stage2_rvv_base_standalone_reduce_add_abi_closure/pre-realized-standalone-reduce-add-dry`.
  - Explicit selected-body m1 dry-run passed for `standalone_reduce_add` and
    `computed_mask_standalone_reduce_add`:
    `artifacts/tmp/stage2_rvv_base_standalone_reduce_add_abi_closure/explicit-standalone-reduce-add-m1-dry`.
  - Direct pre-realized route-entry probe failed closed with the expected
    selected-boundary-only diagnostic:
    `artifacts/tmp/stage2_rvv_base_standalone_reduce_add_abi_closure/direct-pre-realized-standalone-reduce-add-fail`.
  - Evidence JSON for the selected-boundary add artifacts reports
    `route_entry_realization: false` and includes the new provider-derived leaf
    mirrors.
- Real RVV evidence:
  - `ssh rvv` generated-bundle correctness passed for all four add variants
    over counts `0`, `1`, `16`, `23`, and `257`, with signed seeds `-11` and
    `17`, including tail preservation and all-inactive-mask checks for
    computed-mask variants:
    `artifacts/tmp/stage2_rvv_base_standalone_reduce_add_abi_closure/pre-realized-standalone-reduce-add-ssh`.
- Non-regression:
  - Generated-bundle dry-run passed for standalone min/max, computed-mask
    standalone min/max, and runtime-scalar computed-mask standalone
    reduce_add/min/max:
    `artifacts/tmp/stage2_rvv_base_standalone_reduce_add_abi_closure/standalone-reduction-nonregression-dry`.
- Final gates:
  - `cmake --build build --target check-tianchenrv -j2` passed, 459/459.
  - `git diff --check` passed.
  - Bounded production touched-file authority scan found no added central
    ad hoc, name-derived, metadata-derived, descriptor-derived,
    ABI-string-derived, script-derived, artifact-name-derived,
    common-EmitC-derived, source-front-door-derived, route-id-derived,
    exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only,
    or legacy-i32-derived executable authority. Test-only scan hits are
    deliberate stale-metadata negative mutations.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/index.md`
  - `.trellis/spec/guides/index.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-abi-closure/task.json`
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-abi-closure/prd.md`
  - `.trellis/workspace/codex/journal-18.md`
- Primary inspection points:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
