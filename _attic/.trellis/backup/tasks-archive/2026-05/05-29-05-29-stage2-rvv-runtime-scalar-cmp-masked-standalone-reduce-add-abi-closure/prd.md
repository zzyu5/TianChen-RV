# Stage2 RVV runtime-scalar computed-mask standalone reduce-add artifact ABI closure

## Goal

Close the artifact/runtime ABI validation gap for
`runtime_scalar_cmp_masked_standalone_reduce_add` and its supported add
variants as one bounded Stage2 RVV target artifact validation module.

The executable route claim must flow through:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv runtime-scalar computed-mask standalone reduction body
  -> RVV plugin-local selected-body realization
  -> provider-derived route facts
  -> target artifact route-family validator
  -> neutral artifact export
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence when runtime/correctness is claimed
```

The task is complete only when the production target artifact route-family
consumer validates add variants from provider-derived facts and fails closed on
stale mirrors. Tests and generated-bundle evidence are required, but they are
not the production authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV runtime-scalar computed-mask standalone
  reduce-add artifact ABI validation closure`.
- Module owner: RVV target artifact route-family validation consumer for
  `runtime_scalar_cmp_masked_standalone_reduce_add`, including supported i32
  LMUL m1/m2 and i64 reduce-add variants.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6a6e0bf2 rvv: close runtime scalar min max ABI validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned or
  parallel agent workflow.

## Current Repository Facts

- The previous archived task closed runtime-scalar computed-mask standalone
  `reduce_min`/`reduce_max` artifact ABI validation in
  `RVVTargetArtifactRouteFamilyValidation.cpp` and added focused target tests,
  generated-bundle dry-runs, real `ssh rvv` evidence, and full
  `check-tianchenrv` coverage.
- That task intentionally limited its detailed validator to min/max so the
  existing runtime-scalar `reduce_add_i64` path would not be rejected.
- Existing archived runtime-scalar reduce-add work already proved the selected
  boundary can generate and run the add artifact, but the production target
  artifact validator did not yet consume the same detailed provider-derived
  add facts as the min/max closure.
- Direct pre-realized route-entry support is globally false for this family and
  must remain fail-closed.
- The relevant long-term specs require target artifact validators to dispatch
  from rebuilt provider descriptions, validate candidate metadata only as
  mirrors, and fail closed on stale provider/candidate facts.

## Requirements

1. Extend the runtime-scalar computed-mask standalone reduction target artifact
   validator from min/max to add variants without reopening unrelated reduction
   coverage.
2. Accept supported i32 LMUL m1, i32 LMUL m2, and i64 reduce-add variants only
   when the rebuilt provider route carries the expected typed body/config,
   route family, route-control, ABI, leaf statement, scalar-result, and
   accumulation facts.
3. Validate provider-derived facts for:
   - operation kind `RuntimeScalarComputedMaskStandaloneReduceAdd`;
   - typed compute op `tcrv_rvv.masked_standalone_reduce`;
   - memory form `RuntimeScalarComputedMaskUnitStrideStandaloneReduction`;
   - element dtype, SEW, source/work LMUL, mask type, and scalar-result vector
     channel;
   - runtime ABI order `cmp_lhs,rhs_scalar,src,acc,out,n`;
   - ABI roles for `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and `n`;
   - runtime scalar RHS compare/splat role;
   - vector source, accumulator, scalar output, and runtime AVL roles;
   - provider mirror
     `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated`;
   - route operand binding plan and binding summary;
   - zero-inactive add contract, seed splat, runtime scalar RHS splat,
     compare, masked merge, reduction, scalar-result store, scalar carry,
     accumulator, result, and accumulation producer-source facts.
4. Validate candidate metadata only as exact mirrors after rebuilding the
   provider route. Missing, stale, swapped, or inconsistent mirrors must fail
   closed with targeted diagnostics.
5. Preserve existing runtime-scalar min/max behavior as non-regression and do
   not widen this round into segment2, conversion, widening dot, MAcc,
   compare/select, memory movement, high-level frontend, or route-entry
   compatibility work.
6. Do not infer route support from exact intrinsic spelling, route ids,
   artifact names, ABI strings, script names, descriptor residue,
   source-front-door state, common EmitC branches, direct route entries, or
   legacy i32 helper authority. Exact intrinsics may only be checked as
   provider-derived target leaf facts after typed route support is established.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module and non-goals.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` production validation
      accepts valid runtime-scalar computed-mask standalone reduce-add selected
      routes for supported i32 LMUL m1, i32 LMUL m2, and i64 variants.
- [x] The validator fails closed for stale or missing typed op, memory form,
      operation kind, dtype/SEW/LMUL, scalar-result vector channel,
      runtime ABI order, ABI roles, provider mirror, route operand binding
      plan/summary, zero-inactive add contract, runtime scalar splat,
      compare/masked merge, reduction/store leaves, scalar carry,
      accumulator/result contracts, and accumulation producer-source facts.
- [x] Target artifact C++ tests include positive coverage for supported
      add variants and focused negative mutations for stale ABI order,
      wrong `rhs_scalar`/`src`/`acc`/`out`/`n` roles, wrong dtype/SEW/LMUL,
      wrong scalar-result channel, wrong provider mirror, wrong route operand
      binding, wrong zero-inactive add contract, wrong runtime scalar splat,
      wrong compare or masked merge facts, wrong reduction/store leaf facts,
      and stale accumulation contracts.
- [x] Generated-bundle dry-runs pass for the add variants with provider mirror,
      route operand binding, source/scalar-result type facts, runtime ABI
      order, and non-empty mask/reduction evidence.
- [x] Direct pre-realized route-entry remains fail-closed for the add variants.
- [x] Real `ssh rvv` correctness passes for counts `0,1,16,23,257` with
      multiple runtime scalar thresholds, accumulator seeds, mixed masks, and
      all-inactive masks.
- [x] Non-regression covers runtime-scalar min/max and existing standalone
      reduce-add/i64 routes.
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

- Do not add new reduction operation coverage beyond the supported add
  variants.
- Do not reopen min/max except as necessary non-regression.
- Do not start segment2, widening conversion, widening dot, MAcc,
  compare/select, memory movement, high-level Linalg/frontend, selected-body
  realization framework rewrites, route-entry compatibility, dashboard/report,
  broad smoke-matrix, or evidence-only work.
- Do not change common EmitC/export code to choose RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.

## Technical Approach

1. Inspect the current min/max runtime-scalar validator and factor or extend its
   expected-fact helpers so add can validate operation-specific provider facts
   without rejecting existing i64 support.
2. Add add-specific expected runtime-scalar standalone reduction binding and
   leaf checks in `RVVTargetArtifactRouteFamilyValidation.cpp`, including
   dtype/config, scalar-result split, route operand binding, zero-inactive add,
   runtime scalar RHS splat, compare, masked merge, reduction/store, scalar
   carry, accumulator, and result contracts.
3. Extend `TargetArtifactExportTest.cpp` with positive add variant coverage and
   targeted fail-closed mutations across provider facts and candidate mirrors.
4. Use `scripts/rvv_generated_bundle_abi_e2e.py` only as generated-bundle
   evidence consumer unless inspection proves the evidence parser is missing
   already-produced provider facts.
5. Run focused build/test/script checks, generated-bundle dry-runs, `ssh rvv`
   correctness, bounded non-regressions, authority scan, `git diff --check`,
   and `check-tianchenrv`.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Generated-bundle dry-run for:
   - `runtime_scalar_cmp_masked_standalone_reduce_add`
   - `runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2`
   - supported i64 runtime-scalar computed-mask standalone reduce-add variant
4. Direct pre-realized route-entry fail-closed check for the same add variants.
5. Real `ssh rvv` generated-bundle correctness over counts `0,1,16,23,257`
   with at least two runtime scalar thresholds and at least two seeds.
6. Bounded non-regression dry-runs for runtime-scalar min/max and existing
   standalone reduce-add/i64 paths.
7. Bounded authority scan over touched RVV target/test/script files.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Completion Evidence

- Production owner changed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
  The runtime-scalar computed-mask standalone reduction payload validator now
  accepts add in addition to min/max, while limiting add support to provider
  supported i32 LMUL m1, i32 LMUL m2, and i64 LMUL m1 typed configurations.
- The validator consumes provider-derived typed op, memory form, operation
  kind, element type, SEW/LMUL, source/scalar-result vector channels, mask
  channel, runtime ABI order/roles, provider mirror, route operand binding
  plan/summary, zero-inactive add contract, seed splat, runtime scalar RHS
  splat, compare, masked merge, reduction, scalar store, accumulation producer,
  accumulator/result, and scalar-carry facts.
- Focused target C++ coverage changed:
  `test/Target/TargetArtifactExportTest.cpp`.
  It now includes positive validation for runtime-scalar standalone reduce_add
  i32 m1, i32 m2, and i64, plus fail-closed mutations for stale typed op, ABI
  order, RHS role, dtype/config, scalar-result channel, provider mirror,
  binding plan, zero-inactive contract, RHS splat, compare, merge, reduction,
  store, producer source, scalar carry, and candidate mirrors.
- Checks passed:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - Manual `tcrv-opt | tcrv-translate | FileCheck` header artifact replay for
    runtime-scalar standalone reduce_add i32 m1, i32 m2, and i64 fixtures.
  - Generated-bundle dry-run for runtime-scalar standalone reduce_add i32 m1,
    i32 m2, and i64:
    `artifacts/tmp/stage2_rvv_runtime_scalar_add_abi_closure/pre-realized-runtime-scalar-reduce-add-dry`.
  - Direct pre-realized route-entry fail-closed probe for the same add variants:
    `artifacts/tmp/stage2_rvv_runtime_scalar_add_abi_closure/direct-pre-realized-runtime-scalar-reduce-add-fail`.
  - Generated-bundle min/max non-regression dry-run:
    `artifacts/tmp/stage2_rvv_runtime_scalar_add_abi_closure/runtime-scalar-minmax-nonregression-dry`.
  - Standalone and computed-mask standalone reduce-add non-regression dry-run:
    `artifacts/tmp/stage2_rvv_runtime_scalar_add_abi_closure/standalone-reduce-add-nonregression-dry`.
  - Real `ssh rvv` generated-bundle correctness for runtime-scalar standalone
    reduce_add i32 m1, i32 m2, and i64 over counts `0,1,16,23,257`, RHS
    scalars `-37,91`, seeds `-11,17`, mixed masks, all-inactive masks, and
    tail preservation:
    `artifacts/tmp/stage2_rvv_runtime_scalar_add_abi_closure/pre-realized-runtime-scalar-reduce-add-ssh`.
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2` passed `459/459`.
- `llvm-lit` was not available in PATH and Python `lit` was not installed, so
  focused lit coverage was replayed manually with the built tools and
  `/usr/lib/llvm-20/bin/FileCheck`.
- Bounded authority scan over touched production/test diff found no new
  executable authority leak. The only forbidden-word hits were deliberate
  negative-test stale mirror strings such as `metadata-derived-*`.
- Spec update review: no `.trellis/spec/` update is needed. The existing RVV
  plugin artifact ABI consumer and standalone reduction contracts already cover
  the behavior added here; this task is an implementation closure of that
  existing contract.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/variant-pipeline/index.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-minmax-runtime-abi-closure/prd.md`
  - `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-boundary/prd.md`
  - `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-computed-mask-standalone-reduce-add-lmul-m2/prd.md`
  - `.trellis/workspace/codex/journal-18.md`
- Primary inspection points:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - directly related runtime-scalar computed-mask standalone reduce-add MLIR
    fixtures and generated-bundle tests.
