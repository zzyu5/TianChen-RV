# Stage2 RVV runtime-scalar masked standalone reduction executable artifact ABI boundary

## Goal

Make the existing selected/pre-realized
`runtime_scalar_cmp_masked_standalone_reduce_add` RVV route executable through
the generated artifact ABI with truthful provider-derived runtime-scalar,
compare-mask, standalone-reduction, scalar-result, header/type, and statement
evidence, or fail closed at the exact stale/missing artifact boundary. This
round is bounded to the runtime-scalar compare-masked standalone reduce-add
workflow and its existing LMUL coverage when shared expectations are already
present.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar masked standalone reduction executable artifact ABI boundary`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `92e6bc50 rvv: tighten product dequant artifact abi evidence`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-product-reduction-dequant-executable-artifact-abi-boundary/prd.md`
  proved the product-reduction dequant artifact ABI boundary by tightening
  provider-owned operand/header binding evidence and running real non-dry-run
  `ssh rvv` evidence. This round must not repeat that route as the main
  achievement.
- Existing focused dry-run script coverage already targets
  `runtime_scalar_cmp_masked_standalone_reduce_add` and
  `runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2`.
- Existing direct pre-realized route-entry coverage already expects
  `--direct-pre-realized-route-entry` to fail closed for the runtime-scalar
  masked standalone-reduction add variants.
- The spec contract requires this path to keep authority in selected typed
  `tcrv_rvv` body/config/runtime facts, RVV standalone-reduction family facts,
  runtime-scalar computed-mask facts, provider-built
  `TCRVEmitCLowerableRoute`, Common EmitC neutral materialization, target
  artifact validation, and generated-bundle evidence.

## Requirements

- Keep the authority chain structural:
  selected/pre-realized `tcrv_rvv` body facts ->
  RVV plugin-local standalone-reduction realization ->
  standalone-reduction provider facts and statement plan ->
  `TCRVEmitCLowerableRoute` ->
  Common EmitC materialization ->
  target artifact export ->
  generated bundle ABI ->
  `ssh rvv` correctness evidence when executable behavior is claimed.
- Validate that runtime ABI order, runtime scalar threshold binding, compare
  mask provenance, inactive-lane contract, reduction kind, accumulator/result
  scalar channel, runtime AVL/VL, header declarations, C type mapping,
  operand-binding summary, route-family plan, and statement plan all line up.
- If the current path is dry-run-only, stale, under-validated, or unable to
  execute, make production code changes at the owner-local RVV provider /
  target-artifact boundary and update focused tests.
- If the current production path already executes without source changes,
  justify that the missing blocker was real `ssh rvv` evidence and do not turn
  the task into report-only bookkeeping.
- Preserve direct pre-realized route-entry fail-closed behavior.
- Keep Common EmitC/export neutral; it must not infer standalone reduction
  semantics, reduction kind, mask behavior, scalar-result layout, ABI roles,
  route support, or intrinsic spelling.

## Acceptance Criteria

- [x] A Trellis PRD and context files exist for this bounded task before source
      edits.
- [x] Focused precheck identifies whether the current path needs production
      repair or only missing runtime evidence.
- [x] Positive evidence covers pre-realized runtime-scalar-cmp-masked
      standalone-reduce-add through materialized selected boundary, emission
      plan, target artifact export, generated bundle compile, and `ssh rvv`
      correctness if executable behavior is claimed.
- [x] Fail-closed evidence covers at least one stale or missing executable
      boundary fact such as runtime scalar ABI order, compare mask provenance,
      inactive-lane contract, reduction kind, scalar result channel, header /
      type mapping, operand binding, or statement-plan facts.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] The relevant generated-bundle dry-run lit test passes.
- [x] The direct pre-realized route-entry fail-closed test remains passing.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      descriptor, source-front-door/source-export, direct-C, helper-name, or
      artifact-name route authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.
- [x] If the task is complete, finish/archive the Trellis task and create one
      coherent commit.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains evidence tooling only; it does not implement compiler core,
  dialects, passes, provider facts, target export, or lowering semantics.
- Runtime-scalar masked standalone-reduction facts remain RVV-provider-owned;
  artifact metadata remains a mirror after route construction.
- Common EmitC/export do not infer RVV semantics, dtype, ABI order, intrinsic
  spelling, scalar carry layout, inactive-lane semantics, or support status.
- No new reduction family, dtype/LMUL clone batch, high-level frontend, broad
  generated-bundle matrix, dashboard, performance tuning DB, source-front-door
  positive route, or mass rewrite of unrelated providers is introduced.

## Technical Approach

Use the existing production path and repair only the stale executable boundary
if repository evidence shows one:

```text
pre-realized runtime-scalar compare-masked standalone-reduce-add body
  -> selected lowering-boundary materialization
  -> RVV standalone-reduction realization and provider facts
  -> standalone reduction route validation contract and metadata mirrors
  -> TCRVEmitCLowerableRoute
  -> Common EmitC materialization
  -> target artifact route-family validation
  -> generated object/header bundle
  -> generated external C ABI harness
  -> ssh rvv compile/run correctness
```

Planning and implementation steps:

1. Run a focused dry-run precheck for the existing pre-realized route.
2. Inspect the provider, selected-body realization owner, target artifact
   validator, generated-bundle script, and focused tests for stale or missing
   runtime-scalar / mask / reduction / ABI boundary facts.
3. If needed, patch production owner-local code and the focused evidence
   expectations.
4. Run focused C++ tests, focused lit/script evidence, non-dry-run `ssh rvv`
   generated-bundle proof when claiming executable behavior, and bounded old
   authority scans.

## Out Of Scope

- New reduction families or new reduction kinds.
- New dtype or LMUL clone batches beyond existing shared expectations.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Source-front-door or source-artifact positive RVV routes.
- Descriptor-driven or direct-C computation export.
- Common EmitC invention of RVV semantics.
- Making direct pre-realized route-entry mode positive.
- Performance tuning, autotuning databases, dashboards, or broad generated
  bundle matrices.
- Mass rewrite of product/dequant, direct contraction, memory, compare/select,
  conversion, segment2, MAcc, or unrelated standalone-reduction owners.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Relevant evidence/tooling files:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`.

## Completion Evidence

- No compiler source change was required. Focused dry-run and code/fixture
  inspection showed the production boundary already carries the required
  provider-owned facts for `runtime_scalar_cmp_masked_standalone_reduce_add`
  and `runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2`:
  runtime ABI order `cmp_lhs,rhs_scalar,src,acc,out,n`, per-operand
  `abi|hdr` route binding, compare predicate `sle`, mask role/source/form,
  inactive zeroing, standalone reduction scalar-result runtime boundary,
  header declarations, C type mapping, and scalar-result channel facts.
- The single missing blocker for claiming executable behavior was real
  non-dry-run `ssh rvv` evidence. This round produced that evidence without
  relaxing target validation or Common EmitC boundaries.
- Focused positive `ssh rvv` evidence:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --artifact-root artifacts/tmp/codex-stage2-rt-scalar-standalone-reduce-abi-ssh \
    --run-id pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add \
    --overwrite \
    --op-kind runtime_scalar_cmp_masked_standalone_reduce_add \
    --op-kind runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2 \
    --runtime-count 0 --runtime-count 1 --runtime-count 16 \
    --runtime-count 23 --runtime-count 257 \
    --rhs-scalar -37 --rhs-scalar 91 \
    --tcrv-opt build/bin/tcrv-opt \
    --tcrv-translate build/bin/tcrv-translate \
    --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj \
    --ssh-target rvv --connect-timeout 10 --timeout 120
  ```

  returned `rvv_generated_bundle_abi_e2e: success`, recorded
  `ssh_evidence: true`, and printed:

  ```text
  PASS op=runtime_scalar_cmp_masked_standalone_reduce_add counts=0,1,16,23,257 rhs_scalars=-37,91 seeds=-11,17 patterns=0,1 runtime_scalar_computed_mask_standalone_reduce source_preserved tail_preserved
  PASS op=runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2 counts=0,1,16,23,257 rhs_scalars=-37,91 seeds=-11,17 patterns=0,1 runtime_scalar_computed_mask_standalone_reduce source_preserved tail_preserved
  ```

- Focused fail-closed evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --direct-pre-realized-route-entry ...` failed
  as expected with
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized
  op kind(s): runtime_scalar_cmp_masked_standalone_reduce_add,
  runtime_scalar_cmp_masked_standalone_reduce_add_i64,
  runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2; the direct
  route-entry shortcut is retired...`.
- Focused checks passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  build-side lit filter for
  `rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run`
  and
  `rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed`
  passed 2/2;
  build-side lit filter for
  `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add`
  and
  `explicit-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add`
  passed 4/4.
- A direct source-test lit invocation first failed because it bypassed
  build-side `lit.site.cfg.py` and therefore lacked `tianchenrv_obj_root`; the
  repaired invocation ran from `build/test` with the generated site config and
  passed.
