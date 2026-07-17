# Stage2 RVV runtime-scalar-cmp masked indexed gather-load ABI boundary

## Goal

Make the existing RVV runtime-scalar-cmp masked indexed gather-load selected-body route truthful at the executable artifact ABI boundary. The route must either pass through typed `tcrv_rvv` body/runtime facts -> RVV provider validation -> `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact bundle -> generated bundle evidence, or fail closed before target artifact acceptance when runtime scalar, computed mask, indexed gather, inactive-lane, header/prototype, ABI order, dtype/config, or runtime AVL/VL facts are stale or missing.

This task owns only the runtime-scalar-cmp masked indexed gather-load side of the computed-mask indexed memory family. It must not broaden into scatter-store rework, dtype/LMUL expansion, source-front-door paths, performance tuning, or high-level frontend work.

## What I Already Know

- Repository started clean on `main` at `822834f0 rvv: harden runtime scalar indexed scatter ABI boundary`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes direction brief.
- Specs require the RVV authority chain to stay in selected typed `tcrv_rvv` body/config/runtime facts and RVV provider-owned route contracts, with target artifact metadata as mirrors only.
- The existing explicit and pre-realized gather-load MLIR fixtures and generated-bundle dry-run tests already exist for `runtime_scalar_cmp_masked_indexed_gather_load_unit_store`.
- The target artifact test already has a manual runtime-scalar computed-mask indexed gather description, positive provider/candidate validation, binding summary rejection, and stale indexed data memory mirror rejection.
- Compared with the just-finished runtime-scalar indexed scatter boundary, the gather side appears to lack focused C++ fail-closed evidence for stale runtime-scalar mask producer, stale runtime ABI order, and candidate metadata mirrors for operand binding, mask producer, and ABI order.

## Requirements

- Keep route authority in provider-derived typed `tcrv_rvv` body/config/runtime facts, not route ids, helper names, fixture names, candidate metadata, descriptors, or common EmitC semantic branching.
- Preserve common EmitC/export neutrality; common materialization may carry RVV provider payload but must not infer runtime scalar comparison, mask, index, gather, dtype, or ABI semantics.
- Validate the runtime-scalar-cmp masked indexed gather-load executable boundary for both provider route descriptions and candidate metadata mirrors.
- If the current production route validation is already correct, add focused fail-closed C++ target artifact coverage for the missing stale gather-load boundary facts.
- If the new coverage exposes a production gap, repair the target artifact route-family validator or provider fact surface in the bounded gather-load seam.
- Keep generated-bundle dry-run and runtime evidence aligned with explicit selected-body and pre-realized selected-body gather-load paths.

## Acceptance Criteria

- [x] Runtime-scalar-cmp masked indexed gather-load provider validation rejects stale runtime-scalar mask producer facts.
- [x] Runtime-scalar-cmp masked indexed gather-load provider validation rejects stale runtime ABI order facts.
- [x] Runtime-scalar-cmp masked indexed gather-load candidate mirror validation rejects stale operand-binding metadata.
- [x] Runtime-scalar-cmp masked indexed gather-load candidate mirror validation rejects stale runtime-scalar mask producer metadata.
- [x] Runtime-scalar-cmp masked indexed gather-load candidate mirror validation rejects stale runtime ABI order metadata.
- [x] Existing explicit and pre-realized generated-bundle dry-run tests for this route still pass.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] If runtime correctness is claimed, generated bundles compile and run on `ssh rvv` for the route with active/inactive lane, indexed gather, source/passthrough preservation, and tail coverage.
- [x] Bounded old-authority scan over touched files and added diff lines shows no new descriptor/direct-C/source-front-door/legacy `tcrv_rvv.i32_*` route authority.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

- PRD and task context truthfully describe the bounded module owner.
- Source/test changes are focused and committed as one coherent commit unless no source change is proven correct.
- Trellis task status is completed and archived when checks and evidence are complete.
- Workspace journal records the work and evidence.

## Out of Scope

- Broad indexed-memory matrix expansion.
- Runtime-scalar indexed scatter-store rework except as bounded reference.
- Dtype/LMUL clone batches.
- Segment2, reduction, MAcc, dequant, clamp, compare/select, conversion, or unrelated mask route changes.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door positive routes.
- Performance tuning databases, dashboards, reports-only work, or artifact metadata as authority.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/index.md`
  - `.trellis/spec/guides/index.md`
  - `.trellis/spec/guides/capability-first-design-guide.md`
  - `.trellis/spec/guides/plugin-locality-review-guide.md`
  - `.trellis/spec/guides/compute-boundary-review-guide.md`
- Bounded reference read: archived scatter-store task PRD under `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-scatter-store-abi/`.
- Likely implementation touch point: `test/Target/TargetArtifactExportTest.cpp`.
- Relevant existing generated-bundle tests:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-load-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-indexed-gather-load-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-indexed-gather-load-fail-closed.test`
- Completed behavior:
  - Added target artifact provider-fact fail-closed coverage for stale runtime-scalar gather mask producer and runtime ABI order.
  - Added target artifact candidate-mirror fail-closed coverage for stale runtime-scalar gather operand binding, mask producer, and runtime ABI order metadata.
  - No production route semantics changed; the existing validator already rejected the stale facts.
- Evidence:
  - `git diff --check`: pass.
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`: pass.
  - `build/bin/tianchenrv-target-artifact-export-test`: pass.
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`: pass.
  - `build/bin/tianchenrv-rvv-extension-plugin-test`: pass.
  - `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter runtime-scalar-cmp-masked-indexed-gather-load` from `build/test`: pass, 5/5.
  - Explicit selected-body generated bundle on `ssh rvv`: pass for counts `0,1,16,17,257`, `rhs_scalar=-37,91`, patterns `0,1`, with active/inactive lane coverage, noncontiguous indexed gather, `source_preserved`, and `tail_preserved`.
  - Pre-realized selected-body generated bundle on `ssh rvv`: pass for the same counts, scalar values, and patterns with the same preservation evidence.
  - Runtime evidence JSON recorded `remote_compile_succeeded: true`, `remote_run_succeeded: true`, and `status: success` for both explicit and pre-realized generated bundles under `/tmp/tcrv-runtime-scalar-cmp-indexed-gather-load-artifacts/`.
  - Spec update review: no `.trellis/spec/` change needed because the existing computed-mask indexed memory validation contract already requires provider and candidate stale fact rejection for these fields.

## Current Phase

finish
