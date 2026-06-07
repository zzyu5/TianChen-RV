# Stage2 RVV runtime-scalar-cmp masked indexed scatter-store ABI boundary

## Goal

Make the existing RVV runtime-scalar-cmp masked indexed scatter-store selected-body route truthful at the executable artifact ABI boundary. The route must either pass through typed `tcrv_rvv` memory/body facts, runtime scalar compare operands, computed mask facts, masked indexed scatter-store policy, destination/index/value roles, dtype/config/policy, runtime AVL/VL, provider route validation, `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence, or fail closed before target artifact acceptance when any executable-boundary fact is stale or missing.

This task owns only the runtime-scalar-cmp masked indexed scatter-store side of the computed-mask indexed memory family. It must not broaden into gather-load rework, broad indexed-memory matrices, dtype/LMUL clone batches, source-front-door routes, high-level frontend work, or report-only evidence.

## What I Already Know

- Repository started clean on `main` at `cf904a10 rvv: harden runtime scalar indexed gather target mirrors`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes direction brief.
- Specs require RVV route authority to stay in selected typed `tcrv_rvv` body/config/runtime facts and RVV provider-owned route contracts. Target artifact metadata is a mirror only.
- The previous gather-load follow-up closed stale runtime-scalar mask producer, runtime ABI order, and operand-binding mirror coverage for the gather side.
- Existing scatter-store references named in the brief include explicit/pre-realized MLIR fixtures and generated-bundle dry-run tests for runtime-scalar-cmp masked indexed scatter-store.
- Commit history indicates related runtime-scalar indexed scatter ABI work exists (`f00dcaca`, `822834f0`) and must be inspected as current implementation context rather than assumed incomplete.

## Requirements

- Keep route authority in provider-derived typed `tcrv_rvv` body/config/runtime facts, not route ids, helper names, fixture names, candidate metadata, descriptors, artifact names, or common EmitC semantic branching.
- Preserve common EmitC/export neutrality; common materialization may carry RVV provider payload but must not infer runtime scalar comparison, mask, index, scatter-store, dtype, policy, or ABI semantics.
- Validate the runtime-scalar-cmp masked indexed scatter-store executable boundary for provider route descriptions and candidate artifact mirrors.
- If the scatter-store executable path is dry-run-only or under-validated, repair the bounded production seam so explicit and pre-realized selected-body routes export truthful generated RVV artifacts.
- If the production path is already complete, add the exact missing focused fail-closed evidence or executable evidence identified by repository inspection.
- Keep generated-bundle dry-run and runtime evidence aligned with explicit selected-body and pre-realized selected-body scatter-store paths.

## Acceptance Criteria

- [x] Runtime-scalar-cmp masked indexed scatter-store provider validation rejects stale or missing runtime scalar binding / mask producer facts.
- [x] Runtime-scalar-cmp masked indexed scatter-store provider validation rejects stale or missing compare operand role facts when they participate in the generated ABI boundary.
- [x] Runtime-scalar-cmp masked indexed scatter-store provider validation rejects stale or missing index, value/payload, destination mem_window/store role, inactive-lane store policy, dtype/config/policy, or runtime ABI order facts.
- [x] Runtime-scalar-cmp masked indexed scatter-store candidate mirror validation rejects stale operand binding, runtime-scalar mask producer, runtime ABI order, index/value/destination, and inactive-lane store policy metadata when present as target artifact mirrors.
- [x] Explicit selected-body generated-bundle dry-run test for runtime-scalar-cmp masked indexed scatter-store passes and checks the relevant provider/candidate mirrors.
- [x] Pre-realized selected-body generated-bundle dry-run test for runtime-scalar-cmp masked indexed scatter-store passes and checks the relevant provider/candidate mirrors.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] If runtime correctness is claimed, generated bundles compile and run on `ssh rvv` for active/inactive lanes, noncontiguous indexed scatter, destination preservation for inactive lanes, and tail preservation.
- [x] Bounded old-authority scan over touched files and added diff lines shows no new descriptor/direct-C/source-front-door/legacy `tcrv_rvv.i32_*` route authority.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

- PRD and task context truthfully describe the bounded module owner.
- Source/test changes are focused and committed as one coherent commit unless repository inspection proves no source change is needed.
- Trellis task status is completed and archived when checks and evidence are complete.
- Workspace journal records the work and evidence.

## Out of Scope

- Broad indexed-memory matrix expansion.
- Gather-load rework except as bounded reference.
- Dtype/LMUL clone batches.
- Indexed gather-unit-store or indexed scatter-unit-load expansion.
- Segment2, reduction, MAcc, dequant, clamp, compare/select, conversion, or unrelated mask route changes.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door positive routes.
- Performance tuning databases, dashboards, report-only work, or artifact metadata as authority.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Bounded reference read:
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-load-abi-boundary-followup/prd.md`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-load-abi-boundary-followup/implement.jsonl`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-gather-load-abi-boundary-followup/check.jsonl`
- Relevant source/test files from the Hermes brief will be inspected before implementation, especially RVV computed-mask memory selected-body realization, EmitC route family plan owners, route planning/provider, target artifact route-family validation, target artifact C++ tests, generated-bundle ABI script, and explicit/pre-realized scatter-store fixtures.
- Completed behavior:
  - Added focused target artifact provider fail-closed coverage for runtime-scalar-cmp masked indexed scatter-store stale inactive-lane policy, passthrough/store policy, index uniqueness policy, and indexed destination memory form.
  - Added matching candidate mirror fail-closed coverage for stale `tcrv_rvv.inactive_lane_contract`, `tcrv_rvv.masked_passthrough_layout`, `tcrv_rvv.index_uniqueness`, and `tcrv_rvv.indexed_destination_memory_form`.
  - No production validator semantics changed; the existing computed-mask indexed memory target artifact contract already rejected these stale facts.
- Evidence:
  - `git diff --check`: pass.
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`: pass with existing `-Wswitch` warnings in `TargetArtifactExportTest.cpp`.
  - `build/bin/tianchenrv-target-artifact-export-test`: pass.
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`: pass.
  - `build/bin/tianchenrv-rvv-extension-plugin-test`: pass.
  - `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter runtime-scalar-cmp-masked-indexed-scatter-store` from `build/test`: pass, 4/4.
  - Explicit selected-body generated bundle on `ssh rvv`: pass for counts `0,1,16,17,257`, `rhs_scalar=-37,91`, patterns `0,1`, with active/inactive lane coverage, noncontiguous indexed scatter, `source_preserved`, and `tail_preserved`.
  - Pre-realized selected-body generated bundle on `ssh rvv`: pass for the same counts, scalar values, and patterns with the same preservation evidence.
  - Evidence JSON recorded `status: success`, `ssh_evidence: true`, `remote_compile_succeeded: true`, and `remote_run_succeeded: true` for both explicit and pre-realized generated bundles under `/tmp/tcrv-runtime-scalar-cmp-indexed-scatter-store-artifacts/`.
  - Bounded old-authority scan over added diff lines: no matches for descriptor/direct-C/source-front-door/source-export/legacy `tcrv_rvv.i32_*`/`RVVI32M1`/`rvv-i32m1`.
  - Spec update review: no `.trellis/spec/` change needed because the existing computed-mask indexed memory route metadata mirror contract already requires provider/candidate stale fact rejection for these fields.
- Current phase: finish.
