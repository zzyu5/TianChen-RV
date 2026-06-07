# Stage2 RVV composite masked indexed gather-MAcc-scatter executable artifact ABI boundary

## Goal

Make the first missing production boundary for a low-level RVV runtime-scalar computed-mask indexed gather -> masked MAcc/add accumulation -> indexed scatter-store composite body explicit, plugin-owned, and fail-closed. If the current selected-body route/provider/target machinery cannot yet consume that typed multi-operation body as one `TCRVEmitCLowerableRoute`, it must fail before Common EmitC or target artifact export with a precise diagnostic naming the missing composite owner boundary rather than falling through to an isolated gather, MAcc, scatter, route-id, artifact-name, or mirror-only path.

## What I Already Know

- Repository started clean on `main` at `3d59acc5 rvv: harden runtime scalar indexed scatter target mirrors`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes direction brief.
- Specs require the authority chain to stay in selected typed `tcrv_rvv` body/config/runtime facts, RVV-owned realization/provider facts, `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact mirrors, and `ssh rvv` evidence only when runtime correctness is claimed.
- The recent runtime-scalar computed-mask indexed gather/scatter and computed-mask MAcc tasks closed isolated artifact/ABI seams. This task must not repeat another isolated mirror-only closure.
- Current `RVVSelectedBodyMigratedRouteStatementPlan` is an exactly-one-family boundary; existing owners include computed-mask memory and computed-mask accumulation separately, but no composite migrated owner for gather + MAcc + scatter.
- Current `collectRVVSelectedBodyRouteSlice` treats `tcrv_rvv.masked_indexed_load`, `tcrv_rvv.masked_macc`, and `tcrv_rvv.masked_indexed_store` as single selected route terminals, so a composed body can otherwise fail with a generic "exactly one selected compute op" diagnostic rather than the real composite owner gap.
- Current selected-body realization requires exactly one registry-owned pre-realized body. A pre-realized composite assembled from separate indexed gather, computed-mask MAcc, and indexed scatter pre-realized bodies can otherwise fail as "multiple pre-realized bodies" without naming the composite route boundary.

## Requirements

- Add a bounded RVV-owned fail-closed seam for the composite shape at the selected-body route analysis boundary for explicit realized `setvl`/`with_vl` bodies.
- Add a bounded RVV-owned fail-closed seam for the corresponding pre-realized multi-family shape at the selected-body realization owner registry boundary.
- The diagnostic must name the composite masked indexed gather-MAcc-scatter path and the missing production boundary: a composite selected-body realization / route statement-plan / provider contract capable of deriving one lowerable route from typed body facts.
- The implementation must not make Common EmitC infer RVV semantics, must not create a source-front-door positive route, and must not treat route ids, artifact names, ABI strings, helper names, or metadata mirrors as route authority.
- This round does not claim executable target artifact correctness. It may not add generated-bundle success expectations or `ssh rvv` correctness claims for the composite until a real composite route/provider/target contract exists.

## Acceptance Criteria

- [x] Explicit realized typed body containing runtime scalar compare, computed mask, masked indexed gather load, masked MAcc/add accumulation, masked indexed scatter store, runtime AVL/VL, and relevant ABI values fails closed before route construction with a diagnostic naming the composite boundary.
- [x] Pre-realized multi-family composite body combination fails closed during selected-body realization with a diagnostic naming the composite boundary instead of a generic multiple-body registry error.
- [x] Existing runtime-scalar computed-mask indexed gather/scatter and computed-mask MAcc isolated routes continue to pass.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target artifact code or mirrors are touched; otherwise the PRD records why it is not required.
- [x] Focused lit or C++ coverage proves the new fail-closed seam.
- [x] Bounded old-authority scan over touched files and added diff lines shows no new descriptor/direct-C/source-front-door/legacy `tcrv_rvv.i32_*`/`RVVI32M1`/`rvv-i32m1` authority.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

- PRD and task context truthfully describe the bounded module owner and non-executable status.
- Source/test changes are focused and committed as one coherent commit unless the task remains open with an exact continuation point.
- Trellis task status is completed and archived when checks pass.
- Workspace journal records the work and evidence.

## Out of Scope

- High-level Linalg/Vector/StableHLO frontend work.
- New high-level kernel ops or one-intrinsic wrapper dialects.
- Dtype/LMUL clone batches or broad indexed-memory/MAcc matrices.
- Source-front-door positive routes.
- Common EmitC semantic branching for RVV gather/MAcc/scatter.
- Target artifact metadata as route authority.
- Runtime/performance claims or `ssh rvv` correctness claims for the composite before a real composite route/provider/target contract exists.
- Unrelated segment2, reduction, conversion, dequant, compare/select, offload, IME, TensorExt, dashboard, or tuning work.

## Technical Approach

1. Detect the explicit realized composite shape in `collectRVVSelectedBodyRouteSlice` before single-route terminal recording produces a generic selected-compute conflict.
2. Detect the pre-realized multi-family composite shape in the selected-body realization registry before returning a generic multiple pre-realized bodies error.
3. Return precise RVV plugin errors that name the missing composite selected-body realization / migrated statement-plan / provider contract as the continuation point.
4. Add focused tests for both diagnostics and rerun isolated RVV plugin/target checks.

## Decision (ADR-lite)

Context: The existing production route path supports computed-mask indexed gather, computed-mask indexed scatter, and computed-mask MAcc independently. The composite cannot truthfully be emitted as one artifact until a composite selected-body owner and route/provider contract can join memory and math facts into one lowerable route.

Decision: This round will not fake executable support by stitching existing owners through names or metadata. It will introduce the fail-closed composite boundary in production analysis/realization and tests, then leave the next owner as the composite selected-body realization and route statement-plan/provider contract.

Consequences: The compiler now rejects this Stage2 composite at the correct architectural boundary. A later task can replace the fail-closed seam with a real composite owner that consumes typed gather, MAcc, scatter, mask, index, accumulator, ABI, and AVL/VL facts and emits one provider-built route.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Bounded reference read:
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-masked-indexed-scatter-store-abi-boundary/prd.md`
- Relevant source context:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
- Completed behavior:
  - Added an explicit realized-body route-analysis fail-closed seam for a runtime-scalar computed-mask indexed gather -> masked MAcc -> indexed scatter body in one `tcrv_rvv.with_vl` scope.
  - Added a selected-body realization registry fail-closed seam for the corresponding pre-realized multi-family composite assembled from separate runtime-scalar indexed gather, computed-mask MAcc, and runtime-scalar indexed scatter body ops.
  - Added focused C++ smoke coverage that proves both diagnostics name the composite owner/provider boundary rather than falling into generic single-route or multiple-body errors.
  - No Common EmitC, target artifact validator, artifact metadata, route-id authority, source-front-door route, descriptor route, or executable generated-bundle path was added.
- Evidence:
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`: pass.
  - `build/bin/tianchenrv-rvv-extension-plugin-test`: pass.
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`: pass.
  - `build/bin/tianchenrv-target-artifact-export-test`: pass.
  - `git diff --check`: pass.
  - `git diff --cached --check`: pass with no staged changes at the time it was run.
  - Bounded added-diff old-authority scan over touched source/test/PRD files: no matches for descriptor/direct-C/source-front-door/source-export/legacy `tcrv_rvv.i32_*`/`RVVI32M1`/`rvv-i32m1`.
- Runtime evidence:
  - Not claimed. This round intentionally stops before target artifact export for the composite because the missing production boundary is the composite selected-body realization / migrated statement-plan / provider contract.
- Current phase: check/finish.
- Continuation point:
  - Replace this fail-closed seam with a plugin-local composite owner that imports typed gather, MAcc, scatter, mask, index, accumulator, ABI, and AVL/VL facts into one realized body and one provider-built `TCRVEmitCLowerableRoute`, then add target artifact mirrors/generated bundle/`ssh rvv` evidence.
