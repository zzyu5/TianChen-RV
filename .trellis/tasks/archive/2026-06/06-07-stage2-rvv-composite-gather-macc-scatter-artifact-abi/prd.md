# Stage2 RVV composite gather-MAcc-scatter generated artifact ABI boundary

## Goal

Carry the now-realized RVV composite gather-MAcc-scatter selected-body route across the production target artifact and generated-bundle ABI boundary, or fail closed when the export path lacks truthful provider-owned route, header, prototype, ABI, or executable-boundary facts.

This task starts from the previous realization work in commit `8dfe269b`, where the RVV plugin-local selected-body realization owner fused runtime-scalar computed-mask indexed gather, MAcc, and indexed scatter family bodies into one explicit composite `tcrv_rvv` body. The remaining owner is the artifact/export boundary: the target artifact and generated bundle must consume the realized typed body and provider facts rather than infer RVV semantics from route ids, artifact names, metadata, descriptor residue, test names, or common EmitC strings.

## What I Already Know

- The repository has no active Trellis task for this continuation, so this task records the Hermes direction brief before source changes.
- The current HEAD is `8dfe269b rvv: realize composite gather macc scatter bodies`.
- The current RVV authority chain is `tcrv.exec` envelope -> selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality / selected-body realization / route provider -> `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact -> `ssh rvv` evidence when runtime/correctness/performance is claimed.
- `tcrv.exec` binds ABI/runtime roles and selected variants; it does not own compute semantics.
- Target artifact/export code must treat provider route facts and explicit mirrors as validation inputs, not as independent route authority.
- Runtime correctness claims require real `ssh rvv` evidence.

## Assumptions To Validate

- The selected-body realization path already produces a structurally explicit composite body for the positive route fixture.
- The route provider may already have enough route facts for EmitC lowering, but the target artifact/generated-bundle boundary may be missing composite-specific ABI/header/mirror validation or evidence.
- Existing generated-bundle scripts and target artifact tests can be extended narrowly instead of adding a generic composite framework.
- If the route is not executable end-to-end yet, the task should fail closed with targeted diagnostics and leave runtime claims out of the report.

## Requirements

- Inspect the current composite realization, route planning/provider, EmitC statement owners, target artifact validation/export, generated-bundle script, and focused tests before editing source files.
- Preserve plugin ownership of gather/MAcc/scatter semantics. Common EmitC/export code may materialize neutral mechanics only.
- Make the production target artifact/generated-bundle boundary consume realized typed `tcrv_rvv` body and provider facts for the composite route.
- Validate explicit provider-owned route/header/prototype/ABI mirrors for the composite artifact or fail closed with targeted diagnostics.
- Cover ABI order and executable-boundary facts relevant to gather index use, runtime scalar compare, mask policy/source, MAcc payload/accumulator, scatter destination, runtime `n`/AVL, and header/prototype agreement.
- Add or update focused positive generated artifact evidence for the composite route.
- Add or update at least one focused fail-closed check for stale composite executable-boundary facts such as stale scatter index, stale compare scalar, stale mask source/policy, stale accumulator role, stale destination binding, stale header/prototype, stale provider mirror, or wrong ABI order.
- Run focused build/test/script checks and self-repair failures.
- Run bounded old-authority scans over touched added lines and keep the final worktree clean.

## Acceptance Criteria

- [x] A selected explicit composite RVV route exports as a truthful target artifact/generated bundle with provider-owned route/header/ABI mirrors.
- [x] A pre-realized composite path is realized before export and covered by generated-bundle dry-run evidence.
- [x] The artifact/export boundary rejects at least one stale or inconsistent composite executable-boundary fact with a targeted diagnostic.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Relevant lit/script checks for the generated bundle pass.
- [x] `ssh rvv` evidence is collected if, and only if, runtime correctness is claimed.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Final `git status --short` is clean after commit if the task completes.

## Completion Notes

- Implemented composite route/export support across provider route facts, computed-mask memory statement/route-family validation, construction protocol acceptance, target artifact route-family validation, and generated-bundle ABI evidence.
- Added explicit and pre-realized target fixtures for `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with composite op-kind selection, header/prototype/runtime ABI checks, provider mirror metadata checks, emitted C++ route-chain checks, harness generation, self-test coverage, and generated-bundle dry-run evidence.
- Added focused lit coverage for explicit and pre-realized generated-bundle dry-runs.
- Runtime correctness was not claimed; therefore no `ssh rvv` run was required for this PRD item.
- Focused fail-closed coverage is in `build/bin/tianchenrv-rvv-extension-plugin-test`, including stale composite scatter/index boundary rejection.

## Definition Of Done

- Production artifact/export behavior is changed or a precise no-source-change justification is recorded with direct evidence.
- Focused positive and fail-closed tests cover the changed artifact/export boundary.
- The task is finished/archived through the local Trellis convention if complete.
- One coherent commit records the work, unless the task remains open with an exact continuation point.

## Out Of Scope

- No high-level Linalg, Vector, or StableHLO frontend work.
- No source-front-door positive route.
- No generic composite framework.
- No broad matrix over all memory, MAcc, mask, dtype, or LMUL families.
- No descriptor-driven computation or descriptor-driven C/source export.
- No common EmitC invention of gather, MAcc, or scatter semantics.
- No performance tuning database, dashboard, readiness index, or report-only closure.
- No unrelated reduction, segment2, dequant, scalar-broadcast, or standalone MAcc rewrites except as bounded references.

## Technical Notes

- Read first:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-prerealized-realization/`
  - `include/TianChenRV/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h`
  - `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - relevant RVV EmitC memory, MAcc, and statement plan owners
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
