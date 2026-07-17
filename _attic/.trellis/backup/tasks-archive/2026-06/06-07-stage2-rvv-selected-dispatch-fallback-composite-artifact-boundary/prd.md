# Stage2 RVV selected dispatch/fallback composite artifact boundary

## Goal

Connect or harden the production boundary where the RVV composite
gather-MAcc-scatter path enters through real `tcrv.exec` selected
dispatch/fallback structure, runtime guard facts, selected RVV variant facts,
selected-envelope ABI bindings, typed or realized composite `tcrv_rvv` body,
RVV plugin-owned route facts, and target artifact validation.

The task must prevent selected dispatch/fallback mirror strings, artifact names,
route ids, or test names from inventing executable route authority. Mirrors may
remain validation mirrors only.

## What I already know

* Commit `d069c5a9` recorded selected-envelope composite ssh evidence.
* The previous archived composite generated-runtime task proved explicit and
  pre-realized selected-envelope bundles can execute on `ssh rvv`.
* The next bottleneck is the higher execution boundary: actual `tcrv.exec`
  dispatch/fallback facts must line up with selected RVV composite route and
  artifact facts, or the path must fail closed.
* The owner is the real selected dispatch/fallback boundary for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* This task is Stage2 selected-body/dispatch boundary work, not Stage1 i32 route
  authority work, not Stage2 coverage expansion, and not frontend work.

## Requirements

* Inspect current dispatch synthesis, dispatch runtime guard, execution-plan
  coherence, RVV composite selected-body realization, RVV route planning/provider,
  target artifact validation, generated-bundle ABI runner, fixtures, and focused
  tests before source edits.
* Establish one coherent production path if existing executable support is
  present:
  * real `tcrv.exec` dispatch case/fallback facts;
  * runtime guard facts;
  * selected RVV variant facts;
  * selected-envelope `exec_abi_bindings`;
  * typed or realized composite `tcrv_rvv` body;
  * RVV plugin-owned route support/provider facts;
  * target artifact validation and bundle ABI facts.
* If full dispatch/fallback runtime execution is too large for this round,
  finish the structural production boundary and document the exact runtime
  continuation point.
* Fail closed with precise diagnostics when executable claims depend on stale,
  missing, or mirror-only facts.
* Keep common EmitC/export neutral. RVV semantics must remain plugin-owned.
* Keep this task bounded to the composite gather-MAcc-scatter selected path.

## Acceptance Criteria

* [x] A focused positive lit/FileCheck or equivalent test shows actual
  `tcrv.exec` dispatch/fallback facts reaching the selected RVV composite route
  and artifact boundary, or proves that the production seam was already complete
  and validates it without relying on mirror-only metadata.
* [x] Focused fail-closed tests cover stale or missing dispatch case facts.
* [x] Focused fail-closed tests cover stale or missing fallback facts.
* [x] Focused fail-closed tests cover stale or missing runtime guard facts.
* [x] Focused fail-closed tests cover stale or missing selected-envelope
  `exec_abi_bindings` or ABI order facts.
* [x] Focused fail-closed tests cover stale or missing provider support mirror
  facts.
* [x] Focused fail-closed tests cover stale or missing selected variant facts or
  unsupported executable claims.
* [x] If runtime correctness is claimed through this new boundary, run generated
  bundle evidence on `ssh rvv`; otherwise report structural evidence only.
* [x] Run focused build/test targets that cover changed behavior, plus
  `git diff --check`, `git diff --cached --check`, and final clean status.

## Non-goals

* No broad route-family matrix.
* No new gather/scatter/MAcc op family expansion.
* No dtype/LMUL clone batch.
* No IME, offload, TensorExt, future plugin dispatch, or high-level frontend.
* No scalar fallback backend invention beyond the minimal fallback contract
  needed by this selected-dispatch boundary.
* No dashboard, report-only, artifact-index-only, or source-front-door positive
  route.
* No common EmitC invention of RVV semantics.
* No treating selected dispatch/fallback mirror fields, route ids, artifact
  names, or test names as route authority.

## Definition of Done

* PRD, implement/check context, source changes, tests, and task notes match the
  bounded module owner.
* Focused checks pass or any remaining blocker is documented with exact next
  continuation point.
* The Trellis task is finished and archived if complete.
* A single coherent commit is created if complete.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/variant-pipeline/index.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-generated-runtime/prd.md`
* `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
* `include/TianChenRV/Transforms/VariantDispatchSynthesis.h`
* `lib/Transforms/VariantDispatchSynthesis.cpp`
* `include/TianChenRV/Transforms/DispatchRuntimeGuard.h`
* `lib/Transforms/DispatchRuntimeGuard.cpp`
* `include/TianChenRV/Transforms/ExecutionPlanCoherence.h`
* `lib/Transforms/ExecutionPlanCoherence.cpp`
* `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* explicit and pre-realized composite fixtures
* selected-dispatch/fallback negative tests
* focused DispatchSynthesis and DispatchRuntimeGuard tests

## Completion Notes

* No production source change was required. The current production seam already
  rebuilds the RVV provider route from the actual selected `tcrv.exec`
  dispatch-case role, selected variant, same-kernel dispatch/fallback surface,
  runtime guard facts, selected-envelope ABI bindings, and provider route
  description before target artifact export accepts candidate metadata.
* Strengthened the explicit composite target artifact fixture with focused
  fail-closed coverage for missing actual `tcrv.exec.case`, missing actual
  `tcrv.exec.fallback`, missing dispatch-availability runtime guard linkage,
  stale selected dispatch case/fallback mirrors, and missing selected dispatch
  case/fallback mirrors.
* The positive composite FileCheck path continues to show selected
  dispatch/fallback mirrors derived from actual `tcrv.exec` case/fallback facts
  reaching the target artifact header boundary.
* No new `.trellis/spec/` update was needed. The existing variant-pipeline,
  EmitC route, emission-runtime, testing, and validation specs already require
  actual selected dispatch/fallback facts, plugin-owned RVV route construction,
  target artifact mirror validation, and fail-closed metadata-only behavior.
* Runtime correctness through a new boundary was not claimed in this round, so
  no new `ssh rvv` execution was run. Prior selected-envelope ssh evidence from
  commit `d069c5a9` remains the runtime evidence baseline.
* Checks run:
  * `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter`
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative|materialize-dispatch-runtime-guards|variant-dispatch-synthesis|rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run'`
  * `build/bin/tianchenrv-rvv-extension-plugin-test`
  * `build/bin/tianchenrv-target-artifact-export-test`
  * `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  * `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-07-stage2-rvv-selected-dispatch-fallback-composite-artifact-boundary`
  * `git diff --check`
  * `git diff --cached --check`
  * bounded old-authority scan over added test diff lines and task notes.
