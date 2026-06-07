# Stage2 RVV resource-aware Gearbox selected-body realization foundation

## Goal

Add a bounded production compiler/pass foundation for resource-aware RVV
selected-body realization on the runtime-scalar-cmp masked indexed
gather-MAcc-scatter composite path.

The composite selected-body owner must derive resource/schedule facts from the
typed pre-realized `tcrv_rvv` bodies, the selected runtime AVL/VL control plan,
the selected `tcrv.exec` RVV variant, and target/capability facts. The realized
`tcrv_rvv.with_vl` structure must carry those facts before provider route
construction. RVV route planning/provider code may then consume and mirror the
facts, but Common EmitC, target artifact metadata, route ids, helper names, and
emission-plan mirror fields must not invent them downstream.

## What I already know

* The repository starts clean at commit `466fa08d`, whose predecessor task
  proved explicit and pre-realized selected-dispatch composite generated bundles
  on real `ssh rvv`.
* `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief instead of continuing stale work.
* The Stage2 RVV specs require selected-body realization to consume
  hints/config/profile into real `tcrv_rvv` structure when those facts affect
  generated code.
* The composite selected-body owner already recognizes the three pre-realized
  family bodies, verifies their shared typed facts, and materializes one
  realized `tcrv_rvv.with_vl` body containing `setvl`, loads, splat, compare,
  indexed load, masked MAcc, and indexed store.
* The existing route planning/provider path already validates typed body/config
  facts, selected target capability facts, dispatch/fallback envelope facts,
  runtime ABI order, operand binding closure, and provider-owned route-family
  plans before creating `TCRVEmitCLowerableRoute`.
* Existing low-precision Gearbox/resource facts provide a useful local pattern:
  resource facts are attached to real `tcrv_rvv` structure, route planning
  validates them, and target artifact export treats them as mirrors only.

## Requirements

* Add or harden a shared RVV Gearbox/resource fact carrier for the composite
  selected-body path under `include/TianChenRV/Plugin/RVV` and
  `lib/Plugin/RVV`.
* The composite selected-body realization owner must derive authoritative
  resource facts from typed body/config/runtime facts and selected target
  capability/resource facts, not from route ids, artifact names, helper names,
  scripts, emission-plan metadata, or Common EmitC.
* The realized `tcrv_rvv.with_vl` body must carry resource facts for the
  bounded composite consumer, including:
  * selected candidate/schedule id or equivalent plan identity;
  * SEW/LMUL and VL placement policy;
  * accumulator layout;
  * vector register budget and peak live vector group estimate;
  * unroll factor;
  * pipeline/prefetch intent for this bounded foundation;
  * runtime AVL source and runtime ABI order;
  * selected target capability provider/legality mirrors after capability
    validation;
  * legality/rejection facts for fail-closed diagnostics.
* Provider route planning must consume/validate those realized facts before
  producing the composite `TCRVEmitCLowerableRoute`.
* Route description/metadata and target artifact export may mirror the resource
  facts, but must reject stale/missing/unsupported composite resource facts.
* Keep the implementation plugin-local. Common EmitC must remain a neutral
  materializer and must not derive RVV schedule/resource semantics.
* Keep scope to the runtime-scalar-cmp masked indexed gather-MAcc-scatter
  composite path. Do not expand to unrelated RVV families.

## Acceptance Criteria

* [ ] A production source change under `include/TianChenRV/Plugin/RVV` adds or
  hardens the composite resource/Gearbox fact carrier.
* [ ] A production source change under `lib/Plugin/RVV` derives composite
  resource facts during selected-body realization and carries them on realized
  `tcrv_rvv` structure.
* [ ] Route planning/provider code consumes or validates the realized composite
  resource facts before building the composite `TCRVEmitCLowerableRoute`.
* [ ] Target artifact validation/export treats composite resource facts as
  mirrors of provider/realized facts and rejects stale or missing facts.
* [ ] Focused positive IR/lit evidence shows pre-realized composite selected-body
  realization emits or carries the resource facts and route planning mirrors
  them.
* [ ] At least one fail-closed test covers stale, missing, or unsupported
  composite resource facts before provider route or target artifact acceptance.
* [ ] Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
* [ ] Run `build/bin/tianchenrv-target-artifact-export-test`.
* [ ] Run focused lit for the changed realization/planning/artifact path.
* [ ] Run a bounded old-authority scan over touched files and added diff lines.
* [ ] Run `git diff --check` and `git diff --cached --check`.
* [ ] Finish with clean `git status --short` after one coherent commit if the
  task is complete.

## Non-goals

* No generated-bundle evidence-only closeout.
* No broad smoke matrix or script-only harness expansion.
* No new `ssh rvv` runtime claim unless executable behavior changes.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority or source-front-door positive route.
* No global autotuning database, dashboard, readiness state machine, or profile
  system.
* No dtype/LMUL clone batch.
* No IME, Offload, TensorExt, Template/Toy, or future plugin expansion.
* No Common EmitC scheduling invention.
* No artifact metadata, route id, helper name, or mirror field as schedule or
  compute authority.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-selected-dispatch-composite-executable-bundle-boundary/prd.md`

Primary implementation files:

* `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`
* `lib/Plugin/RVV/RVVGearboxSchedules.cpp`
* `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `include/TianChenRV/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h`
* `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

Focused test/evidence areas:

* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* `test/Target/TargetArtifactExportTest.cpp`

## Definition of Done

The task is complete when the composite selected-body realization path carries
resource-aware Gearbox facts as realized `tcrv_rvv` structure, provider route
planning validates those facts before route construction, target artifact export
mirrors them fail-closed, focused checks pass, task notes are truthful, and one
coherent commit records the source change.
