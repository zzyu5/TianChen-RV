# RVV source-front-door authority demotion

## Goal

Demote the legacy RVV vector source front door so it cannot be used by the
production/default source-artifact export flow as RVV route or artifact
authority. The RVV source adapter may remain available as an explicit
plugin-local typed-body materialization seed, but default artifact export must
depend on an already materialized selected typed `tcrv_rvv` body, RVV-owned
route construction, and the common EmitC/target export mechanics.

## What I Already Know

- Current repo root is `/home/kingdom/phdworks/TianchenRV`; current HEAD is
  `1df61bf` and the worktree was clean at task start.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
- `.trellis/spec/index.md` keeps the implementation stack in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck, with Python limited to tooling.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines RVV Stage 1 as route
  authority replacement: bounded i32m1/source/artifact/route metadata must be
  replaced or fail-closed in favor of selected vector-level typed `tcrv_rvv`
  body plus RVV-owned legality, realization, and route construction.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` allows target
  artifact export only from explicit extension-family IR plus materialized
  EmitC routes, and requires source-only/front-door residue to fail closed.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` already keeps
  source-front-door interpretation plugin-owned and common orchestration
  target-neutral. This task needs a generic registration-level way to keep
  explicit source materializer passes public while excluding selected legacy
  passes from default artifact front-door pipelines.
- Commit `1df61bf` already made RVV object/header/bundle target artifact
  validation rebuild the RVV EmitC route from the selected typed body before
  accepting candidate metadata.
- Current `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` materializes typed
  `tcrv_rvv` body from one tail-safe source pattern and rejects stale
  `tcrv_rvv.lowering_seed` and pre-existing `tcrv.exec`/`tcrv_rvv` residue.
- The remaining default-route authority surface is that
  `--tcrv-source-artifact-front-door-pipeline` and
  `--tcrv-source-artifact-bundle-front-door` collect and run the RVV source
  materializer, so source-only RVV MLIR can still reach object/header/bundle
  export through a default front door.

## Requirements

- Add a target-neutral source-front-door registration property that states
  whether a pass may participate in default source-artifact export pipelines.
- Keep common pipeline logic generic: it may filter registrations by that
  property, but must not branch on RVV, TensorExtLite, Toy, dtype, route id,
  artifact name, source op, or arithmetic kind.
- Mark the RVV vector source materializer as an explicit non-default typed-body
  seed. Its direct pass option should remain available for focused
  materialization and emission-plan tests.
- Keep Toy and TensorExtLite source-front-door behavior unchanged unless a
  failure reveals a generic registry issue.
- Make source-only RVV input through default source-artifact pipelines fail
  closed before target artifact export.
- Preserve positive RVV object/header/bundle coverage from already
  materialized typed `tcrv_rvv` selected-body IR.
- Preserve existing selected typed-body candidate validation: stale route
  metadata, stale arithmetic metadata, route id edits, artifact-name tricks,
  metadata-only candidates, and missing selected typed body must not bypass the
  selected-body route builder.

## Acceptance Criteria

- [ ] `SourceFrontDoorPassRegistration` exposes a generic default artifact
      front-door eligibility flag with a safe default for existing plugins.
- [ ] `buildSourceArtifactFrontDoorPipeline` only runs registrations eligible
      for default artifact front-door use.
- [ ] RVV registers
      `tcrv-rvv-materialize-i32m1-vector-source-front-door` as public but not
      default-artifact-front-door eligible.
- [ ] `tcrv-opt --tcrv-rvv-materialize-i32m1-vector-source-front-door` still
      materializes the typed selected RVV body, and explicit chaining to
      `--tcrv-materialize-emission-plans` still produces selected-body-derived
      emission diagnostics.
- [ ] `tcrv-opt --tcrv-source-artifact-front-door-pipeline` on source-only RVV
      input fails closed instead of producing an RVV selected artifact
      candidate.
- [ ] `tcrv-translate --tcrv-source-artifact-bundle-front-door` on source-only
      RVV input fails closed and does not emit complete bundle output.
- [ ] Existing typed RVV IR target artifact tests still export object, header,
      and bundle.
- [ ] C++ registry/plugin tests cover the RVV registration demotion and default
      eligibility behavior.

## Definition Of Done

- Focused C++ and lit checks covering source-front-door registry behavior, RVV
  explicit materialization, RVV source-only default fail-closed behavior, and
  typed selected-body artifact export pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; if not, the exact reason is recorded.
- Trellis task status and journal are updated truthfully.
- One coherent commit is created if the task is complete.

## Technical Approach

Extend the existing source-front-door pass registration surface with a generic
boolean that defaults to default artifact-front-door eligibility. The generic
source-artifact pipeline will run only eligible registrations. RVV will set the
flag to false because its source pass is only a non-default typed-body seed
during Stage 1. This removes the default source-to-artifact RVV lane without
removing the explicit RVV pass or adding a concrete RVV branch in common
orchestration.

## Decision

Context: RVV source materialization can be useful as an explicit bounded seed,
but default artifact/export flows must not treat source pattern matching as
RVV route authority during Stage 1.

Decision: Keep the RVV pass registered as a public explicit pass and add a
generic registration property that excludes it from default source-artifact
front-door pipelines.

Consequences: Already-materialized selected `tcrv_rvv` IR remains the positive
artifact path. RVV source-only inputs now fail closed in default source-artifact
flows. Other plugin source front doors keep their current default behavior via
the safe default unless they opt out later.

## Out Of Scope

- Stage 2 RVV coverage expansion, dtype/LMUL clone batches, new source
  patterns, high-level Linalg/Vector frontend lowering, generic source-to-RVV
  compilation, global tuning/profile systems, dashboards, scalar/IME/Offload
  work, descriptor/direct-C/source-export restoration, and RVV runtime or
  performance claims.
- Fresh `ssh rvv` evidence. This task changes compiler route authority and
  default front-door eligibility, not emitted runtime semantics.

## Technical Notes

- Primary files expected to change:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, targeted tests, and possibly
  relevant `.trellis/spec/` contract text if the registration interface is
  updated.
- Positive typed-body artifact coverage already exists in
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.
- Existing source-to-artifact positive RVV tests should be rewritten into
  explicit materialization or fail-closed coverage; they must not keep the
  old default source authority alive.
