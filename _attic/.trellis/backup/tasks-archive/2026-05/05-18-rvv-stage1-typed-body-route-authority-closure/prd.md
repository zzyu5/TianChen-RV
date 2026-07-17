# RVV Stage 1 typed-body route-authority closure

## Goal

Close RVV Stage 1 for the target artifact/export route by making selected
typed `tcrv_rvv` body validation the only authority for RVV object, header,
and bundle artifact candidates. Route ids, artifact names, candidate metadata,
source-front-door residue, arithmetic metadata, helper fixtures, or old i32m1
names may only mirror the validated body; they must not select, synthesize, or
repair RVV emission semantics.

## What I Already Know

- Current repo root is `/home/kingdom/phdworks/TianchenRV`; task start found a
  clean worktree at HEAD `9a29ee9`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
- `.trellis/spec/index.md` keeps the compiler stack in C++/MLIR/LLVM/
  TableGen/CMake/lit/FileCheck. Python remains tooling/evidence only.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines Stage 1 as route
  authority replacement: bounded i32m1 arithmetic, route ids, artifact names,
  source-front-door patterns, descriptor residue, and intrinsic spellings must
  be replaced or fail closed in favor of selected vector-level `tcrv_rvv` body
  plus RVV-owned legality, realization, and route construction.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  target export to consume selected-path, lowering-boundary, emission-plan,
  route, and runtime ABI metadata coherently, while concrete artifact
  generation remains target-owned and backed by explicit extension-family IR
  plus materialized EmitC route.
- The previous archived task demoted the RVV source front door out of default
  artifact pipelines. The RVV source materializer remains explicit-only as a
  typed-body seed.
- Commit `9a29ee9` is the current baseline after source-front-door authority
  demotion.

## Requirements

- Inspect the current production RVV artifact/export path:
  selected `tcrv.exec` RVV variant -> explicit typed `tcrv_rvv` body -> RVV
  plugin route/body validation -> common EmitC/materialized target artifact
  mechanics.
- Keep common target/export code target-neutral. It may consume generic
  interfaces and route-local validation callbacks, but must not learn RVV
  compute semantics.
- Ensure RVV object, header, and bundle export resolve the selected variant,
  require explicit typed `tcrv_rvv` body, and run RVV-owned route/body
  validation before artifact materialization.
- Reject missing or mismatched route metadata, arithmetic metadata, selected
  variant, source-op provenance, runtime ABI parameters, and fallback-only
  selections before object/header/bundle bytes are emitted.
- Keep default RVV source-artifact and source-bundle front doors fail-closed
  for source-only input.
- Preserve positive typed-body object/header/bundle coverage when the selected
  body and runtime ABI contracts are coherent.
- If inspection finds a bypass, remove it or make it fail closed in the same
  round. If no production bypass remains, encode closure as focused
  fail-closed coverage and spec wording only where useful.

## Acceptance Criteria

- [ ] Default RVV source-artifact and source-bundle front doors still reject
      source-only RVV input.
- [ ] RVV object export cannot succeed from route id, artifact name,
      candidate metadata, or arithmetic metadata unless the selected typed
      `tcrv_rvv` body validates through the RVV route builder.
- [ ] RVV header export derives from the same selected materialized object
      candidate and rejects stale/missing ABI role metadata.
- [ ] RVV bundle export validates object and header components as one coherent
      selected-variant group and rejects one-sided or mismatched component
      metadata.
- [ ] Negative coverage proves missing typed body, wrong route id, wrong
      arithmetic mirror, wrong source-op provenance, selected variant mismatch,
      missing runtime ABI role, and fallback-only selection cannot emit RVV
      target artifacts.
- [ ] Positive typed-body object/header/bundle coverage still passes.
- [ ] Targeted scans of RVV plugin, RVV target support, target export,
      `tcrv-translate`, the RVV generated bundle script, and direct RVV tests
      show no descriptor/direct-C/source-export RVV production artifact path.

## Definition Of Done

- Focused build/test evidence for touched RVV plugin/target/export surfaces.
- Focused lit and/or C++ negative coverage for metadata/source/route-id-only
  candidates failing before artifact/header/bundle export.
- Existing positive typed-body object/header/bundle tests pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the exact reason is
  recorded.
- Trellis task status, context, and journal are updated truthfully.
- If complete, the task is archived and one coherent commit is created.

## Out Of Scope

- Stage 2 RVV coverage expansion.
- New SEW/LMUL/dtype families, add/sub/mul/source clone batches, high-level
  Linalg/Vector/StableHLO frontend work, TensorExt/IME/Offload work, tuning
  databases, reports, dashboards, compatibility wrappers, legacy modes,
  Python compiler-core logic, direct C semantic exporters, source-export
  routes, or common/export code that knows RVV compute semantics.
- Fresh `ssh rvv` evidence unless this round changes runtime object/header/
  bundle behavior or refreshes runtime/correctness claims.

## Technical Notes

- Primary inspection targets from the Direction Brief:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/vector-source-target-artifact-exporters.mlir`,
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`,
  and `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`.
- Previous task PRD:
  `.trellis/tasks/archive/2026-05/05-18-rvv-source-front-door-authority-demotion/prd.md`.
- Expected implementation shape is deletion/fail-closed or focused validation
  reinforcement, not a broad audit report or helper-only round.
