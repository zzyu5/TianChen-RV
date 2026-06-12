# Delete Runtime-Callable Source Bundle Residue

## Goal

Remove the remaining runtime-callable C source bundle residue from the target
artifact export layer after deletion of the generic source artifact front door.
The target artifact bundle/export infrastructure must no longer preserve direct
C source semantics from selected metadata, `directHelperRoute` records, or
synthetic source-route unit fixtures.

## Why Now

The previous round removed `tcrv-export-target-source-artifact`, but live target
artifact code/tests still mention `runtime-callable-c-source` as a bundle or
component artifact kind, keep direct-helper source-route handling, and synthesize
source-route records in tests. That leaves a second selected-metadata-to-C
artifact path after the front door deletion.

## Scope

- Target artifact bundle record collection, validation, manifest serialization,
  and deterministic bundle file naming.
- Direct-helper source-route handling, especially records that preserve source
  component semantics for bundle packaging.
- C++ unit tests and lit fixtures that construct, validate, or accept
  `runtime-callable-c-source` components, synthetic source composites, no-op
  source routes, or deleted source-route names.
- Focused evidence scans and target artifact export checks affected by this
  deletion.

## Non-Goals

- Do not implement a replacement EmitC route.
- Do not add a new artifact format, compatibility mode, wrapper, quarantine, or
  descriptor fallback.
- Do not add new RVV lowering, dispatch source generation, runtime execution,
  or broad cleanup outside the target artifact bundle/export residue.
- Do not restore direct C source routes to make checks pass.
- Do not treat helper/report/guardrail-only changes as completion.

## Requirements

- Delete or fail-close any target artifact bundle path that treats
  `runtime-callable-c-source` as a valid selected-path semantic artifact.
- Remove direct-helper source-route preservation from active bundle/export
  collection and manifest records.
- Delete or rewrite tests whose only purpose is to protect synthetic source
  composite routes, no-op source routes, or source bundle component records.
- Keep remaining artifact packaging constrained to object/header/metadata
  records, or fail closed until a future materialized MLIR EmitC module route
  exists.
- Preserve target-neutral generic routing boundaries: no family-specific core
  branches and no descriptor-to-C computation semantics.
- If a build or focused test fails because the new EmitC architecture is missing,
  report it as a deletion gap rather than restoring source routes.

## Acceptance Criteria

- [x] Focused scan shows no active target artifact code path preserving
      `runtime-callable-c-source` source components as supported selected-path
      bundle artifacts.
- [x] Focused scan shows no active `directHelperRoute` source record handling
      remains in target artifact bundle/export infrastructure.
- [x] Unit/lit tests no longer protect synthetic source composite routes,
      no-op source routes, or deleted source-route names as positive behavior.
- [x] Affected target artifact export C++/lit checks are run and either pass or
      fail only with documented missing new-architecture gaps.
- [x] Trellis task status, context, journal, archive state, and commit are
      updated truthfully for the completed round.

## Technical Context

- Relevant spec root: `.trellis/spec/index.md`
- Lowering/runtime contract:
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- Stack boundary:
  `.trellis/spec/implementation-stack/compiler-stack-contract.md`
- Test policy:
  `.trellis/spec/testing/mlir-testing-contract.md`
- Code starting points from the direction brief:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/EmissionManifest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/ArtifactExport/`

## Evidence Plan

- Search target artifact code/tests for:
  `runtime-callable-c-source`, `directHelperRoute`, source bundle component
  records, synthetic source composites, no-op source route names, and deleted
  source-route names.
- Build and run focused target artifact export checks affected by the deletion.
- Re-run focused checks after self-repair.

## Completion Notes

- Removed `directHelperRoute` from target artifact exporter APIs, bundle records,
  and emission manifest artifact records.
- Runtime-callable/standalone C source artifact kinds now fail closed at target
  artifact exporter registration and bundle component validation.
- Rewrote target artifact bundle discovery/file-name/component-contract tests
  to use metadata/header/object positive coverage; source appears only as
  deletion-state negative coverage.
- Focused checks run:
  `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVBuiltinTargetArtifactExporters tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`,
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`,
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport`
  from `artifacts/tmp/tianchenrv-build/test`, and `git diff --check`.
- Self-repair: fixed one C++ test compile error caused by copying
  `SmallVector<T, 2>` directly into `SmallVector<T, 3>`.
