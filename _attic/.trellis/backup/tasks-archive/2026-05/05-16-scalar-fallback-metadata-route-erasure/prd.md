# Scalar fallback metadata-route erasure

## Goal

Erase the legacy scalar fallback metadata-only route from active TianChen-RV
code, tests, and directly related specs. Scalar fallback may remain as a
generic capability-driven fallback proposal and dispatch/fallback envelope, but
it must no longer materialize a plugin-owned metadata lowering boundary,
metadata-only emission/runtime ABI route, target artifact bundle surface, or
selected executable/ABI substitute.

## Context

- This is a Wrong Logic Deletion Campaign round.
- The previous clean HEAD is `6753a65 chore(rvv): erase no-body probe/bundle fixture authority`.
- The current repository has no pre-existing Trellis current task; this task
  was created from the Hermes brief before source edits.
- Active code currently materializes `tcrv_scalar.lowering_boundary` in
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` and emits scalar metadata-only
  route fields through `buildVariantEmissionPlan`.
- Active target-support registration currently exposes scalar bundle/boundary
  metadata in `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
- Active tests currently protect `portable-scalar-fallback-metadata-route`,
  `host-scalar-fallback-metadata`,
  `metadata-only-host-fallback-boundary`,
  `portable_scalar_fallback_first_slice`, and/or
  `tcrv_scalar.lowering_boundary` as selected-path metadata authority.

## Requirements

- Remove scalar fallback selected-boundary materialization as an active
  compiler route.
- Remove scalar fallback metadata-only emission/readiness/runtime ABI fields as
  an active plugin-owned route.
- Remove scalar bundle/export registration that only exists to preserve the
  metadata-only selected-boundary/export surface.
- Delete or rewrite tests that assert scalar fallback metadata-only boundary,
  metadata-only emission, runtime ABI, target artifact bundle, or probe replay
  route authority.
- Keep only generic scalar fallback capability/proposal/legality/selection
  behavior when it is not used as scalar compute, metadata emission, executable
  route, runtime ABI, or artifact authority.
- If deletion exposes missing new-architecture gaps, report them as gaps
  without restoring the metadata route.
- Update directly conflicting Trellis specs/comments when they instruct future
  agents to keep the deleted metadata-only route.

## Non-Goals

- No new scalar fallback implementation.
- No Common EmitC route.
- No RVV rebuild.
- No new runtime ABI.
- No compatibility wrapper, replacement lowering boundary, replacement
  artifact route, or metadata route.
- No unrelated descriptor, RVV legality, target bundle, or extension rebuild
  work unless directly required to erase the scalar metadata route.

## Acceptance Criteria

- Active code/tests no longer protect
  `portable-scalar-fallback-metadata-route`.
- Active code/tests no longer protect
  `host-scalar-fallback-metadata`.
- Active code/tests no longer protect
  `metadata-only-host-fallback-boundary`.
- Active selected-boundary tests no longer protect
  `tcrv_scalar.lowering_boundary` as a metadata-only scalar fallback route.
- Scalar fallback is not treated as a selected executable/ABI route or artifact
  authority.
- Focused active-surface ref scan is run for:
  `portable-scalar-fallback-metadata-route`,
  `host-scalar-fallback-metadata`,
  `metadata-only-host-fallback-boundary`,
  `portable_scalar_fallback_first_slice`,
  `scalar_fallback_first_slice`, `scalar-plugin`, and
  `tcrv_scalar.lowering_boundary`, excluding `.git`, `build`,
  `.trellis/tasks/archive`, `.trellis/workspace`, `artifacts/tmp`, and run
  artifacts.
- Focused build includes at least `ninja -C build tcrv-opt tcrv-translate`.
- Focused lit/FileCheck coverage is run for affected scalar dialect, plugin
  legality, lowering-boundary/emission-readiness, probe replay, and target
  bundle tests that remain.
- `ninja -C build check-tianchenrv` is attempted or explicitly reported with
  the reason it could not be completed.
- `git diff --check` and Trellis task validation pass.
- The task is finished/archived according to Trellis convention and committed
  as one coherent deletion/refactor commit if complete.

## Technical Notes

- Read specs: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant code surfaces include:
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h`,
  `lib/Dialect/Scalar/IR/ScalarDialect.cpp`,
  `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`,
  and `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
- Relevant initial tests include:
  `test/Dialect/Scalar/lowering-boundary.mlir`,
  `test/Transforms/PluginVariantLegality/plugin-variant-legality-pass.mlir`,
  `test/Transforms/PluginVariantLegality/plugin-variant-legality-empty-registry.mlir`,
  and `test/Scripts/rvv-probe-to-mlir.test`.
