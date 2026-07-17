# Scalar fallback EmitC source authority production route

## Goal

Rewire the default scalar fallback runtime-callable source artifact route so
the scalar callable function body is produced from a verified MLIR EmitC module
translated by `mlir::emitc::translateToCpp`, using the common
`TCRVEmitCLowerableRoute` source authority surface. This closes the scalar
compatibility gap left after the RVV source-authority migration and makes the
scalar component consumed by RVV+scalar dispatch bundles follow the same
extension family ops -> common EmitC -> C/C++ emitter route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean at task creation.
- Task start HEAD is
  `d5924df feat(rvv): use emitc source authority for microkernel source`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as
  `.trellis/tasks/05-12-scalar-emitc-source-authority-production-route/`.
- The predecessor task
  `.trellis/tasks/archive/2026-05/05-12-emitc-source-authority-production-route/`
  completed the direct RVV source authority migration and must remain archived.
- Specs require scalar fallback callable source production to use the typed
  scalar microkernel op's common `TCRVEmitCLowerableRoute`; descriptors remain
  bounded metadata or compatibility mirrors, not computation authority.
- Current code builds a typed scalar `TCRVEmitCLowerableRoute` and verifies it
  materializes to EmitC, but `lib/Target/Scalar/ScalarMicrokernel.cpp` still
  emits the production scalar callable function body via
  `renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction`.
- The current common MLIR Cpp source-authority materializer is already used by
  RVV production source, but it only accepts the RVV runtime-AVL-to-VL route
  shape. Scalar add/sub/mul routes use a runtime-element-count loop over
  `index` with compute and store `emitc.call_opaque` steps.

## Requirements

- Rewire the default scalar runtime-callable source export so the scalar
  callable function body comes from `emitTCRVEmitCLowerableRouteAsCppSource`
  or the same MLIR EmitC / MLIR Cpp emitter authority surface.
- Extend the common source-authority materializer only as needed for the scalar
  runtime-element-count loop shape: typed scalar route ABI mappings, scalar
  indexed load operands, store address operands, ordered `emitc.call_opaque`
  compute/store steps, and recursive/controlled bounded loop emission.
- Preserve scalar-owned helper definitions such as `tcrv_scalar_i32_add`,
  `tcrv_scalar_i32_sub`, `tcrv_scalar_i32_mul`, and store helpers. The
  production callable body may call those helpers, but it must not be
  reconstructed by the legacy custom route-to-C renderer.
- Generated scalar source metadata must identify MLIR EmitC / MLIR Cpp emitter
  as the source authority and preserve `TCRVEmitCLowerableOpInterface` or typed
  scalar source-op provenance.
- RVV+scalar dispatch source/header/object/bundle paths must consume the
  migrated scalar callable component through the existing scalar artifact route;
  dispatch tests should not depend on legacy scalar loop spelling.
- Keep the legacy renderer only as an explicitly named diagnostic or
  compatibility helper. No default scalar or RVV production source call site may
  use it.
- Keep common code generic. Do not add scalar/RVV family semantic branches in
  core orchestration; any new common source-authority support must be expressed
  as route-shape handling through the existing route payload.
- Preserve IR-backed callable ABI layering: `lhs`, `rhs`, `out` come from
  `tcrv.exec.mem_window`; runtime `n` comes from direct
  `tcrv.exec.runtime_param`; emission-plan parameter metadata remains a
  validated mirror.
- Python remains tooling-only; no compiler core, dialect, lowering, emission,
  or plugin registry implementation in Python.

## Acceptance Criteria

- [x] `lib/Target/Scalar/ScalarMicrokernel.cpp` default source export no longer
      calls `renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction` for the
      production scalar callable body.
- [x] The common MLIR Cpp source-authority materializer supports the scalar
      runtime-element-count route shape without family-specific semantic
      branching.
- [x] Generated scalar source records
      `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` and source metadata
      says the body is emitted by `mlir::emitc::translateToCpp`.
- [x] Scalar source preserves generated-interface source-op provenance for the
      typed scalar compute step and fails closed when required provenance is
      missing.
- [x] RVV+scalar dispatch source/bundle fixtures consume the migrated scalar
      component and no longer assert legacy scalar renderer wording.
- [x] Focused C++ and lit/FileCheck coverage proves scalar source authority,
      target artifact export, and dispatch bundle coherence.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis task
      validation pass.
- [x] `check-tianchenrv -j2` is run if focused checks pass and the existing
      build tree is usable.
- [x] One coherent commit records the completed module, or the task remains
      open with the exact remaining production call site, materializer gap,
      failing fixture, or linker issue.

## Non-Goals

- No broad family matrix, benchmark, performance claim, or standalone
  `ssh rvv` evidence round.
- No descriptor-to-C exporter and no descriptor-driven computation semantics.
- No moving scalar, RVV, dispatch, or other extension semantics into core
  orchestration passes.
- No rewriting the RVV direct source authority except for small common API
  compatibility required by the source-authority materializer.
- No helper-only, metadata-only, wrapper-only, smoke-only, report-only, or
  evidence-packaging closeout without changing the scalar default production
  source path.
- No preserving tests whose only purpose is to require the old legacy scalar
  loop spelling, unless explicitly quarantined as legacy diagnostic renderer
  tests.

## Technical Approach

Extend `RouteCppSourceAuthorityMaterializer` so it can materialize two bounded
generic loop shapes from the same `TCRVEmitCLowerableRoute` payload:

- the existing RVV runtime-AVL-to-VL recursive helper shape;
- the scalar runtime-element-count recursive helper shape, where the helper
  guards on `index < n`, materializes route-authored indexed loads and store
  addresses, emits the ordered `emitc.call_opaque` compute/store sequence, and
  recurs with `index + 1`.

Then switch `ScalarMicrokernel.cpp` to call
`emitTCRVEmitCLowerableRouteAsCppSource` for the production scalar callable
body, update scalar metadata comments, and revise scalar/dispatch fixtures to
assert source authority and semantic route evidence rather than legacy loop
formatting.

## Decision (ADR-lite)

**Context**: Scalar already has typed family/body authority and builds a common
EmitC route, but production source output still goes through the legacy
diagnostic C renderer.

**Decision**: Promote the MLIR EmitC / MLIR Cpp emitter adapter to production
source authority for scalar fallback callable source and keep the old renderer
only for diagnostic compatibility tests.

**Consequences**: Generated scalar source formatting and local helper shape may
change according to the MLIR Cpp emitter. Tests should assert source authority,
callable ABI provenance, route-authored callee sequence, and dispatch component
coherence instead of handwritten loop spelling.

## Minimal Validation

- Build focused owners:
  `TianChenRVConversionEmitC`, `TianChenRVScalarTarget`,
  `TianChenRVRVVScalarDispatchTarget` if present,
  `tianchenrv-target-artifact-export-test`, and `tcrv-translate`.
- Run focused C++ tests for EmitC lowerable/source authority and target
  artifact export.
- Run focused lit/FileCheck for scalar microkernel source export, RVV+scalar
  dispatch source/bundle export, target artifact bundle export, and changed
  LinalgToExec / ExecutionPlanning fixtures.
- Run `git diff --check` and `git diff --cached --check`.
- Run
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-scalar-emitc-source-authority-production-route`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-12-emitc-source-authority-production-route/prd.md`
- Initial source focus:
  - `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`
  - `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `test/Conversion/EmitC/TCRVEmitCLowerableInterfaceTest.cpp`
  - scalar, RVVScalarDispatch, TargetArtifactBundleExport, and script fixtures
    that assert scalar source authority or scalar fallback source text.

## Definition Of Done

This task is complete only when scalar production/default source output uses
MLIR EmitC / MLIR Cpp emitter authority, default production call sites no
longer use the legacy diagnostic renderer, RVV+scalar dispatch consumes the
migrated scalar component, focused checks pass, Trellis state is truthful, and
one coherent commit records the module. If unfinished, leave the task open with
the exact missing API/linkage, production call site, or failing fixture.

## Completion Notes

- Extended `RouteCppSourceAuthorityMaterializer` with a generic
  runtime-element-count loop shape for routes whose first step is not
  `runtime-avl-to-vl`. The new shape materializes indexed scalar loads through
  `emitc.subscript` + `emitc.load`, store addresses through `emitc.apply "&"`,
  route-authored `emitc.call_opaque` compute/store steps, and recursive
  `emitc.call` loop progression by `index + 1`.
- Kept the existing RVV runtime-AVL-to-VL source-authority route shape intact.
  The common materializer selects by route step role and does not branch on
  scalar/RVV family names.
- Rewired `lib/Target/Scalar/ScalarMicrokernel.cpp` so scalar production source
  emits the callable body through `emitTCRVEmitCLowerableRouteAsCppSource`.
  The legacy route-to-C renderer remains only under its explicit diagnostic
  helper API and is no longer called by scalar or RVV production target code.
- Updated scalar, RVV+scalar dispatch, target artifact bundle, LinalgToExec,
  ExecutionPlanning, and e2e script fixtures to assert MLIR EmitC / MLIR Cpp
  emitter authority and typed scalar source-op provenance instead of legacy
  handwritten scalar loop spelling.
- No fresh `ssh rvv` execution was performed, so this task makes no RVV
  runtime, correctness, or performance claim.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC TianChenRVScalarTarget -j2`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-emitc-lowerable-interface-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC TianChenRVScalarTarget TianChenRVBuiltinTargetArtifactExporters tianchenrv-target-artifact-export-test tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit:
  `Target/ArtifactExport/scalar-target-source-artifact-routes.test`,
  `Target/ArtifactExport/scalar-target-vmul-source-artifact-routes.test`
- Focused lit:
  `Target/ArtifactExport/target-source-artifact-routes.test`,
  `Target/RVVScalarDispatch`, `Target/TargetArtifactBundleExport`,
  `Transforms/LinalgToExec`, `Transforms/ExecutionPlanning`
- Focused lit:
  `Scripts/rvv-scalar-dispatch-e2e.test`,
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 209/209 tests.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-scalar-emitc-source-authority-production-route`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-12-scalar-emitc-source-authority-production-route`
