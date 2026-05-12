# EmitC Source Authority Owner

## Goal

Close the remaining conversion-layer escape hatch that still exposes the old
route-to-C renderer as a reusable source path. The production generated source
path for bounded RVV, scalar, and RVV+scalar dispatch artifacts already routes
through `TCRVEmitCLowerableRoute` materialization and
`mlir::emitc::translateToCpp`; this round removes the legacy custom C renderer
API from the conversion library so future source work cannot reuse it as an
alternate authority.

## What I Already Know

* Hermes priority 1 is EmitC source authority: typed extension family ops
  should materialize through a common EmitC route and MLIR Cpp emitter source
  authority, not descriptor/direct C export.
* Recent repo evidence shows the main bounded RVV, scalar, and RVV+scalar
  dispatch production source paths already call
  `emitTCRVEmitCLowerableRouteAsCppSource`.
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` still implements
  `LegacyDiagnosticRouteCSourceRenderer`, and the public header still exposes
  `renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction`.
* The only current callers of that renderer are conversion tests. Production
  target code for RVV, scalar fallback, and RVV+scalar dispatch uses the MLIR
  EmitC source authority API.

## Requirements

* Remove the legacy route-to-C renderer from the public conversion/EmitC API.
* Remove the implementation class/function that directly prints C loops from a
  `TCRVEmitCLowerableRoute`.
* Update conversion tests so they assert MLIR EmitC materialization and
  `mlir::emitc::translateToCpp` source authority only.
* Preserve existing public APIs that materialize a `TCRVEmitCLowerableRoute`
  into an EmitC module and emit generated C/C++ source through the MLIR Cpp
  emitter.
* Do not introduce Python compiler logic, descriptor-driven computation, or a
  new family-specific branch in common orchestration.

## Acceptance Criteria

* [x] `rg renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction include lib test`
  has no hits.
* [x] `rg TCRVEmitCLegacyDiagnosticSourceRenderOptions include lib test` has no
  hits.
* [x] `tianchenrv-emitc-lowerable-interface-test` builds and passes.
* [x] Focused target source authority tests still show
  `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`.
* [x] `git diff --check` passes.

## Completion Notes

* Removed `TCRVEmitCLegacyDiagnosticSourceRenderOptions` and
  `renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction` from the public
  conversion/EmitC header.
* Deleted the internal `LegacyDiagnosticRouteCSourceRenderer` implementation.
* Updated `tianchenrv-emitc-lowerable-interface-test` to assert only EmitC
  materialization and MLIR Cpp source-authority behavior.
* Focused lit validation passed for the source-authority artifact routes:
  `target-source-artifact-routes`,
  `scalar-target-source-artifact-routes`,
  `scalar-target-vmul-source-artifact-routes`,
  `rvv-microkernel-family-sub`,
  `rvv-microkernel-family-mul`, and
  `rvv-scalar-i32-vmul-dispatch-generic-route`.

## Out of Scope

* No RVV runtime, correctness, or performance claim.
* No `ssh rvv` validation.
* No new extension family semantics, new compiler stack, or Python-based
  compiler implementation.
* No broad matrix run unless focused checks expose cross-module breakage.

## Technical Notes

* Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Layer classification: `conversion / EmitC` and `target source authority`.
* Current evidence from `.trellis/workspace/codex/journal-4.md` records
  completed RVV, scalar fallback, and RVV+scalar dispatch source-authority
  migrations; this PRD removes the remaining conversion-layer compatibility
  renderer rather than adding another wrapper or helper.
