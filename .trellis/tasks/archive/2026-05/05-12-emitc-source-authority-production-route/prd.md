# EmitC source authority production route

## Goal

Make one bounded production/default generated-source route use MLIR EmitC as
the source authority. For this round, the layer is **target source authority**
plus the minimum **conversion / EmitC** support needed for the direct RVV
runtime-callable binary microkernel source path to obtain its generated C/C++
source from a verified MLIR EmitC module translated by the MLIR Cpp emitter,
rather than from the legacy custom route-to-C renderer.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean at task creation.
- Task start HEAD is `95faeea feat(rvv): enforce dispatch selected-plan route authority`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as `.trellis/tasks/05-12-emitc-source-authority-production-route/`.
- The predecessor task
  `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-selected-plan-route-authority/`
  completed the selected component route authority migration and must remain
  archived.
- Specs require the current main lowering route to stay extension family ops
  -> EmitC ops -> C/C++ emitter -> intrinsic/runtime C/C++ -> native compiler.
  Descriptor-driven computation and route-to-C string rendering are transition
  debt, not the production architecture.
- Current inventory shows `TCRVEmitCLowerableRoute` and verified MLIR EmitC
  materialization already exist, but `lib/Target/RVV/RVVMicrokernel.cpp` still
  calls `renderTCRVEmitCLowerableRouteAsCFunction` for the production function
  body after the EmitC verification boundary.

## Requirements

- Rewire the default direct RVV runtime-callable microkernel source export path
  so source bytes for the bounded generated function come from a verified MLIR
  EmitC module translated through `mlir::emitc::translateToCpp`, or an adapter
  that is explicitly backed by that MLIR EmitC source authority.
- Preserve the production route shape:
  typed TCRV RVV family/body authority -> `TCRVEmitCLowerableRoute` -> MLIR
  EmitC module -> MLIR Cpp emitter/source authority -> generated C/C++ source
  -> clang default compiler path.
- Extend conversion / EmitC support only as far as needed for the bounded
  runtime-AVL-to-VL RVV binary route. The verified EmitC module must model the
  function body sufficiently; do not bypass EmitC with a descriptor-to-C or
  route-to-C string renderer.
- Keep RVV-specific intrinsic names, vector type spelling, selected shape, and
  family mapping in RVV target/plugin-owned code or in route payloads produced
  by that code. Common EmitC support must not add RVV family semantic branches.
- Quarantine any remaining custom renderer as legacy compatibility or
  diagnostic fallback only. The default direct RVV source route must not call
  it.
- Source comments and route evidence must identify source authority as MLIR
  EmitC / MLIR Cpp emitter and preserve generated-interface source-op
  provenance where available.
- Preserve parameter layering: hardware/capability facts remain target
  capability data; selected SEW/LMUL/policy/vector shape remains compile-time
  variant config; runtime AVL/VL and mem windows remain IR/ABI values;
  descriptor fields remain legacy mirrors or compatibility checks only.
- Keep clang/LLVM as the default native compiler path. GCC or vendor compilers
  may remain compatibility only and must not become the default.
- Python remains tooling-only.

## Acceptance Criteria

- [x] The production direct RVV runtime-callable source path no longer calls
      `renderTCRVEmitCLowerableRouteAsCFunction` or `RouteCSourceRenderer` as
      its default source authority.
- [x] A conversion / EmitC source authority adapter materializes a verified
      MLIR EmitC module for the bounded runtime-AVL-to-VL route and emits source
      via `mlir::emitc::translateToCpp`.
- [x] Generated source/evidence comments name MLIR EmitC / MLIR Cpp emitter as
      the source authority and still record `TCRVEmitCLowerableOpInterface`
      provenance for the typed compute step.
- [x] The legacy custom renderer, if retained, is renamed or clearly
      quarantined as a diagnostic/compatibility fallback and is not used by the
      production RVV source export.
- [x] Focused C++ tests prove the bounded route materializes to MLIR EmitC and
      emitted source comes from the MLIR EmitC source authority rather than the
      custom renderer.
- [x] Focused target/export or lit coverage proves the direct RVV bounded
      source artifact records MLIR EmitC source authority and preserves
      fail-closed typed-source metadata behavior.
- [x] Focused changed owners build and focused tests pass.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis task
      validation pass.
- [x] `check-tianchenrv -j2` is run if focused checks pass and the build tree is
      usable.
- [x] One coherent commit records the completed module, or the task remains
      open with the exact missing MLIR API/linkage and continuation point.

## Non-Goals

- No broad RVV matrix, benchmark, new dtype, new LMUL, new arithmetic family,
  generic RVV backend claim, MLIR vector route, LLVM scalable-vector route, or
  performance claim.
- No descriptor-to-C exporter and no descriptor-driven computation semantics.
- No moving RVV/scalar/offload semantics into core orchestration passes.
- No Python compiler internals.
- No helper-only, metadata-only, wrapper-only, smoke-only, report-only, or
  evidence-packaging closeout without changing the production/default source
  path.
- No standalone `ssh rvv` evidence round. Runtime/correctness/performance
  claims require a fresh bounded `ssh rvv` execution through the changed path.

## Technical Approach

Add a small MLIR-backed source authority surface in the EmitC conversion
support library. It will consume `TCRVEmitCLowerableRoute`, materialize a
verified EmitC module for the bounded runtime-AVL-to-VL route, and translate
that module through `mlir::emitc::translateToCpp`. The existing direct RVV
target source export will call that authority for the generated function body.

The first bounded body shape may model the RVV runtime-VL progression with
EmitC control flow rather than preserving the old handwritten `while` text.
The important contract is that the emitted source is produced from MLIR EmitC
ops carrying the route's ABI mappings and ordered call-opaque steps, with
op-interface provenance retained.

## Decision (ADR-lite)

**Context**: The repo already verifies `TCRVEmitCLowerableRoute` materializes
to EmitC, but the production RVV source export still uses a custom C renderer
after that verification.

**Decision**: Promote an MLIR EmitC / MLIR Cpp emitter adapter to production
source authority for the bounded direct RVV source route, and demote the
existing renderer to legacy diagnostic compatibility.

**Consequences**: The generated function source may change formatting and
local variable names according to the MLIR Cpp emitter. Tests should assert the
source authority, intrinsic/dataflow content, typed provenance, and route
behavior rather than the old handwritten loop spelling.

## Minimal Validation

- Build focused owners, as applicable:
  `TianChenRVConversionEmitC`, `TianChenRVRVVTarget`,
  `TianChenRVBuiltinTargetArtifactExporters`, `tcrv-translate`, and affected
  C++ test binaries.
- Run focused C++ tests for EmitC lowerable/source authority and target artifact
  export.
- Run focused lit/FileCheck for direct RVV microkernel generated source and any
  changed RVV source/export fixture.
- Run `git diff --check` and `git diff --cached --check`.
- Run
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-emitc-source-authority-production-route`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-selected-plan-route-authority/prd.md`
  - `.trellis/workspace/codex/journal-4.md`
- Initial source focus:
  - `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`
  - `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
  - `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h`
  - `lib/Conversion/EmitC/TCRVEmitCLowerableInterface.cpp`
  - `lib/Conversion/EmitC/CMakeLists.txt`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/RVV/CMakeLists.txt`
  - `lib/Target/TargetArtifactExport.cpp`
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `test/Conversion/EmitC/TCRVEmitCLowerableInterfaceTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - focused RVV source/export lit fixtures under `test/Target/RVV*` and
    `test/Scripts/rvv-microkernel*.test`

## Definition Of Done

This task is complete only when the direct RVV bounded production/default
source route obtains its generated C/C++ function source from MLIR EmitC / MLIR
Cpp emitter authority, stale typed-source metadata still fails closed, focused
tests prove the authority switch, Trellis state is truthful, and one coherent
commit records the module. If unfinished, leave the task open with the exact
missing API/linkage or production-path call site.

## Completion Notes

- Added `TCRVEmitCSourceAuthorityOptions`,
  `materializeTCRVEmitCLowerableRouteSourceAuthority`, and
  `emitTCRVEmitCLowerableRouteAsCppSource`.
- Linked `TianChenRVConversionEmitC` against `MLIRTargetCpp` and used
  `mlir::emitc::translateToCpp` as the source authority for the bounded
  runtime-AVL-to-VL RVV route.
- Rewired `lib/Target/RVV/RVVMicrokernel.cpp` so the direct/default RVV source
  export obtains its function body from verified MLIR EmitC source authority.
- Renamed the old custom renderer API to
  `renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction` with
  `TCRVEmitCLegacyDiagnosticSourceRenderOptions`; scalar compatibility remains
  on that legacy diagnostic path, but the bounded RVV production route no
  longer uses it.
- Updated generated source metadata and FileCheck fixtures to assert
  `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`, static EmitC helper
  functions, `emitc.if`-modeled runtime-VL progression, and
  `TCRVEmitCLowerableOpInterface` source-op provenance.
- No fresh `ssh rvv` execution was performed, so this task makes no RVV
  runtime, correctness, or performance claim.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC -j2`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVTarget TianChenRVScalarTarget -j2`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-emitc-lowerable-interface-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-target-artifact-export-test tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit filter:
  `rvv-microkernel|target-source-artifact-routes|artifact-export-coherence|rvv-probe-to-mlir`
- Broader lit filter:
  `rvv-microkernel|target-source-artifact-routes|artifact-export-coherence|rvv-probe-to-mlir|rvv-scalar|target-artifact-bundle`
- Focused transform lit for the updated LinalgToExec and ExecutionPlanning
  source fixtures.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 209/209 tests.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-emitc-source-authority-production-route`
