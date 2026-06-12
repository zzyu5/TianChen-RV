# Delete common EmitC source-authority C++ exporter residue

## Goal

Remove the common Conversion/EmitC source-authority surface so that shared EmitC lowering keeps only extension-agnostic in-memory MLIR materialization. The common boundary must no longer expose or test a C++ source-export path that treats route metadata, route kind, RVV intrinsic names, or runtime-avl-to-vl control as authoritative C++ source semantics.

## What I already know

* Current repo root is `/home/kingdom/phdworks/TianchenRV`, branch `main`, with one unrelated untracked `artifacts/` tree already present.
* There is no pre-existing `.trellis/.current-task`; this task was created from the Hermes brief.
* The current emission-runtime spec already frames direct C semantic exporter routes as deleted or fail-closed until a future materialized EmitC route exists.
* `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h` still declares `TCRVEmitCSourceAuthorityOptions`, `materializeTCRVEmitCLowerableRouteSourceAuthority`, and `emitTCRVEmitCLowerableRouteAsCppSource`.
* `include/TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h` and `lib/Conversion/EmitC/TCRVLowerToEmitC.cpp` still expose a common `lowerTCRVEmitCLowerableToEmitCSource` wrapper/result API.
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` still contains a large `RouteCppSourceAuthorityMaterializer` implementation that validates runtime-avl-to-vl shape, emits MLIR Cpp source, and records a `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` marker.
* `test/Conversion/EmitC/TCRVEmitCLowerableInterfaceTest.cpp` still asserts RVV intrinsic names, `tcrv_rvv.*` op names, runtime-avl-to-vl materialization, source-authority source text, and the common lower-to-EmitC source wrapper.
* `test/CMakeLists.txt` still links the EmitC common test against the RVV dialect even though the target should be extension-agnostic after this deletion.

## Assumptions

* The shared, allowed EmitC path is still the in-memory MLIR materialization route that does not act as a public C++ source-export authority.
* No replacement source exporter, plugin ABI provider, or new RVV lowering should be introduced in this round.
* If removal of the deleted path exposes a missing user-facing bridge, that gap should be reported rather than reimplemented here.

## Requirements

* Delete the public common EmitC source-authority API surface from headers and implementation.
* Delete the wrapper source/result API for `lowerTCRVEmitCLowerableToEmitCSource`.
* Remove the `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` marker path and the Cpp emitter translation path from common conversion code.
* Rewrite the common EmitC test to cover only extension-agnostic route verification and in-memory EmitC materialization.
* Remove RVV-specific intrinsic-name, `tcrv_rvv.*`, runtime-avl-to-vl, and public-wrapper source assertions from the common test surface.
* Remove build references to deleted source-authority source files and any unnecessary RVV dialect dependency in the common EmitC test target.

## Acceptance Criteria

* [x] `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h` no longer exposes source-authority options or source-export declarations.
* [x] `include/TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h` and `lib/Conversion/EmitC/TCRVLowerToEmitC.cpp` are removed or otherwise no longer part of the common conversion build.
* [x] `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` no longer contains the source-authority materializer class, the MLIR Cpp source emission wrapper, or the `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` marker.
* [x] The focused ref-scan over `include/TianChenRV/Conversion/EmitC`, `lib/Conversion/EmitC`, `test/Conversion/EmitC`, and `test/CMakeLists.txt` finds no remaining public common source-authority API or test authority for RVV intrinsic names, runtime-avl-to-vl loops, or public C source wrappers.
* [x] The common EmitC test still verifies generic route construction and in-memory EmitC materialization, but does not assert C++ source export behavior as common authority.
* [x] The focused common EmitC build/test target passes, or any remaining failures are reported as expected gaps caused by deleting the old source-authority path.
* [x] `git diff --check` passes.

## Definition of Done

* Source-authority declarations, definitions, wrappers, build references, and tests are removed.
* The remaining common EmitC surface is extension-agnostic and stays within in-memory materialization boundaries.
* The task is validated with focused build/test and diff checks.
* The task is archived and the workspace journal is updated when done.

## Out of Scope

* No replacement plugin ABI provider.
* No new RVV lowering or RVV-specific common source exporter.
* No new C/C++ exporter, compatibility wrapper, legacy mode, quarantine, or descriptor path.
* No broad documentation cleanup outside the immediate boundary touched by the deletion.
* No ssh RVV evidence, runtime/performance evidence, or broad test matrix.

## Technical Notes

* Read before implementation:
  * `.trellis/spec/index.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/tasks/archive/2026-05/05-15-delete-support-i32-rvv-runtime-abi/prd.md`
  * `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`
  * `include/TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h`
  * `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
  * `lib/Conversion/EmitC/TCRVLowerToEmitC.cpp`
  * `test/Conversion/EmitC/TCRVEmitCLowerableInterfaceTest.cpp`
  * `test/CMakeLists.txt`
* Focused validation should include a ref-scan for:
  * `TCRVEmitCSourceAuthorityOptions`
  * `materializeTCRVEmitCLowerableRouteSourceAuthority`
  * `emitTCRVEmitCLowerableRouteAsCppSource`
  * `TCRVLowerToEmitCSourceOptions`
  * `TCRVLowerToEmitCSourceResult`
  * `lowerTCRVEmitCLowerableToEmitCSource`
  * `tcrv_emitc.source_authority`
  * `runtime-avl-to-vl`
  * `__riscv_vadd_vv_i32m1`
* Remaining in-memory EmitC materialization helpers are acceptable only if they do not reintroduce C++ source-export authority.

## Validation Notes

* [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-emitc-lowerable-interface-test -j2`
* [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
* [OK] Focused ref-scan over `include/TianChenRV/Conversion/EmitC`, `lib/Conversion/EmitC`, `test/Conversion/EmitC`, `lib/Conversion/EmitC/CMakeLists.txt`, and `test/CMakeLists.txt` for deleted source-authority symbols, marker strings, runtime-avl-to-vl, common-test RVV intrinsic names, `riscv_vector`, and `tcrv_rvv`
* [OK] `git diff --check`
* [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-delete-common-emitc-source-authority`
* Deletion gap: common Conversion/EmitC no longer has a public C++ source bridge from route metadata; future source output requires a new explicit extension-family IR plus materialized EmitC route contract.
