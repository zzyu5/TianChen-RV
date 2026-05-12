# RVV Common EmitC Source Boundary Production Owner

## Goal

Rewire the RVV production/default runtime-callable microkernel C source route so RVV target code owns typed family route construction, while the common conversion/EmitC layer owns source-authority emission through `lowerTCRVEmitCLowerableToEmitCSource`.

This is a structural migration round. It must change the real RVV source path, not only add helper code, comments, broad smoke coverage, or metadata.

## What I Already Know

* Current HEAD is `d47e730 feat(emitc): add common lower-to-emitc source boundary`.
* The previous archived task created `lowerTCRVEmitCLowerableToEmitCSource` and rewired scalar fallback production source export through it.
* RVV source export still directly imports and calls `buildTCRVEmitCLowerableRoute`, `emitTCRVEmitCLowerableRouteAsCppSource`, and `TCRVEmitCSourceAuthorityOptions` from `lib/Target/RVV/RVVMicrokernel.cpp`.
* RVV route construction is target-owned and must remain target-owned: selected vector shape, SEW/LMUL/policy, intrinsic names, runtime ABI mapping, callee names, required capabilities, dataflow body validation, header/prototype/harness metadata, and RVV family validation stay in RVV target/plugin code.
* The common conversion/EmitC boundary must own the source-authority emission and must fail closed when the MLIR EmitC Cpp emitter authority marker or interface-backed compute provenance is missing.
* Both runtime-callable library source and explicit self-check harness source must use the same common boundary result for the RVV intrinsic body.

## Requirements

* Replace the production/default RVV body emission path in `lib/Target/RVV/RVVMicrokernel.cpp` so it calls `lowerTCRVEmitCLowerableToEmitCSource` instead of invoking `emitTCRVEmitCLowerableRouteAsCppSource` directly.
* Keep RVV-local route construction in `RVVBinaryEmitCLowerable` or equivalent RVV target-local code. The common layer must not learn RVV family names, intrinsic spelling, vector-shape selection, dataflow semantics, ABI roles, or descriptor mirrors.
* Preserve existing RVV route metadata and artifact comments where true, while adding RVV-specific wording that records the common lower-to-EmitC source-authority boundary.
* Remove obsolete direct RVV imports/usings of direct source-authority APIs once production RVV code no longer calls them.
* Keep the explicit self-check harness as RVV-local wrapper code, but source the callable microkernel body from the same common boundary as the library artifact.
* Update focused FileCheck expectations so RVV generated source requires the common boundary marker while preserving `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` and typed RVV op provenance.
* Do not add descriptor-driven computation, descriptor-to-C export, direct C rendering, new RVV families, dtype matrices, benchmarks, performance claims, or compute semantics in `tcrv.exec`.

## Acceptance Criteria

* [x] RVV runtime-callable source generation calls `lowerTCRVEmitCLowerableToEmitCSource`.
* [x] `lib/Target/RVV/RVVMicrokernel.cpp` no longer directly imports or calls `emitTCRVEmitCLowerableRouteAsCppSource` for production/default microkernel source emission.
* [x] RVV generated source records an RVV-local common lower-to-EmitC boundary marker and still records `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`.
* [x] RVV generated source still records typed RVV op provenance, including `TCRVEmitCLowerableOpInterface` for compute steps.
* [x] Runtime-callable library source and self-check harness mode share the same common boundary output for the RVV callable microkernel body.
* [x] Focused build target succeeds: `cmake --build build --target TianChenRVConversionEmitC TianChenRVRVVTarget tcrv-translate tianchenrv-target-artifact-export-test -j2`.
* [x] Focused lit coverage passes for RVV artifact export and `rvv-microkernel-e2e` source checks.
* [x] Dry-run evidence script passes for `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul --run-id codex-rvv-common-emitc-boundary-dry --overwrite`.
* [x] One focused real `ssh rvv` self-check run is attempted with `python3 scripts/rvv_microkernel_e2e.py --self-check-harness --arithmetic-family=i32-vadd --run-id codex-rvv-common-emitc-boundary-ssh --overwrite`. If it fails, the exact command and failure are recorded and no runtime/correctness claim is made.
* [x] `git diff --check` passes.
* [x] `cmake --build build --target check-tianchenrv -j2` passes unless a precise environment/tool blocker is recorded.
* [x] The archived Trellis task validates with `python3 ./.trellis/scripts/task.py validate <archived-task-dir>`.

## Completion Notes

* Rewired RVV direct microkernel source generation to construct the RVV-local `RVVBinaryEmitCLowerable` route and pass it through `lowerTCRVEmitCLowerableToEmitCSource`.
* Removed direct RVV use of `buildTCRVEmitCLowerableRoute`, `emitTCRVEmitCLowerableRouteAsCppSource`, `TCRVEmitCSourceAuthorityOptions`, and the direct materializer include from `RVVMicrokernel.cpp`.
* Kept RVV semantics local: selected vector shape, intrinsic names, dataflow plan, callable ABI mapping, route metadata, required capabilities, and harness wrapper remain in RVV target/plugin code.
* Added RVV generated-source comments and focused FileCheck expectations for `emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority` while preserving `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter` and typed RVV op-interface provenance.
* Verified dry-run i32-vmul source export and real `ssh rvv` i32-vadd self-check harness. The runtime claim is limited to bounded generated RVV i32-vadd self-check executable correctness only.
* Full `check-tianchenrv` passed with 210/210 tests.

## Out of Scope

* No generic RVV backend maturity claim.
* No new RVV arithmetic family, dtype matrix, benchmark, or performance measurement.
* No descriptor-selected computation, descriptor-to-C export, or fallback local C renderer.
* No moving RVV-specific config, intrinsic selection, ABI, dataflow validation, or family validation into common conversion/EmitC code.
* No Python implementation of compiler core, dialects, passes, lowering, or emission.
* No `tcrv.exec` compute semantics.

## Technical Notes

* Relevant specs read: `.trellis/spec/index.md`, `.trellis/spec/architecture/unified-riscv-mlir.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read: `.trellis/tasks/archive/2026-05/05-12-common-extension-lower-to-emitc-pass-owner/prd.md`.
* Primary implementation files: `include/TianChenRV/Conversion/EmitC/TCRVLowerToEmitC.h`, `lib/Conversion/EmitC/TCRVLowerToEmitC.cpp`, `lib/Target/RVV/RVVMicrokernel.cpp`, focused RVV artifact/export tests, and `scripts/rvv_microkernel_e2e.py` only if the existing evidence parser needs marker updates.
* Expected implementation shape: construct `RVVBinaryEmitCLowerable` locally, call `lowerTCRVEmitCLowerableToEmitCSource` with RVV function/loop options, print RVV comments from the returned route, then append the returned source string. The harness appends wrapper code after the same returned callable source.
