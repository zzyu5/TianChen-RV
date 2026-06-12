# Common Extension Interface + Lower-To-EmitC Pass Owner

## Goal

Introduce a common C++/MLIR lower-to-EmitC boundary that consumes
`TCRVEmitCLowerableInterface` / `TCRVEmitCLowerableOpInterface` backed route
payloads and materializes production source through the MLIR EmitC Cpp emitter
source authority. This round must rewire at least one real default source route
through the common boundary so the result is not unused infrastructure.

## What I Already Know

* HEAD `b041c6f` removed the legacy route-to-C renderer and left the worktree
  clean.
* The long-term route is extension family ops -> common EmitC lowering ->
  intrinsic/runtime C/C++ -> native compiler.
* `TCRVEmitCLowerableRoute` already verifies bounded headers, ABI mappings,
  provenance, operands, and `emitc.call_opaque` steps.
* `materializeTCRVEmitCLowerableRouteSourceAuthority` and
  `emitTCRVEmitCLowerableRouteAsCppSource` already materialize an MLIR EmitC
  module and translate it through `mlir::emitc::translateToCpp`.
* RVV and scalar production source exporters currently build routes in
  target-local code and call the source authority directly.
* RVV and scalar typed ops already use `TCRVEmitCLowerableOpInterface` for
  bounded source provenance.
* The previous archived task removed the public legacy renderer; this task
  should converge the remaining source-generation ownership boundary rather
  than reintroducing direct C rendering.

## Requirements

* Add or converge a common lower-to-EmitC API, and expose a pass through
  `tcrv-opt` if the common boundary is pass-shaped.
* The common boundary must consume a verified `TCRVEmitCLowerableRoute` built
  from `TCRVEmitCLowerableInterface` and materialize source only through the
  MLIR EmitC Cpp source-authority path.
* Rewire at least one current default production source route, preferably the
  bounded scalar fallback source route or bounded RVV direct microkernel source
  route, so it uses this common boundary instead of invoking local
  materialization/source-authority orchestration directly.
* Keep family-owned route construction local: intrinsic/runtime callee names,
  ABI mapping, selected config, route metadata, and family validation stay in
  RVV/scalar target or plugin code.
* Fail closed when route construction fails, route verification fails,
  generated op-interface provenance is missing for required compute steps, or
  the common source path cannot prove `mlir_emitc_cpp_emitter` authority.
* Do not add descriptor-driven computation, descriptor-to-C export, direct C
  rendering, Python compiler internals, compute semantics in `tcrv.exec`, or
  extension-family semantic branches in core orchestration.

## Acceptance Criteria

* [x] A common C++ boundary exists for lowerable route -> EmitC source-authority
  emission and is used by at least one production/default source exporter.
* [x] The selected production route still emits
  `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`.
* [x] Focused positive lit/C++ coverage proves the common boundary materializes
  EmitC from a real extension-family lowerable route.
* [x] Focused negative coverage proves invalid or non-interface-backed routes
  fail before source emission rather than falling back to old rendering.
* [x] Focused build/test targets for conversion/EmitC, the affected target
  exporter, `tcrv-opt`, and `tcrv-translate` pass as applicable.
* [x] `git diff --check` passes.
* [x] Trellis task validation passes.

## Completion Notes

* Added `TCRVLowerToEmitCSourceAuthority` API as
  `lowerTCRVEmitCLowerableToEmitCSource`, which builds a verified
  `TCRVEmitCLowerableRoute` from `TCRVEmitCLowerableInterface`, emits through
  the MLIR EmitC Cpp source-authority path, and rejects output that does not
  record `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`.
* Rewired scalar fallback runtime-callable source export to use the common
  boundary for production/default scalar i32/i64 add/sub/mul source paths while
  leaving scalar route construction, callee names, ABI mapping, and family
  validation in scalar target code.
* Added C++ positive and negative coverage for the common boundary, including
  fail-closed missing generated op-interface provenance and route verification
  failures.
* Updated scalar artifact and frontend source-route lit checks to assert the
  new common lower-to-EmitC boundary marker while preserving existing
  `mlir_emitc_cpp_emitter` source-authority checks.
* No `ssh rvv` runtime, correctness, or performance claim was made.

## Out of Scope

* No new RVV family, dtype, operation matrix, benchmark, performance claim, or
  generic RVV backend claim.
* No RVV runtime/correctness/performance claim without fresh `ssh rvv`
  evidence.
* No GCC/vendor compiler default route.
* No descriptor-exit cleanup beyond what is required for the selected route.
* No wrapper-only, metadata-only, helper-only, or smoke-only completion.
* No Python implementation of compiler core, dialects, passes, lowering, or
  emission.

## Technical Notes

* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/unified-riscv-mlir.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous context read:
  `.trellis/tasks/archive/2026-05/05-12-emitc-source-authority-owner/prd.md`.
* Primary layers: `core interface`, `conversion / EmitC`,
  `target source authority`, `tests`.
* Possible bounded implementation shape: add a common
  `TCRVEmitCLowerToEmitC` API in the conversion/EmitC library that takes a
  `TCRVEmitCLowerableInterface`, verifies the built route, emits through
  `emitTCRVEmitCLowerableRouteAsCppSource`, and optionally returns the route
  for target-owned metadata comments.
* If a pass is added, it must remain generic and interface-driven; it cannot
  branch on RVV/scalar/IME/offload names to recover family semantics.
