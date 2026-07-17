# Materialize Plugin Variants Pass

## Goal

Add a bounded public `tcrv-opt` pass surface, `--tcrv-materialize-plugin-variants`, that scans existing `tcrv.exec.kernel` anchors, builds generic `TargetCapabilitySet` objects from their direct capabilities, routes proposal collection through `ExtensionPluginRegistry`, and materializes validated plugin proposals as typed `tcrv.exec.variant` operations through the existing C++ materialization API.

## What I Already Know

- The repository is `/home/kingdom/phdworks/TianchenRV`; HEAD starts at `226c140 feat: consume RVV capability properties in plugin decisions`.
- The current worktree was clean before this task was created.
- Existing proposal collection and validation lives in `ExtensionPluginRegistry::collectVariantProposals`.
- Existing variant creation lives in `materializeVariantProposals` / `collectAndMaterializeVariantProposals`.
- `tcrv-opt` already owns deterministic built-in plugin registry wiring at the tool boundary and supports `--tcrv-disable-builtin-plugins`.
- RVV proposal gating and `tcrv_rvv.required_march` metadata must remain plugin-local in `RVVExtensionPlugin`.

## Requirements

- Add a generic C++/MLIR `ModuleOp` pass named `--tcrv-materialize-plugin-variants`.
- The pass must scan existing `tcrv.exec.kernel` operations and use each kernel as the request high-level operation anchor.
- For each kernel, build `TargetCapabilitySet::buildFromKernel(kernel)`, collect proposals through the injected registry, and materialize proposals through the existing materialization API.
- The default pass factory must use an empty registry for embedded negative tests; `tcrv-opt` must register the public pass with the built-in registry unless disabled.
- The pass must not hard-code RVV, scalar, offload, IME, vendor, dtype, shape, or microarchitecture branches.
- Re-running on already materialized variants must not duplicate symbols; it should skip exact existing proposal variants and diagnose mismatched existing direct variants.
- Diagnostics must use MLIR pass failure and generic context while preserving plugin-provided error text.
- No frontend, compute IR, lowering, runtime ABI, object generation, correctness, hardware execution, or performance behavior is in scope.

## Acceptance Criteria

- [ ] Lit coverage exercises `tcrv-opt --tcrv-materialize-plugin-variants` with built-in plugins.
- [ ] Scalar fallback capability materializes `@scalar_fallback_first_slice`.
- [ ] RVV capabilities with preserved properties materialize `@rvv_first_slice` with generic metadata and plugin-owned RVV attributes.
- [ ] A pipeline with at least one downstream pass remains parseable and functional.
- [ ] Deterministic multi-plugin proposal ordering is covered.
- [ ] Empty-registry behavior is covered by lit or C++ test.
- [ ] Missing RVV property evidence does not create an RVV variant while scalar fallback remains independent when available.
- [ ] Re-running the pass does not duplicate variant symbols.
- [ ] Malformed proposal/capability failure surfaces as a useful pass diagnostic.

## Out Of Scope

- Python implementation of compiler decisions.
- New high-level tensor/tile/kernel dialects or compute ops in `tcrv.exec`.
- RVV/IME/offload/Sophgo/AME hard-coded core branches.
- LLVM lowering, RVV intrinsic emission, inline asm, runtime ABI, object generation, benchmarks, correctness claims, or performance claims.
- `ssh rvv` evidence, unless this task unexpectedly makes an RVV runtime/correctness/performance claim.

## Technical Notes

- Relevant specs: variant pipeline, plugin protocol, capability contract, core dialect contract, RVV plugin, scalar fallback plugin, and MLIR testing contract.
- The pass should likely live in `lib/Transforms/VariantMaterialization.cpp` or a small adjacent transform file and be wired through `Passes.td`, `Passes.h`, `lib/Transforms/CMakeLists.txt`, and `tools/tcrv-opt/tcrv-opt.cpp`.
- Tests should prefer lit/FileCheck for public pass behavior; C++ tests are appropriate for empty-registry or injected mock registry behavior if textual lit cannot cover it cleanly.
