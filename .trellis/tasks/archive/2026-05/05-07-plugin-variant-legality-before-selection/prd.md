# Verify Plugin Variant Legality Before Selection

## Goal

Add a target-neutral C++/MLIR pass that verifies each materialized direct
`tcrv.exec.variant` through its origin extension plugin before variant
selection, dispatch synthesis, selected lowering-boundary materialization, or
emission-plan diagnostics.

## What I Already Know

* Existing registry APIs already expose `VariantLegalityRequest`,
  `ExtensionPlugin::verifyVariantLegality`, and
  `ExtensionPluginRegistry::verifyKernelVariantLegality`.
* Existing RVV and scalar plugins own their plugin-local legality logic.
* The missing compiler surface is a public MLIR pass and execution-planning
  pipeline integration between generic capability-requires checking and variant
  selection.
* Public `tcrv-opt` pass construction must inject the deterministic built-in
  plugin registry at the tool boundary. Default pass construction must retain
  an empty registry for embedded/negative tests.

## Requirements

* Add `tcrv-verify-plugin-variant-legality` as a C++/MLIR module pass.
* The pass must build `TargetCapabilitySet` from each direct
  `tcrv.exec.kernel` capability set and verify direct child variants in IR
  order through `ExtensionPluginRegistry`.
* Diagnostics must be generic and include kernel symbol, variant symbol, origin
  plugin, and plugin-provided legality failure text.
* The pass must not hard-code RVV, scalar, IME, Sophgo, AME, offload, vendor,
  dtype, shape, runtime, microarchitecture, or probe-evidence semantics in core
  transforms.
* Integrate the pass into `tcrv-execution-planning-pipeline` after
  `tcrv-check-capability-requires` and before `tcrv-select-variants`.
* Preserve existing `tcrv-check-capability-requires` behavior.
* Keep RVV legality in RVV plugin code and scalar fallback legality in scalar
  plugin code.
* Do not implement lowering, executable emission, runtime ABI, correctness
  runs, or performance work in this round.

## Acceptance Criteria

* Public `tcrv-opt` can run `--tcrv-verify-plugin-variant-legality` with the
  built-in registry.
* Default/empty registry pass construction diagnoses unknown origins.
* Valid RVV replay capabilities plus scalar fallback continue through the full
  execution-planning pipeline to selected lowering-boundary and emission-plan
  diagnostics.
* Valid scalar fallback-only IR passes legality and continues through existing
  scalar fallback planning.
* Unknown origins, missing RVV typed policy metadata, mismatched
  `tcrv_rvv.required_march`, missing structured `requires`, and missing RVV
  capability reject before selection and before lowering/emission surfaces are
  materialized.
* Invalid selected RVV paths are not silently ignored when scalar fallback is
  available.
* Existing lit and C++ tests continue to pass.

## Out Of Scope

* No Python compiler-internal implementation.
* No new compute op in `tcrv.exec`.
* No executable lowering, LLVM/RISC-V/RVV intrinsic emission, inline asm,
  runtime ABI glue, generated objects, benchmarks, correctness claims, or
  performance claims.
* No new `ssh rvv` evidence claim.

## Technical Notes

* Primary implementation files: `include/TianChenRV/Transforms/Passes.td`,
  `include/TianChenRV/Transforms/Passes.h`,
  `lib/Transforms/CMakeLists.txt`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  new `lib/Transforms/PluginVariantLegality.cpp`, and `tools/tcrv-opt/tcrv-opt.cpp`.
* Required tests live under `test/Transforms`; C++ registry/plugin legality
  smoke coverage already exists under `test/Plugin`.
