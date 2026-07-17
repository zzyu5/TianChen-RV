# RVV EmitC to C/C++ emitter handoff

## Goal

Register a bounded RVV-owned `tcrv-translate` route that accepts an already
materialized MLIR EmitC module and hands it to the upstream MLIR EmitC C/C++
emitter. This completes the immediate post-EmitC handoff for the current RVV
i32m1 first slice without claiming runtime ABI completion, artifact packaging,
native compilation, `ssh rvv` execution, correctness, or performance.

## What I already know

- The current repo has no `.trellis/.current-task` at session start; this task
  was created from the Hermes Direction Brief.
- Current HEAD `a2f3d9e` already materializes explicit RVV i32m1 add
  extension-family ops into a parseable MLIR EmitC module.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` currently registers no target
  artifact exporters and no target translate routes.
- `tools/tcrv-translate/tcrv-translate.cpp` already converts registered
  `TargetTranslateRoute` entries into MLIR translation command-line routes.
- The upstream MLIR EmitC emitter API is
  `mlir::emitc::translateToCpp(Operation *, raw_ostream &, bool)`.

## Requirements

- Add one RVV-owned target translate route through the existing
  `TargetTranslateRouteRegistry`.
- The route must accept only an already materialized EmitC module: `builtin`
  module container plus `emitc` operations.
- The route must fail closed with a clear diagnostic on non-EmitC or
  non-materialized `tcrv.exec` / `tcrv_rvv` inputs.
- The route must invoke the upstream MLIR EmitC C/C++ emitter rather than
  hand-writing C compute bodies.
- The route must be discoverable via existing plugin/built-in target translate
  registry aggregation.
- The focused pipeline must support:
  `tcrv-opt --tcrv-materialize-emitc-lowerable-routes | tcrv-translate <new route>`.

## Acceptance Criteria

- [ ] `tcrv-translate --help` lists the new RVV EmitC emitter handoff route.
- [ ] A focused lit/FileCheck test proves materialized RVV first-slice EmitC
      emits C/C++ source with the expected function ABI, `riscv_vector.h`
      include, and RVV intrinsic calls.
- [ ] A focused negative lit/FileCheck test proves raw non-materialized RVV
      input fails closed with a diagnostic naming the materialized EmitC
      requirement.
- [ ] C++ registry coverage proves the route is registered through the RVV
      plugin hook and through built-in target translate aggregation.
- [ ] Focused build/check commands for `tcrv-opt`, `tcrv-translate`, and the
      touched tests pass, or any failure is reported with the exact blocker.
- [ ] A reference scan confirms descriptor/direct-C/source-export legacy paths
      were not restored.

## Out of Scope

- No new RVV dtype, LMUL, arithmetic family, generic RVV lowering, or runtime
  ABI implementation.
- No descriptors, selected metadata recovery, family registry semantic lookup,
  direct C semantic generation, old artifact exporters, clang compile harness,
  object/bundle packaging, `ssh rvv` evidence, correctness, or performance
  claim.
- No extension-specific semantic branch in generic target orchestration.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant source surfaces:
  `include/TianChenRV/Target/TargetTranslateRegistration.h`,
  `lib/Target/TargetTranslateRegistration.cpp`,
  `lib/Target/Builtin/BuiltinTargetTranslateRoutes.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `lib/Transforms/EmitCLowerableMaterialization.cpp`,
  `lib/Conversion/EmitC/`,
  `test/Conversion/EmitC/rvv-first-slice-materialization.mlir`.
