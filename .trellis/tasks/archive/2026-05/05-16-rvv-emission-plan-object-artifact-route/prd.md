# RVV emission plan to object artifact route

## Goal

For the existing bounded RVV i32m1 add slice, make the path from selected
explicit RVV ops through common EmitC materialization, upstream MLIR C/C++
emission, and RVV target artifact export produce a registered
`riscv-elf-relocatable-object` route. The route must be RVV-owned, consume the
existing EmitC/C++ emitter path, and fail closed with precise compile/toolchain
diagnostics if object compilation cannot complete.

## What I Already Know

- The current task starts from clean `main` at `bff9b6e`.
- No `.trellis/.current-task` existed before this task was created.
- The previous completed task registered the non artifact-backed
  `tcrv-rvv-emitc-to-cpp` route for already materialized EmitC modules.
- `RVVExtensionPlugin::buildVariantEmitCLowerableRoute` already builds the
  bounded i32m1 add EmitC route from explicit `tcrv_rvv` ops.
- `RVVExtensionPlugin::buildVariantEmissionPlan` still fails before producing a
  runtime ABI or artifact route.
- `registerRVVTargetSupportPluginTargetExporterBundles` is still a no-op.
- Local `/usr/bin/clang-20` can compile an RVV intrinsic source with
  `--target=riscv64-unknown-elf -march=rv64gcv -mabi=lp64d` into an ELF
  RISC-V relocatable object.

## Requirements

- Register an RVV plugin-owned target artifact exporter through the RVV target
  support extension bundle.
- Make RVV emission readiness/planning report a supported route only for the
  same explicit i32m1 add slice accepted by the existing EmitC materialization
  route.
- The supported plan must carry coherent route identity, runtime ABI
  kind/name/glue role, artifact kind, and ordered runtime ABI parameters for
  `lhs`, `rhs`, `out`, and `n`.
- The exporter must materialize the existing RVV EmitC route, emit C/C++ through
  the MLIR EmitC emitter, and compile that generated source to a RISC-V ELF
  relocatable object with the available clang toolchain.
- Unsupported RVV shapes and non-materialized/non-EmitC direct translation
  inputs must continue to fail closed.
- Descriptor-driven computation, direct C semantic printers, source-export
  legacy routes, compatibility wrappers, and new dtype/LMUL families remain
  out of scope.

## Acceptance Criteria

- [x] Built-in target artifact exporter registration exposes exactly the new
      RVV object route through RVV target support, not through core
      family-name branching.
- [x] RVV emission plan diagnostics for the explicit i32m1 add fixture are
      `supported` and name the registered object route and
      `riscv-elf-relocatable-object`.
- [x] Generic target artifact export and the artifact-backed direct translate
      route can emit bytes that `file` identifies as a RISC-V relocatable
      object.
- [x] Runtime ABI parameter role/type/name/order checks reject mismatched
      supported plan metadata before artifact bytes are produced.
- [x] Unsupported RVV shapes remain fail-closed before claiming a supported
      object route.
- [x] Existing materialized EmitC -> C/C++ handoff remains covered.
- [x] Changed-surface scan shows descriptor/direct-C/source-export legacy terms
      were not restored in the touched production/test files.

## Out of Scope

- New RVV dtype or LMUL families.
- General RVV lowering, MLIR vector lowering, or LLVM RVV intrinsic IR.
- Descriptor, metadata-only, direct source printer, or handwritten compute C
  routes.
- Runtime execution, correctness, performance, or `ssh rvv` runtime claims.
- New bundle/state-machine/checkpoint protocols.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Likely touched files:
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  focused target/plugin tests, and focused lit fixtures.
