# Lowering Runtime Specs

This layer defines lowering, emission, runtime glue, and toolchain boundaries.

## Pre-Development Checklist

- [ ] Is extension-specific emission implemented by the plugin emission provider?
- [ ] Does the route lower extension family ops through the common EmitC route?
- [ ] Does executable RVV lowering consume typed/realized `tcrv_rvv` body through an RVV plugin-built `TCRVEmitCLowerableRoute`?
- [ ] Are emission-plan diagnostics, status fields, route ids, and artifact metadata mirrors only, never route/dtype/compute authority?
- [ ] Is clang/LLVM the default native compiler, with GCC only a compatibility path?
- [ ] Are compiler flags, headers, libraries, runtime handles, and ABI needs recorded in capability/plugin metadata?
- [ ] Does verifier reject unavailable toolchain/runtime paths before emission?
- [ ] Does dispatch lower to diagnosable host-side decision logic?
- [ ] Does fallback remain available for unsupported capability/runtime/shape cases?
- [ ] Does the work avoid descriptor-to-C emission as a long-term architecture?

## Guidelines Index

| Spec | Description |
|---|---|
| [Unified EmitC Route](./emitc-route.md) | Common extension-family ops -> EmitC -> C/C++ route, compiler defaults, descriptor boundary |
| [Emission Runtime Contract](./emission-runtime-contract.md) | RVV、IME、offload emission and runtime boundaries |

## Quality Check

- Core passes must not call vendor-specific compiler or runtime paths directly.
- Toolchain patch/workaround belongs in plugin adapter.
- Direct descriptor-to-C string export is deleted-route residue or fail-closed
  implementation debt, not a transition lowering route or production system
  path.
- Common EmitC/export materializes provider-built routes. It must not choose
  RVV intrinsics, infer dtype, create RVV schedules, or invent body shape.
- Emission output must be reproducible enough for experiments: selected variant, capabilities, flags, libraries, path, fallback status, and failure reason.
