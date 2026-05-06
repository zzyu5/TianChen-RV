# Lowering Runtime Specs

This layer defines lowering, emission, runtime glue, and toolchain boundaries.

## Pre-Development Checklist

- [ ] Is extension-specific emission implemented by the plugin emission provider?
- [ ] Are compiler flags, headers, libraries, runtime handles, and ABI needs recorded in capability/plugin metadata?
- [ ] Does verifier reject unavailable toolchain/runtime paths before emission?
- [ ] Does dispatch lower to diagnosable host-side decision logic?
- [ ] Does fallback remain available for unsupported capability/runtime/shape cases?

## Guidelines Index

| Spec | Description |
|---|---|
| [Emission Runtime Contract](./emission-runtime-contract.md) | RVV、IME、offload emission and runtime boundaries |

## Quality Check

- Core passes must not call vendor-specific compiler or runtime paths directly.
- Toolchain patch/workaround belongs in plugin adapter.
- Emission output must be reproducible enough for experiments: selected variant, capabilities, flags, libraries, path, fallback status, and failure reason.
