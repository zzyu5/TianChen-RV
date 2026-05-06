# Compute Boundary Review Guide

Use before introducing any new op, dialect, or lowering pass.

## Checklist

- [ ] Does the new construct organize execution or express algorithm semantics?
- [ ] If it is algorithm semantics, why is it not represented by high-level MLIR before TianChen-RV?
- [ ] If it is hardware execution behavior, which extension dialect owns it?
- [ ] Does `tcrv.exec` only contain kernel, variant, capability, hart_parallel, dispatch, or fallback structure?
- [ ] Are matmul/softmax/reduce semantics absent from core dialect?
- [ ] Does the construct preserve plugin-driven variant proposal?

## Allowed In Core

```text
tcrv.exec.kernel
tcrv.exec.variant
tcrv.exec.requires
tcrv.exec.hart_parallel
tcrv.exec.region
tcrv.exec.mem_window
tcrv.exec.dispatch
tcrv.exec.fallback
target/capability/cost/tuning/dispatch metadata
```

## Belongs In Extension Dialect

```text
RVV vector register ops
IME fragment/mma/dot ops
offload buffer/runtime call ops
future custom instruction ops
future vendor runtime ops
```

## Forbidden Core Shapes

```text
tcrv.matmul
tcrv.softmax
tcrv.generic_reduce
tcrv.generic_tile
tcrv.generic_mma
high-level tensor compute in core dialect
```

## Correct Transform Shape

```text
high-level MLIR op
  -> plugin registry proposes execution variants
  -> tcrv.exec.kernel contains extension-specific variants
  -> selected variant lowers through plugin emission provider
```
