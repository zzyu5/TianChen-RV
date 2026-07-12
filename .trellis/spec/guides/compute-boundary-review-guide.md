# Compute Boundary Review Guide

Use before introducing any new op, dialect, or lowering pass.

## Checklist

- [ ] Does the new construct organize execution or express algorithm semantics?
- [ ] If it is algorithm semantics, why is it not represented by high-level MLIR before Weft-RV?
- [ ] If it is hardware execution behavior, which WEFT extension family owns it?
- [ ] Does `weft.exec` only contain kernel, target, capability, variant, requires, region, hart_parallel, mem_window, runtime_param, dispatch, fallback, or diagnostics structure (见 core-invariants I2)?
- [ ] Are matmul/softmax/reduce semantics absent from core dialect?
- [ ] Does the construct preserve plugin-driven variant proposal?
- [ ] In current RVV work, does the path start from hand-authored/selected Weft-RV MLIR plus typed `weft_rvv` body, not a new high-level frontend?
- [ ] Do selected bodies explicitly import/consume `mem_window` / `runtime_param` ABI values?
- [ ] If performance config affects code, is it consumed into realized body structure rather than status metadata?

## Allowed In Core

```text
weft.exec.kernel
weft.exec.target / target attachment
weft.exec.variant
weft.exec.requires
weft.exec.hart_parallel
weft.exec.region
weft.exec.mem_window
weft.exec.runtime_param
weft.exec.dispatch
weft.exec.fallback
weft.exec.diagnostic / diagnostic metadata
target/capability/cost/tuning/dispatch metadata
```

## Belongs In Extension Family

```text
RVV vector register ops
IME fragment/mma/dot ops
offload buffer/runtime call ops
future custom instruction ops
future vendor runtime ops
```

## Forbidden Core Shapes

```text
weft.matmul
weft.softmax
weft.generic_reduce
weft.generic_tile
weft.generic_mma
high-level tensor compute in core dialect
```

## Correct Transform Shape

Current RVV shape:

```text
hand-authored or selected Weft-RV MLIR
  -> weft.exec envelope + selected RVV variant
  -> typed low-level weft_rvv body
  -> RVV plugin legality / selected-body realization / route provider
  -> WEFTEmitCLowerableRoute
  -> common EmitC route
```

Future frontend shape (not built yet):

```text
high-level MLIR op
  -> plugin registry proposes execution variants
  -> weft.exec envelope contains extension-family variants
  -> selected variant lowers through common EmitC route plus family mapping
```
