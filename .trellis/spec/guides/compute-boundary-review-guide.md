# Compute Boundary Review Guide

Use before introducing any new op, dialect, or lowering pass.

## Checklist

- [ ] Does the new construct organize execution or express algorithm semantics?
- [ ] If it is algorithm semantics, why is it not represented by high-level MLIR before TianChen-RV?
- [ ] If it is hardware execution behavior, which TCRV extension family owns it?
- [ ] Does `tcrv.exec` only contain kernel, target, capability, variant, requires, region, hart_parallel, mem_window, dispatch, fallback, or diagnostics structure?
- [ ] Are matmul/softmax/reduce semantics absent from core dialect?
- [ ] Does the construct preserve plugin-driven variant proposal?
- [ ] In current RVV work, does the path start from hand-authored/selected TianChen-RV MLIR plus typed `tcrv_rvv` body, not a new high-level frontend?
- [ ] Do selected bodies explicitly import/consume `mem_window` / `runtime_param` ABI values?
- [ ] If performance config affects code, is it consumed into realized body structure rather than status metadata?

## Allowed In Core

```text
tcrv.exec.kernel
tcrv.exec.target / target attachment
tcrv.exec.variant
tcrv.exec.requires
tcrv.exec.hart_parallel
tcrv.exec.region
tcrv.exec.mem_window
tcrv.exec.dispatch
tcrv.exec.fallback
tcrv.exec.diagnostic / diagnostic metadata
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
tcrv.matmul
tcrv.softmax
tcrv.generic_reduce
tcrv.generic_tile
tcrv.generic_mma
high-level tensor compute in core dialect
```

## Correct Transform Shape

Current RVV Stage1/Stage2 shape:

```text
hand-authored or selected TianChen-RV MLIR
  -> tcrv.exec envelope + selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV plugin legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC route
```

Future frontend shape after RVV maturity:

```text
high-level MLIR op
  -> plugin registry proposes execution variants
  -> tcrv.exec envelope contains extension-family variants
  -> selected variant lowers through common EmitC route plus family mapping
```
