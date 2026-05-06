# Design Boundaries

## Non-Goals

TianChen-RV MLIR must not become:

- another high-level tensor/tile IR;
- a generic compute dialect with `tcrv.matmul`, `tcrv.softmax`, `tcrv.reduce`, `tcrv.generic_tile`, or `tcrv.generic_mma`;
- a collection of hard-coded backend branches in core passes;
- a claim that future hardware extensions require zero work;
- a claim that Sophgo/runtime offload is a RISC-V custom ISA extension;
- a paper story where ordinary parameter search is the primary theory.

## Hardware Truth

### Current primary path

```text
name: rvv-main
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, correctness, and performance path
```

All current "real hardware path" claims should default to this RVV environment unless another environment has been explicitly probed and recorded.

### Later IME/K3 path

K3/IME is a later plugin integration target. It validates whether a new matrix-like extension can be added locally through an IME plugin.

Do not write IME as already-available primary hardware unless the environment has been acquired, probed, and recorded.

### Runtime offload path

Sophgo/RISC-V + offload is a runtime-offload capability case:

```text
RISC-V host -> vendor runtime / C ABI / driver / queue -> accelerator
```

It is not evidence for custom RISC-V ISA execution.

### Future extension slots

AME、future custom ISA、other vendor extensions are extension plugin slots. They should appear as future extensibility targets, not as required current milestones or current hardware evidence.

## Correct Paper Language

Use:

```text
TianChen-RV is a capability-driven execution layer after high-level MLIR.
Capability objects drive variant generation, legality, selection, dispatch, and emission.
Sophgo/offload is modeled as runtime-offload capability.
IME is a later matrix-extension plugin used to evaluate plugin-local integration.
Tuning is part of capability-aware variant selection and variant-local optimization.
```

Avoid:

```text
TianChen-RV is a new tensor IR.
Sophgo is a RISC-V custom ISA extension.
AME is the current primary hardware path.
Any future extension can be added with zero core work.
Retuning alone is the main theoretical contribution.
```
