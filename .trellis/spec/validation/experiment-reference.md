# Experiment Reference

## Status

This spec is a validation reference. It must not decide system structure.

TianChen-RV MLIR is first a capability-driven RISC-V execution layer. Experiments test whether that design holds.

## Hardware Conditions

### RVV main

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, performance, and correctness environment
```

Repeatable bounded hardware/toolchain evidence should be captured with
`scripts/rvv_remote_probe.py`. Its artifacts are written below
`artifacts/tmp/rvv_probe/<run-id>/` and include sanitized command logs plus a
JSON summary of uname/kernel, architecture, hart count, clang/cmake
availability, bounded RISC-V/vector CPU hints, non-interactive sudo
capability, and the minimal hand-written RVV intrinsic compile/run result.

This probe is a prerequisite evidence source for future RVV compiler claims,
but it is not itself a TianChen-RV compiler correctness, runtime, supported
emission, or performance artifact.

### K3/IME later

```text
hardware: K3 / IME-capable RISC-V environment
status: later acquisition/integration
role: IME plugin integration and matrix-extension evaluation
```

### RISC-V Sophgo/offload

```text
hardware: RISC-V host + Sophgo accelerator path
role: runtime-offload capability evaluation
note: not custom RISC-V ISA evidence
```

## Research Questions

### Q1: Can TianChen-RV generate valid code on real RVV hardware?

Objects:

```text
matmul
batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

Comparisons:

```text
MLIR linalg/vector default lowering
LLVM auto-vectorization
scalar/OpenMP baseline
hand-written RVV kernels if available
existing AI-Benchmark RVV kernels if usable
```

Metrics:

```text
correctness
single-thread performance
multi-thread performance
compile success rate
performance over default lowering
variant-local tuning benefit
```

### Q2: Does capability model participate in pass decisions?

Profiles:

```text
RVV only
RVV + offload runtime
RVV + IME, after K3 available
fallback-only profile
```

Expected behavior:

```text
RVV only -> RVV variant + fallback
RVV + offload -> RVV variant + offload variant + dispatch + fallback
RVV + IME -> RVV variant + IME variant + dispatch + fallback
fallback-only -> fallback
```

Metrics:

```text
generated variants match capability
illegal variants are rejected by verifier
dispatch conditions are correct
capability changes alter pass decisions
diagnostics are clear
```

### Q3: Is extension plugin integration local?

Reference process:

```text
system has RVV plugin
add offload plugin
later add IME plugin
measure core pass changes and plugin boundary
```

Metrics:

```text
core pass modified LOC
plugin LOC
new capabilities
new ops/types
new variant generators
supported high-level op count
extension-specific branches in core pass
reuse of tcrv.exec.variant / dispatch / verifier orchestration
```

### Q4: Can runtime-offload capability join the same execution layer?

Objects:

```text
large matmul
conv
transformer block or MLP block if runtime supports
```

Comparisons:

```text
RVV CPU variant
Sophgo offload variant
RVV + offload dispatch
fallback
```

Metrics:

```text
offload threshold
end-to-end latency
host-device transfer overhead
runtime launch overhead
shape-size effect on selection
fallback correctness
```

Hard rule:

```text
This validates runtime-offload capability, not custom RISC-V ISA.
```

### Q5: After IME arrives, can plugin-local matrix-extension integration be shown?

Objects:

```text
matmul
batched matmul
attention qk/av block
MLP dense block
int8/fp16/bf16 dot-like kernels
```

Comparisons:

```text
RVV variant
IME variant
fallback
hand-written or vendor IME kernel if available
```

Metrics:

```text
core pass modified LOC
IME plugin LOC
IME variant generation coverage
IME legality verifier effectiveness
IME emission success
performance over RVV for suitable kernels
```

## Ablation References

Capability model ablation:

```text
with capability-driven variant generation
without capability-driven variant generation, using fixed RVV path
```

Plugin locality ablation:

```text
plugin-based integration
hand-coded extension branches in core pass
```

Variant selection ablation:

```text
static RVV selection
capability-aware variant selection
dispatch with offload threshold
```

## Forbidden Interpretations

Do not claim:

```text
Sophgo offload is RISC-V custom ISA extension.
Ordinary tile-size tuning is the main theory.
AME is current verified primary hardware.
Any future extension never needs core changes.
TianChen-RV is a new high-level tensor IR.
```

Use:

```text
Sophgo offload is runtime-offload capability.
Tuning is a system ability inside capability-aware variant selection.
Current mainline is RVV; later IME validates new extension plugin integration.
Extensions that map to existing interfaces support plugin-local integration.
TianChen-RV is a RISC-V execution layer after high-level MLIR.
```
