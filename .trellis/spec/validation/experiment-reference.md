# Experiment Reference

## Status

This spec is a validation reference. It must not decide system structure.

TianChen-RV MLIR is first a capability-driven RISC-V execution layer. Experiments test whether that design holds.

## Hardware Conditions

具体硬件环境（RVV main、K3/IME later、RISC-V Sophgo/offload）是当前事实，会变；权威定义在 [../capability-model/profiles.md](../capability-model/profiles.md)，不在本验证参考里重复。下面只保留与证据解释相关的、durable 的部分。

Repeatable bounded hardware/toolchain evidence should be captured with
`scripts/rvv_remote_probe.py`. Its artifacts are written below
`artifacts/tmp/rvv_probe/<run-id>/` and include sanitized command logs plus a
JSON summary of uname/kernel, architecture, hart count, clang/cmake
availability, bounded RISC-V/vector CPU hints, non-interactive sudo
capability, and the minimal hand-written RVV intrinsic compile/run result.
The JSON artifact may also include a sanitized `capability_facts` section for
the compiler-facing profile boundary. Those facts are input to the plugin-local
C++ RVV capability profile, which validates them and populates
`TargetCapabilitySet`; they are not themselves compiler internals or proof that
TianChen-RV emitted executable RVV code.

This probe is a prerequisite evidence source for future RVV compiler claims,
but it is not itself a TianChen-RV compiler correctness, runtime, supported
emission, or performance artifact.

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

These objects calibrate RVV coverage and future frontend proof. They do
not make current high-level Linalg/frontend lowering the source authority.
Current RVV codegen claims must still flow through selected `tcrv.exec`
variants, typed `tcrv_rvv` bodies, RVV plugin legality/realization, provider
routes, and common EmitC.

Comparisons:

```text
MLIR linalg/vector default lowering
LLVM auto-vectorization
scalar/OpenMP baseline
hand-written RVV kernels if available
existing AI-Benchmark RVV kernels if usable
```

MLIR Linalg/Vector default lowering is comparison/reference only unless a
future frontend task explicitly selects it.

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
RVV + offload runtime (offload not built yet)
RVV + IME (IME not built yet)
fallback-only profile
```

Expected behavior:

```text
RVV only -> RVV variant + fallback
RVV + offload -> RVV variant + offload variant + dispatch + fallback (once offload is built)
RVV + IME -> RVV variant + IME variant + dispatch + fallback (once IME is built)
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

Reference process (once the second family is built):

```text
system has mature RVV plugin
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

Q4 covers offload, which is not built yet. It must not introduce
source-front-door or offload artifact authority (these routes fail closed,
见 core-invariants I7)。

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

Q5 covers IME, the N2 second-family target, which is not built yet and needs
real IME hardware/toolchain evidence.

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
Structured kernel validation objects are current source-route authority.
Offload or IME dispatch is required before RVV typed-route maturity.
Source-front-door generated artifacts prove RVV maturity.
```

Use:

```text
Sophgo offload is runtime-offload capability.
Tuning is a system ability inside capability-aware variant selection.
Current mainline is RVV; later IME validates new extension plugin integration.
Extensions that map to existing interfaces support plugin-local integration.
TianChen-RV is a RISC-V execution layer after high-level MLIR.
```
