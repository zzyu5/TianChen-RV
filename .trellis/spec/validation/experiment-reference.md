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

## N3 Performance-Claim Discipline (baselines — durable contract)

N3 ("capability/resource-aware tune that measurably wins") claims must obey a fixed baseline discipline so a
number names a real contribution, not an artifact of a weak comparand:

- **Scalar is NEVER a contribution baseline.** vector-vs-scalar measures "we vectorized at all" — which
  MLIR/autovectorization already provide. It may appear only as an internal sanity check, never as a reported
  multiple. (The old "wide ÷ scalar 4–15×" framings are retracted.)
- **Keep the three Wins separate; each has ONE mandated baseline:**
  - **Win-A** = the compiler-automatic *tune* (e.g. max-legal-LMUL width, VLEN→strip selection). Baseline =
    the SAME kernel with the tuned knob OFF (both arms compiler-emitted; only the knob differs).
  - **Win-B** = a generated kernel for an algorithm/layout the framework also ships. The framework may ship TWO
    kernels for one quant (a block-dot AND its own repack), so Win-B MUST be reported as two ORTHOGONAL
    comparands, never collapsed into one column:
    - **Win-B1 = vs the framework's block-dot** — "does the layout/algorithm change help". Where the framework
      ships NO such kernel on that VLEN (e.g. ggml's `case 128: break // TODO` → no q4_0 repack at VLEN128), a
      Win-B1 win is a legitimate GAP-FILL e2e acceleration against the real shipping baseline, NOT a weak-baseline
      artifact — but it is a frontend (added-algorithm) win, not a backend-codegen win.
    - **Win-B2 = vs the framework's OWN repack** (only exists where the framework ships one) — "is our codegen
      competitive". The honest success criterion here is PARITY (matching the expert's hand-written kernel).
    NEVER scalar / naive / `_generic`. Conflating the two comparands in one column (big-vs-block-dot numbers next
    to parity-vs-own-repack numbers) is baseline-mixing and over-states the backend win.
    *(Changing the algorithm/layout is a frontend/library contribution, NOT a backend N3 novelty; weight-storage
    repack is the offline-prepack class — Marlin/AWQ/CUTLASS analog — that even Triton leaves outside the compiler.
    See the frontend-vs-backend discriminator in [system-positioning](../architecture/system-positioning.md) N3
    boundary. Win-B is kept here as honest-measurement discipline, not as an N3 backend claim.)*
  - **Win-C** = an automatic *pass* that changes algorithm structure. Baseline = pass OFF vs ON. A
    hand-authored kernel is Win-B, never relabeled as an automatic-pass contribution.
    **A pass-ON/OFF number is NOT automatically a *structural* win.** If the ON arm changes BOTH the
    structure AND an incidental emission property (e.g. it avoids a memory round-trip the OFF arm's emitter
    happens to incur), the ON/OFF delta conflates the two. To attribute the win to the STRUCTURE you MUST
    decompose against a *competently-emitted baseline of the SAME structure* (e.g. a register-kept-accumulator
    per-iteration reduction). If that same-structure competent baseline TIES the ON arm, the structural
    contribution is **NULL** — report only the pass-ON/OFF number with its real but non-structural mechanism,
    never a "structural" win. *(Verified 2026-06-24: a deferred-vs-per-iteration reduction pass gave 3× ON/OFF,
    but the register-kept per-iter control tied the deferred arm at ≈1.00× — the 3× was entirely a per-iter
    `out[0]` memory round-trip, not reduction-structure latency. Win-C-as-structural-novelty: NOT demonstrated.)*
- **Both harnesses are required and not interchangeable:** an isolated single-core microbench (clean
  ablation) AND a real end-to-end run (catches integration/memory effects). A microbench win that does not
  appear e2e must be disclosed as such; regime-dependence (compute-bound vs memory-bandwidth-bound vs
  **per-block reduction-latency-bound** — the block-quant decode regime is the latter, not bandwidth, and a
  repack that dissolves the per-block reduction wall is what transplants) must be stated, not hidden behind
  the larger number.
- **Every reported cell carries an evidence-status tag** `{measured | presumed | board-pending | N/A-by-construction}`.
  An unmeasured cell defaults to NO claim — never to the success state. "presumed parity / presumed null" is not a
  result, it is an open measurement; banking it as success is the recurring over-optimism failure mode. After any
  refactor, prior numbers are STALE until re-measured on the named profile — a parity/win is a target re-measured
  per build, not a banked metric.
- **A repack / block-as-lane kernel's perf claim must name the VLEN regime AND compare vs the framework's
  ACTUAL same-VLEN baseline** (its hand-tuned VLEN-native kernel if one ships, else its generic fallback) —
  never a different-VLEN kernel. Whether the repack wins is set by **competitor strength × compute-density**,
  NOT by the repack's lane-shape (the 2×8 mf2 form IS the correct VLEN128 tiling; the 2-strip split is
  beneficial ILP — the q8_0 NARROW>WIDE ISO datum). The repack wins only when the same-VLEN fallback is a
  HEAVY kernel it out-streams; against a LEAN fallback or a hand-tuned VLEN-native kernel it LOSES, and the
  tune should DECLINE it (select the fallback = match the framework's own algorithm) — declining = matching
  the framework = loss-avoidance, NOT a backend N3 contribution. Choosing repack-vs-fallback is an
  algorithm/layout choice (frontend/library), NOT capability-driven lowering of a fixed op+layout (the
  backend N3 face); see [system-positioning](../architecture/system-positioning.md) N3 boundary. It is NOT a
  kernel bug and NOT a Win-B speedup. *(Verified 2026-06-24: q4_0 repack WINS
  @VLEN128 vs ggml's HEAVY vec_dot; q8_0 LOSES vs ggml's LEAN vec_dot; q4_K LOSES vs ggml's hand-tuned
  _vl128 — all at the SAME correct 2×8 mf2 shape. See SHAPE-AWARE-REPACK-TUNE-DESIGN.md.)*
- Performance claims still require real `ssh`-hardware evidence on a named profile (I8); engagement of the
  emitted kernel must be proven (e.g. an ENGAGED marker / objdump of the FINAL staged binary), not assumed.
- **A new-hardware-unit / build-swap claim needs a can't-possibly-help control.** When a number compares two
  separately-built artifacts (e.g. an IME-enabled lib vs a non-IME lib), it confounds the new unit with ANY
  global toolchain/codegen difference between the builds. Include a CONTROL regime where the new unit
  *physically cannot* contribute (e.g. M=1 memory-bandwidth-bound decode for a matrix/MAC array); if the "win"
  appears THERE too, it is a global codegen artifact, not the unit. A clean unit-isolation requires the SAME
  toolchain with only the unit toggled (e.g. `SPACEMIT=ON` vs `OFF` on one compiler), NOT two independently
  built libs. *(Verified 2026-06-24: a reconstructed clang IME lib showed 1.4–1.95× "prefill", but also 1.25×
  in M=1 decode where the matrix unit can't help → the gain was clang codegen, not IME; the IME-unit e2e win
  stayed NULL.)*

## Forbidden Interpretations

Do not claim:

```text
Sophgo offload is RISC-V custom ISA extension.
Ordinary tile-size tuning is the main theory.
A vector-vs-scalar speedup is an N3 tune contribution.
A microbench win is an end-to-end win without an e2e measurement.
A pass-ON/OFF speedup is a structural-transform contribution without a same-structure competently-emitted control.
A build-swap speedup (e.g. IME-lib vs non-IME-lib) isolates the new hardware unit without a can't-possibly-help control regime.
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
