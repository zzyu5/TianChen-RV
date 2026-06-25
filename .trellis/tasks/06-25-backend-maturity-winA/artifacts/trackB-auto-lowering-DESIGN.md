# Track B — Auto-Lowering Design: tile/vector-dialect → (auto-vectorize + auto-LMUL + auto-pipeline) → tcrv RVV body → EmitC

Read-only design investigation. No code changed. Citations are `file:line` at investigation time (line numbers drift; symbols are stable).

---

## 0. The framing correction (the single most important thing in this doc)

The brief says "**自动向量化 + 自动选 LMUL + 自动软件流水 → EmitC，取代手写 emitter**". Investigation shows the goal is *almost right but the target is mis-located*: **the hand-written EmitC emitter is NOT the gap, and Track B does not replace it.**

`RVVToEmitC.cpp` is already a mechanical, 1:1 op→intrinsic walk: it `dyn_cast`s each op in the `with_vl` body and emits exactly one `__riscv_*` `CallOpaqueOp` per op (`lib/Conversion/RVV/RVVToEmitC.cpp:914`, the `emitWideningDotReduce` dispatch; the emit body at `:2901`-`:2963` is literally "one `vwmul` + one `vredsum`"). **All the "intelligence" — which intrinsic, which LMUL, the 2-strip ILP, where the loop is — lives in the tcrv RVV-dialect OP STRUCTURE, not in the emitter.** The emitter is a dumb printer.

Therefore:

> **Track B = automatically GENERATE the tcrv RVV-dialect body** (`with_vl` / `setvl` / `widening_dot_reduce` / `load` / `store`) **that the existing EmitC emitter consumes UNCHANGED.** What is hand-written today is not the emitter — it is the *construction of that op structure* (the `SelectedBodyRealizationOwner` / `ScheduleMaterialization` machinery picks intrinsics, LMUL, and strip count by hand, per kernel). That construction is what we auto-generate.

This reframing also lifts the feasibility verdict above what generic SOTA suggests (§4): this codebase **already owns the vsetvl/LMUL/tail-correctness layer** that the SOTA flags as the big risk — hand emitters already emit fixed-LMUL `__riscv_*` under one `SetVLOp`, and the gearbox already chooses LMUL from a capability fact. We are not building a vscale-portable vectorizer; we are pinning LMUL exactly like prior EmitC-RVV work does, which is acceptable here.

---

## 1. First target kernel class: **widening dot-reduce** (`widening_dot_reduce`)

Recommended first block: the **signed widening dot-reduce-add** that `emitWideningDotReduce` lowers (`lib/Conversion/RVV/RVVToEmitCInternal.h:2783`, op def near `include/.../RVVOps.td` widening_dot_reduce). Why it is the cleanest first block:

- **Has a hand-written oracle.** `emitWideningDotReduce` is the byte-exact target to diff against. A "can auto-generate" proof needs exactly this.
- **Smallest correct unit.** One op decomposes to `vwmul` + `vredsum` — two intrinsics, one VL scope, no codebook gather, no AoS block loop, no dequant epilogue. The other emitter families (K-quant, codebook-fp4, deferred-dequant) are far heavier.
- **It is the core of block-dot.** Every q4_0/q8_0/iq* block-dot kernel is a dot-reduce inner. Proving auto-gen here is on the N3 main trunk, not a side branch.
- **A high-level source op exists that lowers to it cleanly:** `arith.muli` (vector) + `vector.multi_reduction <add>` (see §2). This reuses the existing front-door's arith-binary matcher style with one extra op.

Phase-2 generalization target: the *masked* widening dot-reduce (`emitMaskedWideningDotReduce`, `:2965`) and `vector.contract` as the matrix generalization of `multi_reduction`.

---

## 2. Architecture: the dialect / pass chain

**One-sentence chain:** high-level reduction op (`transfer_read ×2 + arith.muli + vector.multi_reduction<add> + transfer_write`) → a **new sibling front-door matcher** (modeled on `RVVVectorSourceFrontDoor`) that builds the tcrv `with_vl`/`setvl`/`widening_dot_reduce`/`store` body, with **LMUL chosen by the gearbox capability fact instead of hardcoded m1** → the **existing `RVVToEmitC` emitter, unchanged** → C with `__riscv_*` intrinsics.

### The precedent we extend: `RVVVectorSourceFrontDoor`

This file already does 90% of the front-end plumbing for the *elementwise* case (`lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`):

- **Input it accepts** (`:479`): exactly `2× vector.transfer_read` + one supported `arith` vector binary op + `1× vector.transfer_write`, rank-1 unit-stride i32, no masks.
- **Output it produces** (`:1391`-`:1458`): a `tcrv.exec.KernelOp` → `VariantOp` → `RuntimeABIValueOp`s → `SetVLOp` → `WithVLOp{ LoadOp; BinaryOp; StoreOp }`, stamped with route `rvv-generic-typed-body-emitc-route-family`. **This is exactly the emitter's input contract.** It then routes to EmitC unchanged.
- **LMUL today: hardcoded `m1`** (`:1135`, `:1150`, `:1441`). No strip-mining, no loop in the body (the emitter synthesizes the VL chunk loop — see §3).

**Track B's stage (a) = a sibling matcher in this file (or a new `RVVReductionSourceFrontDoor`)** that recognizes the reduction shape and emits a `WideningDotReduceOp` body instead of a `BinaryOp` body. The dataflow (load lhs, load rhs, widening_dot_reduce, store lane-0) maps 1:1; the standalone-reduction seed machinery already exists in the emitter (`:579`).

### Stage (b): auto-LMUL — reuse the gearbox capability fact

The front door's hardcoded `m1` is replaced by a call into the existing selection authority:

- **Capability facts** are derived from `-march` + probed ISA hints, NOT march substrings: `deriveMinimumVLEN(...)` returns guaranteed VLEN bits (`include/.../RVVCapabilityProfile.h:162`) and `deriveRVVVersion(...)` returns RVV0.7-vs-1.0 (`:58`).
- **The selection model already exists** in `RVVScheduleMaterializationDescriptor` (`include/TianChenRV/Plugin/RVV/RVVScheduleMaterialization.h:57`): an `enumerate(minimumVLEN, budget)` lambda generates LMUL/strip candidates and `selectGenericSchedule` picks measured-best (from a record) else static-argmin (`lib/Plugin/RVV/RVVScheduleDescriptorRegistry.cpp:213`,`:250`-`:281`).
- **The RVV0.7-vs-1.0 LMUL branch is concrete** in `RVVRepackStripWidthMaterialization.cpp:123`-`:131`: RVV0.7 → whole-LMUL `m1` + 16-lane strip; RVV1.0 → `deriveRepackHalfLanes(vlenBits, …)` (VLEN128→8, VLEN256→16). This is the auto-tune knob the brief wants — and it is already capability-driven.

**Honest gap (§4):** today this enumeration is **entangled per-quant-kernel** (each of q4_0/q8_0/iq4_nl has its own `enumerate` in `RVVScheduleDescriptorRegistry.cpp:38`-`:198`). For a *generic* dot-reduce vectorizer we need a **generic `enumerate(elementWidth, opKind=dot_reduce, capability)`** — i.e. factor the kernel-locked LMUL choice into a reusable descriptor. This is net-new, but it is a refactor of an existing model, not a new cost model from scratch.

### Stage (c): software pipelining — **resolved factual finding**

The upstream MLIR `scf` loop pipeliner (`populateSCFLoopPipeliningPatterns`) **is INAPPLICABLE here**, because **there is no `scf.for` in the body the vectorizer produces.** Verified:

- The `with_vl` body is a single straight-line op chain. **The VL chunk loop (`for (i=0; i<n; i+=vlmax)`) is SYNTHESIZED BY THE EMITTER**, not present in the dialect IR: `rewriter.create<emitc::ForOp>(loc, zero, avlArg, vlmax, …)` at `lib/Conversion/RVV/RVVToEmitC.cpp:616` (helper `emitScopeForLoop`, `:605`). A generated kernel confirms the loop appears only in the emitted C (`artifacts/gate4-candidate-feedback-ssh/.../materialized_rvv_emitc.cpp:24`), driven by setvl-tail chunking.
- **The 2-strip ILP is structural, already in the body:** the generated C shows two parallel `vwmul` op-chains per iteration (`materialized_rvv_emitc.cpp:35` and `:52`, two `__riscv_vwmul_vv` from two loads). That comes from the dialect body holding **two parallel `Load`+`WideningProduct` chains**, NOT from any pipeliner.

**So "software pipelining" in this architecture means: emit N parallel strip op-chains into the `with_vl` body** (the existing 2-strip ILP, generalized to an N picked by the gearbox), exactly the structural multi-strip the codebase already uses. It is a **body-construction decision in stage (a)/(b), not an scf transform.** (If a future design introduces an analyzable `scf.for` at the dialect level, the upstream pipeliner becomes reusable — but that is not today's IR.)

### Dialect/pass summary

| Stage | What | Off-the-shelf? |
|---|---|---|
| Source IR | `transfer_read ×2 + arith.muli + vector.multi_reduction<add> + transfer_write` | Upstream `vector`/`arith` dialects ✅ |
| (a) vectorize → tcrv body | new sibling matcher in `RVVVectorSourceFrontDoor` style | self-built; **strong precedent exists** |
| (b) auto-LMUL | call `deriveMinimumVLEN`/`deriveRVVVersion` + generic `enumerate` | selection model exists; **generic enumerate is net-new** |
| (c) pipeline = N-strip | emit N parallel op-chains in the body | self-built (NOT scf pipeliner — see above) |
| lower → EmitC | `RVVToEmitC` `VariantToEmitCFunc` | **existing, UNCHANGED** ✅ |

---

## 3. Feasibility — honest "off-the-shelf vs build"

### Off-the-shelf and reusable
- **The EmitC emitter, entirely.** It already consumes generated IR as long as it obeys the contract: one `VariantOp`, exactly one `with_vl` scope (it finds the first and breaks, `RVVToEmitC.cpp:73`-`:79`), a `SetVLOp` with runtime ABI AVL, tail/mask-agnostic policy, and only supported body op types (else "unsupported op in with_vl beachhead body"). An auto-vectorizer that emits these ops gets EmitC for free.
- **The front-door pattern.** `RVVVectorSourceFrontDoor` is a working `vector → tcrv-body → EmitC` transform. We clone its structure.
- **The gearbox selection model + capability derivation.** `deriveMinimumVLEN`/`deriveRVVVersion` + the descriptor enumerate/select skeleton.
- **Upstream `vector`/`arith`/`scf`/`memref` dialects** are already registered (`lib/InitTianChenRVDialects.cpp:17`-`:18`).

### Absent / must build
- **No upstream `vector → RVV-intrinsic` lowering exists** (RVV has no upstream dialect; the 2021 RVV-dialect RFC never landed — contrast Arm SVE/SME which ARE upstream). No `convert-vector-to-emitc`. There is **no MLIR pass that turns `vector.multi_reduction`/`vector.contract` into `__riscv_*` calls** — this is fundamentally DIY. (The only existence proof of "vector→RVV-intrinsic-EmitC" is the xDSL paper, arXiv 2603.17800 — and it hardcoded fixed VLEN + fixed LMUL, i.e. exactly our acceptable "pin LMUL" posture.)
- **The reduction matcher** (`multi_reduction<add>` over a widening `muli`) — net-new, but small and modeled on the existing arith-binary matcher.
- **The generic LMUL `enumerate`** for dot-reduce (factored out of the per-quant entanglement).
- **The N-strip ILP body emission** as a parameterized choice (today 2-strip is hand-placed per kernel).

### Risk this analysis de-risks
The generic SOTA names "owning vsetvl/LMUL/tail correctness outside LLVM" as the killer risk. **This codebase already pays that cost and has for many kernels** (hand emitters emit `__riscv_vsetvl_e32m*` + fixed-LMUL intrinsics, all byte-exact-verified on `ssh rvv`). So the risk is largely retired; the residual work is front-end, not correctness-layer.

---

## 4. Validation plan

**Oracle:** the hand-written `emitWideningDotReduce` output for a chosen dot-reduce shape.

1. **MVP gate (correctness, fixed m1):** feed the *high-level source op* (`multi_reduction<add>` form) through the new front-door matcher at **fixed m1**, lower via the unchanged emitter, and **byte-exact diff the emitted C against the hand-written `emitWideningDotReduce` kernel** (lit / FileCheck). 
   - **Subtlety that must be honored:** the test MUST start from the high-level vector op, NOT from a hand-built tcrv body. If we hand-build the `with_vl` body, byte-exact is trivial (same emitter) and proves nothing about auto-generation. The diff that counts is `high-level op → [our new matcher] → C` vs `hand kernel → C`.
   - **What this proves:** the backend can auto-generate a *correct* dot-reduce. **It does NOT prove the auto-tune.**
2. **Phase-2 gate (LMUL tune, where the N3 claim lands):** drive stage (b) so the same source op emits the gearbox-selected LMUL/strip on VLEN128 vs VLEN256, and assert **两板 (K1/VLEN256 + a VLEN128 board) byte-exact** vs the matching hand kernels per board. This is where capability-aware selection (the N3 surface) is actually demonstrated.
3. **Phase-3 gate (perf parity):** with N-strip ILP + (if introduced) pipelining, micro-bench the auto-generated kernel `ssh rvv` vs the hand-written one; require **perf ≥ hand-written** at matched VLEN. Report kernel-micro and e2e separately (memory: kernel wins do not transplant to memory-bound e2e).

---

## 5. Phasing, effort, risk

**MVP (correctness only, fixed m1):** new reduction matcher + body builder cloning `RVVVectorSourceFrontDoor`; reuse emitter as-is; one lit byte-exact test from the high-level op. *This is the "backend can auto-generate one correct kernel" proof.* Lowest risk — every downstream piece already exists.

**Phase 2 (auto-LMUL):** factor a generic `enumerate(elementWidth, dot_reduce, capability)` out of the per-quant registry; wire `deriveMinimumVLEN`/`deriveRVVVersion`; 两板 byte-exact. *This is the N3-bearing phase.* Medium risk: the entanglement refactor + a second board.

**Phase 3 (perf parity):** parameterize N-strip ILP in the body; optional structural pipelining; `ssh rvv` micro perf ≥ hand. Highest risk: matching hand-tuned ILP placement; the ILP/pipeline structure is hand-art today.

**Biggest single risk:** the whole vector→RVV-intrinsic front-end is genuinely DIY (no upstream pass) — but it is **de-risked** because (i) the EmitC emitter + gearbox selection + capability derivation already exist and are byte-exact-proven, and (ii) `RVVVectorSourceFrontDoor` is a working precedent for the exact `vector → tcrv-body → EmitC` shape. The trap to avoid is over-claiming "we built a Triton backend" (buddy-mlir/triton-cpu already do vector→LLVM→RVV); the defensible N3 novelty is narrower and real: **capability-aware LMUL selection at the MLIR level emitting RVV-intrinsic EmitC, reusing gearbox capability facts** — unbuilt upstream.

**Recommended first step:** build the MVP (§4 gate 1) — clone `RVVVectorSourceFrontDoor`'s matcher into a reduction sibling that emits a fixed-m1 `widening_dot_reduce` body, and land the single lit test diffing `high-level op → C` against the hand emitter. It exercises the entire chain end-to-end with zero auto-tune risk and proves the core claim ("the backend can auto-generate, not hand-write, a correct dot-reduce").
