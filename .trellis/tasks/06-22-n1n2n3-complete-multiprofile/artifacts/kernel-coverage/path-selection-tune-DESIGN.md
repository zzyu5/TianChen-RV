# Path-selection tune — capability/resource-aware {repack, block-dot} SELECTION (DESIGN, 2026-06-24)

READ-ONLY design. No `lib/` edits. Companion to (and operationalizes Option B of)
`SHAPE-AWARE-REPACK-TUNE-DESIGN.md`. File:line build steps below.

## The contribution (frame EXACTLY this way — selection-correctness, NOT a Win-B)
The static **always-repack** choice is WRONG on 3 of the measured cells. A measured-best
selector over the two real compiler-emittable algorithms {repack, block-dot}, per
`(quant-type, VLEN, M-regime)`, KEEPS repack where it wins and DECLINES it where it loses:

| cell | static (always-repack) | measured-best | result of the selection |
|---|---|---|---|
| q4_0 @ VLEN128, M==1 | repack | **repack** (keep) | win kept: 1.22x->~2.6x e2e |
| q8_0 @ VLEN128, M==1 | repack (0.6–0.7x LOSS) | **block-dot** (decline) | tie ggml (no speedup) |
| q4_K @ VLEN128, M==1 | repack (0.47–0.66x LOSS) | **block-dot** (decline) | tie ggml |
| q4_0 @ K1 VLEN256, M==1 | repack (0.74x LOSS) | **block-dot** (decline) | **flips a real e2e regression to tie** |
| q4_0 @ VLEN128, M>>1 | repack | **repack** (keep) | GEMM 5.68x (prefill) |

DECLINING = select block-dot = MATCH ggml's native kernel (no speedup). The value is the
capability/resource-aware SELECTION — the algorithm-path analogue of the proven Win-A
VLEN-strip selection (1.31x) and the N1 VLEN->LMUL-family selection. Report ONLY as
selection-correctness ("measured > static argmin, which mis-picks 3 cells"); NEVER as a new
Win-B. Per project MEMORY (winc-structural-null, kernel-wins-dont-transplant): no Win-B reclaim.

---

## Architecture reality (what the inspection FOUND — load-bearing, refutes the obvious framing)

### F1. The algorithm path is fixed by OP IDENTITY in the input IR — no tcrv pass chooses it.
The repack-vs-block-dot choice is **already committed in the dialect IR the compiler is handed**.
Evidence — `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir`:
- a `tcrv.exec.variant @ggml_repack_gemv_q4_0_q8_0` whose body holds
  `tcrv_rvv.repack_gemv_q4_0_q8_0 {kind = "ggml_repack_gemv_q4_0_q8_0", weight_interleave = 16, weight_block_stride = 288 ...}`
- vs `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir`:
  `tcrv_rvv.q4_0_q8_0_block_dot {kind = "ggml_q4_0_q8_0_block_dot", weight_block_stride = 18 ...}`.

These are **distinct ops with distinct `kind` strings**, validated (not produced) by the dialect
verifiers in `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (each verifier pins ONE kind, e.g.
:1597 `ggml_repack_gemv_q4_0_q8_0`, :919 `ggml_q4_0_q8_0_block_dot`).

The lowering dispatch — `lib/Conversion/RVV/RVVToEmitC.cpp:334–406`, the
`kBlockDotKernels[]` recognizer table (`{recognize, emit}` pairs, first-match-wins) — is a
**faithful lowering** of whichever op is present (`isRepackGemvQ4_0Q8_0Body` -> `emitRepackGemvQ4_0Q8_0`;
`isQ4_0Q8_0BlockDotBody` -> `emitQ4_0Q8_0BlockDot`). It does NOT choose between them; the op identity
already determines the branch. The choice is **upstream of the compiler**, made by the IR producer.

### F2. M-regime is structurally encoded as op identity — NOT a runtime value. (GOOD NEWS)
`repack_gemv_*` (M==1 decode) vs `repack_gemm_*` (M>>1 prefill) are **separate ops / separate variants**
(see the GEMV/GEMM emitter pairs at `RVVToEmitCBlockQuantLinear.cpp:2674`/`2122`, `3167`, `4135`/`5342`).
So "M-regime" at selection time is *known by which op was produced* — no need to read a runtime M.

### F3. VLEN is already a compile-time capability fact. (GOOD NEWS)
`plugin::rvv::deriveMinimumVLEN(march, isaVectorHints)` (`RVVCapabilityProfile.cpp:253`) and
`deriveHasZvl128b(march, isaVectorHints)` (consumed in `RVVScheduleDescriptorRegistry.cpp:154`)
derive guaranteed minimum VLEN / Zvl128b from the selected `-march` (+ probed hints). VLEN128 vs
VLEN256(K1) is a static fact at this layer. **Both selector inputs (VLEN, M-regime) are already
compile-time facts — no blocker on the inputs.**

### F4. The path is COUPLED to the weight memory LAYOUT, not just the kernel body. (THE REAL BLOCKER)
Repack and block-dot consume **different weight bytes**:
- repack reads interleaved `block_q4_0x16`: `weight_block_stride = 288` (= 16×18),
  `weight_interleave = 16`, weights pre-transformed by ggml `from_float_to_mat`/`ggml_repack`.
  The repack emitter does NOT interleave in-kernel — it reads already-repacked bytes
  (`RVVToEmitCBlockQuantLinear.cpp:2674+`, "16-way interleaved repack reads the SAME bytes").
- block-dot reads plain `block_q4_0`: `weight_block_stride = 18`, no interleave.

So **"decline repack -> block-dot" is not a lowering op-swap** — block-dot cannot read repacked
weights. The path decision is bound to a model-load/producer-time weight-PACKING-TYPE decision
(`GGML_TYPE_Q4_0` plain vs the repacked extra-buffer type). The integration is NOT an in-repo
auto-selecting ggml bridge; it is: the tcrv compiler **emits a repack kernel as a C/C++ adapter**
(archived evidence: `emitted_adapter_gemv.cpp`, `arch_repack.cpp`, `transform_repack.py`,
`cmake_inject.py` under `.../fedora-rvv07/rvv07-perfile-build/` and `.../emit-repack-gemv/`) which a
**build-side injection** swaps into ggml `arch/riscv`, AND the model is repacked to `x16`. "Decline"
= do NOT emit/inject the repack adapter and do NOT repack the weights for that cell; ggml's native
block-dot stays. **Effort is dominated by this producer/packing coupling, not by the gate logic.**

---

## CONFIRMED answers to the task questions

### Q1 — Exact materialization/selection site where the gate must live.
There is **no existing in-compiler selection point between repack and block-dot** (F1). CONFIRMED by
inspecting what the FrontDoor consumes: `RVVVectorSourceFrontDoor.cpp` is an **elementwise vector-op
source materializer** (`materializeRVVVectorBinarySourceKernel` :1380, compare-select :1465+) — it
builds a kernel from an abstract `VectorBinarySourceMatch` and emits a two-case
`createDispatch(selected RVV variant, conservative scalar fallback)` (:1456). **Those two cases are
the SAME algorithm at two capability tiers (RVV vs scalar), NOT {repack, block-dot}.** The FrontDoor
does NOT consume the block-quant contraction ops at all — repack/block-dot arrive as
**already-committed dialect ops** in the input IR (F1; the `.mlir` test inputs already name
`tcrv_rvv.repack_gemv_q4_0_q8_0` with its stride-288/`nc`-operand ABI). It therefore CANNOT synthesize
the layout-incompatible block-dot sibling (stride-18, no `nc`) from the repack op it would be handed.

So the "always-repack" bug lives at the **PRODUCER / emit-plan + weight-packing boundary** — which
`tcrv.exec.variant` (+ which `kind` op) the compiler is *asked to materialize*, plus the build-side
decision to repack the model to `x16` (F4). The contraction path is committed **before** the compiler
sees the IR. The gate has TWO honest placements; pick by how much input-IR restructuring is in scope:

- **(PRIMARY — load-bearing, no input restructuring) Producer / emit-plan layer (S4).** The selection
  lives where the emit-plan chooses WHICH contraction variant `.mlir` to feed the compiler AND whether
  to repack the weights: the adapter-emit + injection harness
  (`.../fedora-rvv07/rvv07-perfile-build/{cmake_inject.py,transform_repack.py,arch_repack.cpp}`).
  `selectContractionAlgorithm` runs here; declining = feed the block-dot variant + skip weight repack.
- **(OPTIONAL — only if a future abstract contraction-request op is added) In-compiler dispatch.** If
  the input contract is restructured to an *un-committed* `tcrv` contraction op (analogous to the
  abstract `VectorBinarySourceMatch` the FrontDoor consumes for elementwise), THEN a FrontDoor-style
  two-case `exec.dispatch` could author BOTH bodies and select branch-free. Until that op exists, an
  in-compiler `exec.dispatch` over the already-committed op is **audit-only theater**, not a selection
  point (it cannot rebuild the layout-incompatible sibling). The `selectContractionAlgorithm` function
  is reusable verbatim in both framings — only the SITE moves; the load-bearing site today is the producer.

It is NOT `deriveRepackHalfLanes` (`RVVRepackStripWidthMaterialization.cpp:78`) — CONFIRMED the wrong
layer: width-only (`half_lanes = vlen/16` clamped to interleave), and it runs by walking an
*already-existing* `GgmlRepackGemmQ40Q80Op` (`module.walk` at :126+). It assumes the repack op was
already chosen; it cannot un-choose it.

### Q2 — Candidate-enumeration mechanism: register {repack, block-dot} as competing candidates.
**CONFIRMED BLOCKER (the task explicitly asked to surface this): `RVVScheduleDescriptorRegistry`
CANNOT express two competing algorithms for one op.** `lookupRVVScheduleDescriptor`
(`RVVScheduleDescriptorRegistry.cpp:37–144`) is a **one-`kernelKey`→one-descriptor** map; the
descriptor's `enumerate(hasZvl128b, budget)` returns `GenericScheduleCandidate`s that are **SHAPE
knobs (SEW/LMUL/strip-elision/activation_cols) of a SINGLE algorithm**, stamped onto an existing op
(`stampRVVSchedule`, :223). It selects a shape within a fixed kernel; it cannot swap op kinds or weight
layouts. This is the **Win-A axis**, structurally distinct from path selection. Do NOT bend the gate
into this registry.

The MLIR vocabulary that COULD express the choice is the existing **`tcrv.exec.dispatch` /
`tcrv.exec.variant`** machinery (`include/TianChenRV/Dialect/Exec/IR/ExecOps.td:134` VariantOp, :271
DispatchOp): it groups multiple variant cases + one fallback, with optional `condition`/`guard`/`policy`
per case (:289). Today it is **wired only for the plugin conservative-fallback** (selected-RVV vs
scalar, SAME algorithm) in the FrontDoor (:1456). **It is REUSABLE, but only under the restructured-
input framing** (Q1 optional branch): expressing {repack, block-dot} as two variant cases requires an
*un-committed* abstract contraction request op so the compiler can author BOTH layout-incompatible
bodies. Over the already-committed op it is **audit-only**, not a real selection mechanism.

Therefore, in the load-bearing (producer) framing, the "candidate enumeration" is NOT an MLIR
construct at all: it is the emit-plan choosing between two **pre-authored, layout-distinct variant
`.mlir` inputs** ({repack `x16`, block-dot plain}) via `selectContractionAlgorithm`, plus the
weight-packing flag. Branch-free / I1/I3-clean holds trivially — exactly one variant is compiled per
cell; there is no runtime `if` in emitted core code. The audit of the choice rides the
`low_precision_resource` mirror (Q4).

### Q3 — Gate signature + static prior.
```
enum class ContractionAlgorithm { Repack, BlockDot };

ContractionAlgorithm selectContractionAlgorithm(
    QuantType quant,        // q4_0, q8_0, q4_K, ...
    MRegime  mRegime,       // Decode (M==1, gemv op) | Prefill (M>>1, gemm op)  [F2: op identity]
    std::int64_t minVLEN,   // deriveMinimumVLEN(march, isaVectorHints)          [F3]
    const std::optional<RepackPathMeasurement>& measured);  // advisory tuning record, authority where present
```
Static prior (no-measurement profiles) — prefer **repack** ONLY when ALL hold; else **block-dot**:
1. **no ggml hand-tuned VLEN-native kernel** for `quant` at this VLEN (q4_K@VLEN128 has hand-tuned asm
   `ggml_vec_dot_q4_K_q8_K_vl128` -> FALSE -> decline), AND
2. **block-dot fallback is "heavy"** (q4_0 block-dot = nibble + per-block vredsum + scattered reads ->
   repack out-streams it; q8_0 block-dot is LEAN, 1 vwredsum/block -> nothing to remove -> FALSE -> decline), AND
3. **(VLEN == 128) OR (mRegime == Prefill)** (the q4_0@K1-VLEN256 decode loss 0.74x -> FALSE at
   VLEN256 decode -> decline -> tie; prefill GEMM keeps repack regardless of VLEN, 5.68x).

**Measurement is the authority where available** (the `RepackPathMeasurement` record): a measured
ratio < 1.0 for a cell forces block-dot even if the static prior says repack, and vice-versa. The
static prior is the fallback for un-measured profiles, mirroring `selectGenericSchedule`'s
"measured-best (still-legal record) else static argmin" (`RVVScheduleDescriptorRegistry.cpp:214–220`).

This prior is a small static table keyed `(quant, VLEN-class, M-regime)` matching the win/loss matrix
above; it is capability-derived, not magic constants — encode (1) as a per-quant
`ggmlHandTunedVLENNativeExists(quant, VLEN)` fact and (2) as a per-quant `blockDotIsHeavy(quant)` fact.

**Latent-mispick caveat (M>>1 unmeasured):** rule 3 keeps repack for ANY prefill cell, so
**q4_0@K1-VLEN256 *prefill* is auto-kept though it is never measured** (only the decode cell was). One
line of honesty: measurement governs where present; unmeasured K1-prefill defaults to repack and is a
known latent mispick risk — flag it, don't claim it as validated.

### Q4 — Is `low_precision_resource` the right carrier? **PARTIAL: right to EXPRESS, wrong to SELECT.**
`low_precision_resource.*` (`lib/Plugin/RVV/EmitC/RVVEmitCRouteMetadata.cpp:116+`) is a **MIRROR /
artifact-metadata output** — it *describes* a selection already made (`candidate_set`,
`selected_candidate`, `candidate_count`, `legal_candidate_count`, `selected_candidate_index`,
`selection_reason`, `route_family_plan`). The selection it mirrors is computed by
`RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp` /
`RVVContractionSelectedBodyRealizationOwner.cpp` over `RVVLowPrecisionContractionResourceCandidate`
(`include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`), whose fields are **resource SHAPE**
(sourceSEW/LMUL, productLMUL/EMUL, accumulatorCount, unrollFactor, tail/mask, packingLayout) of ONE
primitive. **There is NO field for algorithm-path / op-kind.** So:
- **Wrong place to ENUMERATE or SELECT** the path — its candidates are LMUL shapes, not algorithms.
  (This is the same channel gate4 + the e2e packed-i4 oracle consume — it's the within-algorithm
  resource axis, per project MEMORY low-precision-resource-is-N3-evidence. Don't conflate.)
- **Right channel to EXPRESS/AUDIT** the path decision once made: extend the mirror with a new
  family field — `tcrv_rvv.low_precision_resource.contraction_algorithm` = `repack|block-dot` + a
  `path_selection_reason` — so the chosen path is audited in the same artifact stream as the shape
  selection. **A NEW selection input + new audit field is needed regardless** — the existing
  candidate struct cannot carry the choice. (Quick check whether `routeFamilyPlanID` /
  `route_family_plan` already distinguishes a repack vs block-dot family before adding a field; if it
  does, ride it — but the SELECT logic is new either way.)

### Q5 — How VLEN and M reach selection: F3 (VLEN via `deriveMinimumVLEN(march)`) and F2 (M via gemv/gemm
op identity). Both already compile-time. No blocker on inputs.

---

## ORDERED build-incremental steps (each COMPILES; default-identity first = no behavior change)

> NOTE the gating reality (F4): a pure-MLIR `default-identity rewrite pass then flip` is INSUFFICIENT
> for the e2e correction, because declining repack must also stop the **weight repacking** (a
> producer/build decision). Steps split into (A) compiler-expressed decision [pure tcrv, testable in
> `tcrv-opt`] and (B) producer/build-honored packing [the emit-plan + injection harness].

**S0 (pure, no behavior change).** Add `ContractionAlgorithm` + `selectContractionAlgorithm` as a
free function in a new `RVVContractionPathSelection.{h,cpp}` under `lib/Plugin/RVV/`. Wire NOTHING.
Default the static table so every current cell returns its CURRENT path (q4_0/q8_0/q4_K @128 -> Repack
= today's always-repack). Unit-test the table. COMPILES; identity.

**S1 (run the decision at the producer / emit-plan, identity-default).** Call
`selectContractionAlgorithm` in the emit-plan that chooses which contraction variant `.mlir` to feed
the compiler (the adapter-emit harness, S4 layer). With the S0 identity table it returns the current
path for every cell -> the compiler is handed the SAME variant as today -> byte-identical emitted code.
Assert the harness logs the decision per cell. COMPILES; identity. (NOTE: an in-compiler two-case
`exec.dispatch` is **deferred / optional** — it is audit-only over the already-committed op, see Q1; do
NOT build it as the gate. If pursued later it needs a new abstract un-committed contraction request op.)

**S2 (audit channel).** Extend the `low_precision_resource` mirror
(`RVVEmitCRouteMetadata.cpp`) with `contraction_algorithm` + `path_selection_reason` from the S1
decision (or reuse `route_family_plan` if it already distinguishes — verify first). Pure metadata
append; existing CHECK lines unaffected (new lines only). COMPILES.

**S3 (flip the selection per the measured table).** Change the S0 static table so the prior of Q3
fires: q8_0@128 -> BlockDot, q4_K@128 -> BlockDot, q4_0@K1-VLEN256-decode -> BlockDot; q4_0@128 and
all Prefill -> Repack. Now S1 selects the block-dot dispatch case for the declined cells. Update lit
tests to assert the declined cells select block-dot. COMPILES; first BEHAVIOR change (which variant
the compiler commits to).

**S4 (producer/build honor — the F4 coupling, the part that moves e2e).** In the emit-plan / adapter
harness (the `cmake_inject.py` / `transform_repack.py` family used for the live ggml injection), gate
BOTH (a) which adapter is emitted/injected and (b) whether the model weights are repacked to `x16`, on
the S1 decision. Declined cell -> emit no repack adapter, leave ggml native block-dot, do NOT repack
weights. This is where the q4_0@K1 0.74x->tie e2e correction actually lands. (Build-harness change,
not `lib/`.)

---

## VALIDATION harness

**Compiler/IR (lit, in `tcrv-opt`, no hardware):**
- New tests on the two-case dispatch: q4_0@VLEN128 selects `repack`; q8_0@128 & q4_K@128 select
  `block-dot`; q4_0 with K1/VLEN256 `-march` selects `block-dot`; q4_0 prefill (gemm op) selects
  `repack`. Assert via the audit field (`contraction_algorithm`).
- Byte-exact guard (project MEMORY build-incremental-unreliable): after S1 with identity table, the
  existing repack/block-dot lowering tests must remain byte-identical (forced/clean rebuild;
  block-dot fingerprint truth = `f810ce6b`, full = `cb04b219`).

**Micro (hardware):**
- `ssh rvv` (VLEN128): q4_0@128 repack still ~1.22x vs ggml block-dot (win kept); q8_0@128 &
  q4_K@128 declined == ggml block-dot byte/cycle parity (tie, no regression).
- `ssh k1` (VLEN256): q4_0 decode declined == ggml native (the 0.74x cell now ties, no 0.74x).

**E2E (the one change that CAN move e2e — it changes which kernel runs; report SEPARATELY from micro,
per MEMORY kernel-wins-dont-transplant):**
- coherent-llama decode on the **declined regimes**: q4_0@K1-VLEN256 must show the 0.74x regression
  GONE (ties ggml) — a real e2e correction. q8_0@128 / q4_K@128 declined must not regress vs ggml.
- q4_0@VLEN128 e2e win preserved (the kept cell). Frame as "selection removed 3 mis-picks", never as a
  new Win-B.

---

## RISK / effort / blockers

- **BLOCKER (registry):** `RVVScheduleDescriptorRegistry` cannot express two competing algorithms for
  one op (one key -> one shape-candidate descriptor). CONFIRMED. Use `tcrv.exec.dispatch`/variant
  instead; do not force the gate into the registry.
- **BLOCKER (the dominant cost — weight-layout/producer coupling, F4):** declining repack must also
  stop weight-repacking (`from_float_to_mat` -> `x16`), a model-load/build decision, not a lowering
  op-swap. The pure-MLIR gate (S0–S3) is necessary but NOT sufficient for the e2e correction; S4
  (build/producer honor) is where effort concentrates and is the only part that moves e2e.
- **NON-blocker (good news):** both selector inputs are already compile-time facts — VLEN via
  `deriveMinimumVLEN(march)` (F3), M-regime via gemv/gemm op identity (F2). No runtime-M problem.
- **`low_precision_resource`:** reuse as the AUDIT carrier (new `contraction_algorithm` field), not as
  the selector — its candidates are LMUL shapes (Win-A axis), no path field.
- **Effort:** S0–S2 (express + audit, identity) ~1 focused session (pure tcrv, lit-testable). S3
  (flip table) small. S4 (producer/build honor + e2e re-measure on rvv & k1) is the multi-session
  bulk — owns the weight-packing gate and the hardware e2e runs. Total: the SELECTION logic is small;
  the cost is the packing coupling + honest dual (micro+e2e) validation on two profiles.
- **Honesty guard (project MEMORY):** report as selection-correctness only. Declining ties ggml (no
  speedup); the q4_0@K1 e2e improvement is a *regression removal*, not a new win. No Win-B reclaim.

## Critical files (file:line)
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` :1380 materializeRVVVectorBinarySourceKernel, :1456
  createDispatch — **elementwise source materializer; its two-case dispatch is RVV-vs-scalar (same
  algorithm), NOT the contraction gate. Does NOT consume repack/block-dot ops. NOT the gate site.**
- `lib/Plugin/RVV/RVVScheduleDescriptorRegistry.cpp` :37–144 one-key->one-shape-descriptor — **the
  confirmed registry blocker** (Win-A shape axis, not path).
- `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp` :78 deriveRepackHalfLanes, :126 module.walk —
  **the wrong layer (width-only, post-op)**.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteMetadata.cpp` :116+ low_precision_resource mirror — **audit
  carrier (add `contraction_algorithm`)**.
- `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyLowPrecisionResource.cpp` /
  `RVVContractionSelectedBodyRealizationOwner.cpp` :380–510 — shape-candidate select+mirror (Win-A;
  no path field).
- `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` — `RVVLowPrecisionContractionResourceCandidate`
  (resource-shape struct, no algorithm-path field).
- `lib/Conversion/RVV/RVVToEmitC.cpp` :334–406 kBlockDotKernels recognizer table — **faithful lowering
  by op identity (proves the path is pre-decided)**.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` :919/:1597 kind validators (repack vs block-dot kinds).
- `include/TianChenRV/Dialect/Exec/IR/ExecOps.td` :134 VariantOp / :271 DispatchOp — **the reuse target
  for expressing the choice**.
- Weight-layout coupling (F4): `weight_block_stride=288`/`weight_interleave=16` (repack `x16`) vs
  `weight_block_stride=18` (plain block-dot) — `test/Conversion/RVV/rvv-to-emitc-repack-gemv-q4-0-q8-0.mlir:33`
  vs `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot.mlir:34`.
- Producer/build honor (S4): `.../fedora-rvv07/rvv07-perfile-build/{cmake_inject.py,transform_repack.py,arch_repack.cpp}`.
