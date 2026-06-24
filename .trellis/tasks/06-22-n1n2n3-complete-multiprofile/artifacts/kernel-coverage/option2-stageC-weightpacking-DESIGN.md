# Option-2 STAGE C — compiler-driven plain→x16 weight materialization + the real producer (DESIGN, 2026-06-24)

READ-ONLY design. NO `lib/` edits. Stage C of "option-2": the DEEP, e2e-MOVING step.
After B, the abstract `GgmlQuantContractionOp` carries PLAIN `block_q4_0` weights (stride 18)
and the repack-selected cell is audited `path_materialization="deferred-stage-c"`, still
emitting the block-dot body. Stage C must (1) make the repack-selected request a REAL, runnable
`GgmlRepackGemvQ40Q80Op` consuming x16 weights (stride 288), and (2) wire the REAL producer that
authors the abstract op in the live llama.cpp pipeline (vs the stage-A/B hand-authored fixtures).
This is the ONLY stage that moves end-to-end performance. Companion to
`option2-stageA-abstract-op-DESIGN.md`, `option2-stageB-selection-DESIGN.md`,
`path-selection-tune-DESIGN.md` (F4 = the weight-layout coupling).

---

## 0. THE LOAD-BEARING FACT C IS BUILT ON (confirmed by inspection — reshapes the whole verdict)

The task framing says "compiler-driven plain→x16 materialization." Inspection of the EMITTER,
the EMITTED kernel signature, and the llama.cpp `repack.cpp` machinery shows the plain→x16
transform is **NOT something the compiler authors for the e2e path — ggml OWNS it at model-load
time, and the compiler-emitted kernel CONSUMES the already-packed bytes.** Evidence:

- **The repack-GEMV emitter reads ALREADY-packed bytes; it does NOT pack.**
  `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:2734+` reads the x16 facts off the op
  (`weightStride=288`, `weightQuantOffset=32`, `weightInterleave=16`, `half_lanes`) and emits
  disjoint `vle8` sub-loads of the repacked `qs` — with the explicit comment *"The repacked
  nibbles already carry the ^0x88 offset-binary bias"* (:2799). No in-kernel interleave, no
  in-kernel `^0x88`.
- **The emitted kernel SIGNATURE takes a pre-packed `vx`.**
  `…/emit-repack-gemv/emitted-repack-gemv.cpp:4`:
  `tcrv_emitc_ggml_repack_gemv_q4_0_q8_0_kernel_…(size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)`
  — `vx` is the already-interleaved `block_q4_0x16` base. There is NO packing prologue argument.
- **ggml packs plain→x16 at MODEL-LOAD time via its extra-buffer-type machinery.**
  `llama.cpp/ggml/src/ggml-cpu/repack.cpp`: `init_tensor` (:4727) stamps
  `tensor->extra = ggml_repack_get_optimal_repack_type(tensor)`; `set_tensor` (:4739) calls
  `tensor_traits->repack(...)` → for q4_0-16x1 that is `repack_q4_0_to_q4_0_16_bl` (:3358/:3939)
  → `make_block_q4_0x16` (the `^0x88` bake, verbatim-inlined at
  `…/emit-repack-gemv/verify_emitted_gemv.cpp:63`). This runs ONCE at load, before any compute.
- **The build harness LEAVES ggml's packing untouched.** `transform_repack.py` (97 lines) is a
  surgical TEXT rewrite of `arch/riscv/repack.cpp` that swaps only the **compute-kernel body**
  (injects `if (vlenb*8==128) tcrv_emitc_…(); return;` into `ggml_gemv_q4_0_16x1_q8_0`). It does
  NOT touch `make_block_q4_0x16` or `get_optimal_repack_type`. `cmake_inject.py` (44 lines) only
  sets per-file `-march=rv64gc_xtheadvector -D__riscv_zvfh` on `repack.cpp`/`arch/riscv/repack.cpp`/
  `ggml-cpu.c`. `arch_repack.cpp` (189 lines) is the replacement `arch/riscv/repack.cpp` that
  `#include`s the emitted `.inc` and dispatches to it.

**Consequence:** "compiler-driven plain→x16 materialization" as the e2e mechanism is a
**category framing error in the task prompt** (honest correction): the compiler does NOT
materialize the x16 bytes on the e2e path. ggml does, at load. What the compiler drives is
**which compute kernel runs over those bytes**, and — the actual F4 cost — **the build/load
cooperation that decides WHETHER a tensor is given the x16 (repack) buffer type at all.**

---

## 1. WHERE the plain→x16 packing lives TODAY (the map)

| Layer | What it does | File:line | Compiler-owned? |
|---|---|---|---|
| ggml load-time repack | `init_tensor` stamps repack buffer type; `set_tensor` calls `make_block_q4_0x16` (the ^0x88 bake) | `repack.cpp:4727/4739/3358/3939` | **NO — ggml owns it** |
| ggml repack-type SELECTION | `get_optimal_repack_type`: VLEN256 → `q4_0_16x1` (:4593 `case 256`); **VLEN128 → `case 128: break // TODO` (:4592) = NO repack upstream** | `repack.cpp:4528–4595` | **NO — gated by `__riscv_zvfh` + `ggml_cpu_has_riscv_v()`** |
| compute kernel (the swap) | `ggml_gemv_q4_0_16x1_q8_0` body → injected `tcrv_emitc_…` call | `arch/riscv/repack.cpp:206` / harness `transform_repack.py:48` | the EMITTED kernel is compiler-owned; the dispatch swap is build-harness |
| the EMITTED kernel | reads pre-packed x16 `vx`, no packing | `RVVToEmitCBlockQuantLinear.cpp:2674+` | **YES (compiler) — but consumer, not producer of x16** |

**The VLEN128-vs-VLEN256 asymmetry is the crux of F4 and was confirmed:**
- **VLEN256 (K1): ggml AUTO-repacks q4_0 to x16** (`case 256` returns `q4_0_16x1`). So a
  K1-decode "decline to block-dot" cell (the 0.74x→tie regression-removal) requires **SUPPRESSING**
  the load-time repack so plain stride-18 bytes exist for ggml's native `vec_dot_q4_0_q8_0`.
- **VLEN128 (C920): ggml does NOT repack q4_0 upstream, and force-defining `__riscv_zvfh` ALONE
  does NOT fix it.** The `case 128: { break; }` sits INSIDE the `#if defined __riscv_zvfh` block
  (`repack.cpp:4589–4597`): force-defining the macro makes the `switch` COMPILE and lets `case 256`
  (K1) return the traits, but **`case 128` still `break`s → nullptr → no q4_0 repack at VLEN128**.
  `supports_op` (:4779) gates on `get_optimal_repack_type` being non-null, so this kills the whole
  VLEN128 repack dispatch. The function-local `static q4_0_16x1_q8_0` trait DECL registering ≠ the
  `switch` RETURNING it. The fedora-rvv07 `ENGAGED`-at-128 markers (`llama-e2e-rvv07.log:97`) prove
  it WAS enabled in that build — so a 4th edit (beyond the 3 named harness files) must have flipped
  `case 128 → return q4_0_16x1`; that edit is NOT in the archived harness here (lost or in the
  fedora-era `repack.cpp` snapshot). **HONEST DEFAULT for C: enabling VLEN128 repack-keep is a NEW
  build-harness edit C4 must add — patch `case 128` to return `&q4_0_16x1_q8_0` (a one-line ggml
  source edit), not merely a `-D__riscv_zvfh`.**

So "decline" = make `get_optimal_repack_type` return the non-repack outcome for that tensor;
"keep" = make it return `q4_0_16x1`. **Both directions are a LOAD-TIME, per-TENSOR buffer-type
decision — not an in-IR op swap, not a compiler-emitted packing kernel.**

---

## 2. The plain→x16 materialization MECHANISM — (a) / (b) / (c)?

**Honest answer: (c) — a build/load-harness cooperation the compiler only TRIGGERS, NOT an
in-compiler packing kernel.** Justification, ruling out (a) and (b):

- **(a) compiler-emitted packing prologue at load — REJECTED for the e2e path.** The compiler
  CAN emit a `make_block_q4_0x16`-equivalent EmitC kernel (it already emits the consumer kernel;
  the producer is strictly simpler — a byte interleave + `^0x88`). BUT on the e2e path ggml has
  ALREADY packed the weights at load via its OWN `make_block_q4_0x16`; a second compiler-emitted
  packer would be redundant AND would have to REPLACE ggml's `repack()` callback (a deeper ggml
  surgery than the compute-kernel swap). It buys nothing e2e: the bytes are identical. **(a) is
  real and demonstrable in ISOLATION (the partial-C milestone, §6) but is NOT the e2e mechanism.**
- **(b) in-IR transform — REJECTED, structurally impossible at this layer.** The abstract op
  carries a runtime `weight_base` SSA *pointer*; the bytes behind it are produced at model-LOAD,
  long after the compiler has emitted C and exited. An MLIR pass cannot interleave bytes that do
  not exist at compile time. The IR can only *assert* the layout (change the op + attrs); it
  cannot *perform* the packing. (This is exactly why the bridge in §3 is trivial and the
  materialization is not.)
- **(c) build/load cooperation the compiler TRIGGERS — ACCEPTED (the e2e reality).** The
  compiler's role is to (i) emit the repack-consumer kernel (done) and (ii) emit, as a
  side-artifact of the repack-SELECTED decision, the **build directive** that the producer/harness
  honors: "for this (tensor, VLEN, quant) give it the x16 repack buffer type AND inject the
  emitted kernel into ggml's dispatch." The compiler DRIVES (it decides + emits the kernel + emits
  the directive); the build/load harness EXECUTES (enables the repack buffer type, runs
  `make_block_q4_0x16` at load). The honest phrase is **"compiler-DRIVEN, harness-EXECUTED"** —
  not "compiler-materialized."

**Where the code goes (mechanism (c)):**
- The compiler side is DONE (the emitter + B's audit attr `path_materialization`). C adds the
  **emit-plan → harness directive**: a small generated manifest (e.g.
  `tcrv_repack_plan.json`: `{quant, vlen_class, m_regime → {repack: bool, kernel: <emitted .inc>}}`)
  the compiler writes alongside the emitted `.inc`, derived from B's `contraction_algorithm` +
  `path_selection_reason` audit attrs.
- The harness side is the EXISTING `cmake_inject.py`/`transform_repack.py` family, EXTENDED to:
  (i) honor the manifest's `repack` flag by enabling/suppressing the `get_optimal_repack_type`
  outcome (the VLEN128 force-define for keep; a `case`-suppress patch for K1 decline), and
  (ii) inject the named emitted kernel into the dispatch only for kept cells.

---

## 3. THE BRIDGE — repack-selected `GgmlQuantContractionOp` → valid `GgmlRepackGemvQ40Q80Op`

This is the IN-IR half, and it is **TRIVIAL** — precisely because the materialization (§2) lives
OUTSIDE the IR. In the Stage-B pass `RVVLowerQuantContractionPass`, the `Repack` branch (which B
left as an audited block-dot stub) is replaced by an actual `create<GgmlRepackGemvQ40Q80Op>`:

```
case Repack:
  half = deriveRepackHalfLanes(minVLEN);   // RVVRepackStripWidthMaterialization.cpp:78 (8@128, 16@256)
  create<GgmlRepackGemvQ40Q80Op>(
      weight_base,        // SAME SSA pointer the abstract op carried (the IR cannot tell
                          //   plain from x16 — see below; the bytes are x16 at RUNTIME by §2)
      activation_base, output, element_count,
      /*row_count=*/c1,   // GEMV: nr==1 (decode); a constant-1 index
      column_count,       // the nc the abstract op CARRIED — this is WHY stage A carries nc
      /*output_row_stride=*/c0, vl)
    with attrs:  kind="ggml_repack_gemv_q4_0_q8_0", scale_model="dual-fp16-per-block-d_x.d_y",
                 qk=32, weight_block_stride=288, activation_block_stride=34,
                 weight_quant_byte_offset=32, activation_quant_byte_offset=2,
                 weight_interleave=16, half_lanes=half,
                 (optional) integer_core_lmul="mf2"|"m1";
  erase abstract op;
```

The repack-gemv verifier (`RVVDialectWideningOps.cpp:1718`) pins exactly these (288/34/32/2/16,
`half_lanes∈{8,16}` dividing 16) — the `create` above satisfies it fail-closed.

**The load-bearing honesty point (advisor-locked):** `weight_base` is the SAME SSA pointer value.
**The IR cannot distinguish a plain-stride-18 base from an x16-stride-288 base** — both are a
`const uint8_t *`. The op's `weight_block_stride=288` attr is an ASSERTION about bytes that the
HARNESS (§2) must have made true at load. So this bridge does NOT "materialize" anything; it
COMMITS the IR to the x16 contract and TRUSTS the harness to have packed. **If the harness does
not enable the x16 repack buffer type for that tensor, the emitted kernel reads stride-288 over
stride-18 bytes → garbage.** The bridge is correct ONLY in lockstep with §2's harness directive.
This is the irreducible coupling.

---

## 4. THE REAL PRODUCER — what authors the abstract op in the live pipeline

**Stage A found NO `rewriter.create<…ContractionOp>` anywhere in `lib/`** — every concrete
contraction op is hand-authored in `.mlir` fixtures, and upstream by the external adapter-emit
harness. So the abstract `GgmlQuantContractionOp` has **no producer in the codebase today**; the
A/B fixtures hand-author it. The real producer is a **NEW component**.

**What it must do:** for each q4_0 `MUL_MAT` weight tensor in the live model, emit a
`tcrv.exec.variant` holding a `GgmlQuantContractionOp` with the plain facts (stride 18, the
quant/m_regime/nc the model exposes), feed it to `tcrv-opt --tcrv-rvv-lower-quant-contraction
--tcrv-rvv-lower-to-emitc`, and collect the emitted `.inc` + the §2 manifest. This is exactly the
role the existing **adapter-emit harness** (`emitted_adapter_gemv.cpp` author) plays today, but
upstream of the abstract op instead of the concrete one.

**What it takes to build it:** the producer is the SAME class of artifact as the existing
`cmake_inject.py`/`transform_repack.py`/`arch_repack.cpp` harness — a build-side Python/C++ tool,
NOT `lib/` MLIR. It must enumerate the model's quantized mul_mat tensors (their quant type, dims →
VLEN-class via the target `-march`, decode-vs-prefill is per-CALL not per-tensor — see §5 risk),
author the abstract-op `.mlir`, drive `tcrv-opt`, and wire the result into the ggml build. Effort
is dominated by the model-enumeration + the ggml-build integration, NOT by the MLIR.

---

## 5. THE DEEPEST RISK — per-TENSOR layout vs per-CALL M-regime (the load-time boundary)

`get_optimal_repack_type` is keyed by **TENSOR** (type + dims + VLEN) and runs ONCE at load. But
**M-regime (decode vs prefill) is per-`mul_mat`-CALL on the SAME weight tensor.** One tensor → ONE
layout for its entire lifetime. Therefore:

- **q4_0 @ K1-VLEN256 decode→block-dot (plain) and prefill→repack (x16) CANNOT both hold on one
  tensor copy.** The B selection table keeps repack for ALL prefill (fact 3) but declines K1-VLEN256
  decode — these two selections demand DIFFERENT load-time layouts of the SAME tensor. They are
  **NOT independently composable.** The decode-decline "no pre-pack" validation cell (§6) implies
  the tensor is PLAIN at load → its prefill calls on that run CANNOT use repack. Presenting
  decode-decline and prefill-keep as separately achievable on one model is a category error.
- **Honest resolution options (all costly):** (i) pick ONE layout per tensor by the DOMINANT
  regime (decode dominates token generation → plain at K1 → lose prefill repack); (ii) keep TWO
  copies of the weight tensor (memory cost); (iii) accept the B "latent mispick" flag and pin K1
  to whichever the e2e measurement favors. **This is the load-time-boundary risk the task asked
  to surface, and it has no clean in-compiler fix** — it is inherent to ggml's per-tensor buffer
  type.

**Second-deepest risk — the repack e2e blocker is GEMM/PREFILL-scoped, NOT decode (corrected).**
The C920 seal defect is **isolated to the GEMM (prefill) repack path, not the GEVM (decode)
path** (`FINDING.md:212–228`, MEMORY fedora-rvv07-e2e-seal-open): **GEVM/decode RVV0.7 is bit-exact
and coherent on the C920** (the q4_0 decode hot path runs the emitted RVV0.7 GEVM, validated
numerically, tg128≈3.71 t/s), and the repack-kept **decode** win is RECORDED REAL (path-selection
doc: "q4_0@VLEN128, M==1 repack keep 1.22x→2.6x e2e"; MEMORY kernel-wins-dont-transplant lists
"q4_0 GEVM 1.22→2.6×" as a TRANSMITTED e2e win). The broken path is **GEMM/prefill** — either the
`m1→i16m2→i32m4→f32m4` chain-shift emission defect (`FINDING.md:225`) or the interleaved
activation mat-quantizer `ggml_quantize_mat_q8_0_4x8` (the q8_0x4 GEMM stream, NOT the plain q8_0
decode stream the GEVM verifier pins). The coherent landing was GEMM→scalar `_generic` (correct
prefill) + GEVM→emitted RVV0.7 (validated decode). **So C's repack-KEPT DECODE win stands; only the
repack-KEPT PREFILL (GEMM) e2e is gated on a ggml/emitter GEMM fix C does not need for the decode
deliverable.** Do NOT blanket-claim the decode 1.22→2.6x is blocked; the seal-OPEN issue is a
GEMM/full-seal routing matter MEMORY tracks separately.

---

## 6. E2E VALIDATION PLAN (the ONLY e2e-moving stage — report micro and e2e SEPARATELY)

Per project MEMORY (kernel-wins-dont-transplant, winc-structural-null): kernel microbench and e2e
decode are SEPARATE claims; never transplant.

**Cell 1 — q4_0 @ K1-VLEN256 decode 0.74x→tie (decline→block-dot, NO pre-pack). [`ssh k1`]**
- The compiler-driven selection picks `block-dot` (B reason `…vlen256-decode-k1-loss`). The
  harness directive (§2) must SUPPRESS ggml's `case 256` auto-repack so the K1 tensor stays PLAIN
  stride-18 → ggml's native `vec_dot_q4_0_q8_0` runs.
- GATE: coherent-llama decode on K1 shows the 0.74x regression GONE (ties ggml native, no 0.74x).
  This is a **regression REMOVAL**, not a new win (honesty guard). This is the most defensible
  e2e-moving result and aligns with the `FINDING.md` "off-repack = coherent" reality.

**Cell 2 — q4_0 @ VLEN128 decode repack-KEPT. [`ssh rvv` C920]**
- Selection picks `repack`; harness ENABLES the VLEN128 x16 buffer type (the `case 128 → return
  q4_0_16x1` ggml patch from §1/§5, plus per-file vector march + `__riscv_zvfh` force-define) +
  injects the emitted GEVM kernel.
- GATE (micro, `ssh rvv`): emitted q4_0 GEVM bit-exact vs ggml generic 16x1 (norm=0 — shown).
  GATE (e2e DECODE): the q4_0@128 decode repack-kept win is RECORDED REAL (1.22→2.6x, GEVM
  coherent — §5 correction); this cell IS achievable and is a genuine e2e-moving keep. The
  **prefill (GEMM)** repack-kept e2e is the part gated on the GEMM-emission/quantizer fix (§5),
  so report the DECODE keep as the deliverable and flag prefill-keep as GEMM-blocked.

**Compiler/IR gate (lit, no hardware, the in-compiler proof that selection moves correctly):**
- Forced/clean rebuild (MEMORY build-incremental-unreliable). After C, the repack-selected fixture
  emits a REAL `tcrv_rvv.repack_gemv_q4_0_q8_0` op (CHECK its 288/16/32 attrs + `half_lanes`),
  NOT the deferred block-dot stub. The decline fixtures still emit block-dot byte-identical
  (`f810ce6b`). A CHECK that the manifest names the kept cell's kernel.

---

## 7. PRECISE ORDERED BUILD STEPS (each with its gate)

> Split into IN-COMPILER (C1–C2, lit-testable in `tcrv-opt`, no hardware) and HARNESS/PRODUCER
> (C3–C5, the F4 build/load coupling, the e2e-moving + multi-session bulk).

**C1 — the in-IR bridge (repack branch creates the real op).** Replace B's audited block-dot
stub in the `Repack` branch of `RVVLowerQuantContractionPass` with `create<GgmlRepackGemvQ40Q80Op>`
(§3): wire `half_lanes` from `deriveRepackHalfLanes(minVLEN)`, the carried `nc`, the 288/34/32/2/16
attrs. GATE: a repack-selected lit fixture emits the real `repack_gemv` op passing its verifier;
the decline fixtures unchanged (block-dot `f810ce6b` intact, forced rebuild). COMPILES; lit-only;
**byte-exact on decline, NEW op on keep — first IR divergence from B.** No e2e yet.

**C2 — the manifest emit (compiler-driven directive).** From B's `contraction_algorithm` +
`path_selection_reason` audit attrs, emit a `tcrv_repack_plan` manifest (§2) listing per-cell
`{repack: bool, kernel}`. GATE: lit/unit test that the manifest matches the selection table.
COMPILES; no behavior change in emitted C; the directive is a side-artifact.

**C3 — the real producer (NEW build-side component).** Build the tool (§4) that enumerates the
model's q4_0 mul_mat tensors, authors the abstract-op `.mlir`, drives `tcrv-opt`, collects the
`.inc` + manifest. GATE: on a real gguf, the producer emits the same `.inc`/`.mlir` the
hand-authored A/B fixtures do (parity with today's adapter-emit). Build-harness, not `lib/`.

**C4 — harness honors the manifest (the F4 load coupling — the e2e-moving step).** Extend
`cmake_inject.py`/`transform_repack.py` to: (i) for KEEP cells at VLEN128, **patch
`get_optimal_repack_type`'s `case 128: break` → `return &q4_0_16x1_q8_0`** (a NEW one-line ggml
source edit — force-define alone is INSUFFICIENT, §1/§5) plus the per-file vector march +
`__riscv_zvfh`; for KEEP at VLEN256 confirm `case 256` auto-returns; then inject the named emitted
kernel; (ii) for DECLINE cells, SUPPRESS `get_optimal_repack_type` for that tensor so plain bytes
survive (the §5 K1 decode case). GATE: build engages the right path per cell (the `ENGAGED`/
off-repack markers in `llama-e2e-rvv07.log` style). This is where the load-time boundary (§5) bites.

**C5 — e2e measure on two profiles (separate from micro).** `ssh k1`: Cell 1 (decode decline,
0.74x→tie, regression removal). `ssh rvv`: Cell 2 decode repack-kept (the RECORDED 1.22→2.6x GEVM
win, §5 — achievable) + micro bit-exact (done). Prefill (GEMM) repack-kept is gated on the §5
GEMM-emission fix — flag, don't block the decode deliverable on it. GATE: K1 decode coherent + no
0.74x; C920 decode keep ≥ recorded; report micro and e2e SEPARATELY; no Win-B reclaim.

---

## 8. HONEST FEASIBILITY / EFFORT / PARTIAL-MILESTONE VERDICT (the most important output)

**Is C a clean in-compiler capability? NO — it is IRREDUCIBLY coupled to the build/load harness.**
The in-IR bridge (C1) is trivial and clean (a `create<repack_gemv>` the IR cannot even tell is
"wrong" — bytes are plain or x16 only at runtime). But the thing that ACTUALLY moves e2e — the
plain→x16 layout — is owned by ggml's load-time `make_block_q4_0x16` + `get_optimal_repack_type`,
gated by a build-side march/macro decision. The compiler can DRIVE (decide + emit kernel + emit
directive) but cannot MATERIALIZE. **Mechanism (c), not (a)/(b), is the honest e2e answer.** Any
framing that calls C "compiler-driven plain→x16 materialization" as if the compiler packs the
bytes e2e is over-claiming; the defensible claim is "compiler-DRIVEN selection + harness-EXECUTED
layout."

**TRUE effort: 3–5 focused sessions, gated on a ggml-side prerequisite outside our control.**
- C1 (in-IR bridge) ~½ session, lit-only. C2 (manifest) ~½ session.
- C3 (real producer) ~1–1.5 sessions — the model-enumeration + ggml-build integration is the bulk.
- C4 (harness honor, F4) ~1 session, but entangled with §5 — and the VLEN128 keep needs a NEW
  `case 128 → return q4_0_16x1` ggml patch (force-define alone insufficient; small but must be added).
- C5 (e2e two-profile) ~1 session of `ssh` runs — the K1 decode DECLINE and the C920 decode
  repack-KEPT (1.22→2.6x, recorded) are both achievable; only the PREFILL (GEMM) repack-keep is
  gated on the §5 GEMM-emission fix.

**The deepest risks (ranked):**
1. **Per-tensor layout vs per-call M-regime (§5).** One weight tensor → one load-time layout;
   decode-decline and prefill-keep on the same tensor are mutually exclusive. No clean in-compiler
   fix. This caps what "selection per (quant, VLEN, M-regime)" can actually deliver e2e.
2. **The repack-KEPT PREFILL (GEMM) path is not e2e-coherent on the C920 (§5) — but DECODE is
   fine.** The GEVM/decode repack-kept win is RECORDED REAL (1.22→2.6x, bit-exact); only the GEMM
   (prefill) repack path is broken (chain-shift / interleaved-quantizer defect). So C has TWO
   defensible e2e deliverables: the K1 decode DECLINE (0.74x→tie regression removal) AND the
   C920 decode repack-KEEP (1.22→2.6x). Only prefill-keep is GEMM-blocked. Do NOT collapse this
   into "decline is the only defensible claim."
3. **The load-time boundary itself.** "Decline/keep" is a per-tensor buffer-type decision made at
   load by a function (`get_optimal_repack_type`) the compiler can only INFLUENCE via build-side
   march/macro patches — the coupling the task suspected, confirmed.
4. **The producer is a NEW build component**, not a `lib/` pass; its cost is integration, not MLIR.

**A meaningful PARTIAL-C (the honest smaller milestone — RECOMMENDED build-now target):**
**"Compiler-driven plain→x16 materialization, demonstrated in ISOLATION on the microbench, via
mechanism (a)."** The compiler EMITS a `make_block_q4_0x16`-equivalent packing kernel (it already
emits the harder consumer kernel) and proves it BIT-EXACT against ggml's `make_block_q4_0x16`
using the EXISTING `…/emit-repack-gemv/verify_emitted_gemv.cpp` harness (which already inlines the
reference packer at :63). This:
- demonstrates the compiler CAN drive the materialization (mechanism (a) is real), satisfying the
  "compiler-driven" intent on a clean, isolated, lit+microbench path;
- needs NO llama.cpp producer (C3), NO load-time coupling (C4), NO hardware-e2e (C5), NO ggml
  repack-subsystem fix;
- is honestly scoped: it does NOT move e2e and does NOT resolve the §5 load coupling — it is the
  in-isolation proof, the analogue of stage A's byte-exact identity proof.
- effort ~1 session (emit the packer op+verifier+emitter on the consumer-emitter template, prove
  bit-exact vs the reference).

**Build-now vs multi-session recommendation:** ship **C1 (in-IR bridge) + the partial-C isolated
packer** as the honest, ~1.5-session in-compiler milestone that makes the repack-selected request
REAL in IR and proves the compiler can drive the packing in isolation. DEFER C3–C5 (the producer
+ harness honor + two-profile e2e) as the multi-session bulk, AND surface the e2e constraints
(§5 per-tensor/per-call layout + the VLEN128 `case 128` ggml patch + the prefill-GEMM defect)
BEFORE committing to the full e2e sweep. The achievable, defensible e2e-moving deliverables are
BOTH the K1 decode DECLINE (0.74x→tie regression removal) AND the C920 decode repack-KEEP
(1.22→2.6x, recorded); only the prefill-GEMM repack-keep is gated on a separate GEMM-emission fix.

---

## 9. CRITICAL FILES (file:line)
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:2734+` — repack-GEMV emitter; reads x16 facts
  off the op, NO in-kernel packing; :2799 "*repacked nibbles already carry ^0x88*" (proves consumer).
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:4266` `GgmlRepackGemvQ40Q80Op` — the C target op.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:1718` repack-gemv verifier — the exact x16 pins
  (288/34/32/2/16, `half_lanes∈{8,16}`) the bridge `create` must satisfy.
- `lib/Plugin/RVV/RVVRepackStripWidthMaterialization.cpp:78` `deriveRepackHalfLanes` — the
  `half_lanes` derivation C1 wires into the bridge.
- `llama.cpp/ggml/src/ggml-cpu/repack.cpp:3358/3939` `repack_q4_0_to_q4_0_16_bl`→`make_block_q4_0x16`
  (the ^0x88 bake = the plain→x16 transform ggml owns at load); :4727 `init_tensor`, :4739
  `set_tensor`→`repack()` (load-time trigger); :4528/4592/4593 `get_optimal_repack_type`
  (**VLEN128 `case 128: break // TODO` = no upstream repack; VLEN256 `case 256` = auto-repack** —
  the F4 decline/keep gate).
- `…/emit-repack-gemv/emitted-repack-gemv.cpp:4` — emitted kernel signature (pre-packed `vx`, no
  prologue arg); `…/emit-repack-gemv/verify_emitted_gemv.cpp:63` — inlined reference
  `make_block_q4_0x16` (the partial-C bit-exact oracle).
- `.../fedora-rvv07/rvv07-perfile-build/{cmake_inject.py(:44),transform_repack.py(:97),arch_repack.cpp(:189)}`
  — the build harness C4 extends (per-file march + kernel-swap; does NOT touch ggml's packing).
- `.../fedora-rvv07/FINDING.md:212–228` — **the defect is GEMM/PREFILL-scoped: GEVM/decode RVV0.7
  bit-exact + coherent (decode repack-keep win REAL); GEMM/prefill numerically broken (chain-shift /
  interleaved-quantizer). Coherent landing = GEMM→scalar + GEVM→emitted.**
- `.../fedora-rvv07/llama-e2e-rvv07.log:97` — the VLEN128 `ENGAGED` markers (proves a `case 128 →
  return q4_0_16x1` patch existed in that build; the §1/§5 force-define alone is NOT sufficient).
