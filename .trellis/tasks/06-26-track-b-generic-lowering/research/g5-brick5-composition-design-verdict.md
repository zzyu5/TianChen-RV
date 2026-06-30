# G5 brick-5 design verdict — q4_K super-block loop scaffold as a generic composition op

> Track B (G3) design note. **DESIGN/ANALYSIS only — no code changed, nothing built.**
> Question: should "brick 5" (the q4_K super-block FOR-loop scaffold + per-block setup)
> become a generic composition op, or is q4_K body auto-construction meaningfully
> complete at the 6 component ops? Decides the boundary of the mechanical-extraction phase.

## TL;DR — **VERDICT (B): DO NOT BUILD brick 5.**
q4_K body decomposition is meaningfully complete at the 6 component ops + the existing
whole-body op `GgmlBlockDotQ4KQ8KOp`. A loop-scaffold op is degenerate mechanically
(duplicates `GgmlBlockDotQ4KQ8KOp`), byte-identical plumbing as a composition refactor
(no measurable behavior, no cross-format generalization), and a *different* (N3
selection) axis as a higher abstraction. `GgmlBlockDotQ4KQ8KOp` already IS "the" generic
high-level construction entry; the bricks already prove the body decomposes into named,
typed, verifier-checked, byte-exact generic dataflow steps (the capability flip on brick 3).
Next Track-B value is **G5 beat**, not more bricks.

---

## 0. Op-name correction (load-bearing — the prior note + the task prompt are both imprecise)
Both this task's prompt and the prior scope-check (`g3-q4k-feasibility.md:46`) say
`emitQ4_KQ8_KBlockDot` "lowers from the existing recognized op `GgmlBlockDotQ4KQ8KAux32Op`."
**That is the wrong op.** Two distinct first-class ops exist:

- `GgmlBlockDotQ4KQ8KAux32Op` — mnemonic `tcrv_rvv.q4_k_q8_k_aux_partial`
  (`RVVOps.td:6874`). The **K4a INTEGER-CORE PARTIAL** only (nibble unpack + bit-dance +
  scaled i32 dot → writes `aux32[8]` + 16 decoded scale/min bytes). fp32 fold / MIN term /
  ABI are **out of scope** by its own description (`:6889-6890`). Recognized by
  `isQ4_KQ8_KAux32PartialBody` (`RVVToEmitC.cpp:1564`), lowered by `emitQ4_KQ8_KAux32Partial`
  (`RVVToEmitCKQuant.cpp:1695`), dispatched at `RVVToEmitC.cpp:406-407`.
- `GgmlBlockDotQ4KQ8KOp` — mnemonic `tcrv_rvv.q4_k_q8_k_block_dot` (`RVVOps.td:6980`).
  The **COMPLETE K4b full block dot** (integer core + deferred fp32 fold + MIN term + ABI
  `*s`, `:6986-7047`). This is what `emitQ4_KQ8_KBlockDot` actually lowers from
  (`RVVToEmitCKQuant.cpp:2747-2751`), recognized by `isQ4_KQ8_KBlockDotBody`
  (`RVVToEmitC.cpp:1578`), dispatched at `RVVToEmitC.cpp:408-409`.

The degeneracy conclusion is unchanged, but **strengthened**: there are *already two* peer
whole-thing ops (the integer-core partial AND the full block dot). A mechanical brick-5 would
be a *third* op doing what the full block dot already does.

---

## 1. Layering map (file:line cites)

Three real levels, plus the brick witnesses:

```
LEVEL 3  (abstract, algorithm-UNCOMMITTED source — N3 axis)
  GgmlQuantContractionOp  "tcrv_rvv.quant_contraction"        RVVOps.td:4076
    - block-quantized contraction REQUEST, NOT committed to a kernel ALGORITHM
    - option-2 stage-A abstraction over {block-dot, repack-gemv}; q4_0 ONLY today
    - NOT in the EmitC recognizer table; lowered by --tcrv-rvv-lower-quant-contraction
      BEFORE EmitC, via IDENTITY, to a concrete contraction op (:4108-4113)

LEVEL 2  (concrete, format-shaped source op — the q4_K construction entry)
  GgmlBlockDotQ4KQ8KOp    "tcrv_rvv.q4_k_q8_k_block_dot"      RVVOps.td:6980
    - ONE typed generic op carrying ALL q4_K structural facts as attrs
      (strides 144/292; offsets d@0 dmin@2 scales@4 qs@16 q8@4 bsums@260; QK_K=256)
      + the capability knob integer_core_lmul {mf2,m1,m2}  (:7062-7074)
    recognize: isQ4_KQ8_KBlockDotBody                        RVVToEmitC.cpp:1578
    dispatch:  RVVToEmitC.cpp:408-409
    lower:     emitQ4_KQ8_KBlockDot                          RVVToEmitCKQuant.cpp:2743

LEVEL 1  (the hand-written monolith body = the "brick-5 scaffold")
  emitQ4_KQ8_KBlockDot                                       RVVToEmitCKQuant.cpp:2743-3095
    nb = n/QK_K                                              :2845
    declare scratch  aux8[256] / utmp[4] / sums8[8]          :2850-2886
    declare+zero carried accs  sums(vec) / sumf(scalar) ONCE :2888-2918
    blockBaseValue lambda (per-block addr arithmetic)        :2920-2928
    super-block FOR loop  (emitc::ForOp over nb)             :2981
      xb,yb per-block bases                                  :2989-2992
      emitQ4_KSuperBlockAux32Core  (bricks 1-3 helper)       :2998
      load dy                                                :3004-3030
      emitQ4_KMinTermBsumsDot      (brick 4a helper)         :3040
      emitQ4_KSumsFoldScaleD       (brick 6 helper)          :3050
      emitQ4_KMinTermSubtract      (brick 4b helper)         :3061
    post-loop  emitQ4_KHorizontalFold (brick 7 helper)       :3082
    *s = sumf                                                :3087-...

LEVEL 0  (the 6 component ops = byte-exact WITNESSES, NOT the production path)
  Q4KNibbleUnpackOp       RVVOps.td:6391   (brick 1)
  Q4KScaleMinBitDanceOp   RVVOps.td:6461   (brick 2)
  Q4KScaledDotOp          RVVOps.td:6537   (brick 3 — carries the mf2<->m2 flip, :6585-6602)
  Q4KMinTermOp            RVVOps.td:6629   (brick 4)
  Q4KSumsFoldScaleDOp     RVVOps.td:6715   (brick 6)
  Q4KHorizontalFoldOp     RVVOps.td:6798   (brick 7)
    each: own recognizer + own emit; emit calls the SAME shared helper the monolith calls
    (byte-identity by construction), and lowers STANDALONE declaring its OWN scratch + a
    local sink as the observable (e.g. RVVOps.td:6757-6758, 6828-6831).
```

**The key structural fact for this decision:** the monolith calls **raw shared helper
functions** (`emitQ4_KSuperBlockAux32Core`, `emitQ4_KMinTermBsumsDot`,
`emitQ4_KSumsFoldScaleD`, `emitQ4_KMinTermSubtract`, `emitQ4_KHorizontalFold`) — **not** the
component ops. The 6 ops do not appear in the production lowering path at all; they are
first-class *witnesses* that each region is a legitimate generic op with a byte-exact
standalone lowering. The actual q4_K kernel is still produced by a hand-written monolith
(`RVVToEmitC` is "recognize + delegate to the hand-written emitter"; PRD §现状基线 line 20).

---

## 2. Degeneracy of a *mechanical* brick-5 — CONFIRMED

"Brick 5" = the super-block FOR-loop scaffold + per-block setup. In the code that is exactly
the body of `emitQ4_KQ8_KBlockDot` **minus** the helper calls: the `nb` divide
(`:2845`), the scratch decls (`:2850-2886`), the carried-accumulator decls (`:2888-2918`),
the `emitc::ForOp` container (`:2981`), and the per-block address lambda (`:2920-2928`).

A mechanical "extract the loop that calls all the component helpers" op would re-emit the
scaffold **and** call helpers 1-7 in order — i.e. it would reproduce the *entire*
`emitQ4_KQ8_KBlockDot` body. Its lowering would be byte-identical to `GgmlBlockDotQ4KQ8KOp`'s
lowering (same helpers, same order). So it is a **second op that does what
`GgmlBlockDotQ4KQ8KOp` already does, with no novelty** = DEGENERATE. The byte-identity gate
holds vacuously; the result adds nothing. (Confirmed; matches the prior scope-check
`g3-q4k-feasibility.md:46`, modulo the op-name correction in §0 — and the redundancy is in
fact worse than that note states, since two whole-thing ops already exist.)

---

## 3. Is there a NON-degenerate, value-adding brick 5? — candidate analysis

### (i) A composition op fed by a HIGHER-level generic source, auto-constructing the body by composing bricks 1-7
Two sub-forms:

- **(i-b) via a `linalg`/`vector` contraction or `GgmlQuantContractionOp`.** A higher-level
  generic source *already exists*: `GgmlQuantContractionOp` (`RVVOps.td:4076`). But by its own
  description it is an **algorithm-SELECTION** abstraction (block-dot vs repack-gemv), lowered
  by a **pre-EmitC stage-A pass via IDENTITY** to a concrete op (`:4108-4113`), q4_0-only
  today. Extending it to q4_K would add **N3 path-selection** value (choose q4_K block-dot vs a
  future q4_K repack-gemv) — a *different novelty axis on a different op*, and **moot today**
  because no q4_K repack target exists, so it would be a single-target identity lowering. It is
  not "brick 5" and not Track-B body construction.
  - Is `GgmlBlockDotQ4KQ8KOp` "already the generic high-level op," or ggml-shaped/too-specific?
    It is format-specific **by necessity**: q4_K's 6-bit bit-dance, the 144/292 strides, the
    byte offsets, QK_K=256 are intrinsic to the format. No format-agnostic descriptor can
    *yield* the q4_K bit-dance without a q4_K-specific lowering rule — and that rule **is**
    `emitQ4_KQ8_KBlockDot` + the bricks. Pushing the source op higher only adds a dispatch
    layer above the construction logic; the per-format construction stays. So
    `GgmlBlockDotQ4KQ8KOp` already serves as "the" high-level construction entry for the kernel.

- **(i-c) a loop op whose body literally nests the 6 component ops, then lowers each.** The
  component ops' standalone lowerings declare their **own** scratch + sinks (witnesses). To
  compose them into a working kernel they must share scratch (aux8/utmp/sums/sumf/aux32) and
  thread real SSA values — a **second, different lowering mode per brick**, substantial new
  machinery. The output is **still byte-identical** (same helpers underneath). Net: it
  relocates hand-coding from "emitter calls helpers" to "pass builds brick IR + threads
  scratch," with **zero measurable behavior change and zero cross-format generalization** (the
  brick order, the 4a/6/4b interleave, the scratch layout are all q4_K-specific facts that stay
  hand-authored). This is the same trap the PRD's must-NOT names ("把 recognize+delegate 当
  '自动构造'") and that `n1-substrate-emission-not-maturity` warns about: byte-exact emitter
  reshaping ≠ maturity.

### (ii) Refactor the monolith to BUILD its body by composing the 6 component ops (not raw helpers)
Same as (i-c): byte-identical plumbing. The monolith already gets byte-identity from sharing
helpers with the ops (`g3-q4k-feasibility.md` GATE1/GATE2 results). Re-expressing "call helper"
as "emit op then lower it" changes no emitted C, adds a per-brick composed-lowering mode, and
buys structural purity only. No novelty over (a) `GgmlBlockDotQ4KQ8KOp` or (b) bricks 1-7.

### (iii) Other framings
- **Brick 5 as a standalone peer op** (recognize + lower in its own `with_vl`) is just §2's
  mechanical form = degenerate.
- The genuinely open Track-B value is **not** another brick. It is **G5
  (cm-shape-through-Track-B)**: a wider shape that the generic mechanism *synthesizes* and that
  *measurably beats* ggml on board (PRD §In-scope G5, §DoD). That is a beat/measurement axis,
  orthogonal to whether the q4_K body is expressed as a monolith, helpers, or composed ops.

---

## 4. VERDICT — (B) DO NOT BUILD brick 5

**q4_K body auto-construction is meaningfully complete at the 6 component ops.** A loop-scaffold
op adds no novelty over (a) the existing whole-body op `GgmlBlockDotQ4KQ8KOp` and (b) the proven
components 1-7:
- mechanically it duplicates `GgmlBlockDotQ4KQ8KOp` (and the integer-core partner
  `GgmlBlockDotQ4KQ8KAux32Op`) — degenerate;
- as a composition refactor (i-c / ii) it is byte-identical plumbing — no measurable behavior,
  no generalization, the `n1-substrate` non-maturity trap;
- as a higher abstraction (i-b) it is the N3 algorithm-selection axis on `GgmlQuantContractionOp`
  — a different novelty, moot today (no q4_K repack target).

**What `GgmlBlockDotQ4KQ8KOp` already provides as "the" high-level construction entry:** a single
generic typed first-class op that carries every q4_K structural fact as a typed attribute plus
the `integer_core_lmul` capability knob (the mf2/m1/m2 flip), from which the compiler constructs
the complete RVV body. That already satisfies "construct the body from a generic high-level op."
The 6 component ops then prove that body decomposes into named, typed, verifier-checked,
byte-exact generic dataflow steps — including the real VLEN/LMUL capability flip on brick 3
(`Q4KScaledDotOp.integer_core_lmul`, `RVVOps.td:6585-6602`).

**Honesty note:** the production q4_K lowering is still the hand-written monolith; the bricks are
byte-exact *witnesses*, not the production path. So "auto-construct, not hand-write" is *proven
decomposable* for q4_K, not *realized in production* — and brick 5 in any form above does NOT
change that (it either duplicates the monolith or rebuilds it as byte-identical IR).

**Boundary call:** the clean mechanical-extraction phase is COMPLETE at 6/7. Stop here.

**Next Track-B step (do NOT read (B) as "stuck"):** pursue **G5 (cm-shape-through-Track-B)** —
synthesize a wider shape via the generic mechanism and *measure* a beat on board (PRD §G5/DoD).
Extending `GgmlQuantContractionOp` to q4_K is a legitimate but *separate* N3 selection move, and
only becomes non-moot once a q4_K repack-gemv target exists.
