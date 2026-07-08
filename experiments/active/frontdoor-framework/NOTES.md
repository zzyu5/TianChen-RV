# frontdoor-framework — method + tq2_0 first-point decomposition

C2 evidence line **layer B**: the cost to lift a decode family from a **direct-emit
bypass** to a **front-door typed-region CONSTRUCTION**, split so that the reusable
portion (paid once) is separated from the family-specific portion (paid per family).
The load-bearing C2 claim is that the split predicts a **decreasing marginal cost** for
the next decode family, and that the drop is dominated by **structural distance of the
decode leaf**, not by re-paying the apparatus.

## Decomposition rule (how a flip commit's LOC is split)

A front-door-ization flip touches the shared apparatus. Its insertions fall into three
tiers with different amortization:

1. **Reusable framework (tier-1; paid ONCE; next family pays 0):** the generalizations of
   the SHARED verifiers + dispatch that make the front door accept *more than one* decode
   family. Once present, a sibling family reuses them verbatim.
2. **Copy-adapt template (tier-2; per family, but ~copy):** the construction-function
   scaffold and the core-brick shell. Family-named, but a sibling copies the shape and
   edits only the decode leaf, so the *effective* re-pay is a fraction of these LOC.
3. **Family-specific decode leaf (tier-3; per family, re-paid in full if the decode
   differs):** the constants, the decode-brick math, and the emission decode body. A
   sibling with a *structurally distant* decode (e.g. base-3 vs base-2) re-pays most of it.

The ledger CSV collapses this to two columns for machine anchoring —
`reusable_framework_LOC` = tier-1, `family_specific_LOC` = tier-2 + tier-3 — and carries
`of_which_copy_adapt_template_LOC` = tier-2 so the *effective* next-family cost
(< `family_specific_LOC`) stays visible.

## tq2_0 / ternary — first point (commit 7a4250c5, C_construct 33→34)

**Gross (reproducible):** `git show 7a4250c5 --numstat -- lib/ include/` =
**+948 / −717 (net +231)** across six shared files:

| file | +add | −del | role in this flip |
|---|---:|---:|---|
| `lib/Plugin/RVV/RVVLowerQuantContraction.cpp` | 361 | 3 | front-door CONSTRUCTION (dispatch + two ternary construction fns + constants) |
| `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` | 250 | 378 | ternary core-brick verifiers + abstract/loop-body verify generalization − retired op verifiers |
| `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | 126 | 188 | ternary core-brick ODS − retired direct-emit op ODS |
| `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | 154 | 88 | ternary emission leaf (byte-exact re-home) − retired direct emitters |
| `lib/Conversion/RVV/RVVToEmitCInternal.h` | 38 | 29 | emitter decls |
| `lib/Conversion/RVV/RVVToEmitC.cpp` | 19 | 31 | dispatch wiring |

**Retirement offset:** the −717 deletions are almost entirely the **retired direct-emit
bypass** — `emitRepackGem{v,m}TQ20Q8K` / `isRepackGem{v,m}TQ20Q8KBody` /
`GgmlRepackGem{v,m}TQ20Q8KOp` (ODS + verifiers + emitters). This is a *bypass removal*
(the first bypass retirement of the G3 program), not a construction cost, so it is banked
separately and does not net against the front-door build.

### tier-1 — reusable framework (≈60 LOC; paid ONCE)

The amortizable "front door now admits a 2nd decode family" delta:

- **`GgmlQuantContractionOp::verify` family-dispatch** (`RVVDialectWideningOps.cpp`,
  hunks @@ −1683,13 +1831,27 @@ / @@ −1697,30 +1859,65 @@): the abstract-op verifier
  went from *"accept only scale_model `dual-fp16-per-block-d_x.d_y` (q4_0)"* to *"dispatch a
  decode FAMILY off the `scale_model` STRUCTURAL key (q4_0 ∨ ternary), keyed off a required
  WHAT attr, NOT the optional `quant` format label."* The family-branch skeleton (`isQ40Family`
  / ternary select) ≈ 15 LOC is tier-1; the ternary-specific fact pins in the same hunk are
  tier-3 (see below).
- **`TypedRepackGem{v,m}LoopBodyOp::verify` `isTernaryFold` predicate**
  (`RVVDialectWideningOps.cpp`, @@ −15572,12 +15429,19 @@ / @@ −15947,11 +15811,19 @@): the
  region verifier now admits a non-q4_0 fold family (`!isTernaryFold && scale_model != …` gate)
  ≈ 15 LOC.
- **`lowerOne` scale_model → decode-family dispatch + ternary fail-closed**
  (`RVVLowerQuantContraction.cpp`, @@ −221,12 +255,33 @@): the `isTernaryTQ20 ? …Ternary : …`
  fork + the I7 fail-closed that REJECTS a ternary request lacking a repack strip width (never
  silently mis-lowers ternary weights into a q4_0 block-dot) ≈ 30 LOC.

Next decode family (tq1_0) re-pays **0** of tier-1: the hooks are already in place (commit
msg: "钩子已就位").

### tier-2 — copy-adapt template (≈400 LOC; per family but ~copy)

- **`lowerToRepackGemvTernary` + `lowerToRepackGemmTernary`** construction bodies
  (`RVVLowerQuantContraction.cpp`, @@ −638,6 +693,309 @@, ≈309 LOC): build the typed
  `typed_repack_gem{v,m}_loop_body` region (ABI operands, block-fact attrs, region block-args,
  the sole in-region core brick with block_index / strip_row_offset anti-bypass ties, the
  accumulator-passthrough yield), plus the GEMM `nr`/`bs` runtime-ABI materialization. This is
  a near-clone of the q4_0 `lowerToRepackGem{v,m}` shape — a sibling copies it and edits the
  decode leaf, so the *effective* re-pay is a fraction (~20–30%).
- **`RepackGem{v,m}TernaryCoreOp` ODS + verifier shells** (`RVVOps.td` + `RVVDialectWideningOps.cpp`
  lines ~92–190, ≈100 LOC of shell): the op arity / attr surface / anti-bypass tie structure is
  reusable; only the decode_model string + decode-specific attr pins change per family.

### tier-3 — family-specific decode leaf (≈490 LOC; re-paid if decode differs)

- **ternary block-fact constants** (`RVVLowerQuantContraction.cpp`, @@ −112,6 …, ≈34 LOC):
  `kTernaryTQ20*` ×6 stride/offset + 2 `scale_model` strings + doc.
- **ternary verify fact-pins** inside `GgmlQuantContractionOp::verify` (the ternary
  stride/offset structural pins in the tier-1 hunk, ≈40 LOC).
- **ternary emission leaf** (`RVVToEmitCBlockQuantLinear.cpp` +152): the byte-exact ternary
  GEVM/GEMM emit body, re-homed under the core-brick identity (mostly a *move* of the retired
  direct emitter's body — byte-exact per the commit's oracle equality, so low genuinely-new
  math but full re-pay for a differently-decoded sibling).
- **decode/decl plumbing** (`RVVToEmitC.cpp` +19, `RVVToEmitCInternal.h` +38, ≈56 LOC).

## C2 prediction — the tq1_0 datapoint (to be measured at its flip)

Framework (tier-1) is **paid**; tq1_0 re-pays 0 there. The construction template (tier-2) is
**copy-adaptable**; tq1_0 re-pays only a fraction. The open question is tier-3: the commit
itself flags tq1_0 as a "base-3 双叶参数化 = 等体量第二轮" (equal-volume second round) — i.e.
its decode leaf (5 trits packed per byte in base-3, double-leaf unpack) is **structurally
distant** from tq2_0's 2-bit field extract, so its tier-3 is expected to be substantial, NOT
near-zero.

- **Prediction:** tq1_0 marginal ≈ **0 (tier-1) + fraction (tier-2) + substantial (tier-3)**
  → a LOWER total than tq2_0's +948 gross (framework + template amortized) but **not** a
  q5_K-grade near-free reuse, because the decode leaf is structurally far.
- This is the C2 headline from `docs/method/C2_marginal_cost_ledger.md` restated on the
  front-door axis: **marginal cost ∝ structural distance of the decode leaf to the covered
  primitive space, with the apparatus amortized to ~0.** The tq1_0 measurement will either
  confirm (tier-3 dominates, total drops mostly via tier-1/2 amortization) or refine it.

## labor proxy — 2 authoring passes (the emission↛construction discipline)

tq2_0's first authoring pass produced **emission only** — a hand-authored typed region in the
TEST, lowered byte-exact — which the adversarial-verify caught PRE-commit as *test-authored,
not compiler-CONSTRUCTED* (the region was written in the input IR, not produced by the pass).
The second pass added the real **CONSTRUCTION** (the pass CONSTRUCTS the typed region from the
abstract `quant_contraction`). This 2-pass structure is the coarse `labor_proxy_passes=2`
signal and is archived as the **first live [F-EMIT] / six-state enforcement case** in the
result-tables **T8 口径** discipline notes (`experiments/active/result-tables/MANIFEST.md`).
It is the reason "emission ≠ construction" is a load-bearing distinction for C_construct
(strong sense): the +1 to C_construct (33→34) is credited ONLY because the second pass made
the region pass-CONSTRUCTED.
