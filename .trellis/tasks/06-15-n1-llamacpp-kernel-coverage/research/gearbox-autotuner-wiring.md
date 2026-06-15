# Research: How the Gearbox autotuner works, and how to wire Q4_0 kernel-shape selection into it

- **Query**: STRAND 1 — how does our N3 Gearbox autotuner enumerate/prune/select today, how does a chosen schedule get stamped onto the emitted body, and what is the cleanest path to turn the Q4_0 block-dot's `integer_core_lmul` (today a MANUAL attr), plus a multi-block factor, into AUTOTUNER OUTPUT?
- **Scope**: internal (C++ plugin code)
- **Date**: 2026-06-15

> **Headline:** The autotuner's enumerate→prune→select machinery already exists and is
> exactly the right shape, but it is **two disconnected halves** today: (1) the LMUL-rung
> enumerate/prune/select functions in `RVVGearboxSchedule.h` are **unit-tested standalone
> and NOT wired into any live realization** (the header says so explicitly at lines
> 1893–1899), and (2) the live "schedule→body attr stamping" path that DOES run lives in
> `RVVContractionSelectedBodyRealizationOwner.cpp` but is for the **i8 direct-contraction**
> op, not Q4_0. The Q4_0 op's `integer_core_lmul` is supplied **by hand in the source MLIR**
> — no pass computes it. The clean fix: add a small **Q4_0 schedule producer pass** that
> mirrors the existing enumerate→prune→select *structure* and the existing source→destination
> *attr-stamping* path, stamping the chosen schedule onto the `q4_0_q8_0_block_dot` op (which
> `emitQ4_0Q8_0BlockDot` already reads for `integer_core_lmul`).
>
> **Selection target — set by the STRAND 2 measurements
> (`research/q4_0-winning-shape-experiment.md`):** the robust resource-best shape is
> **`(integer_core_lmul = m1, multi_block = 2)`**, which is **~11% faster than the current
> hand-set single-block m1** (1398 vs 1564 ns; mf4 anchor is ~4.9× slower still). So **two**
> knobs are worth selecting: `integer_core_lmul` (mf4→**m1**, fewest reductions/block) AND a
> **2-block unroll** (the robust latency-overlap sweet spot). multi_block=4 and the
> deferred-i32 form REGRESS (1.30x / 1.61x) and are pruned. No robust shape beats ggml
> (ggml stays ~20% ahead — the only ggml-beating shapes drop VLEN-robustness, forbidden by
> the ship constraint), but the autotuner discovering a robust shape **~10% better than the
> manual 140946cc anchor** is the achievable N3 lamp. This justifies adding one new bounded
> attr — `multi_block_factor` (selected value 2) — alongside the existing `integer_core_lmul`.

---

## 1. How the autotuner enumerates / prunes / selects today

The reference implementation is the N3 max-legal-LMUL selector for the low-precision
widening product-reduction contraction, in
`include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h:1879–2085`. It is a textbook
**enumerate → prune → select**:

- **Enumerate** — `enumerateRVVLowPrecisionAccumulatorLMULRungs(vectorRegisterBudget,
  reserveRegisterCost)` (h:1952). Walks the candidate i8 source rungs `{mf4, mf2, m1, m2}`,
  widens each to its i16 product LMUL (EMUL 2×, `getRVVNextWiderLMUL`, h:1931) and then to
  its i32 accumulator LMUL (EMUL 4×), and builds an `RVVLowPrecisionLMULRung` (h:1904) with
  the per-rung register-cost facts: `accumulatorRegisterCost`, `productRegisterCost`,
  `reserveRegisterCost`. Rungs whose widening would exceed the m8 LMUL cap are dropped.
- **Prune** — each rung's `isLegal` is set by the **vreg budget fact**:
  `acc_regs + product_regs + reserve ≤ vectorRegisterBudget` (h:1976–1979), at A=1 (single
  accumulator). `vectorRegisterBudget` is the architectural 32-vreg file size
  (`kRVVLowPrecisionResourceVectorRegisterBudget = 32`, h:746), **not** a hand-picked magic
  constant.
- **Select** — `selectRVVLowPrecisionMaxLegalAccumulatorLMULRung(rungs)` (h:1992) returns
  the **widest legal** rung (max `accumulatorRegisterCost` among `isLegal`), or `nullopt` if
  every rung was pruned. The rule "widest legal at A=1" encodes the measured fact that wider
  LMUL hides the accumulate-chain latency and A>1 buys nothing but vreg waste.

A **second, distinct** cost model exists for the i16 single-widening dot-reduce family:
`enumerateRVVDotReduceDeferredWideLMULRungs` / `selectRVVDotReduceDeferredWideMaxLegalLMULRung`
(h:2040–2081), where `accumulatorLMUL == productLMUL` (the i32 product *is* the deferred
accumulator, one widening), so its prune is `acc_regs + reserve ≤ budget` (no separate
product group). The two families deliberately have **separate enumerators** because their
peak-live cost structures differ (h:2006–2020). *(Aside: the Q4_0 "deferred-i32-accumulator"
robust shape STRAND 2 measured — strip-accumulate into an i32 vector then one reduce — has
exactly this deferred-wide cost structure; STRAND 2 measured it slower than ggml AND slower
than the simple 2-block form, so it is not a target here.)*

**Honest scope (header lines 1893–1899):** these rung functions are *unit-tested standalone*
and **NOT yet wired into the live realization path**. The realized typed body is still pinned
to the narrow i8mf4 per-iteration-vwredsum chain. So selecting a wide rung today would not
realize a faithful body. They are the *shape* of the autotuner, proven in isolation.

## 2. How a SELECTED schedule gets stamped onto the emitted body (the live path)

The live "schedule → body attr" stamping path is the **realization owner**:
`lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, specifically
`materializeLowPrecisionResourceRealizationAttrs(...)` (cpp:340–909). Its mechanism:

1. **Read the planning facts off the `source` op** — it requires the candidate-set attr,
   the `vector_register_budget` fact, the selected-candidate id, etc. (cpp:356–407). The
   facts are the `tcrv_rvv.low_precision_resource.*` string/integer attributes declared en
   masse in `RVVGearboxSchedule.h:48–383`.
2. **Re-build + re-select the candidate enumeration** via
   `buildRVVLowPrecisionProductReductionResourceCandidates(...)` +
   `selectRVVLowPrecisionProductReductionResourceCandidate(...)` (cpp:375–381), then
   cross-checks the source op's recorded selection against the freshly computed one
   (fail-closed: stale/over-budget facts are a hard error, cpp:421–425, 519–541).
3. **Validate** the selected candidate against the provider primitive facts (intrinsic
   spellings, layouts, store-VL) — `validateLowPrecisionResourceCandidatePrimitiveFacts`
   (cpp:194–297) and the long block cpp:435–812.
4. **STAMP the realized schedule onto the `destination` op** via `destination->setAttr(...)`
   (cpp:814–907): the realization producer/decision, candidate counts + selected index,
   producer/consumer scopes, and crucially the **realized schedule knobs** —
   `realized_unroll_factor` (cpp:834), `realized_vsetvl_region_count` (cpp:837),
   `realized_peak_live_vector_groups` (cpp:840), plus region indices/phases. For packed-i4 it
   also stamps the schedule-decision contract/decision/reason (cpp:896–906).

So the pattern is: **facts on the source op → re-derive+validate the selection →
`setAttr` the chosen knobs onto the destination op**, which the EmitC lowering then reads.
This is the template the Q4_0 path should follow. (`copyLowPrecisionResourceAttrs`,
cpp:36–41, shows the simpler "forward all resource attrs" helper for ops that just relay.)
Note `realized_unroll_factor` already exists as a stamped knob — the Q4_0 multi_block factor
is the same kind of fact.

## 3. Where the Q4_0 `integer_core_lmul` comes from TODAY (the gap)

- **The op** `tcrv_rvv.q4_0_q8_0_block_dot` is defined in
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3873–3946`. `integer_core_lmul` is an
  `OptionalAttr<StrAttr>` (td:3937), documented as "a bounded resource/scheduling fact — the
  *how*, not the *what*" (td:3910–3921): mf4 default / m1 anchor, byte-exact either way.
- **The verifier** (`lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:793, 852`) only **bounds**
  it to `"mf4"|"m1"` (fail-closed: "m2" etc. rejected). It does not *choose* it.
- **The EmitC lowering** `emitQ4_0Q8_0BlockDot` (`lib/Conversion/RVV/RVVToEmitC.cpp:3755`,
  read site at 3789–3794) reads `blockDot.getIntegerCoreLmul()`, defaults to `"mf4"` when
  absent, and maps it to the i8 source / i16 product LMUL spellings. It is a pure consumer.
  It does **not** today emit a multi-block outer loop (block step is always 1).
- **Who supplies `integer_core_lmul`?** **Nobody computes it.** It is written **by hand in
  the source MLIR** (`test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-m1.mlir:33`,
  `test/Dialect/RVV/q4-0-q8-0-block-dot-dataflow.mlir:141`). There is **no producer pass**
  between the op being built and the lowering. This is precisely the "set MANUALLY (commit
  140946cc)" the task calls out — a hand-placed attr, not an autotuner output.

So the gap: the enumerate/prune/select *engine* exists (§1), the stamp-onto-body *path*
exists (§2) — they don't meet on the Q4_0 op. And per STRAND 2 the live Q4_0 schedule space
that matters is **two robust knobs** — `integer_core_lmul` (exists) and a **multi_block
factor** (must be added) — both selected at the resource-best robust values (m1, 2).

## 4. Concrete wiring proposal (turn the manual attrs into Gearbox-selected outputs)

**Add a Q4_0 schedule producer pass** (a new resource-planning owner, modeled on
`RVVContractionSelectedBodyRealizationOwner.cpp`) that runs *before* `RVVToEmitC`, on the
`tcrv_rvv.q4_0_q8_0_block_dot` op, and does:

1. **Enumerate** the Q4_0 shape set `{ integer_core_lmul ∈ {mf4, m1} } × { multi_block ∈
   {1, 2, 4} }` (unroll fixed at 1). Mirror the *structure* of
   `enumerateRVVLowPrecisionAccumulatorLMULRungs` (h:1952): per shape record the reduction
   count (mf4 → 4 `vwredsum`/block at VLEN=128; m1 → 1), the peak-live vreg footprint
   (STRAND 2 measured: every robust shape ≤ 5 vregs — multi_block doesn't blow the budget
   here because each block's strip loop is single-trip and its product/reduce retire before
   the next), and a **VLEN-robustness flag** (true iff the shape keeps the inner strip loop +
   `sumi`-carry seed).
2. **Prune** by two facts: (a) the vreg-budget fact `peak_live ≤ 32`
   (`kRVVLowPrecisionResourceVectorRegisterBudget`, h:746) — never binding here (all robust
   shapes ≤ 5 vregs); and (b) the **VLEN-robustness fact** — drop any shape whose
   `robustness flag` is false. This robustness prune is the Q4_0-specific addition over the
   i8 enumerators; it is what keeps the selection inside the ship constraint (the
   ggml-beating non-robust mb4 is pruned here, correctly).
3. **Select** the resource-best legal shape per STRAND 2's measured ordering: among the
   robust shapes, `(integer_core_lmul=m1, multi_block=2)` is fastest (1398 ns, 1.20x ggml,
   ~11% faster than single-block m1); m1×1 = 1564, m1×4 = 1525 (regresses), mf4×* ≈ 5.2x ggml.
   The rule is "fewest reductions/block (→ m1), then the multi_block with the best measured
   latency-overlap (→ 2), pruning regressing wider unrolls" — structurally the same
   "resource-best legal, with a measured tiebreak" shape as
   `selectRVVLowPrecisionMaxLegalAccumulatorLMULRung` (h:1992).
4. **Stamp** the chosen knobs onto the op via `setAttr`, the same mechanism as
   `materializeLowPrecisionResourceRealizationAttrs` (cpp:814–907): set `integer_core_lmul`
   (exists — the clean plug-in point), set the **new** `multi_block_factor` attr (=2), plus
   the resource-provenance attrs (`realized_peak_live_vector_groups`,
   `realized_unroll_factor`-analogue for multi_block, candidate set, selected candidate,
   selection reason, vreg budget) for the audit trail / N3 evidence, exactly as the i8 path
   stamps them.

**What changes downstream:** two small changes.
- `RVVOps.td`: add `multi_block_factor : OptionalAttr<I64Attr>` to the op (mirroring
  `integer_core_lmul`'s "bounded how-not-what" pattern), and the verifier
  (`RVVDialectWideningOps.cpp:852`) bounds it to `{1, 2}` (the robust legal set; 4 is a
  measured regression, fail-closed I7).
- `emitQ4_0Q8_0BlockDot` (RVVToEmitC.cpp:3755): when `multi_block_factor == 2`, step the
  outer `emitc.for` by 2, emit the two per-block strip-loop integer cores + the two
  `sumi`-carry `vwredsum`s + the two in-order folds + an `nb%2` scalar-tail block — the
  structured, VLEN-robust mb2 form proven in STRAND 2. When absent/1 it emits the current body
  byte-identically (additive constraint).

**Why this is the cleanest path:**
- `integer_core_lmul` is **already** the attr the lowering keys on for the LMUL axis — the
  producer pass just *computes* it instead of the IR author hand-writing it.
- The enumerate→prune→select code is a small, self-contained mirror of the existing,
  unit-tested rung functions (h:1952–2081) — same `budget`/`isLegal`/`resource-best-legal`
  shape, with one added prune (VLEN-robustness) and one measured tiebreak (multi_block=2).
- The stamping reuses the proven `setAttr`-the-knobs pattern (cpp:814–907), so the N3
  audit-trail attributes come along for free and make the choice **provable resource-aware
  selection**, not a manual constant. The multi_block factor is the same kind of stamped knob
  as the existing `realized_unroll_factor`.

**Honest framing:** this is a **new selection path** for a new op, not flipping an existing
switch. The existing rung functions can't be reused verbatim (different cost structure;
they're also not yet live-wired for *any* op). What we reuse is the *pattern* (enumerate the
candidate space, prune by the 32-vreg fact **+ VLEN-robustness**, select resource-best-legal
with a measured tiebreak, stamp the chosen knobs via setAttr) — plus one existing attr
(`integer_core_lmul`) as a ready plug-in point. The lift: the new producer pass + 1 new
bounded op attr (`multi_block_factor`) + its verifier bound + the mb2 emission branch in
`emitQ4_0Q8_0BlockDot`. The selected output `(m1, multi_block=2)` is **~10% faster than the
hand-set 140946cc m1 anchor** — so the pass makes the kernel both *derived/provable* AND
*measurably better*, which is the N3 lamp on a real llama.cpp kernel (even though it does not
beat ggml — see STRAND 2's honest ceiling).

---

## Files / lines cited

| File:line | Role |
|---|---|
| `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h:1879–2004` | i8 LMUL rung enumerate/prune/select (the reference engine); :1893–1899 = "NOT live-wired" honesty note |
| `…/RVVGearboxSchedule.h:1916–1943` | `getRVVLMULRegisterFootprint` / `getRVVNextWiderLMUL` (the LMUL arithmetic the cost model uses) |
| `…/RVVGearboxSchedule.h:2040–2081` | 2nd-family (i16 dot-reduce) deferred-wide enumerate/select (separate cost model; matches the Q4_0 deferred-i32-acc shape, which STRAND 2 measured slower than both ggml and mb2) |
| `…/RVVGearboxSchedule.h:746` | `kRVVLowPrecisionResourceVectorRegisterBudget = 32` (the prune budget fact) |
| `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp:340–909` | the live "facts→re-select→validate→`setAttr` knobs onto body" stamping path (the template) |
| `…RVVContractionSelectedBodyRealizationOwner.cpp:814–907` | the `destination->setAttr(...)` schedule-knob stamping block (incl. `realized_unroll_factor` cpp:834 — the model for the new multi_block knob) |
| `…RVVContractionSelectedBodyRealizationOwner.cpp:375–381` | `build…Candidates` + `select…Candidate` (live enumerate+select) |
| `include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3873–3946` | `q4_0_q8_0_block_dot` op def; :3937 = `integer_core_lmul` attr; :3910–3921 = its "how-not-what" doc (the pattern the new `multi_block_factor` attr follows) |
| `lib/Conversion/RVV/RVVToEmitC.cpp:3755–3814` | `emitQ4_0Q8_0BlockDot`; :3789–3794 = reads `getIntegerCoreLmul()` (pure consumer; block step always 1 today) |
| `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp:793, 852` | verifier bounds `integer_core_lmul` to `mf4`/`m1` (fail-closed; the model for bounding the new `multi_block_factor`) |
| `test/Conversion/RVV/rvv-to-emitc-q4-0-q8-0-block-dot-m1.mlir:33`, `test/Dialect/RVV/q4-0-q8-0-block-dot-dataflow.mlir:141` | proof that `integer_core_lmul` is supplied BY HAND in source IR today (no producer) |

## Caveats / Not found

- No existing pass computes `integer_core_lmul` for the Q4_0 op — confirmed by grepping all
  of `lib/`/`include/` for build/set sites; the only writers are hand-authored test MLIR.
  There is no `multi_block_factor` attr at all today (the op always emits block step 1).
- The existing rung enumerators are unit-tested but not invoked by any live pass (per the
  header's own note and confirmed by grepping their call sites — they appear only in the
  header and standalone tests). So "wire the autotuner in" is a real lift for *both* the i8
  path and a fortiori the new Q4_0 path; this research scopes only the Q4_0 wiring.
- **The selection target is `(integer_core_lmul=m1, multi_block=2)`** — set by the STRAND 2
  measurements (`microbench_robust_variants_n4096_n8192.txt`). It is the robust resource-best
  shape (~11% faster than the manual single-block m1) but **does not beat ggml** (~20% short);
  ggml is beatable only by the non-robust mb4, which the robustness prune correctly removes.
  An earlier draft of this file mis-ranked the robust shapes (claimed multi_block was not a
  robust lever); corrected here from the raw data — robust multi_block=2 is a real win over
  the single-block anchor.
- This file does NOT propose the producer pass's exact registration/ordering relative to the
  RVV pass pipeline — that is an implementation detail for the `implement` agent (the pass
  must run after the op is built and before `--tcrv-rvv-lower-to-emitc`).
