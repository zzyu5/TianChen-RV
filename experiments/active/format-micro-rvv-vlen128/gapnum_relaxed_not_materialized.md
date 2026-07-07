# [GAP-NUM] relaxed-tier iq4_xs / tq1_0 — NOT EXECUTABLE (relaxed body not materialized), 裁决二.2

**Directive 裁决二.2** asked for a `relaxed` vs `strict` (+ vs factory) measurement on iq4_xs / tq1_0 with an
honest tax decomposition (gap = numeric-tax + true-diff) and a per-cell ULP bound. **This is not executable
in the current tree: the relaxed numerics tier has no materialized body for iq4_xs or tq1_0.**

## Source-of-truth (code, not opinion)
- `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h:2764` — `enum class RVVNumericsTier { Strict, Relaxed };`
  The `Relaxed` variant is the §5 reassociation form (premultiplied scales, vfmacc/vfredusum), FAIL-CLOSED
  OFF by default, admitted only behind the `numerics.reassoc_ok` policy fact (`RVVOps.td:9923`).
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:7182-7188` — the emit-time surface gate REJECTS a relaxed
  request everywhere except the q8_0 path, fail-closed:
    > `numerics_tier "relaxed" is currently materialized only for the q8_0 (sumi_times_scales)
    >  deferred-ordered flat block-dot body; every other path must use the strict tier (later steps)`
- `grep -c 'relaxed|numericsTier|numerics_tier'` in the two emitters that actually produce iq4_xs / tq1_0
  code — `RVVToEmitCGridCodebook.cpp` (iq4_xs, grid-codebook) and `RVVToEmitCTernaryBinary.cpp` (tq1_0, ternary)
  — is **0 / 0**. There is no relaxed emit branch for either format.

## Consequence
- iq4_xs (constructed super-block **grid-codebook** decode) and tq1_0 (constructed super-block **ternary**
  block-dot) emit only the **strict** byte-exact body. A `numerics_tier="relaxed"` request on either fail-closes
  (`notifyMatchFailure`) — it does NOT silently emit strict (the code explicitly forbids "IR-says-relaxed /
  emit-does-strict" as a lie). So there is **no relaxed variant to time against strict**, and therefore **no
  "numeric tax" and no per-cell ULP bound to record** for these two formats. The directive's premise (that a
  relaxed tier exists for iq4_xs/tq1_0) does not hold at HEAD.

## What IS true / where a relaxed-vs-strict tax study would be valid
- The relaxed tier is materialized **only for q8_0** (sumi_times_scales deferred-ordered flat block-dot). A
  relaxed-vs-strict tax + ULP-bound study is executable **there** (q8_0), not on iq4_xs/tq1_0. That would be a
  separate立项 (build the relaxed body for grid-codebook / ternary first, behind a declared ULP bound + the
  `numerics.reassoc_ok` gate), then measure. No board run was spent on item 4 — the blocker is structural
  (emitter maturity), verifiable by the greps above, and needs no hardware to establish.
- Correctness footing already on record for these two formats (batch2c numerics triage, non-defect): iq4_xs =
  ours ULP0 vs the ggml `*_generic` oracle (50000/50000), the FACTORY-vl128 side reassociates off the oracle;
  tq1_0 = ours ULP0 vs oracle every regime. i.e. ours is already the FAITHFUL (strict) side — the very thing a
  relaxed tier would trade away. (`gap_triage_batch2c/numerics_triage_divergent.txt`.)

## Verdict
[GAP-NUM] relaxed iq4_xs/tq1_0 = **BLOCKED (relaxed body not materialized for these paths; construction-queue,
not a measurement)**. Honest, board-free, code-verifiable. Recorded so no future session re-attempts a board run
against a non-existent relaxed variant.
