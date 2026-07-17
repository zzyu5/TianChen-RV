# Win-A (compiler-automatic Gearbox tune) — fresh re-measure + adversarial correction

**2026-06-22.** Re-measured the i16 max-LMUL ablation FRESH this session on ssh rvv
(directly observed, not archived) + 3 adversarial skeptics. Fresh stdout:
`fresh-lamp-3way-ssh-rvv-stdout.txt`. Board: riscv64, native clang 18.1.3, load ~0.33,
genuine-scalar TU objdump vector-op count = 0 (verified twice), all 6 CORRECTNESS ok.

## Fresh numbers (match archived → the microbench win is now directly-observed)
| n | wide_vs_scalar | wide_vs_periter (clean compiler ON/OFF) | wide_vs_narrow (vs competent naive RVV) |
|---|---|---|---|
| 257 | 4.26 | 6.70 | 3.01 |
| 256 | 5.53 | 8.06 | 3.00 |
| 1024 | 6.87 | 10.15 | 3.51 |
| 4096 | 7.52 | 11.13 | 3.78 |
| 16384 | 4.91 | 6.96 | 2.53 |
| 65536 | 4.14 | 6.11 | 2.06 |

Ratios match the 06-14 archive (n=4096 wide_vs_periter 11.13 now vs 11.16 then); absolute
ns ~2× faster (lighter board), ratios stable. **The N3 microbench ablation is real and
reproduced same-session.**

## CORRECTION (adversarial verdict 1, FATAL on the over-precise sub-claim)
My earlier framing "the ONLY thing that changes ON vs OFF is the selector picking the wide
**LMUL rung** (a pure LMUL tune of the same algorithm)" is **WRONG**. Code
(`RVVContractionSelectedBodyRealizationOwner.cpp:2043/2604/2634/2639`,
`RVVToEmitC.cpp:2795-2808`, `RVVToEmitCInternal.h:99-105`):
- accumulatorLMUL==m8 → realizes the **deferred-wide** chain (vwmul i32m8 → loop-carried
  `deferred_accumulate` vadd.vv → ONE trailing reduce).
- any constrained budget → falls through to the **per-iteration-vredsum** realization
  (`emitWideningDotReduce`: a vredsum EVERY iteration).
These are **two different reduction ALGORITHMS** recognized by different structural ops, not
one algorithm at two LMUL widths. So the clean compiler ON/OFF gap (wide_vs_periter, 6–11×)
is **dominated by the deferred-accumulate-vs-per-iter-reduce algorithm flip**, with the
max-LMUL width on top — NOT an LMUL knob. **Both ON and OFF are compiler-emitted** (so
"compiler-sourced, not hand-written" is true for both — unlike Win B's hand-C). The honest
mechanism statement: *the resource-aware selector automatically chooses the
deferred-wide-max-LMUL realization over the per-iteration-reduce realization.*

## CORRECTION (verdict 2, MAJOR — but my number was already right)
- `wide_vs_periter` (6–11×) is the clean compiler-ON-vs-OFF ablation, BUT the OFF default
  (per-iter-reduce) is **sub-scalar (0.65× scalar)** — an embarrassingly weak prior default.
  Honest label: "vs our prior default emission," NOT the headline.
- **Honest headline = `wide_vs_narrow` 2.06–3.78× vs competent naive RVV** + `wide_vs_scalar`
  4.14–7.52× vs genuine scalar. (I already headlined 2–4× and refused the 6–11× — correct.)
- BUT `narrow-deferred` (the competent naive) is **HAND-WRITTEN** (`lamp_driver.c:16-19`), so
  the 2–4× is a **yardstick** (compiler output beats a competent hand baseline), not a
  compiler-ON/OFF ablation.

## The deepest honest gap (synthesis of v1+v2)
There is **no clean "same good algorithm, vary only LMUL width, all compiler-emitted"
ablation**: the compiler's narrow/budget-constrained path is the *worse per-iter-reduce
algorithm*, and the only same-algorithm-narrow variant (narrow-deferred) is hand-written.
So today:
- clean compiler ON/OFF ablation (6–11×) → weak sub-scalar baseline;
- fair-baseline number (2–4×) → vs a hand-written kernel, not the compiler's OFF.
**Fix to make the LMUL-width ablation bulletproof:** make tcrv-opt *also* emit a competent
**narrow-deferred** realization (same deferred-accumulate algorithm, narrower LMUL), so a
pure-LMUL-width ablation is all-compiler. Concrete N3-strengthening task.

## Regime (verdict 3, MINOR/inverted) — caveat holds
This win is on the **standalone i16/i8/u8 contraction microbenches**, NOT the llama q4_0 hot
path. In llama the wired compiler-automatic tune (measured>static block-dot select) gives
~1.0–1.6× (q5_1 1.62×); max-legal-LMUL is **not** wired into q4_0 block-dots (PRIZE gap P4;
naive block-dot integration ran ~1.8× slower). The llama 5.98×/decode is **Win B** (repack
kernel-swap), explicitly NOT this tune. (v3 noted I *understated* llama impact by omitting
the Win-B e2e — but that's a different mechanism, so the N3 regime caveat stands.)

## i8 caveat
i8 ablation ON÷OFF was never timed same-compiler (narrow path vestigial at budget 32) — only
the i16/u8 ablations are clean. Don't lump i8 into "ablation win" without this note.
