# Win-A "2–4× vs naive RVV" — explained clearly (the N3 headline)

This is the answer to "microbench 上相对 naive RVV 2–4× 没讲清楚和明白，是不是需要重新做实验?"
**Yes, the experiment was redone** (the OFF arm is now compiler-emitted, not hand-written), and here is
exactly what the number means, what's compared, and why it's honest.

## What Win-A IS (the compiler-automatic optimization, NOT a kernel)
The Gearbox tune automatically selects the **widest legal accumulator LMUL** for a contraction
(i16 dot-reduce): load → `vwmul` to a wide i32m8 product → loop-carried wide accumulate → ONE trailing
reduce. The *only* thing that changes between "tune ON" and "tune OFF" is the gearbox's LMUL choice —
the body is verbatim `tcrv-opt` output in both cases. This is distinct from Win-B (the repack
kernel-swap, 5–6× e2e) and from the VLEN-strip selection (1.48× in llama).

## The number: 3 framings (all measured on real ssh hardware), pick the honest one
| comparison | what it is | rvv (SG2044, VLEN128) |
|---|---|---|
| **wide ÷ narrow-deferred** | the tune's LMUL-width win vs a **competent naive RVV** (same deferred-accumulate algorithm, narrow LMUL) | **2.27–3.79×** ← THE honest headline |
| wide ÷ genuine-scalar | vs 0-vector scalar (objdump-verified) | 4.0–7.5× |
| wide ÷ per-iter-reduce | vs the compiler's *prior weak default* | 5.9–11.2× — do NOT headline (the OFF default is *slower than scalar*, 0.65×, an inflated baseline) |

**The "2–4×" = wide ÷ narrow-deferred = 2.27–3.79×.** It is the LMUL-width contribution: the same
deferred-accumulate algorithm at the gearbox's wide LMUL (m8) vs a competent narrow LMUL (m1).

## Why the redone experiment matters (the honesty fix)
**Before**: the narrow-deferred (OFF) arm was HAND-WRITTEN C, so "2–4×" was a yardstick (compiler-output
vs hand-kernel), not a clean pass ON/OFF ablation. **Now** (commit 3d2a2b3f): tcrv-opt emits the
narrow-deferred body too (gearbox at a constrained vreg-budget auto-selects mf2→m1). So BOTH arms are
verbatim compiler output, SAME algorithm, only LMUL differs → a **clean all-compiler ablation**. Measured
fresh on rvv (commit 709bb69d): wide÷narrow **2.27–3.79×** across n∈{256..65536}, both numerically exact
vs a scalar oracle. The budget VALUE is an injected modeled vreg-profile fact; the *rung selection given
that budget* is automatic — that selection IS the tune.

## Cross-profile (3 real chips, 2 ISA generations) — the tune wins everywhere
| chip | wide_vs_scalar | wide_vs_narrow | note |
|---|---|---|---|
| rvv SG2044 (RVV1.0, VLEN128) | 4.0–7.5× | **2.1–3.8×** | the clean all-compiler ablation |
| K1 X60 (RVV1.0, VLEN256) | 8.4–15.3× | 1.8–3.6× | weaker scalar core amplifies vs-scalar |
| Fedora C920 (RVV0.7, VLEN128) | 3.4–7.2× | **1.4–1.7×** | see baseline note ↓ |

**The Fedora `wide_vs_narrow` looks smaller (1.4–1.7× vs rvv's ~3×) — but the tune did NOT weaken.**
RVV0.7 has no fractional LMUL, so the naive baseline there was floored mf2→m1 (the strongest RVV0.7 can
express), making it ~2× stronger (8 vs 4 lanes/strip). Proven from the data: `narrow_vs_scalar` is
2.40–4.27 on C920 vs 1.39–2.02 on rvv — the baseline doubled; the wide side held. So the compression is
a stronger baseline, not a weaker tune.

## "Why did dynamic testing 反驳 you?" — it didn't reject the tune; it rejected a DIFFERENT claim
The e2e regression (repack 0.74× on K1) is **Win-B** (the repack kernel-swap), not Win-A (this tune).
It's X60-microarch-specific (clang autovec is strong on X60). The Win-A *tune* (this 2–4×) is a
microbench/selection result and stands on all 3 chips. The honest separation: Win-A tune = 2–4× selection
win (solid, 3 chips); Win-B repack = 2.49–6× e2e on SG2044 but X60-regression on K1; VLEN-strip = 1.48×
in llama decode on K1. Three different mechanisms, three different numbers — never conflated.

## TL;DR
**"2–4×" = the compiler-automatic max-LMUL tune vs competent naive RVV, both arms now compiler-emitted
(clean ablation), = 2.27–3.79× on rvv, holding (1.4–3.8×) across rvv/K1/Fedora. The experiment WAS redone
to make the OFF arm compiler-emitted. The e2e wobble was Win-B, a separate mechanism.**
