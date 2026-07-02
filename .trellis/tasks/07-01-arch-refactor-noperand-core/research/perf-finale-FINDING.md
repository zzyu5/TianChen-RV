# Perf finale (metric ③) — gate4 on the new `ssh rvv` board, 2026-07-02

READ-the-truth measurement. Real riscv64 hardware (`ssh rvv`), gate4
(`scripts/rvv_generated_bundle_same_target_measure.py`), `clock_gettime(MONOTONIC_RAW)`,
warmup=2 repeat=5 iterations=8 best-of, correctness-guard-before-timing ✓.

## Board (changed since the archived campaign)
- New host: openEuler `6.12.66` riscv64, `/usr/bin/clang 17.0.6`, 64 cores, 99.5% idle at run.
  (Old board was ubuntu clang-18.1.3 @ 192.168.8.72; project memory "rvv down since 2026-06-28"
  is STALE — the board is back, rebuilt.)
- **VLEN = 128** (VLENB=16, rv64gcv / Zvl128b).

## The clean 3-way (both N3 axes disambiguated)
Prior infra caveat (`.../06-14-.../research/perf-measurement-infra.md`): the default baseline
`-O2 -march=rv64gcv` may be **clang-autovectorized**, so "beat scalar" was never cleanly measured
and there was no naive-RVV axis. Closed here by running the baseline TWO ways — the generated bundle
is explicit-intrinsic RVV (28 `__riscv_v*`), so `-fno-vectorize` pins ONLY the scalar baseline,
leaving generated code byte-invariant (verified: generated ns identical across both runs):
- **default `-O2 -march=rv64gcv`** ⇒ baseline = clang **autovectorized** loop = a legitimate **naive-RVV** baseline.
- **`+ -fno-vectorize -fno-slp-vectorize`** ⇒ baseline = **true scalar**.

### `widening_product_reduce_dequantize_f32` — selector picked WIDE LMUL (`i8m2`→`i16m4`→`i32m8`→`vredsum`)
| n | true-scalar ns | autovec(naive-RVV) ns | generated ns | vs scalar | vs naive-RVV |
|---|---|---|---|---|---|
| 257   | ~ (win)   | 125    | 70    | 4.46×  | 1.75× |
| 4096  | 4747.5    | 1452.5 | 440.0 | **10.79×** | **3.28×** |
| 65536 | 75902.5   | 23667.5| 9847.5| 7.17×  | 2.40× |

**⭐ Beats scalar 10.8× AND clang-autovectorized-RVV 3.3×** — both N3 axes measured on real hardware.
The N3 spec bar ("实测赢 scalar 且赢 naive RVV") is **demonstrated ACHIEVABLE on one kernel** (not
"cleared" as a project claim — its sibling clamp fails the autovec axis, below). n-scaling is physical:
peak vs-autovec 3.3× at n=4096 (compute-bound, cache-resident), compressing to 2.4× at n=65536
(memory-bandwidth binds).

**clang-autovec is a FAIR naive-RVV proxy (objdump-verified, not inferred from timing).** Disassembly
of the `-O2 -march=rv64gcv` baseline hot loop (`baseline_product_reduction_dequant_v1`): clang DID
vectorize competently — `vle8.v → vsext.vf4 → vmacc.vv → vredsum.vs`, 13 vector insns; the pinned
`-fno-vectorize` build has **0** vector insns (confirms the control). Our 3.3× edge over it is a real
ALGORITHMIC advantage, not a strawman: we emit `vwmul_vv` (i8×i8→i16 widening-multiply, ×2 widen) +
wide `m2/m4/m8` LMUL, while clang eagerly sign-extends int8→i32 (`vsext.vf4`, ×4) then `vmacc` at
narrow LMUL — halving effective throughput. So "beats naive-RVV" = beats a competent clang-autovec
lowering; it is a proxy for, not literally, the spec's hand-written naive-RVV.

### `widening_product_reduce_dequant_clamp_f32` — selector picked NARROW LMUL (`i8mf4`→`i16mf2`→`vwredsum i32m1`)
| n | true-scalar ns | autovec(naive-RVV) ns | generated ns | vs scalar | vs naive-RVV |
|---|---|---|---|---|---|
| 257   | 315.0   | 125.0   | 215.0  | 1.47× | 0.59× |
| 4096  | 4745.0  | 1447.5  | 2910.0 | 1.63× | **0.50×** |
| 65536 | 75930.0 | 26207.5 | 46462.5| 1.63× | 0.56× |

**Beats true scalar (1.6×) but LOSES to naive-RVV/autovec (0.5–0.6×).**

## ⭐ The blind-selector bug, MEASURED (not guessed)
Two near-identical kernels (same widening-product-reduce shape; clamp only adds an f32
`vmflt`+`vmerge` min/max tail) got **opposite LMUL** from the capability-blind static selector:
dequantize→WIDE (m2/m4/m8), clamp→NARROW (mf4/mf2). Result: dequantize 440ns, clamp 2910ns at
n=4096 — a **6.6× gap for merely adding a clamp**, far more than the clamp arithmetic costs. The
narrow integer-core LMUL is the culprit, and the board proves it: narrow-clamp loses to clang's own
autovectorizer.

This is exactly the "cold-start capability-blind static argmin picks a bad variant" finding — now
**evidence-backed on hardware**, not a fabricated heuristic. Root cause is a **reduction-STRATEGY**
divergence, not just an LMUL knob: the two pre-realized bodies differ structurally — dequantize uses
`widening_product → widening_accumulate(i32m8) → vredsum` (wide accumulator), clamp uses
`widening_product → vwredsum(i16mf2→i32m1)` **directly** (narrow widening-reduce, no wide accumulate).
The clamp's compare+select tail is POST-reduce and cheap, so the wide-accumulate integer core
(proven fast by dequantize) should apply to clamp too → hypothesis: ~500–700ns, beats autovec.

**⚠ CRITICAL framing (over-optimism guard, cf. [[winc-structural-null]]):** feeding a hand-authored
wide-accumulate clamp body via gate4 `--candidate-input` and measuring it win proves only that **a
better variant EXISTS and measurably wins** (opportunity-evidence for measured-argmin tuning). It does
**NOT** prove **the compiler's selector picks it** — those are different claims. The actual selector
fix lives one layer down: **why does `--tcrv-rvv-materialize-gearbox-schedules` pick direct-`vwredsum`
for the clamp op but wide-accumulate for dequantize?** Almost certainly the compare+select tail
defeats a pattern-match in the schedule pass's strategy selection. That is a separate, deeper thrust —
SCOPED here, NOT opened this run (handed to the user as the next steer, same discipline as Track-B).

## Honest scope / caveats
- These are the **N3 product-reduction/dequant lamp kernels** (gate4's two op-kinds), NOT the 24-op
  block-dot zoo. Block-dot perf is a separate, larger measurement (no gate4 harness for it yet).
- KERNEL microbench only. Per project memory (`kernel-wins-dont-transplant-to-e2e`), a compute-bound
  kernel win does NOT presume an e2e decode win — e2e is memory-bound and reported separately, never
  transplanted by assertion. No e2e claim here.
- Idle-cert: 99.5% id at run; best-of-N; correctness-guard-before-timing ✓ (24/48 CORRECTNESS lines).
- Artifacts (git-ignored, reproducible): `artifacts/tmp/gate4-same-target-measurement/perf-finale-20260702`
  (autovec) + `...-pinnedscalar-20260702` (true scalar).
