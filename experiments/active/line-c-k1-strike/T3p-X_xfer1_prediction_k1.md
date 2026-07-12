# T3p-X — [XFER-1] 三杠杆 k1/VLEN256 预测-实测对照表 (PRE-BOARD registration)

> **Line C · 1b deliverable**. Extends `t3p-pattern-ablation/` (P4 = S6 XFER-1 3-class, 7/7 rvv-measured) with a
> **k1-specific PRE-BOARD prediction** for three repack-GEMM (prefill matmul) levers:
> **S6** (output-tiling register-cliff min-fold, spill→0) · **col-outer** (loop-interchange, cache-keyed) ·
> **vl16** (VLEN256-native vl=16 lane-fill vs deployed half-wide vl=8, VLEN-keyed).
> **In-tree analysis · zero board · zero lib edit · [NG-4] kernel-axis, not e2e beats.** Predictions registered
> BEFORE any k1 board run; `_measured_` column left open for main-session board follow-up.
> Evidence tier: `[k1-M]`=board-measured-on-k1 · `[rvv-M]`=rvv-measured (S6 3-class transferred) · `(pred)`=untested-on-k1.

## count summary (19 candidate formats · repack-GEMM axis)
| lever | HOLDS | NULL | NO-OP | LOSS |
|---|---|---|---|---|
| **S6**       | 3 (q4_K,q5_K,q2_K) | 2 (q6_K,q3_K) | 14 (iq4_nl/xs + 7 IQ-gather + 5 FLAT) | 0 |
| **col-outer**| 0 | 10 (5 K-quant + 5 FLAT) | 9 (2 codebook + 7 IQ-gather) | 0 |
| **vl16**     | 6 (q4_K,q5_K,q2_K,q5_0,q5_1,q8_0) | 2 (q6_K,q3_K) | 4 (iq4_nl/xs,q4_0,q4_1) | 7 (IQ-gather family) |

## evidence-tier breakdown (honesty ledger)
- **(a) already-measured-on-k1** (3): `q4_K/S6` (t4a objdump spill 81→4 + 3.106×) · `q5_K/S6` (t4a 1.916×) ·
  `q4_K/vl16` (Win-K1-VLEN / FU-1 1.085× vs hand-brick, byte-exact). Plus **1 measured-NULL-on-k1**: `q4_K/col-outer`
  (FU-1 k1 stack-margin 1.002×, CI crosses 1.0 ⇒ col-outer is genuinely rvv-cache-specific, no k1 margin).
- **(b) transferred-from-rvv-measured** (S6 3-class, 7/7 pred==meas on rvv): `q2_K/S6` HOLDS · `q6_K/S6` NULL ·
  `q3_K/S6` NULL · `iq4_nl/S6` NO-OP · `iq4_xs/S6` NO-OP.
- **(c) pure-prediction untested-on-k1**: all `(pred)` cells (col-outer for non-q4_K, vl16 for non-q4_K,
  entire IQ-gather + FLAT rows). Honestly registered as predictions, not results.

## load-bearing caveats (for the board run to falsify)
1. **col-outer predicted DEAD on k1** — the rvv 2.47× / LLC-miss↓5.2× win fixed a k1-ABSENT H-B memory regression;
   FU-1 already measured k1 stack-margin 1.002× (null). numHalves 2→1 @VLEN256 changes strip/re-stream structure.
   Sharpest transfer-boundary claim.
2. **vl16 HOLDS carries a "deployed-half-wide" precondition** — vl16 only recovers width where the shipped kernel is
   currently vl=8. Confirmed vl=8 for q4_K (sealed source emitted @VLEN128 config, half=8) + q8_0 (G5-M1b vl=8 deploy).
   If a format is re-emitted with `zvl256b` capability fact, vl16 is already native ⇒ lever collapses to NO-OP.
3. **vl16 reg-account (peak_hot=6d+7=31,v30) is q4_K-specific** — transfers to q2_K/q5_K (same dmin/bsums-min
   accumulator family, S6-staged) but NOT to q6_K/q3_K (weight-recon-spill-saturated ~900 spills; lane-fill orthogonal → NULL).
4. **IQ-gather vl16=LOSS is directional** — wider vl on a vluxei indexed-gather-bound kernel adds elements at ~1/cyc
   gather while raising register pressure ⇒ net-negative, consistent with Line A rvv IQ LOSS. S6/col-outer NO-OP there
   (spill is gather-memory, not the accumulator fan those levers address).

## durable files (this cell)
- `T3p-X_xfer1_prediction_k1.md` — this analysis (count summary + evidence tiers + falsification caveats).
- `T3p-X_xfer1_prediction_k1.csv` — the 19-format × 3-lever prediction grid (main-session may promote to result-tables + add `_measured_` col).
