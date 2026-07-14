fmt,nr,K,nc,rhot,rcm,rcb,ohg,phg,oiqr,piqr,owgb,pwgb,nf
q2_K,4,2048,512,0.3775,0.3669,0.3779,1.7789,4.7126,8.14,0.24,0.132,0.359,60
q2_K,8,2048,512,0.3850,0.3539,0.3752,1.8020,4.6804,1.56,0.40,0.065,0.185,300
q2_K,16,2048,512,0.3896,0.3691,0.3818,1.8035,4.6295,0.92,0.94,0.034,0.093,280
q2_K,64,2048,512,0.3957,0.3888,0.3927,1.7981,4.5436,1.18,0.39,0.009,0.023,140
q3_K,4,2048,512,0.0503,0.0518,0.0520,0.1743,3.4641,0.23,0.53,0.018,0.342,300
q3_K,8,2048,512,0.0495,0.0518,0.0516,0.1746,3.5282,0.13,0.44,0.009,0.175,300
q3_K,16,2048,512,0.0488,0.0505,0.0503,0.1744,3.5766,0.28,0.59,0.005,0.091,240
q3_K,64,2048,512,0.0494,0.0495,0.0496,0.1742,3.5278,0.40,0.34,0.001,0.023,280
q4_K,4,2048,512,0.3551,0.2351,0.2448,1.7122,4.8217,1.53,0.86,0.150,0.639,300
q4_K,8,2048,512,0.3604,0.2924,0.2961,1.7286,4.7959,1.29,0.70,0.093,0.318,280
q4_K,16,2048,512,0.3843,0.3900,0.3917,1.7298,4.5016,0.62,1.23,0.052,0.134,280
q4_K,64,2048,512,0.4004,0.3904,0.3905,1.7339,4.3307,0.16,0.41,0.014,0.037,260
q5_K,4,2048,512,0.7138,0.6802,0.6922,1.0445,1.4634,2.06,1.54,0.131,0.192,60
q5_K,8,2048,512,0.7125,0.7588,0.7628,1.0405,1.4604,0.58,0.81,0.082,0.108,260
q5_K,16,2048,512,0.7633,0.7816,0.7819,1.0419,1.3649,0.17,0.30,0.043,0.054,280
q5_K,64,2048,512,0.7664,0.7672,0.7681,1.0359,1.3516,0.82,0.10,0.011,0.014,260
q6_K,4,2048,512,0.0378,0.0400,0.0395,0.1423,3.7686,0.71,1.27,0.027,0.677,280
q6_K,8,2048,512,0.0365,0.0393,0.0391,0.1416,3.8784,0.59,2.10,0.014,0.356,280
q6_K,16,2048,512,0.0371,0.0393,0.0391,0.1416,3.8148,1.05,1.30,0.007,0.182,60
q6_K,64,2048,512,0.0377,0.0383,0.0384,0.1408,3.7400,0.37,0.36,0.002,0.047,240

## census (all nr) — ours/opp ratio, gcc-15.2 symmetric, VLEN128, core8

| fmt@rvv | nr(M) | hot | cold_med | cold_best | ours_hot GMAC/s | opp_hot GMAC/s | cold iqr% o/opp | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| q2_K | 4 | 0.378× | 0.367× | 0.378× | 1.7789 | 4.7126 | 8.14/0.24 | LOSS |
| q2_K | 8 | 0.385× | 0.354× | 0.375× | 1.8020 | 4.6804 | 1.56/0.40 | LOSS |
| q2_K | 16 | 0.390× | 0.369× | 0.382× | 1.8035 | 4.6295 | 0.92/0.94 | LOSS |
| q2_K | 64 | 0.396× | 0.389× | 0.393× | 1.7981 | 4.5436 | 1.18/0.39 | LOSS |
| q3_K | 4 | 0.050× | 0.052× | 0.052× | 0.1743 | 3.4641 | 0.23/0.53 | LOSS |
| q3_K | 8 | 0.050× | 0.052× | 0.052× | 0.1746 | 3.5282 | 0.13/0.44 | LOSS |
| q3_K | 16 | 0.049× | 0.051× | 0.050× | 0.1744 | 3.5766 | 0.28/0.59 | LOSS |
| q3_K | 64 | 0.049× | 0.050× | 0.050× | 0.1742 | 3.5278 | 0.40/0.34 | LOSS |
| q4_K | 4 | 0.355× | 0.235× | 0.245× | 1.7122 | 4.8217 | 1.53/0.86 | LOSS |
| q4_K | 8 | 0.360× | 0.292× | 0.296× | 1.7286 | 4.7959 | 1.29/0.70 | LOSS |
| q4_K | 16 | 0.384× | 0.390× | 0.392× | 1.7298 | 4.5016 | 0.62/1.23 | LOSS |
| q4_K | 64 | 0.400× | 0.390× | 0.391× | 1.7339 | 4.3307 | 0.16/0.41 | LOSS |
| q5_K | 4 | 0.714× | 0.680× | 0.692× | 1.0445 | 1.4634 | 2.06/1.54 | LOSS |
| q5_K | 8 | 0.713× | 0.759× | 0.763× | 1.0405 | 1.4604 | 0.58/0.81 | LOSS |
| q5_K | 16 | 0.763× | 0.782× | 0.782× | 1.0419 | 1.3649 | 0.17/0.30 | LOSS |
| q5_K | 64 | 0.766× | 0.767× | 0.768× | 1.0359 | 1.3516 | 0.82/0.10 | LOSS |
| q6_K | 4 | 0.038× | 0.040× | 0.040× | 0.1423 | 3.7686 | 0.71/1.27 | LOSS |
| q6_K | 8 | 0.036× | 0.039× | 0.039× | 0.1416 | 3.8784 | 0.59/2.10 | LOSS |
| q6_K | 16 | 0.037× | 0.039× | 0.039× | 0.1416 | 3.8148 | 1.05/1.30 | LOSS |
| q6_K | 64 | 0.038× | 0.038× | 0.038× | 0.1408 | 3.7400 | 0.37/0.36 | LOSS |

## opponent machine-probe (VLEN128 dispatched = block-dot; repack-gemm VLEN256-gated-off)

- q2_K: block-dot main@0000000000092316 vl128=0000000000091cb6 | unused_repack_gemm=T,T,T,
- q3_K: block-dot main@00000000000930a6 vl128=000000000009232c | unused_repack_gemm=none
- q4_K: block-dot main@000000000009312c vl128=none | unused_repack_gemm=T,T,T,
- q5_K: block-dot main@0000000000093146 vl128=none | unused_repack_gemm=T,
- q6_K: block-dot main@00000000000935de vl128=none | unused_repack_gemm=T,

## L1' q6_K@rvv GEMM whole-K-nest roll — recovery annotation (2026-07-14, does NOT change any status)

The q6_K@rvv GEMM deepest LOSS (0.037× vs stock block-dot) root-caused to gcc-15.2
vsetvli-storm + spill on the giant full-unroll body (NOT MAC quality; vwmacc~2304 equal).
The whole-K-nest roll (capability-keyed `emit_loop_schedule="rolled"`, deployed emitter,
byte-exact) was implemented + G2 board-validated — casefile
`../L1prime-q6k-gemm-roll-deploy/evidence.md`:
  - on-silicon byte-exact rolled ≡ unrolled (memcmp=0, all shapes)
  - vsetvli 6278 → 21 (-299×), spill -7.7×, obj 299632B → 22768B
  - cold GEMM prefill recovery ~6.7× (roll vs unroll), landing 0.037× → ~0.26× vs stock
The q6_K rows STAY LOSS (0.26× < parity): this is a kernel-axis emitter-maturity RECOVERY
of the self-inflicted [CASE-COMPILER-ASYMMETRY] codegen collapse, NOT a WIN / NOT
perf-covered green / NOT an e2e beat. Density floor (vl=8 dual-strip vwmacc + 6-bit
dual-plane decode vs stock wide block-dot) unchanged; parity needs a density attack [远期].
Default (no stamp) ships frozen unrolled = byte-identical (zero drift). q3_K (shares the
no-min GEMM dispatch) is NOT wired to roll (capability-guard: q6_K-only) — same-class deferred.
