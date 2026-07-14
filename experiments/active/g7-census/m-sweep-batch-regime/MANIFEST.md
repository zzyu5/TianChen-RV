# MANIFEST — G7 §3 M-sweep batch-regime

**Line**: G7 终编成令三·第三段 开局役 — M-sweep batch-regime 特征化 + kernel-axis micro.
**Axis**: kernel-axis micro (2nd track), INDEPENDENT of perf-covered 9/83. Not e2e; potential-判 only.
**Verdict**: compute-bound crossover REAL but board/format-specific (weight-cache amortization across M
rows). **q5_K@rvv** = single clean memory→compute transduction (cold M1 0.896 LOSS → M≥4 1.12–1.29 WIN,
converging to hot compute-bound asymptote 1.35). k1 flat (compute-bound at M=1). q4_K@k1 hand-brick flat
0.622 (compute-quality gap). See `evidence.md`.

## Files
| file | role |
|---|---|
| `evidence.md` | full analysis, M-sweep tables, roofline class, e2e-transduce potential list, hygiene |
| `summary_msweep.csv` | 145 rows: lane×board×fmt×M → ratio, GMAC/s, IQR, ns |
| `parse_msweep.py` | log→CSV+roofline-signal parser (pure parse, no board contact) |
| `board/run_vecdot_msweep_rvv.sh` | rvv block-dot M{1,2,4,8,16,32}, reuses SEALED vecdot-rvv driver+kernels |
| `board/run_vecdot_msweep_k1.sh` | k1 block-dot M{1,2,4,8,16,32} (ships remote body as bash file — k1 login shell = zsh) |
| `board/run_kquant_gemm_msweep_rvv.sh` | rvv repack-GEMM sweep (NOT run — kernels are compiler-emitted, not persisted; used EXISTING census instead) |
| `board/run_q4k_handbrick_msweep_k1.sh` | q4_K@k1 repack-GEMM vl=8 vs hand-brick 16x1, pure re-run of board binary |
| `board/raw/rvv_vecdot_msweep.log` | FRESH rvv block-dot (VERIFY_FAIL=0, 60 CENSUS) |
| `board/raw/k1_vecdot_msweep.log` | FRESH k1 block-dot (VERIFY_FAIL=0, 60 CENSUS) |
| `board/raw/rvv_kquant_gemm_msweep.log` | EXISTING census (batch1-kquant-rvv, nr 4/8/16/64) copied in as repack-GEMM lane |
| `board/raw/k1_q4k_handbrick_msweep.log` | FRESH q4_K@k1 handbrick nr{4,8,16,32} |

## Provenance / integrity
- Drivers/kernels = already-SEALED weft-emitted (vecdot_census_driver.c + per-fmt block-dot kernels,
  kquant census, q4k handbrick s6_q4K.c). NO re-emit, NO emitter/ODS/lib edit, NO git.
- rvv repack-GEMM kernels are regenerated-on-demand by the compiler (not persisted) → to honor the
  "no new emit" rule they were NOT regenerated; the byte-exact existing census (wider M range 4→64)
  was used instead.
- q4_K@k1 = vl=8 core (kernel-sym LOSS); sealed vl=16 winner source unavailable on-box → follow-up.
- Byte-exact: all fresh runs 0 mism / 0 ULP (ZERO-MODEL vs stock ggml). No counts changed.
- Boards restored: 0 stray procs (ps-exact), ggml libs read-only untouched, my scratch removed.
