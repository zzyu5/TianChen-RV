# MANIFEST — G7 L1 Batch-1 K-quant@rvv cold-start census

**Line**: G7 终编成令三·第一段（全量 kernel 冷启动普查）· Batch-1（K-quant@rvv·rvv-lane·板批主力）.
**Date**: 2026-07-14 · **HEAD**: 677a2f7f (branch refactor/full-refactor-m1) · **Board**: rvv (openEuler riscv64 VLEN128 64c core8 gcc-15 域).
**Verdict**: q{2,3,4,5,6}_K@rvv repack GEMM 全 20 cell **LOSS** (0.037×–0.78×) · byte-exact 全过 · 全家族 [CASE-COMPILER-ASYMMETRY] 兑现. See `evidence.md`.

## Files
| file | role |
|---|---|
| `evidence.md` | ★ full census table + opponent probe + byte-exact + objdump attribution + T9 reconcile + 3-exit-per-format |
| `census_table.md` | parsed CSV + markdown (from parse_census.py) |
| `kquant_gemm_census_driver.c` | unified 5-fmt hot(best-of-32)+cold(224MiB-flush paired median-N) driver (weft_emitc symbols) |
| `run_rvv_kquant_census.sh` | stage+compile(gcc-15.2)+objdump-seal+opp-nm-probe+sweep(nr{4,8,16,64}) |
| `run_rvv_kquant_verify.sh` | byte-exact gate (q2/q3/q4/q6 fresh verifiers vs stock ggml) |
| `parse_census.py` | raw CENSUS lines → CSV + md table |
| `verify/verify_q{2,3,4,6}K.c` | copied from tools/e2e-harness/board + sed tcrv_emitc→weft_emitc (untracked copies) |
| `kernels/repack_gem{m,v}_q{2,3,4,5,6}_K_q8_K.kernel.c` | regenerated GEMM+GEVM kernels (weft-opt) |
| `raw/census_full_sweep.txt` | raw sweep (objdump seals [A] + opp probe [B] + 20 CENSUS rows [D]) |
| `raw/verify_byteexact.txt` | raw byte-exact (q2/q3/q4/q6 INT mismatch=0 + NORM) |

## Kernel provenance (regenerated @HEAD 677a2f7f — read-only build, no rebuild)
```
OPT=build-weft/bin/weft-opt            # renamed build (weft dialect; old build/bin/tcrv-opt STALE)
TR=/usr/bin/mlir-translate-20
for f in q2 q3 q4 q5 q6; do
  $OPT test/Conversion/RVV/rvv-to-emitc-repack-gemm-$f-K-q8-K.mlir --weft-rvv-lower-to-emitc \
    | $TR --mlir-to-cpp > kernels/repack_gemm_${f}_K_q8_K.kernel.c
done   # (gemv fixtures likewise, for the q2/q3/q6 verifiers)
```
GEMM kernel md5 (gcc-15.2 -O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on):
```
d9dee831 q2_K  54beac58 q3_K  f373bdb5 q4_K  41368b6d q5_K  bf477693 q6_K
```
weight_block_stride byte-verified vs fixture attrs: q2=1344 q3=1824 q4=2304 q5=2816 q6=3360 (all act=1168).

## Opponent (as-shipped, machine-judged)
`build-gcc15-rv64gcv/bin/libggml-cpu.so md5=d1adc634c2ca04ffc389536b30d9d9c4` (llama.cpp commit f3e1828, gcc-15).
@VLEN128 dispatched = block-dot `ggml_vec_dot_qX_K_q8_K` (repack GEMM trait = NULLPTR for all 5; repack GEMM symbols exist but VLEN256-gated/NEON-only — disclosed unused stronger path).

## Reproduce
```
# regenerate kernels (above), then:
MODE=full HITERS=4 REPS=12 bash run_rvv_kquant_census.sh   # → raw/census_full_sweep.txt
bash run_rvv_kquant_verify.sh                              # → raw/verify_byteexact.txt
python3 parse_census.py raw/census_full_sweep.txt          # → census_table.md
```

## Restore / hygiene
- Main tree + build/ + build-weft/ UNTOUCHED (weft-opt invoked read-only). No git add/commit.
- Board: scratch `/tmp/g7_census_kquant_{rvv,verify}` removed. No stray procs. stock lib md5 测前==测后 (d1adc634, read-only linked). No vLLM co-tenant restart. governor left as-found (performance).
