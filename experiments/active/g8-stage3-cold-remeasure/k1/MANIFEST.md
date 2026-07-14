# MANIFEST — g8-stage3-cold-remeasure/k1 (G8 §六 cold A/B 重测)

- `evidence.md` — 主证据：k1 28 格 0.8 达标表 + 输格清单 + DEQ 18 + flag 处置 + 体例自检。
- `MANIFEST.md` — 本文件。
- `raw/vecdot_build_seal.txt` — FLAT+K-quant vec_dot build seal（ours clang-18 -O2 finalized·so-hash 871169a0·对手 dispatched-sym probe）。
- `raw/vecdot_run.log` — FLAT 5 + K-quant vec_dot 5 cold A/B raw（VERIFY byte-exact 0/0 + CENSUS M=1/M=8）。
- `raw/iqfp4_dequant_build_seal.txt` — iq/fp4 vec_dot 4 + dequant 18 build seal（ours + opp objdump probe）。
- `raw/iqfp4_dequant_run.log` — iq/fp4 vec_dot 4 + DEQ 18 cold raw（byte-exact 0/0）。
- `raw/bclass_forward_build_seal.txt` — forward-op 9 build seal（opp-class objdump + ours LMUL probe）。
- `raw/bclass_forward_run.log` — forward-op 9 hot/cold shape sweep 512-16384（anchor 4096·byte-exact/ULP-gate）。

回填目标（非本目录）：`../../result-tables/T3_B_board_B_rvv1.0_vlen256.csv`（perf 列·46 行·备份 /tmp/T3_B_backup_pre_g8s6.csv）。
follow-up #1：K-quant GEMM@k1 5 格 vs REAL hand-brick（需 ours repack-GEMM 重导出 + 16x1/8x8 交织 harness）。
