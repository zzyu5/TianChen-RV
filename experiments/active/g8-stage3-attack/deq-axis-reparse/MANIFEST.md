# MANIFEST — g8-stage3-attack/deq-axis-reparse (G8 §六.3 DEQ-AXIS 对手 re-parse + auto-promote)

- `evidence.md` — 主证据：18格×双板对手符号级成色表 + auto-promote 候选 + rvv 15 真向量候选 0.8 判定（9 PASS/6 具名-X）+ T3 回填建议 per-board + 板卫生 + 体例自检。
- `MANIFEST.md` — 本文件。
- `raw/deq_metrics_rvv.txt` — rvv 板 stock base.so（md5 1b4580c4）18 dequant 符号 whole-symbol objdump 机判（tot/rvv/vset/gather/mac·md5 before==after·loadavg）。
- `raw/deq_metrics_k1.txt` — k1 板 stock base.so（md5 00267134·decree cpu.so 871169a0 确认）18 dequant 符号机判（全 rvv=0 SCALAR·md5 before==after）。

回填目标（非本目录·主会执行）：
- rvv → `../../result-tables/T3_A_board_A_rvv1.0_vlen128.csv`（18 dequant 行 col36·**15 行 DEQ-AXIS→in-denom** domain=gcc-15.2-deploy + verdict·3 行 iq2_s/iq2_xs/nvfp4 维持 DEQ-AXIS）。
- k1 → `../../result-tables/T3_B_board_B_rvv1.0_vlen256.csv`（18 dequant 行 col36·**0 变更**·全维持 DEQ-AXIS·对手全 scalar 零候选）。

A/B 出处（复用·同 opponent 二进制）：`../../g7-census/iqfp4-dequant-rvv/summary_dequant_rvv.csv`（gcc-15.2 对称 deploy 域·cold streaming·byte-exact 0/0·2026-07-14）。
