# G5-M2 q4_K 曳光弹 — L-接线② correctness-carrier · CASEFILE

> **campaign**: G5 接线战役 · **M2 L-接线② 首格 = q4_K**（验 deploy 模板 + K-quant framing 纠偏）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree f3e1828（restored baseline deb61a29/99131cf7）
> **workflow**: `wyhnspmt6` · **HEAD (TianChen-RV)** = 609d79c4（我方零 tracked 源改动）
> **结论**: correctness **GREEN**（我方 emitted vl=8 = VLEN128 correctness-carrier 第 2 例·[GAP-Q4K-VLEN128] 破损上游确证）· **perf < parity 全相 → yellow-kernel-axis · perf-covered 维持 3/84**（correctness 门绿前不入台账·q4_K repack 慢于 stock scalar vec_dot）。
> **登记性质**: correctness-carrier（C1 合取存在性·VLEN128-correct-repack）· **非 perf-covered 绿格**。

## durable files
- `evidence.md`（recon + emit + deploy + perf 分相 + A-tree restore + L-接线②建法·全数据）
- `.gitignore`（gitignore 大 emitted .inc·regenerable）

> emitted `.inc` = **gitignored**（`tcrv_emitted_gemm_q4_K.inc` md5 6cbd9c19·1.8MB / `tcrv_emitted_gevm_q4_K.inc` md5 e909a9bd·729KB·regenerable via evidence.md §二 emit recipe·避 repo bloat）。board harness 脚本住 `tools/e2e-harness/board/g5-m2-q4k/`。
