# G5-M2 q5_0 曳光弹 — L-接线② 净新 scaffold · CASEFILE (phase-1 recon+emit+design)

> **campaign**: G5 接线战役 · **M2 L-接线② q5_0**（首个真净新上游 scaffold·验证 template 建 upstream scaffold 能力=C1 extensibility）
> **board**: `ssh rvv` openEuler VLEN128 gcc-15.2.0 · A-tree f3e1828（**零改动**·read-only recon + host emit only）
> **workflow**: `wtpdeaoes`（phase-1）· **HEAD (TianChen-RV)** = a7cacf68
> **结论 (phase-1)**: **upstream_absent 确证**（q5_0 GEN+ARCH grep=0·block<K,N> 模板无法表达第 5 位 qh·与 q4_K 全-present 决定性相反）· **kernel EMITTED**（vl=8 全 AVL·zero 16/64·VLEN128-safe）· **12-piece 净新 scaffold DEFERRED**（downgrade rule·远超 q4_K deploy·correctness-critical = make_block_q5_0x16 interleaver·MIRAGE trap 若 qh bit-order 微错）。
> **perf 预期（诚实·非预设绿）**: FLAT kernel-axis 1.23× gcc-symmetric 小 margin·**可能 washes to yellow**（micro↛e2e·q4_K 先例）·perf 只在 correctness GREEN 后。

## durable files
- `evidence.md`（recon + emit recipe + 12-piece 净新 scaffold recipe·全锚点·供 q5_1 复用）
- `tcrv_emitted_gemm_q5_0.inc`（host-emitted GEMM·md5 f03c6566·VLEN128-safe·27KB）
- `tcrv_emitted_gevm_q5_0.inc`（host-emitted GEVM·md5 f3892049·VLEN128-safe·20KB）

> .inc 小（47KB total·直接 tracked·不同于 q4_K 2.5MB gitignored）。scaffold 建设 harness 将复用 `tools/e2e-harness/board/g5-m2-q4k/`（3 挂点模板不变·adapt names）。
