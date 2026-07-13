# MANIFEST — g7-l1-gevm / P0-byte-audit

- campaign: G7 主战役 (L1 GEVM plan 构造) · **P0 字节审计** = plan 立项设计输入.
- role: 裁 [GAP-REPACK-GEVM] 布局假设 (H1 GEVM 结构不成熟 vs H2 布局字节税) — 决定 GEVM plan 攻结构还是攻布局.
- board: `ssh rvv` openEuler VLEN128 64c · cores 8-15 disjoint-pin · DVFS perf-gov · co-tenant vLLM/kernspan 未重启.
- account: **kernel-axis DRAM 流量审计** (decode M=1 GEVM regime) · ours(部署 emitted repack-GEVM) vs stock(as-shipped) vs 理论下限.
- 口径: **DRAM bytes = cache-misses × 64** (校准 +0.05% · raw/calib/) · 两点差分 per-token (N=16,64·--no-warmup·抵消装载).
- NO git · board reversible (只 swap sealed ON .so · 测后 byte-exact restore live=05a62e6a · md5 双证).

## canon 依据 (读死)
- [K-10] 结构级/参数级判据 · 实证① GEMM 兼职 GEVM = 结构级 (`.trellis/spec/architecture/core-invariants.md`).
- [PAT-2] P9 GEVM regime-plan (`schema/pattern-registry.v1.json` · ratified-pending-construction).
- [GAP-REPACK-GEVM] 归因修正 (T8 row 2026-07-13·主因=GEVM plan 缺席·布局降为待审计假设·反证 stock repack-GEVM@k1 decode 1.284×).
- 起点 casefile: `experiments/active/l2-kquant-rvv-systemacct/q4_K` (系统账 decode 0.499×) · T9 回退格 decode 数 · g5-wiring M2-q5_0/q5_1/q4_1 (flat scaffold + sealed ON .so).

## provenance (sealed ON .so · 板上·swap-only 无 rebuild)
- q4_K: /tmp/dkgv/libggml-cpu.so.q4kON.clangrepack (md5 915e6733·emitted GEMM+GEVM VLEN128 vl=8·g5-wiring M2 6cbd9c19/e909a9bd) · stock=/tmp/dkgv/...q4kOFF.gcc (05a62e6a).
- q5_0: /tmp/g5_q5/libggml-cpu.so.q5ON (8d136dd9) · q5_1: /tmp/g5_q51/libggml-cpu.so.q5ON (38377d3a) · q4_1: /tmp/g5_q41/libggml-cpu.so.q4ON (a3774cdf).
- engage 逐格实证: "TCRV G5-M2 EMITTED GEVM(qXX_16x1 VLEN128 compiler-emitted vl=8) ENGAGED".
- models: /home/ubuntu/models/DeepSeek-R1-Distill-Llama-8B-{Q4_K_M,Q5_0,Q5_1,Q4_1}.gguf (8.03B params).

## durable files
- evidence.md — 逐格三值表 + 结构轴伴随证据 + 逐格 H1/H2 + 聚合结论 + GEVM plan 设计输入 + 污染/restore.
- raw/calib/counter_calibration.txt — cache-misses×64 = DRAM bytes 校准 (4 GiB stream +0.05%).
- raw/q4_K/twopoint_bytes.txt — anchor 全量 (ours+stock+theory·两点原始计数·H1 判读).
- raw/stock_roofline_flat.txt — q5_0/q5_1/q4_1 stock roofline 两点 + format-spec theory.
- raw/ours_flat.txt — q5_0/q5_1/q4_1 ours (部署 sealed ON·两点·engage banner·ours/stock 比).

## 结论 (一句)
**5/5 格 → H1** (q4_K/q5_0/q5_1/q4_1 @rvv 字节 0.995–0.998× ≈ 理论 roofline + **q5_K@k1 补格** 解析 footprint 恒等我方/stock=1.000·H2 布局税逐格 REJECTED)；decode win↔loss 判别键**全在结构轴** (insn/IPC/cycles)、从不在字节轴 (q5_K@k1 直证 1.99× 指令/token→0.729×)。**GEVM plan 攻结构不攻布局·跨板成立 (rvv/VLEN128 + k1/VLEN256 同机制)**；[PAT-2] P9 目标"字节最少化"精化为"M=1 带宽→吞吐结构效率"；P1 首格 = q4_K (审计浪费最大·0.50×·IPC 0.29)。q5_K@k1 补格 = `q5K-k1-补格/` (X60 PMU 无字节计数器→退化解析+结构轴·H1-implied 升 H1-measured)。

## reversibility
- 板改动仅 swap LIVE libggml-cpu.so.0.15.1 (测中 ON·测后 restore stock)。source 全程 pristine·无 rebuild·NO git。
- restore 证: live md5=05a62e6a (== systemacct cert-1 OFF-pristine byte-exact)·残留 tcrv=2 个 q4_0 baseline(shipped·非 stray)·0 q4_K/q5/q4_1 emitted 残留·0 遗留进程。
