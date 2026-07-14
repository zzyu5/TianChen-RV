# [G7-L3 货架B] FLAT 5 @k1 路由确认 + q4_1/q5_0/q5_1 结构缺口

> Board `ssh k1` SpacemiT X60 VLEN256 · stock clang-18 · shared source `repack.cpp` md5 `3cac40aa…` (baseline·NEVER edited) · shipped stock lib `871169a0…`.
> 派生自 `ggml_repack_get_optimal_repack_type`（`/home/bianbu/tcrv-k1-llama/ggml/src/ggml-cpu/repack.cpp:4528`）+ stock lib objdump/nm。

## 路由确认（全 5 格·objdump/nm 证）

| 格 | k1 stock dispatch | 出货路径 | objdump 证 | 对手身份 | e2e 可测法 |
|---|---|---|---|---|---|
| **q4_0** | `case256 → q4_0_16x1_q8_0` | **stock 16x1 repack** | gemm/gemv_q4_0_16x1 present·VLEN256-native e32m2/e16m1/e8mf2 | stock block-dot（关 dispatch） | ✅ **repack-vs-block-dot**（模型在·已测·§q4_0/evidence.md） |
| **q8_0** | `case256 → q8_0_16x1_q8_0`（case128 break//TODO） | **stock 16x1 repack**（VLEN256-only） | gemm/gemv_q8_0_16x1 present·VLEN256-native e32m2/e16m1/e8mf2 | stock block-dot | ✅ **repack-vs-block-dot**（模型在·测量中·§q8_0/evidence.md） |
| **q4_1** | 无 branch → nullptr | **generic block-dot** `ggml_vec_dot_q4_1_q8_1` | nm: 零 q4_1 repack·仅 vec_dot | —（stock 无 repack 对位） | ✗ 需**部署 net-new emitted repack**（见下） |
| **q5_0** | 无 branch → nullptr | **generic block-dot** `ggml_vec_dot_q5_0_q8_0` | nm: 零 q5_0 repack·仅 vec_dot | — | ✗ 需 net-new emitted repack |
| **q5_1** | 无 branch → nullptr | **generic block-dot** `ggml_vec_dot_q5_1_q8_1` | nm: 零 q5_1 repack·仅 vec_dot | — | ✗ 需 net-new emitted repack |

**★关键分裂**：k1 FLAT 家族一分为二 —
- **q4_0 / q8_0**：k1 stock **出货 16x1 repack**（case256 ON·与 rvv 不同——rvv q4_0 case128 break·q8_0 VLEN128 破损）。∴ k1 上「repack approach vs block-dot」有 as-shipped 对位·可直接 e2e（同 [WORK-ITEM-K1-KQUANT-E2E] q4_K 2.644× 方法）。
- **q4_1 / q5_0 / q5_1**：k1 stock = **纯 block-dot**（零 repack·同 rvv 的 net-new 情形）。stock 无 repack 对位 ⇒ **repack-vs-block-dot 对照不存在**（两侧都会是 block-dot）。要产 k1 e2e 数**必须先部署我方 net-new emitted repack**（ON=our-repack vs OFF=block-dot），无捷径。

## q4_1/q5_0/q5_1 dual-board e2e = 部署-net-new 跟进项（本批未完成·如实披露）

要在 k1 补这 3 格 e2e，需（每格·非本批可完成的量级）：
1. **VLEN256 kernel emit**：rvv 侧 .inc 是 vl=8 VLEN128（`experiments/active/g5-wiring/M2-q5_0|q5_1` + q4_1）。k1 native 需 `march=zvl256b` vl=16 重 emit（vl=8 在 VLEN256 可跑但半宽·非部署==证过）。
2. **net-new scaffold 部署 patch**（适配 k1 tree）：block struct + interleaver（make_block_q5_0x16 / q8_1x4）+ q8_1 mat-quant 基建 + trait 注册 + **dispatch case256 intercept**（q4_1/q5_0/q5_1 当前无 branch）+ banner。rvv 的 deploy_patch 是 rvv-tree-specific·需移植。
3. **模型 provisioning**：k1 无 q4_1/q5_0/q5_1 gguf（仅 q4_0/q8_0/Q2_K/Q4_K/Q5_K）。可 `llama-quantize --allow-requantize` 从 tinyllama 造（决策卡④）。

∴ 本批（货架B 首批）交付 = **路由确认（全5）+ q4_0/q8_0 dual-board e2e（stock 出货 repack·可直接对照）**。q4_1/q5_0/q5_1 的 k1 dual-board = **明确的 deploy-net-new 跟进 work-item**（结构缺口·非测量失败）。其 rvv perf-covered 绿不受影响（rvv 侧净新 scaffold 已封·`g5-wiring/M2-q5_0|q5_1|q4_1`）。

## 对手成色注（令六 lint）
- q4_0/q8_0 @k1 对手 = stock **generic block-dot**（弱-中对手·非 hand-brick）。区别 q4_K@k1（对手 = 真出货 hand-brick 16x1 repack·成色最硬）——此处 q4_0/q8_0 的 stock「repack」就是被测的 winner·对手是 block-dot fallback。
- 大倍数（q4_0 5.18× prefill）反映 stock block-dot 弱·非 tuned-kernel 对比。
