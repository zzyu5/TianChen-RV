# Research: ★q4_K vec_dot 弧线是否 transfer 到 decode (证·非假设)

- **Query**: PRD §二.3 — vec_dot 四/五次定格 (aux8→serial→scalar-min-term→memory-stall→m2 register-pressure·lever m1-宽度多流) 是否适用 decode M=1？
- **Scope**: internal (objdump 结构证据·逐层判 transfer)
- **Date**: 2026-07-18

## 结论一句话

**PARTIAL transfer (部分·须分层)**：vec_dot 的 **surface 三墙 (levers 1-3) NOT transfer** (decode leaf 结构上已消除)；vec_dot 的 **deepest 墙 (第 5 次定格 memory-scheduling 轴) 结构 transfer** (decode leaf 同轴)；但 transfer **动态上未证** (decode 无 perf-stat)。**禁把 vec_dot 「best 0.186 memory-stall floor」直接外推为 decode 的墙数字或墙性质。**

## 逐定格 transfer 判 (带 objdump 证据)

| vec_dot 定格 (ISSUE-109) | 墙/lever | decode 是否 transfer | objdump 证据 |
|---|---|---|---|
| 定格 1 | aux8 权重 scratch roundtrip (register-fusion lever) | **NOT transfer** | decode leaf **0 权重 vse8** (register-resident 内建)·vec_dot 16× vse8 到 `int8_t[256]` |
| 定格 2 | i32 单累加器 serial 链 (vwredsum lever) | **NOT transfer** | decode leaf **4-way i32m2 累加器** (v34/v36/v38/v40)·非单链 |
| 定格 3 | scalar min-term 16-deep serial MAC (minterm-vec lever) | **NOT transfer** | decode leaf min/scale **已向量化** (`vwmacc_vx` + fp fold epilogue)·0 scalar `lh` |
| 定格 4 | memory-stall floor (三指令流 lever 全 cold-inert) | **结构 transfer·动态未证** | decode leaf 同 weight-DRAM-streaming (窄 load 主导)·但 **无 perf-stat 证 memory-stall** |
| 定格 5 | 窄→宽 load + 寄存器驻留多流 MLP·墙=m2 register-pressure·lever=**m1-宽度多流** | **结构 transfer (最相关)** | decode leaf `576× vle8_v_u8mf2` = 窄 8B load (对手宽 16B)·clang 低 MLP·= vec_dot 第 5 定格同轴 |

## 为何 surface 不 transfer (关键·反直觉)

decode leaf 是 **不同发射路径** (`typed_repack_gemv_loop_body`·front-door `RVVToEmitCKQuant.cpp`)·**不是** vec_dot 的 block-dot 核 (`typed_super_block_block_dot_loop_body`)。这条 repack-gemv 路 **原生就**：
- 寄存器驻留解包 (无 aux8 roundtrip) = vec_dot 花 lever 1 (register-fusion) 才达到的状态·decode 免费。
- 4-way 累加器 = vec_dot 花 lever 2 (vwredsum) 才达到的 ILP。
- 向量化 scale = vec_dot 花 lever 3 (minterm-vec) 才达到的。

⟹ **decode leaf 结构上已在 vec_dot「三 lever 后」的位置**·vec_dot 前三定格的墙对 decode 不存在。把 vec_dot 的 5 次定格叙事整体套到 decode = **错误外推** (不同核·前三墙 N/A)。

## 为何 deepest 结构 transfer

1. **同一对手** (rvv q2/q3/q4/q6_K)：decode 对手 = `ggml_vec_dot_qX_K_q8_K_vl128` = 与 vec_dot 弧线**完全同一符号** (tot 220·vset 7·mac 35·宽 e8m1 load + 手排 MLP)。对手成本中心 (宽 load + intra-super-block MLP·`01-opponent-cost-centers.md` §一) 100% transfer。
2. **同轴我方短板**：decode leaf 窄 e8mf2 (8B) load + clang 低 MLP = vec_dot 第 5 定格未试的 memory-scheduling 轴同型。
3. **同 memory 主导**：M=1 GEVM weight-DRAM-streaming 主导 (权重字节 >> 激活·intensity=1)·与 vec_dot M=1 同 (`02` §二 prefill-vs-decode 摊销对照坐实差别 = 内存不摊销·非 fold 算术)。

## 为何 transfer 仍「未证」(诚实闸)

- **无 perf-stat**：vec_dot 定格 4 (memory-stall) 是 IPC 0.08 / 96% backend-idle 实证。decode leaf 只有 cold ms + vsetvl (A2-b4/b5)。**decode 是否 memory-stall-bound = 未测**·可能反是 unroll/frontend-bound (711KB 全展开·1024 vwmacc·icache 压力)。
- **decode-独有因子** vec_dot 无：(i) x16-interleave 在 M=1 零收益 (转置税无行摊)·(ii) hl8 半宽 @VLEN256 (k1·[ISSUE-019])·(iii) 16-sub-block 格 (q2/q3/q6) 比 8-sub-block (q4/q5) fold 次数 ×2。
- **q4_K@k1 反例警示**：batch4 q4_K@k1 = 1.535 PASS·batch5 q2/q3/q5/q6@k1 全 ≤0.96 ⟹ **家族内都不普适·禁逐格外推** (判别键 sub-block 数·对手强度)。vec_dot floor 亦然。

## ⟹ transfer 的正确用法 (给 main)

- **可 transfer**：对手成本中心 (宽 load+MLP·同符号)·memory-scheduling 轴的**存在性** (decode 同窄-load 短板)·施工原理 (宽 load + 多流·见 `05`)。
- **禁 transfer**：vec_dot 的 surface 三墙叙事·具体 floor 数字 (0.186)·「memory-stall」墙性质 (decode 未 perf-stat)·家族逐格 cold。
- **transfer 的第一验证步** = perf-stat decode leaf (IPC/backend-idle) 证 memory-bound → 才坐实定格 4/5 真 transfer。本 objdump-only task 做不到。

## 出处
- decode/vec_dot leaf histogram 对比：`02-our-decode-cost-center.md` §0
- ISSUE-109 五次定格：`发射器与架构.md:147-154`·归档 task `07-18-q4k-mlp-attack` (第 5 定格 m1-宽度多流)
- q4_K@k1 反例：`A2-batch5-kquant-decode-M1.md:28-29`
