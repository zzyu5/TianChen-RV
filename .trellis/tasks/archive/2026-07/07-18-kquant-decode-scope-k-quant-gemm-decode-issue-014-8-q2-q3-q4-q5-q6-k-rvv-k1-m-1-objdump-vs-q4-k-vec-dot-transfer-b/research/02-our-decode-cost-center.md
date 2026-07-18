# Research: 我方 decode 真成本中心 (objdump·与 q4_K vec_dot 弧线对比·同墙/异墙)

- **Query**: ISSUE-014 成本中心留白 — M=1 decode 我方 leaf 的真成本中心 (fold 非物理地板已四腿证) + 与 vec_dot memory-stall 弧线对比
- **Scope**: internal (只读·objdump leaf 源 + 仓内 perf/cold·零板攻)
- **Date**: 2026-07-18

## ★决定性结构事实：decode leaf ≠ vec_dot leaf (两个不同发射路径)

**判别键 = 核 body 形态** (A2-batch4 §0 铁线：符号串含 "vec_dot" 但 body 不同)：

| | vec_dot leaf | decode leaf |
|---|---|---|
| 文件 | `experiments/active/g7-census/vecdot-rvv/kernels/q4_K.kernel.c` (25KB) | `A2-batch4-gemm-decode-M1-raw/leaves/q4_K_gevm_hl8.c` (711KB·全展开) |
| body op | `typed_super_block_block_dot_loop_body` | `typed_repack_gemv_loop_body` |
| 发射器 | block-dot 路 | `RVVToEmitCKQuant.cpp` front-door repack-gemv |
| **权重 scratch** | **`int8_t v7[256]` aux8·16× vse8 存 + 16× vle8 重载 = roundtrip** | **0 权重 store·576× `vle8_v_u8mf2` 直载→寄存器内 vand/vsrl 解包→vwmacc** |
| 累加器 | 单 i32 链 (v82·serial) | **4-way i32m2 累加器** (v34/v36/v38/v40) |
| min/scale | scalar min-term (16× `lh` + 16 mul serial MAC) | **向量化** (`vwmacc_vx` + fp fold epilogue·8 `vle16` scale) |

**objdump 计数直证** (grep intrinsic histogram)：
- vec_dot leaf：`16× __riscv_vse8_v_i8m2` (存 aux8) + `8× vsetvl_e8m2` + `16× vle8_v_i8mf2` (重载·窄) = **store→load roundtrip 坐实**。
- decode leaf：store 仅 `4× vse32` (最终 f32 结果)·**零权重 vse8**·loads = `576× vle8_v_u8mf2` (窄 8B·全 mf2) + `8× vle16` (scale)·compute = `1024× vwmacc_vx_i16m1` + `608 vand` + `560 vsrl` + `64 vwmacc_vv_i32m2`。

## 一、decode leaf 循环拓扑 (读 `q4_K_gevm_hl8.c:4-70`)

```
for v9 in col_group_count (= nc/16 = 32):        // 输出列组
  v13,v15 = fp32m2 累加器 (vl=8·hl8 半-lane)
  for v17 in block_count (= K/256 = 8):          // super-block
    v23 = *act_scalar                            // M=1 单激活标量
    vle16 6-bit scale → vfwcvt → vfmul           // fp32 fold prep
    v34/v36/v38/v40 = 4× i32m2 零累加器           // 4-way ILP
    scale_min_unpack_superhalf                   // 6-bit scale/min 解包
    <vle8 mf2 权重 → vand/vsrl 解 nibble → vwmacc_vx> // 核心点积
```
- **vl = 8 恒定** (hl8·half_lanes=8)：VLEN128 mf2 满 8·**VLEN256 半用** (应 16·[ISSUE-019] 半宽缺口)。
- 权重经 **x16-interleave 布局** (stride 2304)·但 M=1 只有 1 行激活 ⟹ interleave 的转置摊销收益 = 0 (无 nr 行可摊)。

## 二、真成本中心诊断 (objdump·ISSUE-014 留白的填充候选)

**先排除 (ISSUE-014 四腿证·成本中心不是这些)**：
- ❌ fold 算术：纯算术地板 ratio_floor = 0.889–0.941 > 0.8 (leg①)·fold 单独产不出 0.05–0.36 named-X。
- ❌ 「权重位重建 roundtrip」：**vec_dot 的 aux8 roundtrip 在 decode leaf 结构上不存在** (0 权重 store·objdump 直证)。ISSUE-014 明禁写「成本中心 = 权重位重建」——decode leaf 印证此禁令 (它已寄存器驻留)。
- ❌ 单累加器 serial 链：decode leaf 已 4-way i32 累加器。
- ❌ scalar min-term：decode leaf 已 `vwmacc_vx` 向量化 scale + fp fold。

⟹ **decode leaf 结构上已在 vec_dot「三 lever (register-fusion/vwredsum/minterm-vec) 后」的状态** (roundtrip 消·serial 链消·min-term 向量化)·**这些 vec_dot 弧线的 surface 墙对 decode 是 no-op**。

**残留真成本中心 (objdump 结构推断·两条·分层)**：

**(A) 窄 load + 低 MLP 的 weight-DRAM-stream** — 与 vec_dot 5 次定格同轴：
- decode leaf = `576× vle8_v_u8mf2` = **窄 8B load** (对手宽 e8m1 16B 的一半)·同数据发两倍 load 条数。
- clang 从 C intrinsic 调度·load 紧贴消费者 (vle8→vand→vwmacc 紧循环)·低 MLP。
- 量级：M=1 GEVM 权重字节 (nc×K quant·q4_K 720KB weight cold) >> 激活字节 ⟹ **weight-DRAM-streaming 主导**·arithmetic intensity = 1 (无行复用)。

**(B) M=1 无行摊销 = 内存级 (非 fold 算术)** — 与 prefill 对照坐实：
- **同一 repack-gemv 机制**：prefill (nr≥4·batched) `COLD` dict = q4_K 1.114 / q5_K 1.067 / q2_K 1.112 / q3_K 1.399 / q6_K 0.960 = **全 PASS/parity**；decode (M=1) = 全 named-X。
- ⟹ 差别 = **行摊销 = 权重流的 arithmetic intensity**：prefill 一份权重 tile MAC 对 nr 行 (intensity ∝ nr)·decode 一份权重 load 只用一次 (intensity = 1) → 纯 memory-bound。**这不是 fold 算术不摊销 (ISSUE-014 已证 >0.8)·是权重流内存不摊销**。摊销失败后·谁赢由 **load 宽度 + MLP** (即 (A) 轴) 决定。

## 三、同墙/异墙裁定 (vs q4_K vec_dot 弧线)

**异墙 (surface·NOT transfer)**：vec_dot 的 aux8 roundtrip / serial 链 / scalar min-term (levers 1-3) = decode leaf 已内建消除·**结构上不同核**。

**同墙 (deepest·结构 transfer)**：vec_dot 第 5 次定格 (窄 load + 低 MLP over weight-DRAM stream·墙 = m2 register-pressure·lever = m1-宽度多流) = **decode leaf 同轴** (窄 e8mf2 load·weight-streaming·clang 低 MLP)·且 rvv q2/q3/q4/q6_K 对手 = 与 vec_dot 完全同一符号 (`_vl128` 宽 load + 手排 MLP)。

**★必带诚实缺口 (决定裁定的确定性)**：**decode leaf 从未 perf-stat** (IPC/backend-idle/cache-miss)。vec_dot 的 memory-stall 是 perf-stat 实证 (IPC 0.08·96% backend-idle)；decode 只有 cold ms + vsetvl 计数 (A2-b4/b5)·**无动态瓶颈证据**。⟹ 「decode 同 memory-stall floor」是 **objdump 结构推断·非实测**·可能 decode 反而是 unroll/frontend-bound (全展开 711KB·1024 vwmacc)。**填 ISSUE-014 留白的第一步 = perf-stat decode leaf** (本 objdump-only task 做不到·省板窗)。

## 出处
- leaf 源：`A2-batch4-gemm-decode-M1-raw/leaves/q4_K_gevm_hl8.c` (histogram + :4-70)·`experiments/active/g7-census/vecdot-rvv/kernels/q4_K.kernel.c` (:11-12 aux8·16 vse8)
- prefill 对照：`recon_master_rebuild.py` `COLD` dict (176-177)·decode：`GEMM_DECODE` (183-200)
- vec_dot 弧线：ISSUE-109 (`发射器与架构.md:147-161`)·归档 task `07-18-q4k-mlp-scope` `research/02-our-serialize-rootcause.md`
- ISSUE-014 四腿：`性能与测量.md:114-120`
