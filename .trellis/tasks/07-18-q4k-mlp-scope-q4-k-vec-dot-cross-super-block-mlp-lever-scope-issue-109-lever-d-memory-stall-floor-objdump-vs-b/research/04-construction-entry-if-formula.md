# Research: 施工入口（公式墙 → 必攻）— memory-scheduling 轴 register-resident 宽多流 vec_dot Emission Plan

- **Query**: 若公式墙·我方 emit 能否发射 MLP 结构？涉哪些 brick/结构改？byte-exact 计划？是否回门 schema？
- **Scope**: internal（施工入口 + 可行性·非代码改）
- **Date**: 2026-07-18

## 一、施工性质 = 独立大构造（[K-10] 结构级·非旋钮）

memory-scheduling 轴要改**迭代空间拓扑 + load 粒度 + 寄存器驻留**——满足 [K-10]「迭代空间拓扑 / 布局契约 / 优化目标任一变 → 独立 Emission Plan」。**不是** `integer_core_lmul` 旋钮（那是指令流轴·已 EXHAUSTED）。

## 二、构造 = 把三 dormant lever + 两未试件**叠成一个**镜像对手结构的核

三 lever 此前**各自单测**（叠在 baseline 上·各 cold inert），**从未 STACK**·更从未加 load 宽度/MLP。施工 = 一次性叠：

| # | 组件 | 来源 | 对手对应特征 |
|---|---|---|---|
| 1 | 权重寄存器驻留（消 aux8 roundtrip） | 复用 register-fusion emit（`="fused"` dormant） | 特征 2 |
| 2 | per-sub-block vwredsum 归约（消 serial 链） | 复用 vwredsum emit（`="vwredsum"` dormant） | 特征 6 |
| 3 | 向量化 min-term + scale-fold | 复用 minterm-vec emit（`="minterm-vec"` dormant） | 特征 5·7 |
| 4 | **宽 load（e8mf2 8B → e8m1/m2 16–32B）** | **新** | 特征 1 |
| 5 | **寄存器驻留多流 load 提前发射（intra-super-block MLP）** | **新**（循环重构·独立 load streams 展开·load 与消费者解耦） | 特征 3 |

**核心新工 = #4+#5**：把 Region B 的窄紧循环重构为「先把整 super-block 的宽 load 发到多个独立寄存器（多流·让 clang 有多个 outstanding load 可挂）·再统一消费」，匹配对手 asm 的 load-hoist 形态。#1–#3 是把已证 dormant emit 从「单独」变「同核共存」。

## 三、触碰集 / brick 改

- **主 TU**：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp`（q4_K vec_dot super-block block-dot bricks·**独占**·与 dequant emit `RVVToEmitCForwardElementwise.cpp` / `GridCodebook.cpp` 不相交）。
- **shared bricks·硬门**：默认 sealed 路径（md5 `892b6cf8`）**必 byte-identical**——新结构走 gate 后（见 §五）·gate 外零改。
- 可能触 `RVVToEmitCInternal.h`（若新循环形态需新 loop-body 结构描述）+ test/lit fixture。

## 四、byte-exact 计划（[K-5] ZERO-MODEL·整数免费）

- q4_K vec_dot 是**整数量化点积**（int8×int8→int16→int32·min 项 int16·**int32 累加零舍入**）。宽 load / 多流 / 归约粒度改的只是**整数加法的分组顺序**——整数加法结合律 ⟹ **byte-exact 免费**（前例：vwredsum/minterm-vec 均报 ULP=0）。
- 唯一须守：fp scale-fold（`d`/`dmin` × int32 结果）的**每-sub-block fold 顺序**须与 oracle 一致（fp 非结合）。对手/我方现结构均 per-super-block fp fold·保持即可。
- 门：`vec_dot.sh rvv verify q4_K` → 3-way byte-exact ALL=true + 4-arm 反空心 + **报 ULP=0**。

## 五、是否回门 schema

- **默认无新 schema**：复用现 `integer_core_lmul` gate 命名空间（同三 dormant lever 范式·新增一个 gate 值如 `="mlp-wide"`·dormant/gated/regression-free·默认 byte-identical）⟹ **不必回门**。
- **回门 candidate（条件触发·须单列）**：若 #5 多流重构需在 **IR 层显式表达** unroll/pipeline/load-stream 因子（按 [K-10]/[ISSUE-033]「宽度/结构读自 IR·非发射器焊死字面量」纪律）·则新增一个 IR 结构 attr = **新 schema = 回门**·**单列**（同阶段2 路 B / [ISSUE-116] 范式·cross-layer·agent 不自决·须裁）。
- **判别**：若新结构能从现有 IR 向量类型/loop-body op 推出（如复用 `accVecType.getLmul()` fail-closed 读法·[ISSUE-033] 域内存在性证明的范本 `RVVToEmitCDeferredDequant.cpp`）→ 无需回门；若须新增一等结构因子 → 回门单列。

## 六、可行性 + 地盘预期（诚实）

- **可行性 = 中·gated on clang 调度成熟度**（`03-verdict` §caveat 的脾气墙-风险子墙）：结构（对手证算法可达·byte-exact 整数免费）可发；**但「clang 是否保住 emit 的 MLP 调度」是本构造的真验收门**（register-fusion 已示 clang 不会自发产 MLP）。⟹ 攻坚**先测不预告**（[§五.15]）：
  - cold ≥0.8 → **q4_K vec_dot 公式墙翻**（关 ISSUE-109·消具名-X·真硬赢强手调 vl128·+ q6_K/q2_K/q3_K/q5_K same-family 就绪·地盘 +1↑）。
  - cold <0.8 且 objdump 证 clang 再串行化（load 未 hoist·MLP 未达）→ 降为**脾气墙·5 次定格**（`05-boundary-registration`）。
- **地盘预期**：翻则命中面 = q4_K/q6_K/q2_K/q3_K/q5_K vec_dot@rvv（同 weight-reconstruction floor·2 板测数据点已证同 floor·**须逐格补 oracle+板测·禁外推**）+ @k1 VLEN256 半宽 co-factor（gated ISSUE-105）。
- **公式占比**：memory-scheduling 是 vec_dot 家族**共享**成本中心 ⟹ 单构造若成→扇出全 K-quant super-block vec_dot·高扇出。

## 七、施工纪律（承 ISSUE-109 三次 re-diagnosis 教训）

- **re-roll trap 铁律（4 次证伪）**：每步先反汇编·证 load 宽度真升 / MLP（outstanding load）真增·未增=无效=停该步具名·**禁硬凑**（[GAP-P1] 宪章规则 2）。
- worktree 隔离·rvv 板·clean rebuild 亲见（[[build-incremental-unreliable]]）·sealed 不动·master 不直写·**禁 commit / 禁 add -A**·0 造数·"某物不存在"禁截断。
