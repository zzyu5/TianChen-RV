# Research: 施工入口 (公式墙→必攻·是否复用 vec_dot lever) + 边界登记 (脾气墙 fallback)

- **Query**: PRD §三.3 — 公式墙 → 施工入口 + 复用 vec_dot m1-宽度/STACK lever？脾气墙 → 三步走完的边界登记建议
- **Scope**: internal (施工入口 + 可行性·非代码改)
- **Date**: 2026-07-18

## 一、施工入口 (墙-2 memory-scheduling 公式墙·必攻)

### ★关键发现：vec_dot 的 lever **机制可复用·因发射器共享同一 knob**

decode 的 gemv 发射路 `typed_repack_gemv_loop_body` 与 vec_dot 的 block-dot 路**共享同一 TU** `lib/Conversion/RVV/RVVToEmitCKQuant.cpp`·且该 TU 已有 vec_dot 用过的 **`integer_core_lmul` knob (mf2/m1/m2·stripWidth 8@mf2 / 16@m1)** (`:146-147,210-211,360-365`)。

- **现状 (objdump 坐实)**：decode leaf 走 **LITERAL mf2 链** (`:854-864` "carries NO integer_core_lmul knob, so it passes the LITERAL mf2 chain")·strip load = `vle8_v_i8mf2` (`:233,378`) = **窄 8B**·= 我方测到的 hl8 leaf。
- **施工入口 = 把 m1 knob stamp 到 gemv/decode 路**：coreLmul mf2→m1 使 `WideningChain` (`:365`) 的 l8 从 mf2→m1 ⟹ **strip load 直接从 8B 宽到 16B** = memory-scheduling 轴 (i) load 宽度杠杆。

### ★与 vec_dot lever 的关键差别 (为何 decode 可能有效·vec_dot 无效)

- vec_dot 上 `integer_core_lmul` knob **cold-inert** (best 0.186)·因 vec_dot 的**load 卡在 aux8 roundtrip 的 e8mf2 重载**·knob 只宽了 compute (vwmul/vwmacc)·**没宽 load**。vec_dot 要宽 load 得另造第 5 定格 "mlp" 构造 (循环重构)。
- decode **无 roundtrip** (`02` §0)·strip load = `vle8` 直载·**coreLmul knob 直接就是 load-宽度旋钮** ⟹ 同一个对 vec_dot 无效的 knob·**在 decode 上直击 memory-scheduling 轴** (窄→宽 load)。**这是 decode 独有的·未试的·闭式杠杆**。

### STACK 施工 (镜像对手·[K-10] 结构级)

| # | 组件 | 来源 | 对手对应 |
|---|---|---|---|
| 1 | 宽 load (mf2→m1·8B→16B) | **复用现 `integer_core_lmul` m1 knob·扩 stamp 到 gemv 路** | 对手宽 e8m1 16B |
| 2 | 多流 load 提前发射 (MLP) | 4-way 累加器已有部分 ILP·须循环重构确保 load hoist (vec_dot 第 5 定格 "mlp" 构造原理) | 对手 ~16 独立宽 load |
| 3 | VLEN256 宽化 hl8→hl16 (仅 k1) | **复用 `deriveRepackHalfLanes` 宽化器** (iq3 已证 4 格翻正·机制①·ISSUE-019/105) | 对手 vl256 满宽 |

### byte-exact 计划 ([K-5] ZERO-MODEL·整数免费)

- decode 核 = 整数量化点积 (i8×i8→i16→i32·int32 零舍入)·宽 load / 宽 strip 只改整数加法分组顺序 ⟹ **byte-exact 免费** (发射器自证 `:1170` "provably bit-exact at every legal LMUL·no fp non-associativity")。fp scale-fold 顺序保 per-super-block 不变即可。
- 门：`kquant_gevm_m1_driver_b5.cpp` gate = ours-leaf vs ggml-opp agreement (现档·A2-b5)·可升 ZERO-MODEL (`kquant_repack_verify_q{2,3,6}K.c` 有 `ref_block`)。

### 施工前必做 (顺序·填 ISSUE-014 留白)

1. **★perf-stat decode leaf** (IPC/backend-idle/cache-miss·rvv + k1)：证 memory-bound (公式墙前提)·**这是本 objdump-only task 做不到的·省板窗·须 main 排板批**。若证 memory-bound → 宽 load lever 直击；若证 unroll/frontend-bound → 换轴 (降展开·仍公式墙)。
2. 宽 load knob (mf2→m1) 板测·objdump 证 load 真宽 (`vle8 mf2→m1`)·先测不预告 ([§五.15])。
3. k1 叠 VLEN256 宽化。

### 回门 schema

- **默认无新 schema**：复用现 `integer_core_lmul` knob 命名空间 (同 vec_dot 范式·gated/regression-free/默认 byte-identical)。
- **回门 candidate (条件·单列)**：若多流 MLP 须 IR 层显式 unroll/pipeline 因子 (按 [K-10]/[ISSUE-033] 宽度读自 IR 非焊死) → 新增结构 attr = 回门·单列·agent 不自决须裁。

### 可行性 + 地盘 (诚实)

- **可行性 = 中**：knob 已存·byte-exact 免费·但 **gated on clang 是否保住宽多流调度** (vec_dot 第 5 定格已示 clang 会保 MLP 但撞 m2 register-pressure)。decode 4-way 累加器 + m1 宽度须验不撞同 register-cliff (q5_K 已有 [GAP-Q5K-VLEN128-QH-REGCLIFF])。
- **地盘**：memory-scheduling 是 K-quant super-block decode **共享**成本中心 ⟹ 单构造成 → 扇出全 8 格 (须逐格补 oracle + 板测·禁外推·q4_K@k1 反例警示家族不普适)。
- **成色天花板**：q5_K@rvv/k1 + q5_K/q6_K@k1 对手弱/身份存疑 ⟹ 翻了也须标非硬赢/先核身份。真硬赢价值集中在 q2/q3/q4/q6_K@rvv (vl128 手调) + q3_K@k1 (vl256 手调)。

## 二、边界登记 (仅当攻坚后满足 [K-4] 三步·非现在)

### 墙-1 gcc-death (rvv 5 格) = 脾气墙·**现在即可登记** (三步已走·独立于墙-2 攻坚)

- verdict：登记为 **[CASE-KQUANT-GCC-CODEGEN] 脾气墙** (机制②)·deploy overlay·gcc→clang ~5.5× 修复但 <0.8 不足。**非攻坚核心·非认输核心**——它是 deploy 现实叠加·真攻坚打墙-2。

### 墙-2 memory-scheduling (全 8 格) = **禁现在登记** (违 [K-4] 三步·杠杆非空)

仅当 `施工 §一` 板测后满足全部才登记为脾气墙：
1. objdump 对手✓ (已·`01`)。
2. 定杠杆 = memory-scheduling (宽 load + 多流 + VLEN256)✓ (已·本文)。
3. **板测证伪**：owned 宽 load/多流 emit 构造 + 板测·objdump 证 clang 把宽多流 load 再串行化 (load 未 hoist·cold 仍 <0.8 inert)·**且 perf-stat 证 memory-bound** ——**此步未做·是登记闸**。

若三步走完 → 登记形态 = **具名-X + memory-scheduling clang-调度脾气墙** (同 vec_dot 第 5 定格 fallback·[ISSUE-100]/[ISSUE-107] compiler-maturity 族)·**禁「架构不可达」** (对手同硬件达·结构可达) **禁「honest-null」** (那是结构不可达如 [ISSUE-035])·mechanism-level 具名·禁逐格外推计数。

## 出处
- 发射器 knob：`lib/Conversion/RVV/RVVToEmitCKQuant.cpp:146-147,210-211,233,360-365,854-864,1170`
- 宽化器：`deriveRepackHalfLanes` (iq3 机制①·`K-attack-fanout-ledger.md:13`)
- vec_dot lever 对照：ISSUE-109 (`发射器与架构.md:147-154`)·[K-4] (`覆盖状态机与选择归因.md:20-21`)
- gcc-death：机制② (`K-attack-fanout-ledger.md:14`)
