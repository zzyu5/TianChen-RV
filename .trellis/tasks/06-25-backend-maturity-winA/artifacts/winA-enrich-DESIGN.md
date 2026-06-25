# Win-A 丰富化设计 — 后端能力驱动 lowering 的成熟化

> 任务定位：Win-A = 我们对标 Triton 后端的那层旋钮（**能力驱动的 lowering 决策**：
> LMUL / 向量化 / 流水 / 选指令 / tail），拿给定算子自动生成该硬件最优码。
> 这是后端编译器成熟化的主线，**不是选算法**（repack-vs-block-dot 那种已 demote 为前端 / option-2）。
> 本文档只读调查 + 设计，不改 `lib/` 代码。

---

## 0. 现状一句话

block-dot 的 5 个 kernel（q4_0/q8_0/q4_1/q5_0/q5_1）+ 1 个 GEMM-M 走真正的
schedule-descriptor auto-stamp flow（gearbox 枚举 → 剪枝 → 选 → stamp），有 3 个旋钮
（`integer_core_lmul` / `multi_block_factor` / `strip_elision`）；但**默认选择是 capability-blind
的 static cost-model argmin**，gearbox 唯一看到的 capability fact 是 **1 个布尔 `hasZvl128b`**，
measured>static 只在显式传 `--tune-record` 时生效（默认空 = static），而 **~20 个 K-quant / IQ / TQ
block-dot 根本不在这个 flow**（无 `TunableScheduleOpInterface`，shape 靠 emitter 写死）。

---

## 1. 现状精确地图（旋钮 + 选择逻辑 + 可见 fact + auto-stamp 覆盖）

### 1.1 旋钮（block-dot 的 lowering 决策轴）

| 旋钮 | 取值 | 定义 / 枚举点 | emitter 消费点（真改码处） |
|---|---|---|---|
| `integer_core_lmul` | `mf4`,`m1`(q8_0 加 `m2`) | `enumerateBlockDotShapeCandidates` 三重循环 `RVVGearboxSchedule.h:2522` | `RVVToEmitCBlockQuantLinear.cpp:246` `setvlSEW=(coreLmul=="m1")?8:32` + wideLmul 选 |
| `multi_block_factor` | `1`,`2`,`4` | 同上 `kFactors[]` `RVVGearboxSchedule.h:2518` | 外层 blocks/iter 展开 |
| `strip_elision` | `robust`,`elided` | 同上 `kElisions[]` `RVVGearboxSchedule.h:2519` | `RVVToEmitCBlockQuantLinear.cpp:250-271` 分支：elided=单 `vsetvl_e8m1(16)`+1 reduce，无内层 strip 循环 |
| GEMM `activation_cols`(M) | 枚举 vreg-ceiling 内 | `enumerateRVVGemmMCandidates` | GEMM M-block tile |

另有一条**独立、已 live-wire 但与 block-dot flow 不交叉**的轴：

| 旋钮 | 状态 | 定义点 | 消费点 |
|---|---|---|---|
| accumulator-LMUL rung (i8→i16→i32 的 acc 宽度) | 已建 + live-wire，但只服务 `TypedWideningProductReduceDequantize` dispatch | `selectRVVLowPrecisionMaxLegalAccumulatorLMULRung` + `RVVLowPrecisionLMULRung` `RVVGearboxSchedule.h:1903-1968` | `RVVContractionSelectedBodyRealizationOwner.cpp`（按 vreg budget 自动发 i8m2→i16m4→i32m8 宽体） |

### 1.2 选择逻辑（怎么枚举、怎么选）

- **枚举 + 剪枝**：`enumerateBlockDotShapeCandidates` `RVVGearboxSchedule.h:2513`
  - 全笛卡尔积 `{anchors} × {1,2,4} × {robust,elided}`。
  - 剪枝 (a) **legality**：`elided` 只在 `coreLMUL==elidedLegalAnchor && hasZvl128b` 合法
    (`:2536-2537`) —— **这是 capability 唯一进入选择的地方**。
  - 剪枝 (b) **budget**：`vectorRegisterCost <= vectorRegisterBudget(=32)` (`:2539`)，
    在这些轻 kernel 上从不 bind（注释自承 `:2507`），但 prune 机制真实。
- **cost 公式**：`computeBlockDotShapeCostCore` `RVVGearboxSchedule.h:2445-2462`
  ```
  cost = kReductionUnit(600)*reductionsPerBlock
       + kOuterLoopOverhead(500)/min(factor, coreLatencyDepth)
       + kUnrollOverflowPenalty(250)*max(0, factor - coreLatencyDepth)
       + kStripPenalty(elided 40 / robust 170)*factor
       + kBaseConstant(120)
  ```
  - 输入：`reductionsPerBlock`(mf4=4/m1=1, `:2264`)、`factor`、`elision`、`coreLatencyDepth`
    （`= 2 + decodePrefixLength(format)`，`getRVVBlockDotDecodePrefixLength` `:2415`）。
  - **为什么 capability-blind**：公式参数里**没有任何 capability fact**（无 VLEN/ELEN/vreg 数）；
    设计注释 `:2142-2152` 明确把"cost 是 capability-BLIND 的纯结构函数"当作 *anti-lookup* 卖点。
    capability 只通过 legality prune 改变*被准入的候选集*，不改公式。常量是对 ssh-rvv sweep 的
    measurement-calibrated 相对排名（`:2331-2338`），不是绝对 ns 预测。
- **选择策略**：`selectGenericSchedule` `RVVGearboxSchedule.h:3318`
  - (1) measured-best：`recordText` 存在 + `(kernelKey,march)` 命中 + revalidate 仍合法 → stamp 实测形状（带 ns）。
  - (2) 否则 static argmin：`selectGenericMinCostCandidate` `:2248`（legal 集里 cost 最小）。
  - fail-closed：全被剪 → nullopt（I7）。
- **运行入口**：`runRVVScheduleMaterializationViaInterface` `RVVScheduleDescriptorRegistry.cpp:146`
  - `hasZvl128b = deriveHasZvl128b(march, isaVectorHints)` `:154` —— **唯一 capability 输入**（注释自承 `:152-153`）。
  - `recordText = loadRVVBlockDotTuningRecord(tuneRecord)` `:158`，`tuneRecord` 默认 `""`（`Passes.td:395`）→ **默认 static**。
  - walk 全模块，`dyn_cast<TunableScheduleOpInterface>` 自动发现 target（非硬编码 op 列表）`:170`。

### 1.3 可见 fact（gearbox 现在能看到 vs 已可派生但没喂进去）

- **现在喂进 cost/select 的**：仅 `hasZvl128b`（布尔）。
- **已经有派生函数、但没喂进 block-dot cost/legality 的**（在 `RVVCapabilityProfile.h/.cpp`）：
  - `deriveMinimumVLEN(march, hints)` → **真实 VLEN bits**（128/256/512…），`hasZvl128b` 只是 `>=128`
    的布尔截断（`RVVCapabilityProfile.cpp:298-303`）。
  - `deriveSupportedSEWAllowList` → `"8,16,32,64"` 等（`:111`）。
  - `deriveSupportedLMULAllowList` → RVV1.0 全 grid vs RVV0.7 `m1,m2,m4,m8`（`:130`，无 fractional）。
  - `deriveRVVVersion` → RVV1.0 / RVV0.7（`:317`）。
  - 架构 vreg 预算 32 是常量（`kRVVQ40ShapeVectorRegisterBudget` `:2160`）。
- **结论**：VLEN/ELEN/vreg-count/SEW-allowlist/LMUL-allowlist **都已可派生**，但 block-dot 的 gearbox
  只消费了其中 1 bit（`VLEN>=128`）。这正是 spec line 53 列的 resource-fact 清单里大部分没接通。

### 1.4 auto-stamp 覆盖（谁在 flow 内 / 谁靠手 attr）

- **在 flow 内（自动 stamp，6 个）**：`GgmlBlockDotQ40Q80Op`(`RVVOps.td:3873`)、`Q80Q80`(`:5146`)、
  `Q41Q81`(`:5238`)、`Q50Q80`(`:5350`)、`Q51Q81`(`:5921`)、`GgmlGemmQ40Q80Op`(`:4134`)
  —— 都有 `DeclareOpInterfaceMethods<TunableScheduleOpInterface>` + 注册在
  `lookupRVVScheduleDescriptor`（`RVVScheduleDescriptorRegistry.cpp:43-141`）。
- **不在 flow 内（~20 个，emitter 写死 shape，无旋钮）**：
  `GgmlBlockDotQ4KQ8KOp`(`:6360`)、`Q5KQ8K`(`:6487`)、`Q6KQ8K`(`:6141`)、`Q2KQ8K`(`:6582`)、
  `Q3KQ8K`(`:6698`)、`TQ20/TQ10`、全 `IQ*`、`MXFP4/NVFP4`、`Q10` 等 —— **都没有 `TunableScheduleOpInterface`**，
  无 descriptor，walk 直接跳过。K-quant 的 LMUL 是 emitter 端定的（如 `RVVToEmitCKQuant.cpp` 产 i32 m1），
  RVV0.7 的 whole-LMUL anchor 是 `RVVRepackStripWidthMaterialization.cpp:39` 硬 pin `integer_core_lmul="m1"`，
  不是 gearbox 选出来的。
- **关键诚实点（emitter 是 VLEN-tier-blind 的）**：elided cover 发的是
  `vsetvl_e8m1(halfBlock=16)`（`RVVToEmitCBlockQuantLinear.cpp:257-262`），AVL **恒为 16**
  （= quant 格式常量 `qk/2`），**VLEN256 下产出的 C 与 VLEN128 逐字节相同**。emitter 只对布尔
  （elided 合法与否）分叉，不对 VLEN 档分叉。← 这条决定下面候选 (a) 的真实成本与排序。

---

## 2. 四个「丰富 Win-A」候选

> 贯穿性验收标准（来自项目 recurring NULL 教训：Win-C structural NULL / "micro 胜 e2e washes null"）：
> **每个候选必须能命名两板上产出的 C 的具体指令差异**。若说不出"哪条不同的指令出来了"，那它不是 Win-A，
> 是 audit-attr NULL。下面每个候选都标了"emitted-C 分叉点"。
> 分类轴 = **selection-only（只改选择，emitter 已能产对应形状）** vs **emitter-dependent（必须先让 emitter 能发新形状）**。

---

### 候选 (c) 让 measured>static 成默认 / 真闭环 ★ 推荐先做（selection-only，divergence 已存在）

- **改什么**：把"per-(kernel,march) 实测记录在已知 march 上默认被消费"做成真闭环：
  (1) 在仓内固化一份 `ssh rvv` 跑出的 block-dot tuning record（覆盖 q4_0..q5_1 × {rv64gcv, rv64gc_zve32x}），
  (2) 让 default pipeline 在 march 已知时把 `--tune-record` 指向它（unknown march → 仍 fail-back static），
  (3) 把 `tune_block_dot.py` 的 dump→measure→record 跑通并把 record 纳入版本控制 + 一个 lit/CI 校验 record 仍 revalidate。
- **入口**：选择策略 `selectGenericSchedule` `RVVGearboxSchedule.h:3318`（已实现，不改逻辑）；
  record 读 `loadRVVBlockDotTuningRecord` + `lookupGenericTuningRecord`；默认空在 `Passes.td:395`；
  驱动 `…/06-22-…/k1-vlen256/tune_block_dot.py`。
- **为什么是 lowering 成熟化而非选算法**：这是 spec line 63「实测 > 静态：实测赢家是该 key 的*形状* authority，
  static 降级为 pruner + fallback」的直接落地。选的仍是同一算子的 **shape 旋钮**（LMUL/factor/elision），
  不是换算法。它把"static cost 每加一个 kernel 就补一维 = curve-fit treadmill"（line 63 原话）换成真机定档。
- **emitted-C 分叉点（已存在，最强证据）**：q4_1 的 record 已注释 "FIXES the static mis-pick"
  （`RVVScheduleDescriptorRegistry.cpp:78-79`）——即 static argmin 选错形状、measured 选对，**两者 emit 不同的
  factor/elision 形状 → 不同的 vsetvl/展开**。divergence 是既成事实，本候选只是让它默认生效。
- **风险**：**最低**。逻辑已建，纯属"固化 record + 接默认 + CI 守门"。风险点是 record 的 march-key 必须严格匹配
  （否则静默退 static）；record 必须先过 byte-exact gate 再排名（line 63 要求）。
- **byte-exact 门**：record 的每个候选在写入前过 `_generic` 逐字节 gate（驱动已做）；接默认后，
  stamp 出的 IR 在"有 record"与"record 缺失退 static 且恰好同形状"两路下 byte-identical（forced clean rebuild + BEFORE/AFTER-EQUALITY，按 memory 的 build 纪律，别用绝对指纹）。
- **两板验证点**：rv64gcv（Zvl128b）与 rv64gc_zve32x（非 Zvl128b）各取 record-best 形状，ssh-rvv 实测 best-of-N
  胜 `_generic` static 形状；非 Zvl128b 板必须仍只在 robust 集里选（elided 被剪），证明 capability 轴未被 record 旁路。
- **工作量**：小～中（record 生成是离线一次性 + CI wiring；无 emitter 改动）。

---

### 候选 (a) cost model 真能力/资源感知（VLEN 档驱动结构事实，替代 1 bit）（emitter-dependent）

- **改什么（按 spec 的 anti-lookup 纪律 reframe）**：**不**给 `computeBlockDotShapeCostCore` 加 VLEN 参数让公式分叉
  （那直接违反 `:2142-2152` 的 capability-blind 卖点 + line 55「resource fact 驱动*枚举+剪枝*，不是更胖的公式」）。
  正确做法：让 **结构事实 + 候选集 + legality 按 VLEN 档参数化**，再喂进同一 blind 公式：
  - 把 `deriveMinimumVLEN` 的真实 bits 接进来（替代 `hasZvl128b` 布尔），
  - 在 VLEN256 上新增/准入"宽 cover"候选：一次 `vsetvl_e8m1(32)` 覆盖**两个 half-block**（VLEN256 的 VLMAX≥32），
    它的 `reductionsPerBlock` / `multi_block` 结构事实与 VLEN128 的单-half-block cover 不同 → 同一 blind 公式自然选出不同形状。
- **入口**：legality/结构事实 `enumerateBlockDotShapeCandidates` `RVVGearboxSchedule.h:2513-2543`；
  VLEN 派生 `deriveMinimumVLEN`（`RVVCapabilityProfile.cpp`，已存在，只是没接 block-dot）；
  **emitter 必须改**：`RVVToEmitCBlockQuantLinear.cpp:257-262` 的 `sizeLit(halfBlock=16)` 要能在 VLEN256 档发 `32`。
- **为什么是 lowering 成熟化而非选算法**：直接落地 spec line 53「候选空间由编译器可见的 capability+body facts 推导」
  的 VLEN 维 + line 55「真 resource-aware ⟺ 枚举并按 resource facts 剪枝」。选的仍是 block-dot 的 cover 形状，不是换算法。
- **emitted-C 分叉点（必须先建，否则 NULL）**：当前 elided cover 恒发 `vsetvl_e8m1(16)`，VLEN256 与 VLEN128
  逐字节相同（§1.4 已证）。所以**本候选若只改 selector 而不改 emitter = structural NULL**（重蹈 Win-C）。
  真分叉 = VLEN256 板上 emit `vsetvl_e8m1(32)` + 半数 reduce 次数。这是 emitter-dependent，成本在 emitter 不在 wiring。
  - **scale-boundary 注意**：`halfBlock=qk/2=16` 是一个完整 q4_0 权重块 = 一个 scale 单元；AVL=32 跨**两块/两 scale**，
    属 multi-block 范畴（不是更宽的单块 cover），宽 cover 机制与 scale 边界交互，确切的 emitted 形态需在实现时设计。
    这只会**加强 (c)>(a) 的排序**（(a) 的 emitter 设计更重）。
- **风险**：中～高。NULL 陷阱（必须先有 emitter 的宽-cover 形状）；正确性（VLEN256 单 vsetvl 跨两 half-block 的 nibble
  decode + 跨 block scale 不能串味——half-block 是 scale 边界）；需要真 VLEN256 板（k1-vlen256 profile 已有 artifacts）。
- **byte-exact 门**：新宽-cover 体先过 `_generic` byte-exact；VLEN128 路必须**不受影响**（仍发 16-AVL，逐字节不变）——
  即"加 VLEN256 候选"对 VLEN128 stamp 零回归是硬门。
- **两板验证点**：VLEN128(rv64gcv) emit 不变 + 实测不退；VLEN256(k1/rv64gcv_zvl256b) emit 出 32-AVL 宽 cover
  且实测胜 16-AVL 形状。两板 emit **必须不同**（否则 NULL）。
- **工作量**：中～大（selector 小，emitter 宽-cover + 正确性 gate 大）。

---

### 候选 (b) 更多 kernel 接入 gearbox 自动选（K-quant block-dot：手 attr → 自动 stamp）（wiring 易、emitter 难）

- **改什么**：给 `GgmlBlockDotQ4KQ8KOp`（先一个）加 `DeclareOpInterfaceMethods<TunableScheduleOpInterface>` +
  `getScheduleKernelKey()="q4_K"` + 在 `lookupRVVScheduleDescriptor` 注册一个 K-quant descriptor（anchor 集 / 每-anchor
  reduction count / vreg footprint / 量化格式 → latency depth）。然后 K-quant 的 LMUL/factor/elision 由 gearbox 选 + stamp。
- **入口**：op 声明 `RVVOps.td:6360`；descriptor 注册 `RVVScheduleDescriptorRegistry.cpp:43`（仿 q4_0 那 6 行）；
  emitter `RVVToEmitCKQuant.cpp` 必须按 stamp 的 LMUL 发不同体。
- **为什么是 lowering 成熟化而非选算法**：把"K-quant 的 shape 由 gearbox 按能力选"是 Win-A（shape 旋钮）；
  注意**别越界**：选"K-quant block-dot vs repack"是前端（option-2），本候选只做 *block-dot 内部 shape* 那一侧。
- **emitted-C 分叉点 + 诚实警告**：wiring 本身 *mechanically trivial*（加接口 + 6 行 descriptor）。但 **hollow 除非
  K-quant emitter 能真发多个 LMUL 形状**——而宽-LMUL K-quant emit 正是 memory 记的"timed out 2×、multi-day 难点"。
  spec line 55 明说"固定单候选 / 固定 LMUL 只是 MVP 占位，不是 N3"。所以 **(b) 的真实成本在 emitter 的 wide-LMUL
  K-quant 发射，不在接线**；只接线会得到一个"形式 tunable、实则单候选"的假成熟。
- **风险**：高（在 emitter）。低（在 wiring）。需诚实区分两者，别用 wiring 假装覆盖。
- **byte-exact 门**：每个新 LMUL 形状过 `_generic` byte-exact；接入后默认 stamp 的形状若与现状同形 → emit 逐字节不变（零回归门）。
- **两板验证点**：至少 2 个 LMUL 形状在 ssh-rvv emit 不同指令 + 各自 byte-exact；宽形状实测胜窄形状（否则是占位）。
- **工作量**：大（emitter 主导，对齐 memory 的 wide-LMUL K-quant 目标）。

---

### 候选 (d) 把已建的 acc-LMUL rung 轴接进 block-dot tunable flow（emitter-dependent）

- **改什么**：现有 `selectRVVLowPrecisionMaxLegalAccumulatorLMULRung`（`RVVGearboxSchedule.h:1903`，按真 vreg budget 选最宽 acc-LMUL，
  已 live-wire 到 `TypedWideningProductReduceDequantize`）只服务那一条 dispatch；把 acc-LMUL 作为**第 4 个 block-dot 旋钮**
  并入 schedule-descriptor flow，让 block-dot 也能按 vreg budget 选 i32 累加器宽度（m1 vs m2…）。
- **入口**：rung 选择器 `RVVGearboxSchedule.h:1903-1968`（已建，复用）；并入点 `enumerateBlockDotShapeCandidates`
  + `NamedKnob` 列表（加 `accumulator_lmul` 轴）；emitter `RVVToEmitCBlockQuantLinear.cpp` 的 i32 reduce/accumulator 段。
- **为什么是 lowering 成熟化而非选算法**：acc-LMUL 是经典寄存器压力驱动的 lowering 决策（line 53 列的
  "accumulator 数量与 reduction layout" + "register-pressure-safe unroll"），按真 vreg budget 剪枝 = 真 resource-aware。
- **emitted-C 分叉点**：block-dot emitter 当前 i32 reduce 固定 m1（`vwredsum…_i32m1`，`:212`）。真分叉 = budget 宽松时发更宽 acc
  → 不同 `vwredsum`/累加 LMUL。**仍是 emitter-dependent**：block-dot 体现在要能发多 acc-LMUL 形状。
- **风险**：中。acc-LMUL 在 block-dot 的 16-元素 half-block 上收益可能很小（reduce 已 LMUL-independent in result）；
  有沦为 NULL/无收益的风险——上板前先确认存在能赢的形状。
- **byte-exact 门**：每 acc-LMUL 形状过 `_generic`；默认形状零回归。
- **两板验证点**：budget=32 与人为缩 budget 两路 emit 不同 acc-LMUL（如已建 rung 的 i32m8 vs 退窄），各 byte-exact。
- **工作量**：中（选择器已建，emitter 多 acc-LMUL 体 + 收益存疑需先验证）。

---

## 3. 价值/风险排序 + 推荐次序

| 次序 | 候选 | 类型 | 价值 | 风险 | 关键判据 |
|---|---|---|---|---|---|
| **1（先做）** | **(c) measured>static 默认闭环** | selection-only | 高（直接落地 line 63 实测胜出，divergence 已实在存在 q4_1 mis-pick） | **低** | 机器已建，纯固化 record + 接默认 + CI 守门，无 emitter 改动，无 NULL 风险 |
| 2 | (a) VLEN 档驱动结构事实/候选集 | emitter-dependent | 高（line 53 VLEN 维 + 真两档分形状） | 中高 | 必须连 emitter 宽-cover 一起做，否则 structural NULL（§1.4 已证 emitter VLEN-blind）|
| 3 | (b) K-quant 接 gearbox auto-stamp | wiring 易/emitter 难 | 高（覆盖面 + memory 的 wide-LMUL K-quant 目标） | 高 | wiring trivial 但 hollow，真成本=emitter wide-LMUL 发射（timed-out 难点）|
| 4 | (d) acc-LMUL 并入 block-dot | emitter-dependent | 中（轴已建，但 block-dot 上收益存疑） | 中 | 上板前先确认存在能赢的 acc-LMUL 形状，否则 NULL |

**推荐先做 (c)**，三条理由：
1. **真体现后端优化且无 NULL**：它是唯一 divergence *已经存在*（q4_1 static mis-pick vs measured-fix）、机器*已建*、
   不依赖任何新 emitter 形状的候选——直接把 spec line 63 的"实测 > 静态、static 降级为 pruner+fallback"做成默认行为。
2. **可两板 byte-exact 验证**：record 先过 `_generic` byte-exact 再排名（line 63 硬性要求），两板（Zvl128b / 非 Zvl128b）
   各验 record-best 形状实测胜 static 形状，且非 Zvl128b 板仍只在 robust 集选（capability 轴未被旁路）。
3. **不越界到前端**：选的全程是同一算子的 block-dot **shape 旋钮**（LMUL/factor/elision），不碰"换算法/repack"那条 option-2 前端线。

做完 (c) 后再上 **(a)**（带 emitter 宽-cover，吃 VLEN256 真两档），它是把 capability 从 1 bit 升到真 VLEN 维的主升级；
(a) 落地后 §1.4 那条"emitter VLEN-blind"的诚实缺口被补上，gearbox 才真正"按 resource facts 剪枝"（line 55 判据）。
(b)/(d) 排后，因为它们的真实成本都压在 emitter 端（wide-LMUL K-quant / 多 acc-LMUL 体），且需先证明存在能赢的形状以避免占位式假成熟。
