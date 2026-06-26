# Track A — gearbox cost model: capability-blind → resource-fact 驱动（只读设计）

> 任务定位：把 gearbox 选 LMUL/factor/strip 的 cost model 从「capability-blind（只 1-bit）」
> 升级成真资源 fact 驱动（spec N3「能力驱动 lowering 选择」后端线）。
> 本文档只读调查 + 设计，不改 `lib/` 代码。

---

## 0. 前提纠正（必须先读 —— 任务前提是 PRE-`minimumVLEN` 快照）

任务现状写「cost model 唯一能力输入是 1 个 bool `hasZvl128b`」。**这描述的是历史快照，
当前代码已不成立**：

- `enumerateBlockDotShapeCandidates`（`RVVGearboxSchedule.h:2634`）现在接收的是
  **真实 `minimumVLEN` bits**（`deriveMinimumVLEN`，0/128/256/512…），不是布尔。它用真 VLEN：
  (a) **elided-cover legality**（`:2695` `elisionVLMAX >= blockLen`）、
  (b) **codebook gather-table 剪枝**（`:2707` `elisionVLMAX >= gatherTableEntries`）、
  (c) **reduction-rank**（`:2669` `ceil(blockLen / rankVLMAX)`）。
- 入口 `runRVVScheduleMaterializationViaInterface`（`RVVScheduleDescriptorRegistry.cpp:213`）
  `deriveMinimumVLEN(march, hints)` 是喂进选择的真值；`hasZvl128b`（`:214`）现在**只剩
  audit stamp 用**（`stampRVVSchedule` 的 `has_zvl128b` 镜像），不再是选择输入。
- `computeBlockDotShapeCostCore`（`:2472`）**确实仍 capability-blind，且这是 BY DESIGN**：
  header `:2142-2152` / `:2463-2471` 把「cost 是纯结构函数、无 capability 参数」当作
  *anti-lookup* 卖点（每加一个 kernel 就补一维公式 = curve-fit treadmill，spec line 55 明禁）。

**纠正后的真问题**（也是 spec line 53/55 的真本意）：fact 进入的是 **枚举 + legality 剪枝**，
**不是**标量公式。已有的 sealed q8_0 VLEN-flip（k1 m1-elided@256 / rvv m2-elided@128，
`q8-vlen-flip-BOARD-FINDING.md`）就是「VLEN fact 驱动选择改变」**已经落地**的证据 —— VLEN 维
这件事已经做完了。所以 Track A 的纯-wiring 范围里，**剩下的可派生-未用 fact 只有一个**，且它是
legality 收紧（防错码），不是新 perf 数。下面把这个结论坐实。

---

## 1. 可派生资源 fact 清单（从 RVVCapabilityProfile / march 现在能派生哪些）

| fact | 派生函数 (file:line) | 已喂进 block-dot 选择? | 类别 |
|---|---|---|---|
| 真实最小 VLEN bits | `deriveMinimumVLEN` `RVVCapabilityProfile.cpp:253` | **是**（legality + gather + rank） | 已用，已封 q8_0/codebook flip |
| Zvl128b 布尔 | `deriveHasZvl128b` `:298`（`= minVLEN>=128`） | 退化为 audit-only | VLEN 的截断，冗余 |
| RVV 代次 (1.0/0.7) | `deriveRVVVersion` `:317` | 否（但 repack 路已用，block-dot 路未用） | 是 LMUL-allowlist 的上游 |
| **LMUL 支持 allowlist** | **`deriveSupportedLMULAllowList` `:224`**（RVV0.7=`m1,m2,m4,m8` 无 fractional；RVV1.0=全 grid 含 mf8/mf4/mf2） | **否**（grep 证实未进 schedule 路） | **唯一可派生-未用-真 fact** |
| SEW 支持 allowlist | `deriveSupportedSEWAllowList` `:186`（`8,16,32` 或 `8,16,32,64`） | 否 | **NULL trap（见 §1.1）** |
| 架构 vreg budget (=32) | 常量 `kRVVQ40ShapeVectorRegisterBudget=32` `RVVGearboxSchedule.h:2160` | 是（但是常量，非派生） | **NULL trap（见 §1.1）** |
| RVV 代次 ta/ma 策略 | `deriveRVVVersion` `:317` → `rvv_version` property `RVVCapabilityProfile.cpp:478`（RVV0.7 LACKS ratified ta/ma；RVV1.0 有） | block-dot 候选空间**无 tail/mask 旋钮** → 不进选择 | **legality-only（见 §2.4）** |
| 实测 VLEN bytes | probe `vlenbBytes` `RVVCapabilityProfile.h:64`（只 stamp 成 capability property `:496`） | 否 | 仅作 ≤minVLEN 的实测注脚，非新轴 |

### 1.1 NULL traps（明确列「为什么是 NULL」本身就是产出，不是缺口）

- **vreg-count / budget = NULL**：RVV ISA 在**每个 profile 上都恰好 32 个 vector register**
  （ISA 不变量，不是 derivable-and-varying 的 fact）。budget=32 已经是对的；「派生 budget」
  得到的还是 32 = 零选择变化。block-dot 这些轻 kernel 的 peak-live ≤6 vregs（`:2293` 自承），
  budget prune 从不 bind。**别把常量包装成派生 fact 假装成熟。**
- **SEW / ELEN / SEW64 allowlist = NULL（对 block-dot）**：所有 block-dot kernel 只用
  SEW 8/16/32（i8 load → i16 product → i32 reduce），**连 zve32x 都全支持**。
  `deriveSupportedSEWAllowList` 对这些 kernel 永远 prune-pass → 零选择变化。
  （SEW64/ELEN 只在别处 f64/i64 体相关，不在 block-dot 轴。）
- **`vlenbBytes`（实测）= 非新轴**：它 ≤ `deriveMinimumVLEN`（保证下界），但选择已经用保证下界
  做 fail-closed legality（`:2645` 注释明说 legality 必须用 GUARANTEED minimum，不能用乐观实测），
  喂实测值会**放松** fail-closed 边界，是 anti-pattern，不做。

---

## 2. 唯一真 fact：LMUL 支持 allowlist → fractional-anchor 剪枝（RVV0.7）

### 2.1 这个 fact 怎么进选择（合法性剪枝，不进 cost rank）

- **进哪类**：**合法性剪枝 (a)**，与现有 elided/gather 剪枝同列，**不进 cost rank**（cost 仍 blind）。
- **入口 file:line**：
  - 候选枚举 `enumerateBlockDotShapeCandidates` `RVVGearboxSchedule.h:2655` 的 anchor 循环：
    对每个 `coreLMUL`，新增一条 legality —— 若该 anchor 的 **EMUL（load LMUL 或 strip LMUL）
    落在 fractional 档（mf8/mf4/mf2）且 target 的 LMUL-allowlist 不含该档** → `isLegal=false`。
    具体两处 EMUL 角度同一 fact：mf4 anchor 的 i8 load 是 `i8mf4`、i16 product 是 `i16mf2`，
    两者在 RVV0.7 都不存在（`deriveSupportedLMULAllowList` RVV0.7 无 mf2/mf4）。
  - 新增一个 descriptor 字段（如 `supportedLMULAllowList` 或直接传 `bool hasFractionalLMUL`）
    +`enumerateBlockDotShapeCandidates` 新增一个入参，由
    `RVVScheduleDescriptorRegistry.cpp:213` 处一次性 `deriveSupportedLMULAllowList(march,hints)`
    （或 `deriveRVVVersion`）派生后传入 —— **复用 §3 单一真值源，不造第二套**。
- **为什么不进 cost rank**：fractional anchor 在 RVV0.7 是**非法**（不存在的指令），不是「更贵」；
  非法候选不参与 argmin，所以是布尔剪枝不是代价。

### 2.2 这个 fact 是否改变某 kernel 的选择？—— 诚实结论：legality 收紧，非 perf 改变

逐一查「fractional anchor 是否曾被 cost model 选中」（决定它是 §4 类别(b) 防错码 还是真选择翻转）：

- **mf4 anchor（q4_*/q5_*/q8_0）永不胜**：mf4 的 reduction count = 4（VLMAX 4 vs 16-blockLen），
  m1 = 1（`:2282`）；cost 里 `kReductionUnit*reductions` 让 mf4 永远比 m1 贵 4×，
  **且 m1 永远合法**。所以 m1 永远是 argmin，mf4 从不被选中。
- **codebook mf2 anchor（iq4_nl/mxfp4）只在 VLEN256 准入**：mf2 的 gather-VLMAX = 8 < 16
  → VLEN128 已被现有 gather 剪枝剪掉（`q8-vlen-flip`/`fp4-mf2-k1` 已封）；只在 VLEN≥256 准入。
  而 **C920（唯一 RVV0.7 硬件）是 VLEN128** → RVV0.7 上 codebook mf2 本来就被 VLEN 剪掉了。
- **净结论**：在**任何 RVV0.7 target 上没有任何 fractional anchor 会被 cost model 选中**
  （要么 cost 更贵且 m1 合法，要么已被 VLEN 剪）。所以这个 fact 是
  **§4 类别 (b) 合法性收紧（防错码）**，**不产生新 perf 数，不翻转任何选择**。
  它的价值是：在一个**假想的 RVV0.7 + VLEN≥256 配置**（spec 允许的能力组合，即便当前无此硬件），
  阻止 cost model 把 codebook mf2 当合法选中并 emit 一条 C920 不存在的 `vle8_v_i8mf2`。

- **诚实降权（reachability 已查）**：`isRVV0p7` 分支当前**只在 repack 路**
  （`RVVRepackStripWidthMaterialization.cpp:112-184`）；block-dot 的
  `enumerateBlockDotShapeCandidates` 路**无 version 分支**。而**唯一的 RVV0.7 硬件 C920 是 VLEN128**
  —— 那里 `minimumVLEN=128` 已让选择落在 m1/m2 whole-LMUL anchor（fractional mf4 cost 更贵、
  codebook mf2 被 VLEN 剪），所以 block-dot 路**在真实 RVV0.7 silicon 上从不命中 fractional anchor**。
  因此这条剪枝是「**防一个真实硬件当前不会到达的 config（RVV0.7 + 假想 VLEN≥256）**」的防御性收紧，
  价值确实低（与 repack 路 `:112` 已有的 RVV0.7 fractional 防护**形式对齐**，但 block-dot 这一侧的
  缺口在真实硬件上不可触发）。这进一步坐实：**Track A 纯-wiring 范围零选择翻转，连这唯一的 legality
  收紧都是低价值防御码**。

### 2.4 ta/ma 策略（任务点名的「mask/tail」）—— 同样 legality-only，强化本文论点

任务点名要查「mask/tail」，它是 RVV0.7↔1.0 的**第 2 条**代次分歧（`RVVCapabilityProfile.h:24-28`：
RVV0.7 LACKS ratified ta/ma policy；`CapabilityProfile.cpp:478` 把它 stamp 成 `rvv_version`
property）。sealed q8 objdump 的 emit 是 `vsetvli ..., e8, m1, ta, ma` —— **ta/ma 烙进了 block-dot
emit，在 RVV0.7 非法**。

- **现状归类**：ta/ma 已经在**别处按 version fact gate**（realization-owner / legality 层的
  `requires tail agnostic, mask agnostic policy` 检查，如 `RVVStandaloneReductionSelectedBodyRealizationOwner.cpp:381/514`、
  `RVVGearboxSchedules.cpp:589`；策略字面是 `stringifyGearboxTailPolicy/MaskPolicy`
  `RVVGearboxSchedules.cpp:288/299`）。
- **对 cost model 是 NULL（legality-only，非 NULL-trap，是「已 gate 在别处」）**：block-dot 的
  **候选空间里根本没有 tail/mask 旋钮**（枚举只有 lmul/factor/elision 三轴）—— 没有「选 ta/ma 还是 tu/mu」
  这个选择，所以它**不可能翻转一个 cost-model 选择**。它和 §2 的 LMUL fact 是**同一类**：RVV0.7
  上的合法性事实，由 version fact 驱动，**强化本文「RVV0.7 的可派生 fact 全是 legality-hardening、
  非选择翻转」的论点**，不削弱它。
- **不需在 Track A 新做**：ta/ma 已 gate；列它是为答任务点名的 mask/tail，并归位为「已覆盖、非 cost-model 轴」。

### 2.3 那「真能改变选择的 perf win」在哪？—— emitter-gated，不是 cost-model-gated

这是给用户的关键诚实点：**在现有 kernel 上，没有任何「可派生-未用」fact 能翻转一个选择**
（VLEN 已经把会翻转的选择都翻完了，且已封）。真正能产生新 perf 数的「选择改变」是
**emitter-gated**：VLEN256 wide-cover（prior `winA-enrich-DESIGN.md` 候选 (a)）—— 当前 elided
cover 恒发 `vsetvl_e8m1(16)`，AVL 恒为 16，**VLEN256 与 VLEN128 产出 C 逐字节相同**（§1.4 已证
structural NULL）。要让它非-NULL 必须先改 emitter 发 `vsetvl_e8m1(32)` 的宽 cover，
那是 emitter 工作，不是 cost-model fact-wiring。**Track A 的纯-wiring 范围 = legality 收紧；
perf upside 在 emitter 后面。**

---

## 3. 单一真值源（复用，不造第二套）

- VLEN：`deriveMinimumVLEN`（`RVVCapabilityProfile.cpp:253`）—— 已是选择 + dialect verifier +
  repack 的共同 authority；§2 的 LMUL fact 同理走 `deriveSupportedLMULAllowList`/`deriveRVVVersion`
  这一条已存在的 authority（repack 路 `:112` 已在用），block-dot 路只是接同一根。
- VLMAX：`getRVVStripVLMAXElements`（`RVVGearboxSchedule.h:2544`）—— 砖#1/#2 已建，legality +
  reduction count + dialect verifier 都从它算，§2 的 fractional-anchor 判定**不引入新 VLMAX 公式**，
  只读 allowlist 是否含该 anchor 的 LMUL/EMUL 档。
- vreg budget：常量 32（ISA 不变量），保持常量，**不**派生（§1.1）。
- LMUL allowlist：`deriveSupportedLMULAllowList`（`RVVCapabilityProfile.cpp:224`）已存在，
  当前只服务 legality gate（`verifyRVVSelectedTargetCapabilityForTypedConfig`），未进 schedule 路 ——
  §2 是把这同一个 authority 多接一个消费者，零新真值源。

---

## 4. 非-NULL + 无回归判据（每个 fact 的归类）

| fact | 归类 | 是否改选择 | 无回归门 |
|---|---|---|---|
| VLEN bits | 已用 | 是（已封 q8_0/codebook flip） | 已封，本 track 不动 |
| **LMUL allowlist** | **(b) 合法性收紧（防错码，且真硬件不可触发，§2.2）** | 否（RVV0.7 无 fractional anchor 会被选中） | 见下 |
| ta/ma 策略 | 已 gate 在别处（realization-owner / legality），无 cost-model 旋钮（§2.4） | 否 | 不在 Track A 新做 |
| SEW allowlist / ELEN / vreg-count / vlenbBytes | NULL（§1.1） | 否 | 不做 |

**LMUL-allowlist 剪枝的硬无回归门**（必须在实现时守）：
- prune **gate 在 version FACT 上**（`deriveRVVVersion`/`deriveSupportedLMULAllowList`），
  **不**在 march 字符串上（I3）。
- **RVV1.0 枚举逐字节不变**：RVV1.0 allowlist 含全 fractional 档 → 新剪枝对 RVV1.0 永远 pass →
  sealed 的 **q8_0 VLEN-flip（m1-elided@256 / m2-elided@128）和 codebook mf2@256 选择必须 byte-identical**。
  这是「已封的 q8_0/FP4 选择不能被破坏」的直接落地。
- **Unknown 代次 fail-open**：`deriveSupportedLMULAllowList` 返回 `""`（无限制）→ 新剪枝 no-op
  → 与现状 byte-identical（mirror 现有 silent-gate 行为）。

---

## 5. 验证计划

> 量级与 §2.2 结论匹配：这是 legality-hardening，**主验证是「两板 byte-exact 零回归」+
> 「RVV0.7 反例不再被选中」**，不是新 perf 数（没有选择翻转 = 没有可测的 micro 胜出）。

1. **两板 byte-exact 零回归（主门）**：
   - RVV1.0 板（rvv VLEN128 + k1 VLEN256）：跑现有 q8_0 / iq4_nl / mxfp4 / q4_0..q5_1 的
     `_generic` byte-exact gate（q8-vlen-flip 那套 2400-check 复用），证明新 LMUL 剪枝对 RVV1.0
     **零选择改变、stamp 逐字节不变**（forced clean rebuild + BEFORE/AFTER-EQUALITY，按 memory
     的 build 纪律，别用绝对指纹）。
2. **RVV0.7 反例（单元 / lit，证防错码真生效）**：
   - 构造 `rv64gc_xtheadvector`（+ 假想 `_zvl256b` 让 codebook mf2 在 VLEN 维准入）的枚举调用，
     断言 `enumerateBlockDotShapeCandidates` 输出里**所有 fractional-anchor 候选 `isLegal=false`**
     （mf4 的 i8mf4/i16mf2、codebook mf2）。对照 RVV1.0 同 VLEN 该候选合法 → 证明剪枝由 version
     fact 驱动。这是纯 host 单元测试（仿 `RVVQ40Q80ShapeSelectionTest.cpp`），不需上板。
   - 与 `RVVRepackStripWidthMaterialization.cpp:112` 的 repack RVV0.7 防护对齐（两路同 authority）。
3. **选择改变处的 micro = 无**：因为 §2.2 证明无选择翻转，**不报 micro 胜出**（报了就是 NULL/虚构）。
   若未来配上 emitter wide-cover（§2.3，prior 候选 a），那才有 micro，但那是另一块砖。

---

## 6. 工作量

| 项 | 量级 | 说明 |
|---|---|---|
| LMUL-allowlist 接进 `enumerateBlockDotShapeCandidates` + descriptor 字段 + 注册处派生 | 小 | 复用 `deriveSupportedLMULAllowList`/`deriveRVVVersion`，仿 minimumVLEN 的接法 |
| RVV0.7 fractional-anchor legality 一条剪枝 + 反例单元测试 | 小 | 纯 host，仿 `RVVQ40Q80ShapeSelectionTest.cpp` |
| 两板 byte-exact 零回归 gate | 中 | 复用 q8-vlen-flip 2400-check 套件，上 ssh rvv + k1 |
| （emitter wide-cover，§2.3 的真 perf win） | **大，不在 Track A 范围** | emitter-gated，单独立砖 |

---

## 7. 推荐（先做哪个）

**最诚实的回答：Track A 的纯-cost-model-fact-wiring 范围里，没有「先做哪个能产生 Win-A 数字」的项**
——VLEN 已经把会翻转的选择全翻完且已封（q8_0/codebook）。若一定要在本范围内做一件事，是
**LMUL-allowlist → RVV0.7 fractional-anchor legality 剪枝（§2，小工作量、纯 wiring）**，但**必须诚实标注**：
它是类别 (b) 合法性收紧（防错码）、不产生新 perf 数、不翻转任何现有选择，**且在真实 RVV0.7 硬件
（C920 VLEN128）上不可触发**（§2.2 reachability），价值低；它只是把 block-dot 路对齐 repack 路
（`RVVRepackStripWidthMaterialization.cpp:112`）已做的 RVV0.7 fractional 防护，**形式补缺而非 Win-A**。

**不做（明确弃）**：派生 vreg-budget（ISA 常量 32）、SEW/ELEN allowlist（block-dot 全 SEW8/16/32 通用）、
喂实测 `vlenbBytes`（放松 fail-closed legality 的 anti-pattern）——全是 §1.1 的 NULL trap。

**真 perf upside 在 Track 之外**：VLEN256 emitter wide-cover（§2.3）是唯一能在现有 kernel 上产生
新「选择改变 + 实测胜出」的方向，但它 **emitter-gated 不是 cost-model-gated**，应作单独的 emitter 砖，
不应混进「cost model fact-wiring」的 Track A 范围谎称成熟。
