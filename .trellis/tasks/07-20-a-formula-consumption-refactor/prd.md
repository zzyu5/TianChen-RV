# A 线：公式层真实消费与模块化重构

## Goal

以现有五类 dequant plan 和已存在的 RVV 解析函数为起点，做一次渐进但真实的公式层重构：把散落在 front door、selector、verifier 和 emitter 中的 g/c/ω 决策收拢到小型 typed decision contract，形成 formula 构造候选/参数、legality 限定合法域、selector 选择、typed stamping、emitter 机械消费的单向链。重构必须改变 authority 结构并被真实 consumer 使用，不能只新增命名、ledger 或 mirror。

## Current Assets

- `RVVGearboxSchedule.h` 已包含 register-pressure、gather-anchor 等闭式规则，并声明五类 dequant FormulaProvider。
- `NibbleDecodePlan`、`CodebookGatherPlan`、`KQuantScaleMinPlan`、`GridLookupPlan`、`TernaryDecodePlan` 五类 typed plan 已存在。
- dequant-row head 已达到 plan 5/5，并有 byte-exact 与 decisive lit 资产。
- VLEN、RVV version、vreg_count 已有真实 capability 消费；repack LMUL measured table 已有少量有效行。
- typed body、route provider、reason、miss→prior/fallback 等骨架已存在。
- A2 已落 `RVVFormulaDecision.h`：Nibble 与 repack accumulator-LMUL 两个
  mechanism-specific typed decision slice 已被 production consumer 使用。
- A3 已落 Codebook 的真实 g+c 决策：canonical layout row、selected RVV capability
  collector、`{mf2,m1,m2}` 有界合法集、complete selected-plan stamp、唯一 backend
  materializer 与 direct/registry/artifact 共享 preparation；三条真实 emitted LMUL
  chain 有决定性测试，但尚未据此宣称真板性能赢家。

## Actual Gaps

- “五类 plan 已存在”不等于全项目公式层已闭环：A2/A3 已形成最小 contract 与
  一个 g+c 完整切片，但尚未推广到 KQuant/Grid/Ternary、SP4/loop 与第二 family。
- 部分 plan 仍是 reproduce-current，真实 c-driven θ 消费不足；不得为追求覆盖制造假 capability 旋钮。
- `GridDecodePlan` 仍被 dequant、block-dot/verifier 与 emitter 多头直接查询，存在 authority 双头。
- repack accumulator-LMUL 的 analytic legality、measured winner 与 consumer 边界已
  在 A2 收口；手工 measured registration 仍住 front door，待 A5 替换为 qualified view。
- Codebook 已形成可机检 complete stamp；SP4/loop-order 仍有 unrealizable `Plain`、
  `ABSENT→S6Tiled`、`reason==measured` override 与 q4_K missing-stamp recompute。
- IME/第二 family 尚未用同一最小公式/选择 contract 完成垂直切片。

## Requirements

- 按子任务逐步完成，不一次重写 RVV。
- 先冻结现状和 owner，再改接口；每刀有决定性正负测试。
- g、c、ω typed 且 owner 分离，只在 plugin-local decision 边界汇合。
- formula 构造 candidate/typed plan，legality 先于 selector。
- qualified measurement 只在合法候选中提供 winner；miss/stale/invalid 回 analytic prior。
- selected result 在 emission 前显式 stamping；emitter 不重算、不查 winner 表。
- verifier 独立验证公式结果但不成为第二构造 authority。
- 选一个第二 family 垂直切片证明 contract 非 RVV 专用。
- 每个 vertical slice 自己完成 caller cutover 与旧入口删除；A4 是跨 slice authority 复核，不是替 A2/A3/A5/A6 延后清垃圾。
- 多 agent 并行以“共同完成同一个 declared slice”为组织单位：实现/authority、
  adversarial/parity 与 root integration 三个责任面相互复核，不把强依赖任务各自
  丢给一个孤立 agent。共享接口和合入只有一个 owner。

## Acceptance Criteria

- [ ] authority inventory 覆盖五类 dequant plan、repack LMUL、SP4、loop-order 和相关 capability。
- [x] 一个小型 C++ typed decision contract 被至少两个 RVV decision 真消费。
- [x] Codebook plan 同时对 typed g 与 c 有决定性 construction/legal/emission 测试；
      未承重维度保持 honest-null。
- [ ] legality、selection、stamping、emission 的单向链有 mutation/negative test。
- [ ] GridDecodePlan/DequantMechanismPlan 的双 authority 有明确关闭路径并完成代表性切片。
- [ ] measurement winner view 有 hit/miss/stale/illegal-winner 测试，且不能扩大候选集。
- [ ] IME 或另一第二 family 有一个垂直切片复用相同最小 contract。
- [ ] 每个已迁移 slice 的旧 helper/overload/caller、compat mirror 和 emitter/selector 旁路为 0，符合父任务 Shared Retirement Gate。
- [ ] 全程 byte-exact/ULP、lit 与必要 deployed regression 无新增失败。

## Child Modules

1. [A1 formula authority freeze](../archive/2026-07/07-20-a1-formula-authority-freeze/prd.md)（已归档）：HEAD owner matrix + characterization/negative tests。
2. [A2 typed decision contract](../archive/2026-07/07-20-a2-typed-decision-contract/prd.md)（已归档）：小型 C++ contract + 两个真实 RVV consumer。
3. [A3 dequant c-driven plans](../archive/2026-07/07-20-a3-dequant-c-driven-plans/prd.md)（已归档）：Codebook 首个真实 g+c complete-stamp 切片。
4. [A4a schedule authority cutover](../07-20-a4-single-authority-stamping/prd.md)：只处理 ISSUE-125 的 SP4/loop-order selected→realized。
5. [A4b dequant plan stamping](../07-21-a4b-dequant-plan-stamping/prd.md)：只处理 ISSUE-122 剩余 KQuant/Grid/Ternary 与 Grid 双头。
6. [A5 qualified winner view](../07-20-a5-qualified-winner-view/prd.md)：版本化 winner token + thin selector。
7. [A6 IME decision slice](../07-20-a6-ime-decision-slice/prd.md)：第二 family 垂直切片。
8. [A7 superseded path retirement](../archive/2026-07/07-20-a7-superseded-path-retirement/prd.md)（已归档）：q1_0 与 K-quant 已替代 monolith 原子退役。
9. [A8 baked-g convergence](../07-20-a8-baked-g-convergence/prd.md)：ISSUE-118/119 发射器格式事实全量收敛。

当前里程碑：A1、A2、A3、A7 已完成并归档。下一原子切片是 A4a；A4b、A5、
A6、A8 仍待推进。A4a/A4b 的拆分是按两个独立 declared decisions 分责，不是把
一个算子族拆成微缝。

每个子任务单独 PRD、touch set 和 commit；父任务不直接实现代码。

## Dependencies and Parallelism

- authority inventory 最先完成。
- 最小 contract 依赖 inventory。
- plan 真实消费与 selector/winner-view 可以在 contract 稳定后按不相交文件并行。
- A4a 先收口 schedule authority；A4b 在共享 materializer API 无冲突后迁剩余
  dequant plan。两者各自以多个 agent 共同实现、独立反例审查、root 集成。
- 第二 family 垂直切片依赖 common contract 和单向 authority。
- A7 可在 RET-1 基线后独立开工，但与 K-quant 性能 emitter 修改串行；A8 的 census 可并行，生产迁移依赖 A1/A2。
- B 线的 bench 基建、独立 hot-kernel 攻坚可并行；B 线产生的新 qualified winner 在 A 线数据面接口稳定后接入。

## Out of Scope

- Formula dialect、通用表达式 DSL、giant optional descriptor。
- 在线 autotuning 或 runtime sparse/MoE policy。
- 一次迁移全部 extension family 和全部格式。
- 为增加公式计数而制造不承重 g/c/θ。
- 在 A 线任务中承诺性能翻正；新硬件数字属于 B 线。

## Technical Notes

- Spec authority: `.trellis/spec/architecture/变体流水线.md`
- Current ledger: `experiments/active/formula-layer-migration/LEDGER.md`
- Formula homes: `include/Weft/Plugin/RVV/RVVFormulaDecision.h`（A2 typed contract）
  与 `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`（既有 closed forms）
- Remaining dequant providers: KQuant/Grid/Ternary in `lib/Conversion/RVV/` and
  `include/Weft/Support/GridDecodePlan.h`; Codebook 已迁入 unique materializer。
- Scattered selection: `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`
- Double-head registry: `include/Weft/Support/GridDecodePlan.h`
- Related issues: ISSUE-117、ISSUE-118、ISSUE-119、ISSUE-121、ISSUE-122、ISSUE-125。
