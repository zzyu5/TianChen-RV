# A 线：公式层真实消费与模块化重构

## Goal

以现有五类 dequant plan 和已存在的 RVV 解析函数为起点，做一次渐进但真实的公式层重构：把散落在 front door、selector、verifier 和 emitter 中的 g/c/ω 决策收拢到小型 typed decision contract，形成 formula 构造候选/参数、legality 限定合法域、selector 选择、typed stamping、emitter 机械消费的单向链。重构必须改变 authority 结构并被真实 consumer 使用，不能只新增命名、ledger 或 mirror。

## Current Assets

- `RVVGearboxSchedule.h` 已包含 register-pressure、gather-anchor 等闭式规则，并声明五类 dequant FormulaProvider。
- `NibbleDecodePlan`、`CodebookGatherPlan`、`KQuantScaleMinPlan`、`GridLookupPlan`、`TernaryDecodePlan` 五类 typed plan 已存在。
- dequant-row head 已达到 plan 5/5，并有 byte-exact 与 decisive lit 资产。
- VLEN、RVV version、vreg_count 已有真实 capability 消费；repack LMUL measured table 已有少量有效行。
- typed body、route provider、reason、miss→prior/fallback 等骨架已存在。

## Actual Gaps

- “五类 plan 已存在”不等于跨项目公式层已建立：provider/plan 类型分散，最小共同 decision contract 尚未形成。
- 部分 plan 仍是 reproduce-current，真实 c-driven θ 消费不足；不得为追求覆盖制造假 capability 旋钮。
- `GridDecodePlan` 仍被 dequant、block-dot/verifier 与 emitter 多头直接查询，存在 authority 双头。
- `selectRepackAccumulatorLMUL` 与 measured registration 仍住 front door；analytic legality、measured winner 与 consumer 的边界未统一。
- selected decision 的 stamping 与 emitter no-redecision 尚未形成可机检的全链不变量。
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

## Acceptance Criteria

- [ ] authority inventory 覆盖五类 dequant plan、repack LMUL、SP4、loop-order 和相关 capability。
- [ ] 一个小型 C++ typed decision contract 被至少两个 RVV decision 真消费。
- [ ] 至少一个 plan 参数同时对 g 与 c 具有决定性测试，或被诚实标为 board-invariant/honest-null。
- [ ] legality、selection、stamping、emission 的单向链有 mutation/negative test。
- [ ] GridDecodePlan/DequantMechanismPlan 的双 authority 有明确关闭路径并完成代表性切片。
- [ ] measurement winner view 有 hit/miss/stale/illegal-winner 测试，且不能扩大候选集。
- [ ] IME 或另一第二 family 有一个垂直切片复用相同最小 contract。
- [ ] 每个已迁移 slice 的旧 helper/overload/caller、compat mirror 和 emitter/selector 旁路为 0，符合父任务 Shared Retirement Gate。
- [ ] 全程 byte-exact/ULP、lit 与必要 deployed regression 无新增失败。

## Child Modules

1. [A1 formula authority freeze](../07-20-a1-formula-authority-freeze/prd.md)：HEAD owner matrix + characterization/negative tests。
2. [A2 typed decision contract](../07-20-a2-typed-decision-contract/prd.md)：小型 C++ contract + 两个真实 RVV consumer。
3. [A3 dequant c-driven plans](../07-20-a3-dequant-c-driven-plans/prd.md)：codebook/grid 首个真实 g+c 切片。
4. [A4 single authority stamping](../07-20-a4-single-authority-stamping/prd.md)：provider/verifier、selector/emitter 收口。
5. [A5 qualified winner view](../07-20-a5-qualified-winner-view/prd.md)：版本化 winner token + thin selector。
6. [A6 IME decision slice](../07-20-a6-ime-decision-slice/prd.md)：第二 family 垂直切片。
7. [A7 superseded path retirement](../07-20-a7-superseded-path-retirement/prd.md)：q1_0 与 K-quant 已替代 monolith 原子退役。
8. [A8 baked-g convergence](../07-20-a8-baked-g-convergence/prd.md)：ISSUE-118/119 发射器格式事实全量收敛。

每个子任务单独 PRD、touch set 和 commit；父任务不直接实现代码。

## Dependencies and Parallelism

- authority inventory 最先完成。
- 最小 contract 依赖 inventory。
- plan 真实消费与 selector/winner-view 可以在 contract 稳定后按不相交文件并行。
- stamping/emitter authority 收口依赖 plan 与 selector 接口稳定。
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
- Formula home: `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`
- Providers: `lib/Conversion/RVV/RVVToEmitCSupport.cpp`
- Scattered selection: `lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp`
- Double-head registry: `include/Weft/Support/GridDecodePlan.h`
- Related issues: ISSUE-117、ISSUE-118、ISSUE-119、ISSUE-121、ISSUE-122。
