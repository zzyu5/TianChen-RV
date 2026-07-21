# A4a：SP4 / loop-order selected→realized 原子切换

## Goal

只处理 `ISSUE-125` 的一个 declared schedule slice，把 SP4 tiling 与 sibling
loop-order 从当前 `selected != realized` 收口为不可绕过的单向链：

~~~text
typed schedule facts/capability/context
  → formula/provider constructs bounded candidates
  → legality keeps only realizable candidates
  → selector chooses one legal value
  → complete required bounded schedule stamp
  → pre-emission verification/materialization
  → mechanical realization/emission
~~~

本任务不再夹带 KQuant/Grid/Ternary plan 迁移；后者由 A4b 承担。A3 Codebook
是 complete stamp、唯一 materializer 与路由 parity 的参考实现及回归保护，不是
第二轮重写对象。

## Current Counterexamples

- `tilingVariantFeasibleSet` 可产生 `{Plain,S6Tiled}`，但 min-fold emitter 没有
  `Plain` real body；合法域大于可构造域。
- `weft_rvv.tiling_variant` 缺失时 emission 静默按 `S6Tiled` 实现。
- sibling loop-order 仅在 `reason=measured` 时兑现 `col_outer`；同一个 selected
  value 若 reason=prior 会被改回 `row_outer`。
- q4_K 的 schedule stamp 缺失时，emitter 会调用
  `repackColGroupOuterForLayout` 重算 prior。

这些都是当前代码事实，不是兼容行为。A4a 合入时必须一并退役。

## Scope

- 为本 declared slice 明确真实、有限且均可 realization 的 candidate set；若不在本
  任务实现 `Plain` real body，就从 SP4 legal set 删除 `Plain`。
- 让 SP4/loop-order 的 code-affecting selected fields 成为 emission 前 required、
  bounded、all-or-none 的 complete stamp；共享 reader 将字符串属性解析为 C++ enum
  plan 并做语义验证。这里不虚构已新增 ODS typed schedule attribute；reason/provenance
  仅作 mirror。
- 复用现有 canonical capability/selection/preparation 基础设施；不得在 schedule
  emitter 内建立第二 capability parser、winner reader 或 stamp writer。
- pre-emission 层拒绝 missing、partial、错类型、unknown、forged、结构性 stale 或与
  当前 typed inputs 不一致的 stamp；不得修正成另一个值。`measured` 的 ledger lineage、
  freshness 与 qualification 明确属于 A5，不在 A4a 伪装解决。
- emitter 对所有合法 selected values 机械 realization，不再看 reason、format、
  board/march 或 measurement 决定是否兑现。
- direct wrapper、registry clone 与标准 target-artifact export route 必须经过同一
  preparation 入口；不得新增 benchmark-only bypass。真硬件 deployed runtime 是后续
  correctness/performance campaign，不由本地 route 测试代替。
- 更新或删除固化旧错误行为的 fixture，并把它们变成 killing tests。

## Explicit Retirement Set

- min-fold unrealizable `Plain` 假 feasibility（除非本任务同时交付 real body）；
- `ABSENT → S6Tiled`；
- sibling `reason == measured` override gate 与 override comment；
- q4_K missing-stamp `repackColGroupOuterForLayout` 重算；
- optional string-only schedule attr 与任何 silent/default/compat reader；
- `static_order` code-affecting compatibility reason；空合法集不得制造选择；
- 同一 schedule decision 的第二 selector、第二 writer、shadow route 与旧 overload。

不保留 legacy/deprecated/compat alias，也不把旧 fixture 数量当保留理由。无法迁完
本 declared slice 的全部 production caller 时，保持 worktree 未完成或整笔回滚。

## Multi-Agent Collaboration Contract

这是一个由多个 agent 共同完成的任务，不是“一人一个子任务”：

1. implementation/authority agent 负责 formula、legality、complete stamp 与 emitter
   退役集；
2. adversarial/parity agent 独立设计反例、追踪 direct/registry/target-artifact
   路由，并审查实现 agent 的 diff；
3. root/integration owner 负责共享 worktree、touch-set 冲突裁决、强制重链、全量
   验证、spec/issue/ledger 同步与唯一合入。

agent 可按不相交文件并行，但共享接口只有一个 integration owner。任何 agent 不能
以自己的局部测试替代另一责任面的独立审查。

## Primary Touch Set

- schedule formula/selection：`include/Weft/Plugin/RVV/`、
  `lib/Plugin/RVV/FrontDoor/` 中与 SP4/loop-order 直接相关的 symbols；
- selected body/stamp verifier/materializer；
- `lib/Conversion/RVV/` 中 min-fold、sibling loop-order 与 q4_K 的实际 consumer；
- direct/registry/target-artifact parity 与 focused lit/C++ tests；
- ISSUE-125、formula migration ledger 与 authority matrix。

禁止顺手迁移其它 dequant plan、改 measurement winner、建立 Formula IR 或改真板
性能结论。

## Dependencies

- A2 typed decision contract 与 A3 Codebook complete-stamp/preparation 先例已落地。
- A4a 不以 A5 winner-view 为前置；当前 winner 只要仍在合法集即可，A4a 不改判。
- A4b 可在 A4a 的共享 stamp 形态稳定后并行或随后施工，但不能与 A4a 同写共享
  materializer API。

## Acceptance Criteria

- [x] SP4/loop-order 的每个 legal candidate 都有真实 body；selector 不可能返回
      emitter 才拒绝的值。
- [x] 合法 selected value 的 realization 与 reason 无关；`col_outer/prior` 真正产生
      col-outer 实现。
- [x] 删除 SP4 或 loop-order stamp 会在 emission 前 fail closed；不落回默认/重算。
- [x] partial、错类型、unknown、forged、结构性 stale/inconsistent stamp 各有独立负例；
      measurement lineage/freshness 明确留给 A5。
- [x] `ABSENT→S6Tiled`、`reason==measured` override、q4_K missing-stamp recompute
      与 unrealizable-Plain 假候选在 production tree 中为零。
- [x] direct、registry、target-artifact 路由共享 preparation；公共 hook 提供架构覆盖，
      q4_0 full-pipeline object export 给出代表性结构证据。不把它外推为全格式逐项验证，
      也不把本地 artifact export 冒充板上执行。
- [x] measurement 只能在 legal candidates 内选择；不能创造 candidate、绕过
      legality 或借 reason 改 compute。
- [x] A3 Codebook 的 formula、capability collector、unique materializer 与 parity
      tests 无回归，且本任务没有复制它们。
- [x] 合法现役 fixtures 的 compute 语义与 inner body 保持；schedule-sensitive goldens
      按真实 selected order 有意更新；无新增 RVV suite 失败。
- [x] authority census 证明已迁移 slice 的旧 caller、compat bridge、第二 selector/
      writer/dispatcher 和 code-affecting silent default 均为零。

## Verification

- focused formula/selector/verifier/emission lit 与必要 C++ unit；
- direct/registry/target-artifact parity；
- mutation-style killing tests 覆盖 missing/forged/structural-stale/reason
  override/illegal candidate；
- `formula-authority-matrix.test` 与无截断 `rg` census；
- 修改 Conversion/Target 后删除工具并强制重链；若改共享 header/struct layout，执行
  clean build，再跑相关 RVV/EmitC suite 与全量 baseline；
- runtime/performance 没有在本任务主张，故本地测试不冒充硬件证据。

## Out of Scope

- ISSUE-122 的 KQuant/Grid/Ternary plan stamp（A4b）；
- A5 qualified winner view；
- 改变当前性能 winner 或写 measurement/master；
- 实现新的 `Plain` 候选（除非审计证明删除会破坏合法 production 语义并在同一
  原子切换内交付完整 real body）；
- Formula dialect、通用 schedule IR、在线 tuning、runtime sparse/MoE。

## Authority Boundary

- code-affecting authority 只来自 bounded `tiling_variant` / `loop_order` plan；reason
  经一致性验证，但永不改变 realization。
- `*_selection_record` 是非权威 diagnostic mirror。emitter 与 legality 均不读取它；
  A5 负责从 qualified ledger view 重建或校验其 lineage/freshness。在 A5 完成前，record
  不能独立作为性能或 provenance 证据。
- 标准 direct pass、registry clone 与 target-artifact export 都调用共享 backend
  conversion harness，进而调用同一个 RVV `prepareForConversion`；emitter 再逐 op 调用
  同一 validated-plan reader，防止 preflight 后突变绕过。

## Outcome（2026-07-21）

- selector 现在以 `std::optional` 表达空合法集；frontdoor 在无可实现 schedule 或缺失
  target resource facts 时 fail closed，不制造 compatibility choice。
- SP4 的合法集与真实 body 对齐：min-fold=`S6Tiled`，dual-plane/already-lean=`Plain`，
  均为 `only_feasible`；`RVVMeasurementAxis::SP4Tiling` 与 `static_order` 兼容状态退役。
- loop-order 保留两个真实 body；q4_K 与 siblings 只消费一次解析后的
  `selectedColGroupOuter`。旧 reason gate、q4_K stride 重算与 prior override 为零。
- 新增共享 `RVVRepackSchedulePlan` reader/verifier；它把 string attrs 解析为 bounded
  C++ enums 并验证 shape、reason 与 layout prior。它不是 ODS typed attribute，也不承担
  A5 的 measurement lineage/freshness 资格判断。
- flat q4_0/q4_1/q5_1 等 builder 用 loop body 的真实 `fold_model` 构造 g，消除了前门
  以硬编码 `lane_wise_vector_scale` 覆盖 q4_1/q5_1 真实 fold identity 的 authority split。
- 37 份 direct typed-body fixture 已迁到 complete stamp；新增 selected→realized 与
  fail-closed killing tests；q4_0 target-object 代表测试证明标准 artifact route 进入同一
  preparation/consumer 链。没有据此声称 deployed runtime、真机性能或 artifact replay。
- 本任务采用多人同题交叉审查：两个独立 adversarial reviewer 在最终实现上分别检查
  authority 绕过与证据过度外推，结论均为 GO，且共同保留 A5 边界。

### Reproducible verification

- clean default build：304/304 targets 成功；最后的文案清理后完成依赖重链。
- focused schedule/formula/target tests：4/4；受 flat-fold 修正影响的 frontdoor tests：
  7/7；`Conversion/RVV`：305/305。
- `formula-authority-matrix` checker 与 self-test：通过；issue census：125 条、零缺号、
  零重号；`git diff --check`：通过。
- 正式全仓入口 `cmake --build build/weft --target check-weft -j2`：980/983 通过。
  仅余三项既存 `Scripts/rvv-generated-bundle-abi-e2e-*` 失败，与历史 baseline 完全同名；
  本任务未修改其 Python/target domain，且无新增失败。

## Issue Mapping

- 主项：ISSUE-125；
- 关联：ISSUE-034（早期 loop-order fixture 爆炸面）；
- 同类但明确分拆：ISSUE-122 → A4b。
