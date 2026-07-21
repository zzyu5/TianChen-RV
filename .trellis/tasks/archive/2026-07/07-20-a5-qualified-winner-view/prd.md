# A5：Qualified winner view 与薄 selector

## Goal

把 offline measurement 数据平面生成一个版本化、编译期可消费的轻量 winner-token view，使 selector 只在 analytic formula 已构造且 legality 已确认的有限候选中采用合格 winner；miss、stale、schema mismatch 或 winner infeasible 时回 analytic prior/fallback。

## Scope

- 定义 typed measurement key：至少包含 op/format/engine/regime、必要 shape/context bucket、capability instance/version 与 candidate-axis identity。
- 从正式 master/run lineage 生成或 hash-equivalent 构建 compiled winner view。
- qualification 包括 correctness、lineage、freshness、selection-valid、candidate identity 和 winner-in-legal-set。
- selector 输出 provenance：analytic、measured、fallback/reject。
- 运行时不得读取 `experiments/`；编译器只消费版本化生成物。
- schema、source registry、compiled view 只能有一个事实 authority，其他为生成物。
- B1 独占 measurement artifact、writer/read contract、regime、qualification 与 freshness；A5 只消费其产出的 qualified view 并拥有 selector 语义，不另建 master parser/cache。

## Primary Touch Set

- measurement schema/生成脚本或现有 registry generator
- RVV selection adapter/front door
- versioned generated view/fixture
- selector tests
- 不直接改原始 run 数据

## Dependencies

- 依赖 A2 decision contract 与 A4 legality/stamping。
- 依赖 B1 明确 master ownership、qualification 和 freshness 语义。
- 可先并行设计 schema/fixtures，正式接线等待上述接口。

## Acceptance Criteria

- [ ] qualified hit 选择 winner token 且 winner 仍在合法集。
- [ ] miss、stale、unqualified、key mismatch、schema mismatch、illegal winner 全部回 prior/fallback。
- [ ] measurement 无法创建 candidate、route、dtype、mechanism 或 legality。
- [ ] compiled view 可由正式数据确定性重建并有 hash/version 校验。
- [ ] compiled view 接管后，旧手工 winner 表、平行 reader、runtime experiments lookup 与 old-schema production reader 为 0；旧数据只作离线归档/迁移输入。
- [ ] selector/emit 路径不访问 CSV/JSON/experiments filesystem。
- [ ] reason/provenance 能区分 measured/prior/fallback，但不反向定义 compute。
- [ ] 不使用 `MemoArgmin`；无完整 cost vector 时语义明确为 qualified winner lookup。

## Verification

- hermetic selector unit/lit for all branches;
- generator reproducibility and stale/hash negative tests;
- candidate-set mutation test;
- focused build + A4 no-redecision tests.

## Out of Scope

- 新板测、在线 tuning、完整代价矩阵、运行时 profile、扩大候选域。

## Issue Mapping

- 与 ISSUE-098、ISSUE-117 交叉；B1 关闭数据 ownership，A5 只负责编译期消费视图。
