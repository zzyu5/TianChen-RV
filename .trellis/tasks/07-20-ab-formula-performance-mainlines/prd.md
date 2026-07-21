# A/B 双主线：公式层重构与真实性能推进

## Goal

把下一阶段项目推进组织成两条长期但可分批完成的主线：A 线将散落的 g/c/ω、公式、合法性、选择与 emission 决策收拢成真实被代码消费的最小公式层；B 线沿正式 bench、部署 ggml、代表性强对手与 e2e 三类证据持续推进真实性能。两线共享 typed capability、正确性门、measurement lineage 和负边界，但各自以小任务施工，不用一个巨型任务一步推到论文最终状态。

## What I already know

- 用户已明确将后续工程命名为 A 线与 B 线，并要求建立多个 Trellis task、使用多 agent 协作。
- A 线必须包含真实重构，不接受仅用 ledger 或散落 provider 宣称“已有公式层”。
- B 线必须继续真实性能推进；deployed ggml、代表性强对手和 e2e 都是合法且互补的证据。
- 当前两柱不变：P1 可复用扩展模板，P2 高性能知识的可执行利用。
- 当前公式/选择正本是 `.trellis/spec/architecture/变体流水线.md`；迁移资产在 `experiments/active/formula-layer-migration/LEDGER.md`。
- 正式测量入口、master、runs、runs.log 和多个 cell harness 已存在；unsupported 组合 fail-closed。

## Assumptions (temporary)

- A/B 是并列主线；A 线不会被性能短期数字牵着重写，B 线也不会绕开 A 线的合法性和 typed authority。
- 每个子任务应有单一主要责任、清晰 touch set、可独立验收和提交。
- 先建设可复用的最小 contract 与代表性实例，不做大一统 Formula dialect/DSL，也不全仓一次重写。
- 性能任务优先处理已有真实缺口、正式 runner coverage、强对手近门/具名-X 与 e2e 传导，不重新发明一套性能账。
- “渐进”只表示不同 slice 可分批施工；同一 slice 合入时必须原子 cutover，禁止新旧 production path、compat bridge 或第二 writer/dispatcher 并存。

## Open Questions

- 无阻塞问题；先依据代码、ledger、master 与 issues 完成任务分解。需要重大接口裁决时在对应子任务 PRD 中单独提出。

## Requirements (evolving)

- 建立 A 线父任务及若干模块化子任务。
- 建立 B 线父任务及若干模块化子任务。
- 每个子任务写清代码事实、目标、非目标、验收、依赖、主要 touch set 和与 ISSUE 的对应关系。
- 标出可并行项与必须串行的 authority/schema/measurement 依赖。
- 任务只描述当前可交付阶段，不承诺一步完成全部论文实验或所有格式覆盖。

## Shared Retirement Gate

所有 child task 共同遵守 [architecture · 退役与原子合入](../../spec/architecture/退役与原子合入.md) [RET-1]：

- 已迁移 slice 只有一个 decision、dispatch/route 和 measurement authority；
- 旧 production caller、旧 overload/alias、compat/legacy/deprecated bridge、dual-read/dual-write 与 shadow path 为 0；
- code-affecting missing 字段或 stamp fail-closed，不以 `value_or`、march/format/board 名或旧实现恢复；
- 只保留正式合法的语义 fallback/reject，不保留“新路没接完”的兼容 fallback；
- 无法迁移全部 declared caller 时，task 保持未完成或整笔 Git 回滚，不得把中间态合入；
- 失败实验留 run/ledger，已证伪且无合法 cell 的 production strategy、开关和 dormant branch 删除；
- 代码、测试/golden、spec/issue/ledger 与 retired index 在同一 cutover 同步。

## Technical Approach and Task Tree

~~~text
ab-formula-performance-mainlines
├── a-formula-consumption-refactor
│   ├── a1-formula-authority-freeze
│   ├── a2-typed-decision-contract
│   ├── a3-dequant-c-driven-plans
│   ├── a4-single-authority-stamping
│   ├── a5-qualified-winner-view
│   ├── a6-ime-decision-slice
│   ├── a7-superseded-path-retirement
│   └── a8-baked-g-convergence
└── b-real-performance-progress
    ├── b1-measurement-control-plane
    ├── b2-bench-cell-coverage
    ├── b3-kquant-exhausted-strategy-retirement
    ├── b4-dequant-grid-codebook-attack
    ├── b5-gemm-deployed-path
    └── b6-e2e-transduction-regression
~~~

主依赖：

~~~text
A1 → A2 → A3 → A4 ─┬→ A5
 │     └────→ A8    └→ A6
 └──────────→ A7

B1 → B2 ─┬→ B3(retirement)
         ├→ B4 ─┐
         └→ B5 ─┴→ B6

B1 → A5
A3/A4 → B6 paired regression
A6 → B5/B6 的 IME 证据入口
~~~

## Decision (ADR-lite)

**Context**：现有代码已经有五类 dequant plan 和大量性能资产，但 authority 与 measurement 仍分散；若继续以零散 task 追单点，会不断出现“接口存在但没有真实消费”和“性能已做却被重新列为缺失”。

**Decision**：采用 A/B 两条并列父线；A 线八个模块做公式、退役与第二 family 重构，B 线六个模块做正式性能推进。两线只通过 typed decision、qualified measurement 和 paired regression 交叉。父任务不作为巨型实现 owner。

**Consequences**：允许多个 agent/worktree 按不相交 touch set 并行；共享 contract、runner/master writer 和 stamping/emitter 边界必须串行收口。任务数量增加，但每项可独立验收、提交与回滚。

## Recommended Multi-Agent Waves

1. **Wave 1**：A1、B1 与 A7 分独立 worktree；A8 只做 HEAD census/classification，不改共享 contract。
2. **Wave 2**：A2 与 B2 施工；B3/B4 只做瓶颈和对手只读分析。
3. **Wave 3**：A3 与 B3 按不相交的 codebook-dequant / K-quant-retirement 文件在独立 worktree 并行；B4/B5 按不同 emitter/harness 分片施工，真板由单 owner 串行。A5 只准备 B1 view 的 selector adapter，不另建 reader。
4. **Wave 4**：A4 统一收口共享 authority；随后 A5/A6 与已稳定的 B5 按 touch set 并行。
5. **Wave 5**：B6 汇总 A 线前后 paired regression 与 e2e 传导。

## Acceptance Criteria (evolving)

- [ ] A、B 两条父任务已建立并能独立追踪。
- [ ] 每条线至少拆出三个有实际代码/实验落点的子任务。
- [ ] 子任务没有重复 owner，也不把相同 authority 分散到多个并行写者。
- [ ] A 线覆盖真实 formula consumption、legality/selection/stamping/emission authority 与第二 family 验证。
- [ ] B 线覆盖 runner/cell 合格化、代表性强对手攻坚、deployed/e2e 回归与证据入账。
- [ ] 当前已完成资产不会被重新列为“尚未做”。
- [ ] 每个完成 slice 均有 cutover record，旧 production caller 与兼容路径为 0。
- [ ] worktree 中间态未作为主线完成态合入，rollback 只依赖版本控制。
- [ ] task tree 与 issue/spec 索引一致，且工作树经检查后提交。
- [ ] 17 个 task 的 context JSONL 均通过 `task.py validate`。

## Definition of Done

- 父子 task 树可由 `task.py` 查询。
- 所有 PRD 自足，可由后续 agent 直接开工。
- 依赖和并行边界明确。
- 文档链接有效，task 元数据通过项目检查。
- 本次只创建和设计任务，不实施 A/B 线代码或板测。

## Out of Scope

- 一次性完成公式层全仓迁移。
- 本次直接修改 compiler/runtime 实现。
- 本次直接占用真板跑新性能数字。
- 建 Formula dialect、通用表达式 DSL 或在线 autotuner。
- 把 runtime sparse/MoE future work提前并入当前主线。

## Technical Notes

- Canon: `.trellis/spec/canon/暂定-科研主张.md`
- Architecture: `.trellis/spec/architecture/变体流水线.md`
- Formula ledger: `experiments/active/formula-layer-migration/LEDGER.md`
- Measurement: `.trellis/spec/measurement/index.md`
- Issues: `.trellis/spec/issues/index.md`
- Official runner: `tools/bench/bench`
- Parent task is an organizing container; child tasks own implementation and experiments.
