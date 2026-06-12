# Thinking Guides

这些 guides 是开发和审查前的检查清单，不替代各层 specs。

## Available Guides

| Guide | Purpose |
|---|---|
| [Trunk Discipline](./trunk-discipline.md) | 怎么选下一步该做什么——对齐主干，抵抗挑小问题的引力 |
| [Capability-First Design Guide](./capability-first-design-guide.md) | 检查设计是否真的由 capability 驱动 |
| [Plugin Locality Review Guide](./plugin-locality-review-guide.md) | 检查新增扩展是否局部封装 |
| [Compute Boundary Review Guide](./compute-boundary-review-guide.md) | 检查是否误把 TianChen-RV 写成高层 compute IR |
| [Code Reuse Thinking Guide](./code-reuse-thinking-guide.md) | 写新代码前先查：是不是已经有了（本项目曾有大量重复模板）|
| [Cross-Layer Thinking Guide](./cross-layer-thinking-guide.md) | 实现前先想清楚跨层数据流 |
| [Dead-Mirror Removal Guide](./dead-mirror-removal-guide.md) | 去伪删死码前：分清活发射器 vs 死 metadata-mirror 脚手架，closed-graph+full-link 安全删 |

## Quick Routing

- **选下一个 task / 怀疑自己在做枝节：先读 trunk-discipline。**
- 修改 target、profile、variant legality、dispatch、emission path：读 capability-first guide。
- 新增 RVV/IME/offload/future extension：读 plugin-locality guide。
- 新增 op/dialect/pass 表达：读 compute-boundary guide。
- 准备新增一个 contract/owner/route：先读 code-reuse guide（确认不是又一份重复模板）。
- 删除 RVV 字符串机器 / route-validation / metadata-mirror 脚手架（Stage 3 去伪）：读 dead-mirror-removal guide。
- 修改 RVV route、lowering、artifact、测试或 performance layer：读 RVV plugin spec、variant pipeline、EmitC route、testing contract，并确认 typed body authority、selected-body realization、no legacy i32 positive route、no source-front-door current route、no status/artifact authority。

## Rule

如果 guide 与具体 spec 冲突，以具体 spec 为准；如果具体 spec 与 source design docs 冲突，先报告冲突。
