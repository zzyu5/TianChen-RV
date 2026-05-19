# Thinking Guides

这些 guides 是开发和审查前的检查清单，不替代各层 specs。

## Available Guides

| Guide | Purpose |
|---|---|
| [Capability-First Design Guide](./capability-first-design-guide.md) | 检查设计是否真的由 capability 驱动 |
| [Plugin Locality Review Guide](./plugin-locality-review-guide.md) | 检查新增扩展是否局部封装 |
| [Compute Boundary Review Guide](./compute-boundary-review-guide.md) | 检查是否误把 TianChen-RV 写成高层 compute IR |

## Quick Routing

- 修改 target、profile、variant legality、dispatch、emission path：读 capability-first guide。
- 新增 RVV/IME/offload/future extension：读 plugin-locality guide。
- 新增 op/dialect/pass 表达：读 compute-boundary guide。
- 修改 RVV route、lowering、artifact、测试或 performance layer：读 RVV plugin spec、variant pipeline、EmitC route、testing contract，并确认 typed body authority、selected-body realization、no legacy i32 positive route、no source-front-door current route、no status/artifact authority。

## Rule

如果 guide 与具体 spec 冲突，以具体 spec 为准；如果具体 spec 与 source design docs 冲突，先报告冲突。
