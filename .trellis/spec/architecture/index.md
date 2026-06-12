# Architecture Specs

本层定义 TianChen-RV MLIR 的系统定位和长期边界。进入任何实现或实验解释前，先确认这里的 invariants。

## Pre-Development Checklist（判断提示，不是 gate）

- [ ] 这项工作仍把 TianChen-RV 描述为 high-level MLIR 之后的 RISC-V execution layer 吗？
- [ ] 避免了新增核心 `tcrv.matmul`/`tcrv.softmax`/`tcrv.reduce`/`tcrv.generic_tile` 这类高层计算 op 吗？（[core-invariants](./core-invariants.md) I2）
- [ ] capability object 在 pass 决策路径上，而不是注释吗？（I1）
- [ ] core/common 没有按 family 名分支吗？（I3）
- [ ] RVV dtype/config/operation 来自 typed body，而不是 route id/ABI string/artifact name/旧 `i32m1` helper 吗？（I5）
- [ ] 没有把 emission-plan status/dashboard/manifest/artifact metadata 当 route 或进度 authority 吗？（I4）
- [ ] 这一步推进的是哪个 Novelty 主张（N1/N2/N3）？还是相邻枝节？（[trunk-discipline](../guides/trunk-discipline.md)）

## Guidelines Index

| Spec | Description |
|---|---|
| [Core Invariants](./core-invariants.md) | 全项目复用的 9 条硬规则（I1–I9），其他文件引用它 |
| [System Positioning](./system-positioning.md) | 系统定位、统一系统/family 责任、N1/N2/N3 贡献、模块边界 |
| [Design Boundaries](./design-boundaries.md) | 禁止方向、硬件路线口径、论文表述边界 |

## Quality Check

- 系统叙事必须能一句话复述为：unified TCRV RISC-V MLIR for capability-scoped extension execution。
- 任何新增核心概念必须说明它属于 capability/variant/plugin/dispatch/fallback 之一，或解释为何需要扩展核心 interface。
- 出现 high-level compute core op 时视为 architecture violation，除非它明确属于某 extension family 内部执行 op。
- IME / Offload 是 **N2 的关键证据点**（第二个非-RVV family 走通 common 路径），不是"禁止"项；但也别让它们把还没打穿的 RVV 主线晾在半路——取舍读 [trunk-discipline](../guides/trunk-discipline.md)。
