# CLAUDE.md — TianChen-RV

本项目用 **Trellis** 管理，开发者身份 `claude`。改设计 / 代码 / 任务前，按序读：

1. [`.trellis/spec/index.md`](.trellis/spec/index.md) — 项目定位 + Novelty 表（N1/N2/N3）+ spec 层地图。
2. [`.trellis/spec/architecture/core-invariants.md`](.trellis/spec/architecture/core-invariants.md) — I1–I9 硬规则（其他 spec 引用它，不重抄）。
3. 选下一步做什么时读 [`.trellis/spec/guides/trunk-discipline.md`](.trellis/spec/guides/trunk-discipline.md) — 推进 N1/N2/N3 主干，别挑相邻枝节。

**项目**：能力驱动的统一 RISC-V MLIR execution layer（边界见 [AGENTS.md](AGENTS.md) 的 Project Scope）。主栈 C++/MLIR/LLVM/TableGen/CMake/lit；Python 只做 tooling。RVV 是当前真实硬件 family（`ssh rvv`）。

**Novelty（论文主张）**：N1 RISC-V 扩展异构性作为 first-class capability IR；N2 零-core-branch 的 plugin 泛化（用第二 family IME 证明）；N3 capability/resource-aware 的跨 family tune（Gearbox，且需实测胜出）。

**关键纪律**：
- spec 是给 agent 的**稳定契约 + 判断依据，不是状态机/门禁**。当前进度/状态属于 `tasks/` 和 `workspace/` journal，不写进 spec。
- 改代码前先确认推进的是哪个 N1/N2/N3，否则可能是枝节。
- 硬件/性能主张要真 `ssh rvv` 证据。

**工作流**：[`.trellis/workflow.md`](.trellis/workflow.md)（task 生命周期、spec 注入、check loop）。跨会话记忆见项目 memory（已开启）。
