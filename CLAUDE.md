# CLAUDE.md — TianChen-RV

本项目用 **Trellis** 管理，开发者身份 `claude`。改设计 / 代码 / 任务前，按序读：

1. [`.trellis/spec/index.md`](.trellis/spec/index.md) — 项目定位 + 三贡献 C1/C2/C3′ 表（含 N1/N2/N3 ↔ C 的唯一 bridge）+ spec 层地图。
2. [`.trellis/spec/architecture/core-invariants.md`](.trellis/spec/architecture/core-invariants.md) — I1–I9 硬规则（其他 spec 引用它，不重抄）。
3. 选下一步做什么时读 [`.trellis/spec/guides/trunk-discipline.md`](.trellis/spec/guides/trunk-discipline.md) — 单一尺子 = distance to C1/C2/C3′，别挑相邻枝节。

**项目**：能力驱动的统一 RISC-V MLIR execution layer（边界见 [AGENTS.md](AGENTS.md) 的 Project Scope）。主栈 C++/MLIR/LLVM/TableGen/CMake/lit；Python 只做 tooling。RVV 是当前真实硬件 family（`ssh rvv`）。

**论文贡献（headline，三条）**：C1 合取存在性 → 可复制协议；C2 泛化代价 → 边际成本规律；C3′ 能力键控优化模式库 → 带实测与迁移的模板。终态定义与证据门见 [`.trellis/spec/index.md`](.trellis/spec/index.md) 的三贡献表。N1/N2/N3 是命名的机制子主张，映射进 C1/C2/C3′（唯一 bridge 在 index.md），**不得再当三个并列贡献**。

**关键纪律**：
- spec 是给 agent 的**稳定契约 + 判断依据，不是状态机/门禁**。当前进度/状态属于 `tasks/` 和 `workspace/` journal，不写进 spec。
- 改代码前先确认推进的是哪条贡献 C1/C2/C3′，否则可能是枝节。
- 硬件/性能主张要真 `ssh rvv` 证据。

**性能常驻判断规则（G3 裁决植入，仅三条；历史结论住 T8/canon，不在此重复）**：
1. **修性能前先反汇编认瓶颈**：放大倍数由瓶颈形状决定，未击中关键路径 = 小改善（K-quant repack 5× 慢 = 全展开 regfile spill，非指令微质量）。
2. **一切选择键值 per-format 板测定**：直觉投影（如"更宽=更快"/"回卷省 vsetvli"）不可信，[GAP-P1] widen-to-m1 与 re-roll 两次证伪为证。
3. **性能主张绑 相×板×格式×对手身份（探针）×八门状态**：修法失败先查 T8 是否已证伪，别重试已证伪的偏方。

**工作流**：[`.trellis/workflow.md`](.trellis/workflow.md)（task 生命周期、spec 注入、check loop）。跨会话记忆见项目 memory（已开启）。
