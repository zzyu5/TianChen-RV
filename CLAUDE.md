# CLAUDE.md — TianChen-RV

本项目用 **Trellis** 管理，开发者身份 `claude`。改设计 / 代码 / 任务前，按序读：

1. [`.trellis/spec/index.md`](.trellis/spec/index.md) — 项目定位 + 三贡献 C1/C2/C3′ 表（含 N1/N2/N3 ↔ C 的唯一 bridge）+ spec 层地图。
2. [`.trellis/spec/architecture/core-invariants.md`](.trellis/spec/architecture/core-invariants.md) — I1–I9 硬规则（其他 spec 引用它，不重抄）。
3. 选下一步做什么时读 [`.trellis/spec/guides/trunk-discipline.md`](.trellis/spec/guides/trunk-discipline.md) — 单一尺子 = distance to C1/C2/C3′，别挑相邻枝节。

**项目（定位 · 2026-07-10 升级）**：基于 MLIR 的能力驱动可扩展执行层软件栈之**参考模板（reference template）**；RISC-V 量化 LLM 推理为其**首个高性能实例**（主角 = 可扩展性、性能 = 证明书；定位权威 [`docs/canon/TianChen-RV_定位-v2.md`](docs/canon/TianChen-RV_定位-v2.md)，边界见 [AGENTS.md](AGENTS.md) 的 Project Scope）。**模板 ≠ 通用编译器**（[NG-2] 输入止于 kernel 级接口、负载域锁 ggml 型量化推理 kernel 不变）。主栈 C++/MLIR/LLVM/TableGen/CMake/lit；Python 只做 tooling。RVV 是当前真实硬件 family（`ssh rvv`）。

**论文贡献（headline，三条 · 同一"可扩展软件栈参考模板"的三面）**：**C1（头牌）模板协议本体** = 合取存在性 → 可复制扩展接入协议；**C2 模板经济学** = 泛化代价 → 边际成本规律；**C3′ 模板产出质量** = 能力键控优化模式库 → 带实测与迁移的模板（性能数字是 C3′ 证词、不另立贡献）。终态定义与证据门见 [`.trellis/spec/index.md`](.trellis/spec/index.md) 的三贡献表。N1/N2/N3 是命名的机制子主张，映射进 C1/C2/C3′（唯一 bridge 在 index.md），**不得再当三个并列贡献**。

**关键纪律**：
- spec 是给 agent 的**稳定契约 + 判断依据，不是状态机/门禁**。当前进度/状态属于 `tasks/` 和 `workspace/` journal，不写进 spec。
- 改代码前先确认推进的是哪条贡献 C1/C2/C3′，否则可能是枝节。
- 硬件/性能主张要真 `ssh rvv` 证据。

**性能常驻判断规则（G3 裁决植入，仅三条；历史结论住 T8/canon，不在此重复）**：
1. **修性能前先反汇编认瓶颈**：放大倍数由瓶颈形状决定，未击中关键路径 = 小改善（K-quant repack 5× 慢 = 全展开 regfile spill，非指令微质量）。
2. **一切选择键值 per-format 板测定**：直觉投影（如"更宽=更快"/"回卷省 vsetvli"）不可信，[GAP-P1] widen-to-m1 与 re-roll 两次证伪为证。
3. **性能主张绑 相×板×格式×对手身份（探针）×八门状态×账本(kernel/system)×双方编译器身份×板 shipped-baseline 编译器**：**kernel-axis vs-opponent 数【仅编译器对称时有效】**（[CASE-COMPILER-ASYMMETRY] 2026-07-10：rvv S6 1.884× = clang-ours-vs-gcc-shipped artifact、对称 gcc 0.272× 撤回；判别键 = 板出货编译器 rvv=gcc-15 / k1=clang-18）。修法失败先查 T8 是否已证伪，别重试已证伪的偏方。
4. **性能立项前先给 Amdahl 传导预估**（目标占相内时间比例 × 预期改善 = e2e 上限）：上限低于噪声地板的项**只能以机制/方法学名义立项，不得以性能名义**（G3 四问定性：G2 融合 e2e null = norm 占 decode 0.05% × 融合改善 → e2e 上限 +0.05% < 噪声，是 micro↛e2e 档案级正面教材、非失败）。★**Amdahl 传导估算的 kernel 因子输入必须与目标部署【同域】（同编译器 / 同 deployed variant / 同输入路径）**——1.59× projection 喂 clang-micro 1.884 而非部署 gcc 0.334 = garbage-in（[CASE-COMPILER-ASYMMETRY]）；出货 = clang .o 正门。

**决策权限卡（2026-07-09 用户裁定植入，替代此前散落的自决条款；每次压缩恢复后先读）**：
- **【自决直行，不问】**（做完日志记一行：决定+依据+可逆性）：① 已立项战役内的里程碑推进与排序（立项即授权全程，除非撞停机规则或触碰下方"必问"）；② 队列内工作的执行细节/测量/入表/triage 归类；③ 预注册判读的执行（判据已写，结果落地照判照走，含"成功→自动进下一步"类条款）；④ 公开可得、仅为复现/测量所需的模型/数据获取（自行下载+校验+登记；仅付费/许可证存疑/超大占用 >板剩余 1/3 才升级）；⑤ 可逆工程决策（分支/worktree/build 配置/测试增补）；⑥ 两条已立项线之间的资源微调（以触碰集 diff 为证）。
- **【必问，等裁决】**：① 新战役/新立项（含"顺手先做个 X"——X 不在任何已立项任务树上即为新立项）；② canon/红线级变更（数值政策/工具链/措辞宪法/NG 条款/八门定义）；③ 不可逆动作（删 sealed 证据/改历史基线/覆盖 Win 登记/上游 pin bump）；④ 停机规则触发后的去向（两次超时/预注册判读失败无预案）；⑤ 花钱/许可证存疑/板硬件之外的外部世界动作。
- **【灰区】**：不确定属哪类 → 选可逆路径先行 + 标记决策点入日志 + 继续，不 idle 等待；用户回归批量追认或翻案。**禁止把"自决直行"类事项打包成选择题回门**（那是转嫁决策，不是谨慎）。

**工作流**：[`.trellis/workflow.md`](.trellis/workflow.md)（task 生命周期、spec 注入、check loop）。跨会话记忆见项目 memory（已开启）。
