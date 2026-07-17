# CLAUDE.md — Weft-RV

本项目用 **Trellis** 管理，开发者身份 `claude`。**Trellis（`.trellis/`）是唯一权威**：队列、任务、spec、issues、报告交付全在其中。

## 上岗：只读 `.trellis/`

**进入点 = [`.trellis/spec/index.md`](.trellis/spec/index.md)**（根地图：项目定位 + 六层表 + 上岗顺序 + N↔C bridge 正本）。**读序、何时可动手，以该文件的「新 agent 上岗顺序」节为准**，本文件不重抄（禁副本）。

摘要（细节回根地图）：

1. [`.trellis/spec/index.md`](.trellis/spec/index.md) —— 根地图。
2. [`.trellis/spec/governance/index.md`](.trellis/spec/governance/index.md) → [决策权限卡](.trellis/spec/governance/决策权限卡.md) —— **什么能自决、什么必问、什么禁停**。
3. [`.trellis/spec/canon/index.md`](.trellis/spec/canon/index.md) → [核心不变量](.trellis/spec/canon/核心不变量.md) I1–I9 —— **法源**。
4. [`.trellis/spec/issues/index.md`](.trellis/spec/issues/index.md) —— 唯一问题登记簿；开工前查号，「待裁」条目的**保守默认就是现行法**。
5. 按任务性质取一层：[architecture](.trellis/spec/architecture/index.md)（动代码）/ [measurement](.trellis/spec/measurement/index.md)（跑数字）/ [evidence](.trellis/spec/evidence/index.md)（引工件）。

**动手前**：过该层 Pre-Development Checklist + **挂 Trellis 任务并预注册范围**。收工过该层 Quality Check。

## 项目

基于 MLIR 的能力驱动可扩展执行层软件栈之**参考模板**；RISC-V 量化 LLM 推理为其**首个高性能实例**（主角 = 可扩展性、性能 = 证明书）。**模板 ≠ 通用编译器**（输入止于 kernel 级接口、负载域锁 ggml 型量化推理 kernel）。主栈 C++/MLIR/LLVM/TableGen/CMake/lit；Python 只做 tooling。

**定位与三贡献 C1/C2/C3′ 的条文正本** = 根地图 + [canon · 暂定-科研主张](.trellis/spec/canon/暂定-科研主张.md)（【暂定·随论文侧更新·非定论】）。**agent 只做降级与标注，不得发明、替换或"改进"科研主张。** 本文件不复述主张（禁副本）。

## 三条本文件级纪律

- spec 是**稳定契约 + 判断依据，不是状态机/门禁**。当前进度/状态属 `.trellis/tasks/` 与 workspace journal，不写进 spec。
- **spec 只写现行法**：零历史叙事、零编号考古。教训压缩成规则本身。
- 改代码前先确认推进的是哪条贡献 C1/C2/C3′，否则可能是枝节（[思维准则](.trellis/spec/governance/思维准则.md)）。

## 硬件与测量

真实硬件 family 走板（板册与板别约束 = [measurement · 板册](.trellis/spec/measurement/板册.md)，现役 `rvv` / `k1` / `scalar`）。硬件/性能主张要真板证据；本地 build/lit 只是编译器/工具链证据。

**★测量现状（如实）**：measurement 层立 `bench <格> --board <板>` 为**唯一合法测量动作**。**runner 已建**（`tools/bench/bench`·`--self-test` 11/11·干跑验收达成），**但仍无合法的正式测量通道**——两条硬前置待裁：**ISSUE-090**（每格 harness 的落点与被调契约）+ **ISSUE-091**（命令签名**不定位主表行**：行键 = 四元组 `(op,format,engine,regime)`，而 `--board` 推不出 `engine`、格名命中多 op）。三目的地中 `experiments/runs.log` 已在，`master/`·`runs/<run-id>/` 未建。**⟹ 重启测量前须先落这两裁。** 详见 [issues](.trellis/spec/issues/index.md)。测量法（板、对手、门、行 schema、目的地）一律以 [measurement](.trellis/spec/measurement/index.md) 为准，本文件不重抄。

**性能判断规则**（[canon · 测量判据](.trellis/spec/canon/测量判据.md) / [对手与档位](.trellis/spec/canon/对手与档位.md) / [成色与措辞](.trellis/spec/canon/成色与措辞.md)）—— 正本在 canon，此处只列**入口提示**：

1. 修性能前**先反汇编认瓶颈**：未击中关键路径 = 小改善。
2. 一切选择键值 **per-format 板测定**：直觉投影不可信。
3. 性能主张绑 **相×板×格式×对手身份×门状态×账本(kernel/system)×双方编译器身份**；**kernel-axis vs-opponent 数仅编译器对称时有效**（[CASE-COMPILER-ASYMMETRY]）。
4. 性能立项前先给 **Amdahl 同域传导预估**；上限低于噪声地板者只能以机制/方法学名义立项。

## 决策权限

**[决策权限卡](.trellis/spec/governance/决策权限卡.md) 是正本**（四类穷尽：自决直行 / 必问 / 禁停 / 灰区 + 延后裁决制）。**每次压缩恢复后先读它**，本文件不重抄。

三条最易违反的（提示，非正本）：

- **判为「必问」≠ 停下**：正确动作 = **登记 + 采保守默认 + 续推**。**停下等裁决 = 违例。**
- **禁停机制**：「暂停/等你审视/会话已长/里程碑达成」**永不作为选项**出现在裁决请求中。压缩后恢复动作 = 读根地图 + 权限卡 + 继续队首，不请示。
- **禁把自决直行事项包装成选择题回门**（那是转嫁决策，不是谨慎）。

**工作流**：[`.trellis/workflow.md`](.trellis/workflow.md)（task 生命周期、spec 注入、check loop）。跨会话记忆见项目 memory（已开启）。
