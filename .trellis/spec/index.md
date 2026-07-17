# Weft-RV MLIR Trellis Specs —— 根地图

`.trellis/spec/` 是 Weft-RV 的**长期规范（durable spec）**：约束设计、代码、实验解释和 agent 接手方式。**Trellis 是本项目唯一权威**，承载队列 / 任务 / spec / issues / 报告交付。

规范只描述**稳定契约**。当前进度、stage 编号、measurement 状态、campaign / gate 记录、journal / session 引用——这些都**不是 spec**，属于 `.trellis/tasks/`、`.trellis/workspace/` journal、`artifacts/`。

这些 spec 是给 AI agent 读的**判断依据**，不是状态机或门禁。按原则判断、始终对齐主干；不要把条目当成机械打勾的 gate，spec 也不写"做完 X 才能做 Y"这类流程闸门。Spec 给方向和边界，怎么走由 agent 判断。

**本树的三条通用纪律**（各层重申，此处声明其存在）：

1. **只写现行法** —— 零历史叙事、零编号考古、零"某年某月某人裁定"。教训**压缩成规则本身**。
2. **禁副本** —— 一条规则只有一个家。两处写同一条 = 缺陷；合并留一 + 链接。
3. **跨文件引用禁行号** —— 一律用**相对链接 + 标题锚**或**条目编号**（`ISSUE-067` / `[K-5]` / `I3`）。行号会随改写腐坏成**假引文**；假引文比死链更坏——死链会报错，假引文会被信。

## 项目定位

Weft-RV 是**基于 MLIR 的能力驱动（capability-driven）可扩展执行层软件栈之参考模板（reference template）**——为碎片化硬件生态（RISC-V 是极端案例：VLEN 任意、扩展组合爆炸、厂商专有单元各异）给出一个可复制的**栈组织方式**（接入成本可预期 + 正确性机检 + 选择可归因），而不是又一个手写库。**RISC-V 量化 LLM 推理是该模板的首个高性能实例**：真硅上打赢手写出货物是模板质量的证明书，不是定位本身（主角 = 可扩展性；性能 = 证据）。

**模板 ≠ 通用编译器**：输入侧止于 kernel 级接口，负载域锁 ggml 型量化推理 kernel（[NG-2]）。

> **定位红线的条文正本**住 [canon · 暂定-科研主张 T-1](./canon/暂定-科研主张.md)（[G-0] 终极目标 / [G-2] 定位红线，原样）。本节只作导览，判断以该处为准。

作为软件栈，它是 high-level MLIR 之后的能力驱动统一 RISC-V 执行层：把 RISC-V 的目标能力（ISA 扩展、VLEN/uarch、toolchain、runtime/offload）建成 first-class、可查询、可验证、可参与 pass 决策的 MLIR 对象，并用这些能力对象：

1. 驱动 **plugin-local** 的 variant 生成 / 合法性 / 选择 / dispatch —— 驯服 RISC-V 扩展的组合异构性；
2. 参数化一个 **resource-aware 的 tuning / realization 层（Gearbox）** —— 把选中的 extension body 变成调优过的可执行 body。

它**不是**新的高层 tensor/tile IR，也**不是**"一个硬件一个互不相关 backend dialect"的集合。RVV 是第一个完整 family 与硬件证明；IME / offload / 未来 vendor 扩展走**同一条 common 路径**。

规范上的 dataflow spine 与机器全图见 [architecture · 系统定位与边界](./architecture/系统定位与边界.md)；跨文件复用的硬规则见 [canon · 核心不变量](./canon/核心不变量.md)，本树其他文件**引用**它而不重述。

## Novelty（论文主张 — 写 spec / 代码前必须对齐）

> **【暂定 · 随论文侧更新 · 非定论】** 本节三贡献表与下方 bridge 属**科研表述**，重构中、随论文侧更新。**agent 只做降级与标注，不得发明、替换或"改进"任何科研主张。**
>
> **条文全文正本**（[G-0]/[G-2]/[C1-\*]/[C2-\*]/[C3-\*]/[C1-RW] 逐条原样 + 摘编标注）住 [canon · 暂定-科研主张](./canon/暂定-科研主张.md)。本表是**导览与证据门索引**；两处冲突时以 canon 的原样条文为准，并把冲突登记进 [issues](./issues/index.md)。
>
> **本节不得据以判定胜负或数字可否上报**——那属 [canon](./canon/index.md) 其余文件的定法。

论文主张按科研目标总纲 v2 的**三条贡献 C1/C2/C3′ + 成熟编译器**组织；旧的 N1/N2/N3 不作废，而是**下沉为机制轴的构件**（映射见下方 bridge，本树其他文件与 CLAUDE.md 仍按 N1/N2/N3 引用，不推倒）。

**模板叙事**：三贡献是同一"可扩展软件栈参考模板"的三个面——**C1（头牌）= 模板协议本体**、**C2 = 模板经济学**、**C3′ = 模板产出质量**；性能数字是 C3′ 模板产出质量的**证词**，不另立贡献。编号与全部数值不变。

| # | 主张（终态） | 成立所需证据（缺证据就只是工程，不是贡献） |
|---|---|---|
| **C1**（头牌） | **模板协议本体 · 合取机制的存在性 → 可复制扩展接入协议**：一份带关系的能力 schema + 插件协议（接入五件套）+ falsifier 机检 = 可复制的扩展接入协议，同时驱动编译期变体生成与 fail-closed 运行期（装载期解析形态起步）调度守卫，**跨计算范式（向量 SIMD → 整矩阵 MAC）且跨独立家族（向量缺席的标量家族）**原封复用；接入升级为外部贡献者可循的**协议** | 零家族分支由 falsifier 组 [F-1..F-6]（含独立性判据 [core-invariants](./architecture/core-invariants.md) [F-6]、操作门 [F-2′]）机检并 CI 常绿；schema.def 自第二家族起未被接入触及（[F-2′] 逐 PR 审计）；≥3 家族下证据阶梯成立 |
| **C2** | **模板经济学 · 泛化代价 → 边际成本规律**：零分支/零核心改动不变量下，逐家族接入代价形成**边际递减曲线**，并给出结构解释（成本住模式/谓词/测试哪一处）；第二/第三个异质家族（IME/标量/zvfh）接入多顺 = 模板故事最强证据 | 成本 ledger 脚本自动生成且首点可复算（cloc、测试单列）；≥3 数据点成曲线；术语按 [L-2] integrated/independent-attached；对照锚在位。**诚实标注：曲线仍缺（landed X-SCALAR 把 1 点变 ~2 点 / 需 ≥3 点）**——"第三家族小"正是主张本身（边际递减），不是弱点 |
| **C3′** | **模板产出质量 · 能力键控优化模式库 → 带实测与迁移的模板**：模式以能力谓词表达、由机制选出（归因日志）、跨 VLEN/微结构**换键不改条目**地迁移，**正例负例边界齐备、迁移可预测**，对 tuned 框架内核**分相**报告增量 | 注册表是**数据文件**（[PAT-1..3]）；迁移判据双板 diff=0 CI 常绿；对框架自身同-ISA kernel 实测胜出/持平（[core-invariants](./architecture/core-invariants.md) [L-6] vs-framework，过 [PERF-1] 八门）；scalar/naive 只作内部 sanity，**绝不**作贡献倍数。**性能数字在此陈列为模板质量证词，不另立贡献** |
| **成熟编译器** | 覆盖率与正确性门槛达标，成为本负载域内**真正可用的编译器** | 六态阶梯（[core-invariants](./architecture/core-invariants.md) [K-4]）自动读出、四覆盖率指标 + 燃减曲线进 CI；正确性门（字节精确 / ULP 上界 / VLEN 翻转 / objdump golden）全绿。**成熟度进 CI，不进 slides**（见双轴组织原则） |

> **本表读法（标注，非改写）**：
> - 表内 `[PERF-1] 八门` 的**门数称谓待裁**（门体逐项列十项）——引用本门一律用 `[PERF-1]`、**禁带门数**；见 [ISSUE-071](./issues/spec树与治理.md)。
> - 表内指向 [`architecture/core-invariants.md`](./architecture/core-invariants.md) 的四处链接**维持原指**：该文件的去向与 [canon · 核心不变量](./canon/核心不变量.md) 的关系属 [ISSUE-070](./issues/spec树与治理.md)（canon 级 · 待裁），根地图不代裁。**I1–I9 正文两处逐字一致**（谓词：`diff <(awk '/^## I1 /,/^# 附加硬规则/' .trellis/spec/architecture/core-invariants.md) <(awk '/^## I1 /,/^## 附加硬规则/' .trellis/spec/canon/核心不变量.md)` → 唯一差异是两侧各自的 range 终止标题行本身，I1–I9 全部条文零差异），读任一处得到的规则相同。

### 双轴组织原则（引擎轴 × 证据轴）

**条文正本住 [canon · 暂定-科研主张 T-6](./canon/暂定-科研主张.md)**（[G-1] 引擎轴 / 证据轴 / 裁定「成熟度指标进 CI，不进 slides」，原样）。**本处不重抄**（禁副本）。

### N1/N2/N3 ↔ C1/C2/C3′ bridge（本文件桥接，其他文件仍按 N 号引用）

**本节是全项目唯一的 bridge 正本。** 任何其他文件（含 [canon · 暂定-科研主张 T-7](./canon/暂定-科研主张.md)）只链接过来，**不得重抄，也不得另称"唯一 bridge"**。

- **C1 = N1 ∧ N2 的合取存在性证据**：N1（能力异构性作 first-class IR，是 substrate 而非独立卖点）与 N2（零-core-branch plugin 泛化，用第二 family 证明）在**同一 schema** 上合取复用，即 C1 的"合取机制存在性 → 可复制协议"。
- **C2 = 泛化代价新轴**：在 N2 的零核心改动不变量之上，度量**逐家族接入的边际成本**，把工程不变量升级为经验规律。
- **C3′ ⊇ N3 的升级**：N3（capability/resource-aware 跨 family tune）升级为**能力键控模式库**——不止选变体，而是把优化沉淀为带实测与迁移判据的一等注册表对象。

> "execution-variant 容器""plugin 化"本身**不是** novelty —— MLIR 的 dialect + interface 已提供。不要把架构选择当贡献卖点。novelty 只在 C1/C2/C3′，且都以证据为准；成熟编译器是工程面、进 CI 不作 slide 卖点。
>
> **N1 是 substrate，不是独立卖点**：把能力建成可查询对象本身 ≈ LLVM `-mattr`/TTI 已做的工程；其 novelty **只在**它是跨 family 复用的同一事实源——由 N2 的第二 family 证明（→ C1）、由 N3 的 tune 兑现（→ C3′）。抽掉跨 family 复用就塌回纯工程。capability 驱动的 LMUL/形状选择是 enumerate→prune→select→stamp（一个 stamping pass 写 attr），**不是 IR-rewriting transform pass**，别这么描述。

（bridge 之外另有一条不住本文件的规则：**N1/N2/N3 是命名的机制子主张，不得再当三个并列贡献**——住 [canon · 暂定-科研主张 T-7](./canon/暂定-科研主张.md)。）

## Spec Layers（六层）

| Layer | 回答什么问题 | 什么时候读 |
|---|---|---|
| [canon](./canon/index.md) | **法源**：已定法条与终裁的唯一合订——某主张能不能立、某数字能不能报、某格算不算赢；含 [核心不变量](./canon/核心不变量.md) I1–I9 与【暂定】科研主张 | **必读**。落任何判断前 |
| [measurement](./measurement/index.md) | **测量法**：什么算测量、在哪块板测、用哪条流水线、跟谁比、结果写到哪、哪些检查作数 | 要跑出任何数字前（**runner 未建**，见下） |
| [architecture](./architecture/index.md) | **结构定法**：这台机器由哪些工位组成、每个工位今天在代码里**实际是什么**、设计上要成为什么、哪些结构是禁区 | 要动代码 / 动 IR / 加家族前 |
| [evidence](./evidence/index.md) | **证据地图**：某条主张靠什么工件撑着、工件在哪、谁生产它、现状如何；拿到结果怎么解释、什么解释是禁的 | 要引工件 / 写论文素材 / 解释一个结果时 |
| [governance](./governance/index.md) | **怎么做事**：决策权限卡（自决 / 必问 / 禁停 / 灰区）、延后裁决制、队列与简报、Trellis 卫生、思维准则 | **必读**。开工前 |
| [issues](./issues/index.md) | **唯一问题登记簿**：全部已知缺口、待裁事项、被证伪的在册结论、未清欠账，以 `ISSUE-NNN` 在此**且仅在此**登记 | **必读**。开工前查号 + 收工前登记 |

**六层之外**：`.trellis/spec/` 下仍存在旧层目录（`capability-model/` · `core-dialect/` · `plugin-protocol/` · `extension-plugins/` · `variant-pipeline/` · `lowering-runtime/` · `implementation-stack/` · `testing/` · `validation/` · `guides/`）与平铺 `SPEC-*.md` / `ISSUES.md`。**它们不是第七层，也不再有任何条文**：条文已**全部**并入上表六层，这 **41 份文件现在全部是指路牌**——每份只声明"条文迁去哪了"，**零规则、零副本**。

> **谓词（可复跑）**：
> ```bash
> cd .trellis/spec
> ls capability-model/*.md core-dialect/*.md plugin-protocol/*.md extension-plugins/*.md \
>    variant-pipeline/*.md lowering-runtime/*.md implementation-stack/*.md \
>    testing/*.md validation/*.md guides/*.md SPEC-*.md ISSUES.md | wc -l          # → 41
> for f in capability-model/*.md core-dialect/*.md plugin-protocol/*.md extension-plugins/*.md \
>          variant-pipeline/*.md lowering-runtime/*.md implementation-stack/*.md \
>          testing/*.md validation/*.md guides/*.md SPEC-*.md ISSUES.md; do
>   awk 'NR==1 && !/指路牌|已迁出/ {print FILENAME}' "$f"; done | wc -l              # → 0（零反例）
> ```
> 反例数 = 0 ⟺ 41 份**全部**是指路牌。**这是全量谓词，不是抽样**——若将来有人往旧层塞回条文，反例数会立刻 > 0。

**禁在旧层新增或修改规则**——旧层是路径兼容层（防既有引用腐坏成假引文），不是可写面。其**整体删除**的去向属 [ISSUE-070](./issues/spec树与治理.md)（canon 级 · 待裁）：删了对不上外部引用，留着又长期是死路径，故不自裁。

## 新 agent 上岗顺序

**只读 `.trellis/` 即可上岗**，按此序：

1. **本文件**（根地图）—— 项目是什么、六层各管什么、bridge。
2. **[governance/index.md](./governance/index.md)** → [决策权限卡](./governance/决策权限卡.md)。**先知道什么能自决、什么必问、什么禁停**，再动手。要点：判为「必问」= **登记 + 采保守默认 + 续推**；**停下等裁决 = 违例**。
3. **[canon/index.md](./canon/index.md)** → [核心不变量](./canon/核心不变量.md)（I1–I9）+ [非目标](./canon/非目标.md)（[NG-1..NG-8]）。这是**法源**：判断的依据全在此层。
4. **[issues/index.md](./issues/index.md)** 的[全册索引](./issues/index.md)。**扫一遍**——你要碰的东西大概率已经有号了（**禁重复立条**），而「待裁」条目的**保守默认就是现行法**。
5. **按任务性质**取其一（不必全读）：
   - 要动代码 / IR / 家族 → **[architecture/index.md](./architecture/index.md)**（先读其「读法口径」四条：【现状】/【目标】/【定法】禁混写）。
   - 要跑数字 → **[measurement/index.md](./measurement/index.md)**（**本层未授权的测量动作即非法动作**）。
   - 要引工件 / 解释结果 → **[evidence/index.md](./evidence/index.md)**。
6. **[governance/思维准则.md](./governance/思维准则.md)** —— 选下一步做什么时的单一尺子 = **distance to C1/C2/C3′**，别挑相邻枝节。

**何时可动手**：走完 1–4（+ 5 中相关那一层）→ 该层 **Pre-Development Checklist** 逐条过一遍 → **挂 Trellis 任务并预注册范围**（一切测量 / 施工必须挂任务，见 [trellis 卫生](./governance/trellis卫生.md)）→ 动手。收工走该层 **Quality Check**。

**测量的现状（如实）**：measurement 层立 `bench <格> --board <板>` 为**唯一合法测量动作**，但**该 runner 不存在**，其钉死的三处目的地（`experiments/master/` · `experiments/runs/` · `experiments/runs.log`）也均不存在——见 [ISSUE-067](./issues/门与工具.md)（可复跑谓词在条目内）。现役表在 `experiments/active/result-tables/`（[ISSUE-073](./issues/spec树与治理.md)）。**⟹ 今天没有合法的正式测量通道**；测量重启前 runner 是唯一硬前置。

**工作流**：[`.trellis/workflow.md`](../workflow.md)（task 生命周期、spec 注入、check loop）。
