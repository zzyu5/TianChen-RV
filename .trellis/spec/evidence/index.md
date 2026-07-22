# Evidence Specs —— 证据层

**版本**: v6

> **v4**（2026-07-20·B1）：测量记忆证据链补齐 structured T-N、唯一 master publisher 与 control-plane gate 的活锚；旧 design draft/generator 不再作为证据入口。
> **v5**（2026-07-22）：旧六态/C_construct/provenance 工件降为历史 inventory；当前
> formula authority 由 catalog/call graph/behavior tests 证明，strong construction 由
> delete-leaf reconstruction 证明，两者与 correctness/performance 分账。
> **v6**（2026-07-23）：C3 目标扩展为 RISC-V/GPU cross-paradigm realization；当前只有
> RISC-V 具备 implementation/hardware evidence。V2 architecture 与 artifact-neutral task
> 只能作为架构/计划工件，不能支撑 GPU correctness/performance。

> 当前贡献组织已更新为两柱下的 C1 architecture、C2 formula system 与 C3
> cross-paradigm realization。现有“三贡献证据地图”仍包含旧 C2 边际成本与 C3′ 标签，
> 暂作为工件 inventory 使用，不再定义贡献；重索引登记为 ISSUE-123。GPU 尚无证据行。

本层回答两个问题，且只回答这两个：

1. **"某条论文主张靠什么工件撑着、工件在哪、谁生产它、现在什么状态？"** → [三贡献证据地图](./三贡献证据地图.md)
2. **"拿到一个实验结果，该怎么解释它、什么解释是禁的？"** → [工件与实验参照](./工件与实验参照.md)

外加一份把最易随载体消失的知识钉死在本层的登记：[负结果登记](./负结果登记.md)。

## 本层是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 地图：贡献 → 支撑工件 → 仓库路径 → 生产者 → 现状 | 论证：不复述主张、不推理、不下结论 |
| 指针：路径经存在性核实（`test -e`）后才写入 | 数据副本：不转抄任何测量数字 |
| 证据解释口径：怎么读一个结果、什么解释非法 | 测量法：怎么测、在哪测、跟谁比 → [measurement](../measurement/index.md) |
| 成色的随行标注（诚实边界与工件不可分割） | 成色的评判：不判高低、不升格、不降级 |
| 负结果的**结论正本**（自足、不依赖任何报告可达） | 负结果的量值：倍数 / 格数 / 胜负计数 → 主表行 + run-id |

**定义正本不在本层**：三贡献的主张措辞、成色词汇、[F-1..F-6] 判据、[K-4]
构造强度、[COV-2] 分账边界、[L-\*] / [PERF-1] 各条 →
[canon](../canon/index.md)。本层对它们只用**标号 + 短标签**指认，不展开、不改写。

## 使用规则（五条，读本层任何文件前必须知道）

1. **定义在别处**。三贡献名与定义 = **【暂定·随论文侧更新】**，正本住 [canon](../canon/index.md) 论文侧节。本层引用其标号，**禁在此改写、发明、"改进"任何科研主张**。映射本身（工件 ↔ 路径）= 现行法。
2. **数字不住地图**。引用性能/正确性数字 = 引用主表行字段 + run-id，或由相应工具
   **现算**。旧 `C_construct` 等历史口径若被解释也只引用其原始机算工件；当前 formula
   authority 和 strong reconstruction 不靠手填比例，本层不落任何数值。
3. **路径即事实**。表内每条路径已核实存在。核实**不存在**者、**待裁去向**者，只出现在 [三贡献证据地图 §六 缺口](./三贡献证据地图.md#六缺口本节是地图的诚实边界)，不进正文。
4. **成色随行**。带诚实标注的工件（honest-missing / DORMANT / 归因锁 / 边界 caveat），其标注是**登记档不可分割的部分**——引用该工件必带其 caveat，**禁只取结论丢边界**。
5. **门的锚点 = 脚本本体路径，不是 CI job 名**。机检门以其脚本文件定位。

## Pre-Development Checklist

引用证据、写实验、解释结果之前逐条过：

- [ ] 要引的工件在地图里**有行**吗？没有行 = 该工件不是登记证据，先入表再引。
- [ ] 该行带 caveat / 成色锁吗？带则**连 caveat 一起引**（规则 4）。
- [ ] 你要写的数字，是**引行字段 + run-id**，还是手抄的？手抄 = 非法（规则 2）。
- [ ] 你要引的路径，`test -e` 过吗？不存在的路径只能进 §六 缺口（规则 3）。
- [ ] 这个实验是**验证既有系统契约**，还是在反向定义结构？后者违 [canon · 核心不变量](../canon/核心不变量.md) I9。
- [ ] RVV 证据绑到 `ssh rvv` 或另一具名 profile 了吗（I8）？本地 build / lit 只是编译器工具链证据，不是硬件证据。
- [ ] 若涉及 GPU，是否已有真实 GPU family construction、artifact/runtime 与具名 device
  run lineage？只有 V2 文档或 GPU-ready interface 时不得写 GPU 支持/性能。
- [ ] 强义 / 弱义 constructed 的判断有 delete-leaf reconstruction 或等价直接构造
  证据吗？Catalog/provenance 清单不能替代 [L-8]。
- [ ] Formula authority、strong reconstruction、correctness 和 performance 是否分别
  报告，未从其中一项互推另一项（[COV-2]）？
- [ ] micro → e2e 传导主张带 Amdahl 四列了吗？带宽受限内核以 **parity 为零假设**（[canon](../canon/index.md) 传导会计）。
- [ ] 性能格标了状态枚举、且**同会话配对**了吗（跨会话不比）？
- [ ] 要写"击败"吗？[PERF-1] 全绿前**任何"击败"措辞不存在**（[NG-4]）。
- [ ] 落在 [Forbidden Interpretations](./工件与实验参照.md#forbidden-interpretations) 清单里了吗？

## 本层文件

| 文件 | 内容 |
|---|---|
| [三贡献证据地图](./三贡献证据地图.md) | 现有工件 inventory；旧贡献标签待按两柱/C1-C3 重索引（ISSUE-123） |
| [负结果登记](./负结果登记.md) | 既有性能/能力边界登记；旧 C3′ 标签待重索引，结论与负结果本身继续有效 |
| [工件与实验参照](./工件与实验参照.md) | 研究问题 Q1–Q5、消融参照、硬件证据探针、基线纪律（CANON 未承载的部分）、Forbidden Interpretations |

## Quality Check

- **实验不反向定义结构**：实验文档不得围着一个方便的结果反推系统结构（[canon · 核心不变量](../canon/核心不变量.md) I9）。
- **结果须报**：能力 profile、选中的变体、合法性 / dispatch 行为、fallback 行为、可复现元数据。
- **四类证据不混**：架构证据 / 运行证据 / 性能证据 / 插件局部性证据，各自分开报。
- **地图零死指针**：正文每条路径 `test -e` 为真；为假或去向待裁的，一律降入 §六 缺口。
- **禁副本**：本层与 [canon](../canon/index.md) / [measurement](../measurement/index.md) 同题者，本层**只留指针**。两处写同一条 = 缺陷。
- **禁自裁**：科研主张类冲突（三贡献命名、[L-7] Win taxonomy 等）**只登记、不发明、不替换、不改进**；去向指 [issues](../issues/index.md) 编号。
- **跨文件引用禁行号**：用相对链接 + 标题锚，或条目编号（`ISSUE-073` / `[K-5]` / `[C1-SHAPE]`）。行号引用会随对侧编辑腐坏成假引文。
