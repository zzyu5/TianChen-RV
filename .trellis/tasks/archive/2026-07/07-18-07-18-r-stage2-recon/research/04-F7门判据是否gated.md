# Research: F-7 门判据 —— 是否已定 / gated on ISSUE-031

- **Query**: "决策密度趋零 + 点不入源" 怎么机检 · F-7 判据是否已定还是 gated on ISSUE-031
- **Scope**: 内部（spec canon/issues/architecture + 收口简报）
- **Date**: 2026-07-18

## 结论（一句 · ★关键判据级警示）

**F-7 门判据【未定】，硬 gated on ISSUE-031（待裁）。** 收口简报明文：「**★F-7 不可用作尺子：其判据 UNRESOLVED（PR-23：两种判据给出相反的重构指令）**」。canon `核心不变量.md` 的门表**只有 F-1..F-6，没有 F-7**（F-7 只在 issues 层作为"待立的门"出现）。∴ §四.3「立 F-7 门」这一步**不是纯工，是判据级待裁**——判据取甲/乙/丙未定，且甲乙给出**相反**的重构指令。

## 证据链

### (1) canon 门表无 F-7
`grep -n '\[F-[0-9]' .trellis/spec/canon/核心不变量.md` 只列 **[F-2′]**（家族接入操作门）与 **[F-6]**（独立家族判据）；连同 evidence 层「[F-1..F-6] 判据 → canon」的指认，**已定义的门止于 F-6**。**F-7 在任何 canon/architecture/measurement 文件都无判据正文**（`grep -rn 'F-7'` 命中仅 issues 层 + 开测篇令文 + 收口简报）。

### (2) ISSUE-031 = 「F-7 决策住址门判据未定（C9 blocked）」· 状态 = 待裁
`.trellis/spec/issues/发射器与架构.md` ISSUE-031：
- **实质**：两种判据给出**相反**的重构指令。
  - **判据甲**「format 名是否进入 if / 三目条件式」：会给 GridCodebook 满分（该 TU format 名 23 次全是表符号、0 次进条件式），但该 TU 恰是重灾区（4/5 格焊死宽度、2 假旋钮）⟹ 甲**结构上抓不住**焊死字面量形态；且甲奖励把 if/else 链降成 constexpr 表来刷门 = **门反向激励**。
  - **判据乙**「format 名是否出现在 emitter 翻译单元（含表初始化器）」：IME 表与"降表方案"双双违规 ⟹「板测 registry 该住哪」必须先有答案（= **ISSUE-037**，未定案）。
  - **判据丙（语义级）**：两条决定性新证据（GridCodebook 焊死字面量 + `.value_or` 可按 TU 切文件刷绿）**强支持丙**，但丙的可执行形态**未落**。
- **另三处须一并裁**：(b) op-identity 分派是否豁免 · (c) 白名单与 GEVM/GEMM 镜像去重先于上门 · (d) 常量分界线（架构不变量白名单 vs 调优阈值驱逐）。
- **保守默认**：草案只落**目标态 + 三条待裁项，不落可执行判据**；C9 节点保持树内、状态 = blocked。**建议执行序 ISSUE-037 → ISSUE-031，去重先于上门。**
- **卡在**：判据 = **canon 级新红线**。**状态 = 待裁。**

### (3) 收口简报明令 F-7 不可用作尺子
`.trellis/事故档案/2026-07-17-大重构收口简报.md` L87：
> **★F-7 不可用作尺子**：其判据 **UNRESOLVED**（PR-23：两种判据给出相反的重构指令）

## "决策密度趋零 + 点不入源" 今日怎么机检 —— 现状

**没有钦定的机检形态**（正是 ISSUE-031 待裁的核心）。已存在的**可复跑谓词**（可作候选判据的原料，但哪个是权威 = 未定）：
- **决策密度**候选谓词：`grep -c 'value_or'`（全域 = BQL 51 / KQuant 5 / ForwardElementwise 2 / RVVToEmitC 1）、`grep -c 'getIntegerCoreLmul().value_or("mf2")'`（BQL 18 / KQuant 2）、`grep -c 'StringSwitch'`、format 名进条件式计数。
- **★但 ISSUE-031(b) 已证 `.value_or` 计数不可做 per-TU 门**：GridCodebook 该项 = 0 看似完美，实则 self-default 被搬到上游 TU 当参数递进来 ⟹ **按 TU 切文件即可刷绿** = 该门对"决策住哪"零分辨力。`.value_or` 计数**只可作跨 TU 全域指标，禁作 per-TU 门**。
- **"点不入源"（focal 决策不入源码）**：DeferredDequant 是范本（三个 0 + 完整形状轴），但把"宽度读自 IR"直接写成门判据，会**提前替用户否决候选③**（改门判据本身仍是活选项）——须裁。
- 对照已有门的写法（可类比但非权威）：F-2′ = `接入 PR diff ∩ schema.def = ∅`（集合判据）；F-6 = `implies 传递闭包 ∩ {rvv.*} = ∅ ∧ 存在 only_feasible 实例`（谓词 + 存在性）。F-7 类比形态 = 某"决策住址"谓词 + 反例并标——但具体谓词就是甲/乙/丙之争，**未定**。

## ISSUE-032 / ISSUE-037 的耦合（连带 gated）
- **ISSUE-032**（F-7 样板锚的反例必须并标 · 待施工）：标量后端 878 行 `value_or=0` 不得暗示"0 形状知识"（它把 planes=4/planeLanes=32 字面量烘进发射器）。**卡在：随 ISSUE-031 定稿落地。**
- **ISSUE-037**（IME constexpr 镜像 → 读 live schema · 待施工）：是**判据乙的前置**（"板测 registry 该住哪"未定案 ⟹ 判据乙无法落地）。**建议先做本条。**

## 划分：F-7 立门的哪部分是判据级、哪部分是工
- **判据级（gated on ISSUE-031，禁自裁）**：F-7 门本体的判据（甲/乙/丙）· "决策密度趋零"的机检定义 · "点不入源"的谓词 · 常量分界线(d) · op-identity 豁免(b)。这些是 **canon 级新红线**。
- **工（可先推，不依赖判据裁决）**：ISSUE-033 ②③④ 的**代码侧修法**（前门显式化 + DeferredDequant 化 + 二点阶梯设防 + GridCodebook 去焊死）——这些是"把决策上收"的实体动作，**做完会降低决策密度**，但**立不立门、按哪个判据算绿 = 判据级待裁**。ISSUE-037（读 live schema）也是工，且是判据乙前置。

## 出处 / 谓词
- `grep -n '\[F-[0-9]' .trellis/spec/canon/核心不变量.md`（只到 F-6）
- `grep -rn 'F-7' .trellis/spec/`（仅 issues + 令文 + 简报）
- ISSUE-031/032/037：`.trellis/spec/issues/发射器与架构.md`
- 收口简报 L87：`.trellis/事故档案/2026-07-17-大重构收口简报.md`
