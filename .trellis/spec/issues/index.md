# Issues —— 唯一问题登记簿

> **版本**：v4
> **本层是什么**：Weft-RV 全项目**唯一的问题登记簿**。一切已知缺口、待裁事项、被证伪的在册结论、未清欠账，都以 `ISSUE-NNN` 在此登记**且仅在此登记**。任务、报告、简报、commit **只引编号**，不重述条目内容；其他任何文件（含姊妹 spec 层、任务交付物）不得另开问题清单。
> **本层不是什么**：**不是工作台、不是强制队列**。Trellis task tree 是按需使用的协调工具；本层只回答「有哪些问题、每条卡在哪、状态是什么」。
> **修改途径**：agent 可在当前授权范围内登记、施工并按事实关闭工程条目；论文主张、公共契约或重大方向变更以当前用户裁决和 canon 为准。

---

## Pre-Development Checklist（落笔前必走一遍）

- [ ] 我要写的问题**已经有号了吗**？→ 先查 [全册索引](#三全册索引)。**禁重复立条**（一条问题一个号）。
- [ ] 我要引用别的文件吗？→ **禁写行号**。用**相对链接 + 标题锚**或**条目编号**（`ISSUE-067` / `[K-10]` / `G-3`）。理由与判例见 [§〇.5](#5-跨文件引用禁行号)。
- [ ] 我要断言「X 不存在 / 零命中」吗？→ **禁用 `head` / `tail` / `grep -A<N>` 等有界窗口命令作依据**；用精确谓词或 `awk` 抽到语法边界。展示可截断，**判断不许**（[§〇.8](#8-x-不存在类断言的核法)）。
- [ ] 我写的数字**带产生它的谓词**吗？没有谓词 → 标 `UNVERIFIED-*`，且**不得被引作权威计数**（[§〇.7](#7-数字纪律)）。
- [ ] 这条读起来**自足**吗？—— 只读本条能否开工？靠「出处」才看得懂 = **缺陷**（[§〇.5](#5-跨文件引用禁行号)）。
- [ ] 这是**科研主张**类（三贡献命名 / 解耦 / 分界层 / 论题位）吗？→ **只登记冲突事实，禁发明、禁替换、禁改进**（[§〇.6](#6-科研主张零发明)）。
- [ ] 是否真的需要用户裁决？→ 只有会改变论文主张、公共契约、外部动作或不可逆方向时才停下询问；其余在授权范围内按事实推进。
- [ ] 我要热改流水线 / 临时加门 / 战役中途换判据吗？→ **禁**，一律入册（[measurement](../measurement/index.md) §3.0 哲学与稳定性条款）。

---

## 〇、使用法

### 1. 编号
`ISSUE-NNN`，稳定、**只增不改**。条目关闭后保留原号并标状态，**号不回收、不重排**。新号由本层顺序分配（当前最大号见 [§三 全册索引](#三全册索引)）。

### 2. 条目字段
活动条目至少写清 {**标题** · **实质** · **影响面** · **状态** · **出处**}；存在前置时补 **卡在**，需要决策且仍可安全推进时补 **保守默认**。已关闭条目可压缩为裁决与可复核事实。

### 3. 状态

| 状态 | 含义 |
|---|---|
| **待裁** | 需要用户或 canon 级裁决；能安全推进的部分按保守默认继续 |
| **待施工** | 已有合法去向，缺的是工，可挂任务 |
| **阻塞** | 前置未就位（结构缺口 / 他条 ISSUE / 板 / 执行面外部依赖）；**前置须写明** |
| **已就绪** | 前置齐备、判据明确，取用即开工 |
| **RESOLVED / 已施工 / 已修 / 已退役** | 终态或历史终态；条目保留，不再占活动阻塞 |

> 新条目优先使用上表中的简洁状态；历史条目保留既有终态字样。ISSUE-076 已关闭，不再维护第二套强制状态机。

### 4. 入册途径
新问题（含新增检查的提案、流水线异议）一律入本层；**禁热改流水线、禁临时加门、禁战役中途换判据**（[measurement](../measurement/index.md) §3.0）。

### 5. ★可寻址 = 条目自足（不是「出处能查到」）
- **判据**：读懂一条、并据以开工，**只需读本条目本身**。凡 `实质` 栏未自足者 = **缺陷**，须把判断依据**搬进条目**。
- **`出处` 只是历史线索，不得是理解该条目的必要条件**。理由：`docs/` 全体归档进 `_attic/`，而 **`_attic/` 是 git-ignored —— 克隆本仓者拿不到**。同理 `PR-NN`（旧登记册行号）、令文、各期报告，**归档后一律不可达**。故本层的 `出处` 栏分两类，**逐条标明**：
  - **仓内谓词** —— 可复跑，是合法理解路径。
  - **历史线索（归档后不可达 · 非理解路径）** —— 仅供反查，**读不到不影响开工**。
- **可复跑优先**：能给谓词的给谓词（`grep` / `awk` / `objdump` / `ssh <板>` 命令连同读数一并写进条目），使新 agent 能**自行复算**而非信任转述。仓库内实体（`lib/` · `include/` · `schema/` · `tools/` · `experiments/` · `test/` · `.trellis/`）是合法锚点；`docs/` 与 `_attic/` **不是**。

#### 5. 跨文件引用禁行号
**一切跨文件引用禁用行号**（`文件:NN`）。合法形式只有两种：**相对链接 + 标题锚**，或**条目编号**（`ISSUE-067` / `[K-10]` / `[PERF-1]` / `G-3` / `§3.4`）。指向代码时用 **`grep`/`awk` 谓词 + 符号名**，不用行号。

> **判例（本规则是赔出来的，不是设计出来的）**：
> 1. 兄弟文件被并行改写后，钉死行号的引文**整批腐坏成假引文** —— 引号里的「原文」在目标行根本不存在，而目标行恰好还是一句像样的话 ⟹ **假引文比死链更坏**：死链会报错，假引文会被信。
> 2. 裸行号判错（自报的函数名/行号与实际相差一个变体，正确定义在另一行）→ 见 **ISSUE-058(b)**。
>
> **推论**：**行号是时点敏感的，写下即开始腐坏。** 同理**禁写「某文件目前有没有某字样」类断言** —— 那把本条的正确性绑上兄弟文件的编辑状态（判例：ISSUE-074）。

### 6. 科研主张零发明
**科研主张类条目**（三贡献命名、论题位、解耦 / 分界层、C3′ 等）在本层**只登记冲突事实，不发明、不替换、不改进**；其正文归 [canon](../canon/index.md) 的【暂定 · 随论文侧更新】节。**拿不准 → 原样搬 + 标暂定。**

### 7. 数字纪律
本层数字均须带**产生它的谓词**（第 5 条）。`UNVERIFIED-*` 前缀 = 该数**未经机算复核**、**不得被引作权威计数**，后缀记来源：`-doc-asserted`（文档自述）· `-audit-asserted`（只读审计自述、脚本未进仓）。**比值类数字须同时钉死分子与分母**；分母无仓库实体 ⟹ 算术不可复核 ⟹ 一律 `UNVERIFIED-*`。

### 8. 「X 不存在」类断言的核法
**禁用有界窗口命令作依据**（`head` / `tail` / `grep -A<N>` / 固定行数窗口）—— 用**精确谓词**或 `awk` 抽到语法边界。展示可截断，**判断不许**（[measurement](../measurement/index.md) §3.6 其余铁律）。

---

## 一、仍需明确裁决的高风险项

> 本节是重点索引，不取代各条本体状态。若条目已被后续事实或用户裁决关闭，应从本节移除并在全册索引保留历史记录。

### 1.1 会动头条 / 主表口径（硬冻结面）

| ISSUE | 一句话裁点 | 文件 |
|---|---|---|
| ISSUE-001 | dequant 零向量发射（族级）要不要立项 —— 属队列外新战役 | [性能与测量](./性能与测量.md#issue-001--dequant-leaf-族级零向量发射该轴输赢--宿主-codegen-抽签) |
| ISSUE-002 | dequant 路径向量内容由宿主编译器产生，该族称「由机制构造」是否仍合 [L-8] 强义 | [性能与测量](./性能与测量.md) |
| ISSUE-003 | 28 格 gcc 车道要不要按单世界 clang 重测（扩测量面 + 可能动头条） | [性能与测量](./性能与测量.md) |
| ISSUE-006 | 先例 7 格（对非部署 `_generic` 测得的 PASS）重测还是降披露列 | [性能与测量](./性能与测量.md) |
| ISSUE-007 | iq 系 gemm_tile 的 tier 分档（同一对手两行两档）改不改 | [性能与测量](./性能与测量.md) |
| ISSUE-009 | 主表注记与 evidence 打架（gather-tax）如何订正 | [性能与测量](./性能与测量.md) |
| ISSUE-010 | 对手 IQR 高的 3 格 verdict 是否标「数据存疑」 | [性能与测量](./性能与测量.md) |
| ISSUE-012 | 手调挑战值 = 每板 12 还是双板合计 ≥12 | [性能与测量](./性能与测量.md) |
| ISSUE-013 | 两档铁令验收未达（12 格从未走过构造→前门）的去向 | [性能与测量](./性能与测量.md) |
| ISSUE-014 | fold@M=1 判定为非物理地板后，墙记与论文侧待判定处的措辞 | [性能与测量](./性能与测量.md) |
| ISSUE-015 | P5 两条作废项的销号与重出 | [性能与测量](./性能与测量.md) |
| ISSUE-017 | decode 逐格 emit-golden 的格数口径 | [性能与测量](./性能与测量.md) |
| ISSUE-018 | 测量欠账表是否整体对账后再入册 | [性能与测量](./性能与测量.md) |
| ISSUE-029 | q8_0@ime 结构不可达的锁定写法追认 | [性能与测量](./性能与测量.md) |
| ISSUE-064 | 上游 ggml iq4_nl 双核数值错是否上报上游（外部世界动作） | [性能与测量](./性能与测量.md) |

### 1.2 队列与立项（队序 / 里程碑 = 用户独占）

| ISSUE | 一句话裁点 | 文件 |
|---|---|---|
| ISSUE-019 | VLEN 专化满展开 leaf 的队列位置（最高扇出 ∧ 高风险） | [性能与测量](./性能与测量.md) |
| ISSUE-022 | C4b 与 Q3 是否合并里程碑（共享 VLEN/码本参数化前置） | [性能与测量](./性能与测量.md) |
| ISSUE-023 | C4 拆两条机制（C4a/C4b）是否动树 | [性能与测量](./性能与测量.md) |
| ISSUE-028 | P3 执行序按实测证据重排（Q1→Q3→Q2）还是照令文序 | [性能与测量](./性能与测量.md) |
| ISSUE-034 | 是否排一条串行线做 loop_order 的 ODS 必填形态 | [发射器与架构](./发射器与架构.md) |
| ISSUE-038 | E7 立项前提已不成立，是否仍要 E7 | [发射器与架构](./发射器与架构.md) |
| ISSUE-053 | 5 项【写死】机制无承接节点（树定格禁自增） | [canon与措辞](./canon与措辞.md) |
| ISSUE-063 | rvv07 板去向 | [性能与测量](./性能与测量.md) |

### 1.3 发射器与台账

| ISSUE | 一句话裁点 | 文件 |
|---|---|---|
| ISSUE-031 | F-7 住址门判据取甲、乙还是丙（语义级）—— 甲/乙给出相反的重构指令 | [发射器与架构](./发射器与架构.md#issue-031--f-7-决策住址门判据未定c9-blocked) |
| ISSUE-040 | 描述符成本入账三件的处置 | [发射器与架构](./发射器与架构.md) |
| ISSUE-041 | C7 扇出台账三行重出/追认 | [发射器与架构](./发射器与架构.md) |
| ISSUE-042 | registry status enum 改名与 `partial` 退役 | [发射器与架构](./发射器与架构.md) |
| ISSUE-043 | 主发射器分阶段重构方案（只读审计已出账，红线轴 gated on ISSUE-031） | [发射器与架构](./发射器与架构.md) |
| ISSUE-124 | q4_0 monolith 保留依据未闭合：abstract contraction 仅有 lit 作者、无真实 producer | [发射器与架构](./发射器与架构.md) |
| ISSUE-125 | SP4 / loop-order selected stamp 与真实 emission 分裂 | [发射器与架构](./发射器与架构.md) |

### 1.4 canon 与措辞

| ISSUE | 一句话裁点 | 文件 |
|---|---|---|
| ISSUE-044 | canon v4 定稿（**检查点一**）—— 多条 ISSUE 的可关闭性依赖它 | [canon与措辞](./canon与措辞.md#issue-044--canon-v4-定稿检查点一) |
| ISSUE-045 | uarch 事实归 schema 还是归测量记忆层（现行三条文与新法正面冲突） | [canon与措辞](./canon与措辞.md) |
| ISSUE-046 | [PAT-3] 判据变更（现判据在本仓平凡真） | [canon与措辞](./canon与措辞.md) |
| ISSUE-047 | 探针 DUAL-AGREE 双法交叉验证是否入 canon 测量纪律 | [canon与措辞](./canon与措辞.md) |
| ISSUE-048 | 在册判别键「sub-block 数 16 vs 8」已被证伪，如何订正 | [canon与措辞](./canon与措辞.md) |
| ISSUE-049 | T-P 构造参数三段链主张强度收窄 | [canon与措辞](./canon与措辞.md) |
| ISSUE-050 | [K-5b] 要件① 语料缺口 + 三处预注册偏离的处置 | [canon与措辞](./canon与措辞.md) |
| ISSUE-052 | v4 之职 = 导入/对账（论文侧已先行落法），是否改为导入 | [canon与措辞](./canon与措辞.md) |
| ISSUE-111 | 数值口径松绑：headline 取最快变体并强制报告 ULP 界 | [canon与措辞](./canon与措辞.md) |

### 1.5 门与工具

| ISSUE | 一句话裁点 | 文件 |
|---|---|---|
| ISSUE-055 | C8 全节 verdict=FAIL，补三项还是撤 | [门与工具](./门与工具.md) |
| ISSUE-056 | C6 falsifier 进不进 CI（进则 CI 立刻红，不进则门形同虚设） | [门与工具](./门与工具.md) |
| ISSUE-058 | 假声明/裸行号四处订正 | [门与工具](./门与工具.md) |
| ISSUE-060 | 两个不依赖 runner 的 workflow 被连带删除（理由对它们不成立）—— 复活还是确认删 | [门与工具](./门与工具.md) |
| ISSUE-082 | 「falsifier 组 ≥3 家族 CI 常绿」措辞失锚（无人值守 CI 已不存在）改不改写 | [门与工具](./门与工具.md) |
| **ISSUE-104** | **★scalar route/parser 前置已由 B2 接通，但 roster 无 engine=scalar 行且 ISSUE-061 未解；dormant contract 不授予真跑资格** | [门与工具](./门与工具.md) |

### 1.6 ★spec 树自身（上岗阻塞面）

| ISSUE | 一句话裁点 | 文件 |
|---|---|---|
| ~~ISSUE-070~~ | **已裁 · 已落地**（2026-07-17 用户裁）：I1–I9 唯一权威本 = [canon · 核心不变量](../canon/核心不变量.md)；重复本已归档 `_attic/`，原路径留指针（sealed `$meta.authority` 所迫）。**连带第 2 问（既有 spec 树与六层归并）未随本裁了结**，仍在 [architecture](../architecture/index.md) 未决项表 B 行 | [spec树与治理](./spec树与治理.md) |
| ~~ISSUE-071~~ | **已裁 · 已落地**（2026-07-17 用户裁）：按实况订正 —— 门体 = **十项**（v2 ①–⑧ → v3 增补 ⑨⑩ 前置；「八门」= v2 化石简称）。一处定义 = [canon · 测量判据](../canon/测量判据.md) §一.2，全树引用一律 `[PERF-1]` 禁带门数 | [spec树与治理](./spec树与治理.md) |
| ISSUE-072 | evidence 层三处「唯一权威」锚在 `docs/`（G-1 性能证词根悬空 / G-3 承重论证载体） | [spec树与治理](./spec树与治理.md) |
| ISSUE-074 | 「五层机器」所指未定（**禁 agent 发明**）+ 裸 `L<数字>` 全局处置 | [spec树与治理](./spec树与治理.md) |
| ISSUE-066 | 事故档案与本册的边界（已修复但须防复发的事项进哪儿） | [spec树与治理](./spec树与治理.md) |
| **ISSUE-103** | **★T-X 六列证据表的 spec 层归属未定**（重构后无 testing 层 · 落 measurement 还是 evidence?）—— 保守默认暂落 `experiments/` 活证据 · 连带 B4 PR-1「采购 no-V 板」= RESOLVED-BY-FACT | [spec树与治理](./spec树与治理.md#issue-103--t-x-六列证据表的-spec-层归属未定重构后无-testing-层) |

---

## 二、Guidelines Index（条文文件）

| Spec | 收录 | 条数 |
|---|---|---|
| [性能与测量](./性能与测量.md) | 主表与对手政策（001–018）· 机制缺口与攻坚（019–030）· 板与外部世界（061–065 · 084）· 攻坚与架构不可达（100–102）· R线与新 census（106–107 · 112 · 117） | 43 |
| [发射器与架构](./发射器与架构.md) | 构造、前门与决策住址（031–043）· 架构未决（081）· K-quant 与 formula/schema 收口（109 · 113 · 115–116 · 118–122 · 124） | 24 |
| [canon与措辞](./canon与措辞.md) | canon、条文冲突与措辞纪律（044–053 · 088 · 111 · 123） | 13 |
| [门与工具](./门与工具.md) | 门体清算（054–060）· runner 与工具挂载（067–069 · 082 · 083 · 085 · 087 · 090–094）· bench 通道与 harness（096–099 · 104 · 105 · 114）· 工具默认失锚（108）· T-X≠S1（110） | 28 |
| [spec树与治理](./spec树与治理.md) | 登记簿治理（066）· spec 树自身的缺口（070–074 · 086 · 089）· 旧强制治理裁决（075–080）· 旧 goal 退役与 T-X 归属（095 · 103） | 16 |

**总条数 = 125**（ISSUE-001..ISSUE-125，**零缺号 · 零重号**）。状态分布（leading-token 口径）：**待裁 60 · RESOLVED 23 · 待施工 15 · 已就绪 8 · 阻塞 8 · 已施工 3 · 已裁 2 · 已修 2 · 回门待扫 1 · APPLIED-DEFAULT 1 · 已退役 1 · 裁准拆分 1**（Σ=125）。本行由 `.trellis/scripts/issues_census.py` 于 2026-07-21 机算刷新；后续禁止手抄沿用。

> **计数谓词（机算 · 禁手写小计）**：按 `### ISSUE-NNN` 切块、取每块最后一个 `- **状态**：` 行统计（子块内的重复状态行不重复计）。2026-07-17 §七③ `docs/` 归档轮收官实测：总数 **94** / 零缺号 / 零重号；分布 **待裁 58 · 待施工 14 · 阻塞 10 · 已就绪 9 · 已裁·已落地 2 · 已退役 1**。〔本轮 +1 = ISSUE-094（`--drift` 漂移门），经用户裁后**已退役**、不占待裁额。同轮并行写入者新增 090–093 并结清 2 条，故前几行的「93 / 60·14·10·9」是彼时口径、非漂移 —— **本行数字禁手抄，一律按上述谓词现算**。〕**同轮订正三处既存漂移**：ISSUE-089 在 [§三 全册索引](#三全册索引) **缺行**（本层自称「唯一入口索引」却查不到该号）· canon与措辞 条数 10→11（088 未计）· spec树与治理 条数 13→14（089 未计）。

---

## 三、全册索引

> 一次查完：号 → 标题 → 状态 → 住哪个文件。**这张表是本层的唯一入口索引**；新号顺序追加。

| ISSUE | 标题（简） | 状态 | 文件 |
|---|---|---|---|
| ISSUE-001 | dequant leaf 族级零向量发射 = 宿主 codegen 抽签 | 待裁 | 性能与测量 |
| ISSUE-002 | dequant 族「由机制构造」是否合 [L-8] 强义 | 待裁 | 性能与测量 |
| ISSUE-003 | recon「零 gcc」门是恒等式 + 28 格 gcc 车道污染 | 待裁 | 性能与测量 |
| ISSUE-004 | gemm_tile 对手政策统一为部署事实：执行销案 | 已施工 | 性能与测量 |
| ISSUE-005 | iq 8 board-cell 争议-pending 已落账 | 已施工 | 性能与测量 |
| ISSUE-006 | 先例 7 格 PASS 绑在非部署符号上 | 待裁 | 性能与测量 |
| ISSUE-007 | iq 系 gemm_tile tier 分档低估对手 | 待裁 | 性能与测量 |
| ISSUE-008 | q8_0 decode @rvv 便宜档成色三备注 | 待施工 | 性能与测量 |
| ISSUE-009 | 主表注记「gather-tax」与 evidence 打架 | 待裁 | 性能与测量 |
| ISSUE-010 | 数据质量存疑格（对手 IQR 过高） | 待裁 | 性能与测量 |
| ISSUE-011 | 悬置 dequant 格已具名（`dequantize_row|iq2_xs|@rvv`） | 已就绪 | 性能与测量 |
| ISSUE-012 | 手调挑战值口径（每板 12 vs 合计 ≥12） | 待裁 | 性能与测量 |
| ISSUE-013 | 两档铁令验收未达：12 格从未走过构造→前门 | 待裁 | 性能与测量 |
| ISSUE-014 | fold@M=1 非物理地板，但墙记指错成本中心 | 待裁 | 性能与测量 |
| ISSUE-015 | P5 两条作废项 + 三处 minor 订正 | 待裁 | 性能与测量 |
| ISSUE-016 | VOID 清偿余额 4 格 | 待施工 | 性能与测量 |
| ISSUE-017 | decode 逐格 emit-golden + 格数口径冲突 | 待裁 | 性能与测量 |
| ISSUE-018 | 测量欠账表须整体对账后才能入册 | 待裁 | 性能与测量 |
| ISSUE-019 | VLEN 专化满展开 leaf 缺口（最高扇出 ∧ 高风险） | 待裁 | 性能与测量 |
| ISSUE-020 | tiny-reduction：弹药就绪未施工 | 已就绪 | 性能与测量 |
| ISSUE-021 | tiny-codebook gather 机制（C4b）· 禁以性能名义立项 | 待施工 | 性能与测量 |
| ISSUE-022 | C4b 与 Q3 是否合并里程碑 | 待裁 | 性能与测量 |
| ISSUE-023 | C4「一箭双雕」部分证伪 → 拆 C4a/C4b | 待裁 | 性能与测量 |
| ISSUE-024 | mxfp4 缺专用 vec_dot 驱动 + oracle | 已就绪 | 性能与测量 |
| ISSUE-025 | nvfp4：dequant 纯标量 + gemm 阻于码本参数化 | 阻塞 | 性能与测量 |
| ISSUE-026 | iq4_nl 两格 = 合取两墙 | 阻塞 | 性能与测量 |
| ISSUE-027 | q4_1 / q8_0 decode @k1 结构缺口 | 阻塞 | 性能与测量 |
| ISSUE-028 | Q3 机制名指错成本中心 + P3 执行序冲突 | 待裁 | 性能与测量 |
| ISSUE-029 | q8_0@ime 结构上不存在合法 IME 对手 | 待裁 | 性能与测量 |
| ISSUE-030 | e2e 第三案未建（q2_K interleaver） | 待施工 | 性能与测量 |
| ISSUE-031 | F-7 决策住址门判据未定（C9 blocked） | 待裁 | 发射器与架构 |
| ISSUE-032 | F-7 样板锚的反例必须并标 | 待施工 | 发射器与架构 |
| ISSUE-033 | 四处已具名的真住址违规（宽度轴） | 阻塞 | 发射器与架构 |
| ISSUE-034 | loop_order「不读者构建失败」形态未实施 | 待裁 | 发射器与架构 |
| ISSUE-035 | strip-width 轴接入 + 记忆层首个真实改判 | 待裁 | 发射器与架构 |
| ISSUE-036 | plan 深层序列化 + round-trip lit 未做 | 已就绪 | 发射器与架构 |
| ISSUE-037 | IME 板测数据 constexpr 镜像 → 读 live schema | 待施工 | 发射器与架构 |
| ISSUE-038 | E7 的立项前提已不成立 | 待裁 | 发射器与架构 |
| ISSUE-039 | 后端注册表的注释与实表不符 | 已修 | 发射器与架构 |
| ISSUE-040 | 描述符成本入账收尾三件 | 待裁 | 发射器与架构 |
| ISSUE-041 | C7 扇出台账三处「已知行」被机算证伪 | 待裁 | 发射器与架构 |
| ISSUE-042 | registry status enum 与分层令文不一致 | 待裁 | 发射器与架构 |
| ISSUE-043 | 主发射器规模问题（54,314 行）· 方案待裁 | 待裁 | 发射器与架构 |
| ISSUE-044 | canon v4 定稿（检查点一） | 待裁 | canon与措辞 |
| ISSUE-045 | uarch 事实 vs schema 准入纪律正面冲突 | 待裁 | canon与措辞 |
| ISSUE-046 | [PAT-3] 判据在本仓平凡真 | 待裁 | canon与措辞 |
| ISSUE-047 | 探针 DUAL-AGREE 双法交叉验证入 canon | 待裁 | canon与措辞 |
| ISSUE-048 | 判别键「sub-block 数 16 vs 8」已被证伪 | 待裁 | canon与措辞 |
| ISSUE-049 | T-P 构造参数三段链主张强度须收窄 | 待裁 | canon与措辞 |
| ISSUE-050 | [K-5b] 语料完备缺口 + 三处预注册偏离 | 待裁 | canon与措辞 |
| ISSUE-051 | 三贡献命名已统一为 C1/C2/C3 | RESOLVED | canon与措辞 |
| ISSUE-052 | v4 之职 = 导入/对账而非重新立法 | 待裁 | canon与措辞 |
| ISSUE-053 | 5 项【写死】机制无承接节点 | 待裁 | canon与措辞 |
| ISSUE-054 | 空心检查清算余量 | 已就绪 | 门与工具 |
| ISSUE-055 | C8 全节 verdict = FAIL · 冻结引用待补三项 | 待裁 | 门与工具 |
| ISSUE-056 | C6 falsifier 保持红且未进 CI · 词表未合并 | 待裁 | 门与工具 |
| ISSUE-057 | 3 例既存 lit 失败（测试套件长期红） | 待施工 | 门与工具 |
| ISSUE-058 | 假声明繁殖 + 裸行号判错（四处订正） | 待裁 | 门与工具 |
| ISSUE-059 | 跨轮基准数不一致（「2 字节之谜」）真因未知 | 待施工 | 门与工具 |
| ISSUE-060 | 两个不依赖 runner 的 workflow 被连带删除 | 待裁 | 门与工具 |
| ISSUE-061 | scalar 板：对手链已建成，缺家族身份验收 | 已就绪 | 性能与测量 |
| ISSUE-062 | T-X 第 5 列「逐竞品产出数」未补齐 | 阻塞 | 性能与测量 |
| ISSUE-063 | rvv07 板去向未定 | 待裁 | 性能与测量 |
| ISSUE-064 | 上游 ggml iq4_nl 双核在 VLEN128 数值错 | 待裁 | 性能与测量 |
| ISSUE-065 | Tier-3 真第三方接入 = 域外 | 阻塞 | 性能与测量 |
| ISSUE-066 | 事故档案与本册的边界 | 待裁 | spec树与治理 |
| ISSUE-067 | bench runner、三目的地与 cell contract 已落地 | RESOLVED | 门与工具 |
| ISSUE-068 | `perf_covered_metrics.py` 成孤儿（无编排者） | 已就绪 | 门与工具 |
| ISSUE-069 | 新 experiments 布局仍有 732/1372 条一致性 findings | 待施工 | 门与工具 |
| ISSUE-070 | ★`core-invariants.md` 违「只写现行法」= 上岗阻塞 | 已裁 | spec树与治理 |
| ISSUE-071 | [PERF-1] 门项数：「八门」vs「十项」 | 已裁 | spec树与治理 |
| ISSUE-072 | evidence 三处「唯一权威」锚在 `docs/`（根悬空） | 待裁 | spec树与治理 |
| ISSUE-073 | 主表住址已统一到 `experiments/master/` | RESOLVED | spec树与治理 |
| ISSUE-074 | 「五层机器」所指未定（禁 agent 发明） | 待裁 | spec树与治理 |
| ISSUE-075 | 旧强制停机/续跑规则退役 | RESOLVED | spec树与治理 |
| ISSUE-076 | 第二套延后裁决状态机退役 | RESOLVED | spec树与治理 |
| ISSUE-077 | `ROADMAP=` 前缀不再强制 | RESOLVED | spec树与治理 |
| ISSUE-078 | 固定简报模板不再强制 | RESOLVED | spec树与治理 |
| ISSUE-079 | 实验预注册与 Trellis task 解耦 | RESOLVED | spec树与治理 |
| ISSUE-080 | task 母子自动聚合不再是全项目门 | RESOLVED | spec树与治理 |
| ISSUE-081 | N-operand route identity 的完成度自相抵 | 待施工 | 发射器与架构 |
| ISSUE-082 | 「falsifier 组 CI 常绿」措辞失锚 | 待裁 | 门与工具 |
| ISSUE-083 | 真二进制门 DORMANT 的常驻登记处 | 阻塞 | 门与工具 |
| ISSUE-084 | 旧 C2 成本曲线第三家族点不再是柱二前置 | RESOLVED | 性能与测量 |
| ISSUE-085 | `tools/TOOLS.md` 已建 | 已修 | 门与工具 |
| ISSUE-086 | 前身证据索引含死指针 + T2 口径 stale | 已就绪 | spec树与治理 |
| ISSUE-087 | 门迁址遗留引用已施工，仍有判据级尾项 | 已施工 | 门与工具 |
| ISSUE-088 | 「对手类」一词同指两个闭合枚举（档位 vs 角色） | 待裁 | canon与措辞 |
| ISSUE-089 | `schema.def` 的 `$meta.authority` 悬空 vs [S-6] 形态哈希 | 待裁 | spec树与治理 |
| ISSUE-090 | cell harness 住址与契约已确定 | RESOLVED | 门与工具 |
| ISSUE-091 | bench 四元行键与签名已确定 | RESOLVED | 门与工具 |
| ISSUE-092 | 判定与对手档值域已确定 | RESOLVED | 门与工具 |
| ISSUE-093 | `experiments/runs.log` 已成为耐久台账 | RESOLVED | 门与工具 |
| ISSUE-094 | `--drift` 漂移门退役（守护对象已被新法废除） | 已退役 | 门与工具 |
| ISSUE-095 | 旧战役 goal（P/C/S/E 五线 + 验收门六条）退役成文站点 | RESOLVED | spec树与治理 |
| ISSUE-096 | bench cell harness 基建已建成 | RESOLVED | 门与工具 |
| ISSUE-097 | regime 已成显式闭合键，空值通配与含混行已退役 | RESOLVED | 门与工具 |
| ISSUE-098 | master writer 已收口为 recon-only；bench 只写 immutable run | RESOLVED | 门与工具 |
| ISSUE-099 | PR-47 真 gcc 污染 8 格：product_reduce 工具前置已闭；3 格待正式重测，5 个 FLAT gemm 路径待裁 | 待裁 | 门与工具 |
| ISSUE-100 | `vec_dot·nvfp4@rvv` 残余 lever 待施工，禁提前判不可达 | 待施工 | 性能与测量 |
| ISSUE-101 | canon 措辞：「判别键 = sub-block 数（16 vs 8）」被 P5-F2 证伪 | 待裁 | 性能与测量 |
| ISSUE-102 | 机制①「VLEN 专化满展开」反汇编证 no-op·真 lever = VLEN256 宽化·re-scope | 待裁 | 性能与测量 |
| ISSUE-103 | T-X 六列证据表 spec 层归属未定（重构后无 testing 层） | 待裁 | spec树与治理 |
| ISSUE-104 | scalar route/parser 已接通；ISSUE-061/roster 仍阻塞真测 | 阻塞 | 门与工具 |
| ISSUE-105 | per-board VLEN fixture 与 k1 半宽欠用已修 | RESOLVED | 门与工具 |
| ISSUE-106 | ★dequant「PASS」多是 lottery-PASS·R线 de-lottery 用 owned 真测替换会降 census PASS（诚实代价·§四.1） | APPLIED-DEFAULT | 性能与测量 |
| ISSUE-107 | ★grid-codebook dequant owned 真向量(HW-gather) 天花板·grid 族通用·标量门这些格到不了 PASS（非架构不可达·有标量-load lever gated 调度成熟） | 待裁 | 性能与测量 |
| ISSUE-108 | recon_t3_disposition.py 默认 reader 已切到 experiments/master | RESOLVED | 门与工具 |
| ISSUE-109 | q4_K vec_dot exact experimental campaign 已证伪并退役；性能格仍为具名-X | RESOLVED | 发射器与架构 |
| ISSUE-110 | T-X enablement 与 S1 scalar 性能账已裁准拆分 | 裁准拆分 | 门与工具 |
| ISSUE-111 | 数值口径松绑: headline 取最快变体+ULP 界·oracle §5 作废（补充令二§二·量税证数值非perf杠杆） | 待裁 | canon与措辞 |
| ISSUE-112 | 11 格 NEEDS-LEVER 具名-X 缺口 | 待施工 | 性能与测量 |
| ISSUE-113 | `integer_core_lmul` optional → required 收尾锁 | 待施工 | 发射器与架构 |
| ISSUE-114 | q4_K/q5_K vec_dot min-term 空心测漏洞 | RESOLVED（B2） | 门与工具 |
| ISSUE-115 | q2_K fold-brick op 的语义归属 | 待裁 | 发射器与架构 |
| ISSUE-116 | decode/flat 前门 stamp 与 6 处 live-default 去烘焙 | 待裁 | 发射器与架构 |
| ISSUE-117 | 柱二 c 轴公式、schema 与归因真实状态订正 | 待裁 | 性能与测量 |
| ISSUE-118 | GridCodebook/Ternary grid 几何描述符 lift | 回门待扫 | 发射器与架构 |
| ISSUE-119 | ForwardElementwise dequant-row 格式几何描述符化 | 待裁 | 发射器与架构 |
| ISSUE-120 | iq2_xs/iq2_s VLEN256 byte-broken 已修 | RESOLVED | 发射器与架构 |
| ISSUE-121 | 4 个 I8 证据线属性零消费的去留 | 待裁 | 发射器与架构 |
| ISSUE-122 | DequantMechanismPlan pre-emission 重算/验证（Codebook 已落，其余待续） | 部分落地 | 发射器与架构 |
| ISSUE-123 | evidence 地图从旧 C2/C3′ 标签重索引 | 待施工 | canon与措辞 |
| ISSUE-124 | q4_0 monolith 保留依据未闭合（abstract contraction 无真实 producer） | 待裁 | 发射器与架构 |
| ISSUE-125 | SP4 / loop-order selected stamp 与真实 emission 分裂 | 待施工 | 发射器与架构 |

---

## 四、姊妹层的「待裁 / 缺口」自设表 → 本层编号

其他层各自带有「待裁」或「缺口」表。**它们是本层的上游来源，不是第二登记簿** —— 本层给号，姊妹层按编号引用。副本张力的**首两例已按 `ISSUE-070`/`ISSUE-071` 裁定消解**（同事项的 canon「待裁」行随裁移除，只留本层一处）；其余行的去留仍按各自条目走。

| 来源 | 本层编号 |
|---|---|
| ~~[canon]「待裁」表 · C3′ vs C3 三贡献命名~~ | ISSUE-051 —— **已裁**：统一 C1/C2/C3，C3′ 退役 |
| ~~[canon]「待裁」表 · [PERF-1] 门数称谓~~ | ISSUE-071 —— **已裁**；该行已从 canon「待裁」表移除（副本消解） |
| ~~[canon]「待裁」表 · `core-invariants.md` 违「只写现行法」~~ | ISSUE-070 —— **已裁**；该行已从 canon「待裁」表移除（副本消解） |
| [canon](../canon/index.md)「待裁」表 · [S-4] uarch vs schema 准入 | ISSUE-045 |
| [canon](../canon/index.md)「待裁」表 · [PAT-3] 迁移判据可执行形态 | ISSUE-046 |
| [canon](../canon/index.md)「待裁」表 · 探针 DUAL-AGREE 归属 | ISSUE-047 |
| [architecture](../architecture/index.md) 「本层登记的未决项」表 未决项 **A**（「五层」所指） | ISSUE-074 |
| [architecture](../architecture/index.md) 「本层登记的未决项」表 未决项 **B**（既有 spec 树与新六层归并） | ISSUE-070（第 2 问）—— **未随 070 裁定了结，仍活** |
| ~~未决项 **C**（core-invariants 夹带叙事/commit 号）~~ | ISSUE-070（第 1 问）—— **已裁**：载体已归档，外壳随件离场 |
| ~~未决项 **D**（[PERF-1] 门项数）~~ | ISSUE-071 —— **已裁**：门体 = 十项 |
| [architecture](../architecture/index.md) 「本层登记的未决项」表 未决项 **E**（N-operand route identity） | ISSUE-081 |
| [evidence](../evidence/index.md) §六 **G-1**（性能证词根悬空） | ISSUE-072 |
| [evidence](../evidence/index.md) §六 **G-2**（「CI 常绿」措辞失锚） | ISSUE-082 |
| [evidence](../evidence/index.md) §六 **G-3**（承重论证载体去向） | ISSUE-072 |
| ~~[evidence] §六 **G-4**（主表目的地未落地）~~ | ISSUE-073 —— **已解决**：主表迁入 `experiments/master/` |
| [evidence](../evidence/index.md) §六 **G-5**（真二进制门 DORMANT） | ISSUE-083 |
| ~~[evidence] §六 **G-6**（旧 C2 曲线第 3 家族点）~~ | ISSUE-084 —— **已重分类**：不再是柱二前置 |
| [evidence](../evidence/index.md) §六 **G-7**（`tools/TOOLS.md` 未建） | ISSUE-085 |
| [evidence](../evidence/index.md) §六 **G-8**（前身索引死指针） | ISSUE-086 |
| [evidence](../evidence/index.md) 旧贡献标签重索引 | ISSUE-123（工件仍有效，贡献映射待迁移） |
| [evidence](../evidence/index.md) §六 **G-10**（旧 `tools/lint/` 反向引用） | ISSUE-087 |
| 旧强制停机/续跑规则 | ISSUE-075 —— RESOLVED |
| 旧第二套延后裁决状态机 | ISSUE-076 —— RESOLVED |
| 旧 `ROADMAP=` 强制前缀 | ISSUE-077 —— RESOLVED |
| 旧固定简报模板 | ISSUE-078 —— RESOLVED |
| 实验预注册与 task 强绑定 | ISSUE-079 —— RESOLVED |
| task 母子聚合作为全项目门 | ISSUE-080 —— RESOLVED |

---

## Quality Check

- **一条问题一个号**：立新条前查 [§三 全册索引](#三全册索引)；重复立条 = 缺陷，合并留一 + 链接。
- **零行号**：全册跨文件引用只用相对链接 + 标题锚 / 条目编号 / `grep` 谓词。行号出现 = 缺陷（[§〇.5](#5-跨文件引用禁行号)）。
- **条目自足**：`实质` 栏读完即可开工；靠 `出处` 才看得懂 = 缺陷。`出处` 逐条标「仓内谓词」或「历史线索（归档后不可达）」。
- **数字带谓词**：无谓词的数标 `UNVERIFIED-*`，且不得被引作权威计数。比值必钉分子与分母。
- **待裁在可安全推进时带保守默认**；若不同选择会实质改变论文主张、公共契约、外部动作或不可逆方向，则明确停在决策点。
- **阻塞必写前置**（他条 ISSUE 号 / 结构缺口 / 板 / 外部依赖），否则无法判可开工性。
- **科研主张不由 issue 发明**：重大研究表述由当前用户裁决与 canon 正本定义；issue 只登记冲突、迁移或施工状态。
- **是否继续由风险决定**：能在既定范围内安全推进就推进；需要新权限或会改变方向时停下询问，不设“禁止停止”规则。
