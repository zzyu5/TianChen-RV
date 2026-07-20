# ISSUES · spec 树与治理

> 本文件是 [issues 层](./index.md) 的条文文件之一。**编号规约 · 六字段 · 状态定义 · 可寻址判据全部住 [index](./index.md)，本文不重抄。**
> 收录范围：登记簿治理（ISSUE-066）· spec 树自身的缺口（ISSUE-070..074 · ISSUE-086）· governance 层移交的待裁（ISSUE-075..080）。
>
> **本文件的判据基线** = **全新 agent 只读 `.trellis/` 能从零上岗**；凡需翻 `docs/` / `CLAUDE.md` / 令文才看懂者 = 缺口（`docs/` 整体归档进 **git-ignored** 的 `_attic/`，克隆本仓者**拿不到**）。

---

## 一、登记簿治理

### ISSUE-066 · 事故档案与本册的边界
- **实质**：现行法令「事故档案原样迁入 Trellis 事故区（防复发资产 · 不去重不改写）」，与本册「唯一问题登记簿」的边界未明。典型件 = **增量构建恶性变种**：源码回退但已编译二进制未回退 → 幽灵符号仍在发射，而未提交的测试已改成期望新形态 ⟹ 测试绿在一个源码里**不存在**的发射器上 = 不会被发现的假绿；判决法 = 把可疑符号在输入里改名重跑，原名仍出现即证明烧在二进制里。该类【已修复但须防复发】的事项进事故区还是同时在本册立条，未定。
- **影响面**：本册与事故区的分工；此类条目现两处都没有稳定归宿。
- **卡在**：用户裁。
- **保守默认**：**已修复 + 防复发** → 事故区（不入本册）；**未修复** → 本册。本条即该边界的占位。
- **状态**：待裁
- **出处**：随令工项的承接（令文 §4.2.7）；令文归档后本条**自足**、不依赖其可达。

---

## 二、spec 树自身的缺口

### ISSUE-070 · `architecture/core-invariants.md` 违反「spec 只写现行法」，且它被判为必读权威全文
- **实质（裁定前状态 · 存证）**：现行铁律 = **spec 只写现行法，零历史叙事、零编号考古**（教训压缩成规则本身）。`architecture/core-invariants.md` 原全文**正面违反**，三类物证（**谓词复跑**：`grep -c 'd0826caa\|G7 全量令\|稻草人' (归档·git show pre-restructure-snapshot:.trellis/spec/architecture/core-invariants.md`) → **4 命中**。★**谓词已随裁定改指归档件**：活树同名路径现为**指针文件**、该 grep 恒 0；**禁用有界窗口命令核此类断言**）：
  - **编号考古 +「某年某月某人裁定」体**：`[K-10]` 的节标题即 `— 结构级 / 参数级判据（canon · G7 全量令 2026-07-13 用户裁 · [GOV-9] 提案+批准一体）`。
  - **commit hash 入正文**：`★G1 精化①/②（d0826caa · L2 实证…）` 两处 —— 规则本身（「绝对 spill 必须以部署编译器测」/「指令数子门是 regime-keyed」）是好规则，但被绑在一个 commit 号上。
  - **战役事故叙事**：`对手类机判（重编令二.2 · q4_K 稻草人事故制度化修复）` + 整段 `b29c269c 证伪` /「测错核之戒」/「双核分立限定（27658b8a · 五复核）」叙事。
- **★为何是上岗阻塞（不只是卫生）**：该文件**被两处 spec 判为新 agent 必读的「权威全文」** —— [canon](../canon/index.md) 「Guidelines Index」的架构不变量条目（「权威全文住 `architecture/core-invariants.md`…此处为规范性一句，**不重抄**」）与 [architecture](../architecture/index.md) 「本层地图」的硬规则条目（「全部正文住 `architecture/core-invariants.md`」）。⟹ 新 agent **被指着去读**一份夹带 commit 号与战役叙事的文件，而那些 commit 与战役档案正随 `docs/` 归档进 git-ignored 的 `_attic/` ⟹ **他读到的引用解析不了**。architecture 层 「本层登记的未决项」表已自认此事（C 行）并明写「**本文不改该文件 · 不代裁、不代改**」。
- **★裁决（2026-07-17 · 用户裁 · 本条 CLOSED）**：**收敛为单一权威本 + 另一本归档 + 引用改指**。落地形态：
  1. **权威本 = [canon · 核心不变量](../canon/核心不变量.md)**。裁决时的上岗级引用共同指向该文件；当前根 `AGENTS.md` 已退役，入口改为 `README.md` + spec 根地图，schema 的 invariant 指针继续指向 canon。
  2. **归档件 = `architecture/core-invariants.md` 原全文**（含其 commit 号 / 令文体 / 战役叙事）→ `_attic/`（plain `mv` · git-ignored · 记于 `_attic/ATTIC_INDEX.md`）。**铁律 1 违规随载体离场**，非靠逐句删改。
  3. **★原路径保留为指针（非删除）**：`schema/capability.schema.v1.json` 的 `$meta.authority` 钉住该路径的 `[S-5]` / `[F-2′]`（v1.2.0 sealed）；`ISSUE-072` 裁决材料 (b) **实测** 改锚 = `breaking` + 新 RFC。用户令明禁「touching sealed 资产」⟹ 全档不可行，**指针是唯一同时满足『单一权威本』与『不动 sealed 资产』的形态**。
  4. **前置验证（防丢法条）**：归档前逐条机核 **12 条附加硬规则在 canon 均有完整家**（[L-6]/[NG-4] → 成色与措辞 · [L-8]/[K-10] → 部署与构造语义 · [K-4]/[SEL-2] → 覆盖状态机与选择归因 · [S-5]/[F-2′]/[F-6]/[NG-3] → 能力模型与插件协议 · [VERIFY-LADDER] → 测量判据 · 对手类机判 → 对手与档位）⟹ **归档零法条损失**，只去重复本。
- **影响面（已结）**：新 agent 上岗可行性 —— 必读集合内不再有夹带 `_attic/` 死引用的文件。
- **遗留（不属本条）**：既有 spec 树与新六层的归并 / 归档关系（原须裁第 2 问）**未随本裁了结**，仍住 [architecture](../architecture/index.md) 「本层登记的未决项」表 B 行。
- **状态**：**已裁 · 已落地**（2026-07-17）
- **出处**：仓内谓词（上列 `grep -n`，可复跑）；承接来源 = [architecture](../architecture/index.md) 「本层登记的未决项」表 B/C 行。

### ISSUE-071 · [PERF-1] 门项数：「八门」vs「十项」
- **实质**：同一个门在 spec 树内**同时以两个项数流通**（谓词：`grep -rn 'PERF-1' .trellis/spec/`）：
  - **称「八门」**：[architecture/core-invariants.md](../architecture/core-invariants.md) 的 `[NG-4]` 正文（「在性能验收门 [PERF-1]（**性能验收八门**，科研目标总纲 §4.4）未全绿前…」）· spec 根 `index.md` 的 C3′ 证据门行（「…过 [PERF-1] **八门**」）。
  - **列「十项」**：[canon](../canon/index.md) §四 的门体自身**逐项列出十项**：① 字节精确（或声明 ULP 界）② VLEN 翻转 lit（128/256 双配置）③ 双板各一次 objdump 验封 ④ micro **且** e2e ⑤ 双板都验证 ⑥ 实验纪律 ⑦ 机制合成归因 ⑧ 措辞门（[L-1]）⑨ 账本合规（[L-9]）⑩ 部署身份五验（[L-10]）；**⑨⑩ 是前置：不满足则整门不予受理**。
  - **实质无歧义**：门体列十项，「八门」是简称；两者不指向两个不同的门，**不阻塞任何施工**。**但** `[NG-4]` 的「八门」还额外把权威指向 **`科研目标总纲 §4.4`** = `docs/` 下即将归档的文件 ⟹ 新 agent 顺着「八门」去查会走进 `_attic/`（git-ignored）。
- **★裁决（2026-07-17 · 用户裁 · 本条 CLOSED）**：**按实况订正称谓 · 一处定义、全树引用**。
- **★实况 = 十项（数门体本身得出，非两处文本互猜——互猜正是本 bug 的成因）**。**门体演进史**（谓词：读归档的三代总纲门体，逐项数）：
  | 载体 | 门体实列 | 自述称谓 |
  |---|---|---|
  | 总纲 v2 §4.4（`(归档·git show pre-restructure-snapshot:docs/canon/Weft-RV_科研目标总纲v2.md`）) | **①–⑧**（8 项） | 「**八项**全绿前」——**当时属实** |
  | 总纲 v3 §4.4（`(归档·git show pre-restructure-snapshot:docs/files) (2)/TianChen-RV_科研目标总纲v3.md`） | **①–⑧ + v3 增补两项前置 ⑨⑩** = **10 项** | 自相抵三称谓：「八项（十项）」· [L-1] 称「八门」· 附 B 称「**十门**」 |
  | **活 canon**（[测量判据](../canon/测量判据.md) §一.2） | **①–⑩**（10 项） | 现「门体 = 十项」 |
  ⟹ **门体从来只有一个**；「**八门**」= **v2 期化石简称**，在 v3 增补 ⑨（账本合规 [L-9]）/ ⑩（部署身份五验 [L-10]）**两条前置**后即失效，而 v3 未订正自身摘要行 ⟹ 三称谓并存。**v4 草案 §6.1 第 5 条已把「门数三称谓」登记为 v3 自身缺陷**（独立佐证：非本轮新说）。
- **落地**：
  1. **一处定义** = [canon · 测量判据](../canon/测量判据.md) §一.2（**门数只在此出现一次** + 称谓订正注）。
  2. **全树引用** = 一律 `[PERF-1]`、**禁带门数**（别处带数即漂移源）。
  3. 「八门」的两处活载体：`[NG-4]` / `[VERIFY-LADDER]` 的旧载体 `architecture/core-invariants.md` **已随 `ISSUE-070` 归档离场**（其「八门」还把权威指向已归档的 `科研目标总纲 §4.4` = 死指针，一并了结）；spec 根 `index.md` C3′ 行的「八门」**已去数**。canon 现役 [NG-4]（[成色与措辞](../canon/成色与措辞.md)）**本就不带门数**，无需改。
  4. **副本表**：[canon · 待裁](../canon/待裁.md) 原第 4 行（同事项）**已随裁定移除**，唯一登记簿原则恢复。
- **影响面（已结）**：`[PERF-1]` 全部引用面 + 唯一登记簿原则。
- **状态**：**已裁 · 已落地**（2026-07-17）
- **出处**：仓内谓词（`grep -rn 'PERF-1' .trellis/spec/`，可复跑）+ 归档三代总纲门体逐项计数；承接来源 = [architecture](../architecture/index.md) 「本层登记的未决项」表 D 行 + [canon](../canon/index.md)「待裁」表同事项行。

### ISSUE-072 · evidence 层三处「唯一权威」锚在 `docs/`（根悬空）
- **实质**：[evidence](../evidence/index.md) 是「三贡献 → 证据工件指针地图」，但其数处**唯一权威**锚点住在 `docs/` 下 —— 而 `docs/` 全体归档进 **git-ignored** 的 `_attic/` ⟹ **克隆本仓的新 agent 拿不到，指针整段失效**。该层已自设两条必裁项（住其 §六 缺口表，**按 G 编号寻址、不按行号**）：
  - **G-1 · 性能证词的根悬空**：`docs/reports/SEALED-WIN-REGISTRY.md` 是 **C3′ 性能证词的唯一权威登记**（Win #1 Win-K1-VLEN · Win #2 q4_K@k1 e2e prefill，逐字锁定措辞），且被列为**可达性根集合成员**、被判「留仓库」；但它**现住 `docs/` 下**，且**零脚本消费** —— 曾唯一与之接触的 `check_docs_canon.py` 也仅做文件名命名豁免、不读其内容，且该脚本**现已入 attic**（见 ISSUE-054⑥）⟹ 此刻全仓**无任何脚本引用它**。⟹「被脚本消费的数据」这条留仓理由**不被事实支撑**，**新家未定 → 整张地图的性能证词根悬空**。
  - **G-3 · 承重论证载体去向**：三贡献的承重**论证**主体住 `docs/reports/` 与 `docs/method/`（构造协议 + 摊销曲线 · 模式库综合 · 五负结果 · regime-split · 两 casefile · 三份接入实录 · C2 口径 · 归因锁 · ZERO-MODEL 法）。它们是**论证/报告**、非「被脚本消费的数据」→ 按可达性判据应归档；但**归档后本地图大半指针指向 `_attic`**。
    **★已出 G-3 范围（结论已迁入 evidence 层正文、不再依赖 docs 可达）**：五条 C3′ 负结果 → §4.7 · [F-1..F-6] → 工件映射 + [C1-SHAPE] 命名碰撞 → §2.2 / §2.1 · 相邻门 → §5.8。docs 侧原件**降为可选参考**（**禁再在 docs 留同内容副本**）。
  - 另 evidence 层 §2.1 / §2.2 把 **[C1-SHAPE] ↔ [F-1] 命名碰撞的唯一权威记录**与 **F → 工件的唯一映射索引**都锚在 `(归档·git show pre-restructure-snapshot:docs/method/FALSIFIER-INDEX.md`（同样归档去向未定）——) 惟此两项的**结论正本已迁入 evidence 层**（见上）。
- **须裁**：「迁入六层 / 事故区」vs「破例留仓」。
- **影响面**：evidence 全图的可用性 = C1/C2/C3′ 三贡献证据链能否被只读 `.trellis/` 的 agent 走通。
- **卡在**：与归档令的可达性判据正面耦合 = canon 级。
- **保守默认**：evidence 正文已逐条标「归档去向待裁」，**指针按现路径钉死、不预写尚不存在的新路径**；**禁 agent 自行搬迁或发明新家**。

#### ★ 2026-07-17 `docs/` 归档轮的处置（**执行了保守默认 · 未裁本条**）

`docs/` 已按令 §七③ / §4.2.7 归档（124 文件 → **67 attic + 51 事故区 + 6 留仓**；台账住 [`_attic/ATTIC_INDEX.md`](../../../_attic/ATTIC_INDEX.md) §三.2）。本条**未被裁**，其两问（G-1 新家 / G-3 承重论证载体去向）**仍全额待裁**。本轮对本条相关件做的是：

1. **G-1 的 `SEALED-WIN-REGISTRY.md` = 未动**（仍住 `docs/reports/`）。**理由不是「被脚本消费」**（本条已实测推翻该理由，本轮复核其四条谓词**全为真** —— `test -e tools/gates/check_docs_canon.py`=假 · `grep -rln SEALED-WIN-REGISTRY tools/ .trellis/scripts/ schema/`=**零命中**），而是：**§4.1.2 根集合逐字含「SEALED 登记」⟹ 它是根，而 attic 判据是「从根引用不到的一律入 attic」，根本身永不入 attic**；且本条保守默认**明令「禁 agent 自行搬迁或发明新家」** ⟹ 搬 = 自裁、归档进 git-ignored 区 = **正是 G-1 描述的伤害**、留在原地 = 唯一不预判的动作。
2. **G-3 的「两 Win 的 docs 侧证据腿与加固报告」= 未动**（4 份，随其根同留）。**该 4 份不是 agent 挑的** —— 是 `SEALED-WIN-REGISTRY.md:17/21/47` 的**证据指针逐字点名**的，按 §4.1.2「被引用的留下」机械可达。
3. **G-3 的 `docs/method/C2_marginal_cost_ledger.md` = 未动**：[architecture · 插件协议](../architecture/插件协议.md) §接入五件套交付物表把它列为「⑤ C2 ledger 行」的**落点**，并自带「★该文件归档去向待裁……**落点以裁定为准**」。
4. **G-3 的「★已出本条范围」部分 = 已按其自身授权归档**（该段逐字写「docs 侧原件**降为可选参考**，**随 `docs/` 归档即可**」）：`FALSIFIER-INDEX.md`（[F-1..F-6] 工件映射 + [C1-SHAPE] 命名碰撞，正本已在 evidence §2.1/§2.2；**G-8③ 的「已核实不破门」本轮复跑再确认** —— `check_family_locality.py` default 门 **exit=0**）· 三份接入实录 · C2 两 regime 口径 · 摊销规律 · q4_0 归因锁 · ZERO-MODEL 法 · 五条 C3′ 负结果。
5. **G-3 的两 casefile（[CASE-MICRO-E2E] / [CASE-COMPILER-ASYMMETRY]）= 迁入 [事故档案](../../事故档案/README.md)**。**这【不是】自裁 G-3 的「迁入六层/事故区 vs 破例留仓」** —— 而是**执行令文 §4.2.7 的明令**「事故档案**原样迁入 Trellis 事故区**（防复发资产，不去重不改写）」。令文即用户裁决，其对**事故档案子集**的去向已明定；G-3 剩余（非事故档案的承重论证载体）**不受影响、仍待裁**。
6. **未新增副本**：G-3 明令「**禁再在 `docs/` 留同内容副本**」—— 本轮零复制，全部为 `mv`。

**⟹ 本条状态不变（待裁）。** 待裁面**已收窄为**：① 那 6 件留仓件的最终新家（`docs/` 是否作为「数据/工件岛」长期存在，还是迁 `experiments/` 数据区 / evidence 层）；② G-3 中**未随事故区出清**的残余承重载体（如 `2026-07-10-C3-pattern-library-evidence.md` 模式库综合叙述，现已入 attic —— 若裁「破例留仓」须从 attic 取回）。
- **状态**：待裁
- **出处**：仓内 —— [evidence](../evidence/index.md) §六 缺口表 **G-1 / G-3** 行（该层在 `.trellis/` 内，可直接读到）。

### ISSUE-073 · 主表住址已统一到 `experiments/master/`

- **实质**：旧版 measurement 与 evidence 曾分别指向目标路径和迁移前路径。
- **裁决与现状**：迁移已完成；现役主表统一住 `experiments/master/`，原始运行住 `experiments/runs/<run-id>/`，台账住 `experiments/runs.log`。`experiments/active/result-tables/` 只可作为历史来源或旧工件位置，不再是正式写入目的地。
- **影响面**：runner、recon、evidence 指针和报告引用。
- **状态**：RESOLVED
- **出处**：仓内路径与 [measurement](../measurement/index.md) 的单一目的地法。

### ISSUE-074 · 「五层机器」所指未定（**禁 agent 发明**）
- **实质**：令文令 SPEC-ARCHITECTURE = 「五层机器」，现状/目标双栏分开写。但**「五层」这个表述在仓内无所指** —— architecture 层已就此立法三条（住其 「命名未决 · 「五层」的所指未裁」节与 「命名未决 · 「五层」的所指未裁」节）：**不定义「五层」、不承认它是既立术语、层内禁用该词指代架构**，并按**实际代码层次**组织；**该命名的所指须用户裁，禁 agent 发明** —— 裁定前任何 spec 把「五层」当既立术语使用**都属越权发明**。
  **相邻事实（同层「「五」已有既存所指」与「裸 `L<数字>` 编号禁用」两节）**：① 「五」已有既存所指（模板级「五大件」与单家族级「五件套」**不是同一个五**，且前者口语称「五」实为 6 件）⟹ 再造第三个「五」直接撞上既有消歧纪律；② 裸 `L<数字>` 在现行文本中**至少 4 个互不相关所指**（性能轴 / IR-后端轴 / 战役里程碑 / 战役队列优先级），且 IR 轴的 `L2`/`L3` **被使用但从未定义** ⟹ 就算想反推「五层」是哪五层，现有 `L*` 记号本身就是混叠的，反推必然是发明。
  > **本条禁写「某文件目前有没有『五层』字样」类断言** —— 那是时点敏感的，写下即开始腐坏，且把本条的正确性绑上兄弟文件的编辑状态。**裁的是词的所指，不是词的出现次数。**
- **须裁**：「五层机器」的五层各是什么（或：废除该命名，按 architecture 层现行的实际代码层次组织并改令文措辞）。**裸 `L<数字>` 编号的全局处置**一并裁（architecture 层 「裸 `L<数字>` 编号禁用」节 已禁用该记号，全局处置提案归本条）。
- **影响面**：architecture 层的组织骨架 + 一切 `L<数字>` 记号的可读性。
- **卡在**：**命名属 canon 级**（[GOV-9] 命名冻结）；且 architecture 层已明令**禁 agent 发明**。
- **保守默认**：architecture 层**维持按实际代码层次组织** + 标注该命名待裁，**不硬凑五层**；泛指该机器的结构定法时用替代说法（见 architecture 层 「命名未决」节 对照表），**禁 agent 自行指派五层所指**。
- **状态**：待裁
- **出处**：仓内 —— [architecture](../architecture/index.md) 「命名未决」节 / 「命名未决 · 「五层」的所指未裁」节 / 「「五」已有既存所指」节 / 「裸 `L<数字>` 编号禁用」节 与 「本层登记的未决项」表 A 行（该层在 `.trellis/` 内，可直接读到）。

### ISSUE-086 · 前身证据索引含已验证死指针 + T2 口径 stale
- **实质**：evidence 层的迁移底本（`(归档·git show pre-restructure-snapshot:docs/reports/2026-07-10-paper-evidence-index.md`）内) **4 处路径经核实不存在**，且其 T2 口径已 stale（按 `seq` 区间描述，与现行单分母制口径不符）⟹ **禁照抄该底本**。已核实的死指针 → 正解对照（**结论已迁入本条，不依赖底本可达**）：
  | 死指针 | 正解 |
  |---|---|
  | `include/TianChenRV/Plugin/RVV/RVVRepackTilingSelection.h` | `include/Weft/Plugin/RVV/RVVRepackTilingSelection.h`（改名后） |
  | `docs/canon/TianChen-RV_定位-v2.md` | `(归档·git show pre-restructure-snapshot:docs/canon/Weft-RV_定位-v2.md`) |
  | `docs/canon/TianChen-RV_执行总纲v2.md` | `(归档·git show pre-restructure-snapshot:docs/canon/Weft-RV_执行总纲v2.md`) |
  | `experiments/active/kquant-family-closure/transmission_account.md` | `experiments/archive/l1-kquant/kquant-family-closure/transmission_account.md`（已归档） |
  **阅读纪律（承接改名事实）**：改名前的历史记载用旧名 = 当时准确，**勿改**；改名后一律 `weft` / `Weft`。
- **影响面**：凡以该底本为源起草的证据表述。
- **卡在**：无 —— 底本随 `docs/` 归档即闭（**本条的正解对照已自足**）。
- **保守默认**：**禁引该底本**；需要「工件 → 路径」映射一律读 [evidence](../evidence/index.md) 正文。
- **状态**：已就绪（随 `docs/` 归档 + 本条正解对照落地即可关闭）
- **出处**：仓内 —— [evidence](../evidence/index.md) §六 缺口表 G-8 行 + §6.1 核实不存在的路径 表 = 本条的承接来源。

---

## 三、governance 旧强制流程条目的裁决（6 条）

> 2026-07-20 用户明确裁决：GPT 不必严格走 Trellis 流程；`.trellis/spec/` 是项目契约，task/queue 是按需使用的协作工具。下列六条因此按新的治理边界关闭。

### ISSUE-075 · 「停机规则」在本仓无 canon 定义站点

- **实质**：旧治理曾假设“禁止停止、只有检查点可停”，但这不是项目产品或科研不变量。
- **裁决**：删除代理强制续跑规则；当前请求完成即可交付，真正阻塞时说明缺少的选择、权限或外部条件。
- **状态**：RESOLVED
- **出处**：[governance](../governance/index.md) 与当前用户裁决。

### ISSUE-076 · 延后裁决登记的载体与 schema

- **实质**：旧治理同时维护两套状态枚举，增加了不必要的协议负担。
- **裁决**：已知项目缺口继续进 `issues/`；临时判断可直接写进相关 spec、实验计划或任务说明，不再要求第二登记册或七值状态机。
- **状态**：RESOLVED
- **出处**：[governance](../governance/index.md) 与当前用户裁决。

### ISSUE-077 · 队列节点身份前缀 `ROADMAP=`

- **实质**：`ROADMAP=` 是旧战役编排遗留字面，不应成为普通开发任务的强制身份协议。
- **裁决**：不再规定该前缀。若某次战役主动采用 task tree，可在该战役内自定稳定标识。
- **状态**：RESOLVED
- **出处**：[队列与简报](../governance/队列与简报.md)。

### ISSUE-078 · 强制简报头条与机算宿主

- **实质**：旧治理要求每次简报都携带固定头条三数和黄格分类，但它与多数局部开发请求无关。
- **裁决**：简报按当前任务风险和用户需要提供；论文数字与性能数字仍必须来自可复算工件，普通开发状态不强制套固定模板。
- **状态**：RESOLVED
- **出处**：[队列与简报](../governance/队列与简报.md)。

### ISSUE-079 · 预注册无 `task.py` 工具位

- **实质**：测量预注册有价值，但不应等同于“一切工作必须挂 Trellis task”。
- **裁决**：正式实验仍须在首次测量前固定 cell、板、对手、正确性门和判定规则；载体可为 task PRD、独立实验计划或 runner manifest。普通代码/spec 修改无需预注册。
- **状态**：RESOLVED
- **出处**：[trellis 卫生](../governance/trellis卫生.md) 与 [measurement](../measurement/index.md)。

### ISSUE-080 · task 母子状态无自动聚合

- **实质**：task tree 现为可选协调工具，母子状态自动聚合不再是整个仓库的质量门。
- **裁决**：采用 task tree 的战役自行维护一致性；不为可选工具强加全项目脚本前置。
- **状态**：RESOLVED
- **出处**：[governance](../governance/index.md) 与当前用户裁决。

---

### ISSUE-089 · `schema.def` 的 `$meta.authority` 指针已悬空，但改它会撞 [S-6] 形态哈希

- **状态**：**待裁**（canon 级 · 禁 agent 自裁）
- **实质**：`schema/capability.schema.v1.json` **就是 `schema.def` 本体**。其 `$meta.authority`
  指向的 spec 路径在六层收口后**已不存在**（旧 `capability-model/` 层已归档）⟹ **悬空锚**。
  但 **[S-6] 的冻结哈希 = `sha256(json.dumps(obj, sort_keys=True))` 覆盖【整个对象，含 `$meta`】**
  ⟹ **仅改这个文档指针就会让形态哈希漂移**，`schema-def-f2prime-gate-redteam` 当场变红（实测 950/4）。
- **为何不能自决**：修复须**同时 bump [S-6] 版本 + 更新 VERSIONLOG** = **canon 级动作**。
  且 `schema.def` 自第二家族起未被接入触及是 **C1 的承重证据**（[F-2′] 逐 PR 审计）——
  动它的哈希需要用户明示。
- **保守默认（现行）**：**指针保持悬空**，`capability.schema.v1.json` 与 HEAD **字节相同**
  （施工代理已回退其 2 处编辑并自证）。**这是全树【唯一】残留的悬空路径**，其余 39 处已改锚。
- **两个候选去向（供裁）**：
  ① 改锚 + bump [S-6] + 更新 VERSIONLOG（一次性，之后哈希稳定）
  ② 把 `$meta.authority` 移出哈希域（改 [S-6] 哈希口径 = 更深的 canon 变更）
- **相邻事实**：本条与 [ISSUE-070]（I1–I9 双本）、[ISSUE-071]（[PERF-1] 门数称谓）同属
  「六层收口显形的 canon 级待裁」，建议一并裁。

#### ★裁决材料（2026-07-17 实测 · 三条推翻了「bump 一下就了断」的直觉）

**(a) `$meta.authority` 四条中【两条】悬空，且其一【早于本次重构】**（判决性：读 `pre-restructure-snapshot` 标签）：

| 条目 | 快照里 | 现在 |
|---|---|---|
| `capability-model/capability-contract.md#S-5` | ✓存活 | ★悬空（**本次重构归档所致**） |
| `architecture/core-invariants.md#S-5` | ✓存活 | **✓存活** |
| `architecture/core-invariants.md#F-2prime` | ✓存活 | **✓存活** |
| `docs/Weft-RV_科研目标总纲v2.md#S-5` | **★快照里就是死的** | ★悬空（真路径缺 `/canon/`） |

⟹ **悬空条件不是重构回归**：该 schema **本来就带着一条死指针在出货**，无人发现。
⟹ **S-5 仍有活权威**（`core-invariants.md` 在六层内）⟹ **保守默认（不动）不丢失任何东西**。

> **★ `ISSUE-070` 裁定后的交叉订正（2026-07-17 · 本表结论不变、更强）**：`core-invariants.md` 现为**指针文件**（原全文已归档 `_attic/`），其 `[S-5]`/`[F-2′]` 条目**仍在、仍解析**（故上表两行「✓存活」**不变**），只是内容改为**去向**。而 `[S-5]` / `[F-2′]` 的**法条正本** `canon/能力模型与插件协议.md` 与**字段级全文** `architecture/能力模型.md` **本就已在 `$meta.authority` 之内**（v1.2.0 由 ISSUE-089 一并锚定）⟹ **「不动 schema」的保守默认在 070 落地后依然成立且更稳**：即便将来把这两条指针条目删掉，S-5 的活权威仍有两条独立在案。**070 落地实测：`report --check` exit=0 · redteam 8/8 · `[S-6]` 零 bump。**

**(b) ★「改锚 + 版本 +1」会被分类器判 `breaking`，不是 minor**（实测 `classify_schema_change(旧,新) = breaking`）：
`$meta.authority` 是**标量字符串列表**，分类器按「SCALAR list-member (enum) add/remove → breaking」处理
⟹ 改任一条字符串即触发。而 VERSIONLOG 定义 `breaking` = 「major · **requires a new RFC**」。
**⟹ 代价 = breaking 评级 + 新 RFC，且审计链留下「v2: breaking」一行 —— 外部读者会读成
「schema 形态发生过破坏性变更」，而真相只是挪了个文档指针。这【反噬】C1 叙事，与改锚的初衷相反。**

**(c) VERSIONLOG 自身的设计文档区分了两类门**（这条支持"不伤 C1"，但不解决 (b)）：
> *Not to be confused with the per-PR operation gate ([F-2′])：a **family-onboarding** PR whose diff
> intersects `schema/` is the falsifier firing. This log grades **core-author evolution** PRs,
> which are a different PR class.*

⟹ C1 的承重证据是 **[F-2′] 操作门**（家族接入 PR），**[S-6] 报告门**管的是 core-author 演进 = 另一类。
文档指针迁移属后者 ⟹ **bump 本身不伤 C1 的"家族接入未触 schema"主张**；伤的是 (b) 的**评级措辞**。

**(d) ★★真缺陷可能不是「指针错了」，是「没有门检查指针活没活」**：
一条死指针（`docs/Weft-RV_...` 缺 `/canon/`）在库里躺到今天无人知，**因为没有任何检查验证
`$meta.authority` 的存活性**。若立此门，(a) 的两条会当场现形，且此后不再复发。
**但新增检查的唯一途径 = 提案入 ISSUES → 用户裁**（measurement 层「门体系」§3.5.4 自己的法）⟹ **本条即该提案**。

**⟹ 修订后的候选去向（供裁 · agent 不自裁）**：
- **①′ 改锚 + bump + 承受 `breaking` 评级 + 出新 RFC** —— 一次性了断，但审计链留「breaking」一行（措辞反噬）
- **①″ 改锚 + bump + 评级另立 `editorial` 档** —— 需先改 VERSIONLOG 的评级口径（canon 级）
- **② 把 `$meta` 排出哈希域** —— 一劳永逸，但**给未来的静默改动开口子**（`authority` 可被悄悄重定向而封条不响）
- **③ 维持悬空 + 立【authority 存活性门】** —— 承认 (a)(d)：悬空非本次所致、S-5 仍有活权威；
  把力气花在**让这类病以后能被机器抓到**，而非追指针
- **④ 保守默认（现行）**：不动，`capability.schema.v1.json` 与 HEAD 字节相同

---

### ISSUE-095 · 旧战役 goal（P/C/S/E 五线 + 性能验收门六条）退役无成文站点

- **状态**：**RESOLVED（2026-07-18·《测试与收尾总令-开测篇》§〇.5 明裁）** —— 旧 P/C/S/E goal 及验收门六条**正式退役**，其未竟目标已并入开测篇五线（K/S/R/E + 终审）。本条留档为「令冲突解析法」（[决策权限卡](../governance/决策权限卡.md) §〇·补）的判例锚。
- **实质**：一份**旧战役 goal**（"canon v4 双流合订 → P 性能收尾 / C 能力发射重构 / S 标量收口 / E 论文证据工件"
  + "性能总验收门六条"）仍被 Stop hook 逐轮引用作停止条件。但它**与现行令直接冲突**：
  《大重构总令-最终版》§开篇明写「**纯结构与工具轮：零正式测量、零 kernel/发射器功能改动**」「**测量在重构收口、
  用户另下测试总令后才重启**」，§一.1「ROADMAP.md 与 docs/ 全体**不再使用**」。
  **旧 goal 的第 2 项「性能总验收门」需要正式测量 ⟹ 与「本轮零正式测量」不可同时满足。**
- **机核**：`grep -rl '五线退役|旧战役.*退役' .trellis/spec/` = **0**；ISSUES 无此冲突条目 = **0**
  ⟹ **退役无成文站点** ⟹ 每个新 agent / 每次 hook 都会拿旧 goal 来量、都困在同一矛盾里（本役已困 5 轮）。
- **保守默认（现行）**：**以《大重构总令-最终版》+ 用户逐条裁决为现行法**；旧 goal 的五项验收
  **是【测试总令之后】的目标态**，非本轮欠账。本轮交付 = 令文 §七「交付六件」，**已齐**（见 `.trellis/事故档案/2026-07-17-大重构收口简报.md`）。
- **建议裁法（供裁）**：① 确认旧 goal 退役、以现行令为准 → 本条 RESOLVED；
  ② 或明示旧 goal 仍活、要求本轮即测 → 则须同步**撤销「本轮零正式测量」**（二者不可兼得）。
- **★这是一条「规则不对自己生效」的元实例**：与本役另三起同源（`--drift` 门 / 「禁归并」指令 / 版本载体）——
  **一条停止条件若无「它已被后令取代」的成文记录，就会永远拿过时的标准量新工作**。

---

### ISSUE-103 · T-X 六列证据表的 spec 层归属未定（重构后无 testing 层）

- **状态**：**待裁**（**判据级** —— 「T-X 定义该落 measurement 还是 evidence」须先答，非对齐既有事实）
- **实质**：B4 案头稿把 T-X 六列证据的**定义**指向 `.trellis/spec/testing/mlir-testing-contract.md` §T-X；
  但六层重构后 spec **无 testing 层**（[index](./index.md) 六层 = governance/canon/issues/architecture/measurement/evidence）。
  T-X = X-SCALAR enablement 的六列证据格（列1 目标身份 · 列2 构建 · 列3 零向量机检 · 列4 byte-correct ·
  列5 逐竞品产出数 · 列6 参考路径披露），**域 = enablement·非胜负账·[L-6] scalar 永不作贡献基线**。
  候选归属层：**measurement**（它证「零向量 + ZERO-MODEL byte-exact」= [正确性门](../measurement/正确性门.md) 的
  X-SCALAR 特例）或 **evidence**（enablement 证词·连 [ISSUE-062] C2 摊销曲线第 3 独立家族点）。
- **保守默认（现行·已生效）**：T-X 证据表**暂落 `experiments/` 数据格·活证据**
  （`experiments/active/g8-stage3-attack/A3-xscalar-rv64gc/TX-six-column-evidence.md`·本 task 已更新为
  **真硅 (a)物理 no-V 态** —— 超锐板 verified isa 无 v/zve·去 B4 稿「窄豁免」标）；
  **定义指针待裁·不擅建 spec testing 层/文件**（[决策权限卡](../governance/决策权限卡.md) §〇 判据级）。
- **候选裁法（供裁·不代裁）**：① measurement 层加「T-X 证据格」定义（作正确性门的 X-SCALAR 落点）；
  ② evidence 层收（enablement 证词）；③ 保持纯 `experiments/` 活证据·无 spec 定义站点（则 spec 侧只留一条指针）。
- **连带 · B4 稿 PR-1「采购物理 no-V 板」= 重开·事实更正态**（W1 载体批 §2.3.3·裁1 已裁·2026-07-20 落）：
  **改写依据 = 事实「手头无物理 no-V 真硅」已为假**——超锐(scalar)板已**到货**且 verified isa
  `rv64imafdch_zicntr_zicsr_zifencei_zihpm_zaamo_zalrsc_zca_zcd`（无 v/zve·clang-18 装讫）= 真物理 no-V 硅。
  ⚠**措辞铁线**：板是**到货**的不是**采购**的·**不推翻「采购否决」那条硬冻结**（采购仍否决·只是无需采购即有真硅）；
  铁线「validated on silicon」字面成立。**⟹ 落盘完成·下游两条 canon 订正自动解锁进待裁**（agent 不自决·下轮裁）：
  ①**词表订正**（scalar「永不作贡献基线」动分档口径）②**铁线4 订正**（「窄豁免」→「物理 no-V」·牵动 T-X 旗舰措辞）。
  三份 W1 FINDING 已 tracked（`experiments/active/r5.1-w1-carrier/{L3L4-scalar-silicon,scalar-noV-fourway,rvv-Aprime-dualVLEN}-FINDING.md`·`git ls-tree -r HEAD | grep -c FINDING`=8≥4）。
- **出处**：本 task `.trellis/tasks/07-18-07-18-s-scalar-tx-s1`（S 线·scalar 真 no-V 硅 T-X 收口）。
