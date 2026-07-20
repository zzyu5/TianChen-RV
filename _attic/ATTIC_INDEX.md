# ATTIC_INDEX — 归档索引

> 大重构 §4.1：一切"重复/过时/存疑"**移入 `_attic/` 而非删除**，按原路径结构存放。
> `_attic/` 本身 git-ignored；**本索引在库内**。
> **纯派生物例外**（§4.2.3/§4.2.4）：worktree 缓存与 build 目录**直接删除、不 attic**。

**保留判据 = 可达性**（§4.1.2）。根集合 =
{六份 spec ∪ `experiments/master` ∪ SEALED 登记 ∪ 现役 runner/oracle/gates ∪ ISSUES ∪ 现役测试套件}。
从根引用不到的一律入 attic；被引用的留下并标注来源。

---

## 一、直接删除（纯派生物·不 attic·令文明令）

| 原路径 | 原因 | 日期 |
|---|---|---|
| `.worktrees/` (1.3G) | §4.2.3。内含 `cache/<commit-sha>/` ×12 = 并行基建的 pinned-commit 缓存。**核实：git 注册 worktree 数 = 0；12 个缓存目录含未提交内容 = 0**。纯派生物，工具 `tools/bench/new-line-worktree.sh` 按需重建。 | 2026-07-17 |

## 二、查证后【保留】（记录结论，防重复调查）

| 路径 | 结论 | 依据 | 日期 |
|---|---|---|---|
| **★同病第 4 例：`tools/visibility/t1d_dual_instance.py`（2026-07-17 §4.2.8 工具册线补记）** | **修正而非归档** —— 上一行称「3 门」**已修完**，实为**漏网 4 例**：本器的候选表 `("build-weft/bin/weft-opt", "build-demo/bin/weft-opt", "build/bin/weft-opt")` **三者今皆不存在** ⟹ 本器恒 `SETUP ERROR`、其产物 `experiments/active/result-tables/T1d_dual_instance.csv` **恒不可重生**（该表自述「nothing here is hand-written」，却已无法由机器重写 = 事实上退化为手工表）。**已按三门同法修**：`$WEFT_BUILD/bin` → 唯一布局 `build/weft/bin`。**负控实证**：修前 `python3 tools/visibility/t1d_dual_instance.py` → `[t1d] SETUP ERROR: no built weft-opt found`；修后 `--check` 能真正跑起来并报出 `CHECK FAIL: live run diverges`（= 从「跑不动」变成「跑得动且如实报漂移」）。**★该表的重生【未执行】**：其 diff 不止路径，还含 `69 core files → 71` + 三枚 pin 刷新 = **C1 结构证据的数据变更** ⟹ 不得以「修路径」名义顺手改证据数字，已登记 ISSUE-087 判据级②。**教训**：「同类已全修」类断言须对**全仓**复跑谓词，不能只数当时手上那几个。 | 2026-07-17 |
| `build/bin` 发现路径（3 门） | **修正而非归档** —— §4.2.4 布局归一的机械后果 | 门的 binary 发现候选原为 `$WEFT_BUILD/bin` → `build/bin` → `build-weft/bin`。**预存缺陷**：`build/bin`（陈旧 07-12 构建）排在真正在用的 `build-weft/bin` **之前** ⟹ 门一直在拿**陈旧 binary** 当证据。布局归一到 `build/weft/` 后两者皆无 ⟹ 退化为**静默 SKIP 返回 0**（更坏：绿得无声）。**已修**：候选首位改为唯一布局 `build/weft/bin`，删除已不存在的 `build-weft`。**负控实证**（f5/f4/f6 三门）：挪走 `weft-opt` + `--require-binaries` → **exit=2 全部变红**；恢复后复绿。 | 2026-07-17 |
| `docs/` **6 件留仓**（`reports/SEALED-WIN-REGISTRY.md` + 其 4 条证据腿 `2026-07-10-vlen-adapt-m1-k1-sealed-win-candidate.md` / `2026-07-10-k1-seal-e2e-transduction.md` / `2026-07-10-Win-K1-VLEN-加固报告.md` / `2026-07-16-G8-paper-evidence-freeze.md` + `method/C2_marginal_cost_ledger.md`） | **保留** —— **不是我的判断，是 §4.1.2 可达性的机械结果 + 已登记必裁项的保守默认** | **① SEALED 登记 = 根集合成员**：§4.1.2 根集合逐字含「**SEALED 登记**」，而规则是「从根引用不到的一律入 attic；**被引用的留下**」⟹ 根本身永不入 attic。§一.2 亦逐字列其「**留在仓库**」。**② 4 条证据腿 = 从该根可达**：`SEALED-WIN-REGISTRY.md:17/21/47` 的**证据指针**逐字点名这 4 份（Win #1 的 1.085× CI + 五验 + hand-brick 对手身份 + 加固报告；Win #2 的三点隔离 + 1.101× 卷）——**这不是我挑的，是登记册自己点的名**，恰好就是 evidence 层 **G-3 所称「两 Win 的 docs 侧证据腿与加固报告」**。归档它们 = 让 C3′ 唯一权威登记的证据腿指进 git-ignored 区 = **正是 G-1 描述的那个伤害**。**③ C2 ledger = 活落点**：`.trellis/spec/architecture/插件协议.md:245` 列其为接入五件套「⑤ C2 ledger 行」的**落点**，并自带「**★该文件归档去向待裁……落点以裁定为准**」；另被 `schema/family-manifest.v1.json`(3) · `family-dirs.v1.json` · T2 CSV provenance(13) 引。**④ 保守默认由 ISSUE-072 逐字规定**：「指针按现路径钉死、不预写尚不存在的新路径；**禁 agent 自行搬迁或发明新家**」；须裁项 = 「迁入六层 / 事故区」vs「破例留仓」。⟹ **搬 = 自裁 · 归档 = 自裁 · 留在原地 = 唯一不预判的动作**。<br>**★ 特别记：留仓理由【不是】「被脚本消费的数据」。** 令文 §一.2 用该理由留 SEALED，但 **G-1 已实测推翻它**：「**零脚本消费** …… 曾唯一与之接触的 `check_docs_canon.py` 仅做文件名命名豁免、不读其内容，且该脚本现已入 attic」。**本轮独立复核 G-1 四条谓词全部为真**：`test -e tools/gates/check_docs_canon.py` = 假（已在 `_attic/tools/lint/`）· `grep -rln SEALED-WIN-REGISTRY tools/ .trellis/scripts/ schema/` = **零命中** · `test -e .github/workflows` = 假 · `check_family_locality.py` default 门 exit=0。⟹ 留仓的**真**理由 = 上述 ①②③④，**不是** §一.2 那句话。 | 2026-07-17 |
| `docs/PENDING_RULINGS.md` 48 条的落位（§〇.3「已知问题零丢失」） | **零丢失·可归档** —— 48 行**逐条**核毕 | **计数订正**：登记表**真值 = 48 行**（`PR-1`…`PR-46` + `PR-48` + `PR-49`），但全文含 **49** 个 `PR-NN` 记号 —— **`PR-47` 是幽灵**：**无登记行**，只在 `PR-49` 行正文与 `2026-07-17-收束-推进与理解.md:126`「PR-47 **待写**」里现身。**issues 层已自行抓到并承接**（`性能与测量.md:32`：「出处：历史线索 = PR-47（**旧册无此行，本条即其承接**）」）。<br>**48 行分两类，各自可寻址**：**(a) 31 行被 issues 层按 `PR-NN` 直接引**（`PR-3` · `PR-18` · `PR-20`…`PR-46` · `PR-48` · `PR-49`）—— 恰好是全部 `APPLIED-DEFAULT`/移交（**真未决**）者。**(b) 17 行未被引**（`PR-1,2,4..17,19`）—— 恰好是全部 `RESOLVED*` 者，其**裁决已成现行法**，逐条抽验落地：`PR-2`双 regime→`canon/暂定-科研主张`·`PR-4`[I7]→`architecture/core-invariants`·`PR-5`instance-hash→`architecture/变体流水线`·`PR-6`物理墙收严→`canon/缺口与认输`·`PR-7`三档词表→同·`PR-9`双账本→`canon/测量判据`·`PR-10`parity-by-roofline→`canon/成色与措辞`·`PR-11`dequant 标量类→`measurement/对手法`·`PR-12`便宜档→`issues/性能与测量`·`PR-14`Win#2→`evidence/三贡献证据地图`·`PR-15`selection_valid→`canon/覆盖状态机与选择归因`·`PR-17`单世界 clang→9 文件·`PR-19`→`issues/发射器与架构`（**13/14 命中 spec**）。**唯一未命中 spec 的 `PR-16`（`deployed_point` 复合主键）落在【数据】而非文档** —— 实存于 `schema/measurement-memory.v1.json`，**符合 §一.2「schema = 被脚本消费的数据、不是文档」**，非丢失。<br>**两条 RESOLVED 仍带残留、均已在册**：`PR-8` 的 runner 三 gap → `issues/门与工具.md:65` **逐字承接**（「(1) 注册 token 需 repo-admin（无采购）；(2) 无官方 riscv64 runner 二进制；(3) 板上构建需板构建 = 同硬冻结」）· `PR-1` 的 X-SCALAR 窄豁免灰区 → **已升格为现行法** `canon/测量判据.md:108`（「**采购 = 否决**……在飞路径**只有** (b) 窄豁免，且**每次**须显式标注」）+ `evidence/三贡献证据地图.md:64` 的 `narrow-exempt: V-board-run-as-noV` 工件指针。 | 2026-07-17 |
| `.touch-set/` | **保留** —— 是**活基建**，非僵尸 | §4.2.2 要求"查来历 + 是否被现役引用"。**来历**：由"纪律机械化 A1+A3 — touch-set CI + 禁 git add -A + worktree 政策"引入。**现役引用（代码级，非散文）**：`tools/hooks/pre-commit`（打印那条 NOTE 的就是它）· `tools/ci/check_commit_touchset.py`（检查器本体）· `tools/bench/new-line-worktree.sh`（写 `ACTIVE` 标记）· `.gitignore`（忽略 `ACTIVE`）。**作用**：`{commit 触碰的文件} ⊆ {线声明的 globs}`，可真变红。**排查注记**：用 `grep touch-set` 会匹到散文用法（"3-file touch-set"）= 假阳性；真引用须用 `\.touch-set` 或 `TRELLIS_TOUCH_SET_LINE`。 | 2026-07-17 |

## 三、移入 attic

| 原路径 | → attic 路径 | 一句话原因 | 日期 |
|---|---|---|---|
| `.github/workflows/` (4 yml) | `_attic/.github/workflows/` | §4.2.1 明令删除：runner 缺口从未闭合、名不符实。**实证**：`falsifier-binary-gate.yml` 要求 `[self-hosted, weft-build]` runner（自述 DORMANT·从未供应）；`dir-hygiene.yml` 的三门在本地 HEAD 就是 RED（layout 510 findings / index 1061 issues）——若 CI 真在跑，PR 早已全红。四份 attic 副本经 sha256 逐一验与原件字节相同后才 `rm`。門本体已按 §4.2.1 迁 `tools/gates/`。`.github/` 除 workflows 外**空**（`find .github/ -mindepth 1 -not -path '.github/workflows*'` = 0），故整目录移除。 | 2026-07-17 |
| `tools/lint/check_docs_canon.py` | `_attic/tools/lint/check_docs_canon.py` | **判死 §3.5.3 条款①「文本里有没有某字样式验收」**——本体是禁词表 `FRAMING_BANNED = ("实质全收口","实质胜利","毛刺级","大成","publication-grade","capstone")` + `SLOGAN_PATTERNS` 正则。**负控实证（字样defeat）**：探针文件写「本轮达成实质胜利。」→ 命中 `[实质胜利]`；改写成**语义相同（更强）**的「本轮已全面取胜、再无欠账、可宣告收官。」→ **零命中**。同一主张换词即过门 = 该门量的是措辞不是事实，正撞 §3.3.2「验收查世系不查字样」。次因：其标的 `docs/` 按 §4.2.7 整体归档，主体将不存在。 | 2026-07-17 |
| `tools/lint/check_experiments_data_only.py` | `_attic/tools/lint/check_experiments_data_only.py` | **判死 §3.5.3 条款④「各轮重复的检查脚本（合并留一）」**——**非门，是 22 行 deprecated shim**：自述 `SUPERSEDED (org STAGE2, 2026-07-06)`，全部内容 = `import check_experiments_layout as layout; sys.exit(layout.main(sys.argv[1:]))`。零自有判断逻辑，与 `check_experiments_layout.py` 输出逐字相同（baseline 两者末行一致，实证）。留一 = `check_experiments_layout.py`（已迁 `tools/gates/`）。 | 2026-07-17 |
| `tools/lint/check_manifest.py` | `_attic/tools/lint/check_manifest.py` | **判死 §3.5.3 条款④（同上·合并留一）**——**非门，是 23 行 deprecated shim**：自述 `SUPERSEDED (org STAGE2, 2026-07-06)`，全部内容 = `import check_index_consistency as idx; sys.exit(idx.main(sys.argv[1:]))`。零自有判断逻辑。留一 = `check_index_consistency.py`（已迁 `tools/gates/`）。 | 2026-07-17 |
| `tools/rename_to_weft.py` | `_attic/tools/rename_to_weft.py` | **§4.2.8「写不出『被谁调用』即 attic」+ [trellis卫生 §6.3](../.trellis/spec/governance/trellis卫生.md) 门退役判别式「它守的东西还在不在」= 【不在】**。本体 = `[RENAME] 裁四`（2026-07-12）的**一次性机械改名迁移工具**（六 token 映射 TianChen-RV/TCRV → Weft）。**死因三条，逐条实测**：① **改名已完成**——`fc72fe53f`「全仓机械改名 · byte-exact 零漂移 · CI 绿 · MOVES 158」；② **标的已不存在**——`git grep -c 'TianChen-RV\|TianChenRV\|TIANCHENRV\|tianchenrv' -- lib include test CMakeLists.txt` = **零命中**（其六 token 在功能面已无一存活，再跑即 no-op）；③ **零引用**——`python3 tools/gates/gen_tools_index.py --zero` 列其为零引用者（全仓除自身外无任何引用）。**且不属 §4.2.8 正面清单**（非 runner / oracle / gate / 必要构建脚本），亦**非按需门**（它不守任何东西，是迁移工具）。**可验地址**：`git show pre-restructure-snapshot:tools/rename_to_weft.py`（**已实跑 `git cat-file -e` 确认可解析**）。★**残留历史旧名是【对】的、勿再跑本工具"清理"**：其 docstring 明载 `.trellis/backup|tasks|workspace`、`docs/reports`、`artifacts/`、`experiments/`、`tools/e2e-harness`、`tools/bench` 为**具名例外**——旧名在当时**事实正确**，改写即毁审计链。 | 2026-07-17 |
| `2026-07-18-项目行动书-r3.md`（根目录·18173B） | `_attic/2026-07-18-项目行动书-r3.md` | **过时令归档·用户明令（2026-07-19「准备下发新行动」）**——r3 行动书（A线律2重构 + B线三块地公式墙必攻 + 交付三项纪律）**已执行完主体**：B线第一块 11 格 dequant de-lottery 完成（标量类-rvv 37→40·+3真地盘）· B线第二块 4 公式墙 vec_dot 全 attacked（3 deployed 地盘 iq2_xxs@rvv+tq1_0@rvv+@k1·手调-rvv 8→10·手调-k1 10→11·iq2_xs 板上证伪维持具名-X）· A线 census→律2 三角色（ISSUE-117/118/119/120 登记·回门待裁）。被 2026-07-19 新行动取代。磁盘保留可查。 | 2026-07-19 |
| `测试与收尾总令-开测篇.md`（根目录·8124B） | `_attic/测试与收尾总令-开测篇.md` | **过时令归档·用户明令（同上）**——《开测篇》5 停止条件战役令（S线 T-X 达成 · K/R 机制证毕 · 测量通道 ISSUE-090/091 待裁）·其战略态已并入 memory `opentest-campaign-strategic-state.md`（r3 阶段 + 第二块闭环 + 方法学）。被新行动取代。磁盘保留可查。 | 2026-07-19 |
| `2026-07-19-行动书-r5.1.md`（根目录·30337B） | `_attic/2026-07-19-行动书-r5.1.md` | **过时令归档·用户 2026-07-20 收尾令**——r5.1 行动书（缺管道纠偏 + A线真重构接管道 + B线继续 + 五路并行调度令 + §四.5 B线三闸）**主体已执行**：五路真并行 done（W1 拔管道判决3/3·W2 [GAP-P1] width选择器闭式2/2·W3 逐条判并做·W4 扩iq2 3入账·W5 dequant收割 grid全族翻45/46）· 两层诚实审计@HEAD（能力层~1.5/5→②③✓·公式层1/5→dequant-row head 5/5·5家全进 plan）· 能力层 version管道+vreg_count正名· DequantMechanismPlan 5家闭环。战略态在 memory `r51-exec-grid-flip-gapp1-formula-migration.md` + 活账本 `experiments/active/formula-layer-migration/LEDGER.md`。磁盘保留可查。 | 2026-07-20 |

### 三.1 六 spec 迁移收口 —— 旧 spec 世界（用户明令「清理干净，不留指路牌」·2026-07-17）

> **共同前提（逐件已核，非推定）**：下列 16 项在移入前**已全部化为「指路牌」**（每份 6–26 行，正文仅剩「本层不再有条文 / 禁在本层新增或修改任何规则」+ 新住址表），**条文本体先前已迁入六层**。移入前逐层核过迁移目的地**真的含条文**（非空壳）：`[PERF-1]`/`T-X`/`[K-5]`/`[F-1]` → `canon/测量判据.md`+`measurement/正确性门.md`；`[S-5]`/Target Profiles → `architecture/能力模型.md`；`[F-3]`/Plugin Locality → `architecture/插件协议.md`；`[SEL-*]`/`[D-*]`/Gearbox → `architecture/变体流水线.md`；八份 guide → `governance/思维准则.md` §一–§九。**移入前已把全仓活消费者改锚完毕**（详见本节末「改锚台账」），移入后全仓零悬空活引用。
> **`architecture/` 是六层之一 —— 不在本次归档范围**（含 `core-invariants.md`，59 处活引用含 `lib/`）。
> **未决登记（不自裁）**：各指路牌自述其归档去向属 **`ISSUE-070`（canon 级·待裁）**；本次移入是**执行用户明令**（「记得最后要清理干净，而不是留下一堆指路牌」「总之要是新状态」），**非 agent 自裁 ISSUE-070**。ISSUE-070 条文本身**未改**，其「归并/归档」半仍挂账待用户裁定收口措辞。

| 原路径 | → attic 路径 | 一句话原因 | 日期 |
|---|---|---|---|
| `.trellis/spec/SPEC-ARCHITECTURE.md` | `_attic/.trellis/spec/SPEC-ARCHITECTURE.md` | 六 spec 迁移的**平铺中间态**，条文已并入 `architecture/` 层；本体已是指路牌。全仓零活引用（唯一提及在 `大重构总令-最终版.md` = 令文本身、非消费者）。 | 2026-07-17 |
| `.trellis/spec/SPEC-CANON.md` | `_attic/.trellis/spec/SPEC-CANON.md` | 同上，条文已并入 `canon/` 层（13 文件）。 | 2026-07-17 |
| `.trellis/spec/SPEC-MEASUREMENT.md` | `_attic/.trellis/spec/SPEC-MEASUREMENT.md` | 同上，条文已并入 `measurement/` 层（7 文件）。 | 2026-07-17 |
| `.trellis/spec/SPEC-EVIDENCE.md` | `_attic/.trellis/spec/SPEC-EVIDENCE.md` | 同上，条文已并入 `evidence/` 层；`evidence/index.md` 与 `三贡献证据地图.md` 保留「承接 SPEC-EVIDENCE v2」**世系注**（非链接、不悬空，故不改）。 | 2026-07-17 |
| `.trellis/spec/SPEC-GOVERNANCE.md` | `_attic/.trellis/spec/SPEC-GOVERNANCE.md` | 同上，条文已并入 `governance/` 层（6 文件）。 | 2026-07-17 |
| `.trellis/spec/ISSUES.md` | `_attic/.trellis/spec/ISSUES.md` | 同上，条目已并入 `issues/` 层（6 文件）= 唯一问题登记簿。 | 2026-07-17 |
| `.trellis/spec/testing/` (2) | `_attic/.trellis/spec/testing/` | 指路牌层。★**该层曾留 `[PERF-1]` 权威**（主会话本轮据此回退过一次 stub）——**本次移入前已复核**：`[PERF-1]`/`T-X`/`[F-1]`/`[K-5]` 全部已在 `canon/测量判据.md` + `measurement/正确性门.md` 落地，`mlir-testing-contract.md` 现仅 18 行指路牌。**唯一真条文** `flat-block-dot-fp-fold-oracle.md` 已先迁 `measurement/浮点折叠oracle.md`（见下）。 | 2026-07-17 |
| `.trellis/spec/validation/` (2) | `_attic/.trellis/spec/validation/` | 指路牌层；条文已整体并入 `evidence/工件与实验参照.md`（该文自述承接）。 | 2026-07-17 |
| `.trellis/spec/guides/` (8) | `_attic/.trellis/spec/guides/` | 指路牌层；八份 guide 条文已整体并入 `governance/思维准则.md` §一–§九（逐份新住址表已核）。**7 处活 Trellis 工具/技能引用已改锚**（见台账）。 | 2026-07-17 |
| `.trellis/spec/capability-model/` (3) | `_attic/.trellis/spec/capability-model/` | 指路牌层；条文已整体并入 `architecture/能力模型.md`（含 `[S-5]` 字段级全文 + 【定法】Target Profiles）。★**曾被 canon 反向指为「字段级权威全文」住址**——该悬空授权已改锚。 | 2026-07-17 |
| `.trellis/spec/plugin-protocol/` (5) | `_attic/.trellis/spec/plugin-protocol/` | 指路牌层；条文已整体并入 `architecture/插件协议.md`（含 `[F-3]` 变更收敛 + Plugin Locality）。**`schema/family-manifest.v1.json` 7 处 + 2 门脚本 docstring 已改锚**。 | 2026-07-17 |
| `.trellis/spec/variant-pipeline/` (2) | `_attic/.trellis/spec/variant-pipeline/` | 指路牌层；条文已整体并入 `architecture/变体流水线.md`（含 `[SEL-*]`/`[D-*]`/Gearbox）。 | 2026-07-17 |
| `.trellis/spec/core-dialect/` (2) | `_attic/.trellis/spec/core-dialect/` | 指路牌层；条文已并入 `architecture/核心方言.md`。零活引用。 | 2026-07-17 |
| `.trellis/spec/extension-plugins/` (6) | `_attic/.trellis/spec/extension-plugins/` | 指路牌层；条文已并入 `architecture/家族现状.md` 等。零活引用。 | 2026-07-17 |
| `.trellis/spec/implementation-stack/` (2) | `_attic/.trellis/spec/implementation-stack/` | 指路牌层；条文已并入 `architecture/实现栈.md`。零活引用。 | 2026-07-17 |
| `.trellis/spec/lowering-runtime/` (3) | `_attic/.trellis/spec/lowering-runtime/` | 指路牌层；条文已并入 `architecture/发射与降级.md`。唯一提及在 `experiments/` 历史台账（`T8_winloss_gap_ledger.csv` 的 `#` 注释行·**已核无脚本读该 CSV**）= 写就的历史事实，按令不改。 | 2026-07-17 |

**移入计量**：41 文件（35 = 10 层 + 6 平铺件）。**整性已验**：attic 副本 vs HEAD blob 逐一 `sha256` 相同（抽验 `SPEC-CANON.md` / `SPEC-MEASUREMENT.md` / `ISSUES.md` = SAME）。**跟踪已断**：`git ls-files _attic` = 1（仅本索引）——`.gitignore:45 _attic/*` 覆盖嵌套路径，`git add -A` 只落 41 个 `D`，未把 attic 拉回跟踪。

### 三.2 `docs/` 整体归档（§七③ + §4.2.7 · 2026-07-17）

> **三分处置**（124 文件 = 67 attic + 51 事故区 + 6 留仓）。**分法不是体裁直觉，是令文自带的两条机械规则**：
> ① **§4.1.2 可达性**（根集合 = 六 spec ∪ `experiments/master` ∪ **SEALED 登记** ∪ 现役 runner/oracle/gates ∪ ISSUES ∪ 现役测试套件；「从根引用不到的一律入 attic；**被引用的留下并标注来源**」）；
> ② **§4.2.7 事故档案原样迁入 Trellis 事故区**（判据与选址理由住 [`.trellis/事故档案/README.md`](../.trellis/事故档案/README.md)，不在此重抄）。
>
> **★ 归档前置已核（非推定）**：`docs/` 现状 **124 文件**（不是先前简报所说的 119 —— 另有一个**未申报**的 `docs/files (2)/` 4 件 = 浏览器下载残留目录，内含 pre-rename 的 v3 canon 家族，被 `docs/canon/Weft-RV_科研目标总纲v4-草案.md:4` 以「导入制前身」引用；引用面全在 `docs/` 内 ⟹ 随之同行入 attic）。

| 原路径 | → attic 路径 | 一句话原因 | 日期 |
|---|---|---|---|
| `docs/canon/` (5) | `_attic/docs/canon/` | §一.1 明令「ROADMAP.md 与 docs/ 全体（**含 canon**）不再使用」。**现行法已在六层**（逐条抽验：`双账本`/`[L-9]`/`[L-10]` 均命中 spec）。**非事故档案**——体裁 = 法条，其教训按 §二 铁律已压缩成 canon 规则本身。 | 2026-07-17 |
| `docs/files (2)/` (4) | `_attic/docs/files (2)/` | **★ PR-29「浏览器残留目录」= 2026-07-17 用户裁【attic】· 本行即其销案记录。** 该目录名 `files (2)` 系**浏览器下载残留**（非人取名），内含 **pre-rename 的旧 canon v3 家族**（`TianChen-RV_科研目标总纲v3.md` 311 行 · 交接 v2 · 实验方案与结果表集 v2 · 性能宪法与测量司法）。PR-29 曾登记其为**「两 canon 家族问题」同源**（「v3 实体在 `docs/files (2)/`，**非** `docs/canon/`」—— 正是它让 `[PAT-3]` 出处被引成不解析的「v3:127」）。**归档消灭该病根**：全仓自此**只剩一个 canon 家族入口 = `.trellis/spec/canon/`**。零根引用（引用者仅 `v4-草案:4,48` 的「导入制前身」与 `E5-T1d:96`，两者本身亦已离开 `docs/`）。**PR-29 销案。** | 2026-07-17 |
| `docs/design/` (1) | `_attic/docs/design/` | 零根引用。iq2 grid repack GEMM 设计稿；其「signs64 blocker 已解」属**陈旧事实随时间被解决**、非我方判断有错 ⟹ 不入事故区。 | 2026-07-17 |
| `docs/method/` (6/10) | `_attic/docs/method/` | 归档 6：`FALSIFIER-INDEX.md`（**evidence 层 G-3 ★已出本条范围**明列「[F-1..F-6] → 工件映射 + [C1-SHAPE] 命名碰撞 → §2.2/§2.1」结论正本已迁，原件「降为可选参考·随 docs/ 归档即可」；且 **G-8③ 已核实不破门**——`check_family_locality.py` 对 `docs/**` 走前缀豁免、不校验存在性，default 门实跑 GREEN，**本轮复跑仍 exit=0**）· `REPOSITORY-MAP-五大件.md`（地图）· `CADENCE-LAW.md` · `LAW-FIRST-EMISSION.md` · `P4-family-integration-doc-TEMPLATE.md` · `x-scalar-ternary-vec-dot-construction.md`。**留 1**（`C2_marginal_cost_ledger.md`·见「查证后保留」）+ **入事故区 3**。 | 2026-07-17 |
| `docs/reports/` (56/102) | `_attic/docs/reports/` | 零根引用且非事故档案者。含全部 `perf-covered-*-green-*` 绿格登记（体裁 = 转绿记账）· 侦察报告（`gap-grid-decode-scout` / `small-m-decode-reuse` / `decode-cost-scout`：**假说被实测证伪 = 正常科学，非事故**）· T3/T6 报告模板 · `paper-material-inventory`（清单）· RENAME 台账 · M4 三件套。 | 2026-07-17 |
| `docs/ROADMAP.md` | `_attic/docs/ROADMAP.md` | §一.1 **明令**「ROADMAP.md 与 docs/ 全体不再使用」。体裁 = 队列 / canon 权威流水账。**内嵌订正确属自指翻转**（`register-cliff` 误名 + headroom 主张 RETRACTED；M1b 证伪 carrier 假说），但**第 4 条体裁测试不过** ⟹ 归档而非入事故区；其**队列职能已由 `.trellis/` 承接**（§一.1 Trellis = 唯一权威）。★**副作用见 ISSUE-094**（`--drift` 锚失依托）。 | 2026-07-17 |
| `docs/PENDING_RULINGS.md` | `_attic/docs/PENDING_RULINGS.md` | 登记册体裁；**§〇.3「已知问题零丢失」已逐条核毕**（见「查证后保留」段的 48 条落位对账）。issues 层 = 唯一问题登记簿（§二.6）；**48 行→零丢失已逐条核毕**（同轮并行写入者另增 090–093、本役增 094，故册内总数随时在动 —— 计数以 [issues/index.md](../.trellis/spec/issues/index.md) 的机算谓词为准，此处不钉死数）。 | 2026-07-17 |

**移入计量**：**67 文件**。**跟踪已断**：`git ls-files _attic` = **1**（仅本索引）—— 用 **plain `mv` 非 `git mv`**（`git mv` 会让 attic 文件**仍被跟踪**，`.gitignore` 只管未跟踪文件）。`docs/canon/` `docs/design/` `docs/files (2)/` 三个空目录已 `rmdir`。

**移入事故区（**非 attic**·令文 §4.2.7 明令）**：**51 文件** → `.trellis/事故档案/`（**在库内·已跟踪**）。判据、选址理由、十大事故簇、「内部指针刻意未改」说明**全住** [`.trellis/事故档案/README.md`](../.trellis/事故档案/README.md)。

**★ 改锚台账（移入前完成）**

| 类 | 处数 | 从 → 到 |
|---|---|---|
| `schema/` 文档指针（**已核非机检字段**：全仓无脚本校验其存在性；**[S-6] 哈希只覆盖 `capability.schema.v1.json` 一件**，其余是 [F-3] 领地） | **8 文件** | `cert-lineage` · `f5-failclosed-baseline` · `family-manifest` · `measurement-memory.{v1,design}` · `pattern-registry` · `perf-covered-category` · `tiling-measurements` → 事故区 / `_attic/docs/`；**改后 8 份 JSON 均 `json.load` 通过** |
| 门 / 工具 docstring 与打印串 | **3 行 / 3 文件** | `tools/gates/check_pat3_registry_diff.py` · `tools/visibility/recompute_ledger_anchor.sh`（仅 `lines.append` 打印、**不读文件**） · `tools/bench/new-line-worktree.sh` |
| 代码注释（**零代码、零 CHECK 行**） | **3 行 / 3 文件** | `lib/Plugin/RVV/CMakeLists.txt` · `include/Weft/Plugin/RVV/RVVRepackTilingSelection.h:459`（**折行指针**·grep 单行匹不到，易漏） · `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q6-K-q8-K-auto-unrolled-vlen128.mlir:6`（**折行**·行 6-7 是 `//` 散文、非 CHECK） |
| 六层内指针 | **2 文件** | `.trellis/spec/issues/spec树与治理.md`（ISSUE-086 死指针对照表的「正解」列 + ISSUE-072 描述）· `.trellis/spec/evidence/三贡献证据地图.md`（G-8③） |
| 留仓件的出向指针 | **2 文件** | `docs/method/C2_marginal_cost_ledger.md`（→ 事故区 X-SCALAR 排期报告 · → `_attic` T1c）· `docs/reports/2026-07-16-G8-paper-evidence-freeze.md`（→ `_attic` paper-evidence-index / G8-全量攻坚收口） |
| 并行线纪律指针（**先前即悬空**·顺手修正） | **3 行 / 3 文件** | `tools/hooks/pre-commit` · `tools/hooks/README.md` · `tools/ci/check_commit_touchset.py`：`docs/并行线纪律-worktree-与触碰集.md`（**该路径从来不存在**·真身在 `docs/reports/2026-07-06-…`）→ `.trellis/事故档案/2026-07-06-并行线纪律-worktree-与触碰集.md`。**自测复跑 exit=0** |

**★ 刻意未改（3 类 · 连同理由留证，防后人当遗漏重查）**

| 处 | 为何不改 |
|---|---|
| **`experiments/` 全部 docs 指针**（~84 处 · `MOVES.md` / 各 `MANIFEST.md` / T2·T3_A·T3_B·T8 CSV 的 provenance 列 / `B*-*.md`） | `experiments/` = **历史实验记录 = 写就的事实**，改写即篡改历史（沿用本索引 §三.1 既立先例：「唯一提及在 `experiments/` 历史台账 = 写就的历史事实，按令不改」）。**已核无脚本读取这些指针**（`grep -rnE "(open\|read_text\|cat \|source \|Path)\(...docs/" experiments/` = **零命中**；`sel3_transcribe.py` 只**写出**这些串到 schema、不读回）。CSV 里是 **provenance 列**，非文件句柄。 |
| **事故区文件的内部自述指针** | 令文 §4.2.7 **明令「原样……不改写」**。判读法住 `.trellis/事故档案/README.md` §四。 |
| **自测 fixture 里的合成路径**（`docs/x.md` · `docs/foo.md` · `docs/policy.md` · `docs/other.md` · `docs/note.md` · `check_family_locality.py:474` 的 `docs/method/FALSIFIER-INDEX.md`） | 是**门自测的合成输入串**，**从来不指真实文件**（`check_commit_touchset.py:345` 当场 `mkdir` 临时 `docs/`）。改它们 = 改测试语义。 |
| **`docs/Weft-RV_科研目标总纲v2.md` 这条【从来不存在】的路径**：现存于 `schema/tiling-measurements.v1.json`（`$meta.authority`）· `schema/pattern-registry.v1.json`（`$meta` note + authority）· `schema/VERSIONLOG.md` · `issues/spec树与治理.md`（= ISSUE-089 自己的裁决材料表，**记录该事实者，非受害者**） | **本轮归档【未】使其悬空 —— 它在 `pre-restructure-snapshot` 标签里就是死的**（真路径缺 `/canon/`；真身 = `docs/canon/Weft-RV_科研目标总纲v2.md`）⟹ 按本索引既立先例「**先前即悬空、非本次归档所致，不在触碰集**」。**⟹ 它不在本轮「零悬空」门的标的内**（该门只管「**我归档的文件**是否还有活引用」，而这条从未指向任何存在过的文件）。<br>★**同轮进展（非本役所为·2026-07-17 交叉记录）**：`capability.schema.v1.json` 的**同名第 4 条** `$meta.authority[3]` 已随 **ISSUE-089 用户裁① 落地**（`08da5e3b3`：改锚六层 + `[S-6]` bump v1.2.0 + VERSIONLOG 记笔）—— 该文件现 4 条 authority **全指 `.trellis/spec/`**，`report --check` 复跑 **exit=0**。**故本行原写的「已登记 ISSUE-089·待用户裁」已过时、就地订正**。**残留**：上列 `tiling-measurements` / `pattern-registry` / `VERSIONLOG` 三处**同名旧路径未随该裁清除**（与 `08da5e3b3` 的「悬空清零」措辞有出入）；三者均属 **[F-3] 领地（非 `[S-6]` 哈希域，`SCHEMA_JSON` 只含 `capability.schema.v1.json` 一件）**，可无风险改锚，**但不在本役触碰集**（先前即悬空 + 属 ISSUE-089 的执行面）⟹ **点名登记、不吞**。 |
| ~~**`tools/gates/emit_maturity_numbers.py` 的 `DOC_CCONSTRUCT_ANCHORS`**~~ | ~~只登记不自改~~ → **已被用户裁翻：该门【退役】，见下「门清算」段**（2026-07-17 回批①）。 |

### 三.3 `ISSUE-070` 裁定件（2026-07-17 用户裁 · **晚于上列各轮·独立动作**）

> **本节不属 §三.1 六 spec 迁移，也不属 §三.2 `docs/` 归档**。§三.1 的前提行「`architecture/` 是六层之一 —— **不在本次归档范围**（含 `core-invariants.md`，59 处活引用含 `lib/`）」**当时属实且未被推翻**：本节归档**不是**把 `architecture/` 移出六层，而是执行**其后**的用户裁决（`ISSUE-070`：I1–I9 双本 → 单一权威本 + 另一本归档 + 引用改指）。**该 59 处活引用全部仍解析** —— 原路径**保留指针文件**，未删。

| 原路径 | → attic 路径 | 一句话原因 | 日期 |
|---|---|---|---|
| `.trellis/spec/architecture/core-invariants.md`（**原全文 · 145 行**） | `_attic/.trellis/spec/architecture/core-invariants.md` | **`ISSUE-070` 裁定件**。归档的是**重复本**：其 I1–I9 与 [`canon/核心不变量.md`](../.trellis/spec/canon/核心不变量.md) **逐字一致**（谓词复跑 `diff` → 唯一差异 = 两侧各自的 range 终止标题行本身，条文零差异）。**权威本 = canon 本**，判据 = **上岗级引用四处全指它**（`CLAUDE.md` 法源入口 · `README.md` · `AGENTS.md` · `schema/family-regex.v1.json` 的 `$meta.invariant`）+ canon = 法源层 + 原 `canon/待裁.md` 4b 行自载「本层 I1–I9 已按『只留规则、删历史外壳』洗净并落此」。**★零法条损失（归档前逐条机核，非推定）**：12 条附加硬规则在 canon 均有**完整**家 —— [L-6]/[NG-4]→`成色与措辞` · [L-8]/[K-10]→`部署与构造语义` · [K-4]/[SEL-2]→`覆盖状态机与选择归因` · [S-5]/[F-2′]/[F-6]/[NG-3]→`能力模型与插件协议` · [VERIFY-LADDER]→`测量判据` · 对手类机判→`对手与档位`。本件同时是**铁律 1 违规载体**（`d0826caa` commit 号 · 「G7 全量令」体 · 「q4_K 稻草人事故」叙事 = **4 命中**）⟹ **外壳随载体离场**，非逐句删改（守「条文本体零改」）。★**原路径保留指针文件（非删除）**：`schema/capability.schema.v1.json` 的 `$meta.authority` 钉其 `[S-5]`/`[F-2′]`（v1.2.0 sealed），而 `ISSUE-072` 裁决材料 (b) **实测**改锚 = `breaking` + 新 RFC；用户令明禁「touching sealed 资产」⟹ **指针 = 唯一同时满足「单一权威本」与「不动 sealed 资产」的形态**。指针内含 `[S-5]`/`[F-2′]` 去向（两者正本 **已在 `$meta.authority` 之内**，v1.2.0 一并锚定 `canon/能力模型与插件协议.md` 与 `architecture/能力模型.md`）⟹ **schema 零改、`[S-6]` 零 bump**（实测 `report --check` exit=0 · redteam 8/8）。 | 2026-07-17 |

**移入计量**：**1 文件**（累计 attic = 67 + 41 + 3 + 1）。**跟踪已断**：`git ls-files _attic` = **1**（仅本索引）—— plain `mv`，非 `git mv`。

### 三.3 门清算 —— `--drift` 漂移门【退役】（§六.4 · 2026-07-17 用户裁「归档收官三件回批」①）

> **★ 判例（立此存照 · 一般原则）**：**凡守护对象已不存在的门，一律退役，不许改造续命 —— 旧世界的检查不迁入新世界。**
> 判别式 = 先问「**这门守的那个东西还在不在**」，再问「它绿不绿」。守护对象已被新法废除 ⟹ **改锚 / 改判据 / fail-closed 化 全部禁止**（那是给尸体续命，会把一个已绝育的病种伪装成活的防线）。

| 被杀的检查 | 死因（一行） | 处置与实测 | 日期 |
|---|---|---|---|
| `emit_maturity_numbers.py --drift`（`DOC_CCONSTRUCT_ANCHORS` + `CCONSTRUCT_RE` + `run_drift()`） | **「数字只活在主表行 + run-id，地图漂移病种已绝育」** —— 该门比对「**文档里手抄的 `C_construct N/M`**」与机算值；而新法（[evidence · index](../.trellis/spec/evidence/index.md) 规则 2「**数字不住地图 · 人工转抄数字非法**」）**已废除其守护对象**：新世界里根本不该存在被对账的转抄数（实测 `grep -rn "C_construct" .trellis/spec/` = 9 命中，**全是定义与口径、零个 `N/M` 数值**）。 | **代码移除**（292 → **249 行**）：切 `DOC_CCONSTRUCT_ANCHORS` / `CCONSTRUCT_RE` / `run_drift()` / `--drift` 参数与派发 / 随之无用的 `import re`；原址留**墓碑注释**。**负控实测**：`--drift` → **exit=2**（argparse 拒绝已退役旗标，**非静默绿**）。**存活 lane 全绿**：默认人读 / `--json` / `--self-test` 各 **exit=0**。**机算数字零漂移**：`C_construct labeled 101/110 = 91.8%` **@ 快照 `3bbd58459`**（[G-1] 快照纪律：本数 = 死亡时点的冻结观测、**非现值**；现值一律由 `emit_maturity_numbers.py` 现算）—— 与切除前**逐字相同** ⟹ **杀的只是 lane，不是计数器**。`check-weft` **954/951/3** 不变。**登记** = ISSUE-094（状态 **已退役**）。 | 2026-07-17 |

> **它归档【前】是诚实的红**（实测在报 3 处真漂移：ROADMAP 自称 84/91 · paper-evidence-index 自称 66/93 · 机算 101/110），**归档后会变无声的绿**（缺锚 → `[SKIP]` → `return 0`）。**但退役理由不是「它会变空心」，而是「它守的东西已经不存在」** —— 前者会诱人去改成 fail-closed（= 续命），后者才导出正确处置。**两者别混。**

**条文搬迁（唯一一件真条文·非指路牌）**

| 原路径 | → 新住址（**六层内·非 attic**） | 判据 | 日期 |
|---|---|---|---|
| `.trellis/spec/testing/flat-block-dot-fp-fold-oracle.md` | `.trellis/spec/measurement/浮点折叠oracle.md` | **判 `measurement/` 而非 `canon/正确性与证书.md`**：① 该文是**可执行的数值折叠规格**（逐条 `t=(float)sumi*d_x; t=t*d_y; sumf=sumf+t` 的钉死算术 + oracle 实现纪律 + lit 陷阱），属「怎么测/怎么对拍」= measurement 层职责；canon 收的是**判断依据法条**（某主张能否成立），非算术本体。② 其自题即 `[K-5]`，而 `[K-5] 正确性门` 的六层住址正是同层 `measurement/正确性门.md` —— 同层相邻、无跨层反指。③ `canon/正确性与证书.md` 收的是「硬件证据通用契约 / 断言来源区分」，与本文体裁不同。**§1–§6 章节号原样保留**（`R100` 字节相同重命名）⟹ 代码注释里既有的 `§1`/`§5` 引用**仍精确解析**，故只改路径、不动 § 号。 | 2026-07-17 |

**改锚台账（移入前完成·全仓零悬空活引用）**

| 类 | 处数 | 从 → 到 |
|---|---|---|
| fp-fold oracle 路径（`lib/` 3 · `include/` 3 · `test/` 7 · `.trellis/scripts/` 2 · ODS `.td` 2） | **16 行 / 11 文件** | `testing/flat-block-dot-fp-fold-oracle.md` → `measurement/浮点折叠oracle.md`（**只改注释/ODS description 内的文档路径**；零代码、零 CHECK 行） |
| Trellis 技能 / 工作流 / 上下文脚本 | **9 行 / 7 文件** | `spec/guides/*` → `governance/思维准则.md`（`.agents/` ×2 · `.claude/skills/` ×2 · `.claude/agents/trellis-implement.md` · `.trellis/workflow.md` · `.trellis/scripts/common/packages_context.py`） |
| 六层内**反向悬空授权**（canon/evidence 指向已 stub 的旧层） | **5 行 / 3 文件** | `capability-model/{capability-contract,profiles}.md` → `architecture/能力模型.md`（`canon/能力模型与插件协议.md` ×2 · `canon/核心不变量.md` ×1 · `evidence/工件与实验参照.md` ×2） |
| `schema/` 文档指针（**已核非机检字段**：无脚本校验其**存在性**） | **9 行 / 2 文件** | `family-manifest.v1.json`（`$meta.falsifier` + 7 × `families[].docs[]`）· `measurement-memory.design.json` → `architecture/{插件协议,变体流水线}.md`；**改后两份 JSON 均 `json.load` 通过**。★`capability.schema.v1.json` **另案·已回退**，见下「刻意未改」 |
| 门脚本 docstring | **2 行 / 2 文件** | `tools/gates/check_family_locality.py` · `.trellis/scripts/check_schema_gate.py` → `architecture/插件协议.md`（并**去掉原 `:76` 行号**，遵「跨文件引用禁行号」） |
| `architecture/系统定位与边界.md` 可复跑谓词 | **1 处** | 原谓词靶 `SPEC-*.md`/`ISSUES.md`（移入后会**报错而非返回「空」**）→ 改递归扫 `.trellis/spec/`；**已实跑复核，结论不变**（仅命中本文自身 = 无第二处落点） |
| `issues/canon与措辞.md` 成因注 | **1 处** | 「该层已化为指路牌」→「原文件已移入 `_attic/`」 |

**★刻意未改（4 处·连同理由留证，防后人当遗漏重查）**

| 处 | 为何不改 |
|---|---|
| ★★ `schema/capability.schema.v1.json` `$meta.authority[0]` = `.trellis/spec/capability-model/capability-contract.md#S-5`（**+ `layering_rule_ref` 的 `(capability-contract.md)`**） | **改了会真红门·已实证并回退**。该文件 **就是 `schema.def` 本体**（`check_schema_gate.py:SCHEMA_JSON`），而 `[S-6]` 的哈希 = `json.dumps(obj, sort_keys=True, separators=(",",":"))` 的 SHA256 **对【整个对象】取、不排除 `$meta`** ⟹ **连文档指针这种纯散文字段一动，shape-hash 即漂移**、与 `schema/VERSIONLOG.md` 的 v1.1.0 记录对不上。**实测**：改后 `check-weft` 从 951/3 变 **950/4**，新红 = `Weft :: Scripts/schema-def-f2prime-gate-redteam.test`（其 check 1「clean schema.def PASSes report --check」+ check 5「reverted schema.def PASSes again」双 FAIL·6/8）；**只回退这两处后复绿 951/3**（`git diff HEAD` 对该文件 = 空）。**为何不顺手 bump VERSIONLOG**：`[S-6]` 要求「规范化序列化 → SHA256 + RFC 版本日志（记 minor·标 extension not modification）」= **canon 级动作**，撞硬规则「禁自裁 canon 级事项 —— 只登记」。⟹ **该指针现悬空指向 `_attic/.trellis/spec/capability-model/capability-contract.md`（原件在，路径已迁）**，**须用户裁**：改锚必须与一次 `[S-6]` 版本日志 bump 同批做。**这是本次收口唯一未闭合的悬空活引用**，故显式立此条防后人误当遗漏。 |
| `lib/Plugin/RVV/FrontDoor/RVVMonolithicBlockDotSourceFrontDoor.cpp` 的 `[flat-block-dot-fp-fold-oracle §5]` | 位于 `llvm::cl::desc(...)` **字符串字面量**（CLI help 文本）= **代码，非注释**，撞「禁碰任何代码/CHECK 行」硬规则。且该处**只有文档名、无路径** ⟹ 不产生悬空路径。**登记待裁**：是否允许改 help 文本内的文档名。 |
| `architecture/系统定位与边界.md` 原样 ① 内的 `[variant-pipeline](../variant-pipeline/generation-selection-tuning.md)` | 该块受文件自身「**以下两段为原样·逐字**」约束，且原文明令「原文内的链接一并原样保留」。**引文本体零改**；改为在**引文之外**加「现住址注」指向 `architecture/变体流水线.md`。 |
| `.agents|.claude/skills/trellis-break-loop/SKILL.md` 的 `cross-platform-thinking-guide.md` / `backend/*.md` / `frontend/*.md` | **Trellis 上游模板的通用文案**，所指文件在本仓**从来不存在**（`guides/` 八份里无此名）⟹ **先前即悬空、非本次归档所致**，不在触碰集。 |
| `大重构总令-最终版.md` | `_attic/大重构总令-最终版.md` | 大重构收口完成,令文历史使命已尽 → 归档(用户 2026-07-17 令「可以归档,准备走 trellis 流程」)。其条文已迁六 spec + 事故档案 + ISSUES;**逐字性可复核** → `git show pre-restructure-snapshot:大重构总令-最终版.md`(239 行)。3 处引用:TOOLS.md 链接锚已改标签形式;收口简报/ISSUE-095 是散文历史线索(§6.4 允许·非死锚)。 | 2026-07-17 |
