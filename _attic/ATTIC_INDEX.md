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
| `build/bin` 发现路径（3 门） | **修正而非归档** —— §4.2.4 布局归一的机械后果 | 门的 binary 发现候选原为 `$WEFT_BUILD/bin` → `build/bin` → `build-weft/bin`。**预存缺陷**：`build/bin`（陈旧 07-12 构建）排在真正在用的 `build-weft/bin` **之前** ⟹ 门一直在拿**陈旧 binary** 当证据。布局归一到 `build/weft/` 后两者皆无 ⟹ 退化为**静默 SKIP 返回 0**（更坏：绿得无声）。**已修**：候选首位改为唯一布局 `build/weft/bin`，删除已不存在的 `build-weft`。**负控实证**（f5/f4/f6 三门）：挪走 `weft-opt` + `--require-binaries` → **exit=2 全部变红**；恢复后复绿。 | 2026-07-17 |
| `.touch-set/` | **保留** —— 是**活基建**，非僵尸 | §4.2.2 要求"查来历 + 是否被现役引用"。**来历**：由"纪律机械化 A1+A3 — touch-set CI + 禁 git add -A + worktree 政策"引入。**现役引用（代码级，非散文）**：`tools/hooks/pre-commit`（打印那条 NOTE 的就是它）· `tools/ci/check_commit_touchset.py`（检查器本体）· `tools/bench/new-line-worktree.sh`（写 `ACTIVE` 标记）· `.gitignore`（忽略 `ACTIVE`）。**作用**：`{commit 触碰的文件} ⊆ {线声明的 globs}`，可真变红。**排查注记**：用 `grep touch-set` 会匹到散文用法（"3-file touch-set"）= 假阳性；真引用须用 `\.touch-set` 或 `TRELLIS_TOUCH_SET_LINE`。 | 2026-07-17 |

## 三、移入 attic

| 原路径 | → attic 路径 | 一句话原因 | 日期 |
|---|---|---|---|
| `.github/workflows/` (4 yml) | `_attic/.github/workflows/` | §4.2.1 明令删除：runner 缺口从未闭合、名不符实。**实证**：`falsifier-binary-gate.yml` 要求 `[self-hosted, weft-build]` runner（自述 DORMANT·从未供应）；`dir-hygiene.yml` 的三门在本地 HEAD 就是 RED（layout 510 findings / index 1061 issues）——若 CI 真在跑，PR 早已全红。四份 attic 副本经 sha256 逐一验与原件字节相同后才 `rm`。門本体已按 §4.2.1 迁 `tools/gates/`。`.github/` 除 workflows 外**空**（`find .github/ -mindepth 1 -not -path '.github/workflows*'` = 0），故整目录移除。 | 2026-07-17 |
| `tools/lint/check_docs_canon.py` | `_attic/tools/lint/check_docs_canon.py` | **判死 §3.5.3 条款①「文本里有没有某字样式验收」**——本体是禁词表 `FRAMING_BANNED = ("实质全收口","实质胜利","毛刺级","大成","publication-grade","capstone")` + `SLOGAN_PATTERNS` 正则。**负控实证（字样defeat）**：探针文件写「本轮达成实质胜利。」→ 命中 `[实质胜利]`；改写成**语义相同（更强）**的「本轮已全面取胜、再无欠账、可宣告收官。」→ **零命中**。同一主张换词即过门 = 该门量的是措辞不是事实，正撞 §3.3.2「验收查世系不查字样」。次因：其标的 `docs/` 按 §4.2.7 整体归档，主体将不存在。 | 2026-07-17 |
| `tools/lint/check_experiments_data_only.py` | `_attic/tools/lint/check_experiments_data_only.py` | **判死 §3.5.3 条款④「各轮重复的检查脚本（合并留一）」**——**非门，是 22 行 deprecated shim**：自述 `SUPERSEDED (org STAGE2, 2026-07-06)`，全部内容 = `import check_experiments_layout as layout; sys.exit(layout.main(sys.argv[1:]))`。零自有判断逻辑，与 `check_experiments_layout.py` 输出逐字相同（baseline 两者末行一致，实证）。留一 = `check_experiments_layout.py`（已迁 `tools/gates/`）。 | 2026-07-17 |
| `tools/lint/check_manifest.py` | `_attic/tools/lint/check_manifest.py` | **判死 §3.5.3 条款④（同上·合并留一）**——**非门，是 23 行 deprecated shim**：自述 `SUPERSEDED (org STAGE2, 2026-07-06)`，全部内容 = `import check_index_consistency as idx; sys.exit(idx.main(sys.argv[1:]))`。零自有判断逻辑。留一 = `check_index_consistency.py`（已迁 `tools/gates/`）。 | 2026-07-17 |

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
