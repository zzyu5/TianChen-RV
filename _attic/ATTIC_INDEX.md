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
