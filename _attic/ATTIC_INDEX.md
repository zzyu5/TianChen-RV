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
| `.touch-set/` | **保留** —— 是**活基建**，非僵尸 | §4.2.2 要求"查来历 + 是否被现役引用"。**来历**：由"纪律机械化 A1+A3 — touch-set CI + 禁 git add -A + worktree 政策"引入。**现役引用（代码级，非散文）**：`tools/hooks/pre-commit`（打印那条 NOTE 的就是它）· `tools/ci/check_commit_touchset.py`（检查器本体）· `tools/bench/new-line-worktree.sh`（写 `ACTIVE` 标记）· `.gitignore`（忽略 `ACTIVE`）。**作用**：`{commit 触碰的文件} ⊆ {线声明的 globs}`，可真变红。**排查注记**：用 `grep touch-set` 会匹到散文用法（"3-file touch-set"）= 假阳性；真引用须用 `\.touch-set` 或 `TRELLIS_TOUCH_SET_LINE`。 | 2026-07-17 |

## 三、移入 attic

| 原路径 | → attic 路径 | 一句话原因 | 日期 |
|---|---|---|---|
| _(待 §四.2 逐目录处置填入)_ | | | |
