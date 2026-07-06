# 并行构建 + 字节精确基线（reliable parallel lines）

**一句话**：并行开发线一律用**从 `HEAD` 分叉的独立 worktree** + **每线独立
`build/<line>/`** + **字节精确基线走缓存的 detached-base 产物、永不 `git stash`**。
配套纪律（触碰集 / worktree 隔离）见
[`../reports/2026-07-06-并行线纪律-worktree-与触碰集.md`](../reports/2026-07-06-并行线纪律-worktree-与触碰集.md)；
本文覆盖的是**构建 + 基线**这三处机制。

机器落地：`tools/bench/`
- `new-line-worktree.sh <line>` — 开线（从 HEAD 分叉 + 装 touch-set 门 + 配 build）
- `configure-line-build.sh <line>` — 单独（重）配某线的独立 build
- `byte-exact-baseline.sh <base-commit> <target>` — 缓存的 pinned-base 产物（**代替 stash**）
- `_build-common.sh` — 共享 configure 咒语（sourced，非入口脚本）

---

## 为什么并行开不了（三个已实证的根因）

| # | 根因 | 症状 | 本文机制 |
|---|---|---|---|
| ① | 字节精确用 **`git stash` 造基线** | stash 是**全局树操作**，卷走*别的*并行线未提交改动（实证：卷走过 sibling 线的 `settings.json`/`.gitignore`）。文件集不相交也救不了。 | `byte-exact-baseline.sh`：detached base worktree + 按 sha 缓存产物，**主树永不 stash** |
| ② | 隔离 worktree 从**陈旧 base**（远端主分支旧点）分叉 | 缺当前战役输入（`experiments/sealed/` 等），逼回主树，隔离失效 | `new-line-worktree.sh`：`worktree add … HEAD`，从**当前 HEAD** 分叉，携带全部已跟踪战役输入 |
| ③ | **单一绝对 build 路径** | 并行 `ninja` 撞同一个 `build/CMakeCache.txt` | 每线 `build/<line>/` 在**自己的 worktree 内**，独立 CMakeCache；`ninja` 互不干扰 |

---

## 机制一：worktree 一律从 `HEAD` 分叉

```bash
tools/bench/new-line-worktree.sh <line>            # 默认顺带 configure build
tools/bench/new-line-worktree.sh <line> --no-configure   # 只开 worktree
```

做的事：

1. `git worktree add -b line/<line> .worktrees/<line> HEAD` —— **从 HEAD、不是 origin**。
   隔离树因此含**全部已跟踪的当前战役输入**（根治 ②）。
2. touch-set：写 `.touch-set/ACTIVE=<line>`（per-worktree，git-ignored），若无则落一份
   **起手 `.touch-set/<line>.txt` 模板**（tracked，开工前填写该线可写的 globs）。
   门（pre-commit hook）**不自动装**：`git rev-parse --git-path hooks` 解析到**共享的**
   主 `.git/hooks`，往那 symlink 会改主树提交行为、且线一撤就悬空；开工时**逐 worktree**
   手动激活 `git config core.hooksPath tools/hooks`（见 `tools/hooks/README.md`）。
3. 若装了 `ccache`/`sccache`，configure 时挂到 compiler-launcher，摊平多树构建成本。
4. 默认调 `configure-line-build.sh <line>` 配好独立 build（机制二）。**不 `ninja`**（构建留给你）。

> `.worktrees/` 与其下所有内容都被 `.gitignore` 忽略，永不进版本库。

## 机制二：每线独立 out-of-tree 构建

```bash
tools/bench/configure-line-build.sh <line>         # 单独（重）配
ninja -C .worktrees/<line>/build/<line> tcrv-opt tcrv-translate
ninja -C .worktrees/<line>/build/<line> check-tianchenrv
```

- build 目录 = `<worktree>/build/<line>/`，**在 worktree 内**，绝对路径唯一、独立
  CMakeCache（根治 ③）。
- LLVM/MLIR 路径从主树 `build/CMakeCache.txt` 派生（对齐仓库实际工具链），缺省回退
  README 默认；此逻辑集中在 `_build-common.sh`，保证**线构建与基线构建 configure 完全一致**
  —— 否则字节 diff 可能反映的是工具链差异而非源码差异，会悄悄废掉字节精确门。
- **stash 一律 worktree-local**：真要 stash，在**线 worktree 内** `git stash`，只动该 worktree，
  不碰主树。

## 机制三：★字节精确基线 —— 废除全局 stash

```bash
# 1) 拿到 pinned base 产物路径（缓存；首次构建、之后命中幂等）
BASE=$(tools/bench/byte-exact-baseline.sh <base-commit> tcrv-opt)

# 2) 字节精确 = 缓存 base 产物 emit  vs  线 worktree constructed 产物 emit
diff <("$BASE"                          IN.mlir --tcrv-rvv-lower-to-emitc) \
     <(.worktrees/<line>/build/<line>/bin/tcrv-opt IN.mlir --tcrv-rvv-lower-to-emitc)
# 空 diff = byte-identical。全程不碰主树、无 git stash。
```

机制（`byte-exact-baseline.sh`）：

1. 把 `<base-commit>` **pin 成解析出的完整 sha**（缓存键无歧义）。
2. 命中 `.worktrees/cache/<sha>/<target>` → 立即返回路径，**不重建**（幂等）。
3. 未命中 → `git worktree add --detach .worktrees/baseline-<sha> <sha>`（把 base 检出到
   **单独目录**，主树工作副本与所有 sibling 线未提交改动**全程不动**）→ `bench_configure`
   → `ninja <target>` → 把 `bin/<target>` 拷进 `.worktrees/cache/<sha>/`（**按 commit-hash 缓存**）
   → 移除临时 base worktree（`--keep-worktree` 可留）。
4. **stdout 只打印缓存产物绝对路径**（其余全走 stderr），可直接 `BASE=$(…)` 组合。
5. `--dry-run` 打印计划（sha / 缓存路径 / 会不会重建）不构建。

**★钉死**：跑 byte-exact verify 的线**一律 worktree**；**主树永不 `git stash` 造基线**。

---

## 并行线开工协议（清单）

1. **开线**：`tools/bench/new-line-worktree.sh <line>`（从 HEAD；自动装门 + 配 build）。
2. **申报触碰集**：编辑 `.worktrees/<line>/.touch-set/<line>.txt`，列该线可写 globs；
   与其余活跃线**真不相交**（共享文件 ODS/verifier/emitter 必须串行，不并行）。
3. **构建**：`ninja -C .worktrees/<line>/build/<line> tcrv-opt tcrv-translate`。
4. **字节精确**：`BASE=$(tools/bench/byte-exact-baseline.sh <base> tcrv-opt)` → 用**缓存 base**
   emit 对 diff。**不 stash、不动主树。**
5. **暂存**：只用**显式路径白名单** `git add <path> …`；活跃并行期**禁** `git add -A`/`git add .`/
   `git commit -a`（pre-commit 门是安全网，纪律靠显式暂存）。
6. **收线**：commit 上 `line/<line>` → merge 回主线 → `git worktree remove .worktrees/<line>`。

## 忽略清单（`.gitignore`）

```
build/       build/*/         # 主 build + 每线 build/<line>/
.worktrees/                    # 所有线 worktree + baseline-<sha> + cache/<sha>/
```
