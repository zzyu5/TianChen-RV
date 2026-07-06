# 并行线纪律 — worktree 隔离 + touch-set 门

**一句话**：并行开发线一律用**独立 git worktree**；每条线**先申报触碰集**（`.touch-set/<line>.txt`）；
主树只跑一条线；活跃并行期**禁 `git add -A`**。机器强制在 `tools/hooks/pre-commit` +
`tools/ci/check_commit_touchset.py`。

教训来源：并行 agent 线一旦【要写的文件集不真不相交】就会交织。共享文件（ODS / verifier /
emitter 天然跨线共享）的改动必须**串行**（先 verify+commit 线1 再起线2）。别信"worktree 天然
隔离"——子代理用绝对路径直写主树，隔离靠**申报的 touch-set + 门**，不靠目录。

---

## 三条硬规则

### 1. 每条并行线 = 一个独立 worktree

```bash
git worktree add ../TianchenRV-<line> -b <branch>     # 新分支线
git worktree add ../TianchenRV-<line> <existing-branch>
git worktree list                                     # 查所有 worktree
git worktree remove ../TianchenRV-<line>              # 收线
```

- **主树只跑一条线**。其余线各自独立 worktree，物理隔离工作副本。
- worktree 隔离的是**工作副本 + 索引 + HEAD**，不隔离 touch-set 意图——所以规则 2 必须配套。
- `core.hooksPath` **不跨 worktree 继承**：在每个 worktree 各装一次 pre-commit
  （见 `tools/hooks/README.md`）。

### 2. 先申报触碰集（touch-set），再动代码

- 在 `.touch-set/<line>.txt` 列该线允许写的路径 glob（一行一个，`#` 注释）。**tracked**，是耐久契约。
- 本 worktree 用 `.touch-set/ACTIVE`（git-ignored，每 worktree 一份）指名当前活跃线：
  ```bash
  echo <line> > .touch-set/ACTIVE          # 或 export TRELLIS_TOUCH_SET_LINE=<line>
  ```
- **并发活跃的多条线，globs 必须真不相交**。要碰共享文件（ODS / verifier / emitter）→
  串行化：先 verify+commit 占用方，再起下一条。

### 3. 活跃并行期禁 `git add -A` / 排除法暂存

- **只许显式路径白名单**：`git add <path> <path> ...`。
- **禁** `git add -A`、`git add .`、`git add -u`、`git commit -a`、以及 `git add -A -- ':!excluded'`
  这类"排除法"暂存——它们会静默把别的线拥有的文件卷进来（历史交织的根因）。
- `pre-commit` 门会拒绝越出 touch-set 的 staged 集；但门是安全网，纪律靠显式暂存。

---

## 机器强制（怎么落地）

| 组件 | 路径 | 作用 |
|---|---|---|
| touch-set 校验器 | `tools/ci/check_commit_touchset.py` | 校验 commit/staged 文件集 ⊆ 线 globs；越界 exit≠0 |
| pre-commit 门 | `tools/hooks/pre-commit` | 提交前拒绝越界 staged 集（复用校验器） |
| 线申报 | `.touch-set/<line>.txt` | 每线允许 globs（tracked） |
| 活跃线标记 | `.touch-set/ACTIVE` | 本 worktree 当前线（git-ignored） |

安装（每个 worktree 各一次）：

```bash
git config core.hooksPath tools/hooks           # 方案 A：整仓
# 或 方案 B：只装这一个 hook
ln -sf ../../tools/hooks/pre-commit "$(git rev-parse --git-path hooks)/pre-commit"
```

自检 / CI：

```bash
python3 tools/ci/check_commit_touchset.py --self-test        # units + 真 git 绿/红
python3 tools/ci/check_commit_touchset.py --staged --line <line>   # 手动查暂存集
python3 tools/ci/check_commit_touchset.py --commit <ref>          # 查某个 commit
python3 tools/ci/check_commit_touchset.py --range A..B            # 查一段
```

CI 可把 `--range origin/main..HEAD`（每个 commit 读自己的 `Touch-Set-Line:` trailer）
接进 PR 门；本仓当前 CI 见 `.github/workflows/falsifier-gate.yml`，touch-set 门可作为并列 job 追加。

## 有意越界

genuine 的跨线单提交（罕见）：

```bash
TOUCH_SET_SKIP=1 git commit ...
```

只在真需要时用，绝不拿来盖交织。越界要在 commit body 说明理由（参考历史 ledger F23 教训：
并行线写 untracked 文件时 `git add -A` + 窄 exclude 危险，应列具体文件）。
