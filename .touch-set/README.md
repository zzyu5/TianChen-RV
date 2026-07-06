# `.touch-set/` — per-line write-scope declarations

Each **parallel work line** declares, up front, the set of path globs it is
allowed to write. The touch-set gate (`tools/ci/check_commit_touchset.py` +
`tools/hooks/pre-commit`) enforces `{ files a commit touches } ⊆ { line globs }`.

Policy: [`../docs/并行线纪律-worktree-与触碰集.md`](../docs/并行线纪律-worktree-与触碰集.md).

## Files

- `<line>.txt` — **tracked**. One path-glob per line; `#` starts a comment.
  This is the durable declaration for line `<line>`.
- `ACTIVE` — **git-ignored, per-worktree**. Contains the single line name this
  worktree is running. The pre-commit hook reads it to pick the active line.
  Set it with `echo <line> > .touch-set/ACTIVE` (or export
  `$TRELLIS_TOUCH_SET_LINE`).

## Glob syntax

```
foo/bar.py       exact file
tools/ci/        trailing slash → directory prefix (whole subtree)
tools/ci         bare dir name also matches the subtree
tools/ci/*.py    *  matches within one path segment (not '/')
.trellis/**      ** matches across '/' (recursive subtree)
?                one non-'/' character
```

## Example `<line>.txt`

```
# line: discipline-tooling — CI scripts, git hooks, discipline docs, trellis cleanup
tools/ci/
tools/hooks/
docs/
.touch-set/
.claude/hooks/
.claude/settings.json
.trellis/tasks/
```

See `_example.txt` for a copyable template. A commit may also carry an inline
`Touch-Set: <glob> <glob>` trailer (unioned with the line file), and
`Touch-Set-Line: <name>` to name the line when checking a historical commit.
