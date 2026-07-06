# tools/hooks — repo-tracked git hook templates

These are **templates**, not active hooks. Git only runs hooks it finds under
`core.hooksPath` (default `.git/hooks/`, which is not tracked). Install once per
clone / per worktree.

## `pre-commit` — touch-set gate

Rejects a commit whose staged file set escapes the active work line's declared
touch-set. Backed by `tools/ci/check_commit_touchset.py`. See the policy in
[`docs/并行线纪律-worktree-与触碰集.md`](../../docs/并行线纪律-worktree-与触碰集.md).

### Install — option A (recommended, whole-repo)

Point git at this directory. Applies to the main worktree; **each linked
worktree keeps its own `core.hooksPath`**, so run it in every worktree too.

```bash
git config core.hooksPath tools/hooks
```

Undo: `git config --unset core.hooksPath`.

> Note: `core.hooksPath` replaces `.git/hooks` wholesale — the `*.sample` files
> there stop being consulted (they were inert anyway).

### Install — option B (single hook, keeps .git/hooks)

Symlink just this hook, leaving the rest of `.git/hooks` intact:

```bash
ln -sf ../../tools/hooks/pre-commit "$(git rev-parse --git-path hooks)/pre-commit"
```

(For a linked worktree, `git rev-parse --git-path hooks` resolves to that
worktree's own hooks dir.)

### Verify the gate is live

```bash
python3 tools/ci/check_commit_touchset.py --self-test   # must print SELF-TEST PASSED
```

### Per-line activation

The hook enforces only when this worktree has an **active line**:

```bash
echo "<line-name>" > .touch-set/ACTIVE      # per-worktree marker (git-ignored)
# or, transient:
export TRELLIS_TOUCH_SET_LINE=<line-name>
```

with a matching `.touch-set/<line-name>.txt` glob list committed to the repo.

### Conscious override

```bash
TOUCH_SET_SKIP=1 git commit ...
```

Use only for a genuine solo cross-cutting commit, never to paper over an
interleaving.
