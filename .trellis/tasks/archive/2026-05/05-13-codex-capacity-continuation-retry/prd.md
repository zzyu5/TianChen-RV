# Codex capacity continuation retry

## Goal

Fix the serial supervisor retry behavior so OpenAI/Codex transient failures
such as `Selected model is at capacity`, stream/network timeouts, or transport
errors can launch one follow-up Codex worker to continue the same task even
when the failed worker already changed the worktree.

The current bug is that `git_status_changed` blocks the retry path. That is too
conservative for capacity/network failures: the second worker should inspect
the dirty worktree, previous run artifacts, and active Trellis task, then finish
or repair the same round rather than forcing a human closeout.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* The last interrupted round wrote coherent compiler/task changes, then Codex
  failed with `Selected model is at capacity`.
* The runner detected a transient failure but skipped retry with
  `codex_transient_retry_skipped_git_status_changed`.
* User expectation: retry is meant to recover exactly this case by starting a
  fresh Codex worker that continues the dirty task.
* HEAD changes should remain a hard blocker; if the first attempt committed,
  the retry must not blindly re-run.
* User also updated local RTK guidance in `AGENTS.md` / `RTK.md`; include that
  workspace instruction update in this control-plane commit without changing
  its contents.

## Requirements

* Keep compiler code untouched.
* Preserve fail-closed behavior for non-transient Codex failures.
* Preserve `git_head_changed` as a retry blocker.
* Do not skip retry solely because `git status --short` changed.
* When retrying after a status change, use a continuation prompt that tells the
  second worker:
  * this is a retry after a transient Codex/model/API failure;
  * do not start from scratch;
  * inspect the current dirty worktree, active Trellis task, and previous run
    artifacts;
  * continue or repair the same task;
  * finish/archive/commit when checks pass;
  * leave the task open with an exact continuation point if checks fail.
* Log whether a retry is clean or continuation-style in loop events.
* Keep the one-retry limit. If the continuation retry also fails transiently,
  stop the loop and preserve artifacts.
* Keep retry from falling back to an unrelated task or base-prompt-only run.

## Acceptance Criteria

* [x] Runner no longer emits `codex_transient_retry_skipped_git_status_changed`
      for a transient Codex failure when HEAD is unchanged.
* [x] Runner still blocks retry when HEAD changed during the failed worker.
* [x] Status-changed retry appends a clear continuation brief to the original
      Hermes Direction Brief.
* [x] Loop events record whether the retry was continuation-style.
* [x] `python3 -m py_compile scripts/codex_serial_supervisor.py` passes.
* [x] A focused local simulation proves `codex_retry_blocker` and continuation
      prompt behavior without launching Codex.
* [x] `git diff --check` and Trellis validation pass.
* [x] Task is finished, archived, and committed as one coherent commit.

## Out of Scope

* No compiler implementation changes.
* No Hermes prompt redesign.
* No change to retry count beyond the existing single retry.
* No automatic retry after a commit / HEAD change.
* No loop restart in this task.

## Technical Notes

Relevant files:

* `scripts/codex_serial_supervisor.py`
* `.trellis/spec/implementation-stack/supervision-loop.md`
* `AGENTS.md`
* `RTK.md`

## Completion Notes

Runner behavior change:

* `codex_retry_blocker()` now blocks only when the failed Codex attempt changed
  HEAD.
* A dirty worktree after a transient Codex failure is no longer a blocker by
  itself. The runner detects that state through
  `codex_retry_needs_continuation()`.
* Dirty retry attempts receive a `Codex Transient Continuation Retry` brief
  appended to the original Hermes brief/delta. The brief points at the previous
  run artifacts and instructs Codex to continue the same Trellis task from the
  live dirty worktree instead of restarting or creating a new task.
* Loop events now include `continuation_retry` on
  `codex_transient_retry_start`.
* The one-retry limit and fail-closed behavior after a second transient failure
  are unchanged.

Validation:

```bash
python3 -m py_compile scripts/codex_serial_supervisor.py
python3 - <<'PY'
# focused import-level simulation for:
# - capacity detection
# - HEAD-changed retry blocking
# - dirty-worktree continuation prompt construction
PY
python3 scripts/codex_serial_supervisor.py run --repo /home/kingdom/phdworks/TianchenRV --artifact-root artifacts/tmp/hermes_codex_supervisor --run-id retry-continuation-noexec --no-exec --prompt-override 'Retry continuation render check'
python3 scripts/codex_serial_supervisor.py status --repo /home/kingdom/phdworks/TianchenRV --artifact-root artifacts/tmp/hermes_codex_supervisor --limit 2
git diff --check
python3 .trellis/scripts/task.py validate .trellis/tasks/05-13-codex-capacity-continuation-retry
```

The user-provided RTK workspace guidance in `AGENTS.md` and `RTK.md` was kept
as-is and included in the same control-plane commit.
