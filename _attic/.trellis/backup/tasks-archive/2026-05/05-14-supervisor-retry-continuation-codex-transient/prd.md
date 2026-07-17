# Supervisor retry continuation for Codex transient failures

## Goal

Make the serial supervisor continue through Codex model/API/stream transient
failures even when the failed worker already changed `git HEAD` or the
worktree. A transient provider failure should not strand a completed or
nearly-completed Trellis round behind a STOP file.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* The latest loop stopped at round 17 with
  `codex_transient_retry_skipped_git_head_changed`.
* The failed worker had already created commit `4df5aaa` and left the
  worktree clean, but Codex then failed with a transient `503 auth_unavailable`
  provider error while producing final output.
* The existing runner can retry dirty worktree continuation, but it currently
  treats changed `HEAD` as a hard blocker.
* The user wants transient failures to retry instead of pausing the loop.

## Requirements

* Keep the change scoped to supervisor control-plane behavior.
* Do not modify TianChen-RV compiler code.
* Codex transient detection remains limited to model/API/stream/network/capacity
  style failures.
* A transient retry must be allowed even if `HEAD` or git status changed.
* If the previous attempt already committed and archived a task cleanly, the
  retry prompt must tell Codex to inspect live repo/artifacts and report the
  completed state instead of creating unrelated work.
* If the previous attempt left dirty work, the retry remains a continuation over
  the existing worktree, not a fresh task selection.
* Keep one retry attempt by default; if the retry also fails transiently, the
  runner may still stop with explicit artifacts.
* Update the durable supervision spec to match the new retry behavior.

## Acceptance Criteria

* [x] `git_head_changed` no longer blocks a Codex transient retry.
* [x] `git status` changes and `HEAD` changes both trigger continuation retry
      instructions.
* [x] Continuation retry prompt records before/after `HEAD` and git status.
* [x] Prompt explicitly handles the "commit already landed, final response lost"
      case.
* [x] Supervision spec no longer requires fail-closed on transient failure after
      changing `HEAD`.
* [x] `python3 -m py_compile scripts/codex_serial_supervisor.py` passes.
* [x] Focused render/status checks pass.
* [x] Worktree is clean after commit except runtime loop artifacts as expected.

## Non-Goals

* No compiler feature work.
* No Hermes prompt architecture rewrite.
* No change to non-transient compile/test/code failures.
* No infinite retry loop.

## Completion Notes

* `codex_retry_blocker` no longer treats repository mutation as a hard blocker
  for transient retries.
* `codex_retry_needs_continuation` now treats either changed `HEAD` or changed
  `git status` as a continuation case.
* Continuation prompt now records before/after `HEAD` and explicitly handles
  the case where the prior attempt already committed and archived the task but
  lost the final response to provider failure.
* The supervision-loop spec now says transient retry is allowed regardless of
  `HEAD` or status changes, with continuation over live repo state and prior
  artifacts.

## Checks

```bash
python3 -m py_compile scripts/codex_serial_supervisor.py
python3 - <<'PY' ... codex_transient_failure_reason / codex_retry_blocker / codex_retry_needs_continuation ...
python3 scripts/codex_serial_supervisor.py prompt --repo /home/kingdom/phdworks/TianchenRV --artifact-root artifacts/tmp/hermes_codex_supervisor --base-prompt scripts/codex_serial_supervisor_prompt.md --run-id 20260514T-retry-render-check
python3 scripts/codex_serial_supervisor.py run --repo /home/kingdom/phdworks/TianchenRV --artifact-root artifacts/tmp/hermes_codex_supervisor --base-prompt scripts/codex_serial_supervisor_prompt.md --run-id 20260514T-retry-noexec-check --no-exec
git diff --check
```
