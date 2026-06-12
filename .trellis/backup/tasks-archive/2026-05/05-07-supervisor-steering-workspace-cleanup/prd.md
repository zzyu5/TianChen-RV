# supervisor steering control and workspace cleanup loop

## Goal

Add deterministic human steering and Hermes self-check control surfaces to the serial Hermes/Codex supervisor runner, then restart the loop with the first worker round dedicated to workspace hygiene.

## What I Already Know

- The current loop uses a persistent Hermes chat session inside one loop process.
- Each official Hermes review still receives a full evidence prompt and must return strict JSON.
- Manual ad-hoc `hermes chat --resume` steering can race with the runner review and is not a reliable control plane.
- The user wants the next first worker round to clean workspace readability: delete confirmed-obsolete `predoc/`, organize confusing tracked tests where appropriate, and avoid leaving temporary test/probe files in the worktree.

## Requirements

- Add a durable manual steering file that `build_review_prompt` includes in every Hermes review.
- Add a read-only Hermes ask/self-check command that packages repo evidence without launching Codex or writing a next worker prompt.
- Resume the latest saved Hermes session across loop restarts unless explicitly disabled or overridden.
- Keep Python limited to runner/supervisor tooling.
- Do not implement TianChen-RV compiler functionality in this task.

## Acceptance Criteria

- [ ] Runner renders prompts with active manual steering included.
- [ ] Runner can package a no-exec worker prompt after the CLI change.
- [ ] Runner can perform or at least render the new Hermes self-check path without launching Codex.
- [ ] `python3 -m py_compile scripts/codex_serial_supervisor.py` passes.
- [ ] `git diff --check` passes.
- [ ] The loop is restarted from a clean HEAD with first-round cleanup steering.

## Out of Scope

- No compiler dialect, pass, lowering, variant, or RVV kernel implementation.
- No restructuring of `test/` by this agent turn; the restarted Codex worker should own that first loop round.
- No broad artifact cleanup in `artifacts/tmp`.

## Technical Notes

- Relevant specs: `.trellis/spec/implementation-stack/supervision-loop.md`, `.trellis/spec/implementation-stack/compiler-stack-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/validation/experiment-reference.md`.
- Existing saved Hermes session: `20260506_231323_fb6341`.
