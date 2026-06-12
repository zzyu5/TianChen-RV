# worker: selected path marker

## Goal

Run one full-access, serial Codex worker round for TianChen-RV MLIR. First inspect the real repository state, latest supervisor review artifacts, specs, predoc, and code; then choose one coherent high-value engineering owner from the compiler spine and leave the repo stronger with focused code, tests/checks, and a clean reviewable commit if the round completes.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current branch is `main`, and the worktree was clean before task creation.
* Recent commits have already added plugin emission readiness checks, plugin variant metadata preservation, RVV dialect/plugin first slices, registry-injected variant selection, generic variant selection planning, cost ranking, and decision metadata preservation.
* User requires a single non-TUI worker: no subagents, spawned agents, parallel agents, or multi-agent workflows.
* If no current task is appended, choose the highest-value coherent owner after inspecting repo audit, review input, git history, specs, predoc, and current code.

## Assumptions (temporary)

* The highest-value owner should avoid redoing completed scaffolding and should likely advance one of capability model, `tcrv.exec`, plugin registry, RVV probe/evidence, variant pipeline, or lowering/runtime diagnostics.
* RVV runtime/performance claims are out of scope unless this round obtains real `ssh rvv` evidence.

## Open Questions

* None blocking yet; derive the owner from repo inspection rather than asking the user.

## Requirements (evolving)

* Owner for this round: add a bounded generic selected-path marker surface for `StaticVariant` and `FallbackOnly` selection plans, then teach emission-readiness to consume that marker before falling back to all variants.
* Preserve the project spine: high-level MLIR op -> target capability model -> extension plugin registry -> plugin-proposed variants -> legality -> capability-aware selection/dispatch -> plugin-owned lowering/emission/runtime glue.
* Keep compiler internals in C++/MLIR/TableGen/CMake; Python is limited to scripts/probes/artifact utilities.
* Keep `tcrv.exec` focused on execution organization, capabilities, variants, dispatch, fallback, and diagnostics.
* Keep extension details plugin-local and avoid extension-specific branches in core orchestration.
* Ensure capabilities participate in compiler decisions.
* Add or update relevant tests for code changes.
* Selected-path metadata must be generic `tcrv.exec` diagnostic/control metadata, not a new compute op and not target-family-specific logic.

## Acceptance Criteria (evolving)

* [x] Latest `repo_audit.md` and `review_input.md` are inspected before owner selection.
* [x] Relevant specs, predoc, and code are inspected before editing implementation files.
* [x] One coherent owner is selected and documented in this PRD.
* [x] Code changes are focused on that owner and include relevant tests or diagnostics.
* [x] Validation commands are run or exact missing toolchain blockers are documented.
* [x] Final state is clean and, if the round is complete, has one coherent commit.

## Definition of Done (team quality bar)

* Tests added/updated where appropriate.
* Lint/typecheck/build/lit checks run when available, or missing dependencies diagnosed.
* Docs/specs updated if durable behavior changes.
* RVV claims include `ssh rvv` evidence; otherwise no RVV runtime/performance claim is made.
* Final report follows the user-requested seven-part format plus invariant checklist.

## Out of Scope (explicit)

* No Python-only replacement for MLIR/C++ compiler internals.
* No subagents or parallel/multi-agent workflow.
* No hardware/runtime/performance claims without corresponding evidence.
* No unrelated temporary files in the final state.

## Technical Notes

* Required initial commands were run: `pwd`, `git status --short`, `git log --oneline -8`, and bounded file listing.
* `.trellis/.current-task` was initially missing, so this task was created to record the round.
