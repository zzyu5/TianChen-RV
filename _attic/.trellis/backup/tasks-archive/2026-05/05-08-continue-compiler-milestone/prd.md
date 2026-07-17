# worker: continue compiler milestone

## Goal

Continue the TianChen-RV MLIR compiler mainline from the current clean HEAD and
land one coherent compiler milestone that makes a real IR boundary, plugin-owned
lowering/emission path, capability decision, variant selection behavior,
runtime ABI boundary, emission artifact, or hardware-evidence claim more real.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD is `79efea5 feat: link dispatch guard to exec runtime param`.
* The worktree was clean before this task was created.
* `.trellis/.current-task` did not exist before this task; this task is the
  active local task for the round.
* Latest Hermes audit/review artifacts are under
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0041-20260508T022949Z/`.
* The latest completed milestone linked selected `tcrv.exec.case runtime_guard`
  to an IR-owned `tcrv.exec.runtime_param` and proved the bounded RVV+scalar
  dispatch self-check on real `ssh rvv` hardware.
* No `repo_audit.md` or `review_input.md` exists outside `artifacts/tmp`; the
  latest artifact copies above are the current review inputs.

## Assumptions

* The next owner should not be smoke-only, report-only, or helper-only.
* The round should prefer a bounded but meaningful compiler step over a tiny
  guardrail unless the guardrail blocks the next compiler behavior.
* Subagents and parallel agent workflows are out of scope for this round.

## Requirements

* Preserve the project spine:
  high-level MLIR op -> capability model -> plugin registry -> variants ->
  legality -> selection/dispatch -> plugin-owned lowering/emission/runtime glue.
* Keep primary implementation in C++ / MLIR / LLVM / TableGen / ODS / CMake /
  lit / FileCheck.
* Keep `tcrv.exec` execution/capability/variant focused and compute-free.
* Keep extension details plugin-local.
* Keep capability data participating in compiler decisions.
* Preserve parameter layering between target capability facts, compile-time
  variant config, runtime SSA/control values, and descriptor-local fixtures.
* Add only focused tests that validate the changed compiler behavior.
* If an RVV runtime/correctness/performance claim is made, back it with real
  `ssh rvv` evidence.

## Acceptance Criteria

* [x] One coherent compiler owner is selected and stated before code edits.
* [x] The selected owner produces code/schema/build behavior, not only docs.
* [x] Relevant Trellis specs are kept aligned when durable behavior changes.
* [x] Focused lit/C++/build checks pass, or exact missing tool diagnostics are
      documented.
* [x] The task is finished or archived before final handoff.
* [x] The final repository state is clean and reviewable.

## Out of Scope

* Rebuilding completed scaffolding from earlier tasks.
* Python-only compiler internals.
* Generic RVV lowering claims beyond the bounded implemented slice.
* Broad negative fixture matrices or dashboard-style evidence packaging unless
  they are the single blocker for the selected compiler owner.

## Technical Notes

* Required initial inspection commands were run:
  `pwd`, `git status --short`, `git log --oneline -8`, and the requested
  bounded `find` listing.
* Latest review artifacts read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0041-20260508T022949Z/repo_audit.md`
  and
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0041-20260508T022949Z/review_input.md`.
* Selected owner: make the RVV+scalar dispatch generic non-source artifact a
  runtime-callable library object rather than a self-check/evidence object.
* `--tcrv-export-rvv-scalar-i32-vadd-dispatch-object` now exports the
  library-style dispatch object directly.
* `--tcrv-export-target-artifact` now routes the validated RVV+scalar dispatch
  plan to that library object; the explicit self-check object helper remains
  direct-only evidence tooling.
* Object compilation derives the clang target from structured RVV
  `architecture = "riscv64"` capability metadata, then uses preserved selected
  `-march` and optional `-mabi` compile facts.
* Validation run: `cmake --build build --target check-tianchenrv -- -j2`
  passed 117/117 tests.
* No real `ssh rvv` evidence was claimed in this round; object emission is a
  local compiler artifact check, not runtime/correctness evidence.
