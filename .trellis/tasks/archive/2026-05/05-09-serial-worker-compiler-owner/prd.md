# serial worker compiler owner selection

## Goal

Run one full-access, non-TUI Codex worker round in `/home/kingdom/phdworks/TianchenRV`: inspect current repo/audit/review/spec/code state, choose the highest-value coherent compiler owner, implement it without subagents, validate with focused checks, and leave a clean reviewable commit.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* User requires serial execution: no subagents, spawned agents, parallel agents, or multi-agent workflows.
* Stable spine: high-level MLIR op -> target capability model -> extension plugin registry -> plugin-proposed variants -> legality -> selection/dispatch -> plugin-owned lowering/emission/runtime glue.
* Primary implementation stack must remain C++ / MLIR / LLVM / TableGen / CMake.
* Python is allowed only for runner/supervisor/probe/artifact utilities, not compiler internals.
* Real RVV runtime/performance/correctness claims require `ssh rvv` evidence.
* If no appended task exists, choose a coherent owner from capability model, `tcrv.exec`, plugin registry, RVV probe, variant pipeline, lowering/runtime, or related compiler milestones.

## Assumptions

* This task records the requested worker round but does not authorize subagents.
* The repo starts clean and has no aligned active `.trellis/.current-task` beyond this task.
* The selected owner should advance an active compiler path, not merely add smoke/report/dashboard artifacts.

## Requirements

* Inspect repo state before edits using the required baseline commands.
* Read latest `repo_audit.md`, latest `review_input.md`, recent git history, TianchenRV specs, and relevant code.
* Do not redo already completed scaffolding.
* Pick one coherent engineering owner and state what compiler path/artifact becomes more real.
* Keep `tcrv.exec` focused on execution/capability/variant orchestration.
* Keep extension details plugin-local and avoid extension-specific core orchestration branches.
* Add only focused validation for changed compiler behavior.
* Create one coherent commit when the round is complete.

## Acceptance Criteria

* [x] Selected owner is justified by current audit/review/spec/code evidence.
* [x] Implementation changes are in C++/MLIR/TableGen/CMake surfaces as appropriate.
* [x] Focused lit/C++/CMake checks pass, or exact missing tool diagnostics are recorded.
* [x] Any RVV runtime claim is backed by `ssh rvv`; otherwise the report says no RVV runtime claim was made.
* [x] Final report follows the requested seven-part format and invariant checklist.
* [x] Worktree is clean after commit.

## Out of Scope

* Subagents or multi-agent workflows.
* Broad negative fixture matrices or dashboard-style evidence packaging.
* Python-only implementations of compiler IR, dialects, operations, types, attributes, passes, plugin registry, capability model, lowering, or emission.

## Technical Notes

* Initial repo inspection found recent commits around RVV microkernel runtime and export.
* No `.trellis/.current-task` existed before this task was created.
* Selected owner: scalar fallback runtime-callable header plus RISC-V relocatable object artifact route.
* Compiler path made more real: a selected scalar fallback microkernel can now cross the same generic artifact boundary as RVV direct and RVV+scalar dispatch paths, producing a declaration-only C header and bounded RISC-V ELF object from the plugin-owned scalar source candidate.
* This is target/export-local scalar work; no core orchestration branch on scalar/RVV is added.
