# codex worker coherent compiler owner

## Goal

Run one full-access, serial Codex worker round for TianChen-RV MLIR. Inspect the current repository, latest Hermes audit/review artifacts, specs, and implementation state; then choose and complete one coherent compiler-owner milestone that makes a concrete MLIR/C++/TableGen/CMake compiler path more real.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Worktree was clean at session start.
* There was no `.trellis/.current-task` before this task was created.
* The current prompt requires a single non-TUI worker and forbids subagents, spawned agents, parallel agents, and multi-agent workflows.
* The project spine is: high-level MLIR op -> target capability model -> extension plugin registry -> plugin-proposed execution variants -> legality verification -> capability-aware variant selection / dispatch -> plugin-owned lowering / emission / runtime glue -> RVV / IME / offload / fallback executable path.
* Primary implementation must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Python is limited to runner, supervisor, probe, artifact parsing, and small support utilities.
* The stable core dialect is `tcrv.exec`; computation and hardware execution details belong to extension dialects/plugins.
* RVV runtime/correctness/performance claims require real `ssh rvv` evidence.
* The latest recent commits are around dispatch runtime guards and RVV capacity selected-plan metadata.

## Assumptions

* The current round should not redo completed scaffolding.
* Tests are validation for the chosen compiler behavior, not the owner itself.
* If no user-specified owner exists, choose the highest-value bounded milestone after reading the latest audit/review and current code.

## Requirements

* Read the repository state and relevant docs/spec/code before editing implementation files.
* Inspect the latest `repo_audit.md` and `review_input.md` under supervisor artifacts before choosing the owner.
* Choose exactly one coherent engineering owner for the round.
* Chosen owner: make RVV `vlenb` / i32 M1 lane capacity participate in the finite RVV i32-vadd microkernel descriptor decision by deriving plugin-owned `tcrv_rvv.element_count` from structured capacity facts when available.
* Concrete path made more real: sanitized RVV capacity facts -> plugin-local C++ `TargetCapabilitySet` -> RVV proposal metadata -> selected lowering-boundary materialized `tcrv_rvv.i32_vadd_microkernel` -> target exporter self-check sample size.
* Keep extension-specific behavior plugin-local; do not add extension-specific branches to core orchestration.
* Preserve the parameter-flow boundaries between hardware facts/capabilities, compile-time variant config, runtime SSA/control values, and descriptor-local fixtures.
* Keep derived `tcrv_rvv.element_count` descriptor-local: it is a bounded fixture/sample size, not high-level shape, global problem size, runtime `n`, AVL, VL, correctness coverage, or performance evidence.
* Add only minimal relevant tests for changed compiler behavior.
* Keep specs aligned when durable behavior changes.
* End with a clean, reviewable state and one coherent commit if the round completes.

## Acceptance Criteria

* [x] Latest audit/review artifacts were inspected and summarized into the task context.
* [x] Relevant specs and implementation files for the chosen owner were read.
* [x] A coherent compiler-owner milestone was selected and justified.
* [x] RVV proposal code derives descriptor-local `tcrv_rvv.element_count` from capacity facts when the structured lane count is available, while preserving the existing bounded fallback when capacity facts are absent.
* [x] Materialized RVV microkernel/export validation continues to enforce descriptor-local element count as bounded metadata and runtime `n` as an ABI/control value.
* [x] Implementation updates are scoped to that owner.
* [x] Minimal relevant lit/C++/CMake checks are added or updated.
* [x] Local validation commands are run, or exact missing tool diagnostics are reported.
* [x] Specs are updated if the implementation changes durable contracts.
* [x] Final report follows the requested seven-section format and invariant checklist.

## Completion Evidence

* `git diff --check` passed.
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2` passed.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `cmake --build build --target tcrv-opt tcrv-translate -j2` passed after plugin code changed, ensuring lit used freshly linked tools.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-capacity-selected-plan-metadata|rvv-microkernel-auto-materialization'` passed from `build/test`.
* `cmake --build build --target check-tianchenrv -j2` passed with 132/132 tests.
* No new `ssh rvv` runtime, correctness, or performance claim was made.

## Out Of Scope

* No subagents, spawned agents, parallel agents, or multi-agent workflow.
* No Python implementation of compiler internals.
* No broad dashboard/report-only, helper-only, or evidence-packaging round unless it directly blocks the compiler milestone.
* No unbacked RVV runtime/correctness/performance claim.

## Technical Notes

* Session-start required inspection commands were run: `pwd`, `git status --short`, `git log --oneline -8`, and `find . -maxdepth 3 ...`.
* `.trellis/.current-task` did not exist before task creation.
* Candidate latest supervisor run paths from lexical listing include `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0072-20260508T165606Z/{repo_audit.md,review_input.md}`; validate latest selection before relying on it.
