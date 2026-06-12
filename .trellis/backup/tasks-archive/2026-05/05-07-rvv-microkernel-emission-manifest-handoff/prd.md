# RVV microkernel emission manifest handoff

## Goal

Inspect the current TianChen-RV MLIR repository state, avoid redoing completed scaffolding, choose one coherent highest-value engineering owner, implement a focused compiler-structure improvement, validate it with local checks, and leave a clean reviewable commit.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* The user requires single full-access serial execution with no subagents, spawned agents, parallel agents, or multi-agent workflow.
* Primary implementation stack must remain C++ / MLIR / LLVM / TableGen / CMake, with Python only for support utilities and probes.
* The stable core dialect is `tcrv.exec`; computation details must remain in extension dialects or plugin-local code.
* Current real hardware mainline is RVV 1.0 via `ssh rvv`; any RVV runtime/correctness/performance claim needs real `ssh rvv` evidence.
* If no appended current task exists, this round should inspect repo audit/review/spec/current code and return to the highest-value capability model, `tcrv.exec`, plugin registry, RVV probe, variant pipeline, or lowering/runtime milestone.
* Initial repo check found a clean working tree and recent commits around RVV explicit microkernel export, RVV smoke probe target export, runtime offload plugin first slice, emission manifest export target, and plugin-owned runtime ABI emission metadata.

## Assumptions

* No Hermes-specific current task was appended; the active task is therefore this worker round.
* Trellis task tracking is allowed as local workflow state, but implementation/checking must remain in the main worker without subagents.
* The selected owner should be narrow enough to implement and verify in one complete round.

## Open Questions

* None currently blocking; the worker should derive the best owner from repo state, latest audit/review inputs, specs, and code.

## Requirements

* Read the current task state if present and keep it aligned with TianChen-RV.
* Inspect latest `repo_audit.md`, `review_input.md`, git history, specs, and current code before selecting work.
* Do not redo completed scaffolding.
* Choose one coherent engineering owner from the allowed project spine.
* Add or update relevant tests for code changes.
* Add diagnostics rather than replacing unavailable MLIR/toolchain paths with Python-only compiler representations.
* Preserve plugin-local extension details and capability-driven compiler decisions.
* End with a clean, reviewable repo state and create one coherent commit if the round completes.

## Acceptance Criteria

* [x] Repo audit/review/spec/current code inspection is recorded in this task.
* [x] One coherent owner is selected with a short rationale.
* [x] Implementation strengthens compiler structure, MLIR integration, tests, or real hardware evidence.
* [x] Relevant tests/checks are run, or exact missing dependency diagnostics are recorded.
* [x] Final report covers the user-requested seven sections and preserved invariants.
* [x] Git status is clean at the end, with one coherent commit if implementation completes.

## Definition of Done

* Tests added or updated where behavior changes.
* Build/lit/unit checks run where available.
* Specs updated only when durable behavior or invariants change.
* No unrelated temporary files are committed.

## Out of Scope

* Python-only implementation of compiler internals.
* Multi-agent execution.
* RVV correctness/performance claims without `ssh rvv` evidence.
* Broad refactors not required for the selected owner.

## Technical Notes

* Required initial commands were run: `pwd`, `git status --short`, `git log --oneline -8`, and the bounded `find` listing.
* Latest complete supervisor audit/review input is `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0005-20260507T081348Z`; r0006 exists but has no `repo_audit.md` or `review_input.md` yet.
* Previous completed commit `914eacd` added the explicit `tcrv_rvv.i32_vadd_microkernel` op and `tcrv-translate --tcrv-export-rvv-microkernel-c`.
* The review tail identifies a remaining deferred item: generic emission manifest support for RVV executable microkernels.
* Selected owner for this round: RVV explicit microkernel emission-plan and emission-manifest handoff.
* Rationale: it directly follows the completed microkernel exporter, strengthens the compiler handoff surface, keeps RVV logic plugin-local, avoids redoing completed scaffolding, and can be validated with lit/FileCheck plus `check-tianchenrv`.
* Boundary: default `@rvv_first_slice` remains unsupported/deferred. Only a selected RVV path with a matching explicit `tcrv_rvv.i32_vadd_microkernel` may return a supported plugin-owned emission plan for deterministic standalone C source export. This does not claim generic RVV lowering, runtime ABI integration, arbitrary kernel emission, correctness, or performance.
* Implementation completed:
  * `RVVExtensionPlugin` now returns supported emission readiness/plans only for a unique matching explicit RVV i32 vadd microkernel attachment.
  * Generic emission-planning and manifest validation now treat only `*.lowering_boundary` ops or explicit lowering-boundary diagnostics as selected lowering-boundary candidates, so executable microkernel attachments are not counted as duplicate lowering boundaries.
  * Added lit/FileCheck coverage for supported RVV microkernel emission-plan materialization, stale/duplicate microkernel diagnostics, and generic manifest serialization of the bounded standalone C source handoff.
* Validation completed:
  * `git diff --check`
  * `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-07-rvv-microkernel-emission-manifest-handoff`
  * `cmake --build build --target check-tianchenrv -j2` passed with 92/92 tests.
