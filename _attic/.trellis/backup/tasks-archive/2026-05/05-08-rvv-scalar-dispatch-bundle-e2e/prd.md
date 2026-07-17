# RVV Scalar Dispatch Bundle E2E Correctness

## Goal

Make the canonical RVV plus scalar dispatch target-artifact bundle path more real by strengthening the path from a minimal `tcrv.exec` kernel through `tcrv-opt` execution planning and coherence, `tcrv-translate` target bundle export, and ssh `rvv` compile/run correctness evidence for both runtime dispatch paths.

## Requirements

* Work as one serial full-access Codex worker with no subagents, spawned agents, parallel agent queues, or multi-agent workflow.
* Keep primary implementation in C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
* Use Python only for script orchestration, remote probes, artifact parsing, fixture generation, or small support utilities.
* Reuse and tighten the existing RVV/scalar dispatch e2e script and bundle lit path rather than adding a parallel harness.
* Prefer the canonical `tcrv-opt --tcrv-execution-planning-pipeline` output and `tcrv-translate` target-artifact bundle index over hard-coded output assumptions.
* If metadata is missing for a bundle-driven remote compile/run, fix the owning C++ target exporter, runtime ABI, transform, or plugin layer.
* Compile and execute compiler-produced RVV plus scalar dispatch artifacts on ssh `rvv`.
* Exercise both meaningful runtime guard paths for i32 vector-add correctness: RVV selected and scalar fallback selected.
* Write non-committed evidence under `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/<run-id>/` with manifest metadata.
* Do not fake RVV correctness or substitute local x86 evidence for an ssh `rvv` runtime claim.
* Create one coherent commit and leave a clean worktree.

## Acceptance Criteria

* [ ] Required repository inspection completed before implementation.
* [ ] The canonical bundle e2e path consumes compiler-produced artifacts from `tcrv-opt` and `tcrv-translate`.
* [ ] The harness covers both dispatch guard paths and validates i32 vector-add outputs.
* [ ] Local `git diff --check` passes.
* [ ] Local `cmake --build build --target tcrv-opt tcrv-translate -j2` passes.
* [ ] Local `cmake --build build --target check-tianchenrv -j2` passes, or any failure is explicitly diagnosed.
* [ ] Relevant RVV/scalar dispatch e2e command runs against ssh `rvv`, or any remote/toolchain blocker is recorded without claiming correctness.
* [ ] Evidence directory is recorded in the final report and not committed.
* [ ] Trellis task is archived before final commit/report.

## Definition of Done

* Focused implementation changes are committed once.
* Tests or lit wrappers cover any changed compiler/script behavior.
* No build outputs, generated evidence, remote logs, generated bundles, credentials, or unrelated files are committed.
* The final report states preserved architecture invariants and exact pass/fail evidence.

## Out of Scope

* RVV performance claims.
* Broad negative matrices.
* Documentation-only, status-only, dashboard-only, or evidence-packaging-only closeout.
* Python implementations of compiler IR, dialects, passes, plugin registry, capability model, variant selection, lowering, emission, runtime ABI decisions, or compiler decisions.
* Generic compute ops inside `tcrv.exec`.
* Hard-coded RVV/scalar/vendor branches in core transforms or generic support code.

## Technical Notes

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Starting HEAD: `bbed9a7 feat: reject ambiguous capability set construction`.
* Starting worktree: clean.
* User-provided project spine: high-level MLIR op -> target capability model -> extension plugin registry -> plugin-proposed execution variants -> legality verification -> capability-aware variant selection / dispatch -> plugin-owned lowering / emission / runtime glue -> RVV / IME / offload / fallback executable path.
* Required specs and source files are listed in the user request and will be inspected before edits.
