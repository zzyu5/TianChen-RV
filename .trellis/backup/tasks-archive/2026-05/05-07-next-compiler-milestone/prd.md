# brainstorm: select next compiler milestone

## Goal

Run one full-access serial Codex worker round in the TianChen-RV MLIR repository. Inspect current audit/review/spec/code state, avoid redoing completed scaffolding, choose the highest-value coherent compiler engineering owner, implement it in the MLIR/C++/TableGen/CMake stack, validate it, and leave a clean reviewable state.

## What I already know

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Project spine: high-level MLIR op -> target capability model -> extension plugin registry -> plugin-proposed execution variants -> legality verification -> capability-aware selection/dispatch -> plugin-owned lowering/emission/runtime glue -> RVV/IME/offload/fallback path.
* Primary stack must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit/FileCheck.
* Python may only be used for scripts/probes/artifact parsing/support utilities, not compiler internals.
* Stable core dialect is `tcrv.exec`; concrete computation belongs in extension dialects or plugin-local op families.
* Current real hardware mainline is RVV 1.0 via `ssh rvv`; any RVV runtime/perf/correctness claim needs real `ssh rvv` evidence.
* The user requires single full-access non-TUI execution with no subagents, spawned agents, parallel agents, or multi-agent workflows.
* Current Trellis state had no active task before this task was created.

## Assumptions (temporary)

* Because no appended concrete current task was provided, this round should select a coherent owner after inspecting `repo_audit.md`, `review_input.md`, specs, predoc, and code.
* The selected owner should be one of capability model, `tcrv.exec`, plugin registry, RVV probe, variant pipeline, or lowering/runtime.
* If local MLIR tools are unavailable, the round should add/strengthen diagnostics instead of replacing compiler internals with Python-only structures.

## Open Questions

* None blocking before auto-context inspection; the user explicitly delegated owner selection when no current task is appended.

## Requirements (evolving)

* Inspect actual repository state before editing files.
* Read latest `repo_audit.md`, `review_input.md`, git history, TianChenRV specs, predoc, and current code before choosing work.
* Do not redo completed scaffolding.
* Choose exactly one coherent engineering owner for this round.
* Add relevant tests for code changes.
* Preserve architecture boundaries: `tcrv.exec` stays execution/capability/variant focused; extension details remain plugin-local.
* Keep capability model participating in compiler decisions.
* If claiming RVV runtime/correctness/performance, include `ssh rvv` evidence; otherwise state no RVV claim.
* End with clean, reviewable state and one coherent commit if changes are complete.

## Acceptance Criteria (evolving)

* [ ] Repository state and latest audit/review inputs are inspected and summarized.
* [ ] One owner is selected with a clear reason tied to current gaps.
* [ ] Implementation changes are focused on that owner.
* [ ] Relevant tests/checks are added or updated.
* [ ] Validation commands are run or exact missing tools are documented.
* [ ] Final report follows the requested 7-part format and invariant checklist.
* [ ] Git status is clean at handoff; if a commit is appropriate, exactly one coherent commit is created.

## Definition of Done (team quality bar)

* Tests added/updated where appropriate.
* Configure/build/lit or targeted checks run where available.
* Docs/specs updated if durable behavior changes.
* No unrelated temporary files or dirty worktree remains.
* Commit created only for a complete coherent round.

## Out of Scope (explicit)

* No Python-only implementation of compiler internals.
* No subagents or multi-agent workflows in this round.
* No unbacked RVV runtime/performance/correctness claims.
* No unrelated broad rewrite outside the selected owner.

## Technical Notes

* Initial git status before task creation was clean on `main` at `d844860`.
* Recent commits: plugin variant proposal orchestration, dispatch-aware capability requires, structured exec dispatch cases, capability query legality pass, extension plugin registry slice, exec organization ops, exec capability verifier slice, MLIR CMake skeleton.
* Further repo/audit/spec/code inspection pending.

## Auto-Context Findings

* Latest supervisor audit for r0008 reports previous HEAD `d844860`, clean post-run worktree, and one completed commit adding plugin variant proposal orchestration.
* Latest review checklist explicitly asks the next worker not to shrink into labels/status-only work and to return to capability model, `tcrv.exec`, plugin registry, RVV probe, variant pipeline, or lowering/runtime milestones.
* Current code already has `VariantProposalRequest`, `VariantProposal`, `ExtensionPlugin::supportsOperation`, `ExtensionPlugin::proposeVariants`, and deterministic `ExtensionPluginRegistry::collectVariantProposals`.
* Current proposal validation only rejects empty variant name and empty origin; it does not yet validate proposal required capability ids/symbol refs against the request capability set.
* `TargetCapabilitySet` already supports lookup by id/symbol and availability checks, including disabled/missing/unavailable status handling.
* Existing `CheckCapabilityRequires` and `tcrv.exec.variant` verifier cover materialized IR, but malformed/unavailable proposal requirements can still leave plugin registry orchestration before materialization without capability-legality diagnostics.
* Local prior audit detected `/usr/lib/llvm-20` MLIR/CMake toolchain and `ssh rvv` reachability, but this owner does not make RVV runtime/correctness/performance claims.

## Selected Owner

Capability-aware variant proposal legality at the plugin registry boundary.

Rationale: the previous coherent slice added generic proposal collection. The next highest-value non-duplicative step is to make those proposals compiler-decision objects by validating their required capability ids/symbol references against the structured `TargetCapabilitySet` before later IR materialization/selection. This strengthens capability model, plugin registry, and variant pipeline without adding target-family branches or Python compiler internals.

## Acceptance Criteria (locked for this round)

* [ ] Registry rejects proposal required capability ids that are empty, unknown, or unavailable.
* [ ] Registry rejects proposal required capability symbol references that are empty, unknown, or unavailable.
* [ ] Diagnostics include plugin name, variant name, requirement kind, missing/unavailable capability detail, and stay target-family generic.
* [ ] Existing happy path still collects enabled supported proposals deterministically.
* [ ] C++/lit tests cover happy path plus invalid id/symbol/unavailable cases.
* [ ] Specs document registry-level capability validation for pre-materialization proposals.

## Completion Evidence

* Implemented registry-side validation for proposal required capability ids and symbol references.
* Added C++ regression coverage for empty, unknown, and unavailable required capability ids.
* Added C++ regression coverage for empty, unknown, and unavailable required capability symbol references.
* Updated plugin protocol and variant pipeline specs to record the pre-materialization validation contract.
* Validation passed:
  * `git diff --check`
  * `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  * `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Acceptance Criteria Status

* [x] Registry rejects proposal required capability ids that are empty, unknown, or unavailable.
* [x] Registry rejects proposal required capability symbol references that are empty, unknown, or unavailable.
* [x] Diagnostics include plugin name, variant name, requirement kind, missing/unavailable capability detail, and stay target-family generic.
* [x] Existing happy path still collects enabled supported proposals deterministically.
* [x] C++/lit tests cover happy path plus invalid id/symbol/unavailable cases.
* [x] Specs document registry-level capability validation for pre-materialization proposals.
