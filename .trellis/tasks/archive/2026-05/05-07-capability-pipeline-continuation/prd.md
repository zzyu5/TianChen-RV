# supervisor continuation: capability pipeline milestone

## Goal

Continue the TianChen-RV MLIR repository as a single full-access serial Codex worker. Inspect current repo audits, specs, predoc, git history, and code; avoid redoing completed scaffolding; then choose and complete one coherent high-value compiler milestone in the capability model, `tcrv.exec`, plugin registry, RVV probe, variant pipeline, or lowering/runtime path.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* User requires serial execution only: no subagents, spawned agents, parallel agents, or multi-agent workflows.
* Primary implementation must remain C++ / MLIR / LLVM / TableGen / CMake, with Python limited to runner/supervisor/probe/artifact support utilities.
* Stable core dialect is `tcrv.exec`; concrete computation belongs in extension dialects/plugins such as RVV, IME, offload, scalar, or future plugins.
* Current real hardware path is RVV 1.0 via `ssh rvv`; any RVV correctness/runtime/performance claim requires real remote evidence.
* The prompt reports a malformed Hermes review JSON fallback; this round must inspect latest `repo_audit.md`, `review_input.md`, git history, specs, predoc, and current code before choosing work.
* Initial repo command output showed a clean working tree on `main`, recent commits around plugin lowering boundaries, RVV metadata, scalar fallback, RVV capability profile, remote evidence probe, built-in RVV registry, and emission diagnostics.

## Assumptions (temporary)

* If no current Trellis task exists, this task records the supervisor-continuation round and its selected milestone.
* If no `repo_audit.md` or `review_input.md` is present, the absence will be documented as an input-state risk rather than blocking useful local progress.
* The selected milestone should strengthen already-present compiler structure instead of recreating project scaffolding.

## Open Questions

* None blocking at task creation time; choose work from repository evidence.

## Requirements (evolving)

* Inspect required repository state before editing implementation files.
* Read AGENTS, README, CMake, Trellis specs, predoc, and relevant source/test files before selecting the milestone.
* Preserve architecture boundaries: `tcrv.exec` remains execution/capability/variant focused; extension details remain plugin-local.
* Make capability decisions visible in compiler-facing structures and diagnostics.
* Add or update relevant tests for any code change.
* Run focused validation; if local MLIR/toolchain pieces are unavailable, document exact missing tools and rely on repository diagnostics where applicable.
* Create one coherent commit if a complete code round is achieved.

## Acceptance Criteria (evolving)

* [x] Latest available audit/review inputs are inspected or their absence is documented.
* [x] One coherent owner is selected from current repo evidence.
* [x] Code/spec/test changes are minimal and architecture-aligned.
* [x] Relevant tests/checks are run or blockers are precisely documented.
* [x] Final report follows the user-requested 7-part format and invariant checklist.
* [x] Repo is clean at handoff, with commit status reported.

## Definition of Done (team quality bar)

* Tests added/updated where appropriate.
* Focused lint/build/test checks attempted and results recorded.
* Docs/specs updated if durable behavior changes.
* Generated evidence, if any, stays under approved artifact paths.
* Worktree is clean or dirty state is explicitly justified.

## Out of Scope (explicit)

* No Python-only replacement of compiler internals.
* No high-level tensor/tile IR expansion inside `tcrv.exec`.
* No RVV runtime/performance claims without `ssh rvv` evidence.
* No subagent, spawned-agent, parallel-agent, or multi-agent execution.

## Technical Notes

* Required first inspection commands already run: `pwd`, `git status --short`, `git log --oneline -8`, and top-level file listing.
* Latest available audit/review artifacts, Trellis specs, predoc, and relevant source/test files were inspected before selecting the capability model owner.

## Selected Owner

Capability model structured-property ingestion from `tcrv.exec.capability` into `TargetCapabilitySet`.

## Owner Rationale

Latest available Hermes audit (`20260506T144732Z-r0030-20260507T004120Z`) shows plugin lowering-boundary routing was completed and committed as `2a2e6e2`. Current source already has `CapabilityDescriptor::properties` and probe-derived RVV properties, but `TargetCapabilitySet::buildFromKernel` only reads `sym_name`, `id`, `kind`, and generic `status`/`availability` from textual MLIR capability ops. That leaves core count, VLEN, dtype/runtime mode, ABI, and toolchain details visible in MLIR attributes but absent from the C++ compiler decision object.

## Concrete Acceptance Criteria

* [x] `TargetCapabilitySet::buildFromKernel` preserves non-core structured `tcrv.exec.capability` attributes as bounded descriptor properties.
* [x] String, integer, bool, float, symbol, and simple aggregate MLIR attributes have deterministic property text suitable for compiler diagnostics and plugin decisions.
* [x] Core identity/status attributes remain first-class fields and are not duplicated as generic properties.
* [x] C++ capability-model test covers textual MLIR capability properties, not only synthetic descriptors.
* [x] Capability-model spec records the durable property ingestion contract.


## Validation Results

* `cmake -S . -B build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir` passed.
* `cmake --build build --target tianchenrv-capability-model-test -j2` passed.
* `build/bin/tianchenrv-capability-model-test` passed.
* `cmake --build build --target check-tianchenrv -j2` passed: 36/36 tests.
* No `ssh rvv` run was made because this round makes no RVV runtime/correctness/performance claim.
