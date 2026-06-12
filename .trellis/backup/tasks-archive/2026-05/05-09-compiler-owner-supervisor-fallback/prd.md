# compiler owner from supervisor fallback

## Goal

Continue the TianChen-RV MLIR compiler implementation as a single full-access Codex worker after a malformed Hermes review fallback. Inspect current repository, latest supervisor audit inputs, specs, and code; then choose and implement one coherent compiler owner that makes a concrete MLIR/C++ compiler path more real rather than adding helper-only scaffolding.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* `git status --short` was clean before creating this Trellis task.
* Current branch is `main`; recent commits are focused on RVV smoke probe/export/profile compile facts and target profile capabilities.
* No `.trellis/.current-task` existed before this task was created.
* Latest supervisor artifacts discovered by mtime:
  * `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0091-20260508T232003Z/repo_audit.md`
  * `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0091-20260508T232003Z/review_input.md`
* The worker prompt requires serial execution, no subagents or parallel agent workflows, and a clean coherent commit at the end when the round is complete.

## Assumptions

* The current owner should be selected from capability model, `tcrv.exec`, plugin registry, RVV path, variant pipeline, or lowering/runtime, based on current code gaps and latest audit evidence.
* Smoke/probe/export-only work is not a valid owner unless it directly unblocks a real compiler behavior change in this same round.
* RVV runtime/performance claims require real `ssh rvv` evidence; purely local tests can validate MLIR/C++ compiler behavior only.

## Open Questions

* None for the user yet; inspect repo/audit/spec/code first and derive the owner locally.

## Requirements

* Chosen owner: relation-provider target/profile capabilities for the scalar fallback and runtime-offload plugins.
* Concrete compiler outcome: a structured `tcrv.exec.target` profile attached with `target = @profile` can explicitly `provide` `scalar.fallback` and `offload.runtime`, and that profile provider can drive plugin proposal, materialized `requires`, plugin legality, variant selection, lowering-boundary materialization, and offload descriptor emission planning.
* Preserve the primary implementation stack: C++, MLIR, LLVM, TableGen/ODS, CMake, lit/FileCheck.
* Keep `tcrv.exec` focused on execution organization, capability, variant, dispatch, fallback, and diagnostics.
* Keep concrete computation and extension details plugin-local.
* Keep target capabilities as structured compiler decision inputs.
* Maintain the parameter separation between hardware facts, compile-time variant config, runtime SSA/control values, and descriptor-local fixtures.
* Prefer a bounded but meaningful compiler owner over another helper-only smoke/probe/report round.
* Add only minimal focused tests for the compiler behavior changed.
* Create one coherent commit if the round reaches a complete reviewable state.

## Acceptance Criteria

* [x] The chosen owner names the concrete compiler path or artifact made more real.
* [x] Scalar fallback proposal and legality accept an available relation-provider capability that satisfies `scalar.fallback`.
* [x] Runtime-offload proposal, legality, boundary materialization, and descriptor emission planning accept an available relation-provider capability that satisfies `offload.runtime` and preserves the required handoff properties.
* [x] A focused execution-planning lit test proves module target-profile relation providers are consumed by scalar/offload rather than requiring exact direct capability symbols.
* [x] A focused artifact-route lit test proves a relation-provider offload profile can drive generic target artifact export for the offload descriptor.
* [x] The implementation changes MLIR/C++/TableGen/CMake compiler surfaces rather than Python-only compiler internals.
* [x] Relevant specs remain aligned with the implementation.
* [x] Focused local build/lit/C++ checks pass, or exact missing tool diagnostics are documented.
* [x] No RVV runtime/performance claim is made without `ssh rvv` evidence.
* [ ] Worktree is clean at final handoff, unless explicitly blocked.

## Definition of Done

* Tests added or updated only where they validate the changed behavior.
* Lint/build/test checks relevant to the touched surfaces are run.
* Durable spec updates are made only if this round changes a stable invariant.
* Commit is created for a complete round.

## Out of Scope

* Broad negative fixture matrices.
* Dashboard/report-only work.
* Python replacement of compiler internals.
* Treating `required_march` string matching as the long-term capability model.
* Claiming RVV correctness, runtime, or performance without real `ssh rvv` evidence.

## Technical Notes

* Required initial commands were run: `pwd`, `git status --short`, `git log --oneline -8`, and the bounded `find` listing.
* Latest `repo_audit.md`, latest `review_input.md`, `AGENTS.md`, `README.md`, `CMakeLists.txt`, relevant `.trellis/spec/**/*.md`, and relevant `include/`, `lib/`, `tools/`, `test/`, `cmake/` files were inspected before selecting the owner.
* Verification run:
  * `cmake --build build --target tcrv-opt tianchenrv-scalar-extension-plugin-test tianchenrv-offload-extension-plugin-test -j2`
  * `build/bin/tianchenrv-scalar-extension-plugin-test`
  * `build/bin/tianchenrv-offload-extension-plugin-test`
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -a Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir` from `build/test`
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -a Target/ArtifactExport/offload-runtime-descriptor-module-profile.mlir` from `build/test`
  * `git diff --check`
  * `cmake --build build --target check-tianchenrv -j2`
