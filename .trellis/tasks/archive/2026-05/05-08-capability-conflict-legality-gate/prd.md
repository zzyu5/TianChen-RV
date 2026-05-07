# capability conflict legality gate

## Goal

Make structured capability `conflicts` relations participate in compiler legality decisions. The current repo preserves `conflicts = [...]` on `tcrv.exec.capability` and exposes it through `CapabilityDescriptor`, but no pass rejects selected variants whose required capabilities conflict with other available target capabilities. This round adds a bounded C++/MLIR legality gate so conflict facts can stop static variants, dispatch cases, and fallbacks before selection/lowering/artifact export.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Required initial inspection was run before edits: `pwd`, `git status --short`, `git log --oneline -8`, and the requested file listing.
* The worktree was clean and HEAD was `28de8d6 feat: bind RVV scalar dispatch ABI roles`.
* `.trellis/.current-task` did not exist before this task was created.
* Latest complete supervisor audit/review pair is `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0032-20260507T211550Z/{repo_audit.md,review_input.md}`.
* The latest audit confirms the previous round only added structured runtime ABI role binding for the bounded RVV+scalar i32-vadd dispatch exporter and left the repo clean.
* `tcrv.exec.capability` already verifies `provides`, `implies`, and `conflicts` as structured arrays of non-empty capability id strings.
* `TargetCapabilitySet` already stores `conflictingIDs`, but current pass decisions only consume availability and provider/implied capability satisfaction.

## Requirements

* Keep the implementation in C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
* Keep `tcrv.exec` compute-free; capability conflicts are target/capability legality metadata, not computation.
* Add a bounded conflict query to the C++ capability model that detects available required capabilities whose conflict ids are satisfied by another available capability in the same kernel capability set.
* Preserve exact-id/provider/implied relation semantics already used by RVV profile capabilities.
* Extend `--tcrv-check-capability-requires` so:
  * unprotected static variants fail if any required available capability conflicts with another available capability;
  * dispatch cases with conflicts require explicit condition/guard/policy metadata, because the runtime guard is the compiler-visible protection surface;
  * dispatch fallbacks fail closed when their required capabilities conflict with another available capability.
* Emit diagnostics that name the requiring variant/case/fallback, the required capability symbol/id, and the conflicting provider symbol/id.
* Do not implement a full solver, preference lattice, or conflict resolution strategy in this round.
* Do not add target-family-specific branches in core orchestration.

## Acceptance Criteria

* [ ] C++ capability model test proves conflict lookup identifies direct-id and relation-satisfied conflict providers.
* [ ] `check-capability-requires` lit tests prove static conflict, unguarded dispatch-case conflict, guarded dispatch-case conflict, and fallback conflict behavior.
* [ ] Existing capability relation and RVV provider tests still pass.
* [ ] Relevant specs document this bounded conflict legality slice without claiming full conflict solving.
* [ ] Local validation includes `git diff --check` and `cmake --build build --target check-tianchenrv -j2`.

## Definition of Done

* Code, focused tests, and spec updates are committed as one coherent compiler-core commit.
* No Python compiler internals, broad smoke/probe matrix, or artifact-only wrapper work is introduced.
* `.trellis/.current-task` is not left pointing at a completed top-level task after the task is archived.

## Out of Scope

* No full conflict solver, profile lattice, provider ranking, tuning model, dispatch condition synthesis, or runtime scheduler.
* No RVV runtime/correctness/performance claim and no new `ssh rvv` evidence unless a future task explicitly runs one.
* No generic RVV lowering, LLVM/RISC-V lowering, object generation, or new export route.
* No changes to supervisor prompts or generated artifacts.

## Technical Notes

* Relevant specs: `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/guides/capability-first-design-guide.md`.
* Main code surfaces: `include/TianChenRV/Support/CapabilityModel.h`, `lib/Support/CapabilityModel.cpp`, `lib/Transforms/CheckCapabilityRequires.cpp`, `test/Support/CapabilityModelTest.cpp`, `test/Transforms/CapabilityRequires/*`.
