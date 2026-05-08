# TargetCapabilitySet identity enforcement

## Goal

Close the support-layer identity gap for synthetic/compiler-created target
capability sets. Parsed `tcrv.exec` IR is already protected by the kernel
verifier; this task makes `TargetCapabilitySet` itself fail closed when compiler
construction attempts to introduce duplicate owning capability identities.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial HEAD is `1f06c54 feat: reject duplicate capability ids`.
* Initial worktree status was clean.
* The previous run added MLIR verifier coverage in
  `lib/Dialect/Exec/IR/ExecOps.cpp` and `test/Dialect/Exec/verify.mlir`.
* Remaining gap: synthetic C++ construction paths can still risk ambiguous
  by-id capability sets unless `TargetCapabilitySet` rejects duplicates itself.

## Requirements

* Add or strengthen a structured C++ insertion/building path in the capability
  model so duplicate owning capability ids are rejected by `TargetCapabilitySet`.
* Use the repository's C++/LLVM error style, such as `llvm::Error` or
  `llvm::Expected`, and keep diagnostics generic.
* Preserve relation-provider semantics: unique concrete/profile capabilities may
  still satisfy abstract requirements through `provides` or `implies`, and may
  list conflicts.
* Preserve the existing parsed-IR verifier behavior from `1f06c54`.
* Migrate real compiler construction paths feeding plugin proposal, legality,
  selection, execution planning, or emission readiness so ambiguous sets cannot
  reach consumers.
* If duplicate symbol names are represented in `TargetCapabilitySet` and can
  corrupt lookup semantics, add the smallest matching duplicate-symbol check.
* Do not add RVV, IME, Sophgo, AME, offload, scalar, vendor, or target-family
  policy branches.
* Do not redesign target-profile or module-level attachment.
* Do not implement compiler decisions in Python.

## Acceptance Criteria

* [x] C++ Support test proves duplicate owning ids are rejected atomically.
* [x] Failed insertion does not mutate existing by-id lookup results.
* [x] Focused compiler-path test proves an ambiguous capability set cannot reach
      plugin proposal, legality, selection, or execution-planning consumers.
* [x] Positive coverage proves relation-provider semantics still work.
* [x] Existing duplicate-id MLIR verifier coverage continues to pass.
* [x] Required build and test commands pass, or exact blockers are documented.

## Definition of Done

* Code changes are in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck surfaces only
  for compiler behavior.
* Focused tests are added or updated.
* Relevant specs are updated only if the behavior contract changes.
* Trellis task is archived before final commit.
* Worktree is clean after one coherent commit.

## Out of Scope

* New RVV runtime, correctness, or performance evidence.
* Broad negative matrices.
* Module-level target attachment redesign.
* New generic compute operations in `tcrv.exec`.
* Helper-only, docs-only, test-only, or report-only closeout.

## Technical Notes

* Required inspection commands were run before compiler edits.
* `.trellis/.current-task` was absent before task creation; `.trellis/tasks`
  contained only archived tasks.
* `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-capability-model-test -j2` passed.
* `build/bin/tianchenrv-capability-model-test` passed.
* `build/bin/tianchenrv-plugin-variant-legality-test` passed.
* `build/bin/tcrv-opt test/Dialect/Exec/verify.mlir --verify-diagnostics
  -split-input-file` passed.
* `cmake --build build --target check-tianchenrv -j2` passed with 126/126
  tests.
* `git diff --check` passed.
