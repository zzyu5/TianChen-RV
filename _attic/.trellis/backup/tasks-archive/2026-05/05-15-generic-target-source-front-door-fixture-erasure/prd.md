# Generic Target Source Front-Door Fixture Erasure

## Goal

Erase active fixture authority for the deleted generic source-only
target-artifact translate option. This round removes tests and active
documentation/spec wording that preserve the old standalone source export front
door as a named contract, while keeping the surviving generic target artifact
and header fail-closed behavior covered through supported commands.

## Background

- The previous scalar cleanup round completed in commit `6e6c3be` and the
  worktree was clean at task start.
- Two active lit tests still invoke the deleted generic source-only option and
  assert the command-line unknown-option spelling as an intentional named
  absence contract.
- Active lowering/runtime spec text still records the deleted option spelling
  as durable route text.
- README text still describes a deleted target-source artifact front door in a
  way that should be generalized to avoid preserving a standalone source route
  anchor.

## Requirements

- Delete or rewrite active lit fixtures whose only source-front-door assertion
  is the deleted option spelling and unknown-option diagnostic.
- Keep current generic target artifact/header fail-closed coverage where the
  command is still a supported active front door.
- Rewrite README/spec wording so standalone direct source export absence is
  described generically, without preserving the deleted option spelling as a
  durable route anchor.
- Do not add a replacement source front door, option alias, wrapper,
  compatibility mode, direct C exporter, descriptor route, source/header/object
  route, bundle route, plugin template, or rebuild implementation.
- If checks expose failures caused by deleting old route authority, report them
  as missing new-architecture gaps instead of restoring the deleted path.

## Acceptance Criteria

- [x] Active tests no longer invoke the deleted generic source-only translate
  option or assert its unknown-option spelling through named source-front-door
  FileCheck prefixes.
- [x] The removed standalone source-front-door route fixture is gone from active
  test discovery.
- [x] README and active specs avoid preserving the deleted option spelling as
  a durable route anchor.
- [x] Surviving generic target artifact/header fail-closed behavior remains
  covered for active supported commands touched by this round.
- [x] Focused active-surface reference scan is clean for the deleted option
  spelling and named source-front-door fixture prefixes, with any remaining
  non-source hits justified as non-authoritative governance.
- [x] Focused build/test coverage for affected targets is attempted and
  results are recorded.
- [x] `check-tianchenrv` is attempted or its current status is reported.
- [x] Trellis validation passes, the task is finished/archived, the worktree is
  clean, and the round is committed.

## Out of Scope

- RVV rebuild, scalar rebuild, dispatch runtime lowering, Common EmitC source
  implementation, executable plugin templates, and RVV remote evidence.
- New source artifact route, alias, wrapper, compatibility mode, descriptor
  path, direct C exporter, self-check route, source/header/object/bundle route,
  or target source artifact route.
- Editing generated run artifacts, `artifacts/tmp`, archived Trellis tasks,
  workspace journals, or supervisor prompt guardrails just to reduce scan
  counts.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Focus files from the direction brief: `README.md`, the standalone source
  route lit fixture under `test/Target/ArtifactExport/`, the RVV/scalar
  dispatch generic-route lit fixture, `test/Target/TargetArtifactExportTest.cpp`,
  and `lib/Target/TargetArtifactExport.cpp`.
- Initial focused scan found active references in the two lit fixtures and
  active lowering/runtime spec text.

## Validation Notes

- Focused active-surface ref scan after deletion: clean for the deleted
  source-only option spelling, the removed route fixture filename, and both
  source-front-door FileCheck prefixes.
- Build: `ninja -C build tcrv-opt tcrv-translate
  tianchenrv-target-artifact-export-test` passed.
- C++ target artifact export test executable passed.
- Focused lit: filtered `Target/(ArtifactExport|RVVScalarDispatch)` run passed
  2 selected tests; the remaining RVV/scalar dispatch generic-route lit file
  passed as an explicit single-file run.
- `git diff --check` passed.
- Trellis context validation passed for `implement.jsonl` and `check.jsonl`.
- `check-tianchenrv` was attempted and remains red with 91 passed / 12 failed.
  The failed tests are existing RVV planning/lowering/script/tool baseline gaps:
  `Plugin/plugin-emission-plan.test`,
  `Scripts/rvv-probe-to-mlir.test`,
  `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  `Transforms/EmissionReadiness/emission-readiness-rvv-builtin.mlir`,
  `Transforms/EmissionReadiness/emission-readiness.test`,
  `Transforms/EmissionReadiness/materialize-emission-plans-rvv-builtin.mlir`,
  `Transforms/ExecutionPlanCoherence/rvv-capacity-stale-boundary-fails.mlir`,
  `Transforms/LoweringBoundary/rvv-i32m1-policy-capability-fails.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-compat.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-malformed.mlir`,
  `Transforms/PluginVariantLegality/plugin-variant-legality-pass-invalid.mlir`,
  and `Transforms/PluginVariantLegality/plugin-variant-legality-pass.mlir`.
