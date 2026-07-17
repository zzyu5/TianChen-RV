# RVV selected vector-shape config boundary cleanup

## Goal

Clean up the RVV plugin/target-owned selected i32 vector-shape configuration
boundary for the existing i32m1 and i32m2 i32 add/sub/mul paths. The compiler
must represent the selected shape explicitly as a C++/MLIR-owned contract, keep
shape-neutral hardware capacity facts separate, and make the linalg-origin
i32-vsub i32m2 path and existing i32m1 paths consume the same structured
boundary before more RVV kernels or shapes are added.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round was a clean worktree at HEAD
  `93ddc96`.
- No `.trellis/.current-task` existed at session start, so this task was
  created as
  `.trellis/tasks/05-10-rvv-selected-vector-shape-config-boundary-cleanup/`.
- The archived
  `.trellis/tasks/archive/2026-05/05-10-linalg-i32m2-vsub-ssh-rvv-evidence-handoff/prd.md`
  task is complete and already collected bounded real `ssh rvv` evidence for
  the frontend-generated linalg i32-vsub i32m2 bundle path.
- The next bottleneck is compiler structure, not more evidence guards.
- Current code already has duplicated finite RVV i32 shape knowledge in
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp` and
  `lib/Target/RVV/RVVMicrokernel.cpp`.
- Current i32m2 linalg tests preserve `rvv.i32_m1_lane_count` /
  `tcrv_rvv.i32_m1_lanes` on an i32m2 path as a capacity fact, which is easy to
  misread as selected m1 config unless the compiler boundary classifies it
  explicitly.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python changes, if any, may only consume emitted compiler
  metadata for evidence validation.
- Introduce or strengthen a plugin/target-owned C++ representation for the
  selected finite RVV i32 vector shape with:
  - shape id;
  - SEW;
  - LMUL;
  - tail policy;
  - mask policy;
  - vector type;
  - vector intrinsic suffix;
  - setvl suffix;
  - required selected-shape capability ids.
- Use that shared representation across RVV proposal, selected variant
  metadata, selected lowering boundary metadata, microkernel materialization,
  emission-plan selected metadata, target artifact export, generated source
  comments, and evidence validation.
- Preserve the distinction between:
  - hardware/profile facts such as VLEN bytes and base i32 M1 lane capacity;
  - selected compile-time vector config such as selected i32m1 or i32m2 shape;
  - runtime SSA/control values such as AVL/VL/runtime `n`;
  - descriptor-local bounded artifact values such as `tcrv_rvv.element_count`.
- If base i32 M1 lane capacity is preserved on an i32m2 selected path, classify
  and name it as a base-capacity fact so it cannot be mistaken for selected m1
  vector-shape config.
- Preserve existing bounded i32 add/sub/mul functionality and existing i32m1
  and i32m2 behavior.
- Update only necessary specs/tests to lock the boundary.

## Acceptance Criteria

- The linalg-origin `i32-vsub` i32m2 planning/export path emits selected
  vector-shape metadata identifying i32m2, not stale i32m1 selected config.
- Existing i32m1 RVV microkernel export paths continue to emit i32m1 selected
  shape metadata through the same C++ representation.
- Stale selected i32m1 metadata on an i32m2 selected capability/body path fails
  before target source artifact output.
- Base i32 M1 lane capacity metadata is renamed or classified as a capacity
  fact distinct from selected shape metadata in C++/MLIR outputs and specs.
- Generated RVV C source comments expose the selected shape config and evidence
  parser validation consumes that compiler-emitted metadata without becoming
  compiler truth.
- Focused lit/FileCheck coverage passes for affected linalg/RVV
  planning/export paths.
- `git diff --check` passes.
- `python3 scripts/rvv_microkernel_e2e.py --self-test` passes if the evidence
  parser expectations change.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passes after focused checks.

## Out of Scope

- Adding new dtypes, new LMULs beyond existing i32m1/i32m2, new arithmetic
  families, or broad benchmark/test matrices.
- Adding performance claims or new benchmark reporting.
- Implementing compiler semantics in Python.
- Adding RVV-specific branches to generic core passes.
- Adding compute semantics to `tcrv.exec`.
- Making a docs-only, test-only, helper-only, or smoke-only commit.
- Collecting new `ssh rvv` runtime evidence unless generated source/header/
  object semantics or claim-bearing artifact behavior changes.

## Technical Approach

- Add one shared RVV target-owned C++ selected vector-shape config surface for
  the finite i32m1/i32m2 shapes and use it from both the RVV plugin and RVV
  target exporter.
- Add plugin-owned selected-shape attributes on materialized RVV variants and
  validate any present selected-shape attributes against the required
  capability-derived shape.
- Copy selected-shape metadata to the RVV lowering boundary and
  plugin-materialized microkernel attachment.
- Add selected-shape entries to emission-plan selected metadata so manifest and
  bundle/export records preserve the compiler-selected shape without relying on
  ad hoc generated-source string checks.
- Rename/classify selected-plan capacity metadata from generic
  `i32_m1_lanes` wording to base i32 M1 capacity wording while preserving the
  underlying `rvv.i32_m1_lane_count` capability as a hardware/profile fact.
- Extend generated source comments with an explicit selected-vector-shape
  comment and have `scripts/rvv_microkernel_e2e.py` validate that emitted
  comment as an evidence consumer.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  - `.trellis/spec/capability-model/capability-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Archived task read:
  - `.trellis/tasks/archive/2026-05/05-10-linalg-i32m2-vsub-ssh-rvv-evidence-handoff/prd.md`
- Primary implementation surfaces:
  - `include/TianChenRV/Target/RVV/`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/EmissionManifest.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `test/Transforms/LinalgToExec/`
  - `test/Target/RVVMicrokernel/`
  - `test/Scripts/rvv-microkernel-bundle-e2e.test`
  - `scripts/rvv_microkernel_e2e.py`

## Validation Plan

- Focused lit/FileCheck for linalg i32-vsub i32m2 planning/export selected
  shape metadata and stale selected-shape rejection.
- Focused lit/FileCheck for RVV i32m1/i32m2 microkernel source export selected
  shape comments.
- Existing focused RVV plugin/target artifact tests touched by the boundary.
- `python3 scripts/rvv_microkernel_e2e.py --self-test` if source comment
  validation changes.
- `git diff --check`.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.

## Current Status

Completed.

## Completion Notes

- Added `RVVI32VectorShapeConfig` as the shared RVV target-owned selected
  vector-shape contract for existing i32m1/i32m2 shapes.
- RVV proposal/materialization, selected boundary, microkernel attachment,
  emission-plan metadata, target source export, and evidence parsing now use
  the same selected-shape fields: shape id, SEW, LMUL, tail/mask policy,
  vector type, intrinsic suffix, setvl suffix, and backing capability ids.
- Renamed the old selected-plan `tcrv_rvv.i32_m1_lanes` metadata to
  `tcrv_rvv.base_i32_m1_lanes` / `base_i32_m1_lanes`, preserving
  `rvv.i32_m1_lane_count` as a target/profile hardware capacity fact.
- The linalg-origin i32-vsub i32m2 path emits i32m2 selected-shape metadata
  while still carrying the separate base i32 M1 capacity fact.
- Stale selected i32m1 shape metadata on an i32m2 selected path now fails
  before target source artifact output.
- Specs updated only for the capability parameter layering, RVV plugin
  selected-shape boundary, and emission parameter claim boundary.

## Validation Results

- [OK] `git diff --check`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emission-readiness-test`
- [OK] focused lit filter for linalg i32m1/i32m2 planning/export, stale
  selected-shape guard, RVV microkernel i32m2 source export, probe, and bundle
  tests (11/11)
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] focused lit filter `rvv-microkernel-bundle-e2e` (1/1)
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` (169/169 tests passed)
- [OK] real `ssh rvv` linalg-origin i32-vsub i32m2 bundle evidence:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub
  --vector-shape=i32m2 --expect-selected-kernel=frontend_i32_vsub
  --run-id codex-shape-boundary-linalg-vsub-i32m2 --overwrite --input
  test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir
  --tcrv-opt artifacts/tmp/tianchenrv-build/bin/tcrv-opt --tcrv-translate
  artifacts/tmp/tianchenrv-build/bin/tcrv-translate --timeout 120`
- [OK] evidence artifact:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-shape-boundary-linalg-vsub-i32m2/evidence.json`
