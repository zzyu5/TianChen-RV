# RVV selected vector capability metadata contract

## Goal

Carry the backing capability ids for a selected RVV vector-shape config through
`selected_plan_metadata`, then validate those ids at the existing RVV
artifact/export preflight boundary. Generated RVV source already records the
selected shape capability ids in comments; this task makes the same capability
evidence visible and fail-closed in the compiler handoff metadata consumed by
source/header/object/bundle routes.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting branch is `main`, with a clean worktree at
  `1fe868a fix(rvv): validate selected config before artifact export`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the supervisor review parse fallback brief before source
  changes.
- Latest audit/input read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0074-20260510T193244Z/repo_audit.md`
  and
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0074-20260510T193244Z/review_input.md`.
- The previous completed task tightened selected RVV vector-shape and runtime
  AVL/VL metadata at the artifact/export boundary, but the selected-plan
  metadata currently covers shape/SEW/LMUL/policies/vector spelling and not
  the backing capability ids.
- `RVVVectorShapeConfig` already owns `sewCapabilityID`, `lmulCapabilityID`,
  `tailPolicyCapabilityID`, and `maskPolicyCapabilityID`.
- `RVVBinaryDescriptor` / `RVVI32BinaryDescriptor` already print these ids in
  generated source comments through `selected_vector_shape_capabilities`.
- `appendRVVVectorShapeSelectedPlanMetadata` is the shared path used by RVV
  emission planning and by RVV/RVV+scalar exporter preflight validators.
- Direct RVV and RVV+scalar artifact candidates already validate expected
  `selected_plan_metadata` entries by exact name/value/role/note.

## Requirements

- Keep the implementation in C++/MLIR/lit/FileCheck. Do not implement compiler
  semantics in Python.
- Extend the shared RVV selected vector-shape metadata descriptor list with
  four backing capability id entries:
  SEW capability id, LMUL capability id, tail-policy capability id, and
  mask-policy capability id.
- The new metadata must be produced by plugin emission planning for finite RVV
  i32/i64 binary routes and propagated into direct RVV, RVV+scalar dispatch,
  header, object, and bundle records through the existing selected-plan
  metadata path.
- The new metadata must be validated by existing RVV target/export candidate
  preflight before source/header/object/bundle output.
- Missing, duplicate, stale, wrong-role, or wrong-note capability metadata must
  fail closed at the target/export boundary.
- Generic target artifact routing and execution-plan coherence must remain
  target-neutral; no RVV capability id branch should move into shared core
  routing.
- Update focused tests and hand-authored fixtures so existing valid cases carry
  the new selected capability metadata and existing negative cases still reach
  their intended diagnostic.

## Acceptance Criteria

- [x] Trellis task exists and is current.
- [x] RVV selected-plan metadata includes selected SEW/LMUL/tail/mask
      capability id entries for generated i32 and i64 RVV binary paths.
- [x] Direct RVV microkernel artifact preflight rejects missing selected
      capability id metadata before artifact output.
- [x] Direct RVV microkernel artifact preflight rejects stale selected
      capability id metadata before artifact output.
- [x] RVV+scalar dispatch artifact preflight rejects stale selected capability
      id metadata before dispatch source/header/object output.
- [x] Bundle index output records the selected capability id metadata for
      direct and/or dispatch bundle routes.
- [x] Existing selected vector-shape config, runtime AVL/VL, capacity, and
      runtime ABI metadata checks continue to pass.
- [x] Shared core/generic target artifact routing stays target-neutral and
      free of RVV semantic branches.
- [x] `git diff --check` passes.
- [x] Focused RVV artifact/export lit and C++ tests pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes before finish/archive.
- [x] Trellis task validation passes before finish/archive.

## Minimal Validation Plan

- `git diff --check`
- Build focused targets if needed:
  `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- Run focused C++:
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit filters for touched RVV/RVV+scalar artifact paths.
- Full project lit gate:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-selected-vector-capability-metadata-contract`

## Out of Scope

- No runtime correctness, throughput, latency, ratio, or performance claim.
- No new RVV, scalar, offload, or generic backend behavior.
- No new vector shape, family matrix, or compiler route beyond existing finite
  i32/i64 RVV binary paths.
- No Python-owned compiler contracts.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  and `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Code surfaces inspected:
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`,
  `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  and `lib/Target/TargetArtifactExport.cpp`.
- While scoping this task I confirmed the earlier suspected
  dispatch-runtime-guard gap is already covered by `tcrv.exec.case` and
  `tcrv.exec.fallback` verifiers, so it is not valid as a new implementation
  task.

## Continuation Rule If Unfinished

Keep the task open and record the exact failing boundary: emission planning
metadata, direct RVV preflight, RVV+scalar dispatch preflight, hand-authored
fixture update, focused lit, aggregate lit, or Trellis validation. Do not
archive an incomplete task.
