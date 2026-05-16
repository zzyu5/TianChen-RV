# RVV Selected Dispatch Artifact Handoff

## Goal

Make the bounded RVV i32m1 add artifact/header bundle consume the selected
execution-plan and emission-plan boundary before materializing the object/header
artifact. The existing i32m1 add object/header route must no longer infer the
export target from the current single direct variant module shape.

## What I Already Know

- Commit `af4ec7c` made the bounded explicit RVV i32m1 add path produce a
  callable object/header bundle.
- Current RVV object/header export still locates exactly one direct
  `tcrv.exec.variant` in `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Generic target artifact candidate collection in
  `lib/Target/TargetArtifactExport.cpp` already understands selected
  `tcrv.exec.dispatch`, selected-path diagnostics, and supported emission-plan
  diagnostics.
- The selected RVV artifact route is bounded to explicit typed RVV i32m1 add
  IR with `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, two `i32_load`s, one
  `i32_add`, and one `i32_store`.

## Requirements

- Export the RVV i32m1 add object/header artifact from exactly one supported
  selected RVV emission-plan candidate.
- Resolve the selected variant from the emission-plan candidate's
  `selected_variant`, not by scanning for a single direct variant in the whole
  module.
- Support selected-path diagnostic input and selected `tcrv.exec.dispatch`
  input with at least one non-selected sibling variant.
- Ignore non-selected sibling variants during object/header materialization.
- Fail closed when multi-variant input is unselected or ambiguous.
- Keep the existing direct single-variant selected-path case covered, but make
  it pass through the materialized emission-plan boundary.

## Acceptance Criteria

- [ ] A selected-path diagnostic fixture with one selected RVV i32m1 add variant
      and a non-selected sibling emits the same object/header ABI metadata as
      the previous direct single-variant case.
- [ ] A selected dispatch fixture with a selected RVV i32m1 add dispatch case
      and a non-selected sibling/fallback exports the object/header bundle from
      the selected RVV emission plan only.
- [ ] Ambiguous multi-variant supported artifact input fails closed instead of
      choosing by direct variant order or module shape.
- [ ] Unselected multi-variant input fails closed before artifact output.
- [ ] Unsupported RVV shapes still fail closed with the bounded route diagnostic.
- [ ] Existing direct single-variant selected-path coverage remains, now through
      the selected emission-plan candidate.

## Out Of Scope

- New RVV dtype, LMUL, or operation families.
- General RVV lowering, MLIR vector lowering, scalar fallback compute semantics,
  dispatch runtime execution, or performance claims.
- Descriptor-driven computation, binary-family registries, direct C compute
  printers, Python compiler-core logic, GCC-default routes, compatibility
  wrappers, or RVV-specific semantic branches in core orchestration.
- Restoring deleted descriptor/direct-C/source-export legacy paths.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and guides under
  `.trellis/spec/guides/`.
- Relevant implementation files:
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, and
  `lib/Transforms/ExecutionPlanCoherence.cpp`.
- The spec tree still contains deletion-period warnings about no supported RVV
  object/header route before the bounded EmitC rebuild. Current HEAD has already
  rebuilt that bounded route; this task is limited to moving its target
  selection authority from direct-module shape to selected emission-plan
  candidates.

## Definition Of Done

- Focused build target(s) compile.
- Focused lit/FileCheck coverage passes for selected-path, selected dispatch,
  ambiguous multi-variant fail-closed, unselected multi-variant fail-closed, and
  unsupported-shape guard.
- `check-tianchenrv` is run if practical.
- Changed-surface scan confirms descriptor/direct-C/source-export legacy terms
  were not restored.
- Task status and journal are updated truthfully.
- One coherent commit is created if the task is complete.
