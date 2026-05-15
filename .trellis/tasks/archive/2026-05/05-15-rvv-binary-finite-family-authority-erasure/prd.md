# RVV binary finite-family authority erasure

## Goal

Delete the active RVV binary finite-family route authority that still lets
selected-plan metadata, finite family records, microkernel operation names,
intrinsic-name helpers, runtime ABI helper calls, or selected-boundary planning
act as the compiler route source. This is a Wrong Logic Deletion Campaign
round: remove or fail-close the old layer and tests that protect it; do not
rebuild a replacement RVV lowering or artifact route in the same round.

## What I already know

- Current repo root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD is `771de92 chore(scalar): erase rvv-scalar finite bridge`.
- Worktree was clean before task creation.
- There was no `.trellis/.current-task`; this task was created from the Hermes
  Direction Brief.
- The current active residue is the RVV binary finite-family layer:
  `RVVBinaryFamilyRecord`, `getRVVBinaryFamilyRegistrationRecords`,
  `RVVBinarySelectedPlan`, `RVVBinaryEmissionIdentity`,
  selected-binary metadata helpers, intrinsic-name helpers, direct typed RVV
  binary family resolution, selected-lowering-boundary materialization tied to
  finite family records, and tests/lit fixtures that preserve that behavior.
- Relevant long-term specs say descriptor/direct-C metadata cannot be the
  executable architecture; future executable support must go through explicit
  extension-family ops and a materialized EmitC route.

## Requirements

- Delete the active RVV binary finite-family record API from target/plugin code.
- Delete or rewrite RVV plugin planning, legality, selected-boundary, and
  selected-emission paths that depend on finite binary family records or
  microkernel names as route authority.
- Keep the RVV plugin behavior that remains bounded to generic plugin identity,
  capability facts, vector-shape capability metadata, target support bundle
  registration, and explicit unsupported diagnostic boundaries.
- Remove selected-binary metadata attributes/helpers from active RVV planning
  and materialization surfaces.
- Delete or rewrite focused tests that protect selected-binary metadata,
  finite microkernel family selection, direct typed-body family resolution, or
  old runtime-callable C source planning.
- If checks expose missing new-architecture RVV lowering or EmitC support,
  report the gap instead of restoring the deleted finite-family route.

## Acceptance Criteria

- [ ] Active code no longer exposes `RVVBinaryFamilyRecord`.
- [ ] Active code no longer exposes `getRVVBinaryFamilyRegistrationRecords`.
- [ ] Active code no longer exposes `RVVBinarySelectedPlan`.
- [ ] Active code no longer exposes `RVVBinaryEmissionIdentity`.
- [ ] Active RVV plugin code no longer publishes selected-binary metadata
      helpers or attributes as planning/emission/boundary authority.
- [ ] Active RVV plugin code no longer resolves direct typed RVV binary family
      bodies by microkernel op name.
- [ ] Active RVV plugin emission-readiness and emission-plan paths are
      unsupported diagnostic boundaries unless a future materialized EmitC
      route exists.
- [ ] Focused RVV binary planning/legality/selected-boundary tests are deleted
      or rewritten so they no longer protect the removed behavior.
- [ ] Ref-scans for the Hermes-listed symbols are clean in active RVV
      plugin/target/test surfaces, except intentional historical/deleted-route
      text that is reported explicitly.

## Non-Goals

- Do not implement a new RVV lowering.
- Do not implement a new EmitC route.
- Do not implement artifact emission, runtime ABI glue, object/header/source
  exporters, or runtime execution support.
- Do not add a generic family registry, compatibility wrapper, legacy mode, or
  replacement finite-family table.
- Do not extend the RVV finite family.
- Do not delete generic RVV dialect definitions, generic capability facts,
  generic plugin registration, generic target artifact registry infrastructure,
  or non-semantic vector-shape/capability utilities unless they are only
  reachable through the finite binary family authority being deleted.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Code surface inspected:
  `include/TianChenRV/Target/RVV/RVVBinaryFamily.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`,
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/CMakeLists.txt`,
  focused plugin/target tests under `test/Plugin/` and
  `test/Target/RVVMicrokernel/`.
- Existing `RVVVectorShape` capability utilities are in scope to retain if they
  only describe compile-time vector-shape capability metadata and not binary
  family route authority.

## Validation Plan

- Focused ref-scans for:
  `RVVBinaryFamilyRecord`, `getRVVBinaryFamilyRegistrationRecords`,
  `RVVBinarySelectedPlan`, `RVVBinaryEmissionIdentity`, `RVVBinaryPlanning`,
  `RVVBinaryVariantLegality`, `RVVBinarySelectedLoweringBoundary`,
  `RVVBinarySelectedEmissionPlanning`, `tcrv_rvv.selected_binary`,
  `selected_binary_family`, `runtime-callable-c-source`,
  `getSetVLIntrinsicName`, `getLoadIntrinsicName`,
  `getArithmeticIntrinsicName`, `getStoreIntrinsicName`,
  `direct typed RVV binary`, and `_microkernel`.
- Build affected tools/targets as available: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and remaining focused RVV plugin
  tests.
- Run relevant lit filters for touched `test/Target/RVVMicrokernel` files.
- Run `git diff --check`, `git diff --cached --check`, and Trellis task
  validation.
