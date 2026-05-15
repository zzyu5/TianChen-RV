# RVV finite selected-source authority deletion

## Goal

Delete the remaining RVV finite selected-source / frontend-lowering authority
as a legality or planning source. RVV selected paths must not become legal, or
select add/sub/mul families, only because `tcrv_frontend_lowering`,
selected-source metadata, finite frontend contracts, or lookup-by-frontend
records name a finite family. Future executable RVV behavior must be rebuilt
from explicit extension-family ops through a materialized MLIR EmitC module
route; this task does not rebuild that route.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `2230ec5`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-15-rvv-body-verifier-descriptor-deletion`
  deleted `RVVBinaryMicrokernelBodyVerifier` and left the tree clean.
- `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp` still accepts finite RVV binary
  variants from selected-source metadata and kernel frontend-lowering metadata
  when no typed RVV body is present.
- `lib/Plugin/RVV/RVVBinaryPlanning.cpp` still resolves proposal family
  selection from `tcrv_frontend_lowering`, preserves variant selected-source
  metadata, and exposes a frontend-lowering proposal overload.
- `include/TianChenRV/Target/RVV/RVVBinaryFamily.h` still stores finite
  frontend markers, frontend contracts, and lookup-by-frontend helpers in the
  RVV family record.
- `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h` still exposes
  finite frontend family contracts and source frontend-lowering contract
  builders.
- Focused RVV plugin tests currently protect positive frontend/selected-source
  family selection and must be rewritten or deleted.

## Requirements

- Remove RVV legality authority from selected-source metadata on
  `tcrv.exec.variant`.
- Remove RVV legality authority from kernel-level `tcrv_frontend_lowering`.
- Remove RVV proposal/planning family selection from
  `tcrv_frontend_lowering`.
- Remove RVV proposal/planning source-kind preservation from
  variant selected-source metadata.
- Remove RVV finite frontend contract pointers and lookup-by-frontend helpers
  from RVV family records.
- Remove finite frontend family contract builders from
  `FiniteBinaryFrontendLowering.h` when they only support selected-source /
  frontend family authority.
- Keep neutral capability facts, selected vector-shape metadata, capacity
  metadata, and unsupported metadata-only RVV lowering boundaries when they do
  not select finite computation or make executable claims.
- Keep actual typed `tcrv_rvv.*` body authority fail-closed / metadata-only;
  typed bodies may identify their family by real op structure, not by
  selected-source metadata.
- Rewrite tests so metadata/frontend-only selected paths are rejected or absent
  rather than accepted.

## Acceptance Criteria

- [x] `tcrv_frontend_lowering` no longer selects an RVV finite binary family in
      RVV planning/proposal code.
- [x] `lookupRVVBinaryFamilyRegistrationByFrontendLowering` and RVV frontend
      contract lookups are removed from RVV family records.
- [x] `FiniteBinaryFrontendContract`,
      `FiniteBinarySourceFrontendLoweringContract`, and their finite family
      builder helpers are removed when no longer used.
- [x] `tcrv_rvv.selected_binary_source_kind` on an RVV variant no longer makes
      the variant legal.
- [x] Selected binary family/dtype/operator metadata without an actual typed
      RVV body fails closed as a missing rebuild gap.
- [x] RVV proposals from ordinary RVV capability facts no longer emit finite
      selected-source metadata or descriptor-local element-count metadata.
- [x] Focused RVV/plugin tests no longer protect successful finite-family
      selection from frontend/metadata-only sources.
- [x] No replacement selected-source authority, compatibility wrapper,
      descriptor path, direct C exporter, Common EmitC rebuild, new extension
      op, or ssh RVV evidence campaign is added.
- [x] Focused ref-scans report remaining occurrences truthfully and show that
      active RVV legality/planning no longer consumes frontend/selected-source
      metadata as authority.
- [x] Focused builds/tests pass, or failures are recorded as expected
      deletion gaps without restoring the deleted authority.
- [x] `git diff --check`, Trellis validation, archive/finish, final clean
      `git status --short`, and one coherent commit are produced if complete.

## Non-goals

- No Common EmitC implementation.
- No new RVV lowering.
- No new extension ops.
- No replacement selected-source authority.
- No compatibility or quarantine mode.
- No direct C exporter restoration.
- No broader finite-family coverage.
- No ssh RVV runtime/correctness/performance evidence.
- No Python compiler-core semantics.
- No deletion of neutral capability facts or non-semantic artifact packaging
  unless directly tied to the selected-source/frontend authority being removed.

## Minimal Evidence

- Bounded ref-scan over directly related RVV/plugin/support/test paths for:
  `tcrv_frontend_lowering`,
  `selected_binary_source_kind`,
  `selected-source`,
  `sourceKind`,
  `smoke_probe_descriptor`,
  `FiniteBinaryFrontendContract`,
  `FiniteBinarySourceFrontendLoweringContract`, and
  `lookupRVVBinaryFamilyRegistrationByFrontendLowering`.
- Focused build targets for touched RVV/plugin tests.
- Focused C++ tests for RVV binary planning, legality, selected lowering
  boundary, and RVV extension plugin behavior.
- Affected lit/FileCheck cases if touched or broken by deletion.
- Full `check-tianchenrv` if coherent after focused validation.
- `git diff --check` and `git diff --cached --check`.
- `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks.
- `.trellis/spec/core-dialect/tcrv-exec-contract.md` keeps computation and
  hardware semantics out of `tcrv.exec`.
- `.trellis/spec/extension-plugins/rvv-plugin.md` currently contains some
  historical finite selected-source wording; code changes in this task follow
  the deletion-campaign direction and leave executable rebuild for a later
  task.
- `.trellis/spec/lowering-runtime/emitc-route.md` states the accepted route is
  extension-family ops -> EmitC -> C/C++ emitter.
- `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  requires extension-family ops and common interfaces rather than descriptor or
  backend-specific core branches.
- The archived body-verifier PRD is the evidence baseline for the immediately
  preceding deletion round.

## Completion Evidence

- Deleted `FiniteBinaryFrontendLowering.h`; no active include/code reference
  remains.
- Removed RVV family frontend-lowering fields, finite frontend contract
  pointers, lookup-by-frontend helpers, frontend proposal overloads, and
  planner `sourceKind` contract exposure.
- RVV planning now rejects non-empty kernel `tcrv_frontend_lowering`; RVV
  variant legality now rejects `tcrv_rvv.selected_binary_source_kind` and
  requires an actual typed RVV body for finite selected metadata.
- Ordinary RVV capability proposals keep vector-shape/capacity metadata only;
  they no longer emit selected binary family/source or element-count metadata.
- Rewrote RVV/plugin and lit tests so remaining frontend/selected-source
  occurrences are deletion-negative or `implicit-check-not` checks.
- Focused deleted-contract ref-scan over RVV/plugin/support/script/lit paths:
  no `FiniteBinaryFrontendContract`,
  `FiniteBinarySourceFrontendLoweringContract`,
  `lookupRVVBinaryFamilyRegistrationByFrontendLowering`, frontend contract
  field, frontend source-kind helper, planner `getSourceKind`, or default
  selected-source materialization helper remains in active code.
- Remaining `tcrv_frontend_lowering` / `selected_binary_source_kind`
  occurrences are RVV deletion-negative tests, rejection constants/messages,
  RVV dialect op-owned direct typed identity validation, or out-of-scope scalar
  tests.
- `cmake --build build --target tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-binary-variant-legality-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-extension-plugin-test -j2` passed.
- `./build/bin/tianchenrv-rvv-binary-planning-test`,
  `./build/bin/tianchenrv-rvv-binary-variant-legality-test`,
  `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Focused lit over touched RVV selected-shape, RVV microkernel, and
  `rvv-probe-to-mlir` cases passed: 4/4.
- Full `cmake --build build --target check-tianchenrv -j2` passed: 114/114.
- `git diff --check` passed.
