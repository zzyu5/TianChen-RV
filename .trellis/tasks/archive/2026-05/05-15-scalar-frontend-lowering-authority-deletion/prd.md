# Scalar frontend-lowering authority deletion

## Goal

Delete scalar fallback frontend-lowering authority as a finite-family proposal,
materialization, legality, or selected-emission source. Scalar fallback must not
select `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, or
`i64-vmul` from kernel metadata such as `tcrv_frontend_lowering`, from
`frontendLowering` fields, or from lookup-by-frontend family tables. Future
scalar executable behavior must be rebuilt from explicit scalar extension-family
ops and the common EmitC/artifact route; this task does not rebuild that route.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `99e7e18`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived RVV PRD deleted finite RVV frontend/selected-source
  authority and left scalar frontend authority explicitly out of scope.
- Current focused inspection found scalar fallback still consuming
  `tcrv_frontend_lowering` in
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
- Current focused inspection found `frontendLowering` and
  `lookupRVVScalarBinaryRegistrationByFrontendLowering` in
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`.
- Current focused inspection found positive scalar plugin tests that rely on
  `tcrv_frontend_lowering` selecting vadd/vsub/vmul families in
  `test/Plugin/ScalarExtensionPluginTest.cpp`.
- `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` still contains
  old wording that treats bounded frontend lowering as a scalar family selector;
  this is obsolete guidance for this deletion round.

## Requirements

- Remove scalar proposal gating authority from non-empty
  `tcrv_frontend_lowering`.
- Remove scalar descriptorless default materialization authority from
  non-empty `tcrv_frontend_lowering`.
- Remove scalar `frontendLowering` fields and lookup-by-frontend helpers from
  RVV-scalar bridge family records when they exist only to select scalar compute
  families.
- Keep neutral scalar fallback capability registration, proposal metadata,
  conservative fallback role, selected boundary metadata, typed scalar
  microkernel op validation, and deleted direct-C route diagnostics when they
  do not select finite computation from frontend metadata.
- Rewrite positive scalar frontend-selection tests so non-empty
  `tcrv_frontend_lowering` is rejected or ignored as deleted authority.
- Record any scalar executable gap as a rebuild gap rather than adding wrappers,
  compatibility selectors, direct exporters, or new lowering routes.
- Update obsolete scalar fallback spec wording that protects frontend metadata
  as selected scalar family authority.

## Acceptance Criteria

- [x] `tcrv_frontend_lowering` no longer selects scalar finite binary families
      in scalar plugin proposal or descriptorless materialization code.
- [x] `frontendLowering` and
      `lookupRVVScalarBinaryRegistrationByFrontendLowering` are removed from
      RVV-scalar bridge family records.
- [x] Positive scalar tests no longer depend on frontend metadata selecting
      vadd/vsub/vmul or i64 families.
- [x] Non-empty scalar `tcrv_frontend_lowering` is rejected or ignored as
      deleted old authority with focused test coverage.
- [x] Existing explicit typed scalar microkernel validation remains typed-op
      based rather than metadata-selected.
- [x] No Common EmitC implementation, new scalar lowering, new scalar extension
      op, descriptor wrapper, compatibility mode, or replacement metadata
      selector is added.
- [x] Focused ref-scans report remaining occurrences of
      `tcrv_frontend_lowering`, `frontendLowering`,
      `lookupRVVScalarBinaryRegistrationByFrontendLowering`,
      `getRequestedScalarBinaryFamily`, and
      `ScalarBinaryFamilyDescriptor` truthfully.
- [x] Focused scalar plugin build/test and affected checks pass, or failures
      are recorded as expected deletion gaps without restoring the selector.
- [x] `git diff --check`, Trellis validation, finish/archive, final clean
      `git status --short`, and one coherent commit are produced if complete.

## Non-goals

- No Common EmitC implementation.
- No new scalar lowering.
- No new scalar extension ops.
- No new route/exporter code.
- No descriptor wrappers or compatibility modes.
- No broad scalar feature coverage.
- No ssh RVV runtime/correctness/performance evidence.
- No restoration of RVV selected-source/frontend authority.
- No deletion of neutral capability descriptors or non-semantic packaging
  unless directly required by removing this scalar selector.

## Minimal Evidence

- Bounded ref-scan over scalar/RVV bridge paths and scalar plugin tests for:
  `tcrv_frontend_lowering`, `frontendLowering`,
  `lookupRVVScalarBinaryRegistrationByFrontendLowering`,
  `getRequestedScalarBinaryFamily`, `ScalarBinaryFamilyDescriptor`, and scalar
  positive frontend-selection tests.
- Focused build target for `tianchenrv-scalar-extension-plugin-test`.
- Focused execution of `./build/bin/tianchenrv-scalar-extension-plugin-test`.
- Affected lit/FileCheck tests if scalar frontend deletion breaks them.
- Full `check-tianchenrv` if practical after focused validation.
- `git diff --check` and `git diff --cached --check`.
- `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- `.trellis/spec/index.md` states descriptor-driven computation is invalid as
  long-term architecture and the main stack is C++/MLIR/LLVM/TableGen/CMake.
- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks and independent backend dialect framing.
- `.trellis/spec/core-dialect/tcrv-exec-contract.md` keeps computation and
  hardware semantics out of `tcrv.exec`.
- `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` is relevant but
  currently contains obsolete frontend-lowering selected-family wording that
  must be corrected by this deletion round.
- `.trellis/spec/lowering-runtime/emitc-route.md` states the correct future
  route is extension family ops -> EmitC -> C/C++ emitter, not frontend metadata
  or descriptor-selected computation.
- Archived PRD
  `.trellis/tasks/archive/2026-05/05-15-rvv-finite-selected-source-authority-deletion/prd.md`
  is the immediate baseline for deleting the analogous RVV authority.

## Completion Evidence

- Removed `frontendLowering` and
  `lookupRVVScalarBinaryRegistrationByFrontendLowering` from
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`.
- Removed scalar plugin lookup-by-frontend, requested-family helper, and
  frontend-driven descriptorless materialization from
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
- Scalar proposal support now declines kernels with non-empty
  `tcrv_frontend_lowering`, and direct proposal/boundary materialization fail
  closed with a deleted-authority diagnostic.
- Descriptorless scalar fallback with no deleted frontend marker still
  materializes the bounded default `tcrv_scalar.i32_vadd_microkernel`.
- Rewrote `test/Plugin/ScalarExtensionPluginTest.cpp` so former vadd/vsub/vmul
  and i64 frontend-selection positives are deletion-negative proposal or
  boundary-materialization checks.
- Rewrote the i64 scalar artifact fixture to use an explicit
  `tcrv_scalar.i64_vmul_microkernel` body instead of
  `tcrv_frontend_lowering`; deleted the stale unused vmul frontend fixture.
- Updated scalar fallback spec wording so `tcrv_frontend_lowering` is a deleted
  old authority, not a scalar family selector.
- Focused deleted-authority ref-scan: no
  `lookupRVVScalarBinaryRegistrationByFrontendLowering`,
  `getRequestedScalarBinaryFamily`, or `ScalarBinaryFamilyDescriptor`
  occurrence remains; `frontendLowering` remains only as the RVV deletion-local
  variable in `lib/Plugin/RVV/RVVBinaryPlanning.cpp`, outside this scalar
  selector.
- Remaining `tcrv_frontend_lowering` occurrences are deletion-negative scalar
  tests, prior RVV deletion-negative tests, script `implicit-check-not`
  coverage, scalar/RVV rejection constants/messages, and spec/PRD deletion
  guidance.
- `cmake --build build --target tianchenrv-scalar-extension-plugin-test
  tcrv-opt tcrv-translate -j2` passed.
- `./build/bin/tianchenrv-scalar-extension-plugin-test` passed.
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  passed.
- `./build/bin/tianchenrv-target-artifact-export-test` passed.
- Manual affected artifact route check:
  `tcrv-opt test/Target/ArtifactExport/Inputs/scalar-i64-vmul-microkernel-source.txt
  --tcrv-materialize-selected-lowering-boundaries
  --tcrv-materialize-emission-plans` passed and materialized an unsupported
  deleted direct-C emission diagnostic from the explicit typed body.
- Manual affected export check:
  `tcrv-translate --tcrv-export-target-source-artifact` on the post-planning
  i64 scalar fixture failed closed with `requires exactly one supported source
  artifact emission-plan route; found none`.
- Full `cmake --build build --target check-tianchenrv -j2` passed: 114/114.
- `git diff --check` passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-15-scalar-frontend-lowering-authority-deletion` passed.
