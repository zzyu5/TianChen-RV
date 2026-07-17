# delete RVV metadata-only first-slice route

## Goal

Erase the RVV metadata-only first-slice route as active compiler authority. A
bare RVV capability/high-level or no-body input must not synthesize
`rvv_first_slice`, `metadata_only_first_slice` legality authority,
`tcrv_rvv.lowering_boundary`, unsupported RVV emission-plan/runtime-ABI fields,
artifact-route metadata, or target-support route metadata.

## What I Already Know

- The worktree started clean at `42883cb chore(scalar): erase fallback metadata route`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp` still proposes
  `rvv_first_slice`, sets `policy = "metadata_only_first_slice"`, materializes
  `tcrv_rvv.lowering_boundary`, and builds unsupported RVV emission plans with
  `rvv-unsupported-metadata-boundary`, `rvv-none-executable-unsupported`,
  `unsupported-plugin-runtime-abi`, `unsupported-emission-runtime-abi`, and
  `unsupported-emission-diagnostic`.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and
  `lib/Dialect/RVV/IR/RVVDialect.cpp` still define and verify the RVV
  lowering-boundary metadata op.
- Several lit and C++ tests still protect the stale selected RVV metadata
  route, including variant materialization, lowering-boundary, plugin legality,
  emission readiness/manifest, RVV microkernel, and RVV/scalar dispatch tests.

## Requirements

- Delete or fail-close the RVV no-body/high-level proposal path that turns
  capability facts into `rvv_first_slice`.
- Remove `metadata_only_first_slice` as an active RVV legality and selection
  contract.
- Remove RVV plugin materialization of `tcrv_rvv.lowering_boundary` from a
  selected metadata-only variant.
- Remove unsupported RVV metadata-route emission-plan/runtime-ABI/artifact
  fields from the RVV plugin route.
- Remove RVV target-support bundle exposure of the deleted lowering-boundary
  route metadata.
- Rewrite or delete tests/spec text that treat the metadata-only RVV route as a
  valid active route.
- Preserve typed RVV dialect verification for explicit `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, and RVV dataflow ops.

## Acceptance Criteria

- [x] Bare RVV capability/high-level input without explicit typed RVV
      extension-family body produces no selected RVV variant.
- [x] The same input produces no RVV lowering boundary, emission plan, runtime
      ABI, artifact route, or metadata-route diagnostics.
- [x] RVV lowering-boundary syntax is no longer an active RVV dialect/compiler
      surface, or any remaining reference is fail-closed and not route
      authority.
- [x] Existing explicit typed RVV dialect tests still verify setvl/with_vl and
      dataflow surfaces.
- [x] Focused ref scan shows no active-surface residue for:
      `metadata_only_first_slice`, `rvv_first_slice`,
      `RVV metadata-only first slice`,
      `rvv-unsupported-metadata-boundary`,
      `rvv-none-executable-unsupported`,
      `unsupported-plugin-runtime-abi`,
      `unsupported-emission-runtime-abi`, and
      `unsupported-emission-diagnostic`, excluding archives/workspace/build/run
      artifacts and non-RVV scalar/offload specs where appropriate.
- [x] Focused build and lit/C++ tests for changed RVV surfaces are run.
- [x] `ninja -C build check-tianchenrv` is attempted or its blocker is
      reported without restoring the deleted route.
- [x] `git diff --check` passes.

## Completion Notes

- RVV no-body/high-level proposal now records a recoverable deleted-route
  decline and produces no RVV variant.
- RVV plugin legality/cost now require an explicit typed `tcrv_rvv` body.
- RVV lowering-boundary op definition/verifier and active materialization path
  were removed; remaining syntax coverage is negative unknown-op/fail-closed
  coverage only.
- RVV emission planning no longer returns unsupported runtime ABI or artifact
  metadata for the deleted route.
- `check-tianchenrv` passes with 89/89 lit tests.

## Out Of Scope

- No new RVV lowering.
- No Common EmitC route.
- No executable plugin template.
- No source-to-RVV rebuild.
- No descriptor, direct C exporter, runtime ABI, ssh rvv evidence, or
  performance work.
- No compatibility alias, helper wrapper, replacement unsupported route, or new
  metadata-only route.
- No scalar fallback metadata-route restoration.
- No finite RVV family expansion beyond deletion needed for this first-slice
  route.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Relevant code inspected:
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/RVVCapabilityProfile.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Transforms/VariantMaterialization.cpp`,
  `lib/Plugin/LoweringBoundary.cpp`, and
  `lib/Transforms/EmissionReadiness.cpp`.
- Deletion gaps must be reported as missing new architecture, not repaired by
  resurrecting descriptor/direct-C/unsupported metadata routes.
