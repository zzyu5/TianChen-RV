# RVV selected-source metadata quarantine deletion

## Goal

Delete the remaining RVV selected-source/source-kind metadata as a recognized
finite-family authority, source identity, or negative quarantine path. Typed RVV
extension-family microkernel structure, selected variant links, and selected
vector-shape/config checks must carry the remaining bounded RVV path.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `1013d12`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-15-no-body-rvv-pipeline-expectation-deletion/prd.md`
  restored full `check-tianchenrv` to 114/114 after deleting public no-body RVV
  pipeline expectations.
- `.trellis/spec/index.md` keeps compiler implementation in the
  C++/MLIR/LLVM/TableGen/CMake/lit stack and rejects descriptor-driven
  computation as the long-term architecture.
- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks and independent backend dialect framing.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says finite RVV binary
  proposals require explicit typed extension-family body authority before
  selected family/dtype/shape/route metadata can matter.
- Current focused scan found active references to
  `tcrv_rvv.selected_binary_source_kind`,
  `stale-selected-source-metadata`, `direct-typed-microkernel-body`,
  `getRVVSelectedBinarySourceKindAttrName`, and `selected-source metadata` in
  RVV planning, legality, dialect ODS/verifier code, RVV C++ tests, RVV target
  comments, and RVV/lowering-runtime specs.

## Requirements

- Remove `tcrv_rvv.selected_binary_source_kind` as a plugin planning or
  legality-recognized field.
- Delete variant-level `rejectDeletedSelectedSourceMetadata` and stop using
  source-kind metadata as a special negative quarantine path.
- Remove directly coupled RVV dialect op-owned source-kind/source-identity
  verifier and ODS plumbing when it only preserves the same old selected-source
  identity.
- Rewrite or delete stale selected-source/source-kind C++ test fixtures instead
  of preserving them as negative quarantine coverage.
- Update active RVV specs/comments so they no longer describe selected-source
  identity or source-kind fields as a current active RVV boundary.
- Keep positive typed-body fixtures working through actual
  `tcrv_rvv.*_microkernel` ops, `selected_variant`, required capabilities, and
  selected vector-shape/config checks.
- Record any newly exposed missing route as a rebuild gap; do not add a
  compatibility selector, helper wrapper, replacement source-kind field, or new
  lowering path.

## Acceptance Criteria

- [x] No active RVV plugin/dialect/test/spec surface uses
      `tcrv_rvv.selected_binary_source_kind`,
      `selected-source metadata`, `stale-selected-source-metadata`,
      `direct-typed-microkernel-body`, or
      `getRVVSelectedBinarySourceKindAttrName` as authority or quarantine.
- [x] `RVVBinaryVariantLegality` no longer has a selected-source quarantine
      rejection branch; variant legality still rejects selected binary metadata
      without an actual typed RVV extension-family body.
- [x] RVV dialect ODS/verifier code no longer makes op-owned source-kind
      metadata meaningful.
- [x] Focused RVV C++ tests pass or expose only truthful deletion/rebuild gaps:
      `tianchenrv-rvv-binary-variant-legality-test`,
      `tianchenrv-rvv-selected-lowering-boundary-test`, and if touched,
      `tianchenrv-rvv-binary-planning-test` and
      `tianchenrv-rvv-extension-plugin-test`.
- [x] Directly affected lit tests pass if code generation or parsing surfaces
      are impacted.
- [x] Focused ref-scan over active RVV plugin/dialect/tests/specs reports the
      deleted selected-source/source-kind tokens truthfully.
- [x] `check-tianchenrv` is run if practical.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
- [x] Trellis task status, context, journal, archive state, and one coherent
      commit are produced if the round completes.

## Non-goals

- No Common EmitC lowering implementation.
- No new RVV lowering.
- No new RVV ops.
- No new artifact/export route.
- No runtime ABI modeling.
- No `ssh rvv` runtime/correctness/performance evidence.
- No helper wrappers, compatibility selectors, or replacement source-kind field.
- No restoration of no-body RVV first-slice synthesis.
- No scalar/offload/direct-C cleanup unless a file is directly coupled to
  deleting this RVV selected-source metadata path.

## Minimal Evidence

- Focused ref-scan over active RVV plugin/dialect/tests/specs for
  `selected_binary_source_kind`, `selected-source metadata`,
  `stale-selected-source-metadata`, `direct-typed-microkernel-body`, and
  `getRVVSelectedBinarySourceKindAttrName`.
- Build and run:
  `tianchenrv-rvv-binary-variant-legality-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`.
- Also build and run:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test` if planning/plugin surfaces are
  touched.
- Run directly affected lit tests if parsing or materialization surfaces are
  impacted.
- Run `check-tianchenrv` if practical.
- Run `git diff --check`, `git diff --cached --check`, and
  `python3 ./.trellis/scripts/task.py validate`.

## Technical Notes

- This round is deletion/refactor-only. The route after deletion remains the
  existing explicit typed RVV microkernel body plus selected-vector-shape/config
  legality path; rebuilding Common EmitC lowering is out of scope.
- Remaining selected binary family/dtype/operator variant metadata is not a
  source-kind authority. It must remain subordinate to actual typed RVV body
  authority if retained.
- If old metadata-only variants become less specially diagnosed after deletion,
  that is intended: selected-source metadata should no longer be a recognized
  active quarantine category.

## Completion Evidence

- Removed `getRVVSelectedBinarySourceKindAttrName` and the backing
  `tcrv_rvv.selected_binary_source_kind` planning constant.
- Deleted `rejectDeletedSelectedSourceMetadata` from
  `RVVBinaryVariantLegality`; selected binary metadata and element-count
  metadata still require actual typed RVV extension-family body authority.
- Removed op-owned selected binary source identity fields from
  `tcrv_rvv.lowering_boundary` and all finite RVV microkernel ODS definitions,
  plus the RVV dialect verifier branch that made those fields meaningful.
- Rewrote RVV C++ tests by deleting stale selected-source quarantine fixtures
  and routing metadata-without-body cases through the existing typed-body
  requirement diagnostic.
- Updated active RVV and lowering-runtime specs so selected-boundary metadata is
  no longer described as route/source-kind authority.
- Added no compatibility layer, helper wrapper, replacement source-kind field,
  new RVV op, new lowering route, Common EmitC implementation, runtime ABI
  model, or `ssh rvv` claim.
- Focused active ref-scan over `lib`, `include`, `test`, and `.trellis/spec`
  found no remaining matches for `selected_binary_source_kind`,
  `selected-source`, `stale-selected-source-metadata`,
  `direct-typed-microkernel-body`, `getRVVSelectedBinarySourceKindAttrName`, or
  `source-kind`.
- Focused build passed:
  `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin
  tianchenrv-rvv-binary-variant-legality-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-rvv-binary-variant-legality-test`;
  `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`;
  `./build/bin/tianchenrv-rvv-binary-planning-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Directly affected lit passed from `build/test`: `python3
  /usr/lib/llvm-20/build/utils/lit/lit.py -sv Dialect/RVV
  Transforms/VariantMaterialization/plugin-variant-materialization-builtin.mlir
  Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape.mlir
  Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape-invalid.mlir
  Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir`
  with 14/14 passing.
- Full `cmake --build build --target check-tianchenrv -j2` passed with 114/114
  tests.
- `git diff --check`, `git diff --cached --check`, and
  `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-15-rvv-selected-source-metadata-quarantine-deletion`
  passed.
