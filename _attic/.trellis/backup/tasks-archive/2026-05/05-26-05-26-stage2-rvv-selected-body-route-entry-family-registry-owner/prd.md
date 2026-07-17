# Stage2 RVV Selected-Body Route-Entry Family Registry Owner

## Goal

Introduce or strengthen a plugin-local RVV selected-body route-entry family
registry/owner so active selected-body direct route-entry families are
registered, validated, and dispatched through a structured RVV-owned boundary
instead of broad one-off predicate/planning/provider wiring. This round centers
on migrating the computed-mask segment2 update route and at least one adjacent
segment2 memory route behind that owner while preserving typed-body authority,
provider-built route construction, fail-closed diagnostics, generated-bundle
behavior, and focused runtime evidence.

## What I Already Know

- Session start had no `.trellis/.current-task`; this task was created from the
  Hermes direction brief.
- Current HEAD is `d8556f0a rvv: add computed-mask segment2 update route`.
- The previous completed route work made computed-mask segment2 update
  executable, but its changed-file shape suggests the next bottleneck is owner
  structure for active route-entry families.
- The prior archived task
  `05-26-05-26-stage2-rvv-selected-body-multi-family-route-composition-owner`
  implemented the composed update route itself. This task should not add another
  new route solely as proof.
- Current RVV authority chain must remain:
  `tcrv.exec` selected RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin-owned selected-body realization / route-family validation /
  route provider -> `TCRVEmitCLowerableRoute` -> neutral common EmitC ->
  generated RVV artifact -> `ssh rvv` evidence when runtime/correctness is
  claimed.
- Registry entries must not become route-id, metadata, script, ABI string,
  artifact-name, descriptor, exact-intrinsic, source-front-door, common EmitC,
  or legacy-i32 authority.

## Requirements

- Define one coherent production owner interface for RVV selected-body
  route-entry families. It may improve an existing owner if one already exists.
- Migrate active production consumers behind that owner, including:
  - computed-mask segment2 update;
  - at least one adjacent segment2 memory route such as computed-mask segment2
    load/store or plain segment2 interleave/deinterleave.
- The owner must dispatch only after typed selected-body structure carries the
  required facts: op kind, dtype/config, memory form, segment factor, mask facts,
  arithmetic facts when applicable, stream roles, runtime `n`/AVL, VL/tail/mask
  policy, operand bindings, and selected RVV capability.
- Realization/planning/provider code must continue deriving route support from
  typed body/config/runtime/mask/memory/arithmetic facts and must fail closed for
  stale metadata, stale route ids, missing producer dependencies, wrong ABI
  roles, mismatched memory forms, segment factor mismatches, unsupported op
  kinds, or incomplete route-family facts.
- Delete or demote duplicated one-off predicate/allowlist logic made redundant
  by the registry. Do not keep a compatibility wrapper that preserves the old
  ad hoc authority.
- Keep common EmitC/export neutral: common materialization may consume
  provider-built route facts but must not choose RVV semantics.
- Update scripts and tests only as active consumers/evidence of the production
  route path; they must not decide support.

## Acceptance Criteria

- [x] Production C++ changes introduce or strengthen the RVV plugin-local
      selected-body route-entry family registry/owner boundary.
- [x] Computed-mask segment2 update and at least one adjacent segment2 memory
      route are migrated behind the owner, reducing one-off selected-body
      predicate/planning/provider wiring.
- [x] Focused C++ and/or lit tests prove migrated consumers still derive route
      support from typed body/config/runtime/mask/memory/arithmetic facts.
- [x] Focused negative tests prove fail-closed diagnostics for unsupported or
      stale metadata/route-id cases and malformed typed-body facts.
- [x] Generated-bundle dry-run covers computed-mask segment2 update and one
      migrated adjacent segment2 path.
- [x] Real `ssh rvv` evidence covers at least the composed computed-mask
      segment2 update unless exact byte-for-byte artifact preservation is
      proven; if preservation is used, one real `ssh rvv` run still covers a
      migrated representative.
- [x] Focused non-regression covers computed-mask segment2 store/load, segment2
      deinterleave/interleave, masked elementwise, reduction, scalar-broadcast,
      conversion, runtime-scalar MAcc, compare/select, MAcc, contraction, and
      base memory route-entry paths.
- [x] Bounded touched-file authority scan finds no new positive legacy-i32,
      source-front-door, descriptor, ABI-string, artifact-name, script-derived,
      metadata-derived, route-id-derived, exact-intrinsic-derived, or
      common-EmitC-derived authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or any blocker is reported with exact command
      output and the task is left open with a precise continuation point.
- [ ] Final git status is clean after one coherent commit if the task is
      complete.

## Out Of Scope

- Adding a new RVV operation solely to prove the registry.
- Broad route matrices, dtype/LMUL clone batches, gather/scatter expansion, or
  Stage 2 coverage expansion beyond the selected consumers.
- High-level frontend/Linalg routes, source-front-door positive routes,
  dashboard/report-only work, global autotuning, or future plugin work.
- Moving RVV semantic choice into common EmitC/export.
- Weakening existing owners for computed-mask segment2 update/store/load, plain
  segment2 interleave/deinterleave, masked elementwise, reduction,
  scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
  contraction, or base memory routes.

## Technical Notes

- Relevant first-read files from the brief:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  archived task
  `05-26-05-26-stage2-rvv-selected-body-multi-family-route-composition-owner`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
- More technical findings will be added after inspecting the active code and
  relevant specs.

## Completion Notes

- Added `RVVSelectedBodySegment2RouteEntryFamilyOwner` in
  `RVVSelectedBodyRealization.h` and implemented the segment2 route-entry
  family registry in `RVVSelectedBodyRealization.cpp`.
- Migrated segment2 direct route-entry eligibility to registry dispatch for:
  computed-mask segment2 load, computed-mask segment2 store, computed-mask
  segment2 update, plain segment2 deinterleave, and plain segment2 interleave.
- Computed-mask segment2 update is now classified by its own owner and is not
  accepted through the adjacent store owner; `arithmetic_kind = "add"` remains
  structural typed-body authority.
- Added C++ coverage for segment2 route-entry owner membership, exact-one
  classification, update-vs-store separation, and stale route metadata no-match
  behavior.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  segment2 route-entry family owner API and test/error contracts.
- Evidence artifacts:
  - `artifacts/tmp/stage2_route_entry_family_registry_owner/direct-pre-realized-computed-masked-segment2-update`
  - `artifacts/tmp/stage2_route_entry_family_registry_owner/direct-pre-realized-computed-masked-segment2-store`
  - `artifacts/tmp/stage2_route_entry_family_registry_owner/direct-pre-realized-computed-masked-segment2-update-ssh`
  - `artifacts/tmp/stage2_route_entry_family_registry_owner/direct-pre-realized-route-entry-nonregression`
  - `artifacts/tmp/stage2_route_entry_family_registry_owner/pre-realized-selected-boundary-extra-nonregression`
- Checks passed:
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  - `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - generated-bundle dry-run for computed-mask segment2 update
  - generated-bundle dry-run for computed-mask segment2 store
  - real `ssh rvv` generated-bundle run for computed-mask segment2 update,
    counts `0,7,16,23,257`, PASS
  - direct route-entry non-regression dry-run for segment2 load/store,
    deinterleave/interleave, standalone reduction, conversion, runtime-scalar
    MAcc, compare/select, MAcc, scalar-broadcast MAcc, contraction, and base
    memory
  - selected-boundary dry-run for masked elementwise and scalar-broadcast
    elementwise
  - production/test touched-file authority scan
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2` (390/390)
