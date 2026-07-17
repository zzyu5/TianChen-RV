# Stage2 RVV compare/select mask route-family target artifact validator migration

## Goal

Register a target-owned RVV compare/select mask artifact route-family validator
and migrate compare/select mask and compare-produced computed-mask memory
artifact acceptance out of `RVVTargetSupportBundle.cpp`. The central RVV target
artifact bridge should remain responsible only for generic route rebuild,
candidate/exporter shape, selected-boundary checks, runtime ABI consistency,
residue rejection, neutral artifact mechanics, and dispatch into the
route-family validator registry.

## Direction Source

- Direction title: `Switch: Stage2 RVV compare/select mask route-family target
  artifact validator migration`.
- Module owner: target-owned RVV artifact route-family validation registry entry
  for compare/select and computed-mask-memory semantic artifact validation.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `ecf90880 rvv: migrate macc artifact validation owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I already know

- Commit `ecf90880` completed the MAcc artifact route-family validator
  migration and left the worktree clean.
- The previous MAcc task proved the target-owned validator pattern: semantic
  MAcc artifact acceptance moved into
  `RVVTargetArtifactRouteFamilyValidation.cpp`, while
  `RVVTargetSupportBundle.cpp` kept generic target bridge mechanics.
- `RVVTargetArtifactRouteFamilyValidation.cpp` currently registers `macc` and
  `widening-dot-reduction` validators.
- `RVVTargetSupportBundle.cpp` still owns compare/select mask family
  classifiers, mask/tail policy constants, provider-fact checks, and
  candidate-mirror checks for:
  - `CmpSelect`;
  - `ComputedMaskSelect`;
  - `RuntimeScalarCompareSelect`;
  - `RuntimeScalarDualCompareMaskAndSelect`;
  - `ComputedMaskUnitLoadStore`;
  - `ComputedMaskStridedStore`;
  - `ComputedMaskStridedLoadUnitStore`;
  - `ComputedMaskIndexedGatherLoadUnitStore`;
  - `ComputedMaskIndexedScatterStoreUnitLoad`.
- The existing compare/select provider route facts include compare predicate,
  compare intrinsic, mask producer/source, mask role, mask memory form,
  select layout, mask/tail policy owner, runtime ABI order, type/header
  mappings, route operand binding facts, provider-supported mirror, and
  computed-mask memory layout facts.
- This round is not allowed to add new RVV operation coverage, selected-body
  realization behavior, intrinsic cases, or runtime claims.

## Requirements

1. Register a production compare/select mask route-family validator in
   `RVVTargetArtifactRouteFamilyValidation`.
2. Dispatch compare/select mask and compare-produced computed-mask memory
   artifact candidates from the central bridge into the route-family validator
   registry.
3. Move compare/select mask semantic provider-fact validation out of
   `RVVTargetSupportBundle.cpp`.
4. Move compare/select mask candidate-mirror validation out of
   `RVVTargetSupportBundle.cpp`.
5. Keep `RVVTargetSupportBundle.cpp` limited to generic bridge
   responsibilities:
   - artifact candidate/exporter shape;
   - selected variant and typed body rebuild;
   - generic rebuilt provider route/candidate consistency entry;
   - selected-boundary and source-op provenance checks;
   - descriptor/direct-C/source-export residue rejection;
   - runtime ABI consistency;
   - neutral materialization/export packaging;
   - metadata evidence listing;
   - dispatch into the route-family validator registry.
6. The compare/select mask validator must consume rebuilt
   `TCRVEmitCLowerableRoute` and
   `RVVSelectedBodyEmitCRouteDescription` facts to validate:
   - route id/provider support and operation family;
   - required RVV headers;
   - VL, vector, mask, index-vector, and C type mappings;
   - runtime ABI order and ABI parameter mirrors;
   - route operand binding plan and operand summary mirrors;
   - compare predicate kind and compare statement callee plan;
   - mask result, role, source, producer source, memory form, and mask/tail
     policy plan owner;
   - select layout and dual-compare mask composition when applicable;
   - inactive-lane and masked-passthrough layout for plain compare/select and
     compare-produced computed-mask memory consumers;
   - computed-mask memory layout for compare-produced masked memory consumers;
   - plain compare-select, computed-mask select, and computed-mask memory
     route-family plan mirrors.
7. Unsupported, stale, missing, or mismatched route id, metadata, predicate,
   mask producer, mask/tail policy, type/header, ABI, statement-plan,
   selected-boundary, source-front-door, direct-C, descriptor, exact-intrinsic,
   common-EmitC, route operand binding, or candidate mirror facts must fail
   closed with targeted diagnostics.
8. Preserve existing MAcc and widening-dot validator behavior as
   non-regressions.
9. Do not introduce direct-route-entry, source-front-door, descriptor,
   route-id, artifact-name, ABI-string, exact-intrinsic, script-derived,
   common-EmitC-derived, pre-realized-fixture-only, or legacy-i32 route
   authority.

## Acceptance Criteria

- [x] `RVVTargetArtifactRouteFamilyValidation` registers and dispatches a
      compare/select mask route-family validator for all in-scope compare/select
      producer and compare-produced computed-mask memory route descriptions.
- [x] `RVVTargetSupportBundle.cpp` no longer owns duplicated compare/select
      mask semantic artifact validation bodies or compare/select mask-specific
      candidate mirror checks.
- [x] The new validator checks headers, type mappings, ABI order/mapping,
      route operand binding mirrors, predicate kind, mask producer/source,
      mask/tail policy plan/owner, select or masked-memory layout, statement
      plan callees, provider-supported mirror, route-family plan mirrors, and
      route/candidate family consistency using rebuilt provider/body facts.
- [x] Focused target artifact/export or plugin tests prove compare/select mask
      provider facts are consumed by the family validator in the production
      path.
- [x] Fail-closed coverage exists for stale or missing provider/header/type/ABI,
      predicate, mask producer, mask/tail policy, statement-plan, provider
      support, route operand binding, and route-family candidate mirrors, using
      existing infrastructure or focused additions.
- [x] Representative non-regression passes for plain compare/select,
      computed-mask select, runtime-scalar compare/select, dual-compare
      mask/select, and compare-produced computed-mask memory artifact consumers.
- [x] Existing MAcc and widening-dot route-family validator behavior remains
      covered and non-regressed.
- [x] `git diff --check` passes.
- [x] Focused build/test targets for `TianChenRVTarget`, `tcrv-translate`,
      `tcrv-opt`, relevant target artifact/export tests, and RVV plugin tests
      pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Bounded authority-leak scan over touched production files and tests
      confirms no executable artifact claim depends on central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [x] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out of Scope

- Migrating elementwise arithmetic, conversion/dtype policy, runtime scalar
  splat-store, segment2, standalone reduction, MAcc, widening-dot, or unrelated
  route families.
- Changing compare/select computation semantics, selected-body realization,
  provider route construction, runtime `n`/AVL values, dispatch/fallback
  behavior, common EmitC materialization, or target artifact format.
- Adding new RVV operation coverage, new selected-body realization, new
  intrinsic cases, dtype/LMUL clone batches, high-level Linalg/frontend
  lowering, dashboards, broad smoke matrices, or evidence-only work.
- Claiming new runtime/correctness/performance evidence without real `ssh rvv`
  execution.
- Reintroducing descriptor-driven computation, source-front-door positive
  routes, direct route-entry compatibility, route-id authority, artifact-name
  authority, ABI-string authority, exact-intrinsic authority, common-EmitC
  semantic authority, script-derived authority, or legacy-i32 authority.

## Technical Approach

1. Keep the Trellis task current and bounded to the supplied Hermes direction.
2. Move compare/select mask family classifiers, mask/tail policy constants,
   provider-fact validators, and candidate-mirror validators from
   `RVVTargetSupportBundle.cpp` into
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Add a compare/select mask validator registration next to the existing MAcc
   and widening-dot validators.
4. Remove the central bridge call to the old compare/select mask provider-fact
   helper and rely on the registry for provider-fact validation.
5. Remove compare/select mask-specific branches from the central candidate
   mirror validator and rely on the registry for candidate-mirror validation.
6. Add or strengthen focused negative tests that mutate compare/select mask
   mirrors and prove failures are produced through the migrated validator path.
7. Run focused target/export and RVV plugin tests, then `git diff --check`,
   authority scan, and `check-tianchenrv`.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Validate/start this Trellis task after PRD and context files are written.
2. Build focused targets:
   `cmake --build build --target TianChenRVTarget tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`.
3. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit filters for directly related target artifact consumers:
   compare/select, computed-mask select, runtime-scalar compare/select,
   runtime-scalar dual compare mask/select, and computed-mask memory artifacts.
5. Run `git diff --check`.
6. Run a bounded authority scan over touched production and test files.
7. Run `cmake --build build --target check-tianchenrv`.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-macc-artifact-route-family-validator-migration/prd.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-compare-select-route-family-owner/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`

Primary evidence consumers:

- `test/Plugin/RVVExtensionPluginTest.cpp`
- target artifact export tests for `cmp_select`, `computed_mask_select`,
  `runtime_scalar_compare_select`,
  `runtime_scalar_dual_compare_mask_and_select`, and compare-produced
  computed-mask memory consumers.

## Open Questions

None blocking. Repository evidence identifies a bounded production gap:
compare/select mask semantic artifact validation still sits in the central RVV
target bridge even though the route-family validator registry now owns MAcc and
widening-dot artifact validation.

## Definition Of Done

Compare/select mask and compare-produced computed-mask memory artifact
acceptance is owned by the target route-family validator registry, the central
bridge contains only generic target artifact mechanics and registry dispatch,
focused fail-closed and non-regression evidence passes, authority scans are
clean, Trellis state is truthful, and one coherent commit records the completed
task.

## Completion Notes

- Production ownership moved to `RVVTargetArtifactRouteFamilyValidation.cpp` via
  a registered `compare-select-mask` validator.
- `RVVTargetSupportBundle.cpp` no longer contains the compare/select mask
  semantic validator helpers or compare/select mask-specific candidate mirror
  branch; it retains generic bridge mechanics, metadata evidence listing,
  residue rejection, ABI/selected-boundary checks, and registry dispatch.
- Focused negative tests cover stale provider support, operand binding, ABI,
  headers, type mapping, predicate/layout, mask producer, mask/tail policy plan
  and owner, and computed-mask memory layout facts.
- Representative compare/select and computed-mask target artifact consumers
  passed, the RVV extension plugin smoke test passed, `git diff --check`
  passed, and `check-tianchenrv` passed 456/456.
- No `ssh rvv` runtime/correctness/performance evidence was claimed.
