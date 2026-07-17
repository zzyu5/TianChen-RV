# Stage2 RVV elementwise arithmetic route-family target artifact validator migration

## Goal

Register a target-owned RVV elementwise arithmetic artifact route-family
validator and migrate elementwise arithmetic artifact acceptance out of
`RVVTargetSupportBundle.cpp`. The validator must cover ordinary elementwise
add/sub/mul, masked elementwise arithmetic, directly related strided
elementwise add, and scalar-broadcast elementwise add/sub/mul artifact mirror
validation, while keeping `RVVTargetSupportBundle.cpp` as a neutral target
artifact bridge.

## Direction Source

- Direction title: `Switch: Stage2 RVV elementwise arithmetic artifact
  route-family validator migration`.
- Module owner: target-owned RVV artifact route-family validation registry
  entry for elementwise arithmetic routes, including plain, scalar-broadcast,
  and directly related strided elementwise artifact validation.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `98875b87 rvv: migrate compare select artifact validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I Already Know

- Commit `98875b87` completed the compare/select mask artifact validator
  migration, `check-tianchenrv` passed 456/456, the prior Trellis task was
  archived, and the current worktree began clean.
- `RVVTargetArtifactRouteFamilyValidation.cpp` currently registers
  `compare-select-mask`, `macc`, and `widening-dot-reduction` validators.
- `RVVTargetSupportBundle.cpp` still owns central semantic helpers for
  elementwise arithmetic artifact validation:
  - `isRVVElementwiseArithmeticRouteFamilyOperation`;
  - `isRVVMaskedElementwiseArithmeticRouteFamilyOperation`;
  - `isRVVStridedElementwiseArithmeticRouteFamilyOperation`;
  - `isRVVElementwiseArithmeticRouteFamilyDescription`;
  - `validateRVVElementwiseArithmeticRouteHeaders`;
  - `validateRVVElementwiseArithmeticRouteTypeMappings`;
  - `validateRVVElementwiseArithmeticRouteABIMappings`;
  - `validateRVVElementwiseArithmeticRouteStatementPlan`;
  - `validateRVVElementwiseArithmeticRoutePayloadFacts`.
- `RVVTargetSupportBundle.cpp` also still owns elementwise and
  scalar-broadcast route-family candidate mirror checks inside
  `validateRVVRouteMetadataMirrorsSelectedBody`.
- Existing provider facts already describe the route surface this task should
  validate: typed config, headers, C type mapping, ABI order and mapping,
  operation kind, memory form, operand binding plan, scalar-broadcast route
  family plan, elementwise arithmetic route family plan, runtime AVL/VL
  control, load/store/compute/splat/strided statement callees, mask facts for
  masked elementwise routes, strided memory facts for strided elementwise
  routes, and `provider_supported_mirror`.
- This round is not allowed to change RVV provider lowering, selected-body
  realization, generated C semantics, intrinsic spelling, route IDs, runtime
  behavior, source-front-door behavior, artifact naming, or executable/runtime
  claims.

## Requirements

1. Register a production `elementwise-arithmetic` route-family validator in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. Dispatch ordinary add/sub/mul, masked add/sub/mul, strided add, and
   scalar-broadcast add/sub/mul target artifact candidates through the target
   route-family validator registry.
3. Move elementwise arithmetic semantic provider-fact validation out of
   `RVVTargetSupportBundle.cpp`.
4. Move elementwise arithmetic and scalar-broadcast elementwise candidate mirror
   validation out of `RVVTargetSupportBundle.cpp`.
5. Keep `RVVTargetSupportBundle.cpp` limited to neutral bridge
   responsibilities:
   - candidate rebuild;
   - selected-boundary and runtime ABI consistency;
   - descriptor, direct-C, source-export, and source-front-door residue
     rejection;
   - neutral materialized route verification and artifact mechanics;
   - metadata evidence listing;
   - registry dispatch.
6. The new validator must consume rebuilt `TCRVEmitCLowerableRoute` plus
   `RVVSelectedBodyEmitCRouteDescription` facts to validate:
   - route family operation and memory form;
   - provider-derived required headers;
   - VL/vector/mask C type mappings;
   - runtime ABI order and route ABI mappings;
   - operation kind and route operand binding mirrors;
   - ordinary vector load, elementwise compute, store, setvl, loop, and runtime
     AVL/VL statement plan facts;
   - scalar-broadcast splat, vector load, elementwise compute, store, setvl,
     loop, and runtime AVL/VL statement plan facts;
   - masked elementwise mask type, compare/merge leaves, mask role/source,
     inactive-lane contract, passthrough layout, and stale mask residue;
   - strided elementwise load/store leaves, stride ABI bindings, strided memory
     layout, stride sources, and stale strided residue;
   - `provider_supported_mirror`;
   - `tcrv_rvv.elementwise_arithmetic_route_family_plan`;
   - `tcrv_rvv.scalar_broadcast_elementwise_route_family_plan`;
   - stale unrelated route-family mirrors such as widening conversion,
     compare/select, MAcc, reduction, segment2, and computed-mask memory when
     the selected route is an elementwise arithmetic route.
7. Unsupported, stale, missing, or mismatched header, type mapping, ABI mapping,
   operand binding, scalar-broadcast residue, strided metadata residue, memory
   form, provider-supported mirror, route-family mirrors, statement-plan callee
   facts, selected-boundary facts, source-front-door facts, descriptor facts,
   or direct-C/source-export residue must fail closed with targeted diagnostics.
8. Preserve existing compare/select, MAcc, widening-dot, conversion,
   runtime-scalar splat-store, segment2, standalone reduction, accumulation,
   and computed-mask memory behavior as non-regressions.
9. Do not introduce central ad hoc, name-derived, metadata-derived,
   descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived,
   common-EmitC-derived, source-front-door-derived, route-id-derived,
   exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only,
   or legacy-i32-derived route authority.

## Acceptance Criteria

- [ ] `RVVTargetArtifactRouteFamilyValidation.cpp` registers and dispatches a
      unique `elementwise-arithmetic` validator for ordinary, masked, strided,
      and scalar-broadcast elementwise route descriptions.
- [ ] `RVVTargetSupportBundle.cpp` no longer contains
      `validateRVVElementwiseArithmeticRoute*` semantic helper ownership or
      elementwise/scalar-broadcast-specific candidate mirror branches.
- [ ] The new validator checks provider-derived headers, type mappings, ABI
      order/mapping, route operand binding mirrors, operation/memory form,
      runtime AVL/VL plan, statement callees, masked facts, scalar-broadcast
      facts, strided facts, provider-supported mirror, and route-family plan
      mirrors using rebuilt provider/body facts.
- [ ] Focused target artifact/export tests prove the production route-family
      validator consumes ordinary elementwise, scalar-broadcast elementwise, and
      strided elementwise artifact facts.
- [ ] Negative coverage exists for stale or missing provider support, route
      operand binding, ABI order, header, type mapping, memory form,
      scalar-broadcast plan, strided memory/stride facts, statement-plan callee,
      and elementwise route-family candidate mirrors.
- [ ] Existing compare/select, MAcc, widening-dot, conversion, runtime-scalar
      splat-store, segment2, standalone reduction/accumulation, and computed-mask
      memory route-family behavior remains covered and non-regressed.
- [ ] `git diff --check` passes.
- [ ] Focused build/test targets for `TianChenRVRVVTarget`, `tcrv-translate`,
      `tcrv-opt`, target artifact/export tests, and RVV plugin tests pass.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] Bounded authority-leak scan over touched production files and tests
      confirms no executable artifact claim depends on central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [ ] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out Of Scope

- Migrating conversion dtype-policy, runtime scalar splat-store, segment2,
  standalone reduction/accumulation, compare/select, MAcc, widening-dot, or
  unrelated validators.
- Changing RVV provider lowering, selected-body realization, runtime behavior,
  generated C semantics, intrinsic spelling, route IDs, source-front-door
  behavior, artifact naming, dispatch/fallback behavior, or common EmitC
  materialization.
- Adding new RVV operation coverage, new selected-body realization, new
  intrinsic cases, dtype/LMUL clone batches, high-level Linalg/frontend
  lowering, dashboards, broad smoke matrices, reports, wrapper-only work, or
  evidence-only tasks.
- Claiming new runtime/correctness/performance evidence without real `ssh rvv`
  execution.
- Reintroducing descriptor-driven computation, source-front-door positive
  routes, direct route-entry compatibility, route-id authority, artifact-name
  authority, ABI-string authority, exact-intrinsic authority, common-EmitC
  semantic authority, script-derived authority, or legacy-i32 authority.

## Technical Approach

1. Keep this Trellis task current and bounded to the supplied Hermes direction.
2. Move elementwise arithmetic family classifiers and provider-fact validators
   from `RVVTargetSupportBundle.cpp` into
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Treat scalar-broadcast elementwise add/sub/mul as an in-scope elementwise
   artifact route-family consumer for target validation, while leaving broader
   conversion dtype-policy validation outside this round.
4. Register the `elementwise-arithmetic` validator next to the existing
   compare-select, MAcc, and widening-dot validators.
5. Remove the central bridge call to
   `validateRVVElementwiseArithmeticRoutePayloadFacts` and rely on registry
   provider-fact validation.
6. Remove elementwise/scalar-broadcast-specific candidate mirror branches from
   `validateRVVRouteMetadataMirrorsSelectedBody` and rely on registry
   candidate-mirror validation.
7. Add or strengthen focused negative target artifact tests that mutate
   elementwise mirrors and prove failures occur through the migrated production
   validator path.
8. Run focused target/export and RVV plugin tests, then `git diff --check`,
   authority scan, and `check-tianchenrv`.
9. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Validate/start this Trellis task after PRD and context files are written.
2. Build focused targets:
   `cmake --build build --target TianChenRVRVVTarget tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit tests for directly related target artifact consumers:
   ordinary add/sub/mul, scalar-broadcast add/sub/mul, masked add/sub/mul, and
   strided add.
5. Run `git diff --check`.
6. Run a bounded authority scan over touched production and test files.
7. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-compare-select-mask-artifact-validator-migration/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`

Primary evidence consumers:

- `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-strided-add.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-masked-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-strided-add.mlir`
- `test/Plugin/RVVExtensionPluginTest.cpp`

## Open Questions

None blocking. Repository evidence identifies a bounded production gap:
elementwise arithmetic semantic artifact validation and elementwise
route-family mirror checks still sit in the central RVV target bridge even
though the target artifact route-family validator registry now owns
compare/select mask, MAcc, and widening-dot validation.

## Definition Of Done

Elementwise arithmetic artifact acceptance is owned by the target route-family
validator registry, the central bridge contains only generic target artifact
mechanics and registry dispatch, focused fail-closed and non-regression evidence
passes, authority scans are clean, Trellis state is truthful, and one coherent
commit records the completed task.

## Completion Evidence

- `RVVTargetArtifactRouteFamilyValidation.cpp` now registers a unique
  `elementwise-arithmetic` route-family validator.
- The validator consumes rebuilt `TCRVEmitCLowerableRoute` plus
  `RVVSelectedBodyEmitCRouteDescription` facts for plain add/sub/mul, masked
  add/sub/mul, strided add, and scalar-broadcast add/sub/mul artifact
  validation.
- `RVVTargetSupportBundle.cpp` no longer owns
  `validateRVVElementwiseArithmeticRoute*` helpers or elementwise /
  scalar-broadcast candidate mirror branches; it retains candidate rebuild,
  central selected-boundary/runtime checks, residue rejection, metadata evidence
  listing, and registry dispatch.
- Focused target lit coverage covers ordinary add, scalar-broadcast add,
  strided add, and masked add target artifact consumers with stale mirror
  negatives for provider support, operand binding, ABI order, header list, type
  mapping, memory form, route-family plan, scalar-broadcast plan, strided
  memory layout, and masked metadata.
- Provider-side validator code checks statement-plan callees for setvl, load,
  scalar broadcast, compute, store, masked compare/merge, and strided
  load/store facts using rebuilt route statements; RVV plugin tests retain
  route/statement callee consistency coverage.
- `git diff --check` passed.
- Focused lit: 4/4 selected target artifact tests passed.
- Focused C++: `tianchenrv-target-artifact-export-test` and
  `tianchenrv-rvv-extension-plugin-test` passed.
- Full check: `cmake --build build --target check-tianchenrv` passed 456/456.
- Bounded authority scan over touched target production files found no
  executable i32m1, route-id-derived, descriptor-derived, source-front-door,
  source-artifact, or exact `__riscv_*_i32m1` authority in the migrated
  elementwise artifact validator path. Remaining `descriptor` matches in
  `RVVTargetSupportBundle.cpp` are the pre-existing forbidden-residue rejection
  diagnostics.
