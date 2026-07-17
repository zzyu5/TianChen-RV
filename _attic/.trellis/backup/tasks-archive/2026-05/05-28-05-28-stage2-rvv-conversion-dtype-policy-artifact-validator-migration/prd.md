# Stage2 RVV conversion dtype-policy artifact validator migration

## Goal

Migrate conversion dtype-policy target artifact validation from
`RVVTargetSupportBundle.cpp` into the target-owned RVV route-family artifact
validator registry. The registry entry must own widening conversion provider
fact checks and candidate metadata mirror checks, while the central support
bundle remains a neutral target artifact bridge.

## Direction Source

- Direction title: `Continue: Stage2 RVV conversion dtype-policy artifact
  validator migration`.
- Module owner: target-owned RVV route-family artifact validator registry entry
  for conversion dtype-policy routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b5f7618d rvv: migrate elementwise artifact validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I Already Know

- Commit `b5f7618d` completed the elementwise arithmetic artifact validator
  migration and left the worktree clean.
- `RVVTargetArtifactRouteFamilyValidation.cpp` currently registers
  `compare-select-mask`, `elementwise-arithmetic`, `macc`, and
  `widening-dot-reduction` validators.
- `RVVTargetSupportBundle.cpp` still owns conversion dtype-policy semantic
  provider-fact validation:
  - `isRVVConversionDtypePolicyWideningRouteFamilyOperation`;
  - `isRVVConversionDtypePolicyRouteFamilyOperation`;
  - `validateRVVConversionDtypePolicyRouteHeaders`;
  - `validateRVVConversionDtypePolicyRouteTypeMappings`;
  - `validateRVVConversionDtypePolicyRouteABIMappings`;
  - `validateRVVConversionDtypePolicyRouteStatementPlan`;
  - `validateRVVConversionDtypePolicyRoutePayloadFacts`.
- `RVVTargetSupportBundle.cpp` also still owns conversion-specific candidate
  mirror validation inside `validateRVVRouteMetadataMirrorsSelectedBody`.
- Existing conversion fixtures cover explicit and pre-realized
  `widen_i32_to_i64`, plus pre-realized `widen_i16_to_i32`, including stale
  conversion plan and stale conversion relation fail-closed cases.
- This round must not change RVV provider lowering, selected-body realization,
  generated C semantics, intrinsic spelling, route IDs, runtime ABI, artifact
  names, or runtime/correctness/performance claims.

## Requirements

1. Register a production `conversion-dtype-policy` route-family validator in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. Dispatch active widening conversion target artifact candidates through the
   target route-family validator registry.
3. Move conversion dtype-policy semantic provider-fact validation out of
   `RVVTargetSupportBundle.cpp`.
4. Move conversion dtype-policy candidate mirror validation out of
   `RVVTargetSupportBundle.cpp`.
5. Keep `RVVTargetSupportBundle.cpp` limited to neutral bridge
   responsibilities:
   - selected-body route rebuild;
   - generic candidate shape and selected-boundary/runtime checks;
   - descriptor, direct-C, source-export, and source-front-door residue
     rejection;
   - neutral materialized route verification and artifact mechanics;
   - metadata evidence listing;
   - registry dispatch.
6. The new validator must consume rebuilt `TCRVEmitCLowerableRoute` plus
   `RVVSelectedBodyEmitCRouteDescription` facts to validate:
   - conversion operation classification;
   - provider support mirror label as mirror-only evidence;
   - route operand binding plan and summary;
   - runtime AVL/VL control plan and runtime ABI order;
   - result dtype/config/policy facts;
   - source dtype/config facts for widening conversions;
   - widening conversion relation with source SEW smaller than result SEW;
   - provider-derived headers;
   - VL/source/result C type mappings;
   - ABI mapping count, order, roles, C names, and value names;
   - pre-loop setvl, loop setvl, source load, widening conversion, and store
     statement facts with selected typed RVV source provenance.
7. The new validator must validate candidate metadata only as mirrors of the
   rebuilt provider facts:
   - `tcrv_rvv.widening_conversion_route_family_plan`;
   - `tcrv_rvv.source_sew`;
   - `tcrv_rvv.source_lmul`;
   - `tcrv_rvv.dest_sew`;
   - `tcrv_rvv.dest_lmul`;
   - `tcrv_rvv.conversion_relation`;
   - `tcrv_rvv.memory_form`;
   - `tcrv_rvv.runtime_control_plan`;
   - `tcrv_rvv.runtime_abi_order`;
   - `tcrv_rvv.required_header_declarations`;
   - `tcrv_rvv.c_type_mapping`.
8. The validator must fail closed on missing or stale provider facts, stale
   scalar-broadcast elementwise residue, stale elementwise/MAcc/widening-dot/
   compare-select/segment2/reduction/accumulation/base-memory route-family
   mirrors, wrong source/result type mappings, wrong headers, wrong ABI order,
   wrong runtime AVL/VL facts, wrong source load/conversion/store statement
   plan facts, descriptor residue, source-front-door residue, route-id-derived
   authority, artifact-name-derived authority, and exact-intrinsic-as-authority.
9. Preserve existing runtime-scalar splat-store, segment2, standalone
   reduction/accumulation, elementwise, compare/select, MAcc, widening-dot, and
   base-memory target artifact behavior as non-regressions.
10. Do not introduce central ad hoc, name-derived, metadata-derived,
    descriptor-derived, ABI-string-derived, script-derived, artifact-name-
    derived, common-EmitC-derived, source-front-door-derived, route-id-derived,
    exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only,
    or legacy-i32-derived route authority.

## Acceptance Criteria

- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` registers and dispatches a
      unique `conversion-dtype-policy` validator for active widening conversion
      route descriptions.
- [x] `RVVTargetSupportBundle.cpp` no longer contains
      `validateRVVConversionDtypePolicyRoute*` semantic helper ownership or a
      conversion-specific candidate mirror branch.
- [x] The new validator checks provider-derived headers, type mappings, ABI
      order/mapping, route operand binding mirrors, runtime AVL/VL plan,
      source/result dtype relation, statement callees, provider-supported
      mirror, and conversion family-plan mirrors using rebuilt provider/body
      facts.
- [x] Focused conversion artifact tests prove registry dispatch,
      provider-fact validation, candidate mirror validation, and stale/residue
      fail-closed behavior.
- [x] Existing explicit and pre-realized conversion artifact fixtures still
      pass without changing generated C/runtime semantics.
- [x] Existing elementwise, compare/select, MAcc, widening-dot, runtime-scalar
      splat-store, segment2, standalone reduction/accumulation, and base-memory
      route-family behavior remains covered and non-regressed.
- [x] `git diff --check` passes.
- [x] Focused build/test targets for `TianChenRVRVVTarget`, target artifact
      export tests, RVV plugin tests, and focused conversion lit tests pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Bounded touched-file authority scan confirms no executable artifact claim
      depends on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out Of Scope

- Adding new conversion route coverage, dtype/LMUL clone batches, or new
  intrinsic cases.
- Changing RVV provider lowering, selected-body realization, runtime ABI,
  generated C semantics, intrinsic spelling, route IDs, source-front-door
  behavior, artifact naming, dispatch/fallback behavior, or common EmitC
  materialization.
- Starting runtime scalar splat-store, segment2, standalone
  reduction/accumulation, compare/select, MAcc, widening-dot, broad target
  cleanup, dashboards, or runtime evidence work in this round.
- Claiming new runtime/correctness/performance evidence without real `ssh rvv`
  execution.

## Technical Approach

1. Keep this Trellis task current and bounded to the supplied Hermes direction.
2. Move conversion dtype-policy family classifiers and provider-fact validators
   from `RVVTargetSupportBundle.cpp` into
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Add a `conversion-dtype-policy` registry entry and consumer predicate next
   to the existing migrated validators.
4. Remove the central bridge call to
   `validateRVVConversionDtypePolicyRoutePayloadFacts` and rely on registry
   provider-fact validation.
5. Remove the conversion-specific branch from
   `validateRVVRouteMetadataMirrorsSelectedBody` and rely on registry
   candidate-mirror validation.
6. Strengthen focused conversion lit tests with at least one stale provider
   support or stale unrelated route-family residue check that must fail through
   the migrated registry path.
7. Run focused build/lit/C++ checks, `git diff --check`, bounded authority scan,
   and `check-tianchenrv` if feasible.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Validate/start this Trellis task after PRD and context files are written.
2. Build focused targets:
   `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit tests for directly related conversion artifact consumers:
   `pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`,
   `pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`, and
   `explicit-selected-body-artifact-widen-i32-to-i64.mlir`.
5. Run `git diff --check`.
6. Run a bounded authority scan over touched production and test files.
7. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-elementwise-arithmetic-artifact-validator-migration/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Primary evidence consumers:

- `test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`
- `test/Plugin/RVVExtensionPluginTest.cpp`

## Open Questions

None blocking. Repository evidence identifies a bounded production gap:
conversion dtype-policy semantic artifact validation and conversion candidate
mirror checks still sit in the central RVV target bridge even though the target
artifact route-family validator registry already owns compare/select,
elementwise, MAcc, and widening-dot validation.

## Definition Of Done

Conversion dtype-policy artifact acceptance is owned by the target
route-family validator registry, the central bridge contains only generic
target artifact mechanics and registry dispatch for this family, focused
fail-closed and non-regression evidence passes, authority scans are clean,
Trellis state is truthful, and one coherent commit records the completed task.
