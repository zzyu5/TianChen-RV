# Stage2 RVV MAcc route-family target artifact validator migration

## Goal

Register a target-owned RVV MAcc route-family artifact validator and migrate MAcc family-specific target artifact acceptance out of `RVVTargetSupportBundle.cpp`. The central RVV target artifact bridge should remain responsible only for generic rebuild, candidate/exporter shape, runtime ABI, residue rejection, and packaging mechanics, while MAcc semantic artifact acceptance is derived from rebuilt provider/body facts through `RVVTargetArtifactRouteFamilyValidation`.

## What I already know

* Current HEAD at task start is `f629768a rvv: add target artifact route-family validation`; the worktree was clean.
* `.trellis/.current-task` did not exist, so this task was created from the Hermes direction brief before source changes.
* The previous archived task `05-28-stage2-rvv-artifact-route-family-validator-boundary` introduced `RVVTargetArtifactRouteFamilyValidation` and migrated widening-dot reduction into the target-owned route-family validator boundary.
* `RVVTargetSupportBundle.cpp` is still expected to contain MAcc family classifiers and MAcc-specific target artifact consumer checks.
* This round is about moving MAcc artifact acceptance ownership, not changing MAcc computation semantics, route construction semantics, selected-body realization semantics, or common EmitC materialization.
* Runtime/correctness/performance claims still require real `ssh rvv` evidence; this task can validate target artifact/export behavior and existing MAcc evidence paths, but must not overclaim runtime performance without new remote evidence.

## Assumptions

* The existing route-family validator registry can be extended with a MAcc family validator rather than redesigned.
* MAcc families in scope are `MAccAdd`, `ScalarBroadcastMAccAdd`, `ComputedMaskedMAccAdd`, and `RuntimeScalarComputedMaskedMAccAdd`.
* Existing target artifact/export and RVV plugin tests contain enough MAcc generated-bundle consumers to prove the validator is in the production path; if not, this task will add focused fail-closed tests rather than broad smoke matrices.
* MAcc candidate metadata remains mirror evidence only. Acceptance must use rebuilt `TCRVEmitCLowerableRoute`, `RVVSelectedBodyEmitCRouteDescription`, and `RVVTargetArtifactRouteFamilyValidationContext` facts.

## Requirements

* Register a production MAcc route-family validator in `RVVTargetArtifactRouteFamilyValidation`.
* Dispatch MAcc artifact candidates from the central bridge into the route-family validator registry.
* Move MAcc family-specific artifact validation bodies out of `RVVTargetSupportBundle.cpp`.
* Keep in `RVVTargetSupportBundle.cpp` only generic bridge responsibilities:
  * artifact candidate/exporter shape;
  * selected variant and typed body rebuild;
  * generic rebuilt provider route/candidate consistency entry;
  * source-op provenance and descriptor/direct-C/source-export residue rejection;
  * runtime ABI parameter consistency;
  * neutral materialization/export packaging;
  * dispatch into the family validator registry.
* The MAcc validator must consume rebuilt route/body facts to validate:
  * required RVV headers;
  * VL, vector type, scalar type, mask type, and C type mappings;
  * ABI order and ABI role mapping;
  * accumulator/result layout;
  * scalar-broadcast facts for scalar-broadcast MAcc;
  * computed-mask facts for computed-masked MAcc;
  * runtime-scalar facts for runtime-scalar computed-masked MAcc;
  * provider-built pre-loop and loop statement plan;
  * intrinsic mirror consistency;
  * route id mirror and provider support mirror;
  * route/candidate family consistency.
* Unsupported, stale, missing, or mismatched MAcc facts must fail closed with targeted diagnostics.
* Preserve widening-dot route-family validator behavior as a non-regression.
* Do not introduce direct-route-entry, source-front-door, descriptor, route-id, artifact-name, ABI-string, exact-intrinsic, script-derived, common-EmitC-derived, pre-realized-fixture-only, or legacy-i32 route authority.

## Acceptance Criteria

* [x] `RVVTargetArtifactRouteFamilyValidation` registers and dispatches a MAcc route-family validator for all in-scope MAcc route descriptions.
* [x] `RVVTargetSupportBundle.cpp` no longer owns duplicated MAcc semantic artifact validation bodies; it retains only generic bridge checks and route-family registry dispatch.
* [x] The MAcc validator checks headers, type mappings, ABI order/mapping, accumulator/result layout, scalar-broadcast facts, computed-mask facts, runtime-scalar facts, provider statement plan, intrinsic mirrors, route id/provider support mirrors, and route/candidate consistency using rebuilt provider/body facts.
* [x] Focused target artifact/export or plugin tests prove MAcc provider facts are consumed by the family validator in the production path.
* [x] Fail-closed coverage exists for stale or missing MAcc headers, type mappings, ABI mapping/order, accumulator/result layout, scalar-broadcast facts, computed-mask facts, runtime-scalar facts, provider statement plan, intrinsic mirrors, route id/provider support mirrors, and route/candidate mismatch, as supported by existing test infrastructure or focused additions.
* [x] Widening-dot validator behavior remains covered and non-regressed.
* [x] `git diff --check` passes.
* [x] Focused build/test targets for `TianChenRVTarget`, `tcrv-translate`, `tcrv-opt`, relevant target artifact/export tests, and RVV plugin tests pass.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.
* [x] Bounded authority-leak scan over touched production files and tests confirms no executable artifact claim depends on central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived authority.
* [x] Trellis task status, journal, and archive/commit state are truthful at the end of the round.

## Out of Scope

* Migrating standalone reductions, segment2, compare/select, widening conversion, memory movement, or unrelated route families.
* Changing MAcc computation semantics, selected-body realization semantics, provider route construction semantics, runtime `n`/AVL values, dispatch/fallback behavior, or common EmitC materialization.
* Adding new MAcc op coverage, dtype/LMUL clone batches, high-level Linalg/frontend lowering, one-intrinsic wrappers, dashboards, broad smoke matrices, or evidence-only work.
* Reintroducing descriptor-driven computation, source-front-door positive routes, direct route-entry compatibility, route-id authority, artifact-name authority, ABI-string authority, exact-intrinsic authority, common-EmitC semantic authority, script-derived authority, or legacy-i32 authority.

## Technical Notes

Required specs and prior context:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* archived task `05-28-stage2-rvv-artifact-route-family-validator-boundary`

Primary production files to inspect:

* `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`

Primary evidence consumers:

* `test/Plugin/RVVExtensionPluginTest.cpp`
* target artifact export tests
* directly related MAcc generated-bundle tests

## Round Result

Implemented MAcc as the second production family in the target-owned RVV
artifact route-family validator registry.

Changed production ownership:

* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` now owns the
  `macc` route-family validator registration and dispatch for `MAccAdd`,
  `ScalarBroadcastMAccAdd`, `ComputedMaskedMAccAdd`, and
  `RuntimeScalarComputedMaskedMAccAdd`.
* `lib/Target/RVV/RVVTargetSupportBundle.cpp` no longer contains the MAcc
  semantic artifact validation bodies or MAcc-specific candidate mirror checks.
  It keeps generic candidate/exporter shape, selected variant and typed body
  rebuild, route/candidate consistency entry, source-op provenance and
  descriptor/direct-C/source-export residue rejection, runtime ABI consistency,
  neutral materialization/export packaging, and route-family registry dispatch.

Moved into the MAcc family validator:

* required header checks;
* VL/vector/scalar/mask/C type mapping checks;
* runtime ABI order and ABI mapping checks;
* accumulator/result layout checks;
* plain MAcc, scalar-broadcast MAcc, computed-mask MAcc, and runtime-scalar
  computed-mask MAcc provider fact checks;
* pre-loop setvl and runtime AVL/VL loop statement-plan checks;
* vector load, scalar splat, compare, MAcc, masked-merge, and store statement
  checks where required by the family;
* route id, provider-supported mirror, operand binding plan, route-family plan,
  runtime/control/header/type/layout/mask/accumulation candidate mirror checks.

Focused test coverage:

* Reused existing scalar-broadcast and computed-mask MAcc target artifact
  negative tests for stale route/provider/binding/ABI/header/type/accumulation
  and layout mirrors.
* Added runtime-scalar computed-mask MAcc target artifact negative tests in
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
  for stale provider support, binding plan, ABI order, headers, type mapping,
  runtime-scalar mask producer, and result layout.

Evidence:

* `cmake --build build --target TianChenRVTarget tcrv-translate tcrv-opt
  tianchenrv-target-artifact-export-test
  tianchenrv-rvv-extension-plugin-test`: passed.
* `build/bin/tianchenrv-target-artifact-export-test`: passed.
* `build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
* Runtime-scalar MAcc focused lit filter
  `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`:
  passed.
* Runtime-scalar MAcc failure filter after the first self-repair:
  passed 6/6.
* `git diff --check`: passed.
* `cmake --build build --target check-tianchenrv`: passed 456/456 after
  cleaning lit output directories from an accidental duplicate concurrent
  check run.

Self-repair:

* The first full `check-tianchenrv` found that the new runtime-scalar MAcc
  validator required `accumulationScalarCarryContract`, which current provider
  descriptions do not emit for this route. The validator was corrected to
  require the provider-derived RHS scalar splat and runtime-scalar mask
  producer facts that are actually present in the rebuilt route description and
  candidate mirrors.
* A later duplicate full-check invocation caused generated lit artifact
  collisions (`file exists`, missing materialized files, and invalid object
  reads). Generated output directories were cleaned and a single full check was
  rerun successfully.

Bounded authority scan:

* `validateRVVMAcc*`, `isRVVMAcc*`, and MAcc target artifact consumer
  diagnostics now appear only in
  `RVVTargetArtifactRouteFamilyValidation.cpp`, not in the central bridge.
* The only `script-derived` hits in touched files are intentional negative
  test substitutions.
* The only descriptor/direct-C/source-export hits in touched production files
  are the existing generic central bridge residue rejections.
* No executable MAcc target artifact claim depends on central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, exact-intrinsic-derived,
  direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
  authority.

Spec update:

* No `.trellis/spec/` update was needed. This round applied the existing
  route-family validator boundary from `rvv-plugin.md`; it did not introduce a
  new durable rule beyond that spec.
