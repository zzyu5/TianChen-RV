# Stage2 RVV standalone reduction/accumulation artifact validator migration

## Goal

Migrate standalone reduction/accumulation target artifact validation from
`RVVTargetSupportBundle.cpp` into the target-owned RVV route-family artifact
validator registry. The registry entry must own standalone reduction provider
fact checks and candidate metadata mirror checks for plain standalone
reductions, computed-mask standalone reductions, and runtime-scalar
computed-mask standalone reductions, while the central support bundle remains a
neutral target artifact bridge.

## Direction Source

- Direction title: `Switch: Stage2 RVV standalone reduction/accumulation
  artifact validator migration`.
- Module owner: target-owned RVV route-family artifact validator registry for
  standalone reduction/accumulation routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `de7fe9e5 rvv: migrate segment2 artifact validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I Already Know

- Commit `de7fe9e5` completed the segment2 memory artifact validator
  migration, left the worktree clean, and moved segment2 provider-fact and
  candidate-mirror validation behind the route-family validator registry.
- `RVVTargetArtifactRouteFamilyValidation.cpp` currently owns the migrated
  route-family validator registry surface and is the correct production owner
  for family-specific target artifact acceptance.
- `RVVTargetSupportBundle.cpp` still owns route-family-specific standalone
  reduction/accumulation predicates and validation functions for headers, type
  mappings, runtime ABI, statement plan, mask facts, accumulation facts,
  scalar-carry facts, runtime-scalar computed-mask variants, and candidate
  metadata mirrors.
- The relevant architecture chain is selected RVV standalone reduction body to
  RVV provider route facts to `TCRVEmitCLowerableRoute` to target route-family
  validator entry to neutral target artifact bundle dispatch to generated
  artifact.
- The RVV plugin spec requires target artifact export to rebuild the provider
  route, dispatch route-family validation from the rebuilt provider
  description, validate provider facts as authority, validate metadata only as
  mirrors, and fail closed on missing/stale provider or mirror facts.
- This round must preserve executable behavior for existing plain standalone
  reduce add/min/max, computed-mask standalone reduce add/min/max, and
  runtime-scalar computed-mask standalone reduce add/min/max artifact paths
  where already supported.

## Requirements

1. Register a production standalone reduction/accumulation route-family
   validator in `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. Dispatch active plain standalone reduction, computed-mask standalone
   reduction, and runtime-scalar computed-mask standalone reduction target
   artifact candidates through the route-family validator registry.
3. Move standalone reduction/accumulation provider-fact validation ownership out
   of `RVVTargetSupportBundle.cpp`.
4. Move standalone reduction/accumulation candidate mirror validation ownership
   out of `RVVTargetSupportBundle.cpp`.
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
   - standalone reduction operation classification and subfamily shape;
   - provider support mirror label as mirror-only evidence;
   - route operand binding plan and summary;
   - runtime AVL/VL control plan and runtime ABI order;
   - provider-derived headers;
   - VL/vector/mask C type mappings, including source and scalar-result
     channel facts;
   - ABI mapping count, order, roles, C names, and value names;
   - one runtime AVL/VL loop with selected typed RVV source provenance;
   - scalar seed splat facts;
   - reduction intrinsic and scalar result store facts;
   - accumulator/result/store-VL layout facts;
   - computed mask compare/merge/mask-role facts for computed-mask routes;
   - runtime-scalar RHS broadcast and scalar-carry facts for runtime-scalar
     computed-mask routes.
7. The validator must validate candidate metadata only as mirrors of rebuilt
   provider facts, including:
   - `tcrv_rvv.provider_supported_mirror`;
   - `tcrv_rvv.route_operand_binding_plan`;
   - `tcrv_rvv.route_operand_binding_operands`;
   - standalone reduction/accumulation route-family plan mirrors;
   - `tcrv_rvv.runtime_control_plan`;
   - `tcrv_rvv.runtime_abi_order`;
   - `tcrv_rvv.required_header_declarations`;
   - `tcrv_rvv.c_type_mapping`;
   - scalar seed, reduction/store, accumulator/result, mask, merge,
     runtime-scalar, and scalar-carry mirrors where applicable.
8. The validator must fail closed on stale route id, missing provider support
   mirror, bad ABI order, bad header/type mapping, bad AVL/VL loop facts,
   missing scalar seed splat, missing reduction/store facts, missing
   mask/compare/merge facts, missing accumulation/scalar-carry facts, stale
   candidate mirrors, descriptor/source/script/artifact-name residue,
   exact-intrinsic-as-authority, and common/central semantic choice.
9. Preserve existing conversion, elementwise, compare/select, MAcc,
   widening-dot, segment2, base-memory, runtime-scalar splat-store, and other
   route-family behavior as non-regressions.
10. Do not introduce central ad hoc, name-derived, metadata-derived,
    descriptor-derived, ABI-string-derived, script-derived, artifact-name-
    derived, common-EmitC-derived, source-front-door-derived, route-id-derived,
    exact-intrinsic-derived, direct-route-entry-only,
    pre-realized-fixture-only, or legacy-i32-derived route authority.

## Acceptance Criteria

- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` registers and dispatches a
      unique standalone reduction/accumulation validator for active plain,
      computed-mask, and runtime-scalar computed-mask standalone reduction
      route descriptions.
- [x] `RVVTargetSupportBundle.cpp` no longer contains standalone
      reduction/accumulation semantic validator ownership or a standalone
      reduction-specific candidate mirror branch, except neutral dispatch.
- [x] The new validator checks provider-derived route id agreement, headers,
      type mappings, ABI order/mapping, route operand binding mirrors, runtime
      AVL/VL plan, scalar seed splat, reduction/store facts,
      accumulator/result/store-VL layout, computed-mask facts, runtime-scalar
      RHS broadcast facts, and scalar-carry facts using rebuilt provider/body
      facts.
- [x] The new validator checks candidate mirrors for at least one plain
      standalone reduction, one computed-mask standalone reduction, and one
      runtime-scalar computed-mask standalone reduction.
- [x] Negative or mutation coverage proves stale route id, missing provider
      support mirror, bad ABI order, bad header/type mapping, bad AVL/VL loop
      facts, missing scalar seed splat, missing reduction/store facts, missing
      mask/compare/merge facts, missing accumulation/scalar-carry facts, stale
      candidate mirrors, descriptor/source/script/artifact-name residue,
      exact-intrinsic-as-authority, and common/central semantic choice fail
      closed.
- [x] Existing standalone reduction artifact fixtures still pass without
      changing generated C/runtime semantics.
- [x] Existing migrated route-family validators remain covered and
      non-regressed.
- [x] `git diff --check` passes.
- [x] Focused target artifact export and RVV plugin tests pass as needed.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Bounded touched-file authority scan confirms no standalone
      reduction/accumulation route-family validation owner remains in
      `RVVTargetSupportBundle.cpp` except neutral dispatch and that no
      executable artifact claim depends on forbidden authority sources.
- [x] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out Of Scope

- Starting runtime scalar splat-store artifact validator migration.
- Segment2 follow-up cleanup.
- Widening dot or MAcc realization changes.
- New standalone reduction operation coverage.
- Dtype/LMUL clone batches.
- Source-front-door routes.
- High-level Linalg/frontend lowering.
- One-intrinsic wrapper dialects.
- Dashboard/report work or broad smoke matrices unrelated to this migration.
- Rewriting the whole target artifact export stack.
- Claiming new runtime/correctness/performance evidence without real `ssh rvv`
  execution.

## Technical Approach

1. Keep this Trellis task current and bounded to the supplied Hermes direction.
2. Inspect the current standalone reduction/accumulation helpers in
   `RVVTargetSupportBundle.cpp` and the migrated validator patterns in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Move standalone reduction/accumulation family classifiers and
   provider-fact validators into `RVVTargetArtifactRouteFamilyValidation.cpp`.
4. Add a standalone reduction/accumulation registry entry and consumer
   predicate next to the existing migrated validators.
5. Remove central support-bundle calls and mirror branches that own standalone
   reduction/accumulation semantics, relying on registry provider-fact and
   candidate-mirror validation instead.
6. Strengthen focused standalone reduction target artifact tests with positive
   and stale/missing mirror cases as needed.
7. Run focused build/lit/C++ checks, `git diff --check`, bounded authority
   scan, and `check-tianchenrv` if feasible.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Validate/start this Trellis task after PRD and context files are written.
2. Build focused targets:
   `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit tests for directly related standalone reduction artifact
   consumers under `test/Target/RVV`.
5. Run `git diff --check`.
6. Run a bounded authority scan over touched production and test files.
7. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-segment2-memory-artifact-validator-migration/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Primary evidence consumers:

- directly relevant standalone reduction/accumulation target tests under
  `test/Target/RVV`.
