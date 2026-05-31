# PRD: Stage2 RVV elementwise arithmetic route-family ABI/provider ownership

## Direction Source

Hermes/user direction switches the next bounded owner to the RVV elementwise
arithmetic ABI/provider boundary after segment2, indexed memory, computed-mask
memory, and compare/select boundaries were closed.

The requested capability chain is:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-local elementwise selected-body realization
  -> realized typed tcrv_rvv elementwise body
  -> elementwise route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC
  -> target artifact ABI mirror validation
  -> generated RVV artifact
  -> ssh rvv correctness evidence when claimed
```

## Initial Repository State

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1f4c8ea4 rvv: close segment2 artifact ABI mirrors`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Repository Findings

- `RVVElementwiseSelectedBodyRealizationOwner` already owns pre-realized
  elementwise/compare-select realization and emits realized typed `setvl`,
  `with_vl`, load/splat/strided-load/compare/binary/masked-binary/store
  structure before route planning.
- `RVVEmitCElementwiseRouteFamilyPlanOwners` already owns ordinary, masked,
  strided, and scalar-broadcast route-family plan derivation and provider-plan
  verification.
- Plain/masked/strided elementwise arithmetic plans carry a typed config
  snapshot: `typedConfigFactsID`, element type, signed C type, element bit
  width, SEW, LMUL, tail/mask policy, and config contract. Provider
  verification compares that snapshot with `analysis.typedConfigFacts` and the
  route description before materialization.
- Scalar-broadcast elementwise plans currently carry runtime control, headers,
  type strings, intrinsics, ABI order, and statement facts, but do not carry the
  same explicit typed config snapshot fields. The description is populated from
  runtime-control facts, but the scalar-broadcast owner boundary does not prove
  those route facts are tied back to the same selected typed config snapshot in
  the same way as ordinary/masked/strided elementwise arithmetic.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already has an
  `elementwise-arithmetic` target validator covering ordinary,
  scalar-broadcast, strided, and masked elementwise routes. It validates
  provider support, route-family plan mirrors, operand binding, memory form,
  runtime ABI order, headers, C type mapping, source/destination memory forms,
  masked facts, strided facts, and stale non-elementwise mirrors.
- Generated-bundle metadata checking already verifies common typed config
  mirrors. For `strided_add`, the script does not require the elementwise
  route-family provider mirrors (`elementwise_arithmetic_route_family_plan`,
  target leaf profile, provider support, header/type summary, source/dest
  memory form) even though production emission and lit target tests emit and
  validate them.

## Module Goal

Close the remaining elementwise arithmetic ABI/provider ownership gap by making
scalar-broadcast elementwise provider plans carry and verify the same typed
config snapshot authority as the rest of the elementwise arithmetic family, and
by tightening generated-bundle evidence so strided elementwise arithmetic checks
the provider-derived elementwise mirrors that target artifacts already consume.

This is a bounded production-plus-evidence task:

- production movement must land in the RVV elementwise route-family provider
  owner code;
- generated-bundle script updates may strengthen evidence checking, but must
  remain mirrors of provider/target facts and must not authorize route support.

## Requirements

1. Add explicit typed config snapshot facts to the scalar-broadcast
   elementwise route-family provider plan: typed config facts ID, element type,
   signed element C type, element bit width, SEW, LMUL, tail policy, mask
   policy, and config contract.
2. Derive those scalar-broadcast facts only from `analysis.typedConfigFacts`
   and the realized typed selected body. Do not derive them from route ids,
   artifact names, metadata, ABI strings, exact intrinsic names, scripts, or
   common EmitC.
3. Validate scalar-broadcast typed config facts before provider
   materialization:
   - typed config facts must exist;
   - element bit width must mirror SEW;
   - element type and signed C type must match SEW and typed config facts;
   - SEW/LMUL/policy/config must mirror runtime AVL/VL control facts;
   - the plan snapshot must mirror `analysis.typedConfigFacts`;
   - route description dtype/config mirrors must be populated from the same
     validated scalar-broadcast family plan.
4. Preserve existing scalar-broadcast support level: no new dtype/LMUL clone
   batch. The current supported scalar-broadcast elementwise route remains
   SEW32 LMUL m1 unless existing tests prove otherwise.
5. Preserve ordinary add/sub/mul, RHS broadcast-load add/sub/mul, masked
   add/sub/mul, and strided add behavior.
6. Preserve target artifact validator semantics. If target validation needs a
   focused mirror check added for this boundary, it must validate provider
   facts as mirrors only after route reconstruction.
7. Tighten generated-bundle evidence for representative elementwise arithmetic
   forms so strided elementwise arithmetic checks the provider-owned
   elementwise route-family plan, provider support, target leaf profile,
   header/type mapping, and source/destination memory-form mirrors.
8. Direct pre-realized route-entry remains unsupported/fail-closed for affected
   forms unless an explicitly existing positive route-entry path already owns a
   representative form.
9. Do not change common EmitC into an RVV semantic owner. Common materializers
   may only consume provider-built route payloads.

## Non-Goals

- Do not start compare-select, computed-mask memory, segment2 memory, indexed
  memory, conversion/dtype expansion, standalone reduction, MAcc/contraction,
  widening dot, high-level Linalg/frontend lowering, one-intrinsic wrappers,
  dashboards, reports, or broad smoke matrices.
- Do not add new elementwise operation coverage, new dtype/LMUL matrices, new
  source-front-door routes, or compatibility wrappers around old i32 route
  authority.
- Do not treat generated scripts, route ids, artifact names, metadata,
  provider-supported mirror strings, exact intrinsic spellings, or common EmitC
  as route authority.
- Do not claim runtime/correctness/performance without real `ssh rvv`
  evidence.

## Acceptance Criteria

- [ ] Production diff lands in the RVV elementwise route-family provider owner
      code, not only tests or scripts.
- [ ] Scalar-broadcast elementwise route-family plans carry typed config
      snapshot facts and provider verification fails closed if those facts are
      missing, stale, or inconsistent with selected typed config/runtime facts.
- [ ] Focused C++ plugin tests cover positive scalar-broadcast provider-plan
      construction and fail-closed stale/missing typed config snapshot
      behavior before provider materialization.
- [ ] Target artifact validation continues to accept ordinary add,
      scalar-broadcast add, strided add, and masked add provider/candidate
      mirrors, and rejects stale provider support, operand binding, memory
      form, mask policy/config mirror, stride, runtime ABI, header/type mapping,
      and non-elementwise route-family mirrors where covered by the focused
      validator.
- [ ] Generated-bundle dry-run covers representative elementwise arithmetic
      forms: plain add, scalar-broadcast add, strided add, and masked add,
      through selected-boundary generation with route-entry realization false
      for forms that are not direct route-entry positives.
- [ ] Direct pre-realized route-entry shortcut remains unsupported/fail-closed
      for the affected non-route-entry forms, or an existing owned positive
      form is reported separately without broadening scope.
- [ ] Real `ssh rvv` generated-bundle correctness runs for representative
      elementwise counts `0`, `1`, exact-VL, tail, and stress cases across the
      currently supported elementwise operation set, or an exact external
      blocker is recorded.
- [ ] Non-regression covers the segment2 five-form and indexed memory
      boundaries from recent completed tasks.
- [ ] Bounded authority scan over touched production/test/script/task files
      finds no new central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [ ] `git diff --check` passes.
- [ ] Focused builds/tests pass, and `check-tianchenrv` passes or the exact
      blocker is recorded.
- [ ] Task status, journal, archive state, and final commit state are truthful.

## Validation Plan

1. Build and run focused RVV plugin tests:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build and run target artifact tests:
   `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
   and `./build/bin/tianchenrv-target-artifact-export-test`.
3. Run selected-boundary generated-bundle dry-run for plain add,
   scalar-broadcast add, strided add, and masked add with route-entry
   realization disabled where applicable.
4. Run direct pre-realized route-entry fail-closed probes for affected forms
   that should not be direct-route positives.
5. Run real `ssh rvv` generated-bundle correctness for representative
   elementwise counts `0,1,16,23,257` unless an exact external blocker occurs.
6. Run segment2 and indexed memory non-regression dry-runs from the recent
   archive pattern.
7. Run bounded authority scans over touched files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/prd.md`.
- Relevant implementation files inspected:
  `include/TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Target/TargetArtifactExportTest.cpp`.

No blocking user question remains. The supplied direction brief, current specs,
archived task boundaries, and code inspection are specific enough for one
bounded implementation round.
