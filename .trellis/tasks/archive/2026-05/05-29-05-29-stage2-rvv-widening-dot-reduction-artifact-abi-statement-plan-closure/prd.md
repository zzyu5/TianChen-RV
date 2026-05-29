# Stage2 RVV widening dot-reduction artifact ABI statement-plan closure

## Goal

Close the target artifact ABI statement-plan validation boundary for the
existing selected typed RVV widening dot-reduction route family. The executable
claim must flow from the selected `tcrv.exec` RVV variant and typed
`tcrv_rvv.widening_dot_reduce` / `tcrv_rvv.masked_widening_dot_reduce` body,
through provider-derived route facts and the rebuilt
`TCRVEmitCLowerableRoute` statement plan, into target artifact validation and
generated RVV bundle evidence.

This task is complete only if production validation in
`RVVTargetArtifactRouteFamilyValidation.cpp` moves for the widening
dot-reduction consumer. Tests, generated-bundle scripts, route ids, artifact
names, exact intrinsic spellings, and candidate metadata are evidence consumers
or mirrors only; they must not become authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening dot-reduction artifact ABI
  statement-plan closure`.
- Module owner: RVV provider-derived route facts and the RVV target artifact
  ABI validator for `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c695537d rvv: close vector reduction ABI statement
  validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned,
  parallel, or multi-agent workflow.

## Current Repository Facts

- The previous vector reduction task closed `reduce_add` artifact ABI
  statement-plan validation by consuming rebuilt provider route statement
  operands/results instead of callee presence or candidate mirrors.
- Existing widening dot-reduction target validation already checks selected
  route payload facts, runtime ABI order/roles, type mappings, ABI mappings,
  headers, layout/relation facts, strided/computed-mask facts, selected typed
  RVV provenance, and callee presence.
- The remaining gap is that
  `validateRVVWideningDotReductionRouteStatementPlan` mostly proves shape and
  callee presence. It does not yet validate exact provider-built operands,
  result names, and C types for pre-loop setvl/seed/store and loop setvl,
  source loads, stride operands, compare/mask construction, widening product,
  masked product, merge, scalar seed, reduction, and output store.
- Specs require target artifact validators to dispatch from rebuilt provider
  description and route, and to treat metadata only as mirrors after route
  construction.

## Requirements

1. Keep provider-built route facts as authority. Candidate metadata may only be
   checked as mirrors after provider route construction.
2. Extend production validation for the widening dot-reduction route statement
   plan so it consumes rebuilt provider facts for:
   - full-chunk pre-loop `setvl` from runtime `n`;
   - pre-loop scalar seed splat from accumulator input and initial scalar store
     to output when the provider route emits those seed/store statements;
   - exactly one runtime AVL/VL loop with provider-derived induction, bounds,
     step, and runtime `n` relation;
   - loop `setvl` from remaining runtime AVL;
   - unit or strided i16 lhs/rhs source loads from provider ABI pointers,
     induction, stride ABI operands when active, loop VL, source vector result
     names, and source vector C type;
   - computed-mask compare lhs/rhs loads, compare statement, mask result name,
     mask C type, predicate source operands, and loop VL when active;
   - widening product operands/results for unit and strided non-computed-mask
     routes;
   - masked widening product operands/results, merge operands/results, and
     inactive-lane zeroing facts for computed-mask routes;
   - scalar seed splat operands/results, reduction operands/results, and output
     store pointer/result/VL facts;
   - selected typed RVV source provenance on all required provider-built
     statements.
3. Preserve existing checks for runtime ABI order, route operand binding,
   memory form, typed compute op, C type mappings, ABI mappings, dot relation,
   layout, store VL, mask/stride facts, and stale non-family facts.
4. Add focused C++ target/export route-clone mutation coverage that corrupts
   exact widening dot-reduction statement operands/results for representative
   unit, strided-input, computed-mask, and computed-mask-strided cases, or a
   justified coherent subset with non-regression for the rest.
5. Keep common EmitC/export neutral. Common code must not infer RVV semantics,
   dtype/config, ABI roles, mask policy, stride policy, route support, or
   intrinsic choices.
6. Generated-bundle selected-boundary dry-run evidence must cover the hardened
   path and keep direct route-entry shortcut fail-closed where exposed.
7. Real `ssh rvv` correctness evidence is required for runtime/correctness
   claims, with counts including `0`, `1`, exact-VL, tail, and stress cases
   over signed i16 inputs, seed/accumulator interaction, stride, and mask
   patterns where applicable.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module and non-goals.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact rebuilt
      provider route statement order, operands, result names, result C types,
      runtime AVL/VL facts, source/result vector C types, mask C type, ABI
      pointer/stride bindings, output store facts, and selected typed RVV
      provenance for widening dot-reduction routes.
- [x] Focused C++ target/export coverage proves positive artifact acceptance
      for the hardened widening dot-reduction cases.
- [x] Focused route-clone mutations fail closed for stale or missing pre-loop
      setvl/seed/store facts, loop setvl remaining AVL, source load pointers,
      stride operands, compare/mask operands, widening product
      operands/results, masked product operands/results, merge
      operands/results, scalar seed operands/results, reduction
      operands/results, output store pointer/result/VL, runtime ABI order,
      C type mapping, route operand binding, stale candidate mirrors, and stale
      non-family facts where applicable.
- [x] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only, and
      legacy-i32-derived authority remain fail-closed or absent.
- [x] Selected-boundary generated-bundle dry-run passes for at least the
      hardened widening dot-reduction path, with `route_entry_realization`
      false/fail-closed where exposed.
- [x] Real `ssh rvv` generated-bundle correctness passes for representative
      widening dot-reduction paths over counts `0`, `1`, exact-VL, tail, and
      stress cases, or an exact blocker is recorded.
- [x] Completed vector reduction, standalone reduction, and prior widening
      dot-runtime closures remain non-regressed through focused evidence.
- [x] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      executable authority.
- [x] `git diff --check` passes.
- [x] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is
      recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Inspect the provider-built widening dot-reduction statement sequence from
   the rebuilt route and encode exact target-side validation against provider
   ABI/type/control facts.
2. Reuse `validateRVVProviderBuiltRouteStep` for concrete callee, operand,
   result-name, and C-type checks, adding small target-local helpers only when
   they reduce repeated pointer/operand construction.
3. Extend `TargetArtifactExportTest.cpp` with route-clone mutations against
   provider statement facts for representative unit, strided, computed-mask,
   and computed-mask-strided paths.
4. Run focused target/export tests, generated-bundle dry-runs, direct
   route-entry fail-closed probes when exposed, real `ssh rvv` correctness,
   non-regression checks, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Out of Scope

- Do not reopen vector `reduce_add`, standalone reduction add/min/max,
  selected-body realization migrations, provider route-family design, high-level
  frontend/Linalg lowering, one-intrinsic wrapper dialects, broad smoke
  matrices, dashboards, or report-only cleanup.
- Do not expand RVV coverage, add new ops, add dtype/LMUL variants, or add new
  route families.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not make descriptors, ABI strings, artifact names, route ids, test names,
  metadata, exact intrinsic spellings, common EmitC code, source-front-door
  markers, or legacy i32 helper names route authority.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Build `tcrv-opt` and `tcrv-translate` if generated-bundle scripts need fresh
   tools.
4. Selected-boundary generated-bundle dry-run for representative widening
   dot-reduction route(s).
5. Direct pre-realized route-entry probe for migrated widening dot path(s),
   expecting false/fail-closed selected-boundary-only behavior where exposed.
6. Real `ssh rvv` generated-bundle correctness for representative widening
   dot-reduction route(s) over counts `0`, `1`, exact-VL, tail, and stress.
7. Focused non-regression for vector `reduce_add`, standalone reduction, and
   prior widening dot runtime closure.
8. Bounded touched-file authority scan.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Completion Evidence

- Production owner moved in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`: the widening
  dot-reduction target artifact consumer now validates the exact rebuilt
  provider route statement plan for all four existing subfamilies instead of
  accepting callee presence. It checks pre-loop full-chunk setvl, accumulator
  seed splat, initial output store, one runtime AVL/VL loop, loop setvl from
  remaining AVL, source loads, strided pointer and stride-byte operands,
  computed-mask compare loads/predicate, masked product, inactive zero splat,
  merge, widening product, loop seed splat, reduction, scalar output store,
  statement result names, result C types, runtime ABI order/role alignment,
  store VL `1`, and selected typed RVV source provenance.
- Focused target/export C++ coverage moved in
  `test/Target/TargetArtifactExportTest.cpp`: route-clone mutations now prove
  fail-closed behavior for stale pre-loop setvl AVL, pre-loop seed operand and
  result, pre-loop store pointer, loop remaining AVL, unit lhs/rhs load
  pointers, widening product operand, loop seed operand, reduction operand and
  result, output store pointer/VL, strided source pointer/stride operands,
  computed-mask compare pointer, mask result, masked product mask operand,
  merge result, computed-mask strided dot source stride/pointer, existing stale
  provider facts, stale candidate mirrors, stale non-family facts, and direct
  route-entry-only authority.
- Focused local validation passed:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
- Selected-boundary generated-bundle dry-run passed for
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` with counts
  `0,1,16,17,257`:
  - `artifacts/tmp/stage2_rvv_widening_dot_statement_plan_closure/pre-realized-all-dry/pre-realized-widening-dot-statement-plan`
- Direct pre-realized route-entry shortcut remained fail-closed for all four
  widening dot-reduction op kinds with the selected-boundary-only diagnostic:
  - `artifacts/tmp/stage2_rvv_widening_dot_statement_plan_closure/direct-route-entry-fail-closed/direct-pre-realized-widening-dot-statement-plan`
- Real `ssh rvv` generated-bundle correctness passed for all four widening
  dot-reduction op kinds over counts `0,1,16,17,257`. Evidence covered signed
  i16 horizontal dot products, scalar seed accumulation, scalar output/tail
  preservation, strided inputs with source stride pairs `2:3`, and
  computed-mask-strided cases with stride pairs `2:3` and `3:2` plus two mask
  and input patterns:
  - `artifacts/tmp/stage2_rvv_widening_dot_statement_plan_closure/pre-realized-all-ssh/pre-realized-widening-dot-statement-plan-ssh`
- Non-regression dry-run passed for vector `reduce_add` and standalone
  `standalone_reduce_add`, `standalone_reduce_min`, and
  `standalone_reduce_max` over counts `0,1,16,17,257`:
  - `artifacts/tmp/stage2_rvv_widening_dot_statement_plan_closure/nonregression-dry/vector-and-standalone-reduction-nonregression`
- Bounded touched-file authority scan passed. Production added lines contain
  no new metadata-, route-id-, descriptor-, source-front-door-, source-export-,
  exact-intrinsic-, direct-route-entry-, legacy-i32-, or common-EmitC-derived
  executable authority. Test added lines containing `metadata_*` are negative
  stale-route mutation inputs only.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 459/459.

## Spec Update Judgment

No `.trellis/spec/` update was needed. The existing RVV plugin, EmitC route,
emission runtime, and testing specs already require provider-derived selected
body route facts, target artifact route-family validators, mirror-only
metadata, direct route-entry fail-closed behavior, and real `ssh rvv` evidence
for runtime/correctness claims. This task applied those existing contracts to
the widening dot-reduction target validator.
