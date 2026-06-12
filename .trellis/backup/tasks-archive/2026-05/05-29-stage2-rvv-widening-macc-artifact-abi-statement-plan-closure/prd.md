# Stage2 RVV widening MAcc artifact ABI statement-plan closure

## Goal

Close the target artifact ABI statement-plan validation boundary for the
existing selected typed RVV `widening_macc_add` contraction route. The executable
claim must flow from the selected `tcrv.exec` RVV variant and typed
`tcrv_rvv.widening_macc` body, through provider-derived route facts and the
rebuilt `TCRVEmitCLowerableRoute` statement plan, into target artifact
validation and generated RVV bundle evidence.

This task is complete only if production validation in
`RVVTargetArtifactRouteFamilyValidation.cpp` moves for the widening MAcc
contraction consumer. Tests, generated-bundle scripts, route ids, artifact
names, exact intrinsic spellings, and candidate metadata are evidence consumers
or mirrors only; they must not become authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening MAcc contraction artifact ABI
  statement-plan closure`.
- Module owner: RVV target artifact route-family validator for
  `widening_macc_add`, plus directly affected target-artifact tests and
  generated-bundle evidence.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `4a5d6c27 rvv: close widening dot ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned,
  parallel, or multi-agent workflow.

## Current Repository Facts

- The previous widening dot-reduction task closed target artifact ABI
  statement-plan validation by checking exact rebuilt provider route statement
  order, operands, result names, result C types, runtime AVL/VL facts, ABI
  pointer/stride bindings, output store facts, and selected typed RVV
  provenance for all four widening dot-reduction variants.
- The adjacent widening MAcc contraction validator already checks route id,
  runtime ABI order/roles, type mappings, ABI mappings, provider support,
  contraction route-family plan, selected typed `tcrv_rvv.widening_macc`
  provenance, source i16/result i32 type facts, layout/relation facts,
  stale-family rejection, and candidate mirror consistency.
- The remaining gap is that
  `validateRVVWideningMAccContractionRouteStatementPlan` still accepts the loop
  payload mostly by callee presence for source load, widening MAcc, and store.
  It does not yet validate exact provider-built operands, result names, and C
  types for pre-loop setvl, loop setvl, lhs/rhs source loads, accumulator load,
  widening MAcc operands/results, and output store.
- Specs require target artifact validators to dispatch from rebuilt provider
  description and route, validate provider-built statement plans, and treat
  metadata only as mirrors after route construction.

## Requirements

1. Keep provider-built route facts as authority. Candidate metadata may only be
   checked as mirrors after provider route construction.
2. Extend production validation for the `widening_macc_add` route statement
   plan so it consumes rebuilt provider facts for:
   - full-chunk pre-loop `setvl` from runtime `n`;
   - exactly one runtime AVL/VL loop with provider-derived induction, bounds,
     step, and runtime `n` relation;
   - loop `setvl` from remaining runtime AVL;
   - lhs/rhs i16 source loads from provider ABI pointers, loop induction, loop
     VL, source vector result names, and source vector C type;
   - accumulator i32 vector load from provider accumulator ABI pointer, loop
     induction, loop VL, accumulator vector result name, and result vector C
     type;
   - widening MAcc operands/results: accumulator vector, lhs/rhs source
     vectors, loop VL, result name, and result vector C type;
   - output store pointer/value/VL facts using the provider output ABI pointer,
     loop induction, provider result vector, and per-iteration loop VL;
   - selected typed RVV source provenance on all required provider-built
     statements.
3. Preserve existing checks for runtime ABI order, route operand binding,
   memory form, typed compute op, source/result type mappings, ABI mappings,
   widening MAcc relation/layout, provider support mirror labels, stale
   route-family facts, and stale candidate mirrors.
4. Add focused C++ target/export route-clone mutation coverage that corrupts
   exact widening MAcc statement operands/results and proves fail-closed
   behavior before artifact export.
5. Keep common EmitC/export neutral. Common code must not infer RVV semantics,
   dtype/config, ABI roles, route support, statement order, or intrinsic
   choices.
6. Generated-bundle selected-boundary dry-run evidence must cover
   `widening_macc_add` and keep direct route-entry shortcut fail-closed where
   exposed.
7. Real `ssh rvv` correctness evidence is required for runtime/correctness
   claims, with counts including `0`, `1`, exact-VL, tail, and stress cases
   over signed i16 inputs and i32 accumulator/result vectors.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module and non-goals.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact rebuilt
      provider route statement order, operands, result names, result C types,
      runtime AVL/VL facts, source/result vector C types, ABI pointer bindings,
      output store facts, and selected typed RVV provenance for
      `widening_macc_add`.
- [x] Focused C++ target/export coverage proves positive artifact acceptance
      for the hardened widening MAcc case.
- [x] Focused route-clone mutations fail closed for stale or missing pre-loop
      setvl AVL, loop setvl remaining AVL, lhs/rhs source load pointers,
      accumulator load pointer/result, widening MAcc operands/result, output
      store pointer/value/VL, runtime ABI order, C type mapping, route operand
      binding, stale candidate mirrors, stale route-family mirrors, and provider
      support facts.
- [x] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only, and
      legacy-i32-derived authority remain fail-closed or absent.
- [x] Selected-boundary generated-bundle dry-run passes for `widening_macc_add`,
      with `route_entry_realization` false/fail-closed where exposed.
- [x] Real `ssh rvv` generated-bundle correctness passes for `widening_macc_add`
      over counts `0`, `1`, exact-VL, tail, and stress cases, or an exact
      blocker is recorded.
- [x] Completed widening dot-reduction statement-plan validation remains
      non-regressed through focused evidence.
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

1. Reuse the existing provider-built route-step validator helper to validate the
   widening MAcc pre-loop setvl and exact loop body sequence rather than adding
   metadata-based checks.
2. Decode the expected widening MAcc ABI order from the rebuilt provider
   description: `lhs`, `rhs`, `acc`, `out`, `n`.
3. Validate the exact provider statement sequence:
   - pre-loop: `setvl(n) -> full_chunk_vl`;
   - loop body: `setvl(n - offset)`, lhs load, rhs load, accumulator load,
     widening MAcc, output store.
4. Extend `TargetArtifactExportTest.cpp` with route-clone mutations against the
   rebuilt route for `widening_macc_add`, mirroring the previous widening
   dot-reduction negative-test style.
5. Run focused target/export tests, generated-bundle dry-run, direct route-entry
   fail-closed probe, real `ssh rvv` correctness, non-regression checks,
   authority scan, `git diff --check`, and `check-tianchenrv`.

## Out of Scope

- Do not start non-widening MAcc statement-plan closure, widening dot follow-up,
  vector reduction, standalone reduction, segment2, conversion, elementwise,
  compare/select, base memory movement, runtime scalar splat-store, new route
  coverage, new dtype/LMUL variants, high-level frontend/Linalg work,
  one-intrinsic wrapper dialects, selected-body realization framework rewrites,
  dashboards, reports, or broad smoke matrices.
- Do not modify common EmitC to infer RVV semantics.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not make descriptors, ABI strings, artifact names, route ids, test names,
  metadata, exact intrinsic spellings, common EmitC code, source-front-door
  markers, direct route entries, pre-realized fixtures, or legacy i32 helper
  names route authority.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. `cmake --build build --target tcrv-opt tcrv-translate -j2` if generated
   bundle scripts need fresh tools.
4. Selected-boundary generated-bundle dry-run for `widening_macc_add`.
5. Direct pre-realized route-entry probe for `widening_macc_add`, expecting
   false/fail-closed selected-boundary-only behavior where exposed.
6. Real `ssh rvv` generated-bundle correctness for `widening_macc_add` over
   counts `0`, `1`, exact-VL, tail, and stress.
7. Focused non-regression for the completed widening dot statement-plan closure.
8. Bounded touched-file authority scan.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Completion Evidence

- Production owner moved in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`: the widening
  MAcc target artifact consumer now validates the exact rebuilt provider route
  statement plan instead of accepting loop payloads by callee presence. It
  checks exactly one pre-loop full-chunk setvl from runtime `n`, exactly one
  runtime AVL/VL loop, loop setvl from remaining AVL, lhs/rhs i16 source load
  pointer/result/type facts, i32 accumulator load pointer/result/type facts,
  widening MAcc operand/result/type facts, output store pointer/value/VL facts,
  runtime ABI order/role alignment, and selected typed RVV source provenance.
- Focused target/export C++ coverage moved in
  `test/Target/TargetArtifactExportTest.cpp`: route-clone mutations now prove
  fail-closed behavior for stale pre-loop setvl AVL, loop setvl remaining AVL,
  lhs/rhs source load pointers, accumulator load pointer/result, widening MAcc
  operand/result, output store pointer/value/VL, existing stale provider facts,
  stale candidate mirrors, stale non-family facts, and provider support facts.
- Focused local validation passed:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
- Selected-boundary generated-bundle dry-run passed for `widening_macc_add`
  with counts `0,1,16,17,257`:
  - `artifacts/tmp/stage2_rvv_widening_macc_statement_plan_closure/pre-realized-dry/pre-realized-widening-macc-statement-plan`
- Direct pre-realized route-entry shortcut remained fail-closed for
  `widening_macc_add` with the selected-boundary-only diagnostic:
  - `artifacts/tmp/stage2_rvv_widening_macc_statement_plan_closure/direct-route-entry-fail-closed/direct-pre-realized-widening-macc-statement-plan`
- Real `ssh rvv` generated-bundle correctness passed for `widening_macc_add`
  over counts `0,1,16,17,257`, with signed widening product accumulation and
  tail preservation:
  - `artifacts/tmp/stage2_rvv_widening_macc_statement_plan_closure/pre-realized-ssh/pre-realized-widening-macc-statement-plan-ssh`
- Widening dot-reduction non-regression dry-run passed for
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` over counts
  `0,1,16,17,257`:
  - `artifacts/tmp/stage2_rvv_widening_macc_statement_plan_closure/widening-dot-nonregression-dry/widening-dot-statement-plan-nonregression`
- Bounded touched-file authority scan passed. Strict added-line scan found no
  new `metadata-derived`, descriptor, source-front-door, source-export,
  direct-route, legacy i32, route-id, script-derived, artifact-name-derived, or
  exact-intrinsic executable authority. `metadata_*` strings appear only as
  negative route mutation inputs.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 459/459.

## Spec Update Judgment

No `.trellis/spec/` update was needed. The existing RVV plugin, EmitC route,
core dialect, and testing specs already require provider-derived selected-body
route facts, target artifact route-family validators, mirror-only metadata,
direct route-entry fail-closed behavior, and real `ssh rvv` evidence for
runtime/correctness claims. This task applied those existing contracts to the
widening MAcc target validator.
