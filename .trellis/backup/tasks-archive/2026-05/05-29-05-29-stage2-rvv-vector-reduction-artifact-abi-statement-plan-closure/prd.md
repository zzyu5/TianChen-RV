# Stage2 RVV vector reduction artifact ABI statement-plan closure

## Goal

Close the target artifact ABI statement-plan validation boundary for the
selected typed RVV vector `reduce_add` route family. The executable claim must
flow from the selected `tcrv.exec` RVV variant and typed `tcrv_rvv.reduce`
body, through provider-derived route facts and rebuilt
`TCRVEmitCLowerableRoute` statements, into neutral target artifact validation
and generated RVV bundle evidence.

This task is complete only if production validation in
`RVVTargetArtifactRouteFamilyValidation.cpp` moves for the vector reduction
consumer. Tests, generated-bundle scripts, route ids, artifact names, and
candidate metadata are evidence consumers or mirrors only; they must not
become authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV vector reduction artifact ABI
  statement-plan closure`.
- Module owner: RVV provider-derived route facts and RVV target artifact ABI
  validator for the vector reduction route family, centered on selected typed
  `tcrv_rvv.reduce` / `reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `05ae1a79 rvv: close standalone min max ABI validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned or
  parallel agent workflow.

## Current Repository Facts

- The archived standalone reduce-add task closed base and computed-mask
  standalone add artifact ABI validation, including provider-derived leaf
  mirrors and real `ssh rvv` evidence.
- The archived standalone min/max task hardened
  `RVVTargetArtifactRouteFamilyValidation.cpp` so standalone min/max
  validation consumes rebuilt provider route statements for setvl, source and
  compare loads, scalar seed splats, inactive neutral splats, merge masks,
  reduction leaves, lane-0 scalar-result stores, and store VL `1`.
- The current vector `reduce_add` target artifact validator already checks
  runtime ABI order/roles, route operand binding summary, memory form
  `VectorRHSLoad`, typed compute op `tcrv_rvv.reduce`, type mappings, ABI
  mappings, layout facts, and that load/reduction/store callees are present.
- The current vector `reduce_add` statement-plan validator does not yet prove
  concrete statement operands/results for the route body: full-chunk setvl
  operand, loop setvl remaining AVL, lhs source load pointer/result, RHS
  seed/accumulator load pointer/result, reduction operands/result, and output
  store pointer/result/VL.
- Current target/export tests have a positive vector `reduce_add` candidate
  and a few stale provider/candidate mirror mutations, but they do not yet
  mutate rebuilt route statement operands/results directly.

## Requirements

1. Keep provider-built route facts as authority. Candidate metadata may only be
   checked as mirrors after route construction.
2. Extend production validation for vector `reduce_add` route statements so
   the target artifact validator consumes rebuilt route facts for:
   - pre-loop full-chunk `setvl` from runtime `n` / AVL;
   - exactly one runtime AVL/VL loop with loop bounds and step derived from
     provider runtime facts;
   - loop `setvl` from remaining runtime AVL;
   - lhs source vector load from `lhs + induction` using provider vector C
     type and loop VL;
   - RHS seed/accumulator vector load from `rhs + induction` using provider
     vector C type and loop VL;
   - reduction intrinsic statement using provider-selected lhs vector, RHS
     seed/accumulator vector, loop VL, result name, and vector C type;
   - output store using `out + induction`, provider result vector, store VL
     `1`, and provider vector/VL C types;
   - selected typed RVV source provenance on pre-loop and loop statements.
3. Preserve existing vector reduction provider checks for runtime ABI order,
   route operand binding, memory form, typed op, C type mappings, ABI
   mappings, accumulator/result layout, store VL, and stale non-family facts.
4. Add focused C++ target/export coverage proving positive vector `reduce_add`
   acceptance and fail-closed mutations for stale rebuilt-route statement
   facts:
   - stale pre-loop setvl AVL operand;
   - stale loop setvl remaining AVL operand;
   - stale lhs load pointer or result;
   - stale RHS seed/accumulator load pointer or result;
   - stale reduction intrinsic operand or result;
   - stale output store pointer/result/VL;
   - stale runtime ABI order or runtime element-count role;
   - stale route operand binding summary;
   - stale candidate mirrors;
   - direct-route-entry-only, artifact-name/script-derived, route-id-derived,
     exact-intrinsic-as-authority, common-EmitC semantic-invention, and
     non-vector-reduction stale-family attempts where applicable.
5. Generated-bundle evidence should exercise selected-boundary vector
   `reduce_add` with `route_entry_realization: false` and
   `pre_realized_body_consumed: true` when the script exposes those fields.
6. Real `ssh rvv` correctness evidence is required for any runtime/correctness
   claim, with counts including `0`, `1`, exact-VL, tail, and stress cases
   over signed inputs and RHS seed/accumulator interaction.
7. Preserve non-regression for completed standalone reduction artifact ABI
   closure as focused evidence only.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module and non-goals.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` consumes rebuilt provider
      route statement operands/results for vector `reduce_add`, including
      pre-loop setvl, loop setvl, lhs load, RHS seed/accumulator load,
      reduction, output store, loop bounds, runtime ABI, C type mappings,
      layout, store VL, and selected-body provenance.
- [x] Focused C++ target/export coverage proves positive vector `reduce_add`
      provider/candidate acceptance.
- [x] Focused C++ fail-closed mutations prove stale rebuilt-route statement
      operands/results are rejected for lhs load, RHS seed/accumulator load,
      reduction intrinsic/result, output store/VL/layout, runtime AVL/VL facts,
      C type mapping, route operand binding, and stale candidate mirrors.
- [x] Candidate metadata remains mirror-only; no acceptance path relies on
      metadata, route id, artifact name, script expectation, ABI string, exact
      intrinsic spelling, direct-route-entry-only state, common EmitC semantics,
      source-front-door residue, descriptor residue, pre-realized-fixture-only
      state, or legacy-i32 authority.
- [x] Selected-boundary generated-bundle dry-run passes for vector
      `reduce_add`, with route-entry shortcut false/fail-closed as applicable.
- [x] Real `ssh rvv` generated-bundle correctness passes for vector
      `reduce_add` over counts `0`, `1`, exact-VL, tail, and stress cases with
      signed lhs/rhs seed interaction, or an exact blocker is recorded.
- [x] Completed standalone reduction artifact ABI closure remains
      non-regressed through focused dry-run or target/export coverage.
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

1. Strengthen `validateRVVVectorReductionRouteStatementPlan` so it validates
   the exact provider-built vector reduction statement sequence and operands
   rather than only callee presence.
2. Reuse the existing target validator helper
   `validateRVVProviderBuiltRouteStep` to compare callee, operands, result
   name, and result C type for the vector reduction pre-loop and loop steps.
3. Add C++ target/export route-clone mutations in
   `TargetArtifactExportTest.cpp` that corrupt rebuilt route statement facts
   after provider route rebuild, proving the production validator catches stale
   operands/results directly.
4. Run focused target artifact tests, generated-bundle dry-runs, direct
   route-entry fail-closed probes if applicable, real `ssh rvv` correctness,
   standalone-reduction non-regression, authority scan, `git diff --check`,
   and `check-tianchenrv`.

## Out of Scope

- Do not reopen standalone add/min/max or computed-mask standalone min/max
  except as focused non-regression evidence.
- Do not start runtime-scalar standalone reduction as a new owner.
- Do not add widening/contraction, segment2, compare/select, memory movement,
  scalar frontend, high-level Linalg/Vector lowering, one-intrinsic wrapper
  dialects, plugin registry work, dashboards, broad smoke matrices,
  report-only work, or evidence-only cleanup.
- Do not add common EmitC RVV semantic choices.
- Do not derive route support from route ids, artifact names, test names,
  exact intrinsic spelling, metadata, descriptors, source-front-door markers,
  or legacy i32 helper names.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Generated-bundle selected-boundary dry-run for vector `reduce_add`.
4. Direct pre-realized route-entry probe for vector `reduce_add`, expecting
   route-entry realization to remain false or fail closed if selected-boundary
   only.
5. Real `ssh rvv` generated-bundle correctness for vector `reduce_add` over
   counts `0`, `1`, an exact-VL case, a tail case, and a stress case.
6. Focused non-regression dry-run or target/export coverage for completed
   standalone add/min/max artifact ABI closure.
7. Bounded touched-file authority scan.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Completion Evidence

- Production owner moved in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`: vector
  reduction target artifact validation now consumes the rebuilt provider route
  statement operands/results for pre-loop setvl, loop setvl, lhs load, RHS
  seed/accumulator load, reduce_add intrinsic, output store, store VL `1`,
  runtime ABI roles, result C type, loop bounds, and selected typed RVV source
  provenance.
- The shared provider-built step validator now reports stale actual result
  names/C types when a statement result mismatches the provider-derived
  expected result. This is a diagnostic-only hardening used by the vector
  reduction fail-closed test.
- Focused target/export C++ coverage moved in
  `test/Target/TargetArtifactExportTest.cpp`: route-clone mutations now prove
  fail-closed behavior for stale pre-loop AVL, loop remaining AVL, lhs load,
  RHS seed/accumulator load, reduce_add operand/result, output store pointer,
  output store VL, runtime `n` role, exact-intrinsic-as-authority, and stale
  typed-op/candidate mirrors.
- Focused C++ validation passed:
  - `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  - `./build/bin/tianchenrv-target-artifact-export-test`
- Tool rebuild passed:
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
- Selected-boundary generated-bundle dry-run passed for vector `reduce_add`:
  - `artifacts/tmp/stage2_rvv_vector_reduction_abi_closure/pre-realized-vector-reduce-add-dry`
  - Evidence JSON reports `pre_realized_selected_body: true` and
    `vector_reduction_boundary.direct_pre_realized_route_entry_supported:
    false`; the vector reduction boundary records provider route facts,
    runtime ABI order, typed compute op, statement plan, and mirror-only
    artifact metadata.
- Direct pre-realized route-entry probe failed closed before target bundle
  export, as expected for selected-boundary-only pre-realized `reduce_add`:
  - `artifacts/tmp/stage2_rvv_vector_reduction_abi_closure/direct-pre-realized-vector-reduce-add`
- Real RVV generated-bundle compile/run/correctness passed through `ssh rvv`
  for vector `reduce_add` counts `0`, `1`, `16`, `23`, and `257`:
  - `artifacts/tmp/stage2_rvv_vector_reduction_abi_closure/pre-realized-vector-reduce-add-ssh`
  - Output marker:
    `tcrv_rvv_generated_bundle_abi_reduce_add_ok counts=0,1,16,23,257`.
- Focused standalone-reduction non-regression dry-run passed for completed
  standalone add/min/max, computed-mask standalone add/min/max, and
  runtime-scalar computed-mask standalone add/min/max:
  - `artifacts/tmp/stage2_rvv_vector_reduction_abi_closure/standalone-reduction-nonregression-dry`
- Final gates:
  - `git diff --check` passed.
  - `cmake --build build --target check-tianchenrv -j2` passed, 459/459.
  - Production added-line authority scan found no new metadata-derived,
    route-id-derived, descriptor-derived, ABI-string-derived, script-derived,
    artifact-name-derived, common-EmitC-derived, source-front-door-derived,
    direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
    executable authority. The only touched-file added-line scan hit was the
    deliberate negative-test exact intrinsic string `__riscv_vadd_vv_i32m1`,
    used to prove exact-intrinsic-as-authority is rejected.
- Spec update judgment: no `.trellis/spec/` update was needed. This task
  implemented the already-recorded RVV target artifact validator boundary and
  did not add a new durable architecture rule.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/index.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-standalone-reduce-add-abi-closure/prd.md`
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-standalone-min-max-artifact-abi-closure/prd.md`
  - `.trellis/workspace/codex/journal-18.md`
- Primary inspection points:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
