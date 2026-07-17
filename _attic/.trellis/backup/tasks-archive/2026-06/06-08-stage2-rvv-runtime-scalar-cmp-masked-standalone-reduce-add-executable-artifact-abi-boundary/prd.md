# Stage2 RVV runtime-scalar-cmp masked standalone reduce-add executable artifact ABI boundary

## Goal

Close one bounded Stage 2 RVV executable artifact/ABI boundary for
`runtime_scalar_cmp_masked_standalone_reduce_add`: selected or pre-realized
typed `tcrv_rvv` body facts must flow through RVV plugin-local realization,
standalone-reduction route planning, provider-built `TCRVEmitCLowerableRoute`,
common EmitC materialization, target artifact export, generated bundle ABI, and
real `ssh rvv` correctness evidence. If current HEAD already owns this seam,
finish only after focused positive, fail-closed, and executable evidence proves
the boundary; otherwise repair the bounded production seam before claiming
runtime behavior.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar-cmp masked standalone reduce-add executable artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial raw `git status --short` had no entries; `rtk git status --short`
  reported `ok` because RTK filtered the empty status.
* Initial `git log --oneline -8` started at
  `1bf35996 rvv: record runtime scalar standalone minmax evidence`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief and started before source edits.
* `.trellis/spec/index.md` keeps the RVV route authority chain at selected
  typed `tcrv_rvv` body -> RVV plugin realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires runtime-scalar
  computed-mask standalone reductions to keep source/work vector facts,
  scalar accumulator/result channel facts, runtime scalar splat/compare facts,
  inactive-lane neutralization, scalar seed/result layout, runtime AVL/VL, and
  operand bindings as RVV provider-owned facts before route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires standalone
  reduction route mirrors such as `reduction_kind`, accumulator/result layout,
  `reduction_store_vl`, route operand binding, headers/types, and runtime
  AVL/VL to be provider-derived mirrors, never route selectors.
* `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  evidence for computed-mask standalone reductions to check provider facts,
  inactive neutral behavior, all-inactive seed preservation, source/tail
  preservation, runtime counts, and real `ssh rvv` compile/run before claiming
  runtime correctness.
* The archived 2026-06-02 reduce-add tasks previously repaired production
  seam issues including accumulator header participation, provider-owned
  inactive neutral helper usage, target fail-closed validation, and `ssh rvv`
  evidence for runtime-scalar masked standalone reduce-add.
* The archived 2026-06-08 min/max executable task proved the sibling
  runtime-scalar masked standalone min/max seam without production source
  changes, after dry-run, fail-closed, target, script, and `ssh rvv` evidence.
* Current code inspection shows existing production hooks for this route:
  `RVVStandaloneReductionSelectedBodyRealizationOwner.cpp` validates
  runtime-scalar pre-realized add/min/max bodies, realizes `rhs_scalar` through
  `tcrv_rvv.splat`, uses `cmp_lhs,rhs_scalar,src,acc,out,n`, and emits
  `tcrv_rvv.masked_standalone_reduce`.
* Current route planning and statement-plan owners already route the
  standalone reduction family through provider plan verification,
  materialization facts, math operand bindings, runtime AVL/VL route-control
  facts, and a standalone reduction statement plan before
  `TCRVEmitCLowerableRoute` construction.
* Current target validation already checks runtime-scalar computed-mask
  standalone reduction statements: loop `setvl`, compare lhs load, RHS scalar
  splat, source load, compare, inactive neutral splat, merge, scalar seed
  splat, signed add/min/max reduction, and scalar-result store.
* Current generated-bundle dry-run tests already cover base add, LMUL m2, i64,
  and direct pre-realized route-entry fail-closed mode; this round still must
  run focused evidence and record whether production source changes were
  required.

## Requirements

* Keep route authority in selected typed `tcrv_rvv` body/config/runtime facts,
  RVV plugin-local realization, standalone-reduction provider plans,
  provider-built route facts, and target validation.
* Preserve and validate runtime ABI order
  `cmp_lhs,rhs_scalar,src,acc,out,n`.
* Validate `rhs_scalar` as `rhs-scalar-value`, realized by `tcrv_rvv.splat`,
  and consumed as the compare RHS in the same runtime VL scope.
* Validate compare predicate `sle`, compare-produced mask role/source/form,
  unit-stride source input, accumulator seed role/layout, reduction kind
  `add`, inactive-lane zero neutralization before reduction, scalar carry
  across runtime VL chunks, scalar lane0 result store, dtype/SEW/LMUL/config,
  policy, runtime AVL/VL, and source/scalar-result channel facts.
* Validate provider route description facts: route operand binding plan and
  summary, target leaf profile, provider-supported mirror, required headers,
  C type mapping, source/scalar-result vector types, scalar C type, reduction
  intrinsic, scalar seed splat, source splat, compare, merge, store, and
  statement-plan leaves.
* Target artifact validation must rebuild/consume provider facts and reject
  stale or missing runtime scalar binding, compare operand role, computed mask
  facts, inactive-lane neutral policy, additive seed/accumulator role,
  header/prototype binding, runtime AVL/VL, statement leaves, or stale mirrors
  before artifact acceptance.
* Generated bundle evidence must prove materialized selected boundary,
  emission plan, target artifact export, header/object ABI agreement,
  harness source shape, generated bundle compile, and `ssh rvv` correctness
  before any runtime correctness claim.
* Common EmitC/export must remain neutral: it may carry provider payloads, but
  must not choose RVV semantics, infer dtype/config, or synthesize runtime ABI.

## Acceptance Criteria

* [x] PRD, `implement.jsonl`, and `check.jsonl` describe the bounded module
      owner and relevant specs.
* [x] Focused production inspection either identifies and repairs a bounded
      reduce-add artifact/ABI seam defect, or records an exact no-source-change
      justification based on provider/target/script evidence.
* [x] Focused dry-run generated-bundle evidence passes for pre-realized
      `runtime_scalar_cmp_masked_standalone_reduce_add`; LMUL m2 and i64 are
      included only if they remain the same coherent seam.
* [x] Focused fail-closed evidence rejects a stale or unsupported executable
      boundary, at minimum the direct pre-realized route-entry shortcut for
      runtime-scalar masked standalone reduce-add.
* [x] Focused target/RVV lit evidence checks materialized selected boundary,
      emission plan, target header artifact, runtime ABI order, route operand
      binding, `reduction_kind = add`, inactive neutral policy, scalar
      seed/result layout, and header prototype.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Generated bundle execution on `ssh rvv` passes before runtime correctness
      is claimed, covering counts `0`, `1`, a VL-boundary count, a tail count,
      and a larger count, with two `rhs_scalar` values and source/seed/tail
      preservation checks.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or
      mirror-only authority path.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis task
      validation pass.
* [x] Task status, journal, archive state, commit, and final worktree status
      are truthful.

## Out Of Scope

* No broad standalone-reduction matrix.
* No dtype/LMUL clone batch; i64 or LMUL m2 only as same-seam references if
  existing checks already bind them.
* No min/max rework except as reference.
* No MAcc, product-reduce, dequant, clamp, segment2, indexed, strided,
  compare/select, widening conversion, memory-route, or unrelated mask-route
  rework.
* No high-level Linalg/Vector/StableHLO frontend work.
* No source-front-door positive route and no descriptor-driven/source-export
  route authority.
* No common EmitC invention of RVV reduction behavior.

## Technical Notes

Specs and context read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-standalone-minmax-executable-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi-boundary/prd.md`

Focused source/test files inspected:

* `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`
* `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-typed-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add-i64.mlir`

## Completion Evidence

No production source change was required. Current HEAD already owns the
runtime-scalar-cmp masked standalone reduce-add executable artifact/ABI seam:
the pre-realized typed body is consumed by the RVV standalone-reduction
selected-body realization owner, `rhs_scalar` is realized through
`tcrv_rvv.splat`, the compare-produced mask feeds
`tcrv_rvv.masked_standalone_reduce {kind = "add"}`, provider planning verifies
standalone reduction family facts before route construction, target validation
checks the rebuilt statement plan and runtime AVL/VL contract before artifact
acceptance, and generated-bundle evidence proves the external ABI on `ssh rvv`.

Checks run:

* [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-08-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-executable-artifact-abi-boundary`
* [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `build/bin/tianchenrv-target-artifact-export-test`
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-standalone-reduce-add'` from `build/test`: 7/7 passed.
* [x] Direct generated-bundle dry-run for pre-realized
      `runtime_scalar_cmp_masked_standalone_reduce_add`,
      `runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2`, and
      `runtime_scalar_cmp_masked_standalone_reduce_add_i64`:
      `rvv_generated_bundle_abi_e2e: dry_run_success`.
* [x] Dry-run evidence JSON spot checks found provider-derived
      `reduction_kind = add`, `runtime-scalar-splat-compare-rhs`,
      `cmp_lhs,rhs_scalar,src,acc,out,n`,
      `masked-standalone-reduction-zero-inactive-lanes-before-reduction`,
      accumulator layouts, route operand binding with `hdr`, source/seed/tail
      contracts, `route_entry_realization: false`, and
      `pre_realized_body_consumed: true`.
* [x] Non-dry-run generated bundle execution on `ssh rvv` passed for
      `runtime_scalar_cmp_masked_standalone_reduce_add`,
      `runtime_scalar_cmp_masked_standalone_reduce_add_lmul_m2`, and
      `runtime_scalar_cmp_masked_standalone_reduce_add_i64`.
* [x] `ssh rvv` evidence covered counts `0,1,16,23,257`, rhs scalars
      `-37,91`, seeds `-11,17`, patterns `0,1`, and reported
      `source_preserved tail_preserved` for all three op kinds.
* [x] Evidence JSON spot checks found `dry_run: false`,
      `remote_compile_succeeded: true`, `remote_run_succeeded: true`,
      `ssh_evidence: true`, `status: success`, `reduction_kind = add`, and
      `runtime_abi_order = cmp_lhs,rhs_scalar,src,acc,out,n` for all three op
      kinds.
* [x] `git diff --check`
* [x] `git diff --cached --check`
* [x] Bounded old-authority scan over the task files found only negative
      checklist/out-of-scope wording and no source-file diff.
* [x] Spec update review: no `.trellis/spec/` change needed because this round
      discovered no new executable contract; it verified existing
      standalone-reduction/runtime-scalar/generated-bundle contracts.

Evidence paths:

* `artifacts/tmp/06-08-runtime-scalar-cmp-masked-standalone-reduce-add-executable/dry-run/pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add/`
* `artifacts/tmp/06-08-runtime-scalar-cmp-masked-standalone-reduce-add-executable/ssh-rvv/pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add/`
