# Stage2 RVV Runtime Scalar Compare Masked Standalone Reduce Add Boundary

## Goal

Close one selected-boundary generated-bundle executable module for
`runtime_scalar_cmp_masked_standalone_reduce_add`. The round must prove that a
runtime scalar threshold binding flows from a selected `tcrv.exec` RVV variant
into a typed/pre-realized `tcrv_rvv` runtime-scalar computed-mask standalone
reduction body, then into RVV plugin-local selected-body realization, provider
route facts, `TCRVEmitCLowerableRoute`, neutral EmitC, target artifact ABI,
generated RVV C artifacts/harnesses, and real `ssh rvv` correctness evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current branch is `main`; the working tree was clean before task creation.
- Latest commit before task creation is
  `7a36d25c rvv: close runtime scalar cmp masked memory boundary`.
- No `.trellis/.current-task` existed when this task was created.
- The archived previous task
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-masked-memory-side-effect-boundary`
  closed the runtime-scalar computed-mask memory side-effect boundary through
  selected-boundary generated-bundle and `ssh rvv` evidence without requiring a
  production C++ repair.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has expectation entries for
  `runtime_scalar_cmp_masked_standalone_reduce_add`, but repository inspection
  found no focused generated-bundle `test/Scripts` coverage for this runtime
  scalar standalone-reduction boundary.
- Focused pre-realized selected-boundary dry-run with current production C++
  passed for `runtime_scalar_cmp_masked_standalone_reduce_add`, with
  `route_entry_realization=false`, `pre_realized_body_consumed=true`, ABI order
  `cmp_lhs,rhs_scalar,src,acc,out,n`, provider-supported mirror metadata, and a
  generated C prototype
  `const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n`.
- The same dry-run also showed a missing evidence-quality gap: generic
  `mask_tail_policy_boundary` and `reduction_accumulation_boundary` summaries
  are populated for computed-mask standalone reduction, but the runtime-scalar
  standalone-reduction variant currently leaves those summaries empty. This is a
  script evidence issue, not proof of missing production route support.
- Relevant long-term specs require:
  - `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` to remain ABI/control
    declarations, not RVV compute, dtype, policy, or route authority;
  - selected pre-realized RVV bodies to be consumed by RVV plugin-local
    realization before provider route construction;
  - direct route-entry support to remain narrower than selected-boundary
    realization; bounded standalone reduction is selected-boundary-only and
    direct route-entry must fail closed;
  - RVV provider facts to own SEW/LMUL, policy, predicate, mask source,
    standalone-reduction family facts, scalar seed/result layout, runtime ABI
    order, operand binding, and statement materialization;
  - common EmitC and target artifacts to consume provider-built routes
    neutrally and mirror metadata only after route construction;
  - runtime/correctness claims to use real `ssh rvv` evidence.

## Requirements

- Treat `runtime_scalar_cmp_masked_standalone_reduce_add` as one coherent
  runtime-scalar computed-mask standalone reduction boundary.
- Use the public selected lowering-boundary path with a pre-realized selected
  body consumed by RVV plugin-local selected-body realization.
- Preserve direct pre-realized route-entry unsupported behavior for this op
  kind; do not add positive direct route-entry support.
- Validate and consume:
  - runtime ABI order `cmp_lhs,rhs_scalar,src,acc,out,n`;
  - runtime scalar threshold binding as an ABI scalar input used by
    `tcrv_rvv.splat` and `tcrv_rvv.compare`;
  - compare predicate facts and compare-produced mask provenance;
  - mask/tail policy, inactive-lane neutralization before reduction, runtime
    `n`/AVL/VL/setvl relation, selected variant origin, required capabilities,
    SEW/LMUL, and provider route-control facts;
  - standalone reduction facts: source payload load, scalar seed from `acc[0]`,
    active masked lane contribution, inactive lane exclusion through neutral
    zeroing, scalar result carried across runtime VL chunks, output store to
    `out[0]`, and non-scalar output sentinel preservation;
  - provider route metadata, target artifact ABI/header/object mirrors,
    generated C signature, argument order, RVV setvl/splat/compare/masked
    standalone reduction body, and harness assertions.
- If production selected-body/provider/target code is already sufficient, keep
  production C++ unchanged and close the missing executable evidence boundary.
- If validation exposes a production mismatch or stale acceptance, repair the
  owning C++ boundary rather than compensating in the script or tests.
- Evidence must fail closed when required facts are absent, stale, swapped, or
  inconsistent.

## Acceptance Criteria

- [x] PRD and Trellis context accurately record the module goal, boundaries,
      non-goals, and production-owner proof or production fix.
- [x] Focused selected-boundary generated-bundle dry-run passes for
      `runtime_scalar_cmp_masked_standalone_reduce_add` with
      `route_entry_realization=false` and `pre_realized_body_consumed=true`.
- [x] Dry-run evidence extracts realized typed facts for runtime scalar splat,
      compare mask, masked standalone reduction, inactive-lane neutralization,
      scalar seed/result layout, runtime AVL/VL, provider route facts, target
      artifact mirrors, and ABI order.
- [x] Direct pre-realized route-entry probe fails closed for
      `runtime_scalar_cmp_masked_standalone_reduce_add` with selected-boundary-
      only diagnostics.
- [x] Evidence checks reject missing or swapped runtime scalar binding, missing
      compare mask, wrong predicate, missing source payload role, wrong seed or
      output role, wrong ABI order, wrong AVL/VL relation, stale provider
      mirrors, artifact/script/name-derived acceptance, exact-intrinsic-as-
      authority, and common-EmitC semantic invention.
- [x] Generated RVV C signature and body prove argument order and reduction
      behavior.
- [x] Real `ssh rvv` generated-bundle compile/run/correctness passes for counts
      including `0`, `1`, exact-vector, tail, and stress cases with at least
      two runtime scalar thresholds and signed input patterns.
- [x] Runtime harness coverage includes mixed active/inactive lanes,
      inactive-lane nonzero payload exclusion, active positive and negative
      payloads, scalar seed preservation, scalar output slot behavior, and tail
      preservation.
- [x] Focused non-regression passes for
      `runtime_scalar_cmp_masked_store`,
      `runtime_scalar_cmp_masked_load_store`, and
      `runtime_scalar_cmp_masked_macc_add`.
- [x] Relevant script/lit/C++ checks pass for changed or validated boundaries.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and the
      script self-test pass.
- [x] Bounded authority scan over touched production/workflow files finds no
      central ad hoc, name-derived, metadata-derived, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived executable authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Final `git status --short` is clean after the coherent commit if the task
      is complete.

## Completion Evidence

- Production C++ owner files were inspected and left unchanged: the existing
  selected-boundary production path already carries the selected pre-realized
  runtime-scalar computed-mask standalone reduction into provider-built route
  facts, neutral EmitC, and target artifact export. The implemented repair is
  evidence/tooling only: script verification now extracts the runtime-scalar
  standalone reduction mask/tail and reduction/accumulation boundaries, and lit
  coverage makes that boundary executable evidence.
- Focused selected-boundary dry-run:
  `artifacts/tmp/stage2_runtime_scalar_cmp_masked_standalone_reduce_probe/pre-dry-v3/probe/evidence.json`.
  It records `route_entry_realization=false`,
  `pre_realized_body_consumed=true`, ABI order
  `cmp_lhs,rhs_scalar,src,acc,out,n`, provider mirror
  `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-standalone-reduction-plan-validated`,
  non-empty `mask_tail_policy_boundary`, non-empty
  `reduction_accumulation_boundary`, and generated prototype
  `void tcrv_emitc_pre_rt_scalar_cm_standalone_reduce_kernel_rvv_pre_rt_scalar_cm_standalone_reduce(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *src, const int32_t *acc, int32_t *out, size_t n);`.
- `ssh rvv` generated-bundle evidence:
  `artifacts/tmp/stage2_runtime_scalar_cmp_masked_standalone_reduce_closure/ssh-rvv/runtime-scalar-cmp-masked-standalone-reduce-add/evidence.json`.
  The remote compile and run commands exited `0`, and the harness covered
  counts `0,1,16,23,257`, thresholds `-37,91`, seeds `-11,17`, mixed active
  and inactive lanes, signed payloads, inactive nonzero exclusion, scalar seed
  preservation, and tail preservation.
- Direct route-entry fail-closed evidence:
  `artifacts/tmp/stage2_runtime_scalar_cmp_masked_standalone_reduce_closure/direct-fail/output.txt`
  contains the selected-boundary-only diagnostic for
  `runtime_scalar_cmp_masked_standalone_reduce_add`.
- Focused non-regression dry-runs passed for
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`, and
  `runtime_scalar_cmp_masked_macc_add`.
- Added script tests:
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`.
- Checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  manual FileCheck replay for the two new lit tests;
  `git diff --check`;
  bounded added-line authority scan;
  `cmake --build build --target check-tianchenrv -j2` with `413/413`.

## Definition Of Done

- Production path is either fixed or explicitly proven to own the
  runtime-scalar computed-mask standalone reduction route.
- Generated-bundle dry-run and `ssh rvv` executable evidence are recorded.
- Direct pre-realized route-entry remains fail-closed for the op kind.
- Relevant tests and bounded scans are run.
- Trellis task status is truthful and archived only after acceptance is met.
- One coherent commit records the completed round.

## Out Of Scope

- New reductions beyond the bounded add case.
- Runtime-scalar masked memory changes beyond non-regression.
- Runtime scalar masked MAcc changes beyond non-regression.
- Widening dot/MAcc, segment2, conversion/dtype/LMUL expansion, high-level
  Linalg/frontend lowering, one-intrinsic wrapper dialects, dashboards, broad
  smoke matrices, or report-only work.
- Direct pre-realized route-entry positive support for standalone reduction.
- Treating artifact names, ABI strings, test names, route ids, script
  constants, mirror metadata, common EmitC output, exact intrinsic spelling,
  source-front-door residue, descriptors, or legacy i32 names as semantic
  authority.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/guides/index.md`
- Relevant prior task:
  - `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-masked-memory-side-effect-boundary/prd.md`
- Likely evidence/tooling files:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
