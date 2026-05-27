# Stage2 RVV Runtime Scalar Dual Compare Mask/Select Executable Boundary

## Goal

Close the selected-boundary executable boundary for
`runtime_scalar_dual_cmp_mask_and_select`. The path must prove that two runtime
scalar threshold ABI bindings flow from a selected `tcrv.exec` RVV variant into
typed `tcrv_rvv` compare-mask-and-select structure, then through RVV provider
facts, `TCRVEmitCLowerableRoute`, neutral EmitC, target artifact ABI/header/
object export, generated RVV C harness, and real `ssh rvv` correctness
evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current branch is `main`; the working tree was clean before task creation.
- Latest commit before task creation is
  `4da37eb6 rvv: close runtime scalar cmp select boundary`.
- No `.trellis/.current-task` existed when this task was created.
- The archived task
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-select-boundary`
  closed the adjacent `runtime_scalar_cmp_select` boundary and kept direct
  pre-realized route-entry support fail-closed.
- Existing repository surfaces already mention
  `runtime_scalar_dual_cmp_mask_and_select` in the RVV dialect verifier,
  explicit/pre-realized target fixtures, and
  `scripts/rvv_generated_bundle_abi_e2e.py`; this task must validate or fix
  the executable evidence path rather than assume those mentions are complete.
- Relevant long-term specs require:
  - selected pre-realized RVV bodies to be consumed by RVV plugin-local
    selected-body realization before provider route construction;
  - runtime ABI roles in `tcrv.exec` to remain ABI declarations, not compute or
    dtype authority;
  - common EmitC to materialize provider-built routes neutrally;
  - generated-bundle evidence to derive compare/select predicate facts from
    typed body/config/runtime/provider facts, not from route ids, artifact
    names, ABI strings, test names, or script expectations.

## Requirements

- Use the public selected lowering-boundary path for
  `runtime_scalar_dual_cmp_mask_and_select`.
- Preserve direct pre-realized route-entry unsupported behavior for this op;
  do not resurrect direct route-entry support.
- Preserve and validate:
  - operation kind `runtime_scalar_dual_cmp_mask_and_select`;
  - runtime ABI order
    `cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n`;
  - two distinct runtime scalar threshold bindings, each as scalar ABI input
    consumed by typed `tcrv_rvv.splat`/compare structure;
  - both compare predicates and their typed compare-mask producers;
  - mask-and composition, composed mask role/source/form, and input mask
    freshness;
  - true/false select value roles, select layout, output memory role, runtime
    `n`/AVL/VL relation, setvl placement, policy, SEW/LMUL, selected variant
    origin, required capabilities, provider route facts, and target artifact
    ABI/header/object mirrors;
  - generated RVV C/C++ loop, setvl, loads, scalar splats, two compares,
    mask-and composition, select/merge, store, and external harness behavior.
- If existing C++ selected-body/provider/target owners are already sufficient,
  prove that with focused generated-bundle/runtime ABI evidence and negative
  fail-closed checks.
- If validation reveals a production gap, fix the owning boundary:
  selected-body realization, route planning/provider, target artifact ABI
  validation, generated RVV C emission evidence, or harness logic as
  appropriate.
- Evidence must fail closed when any required fact is absent, stale, swapped,
  or inconsistent. The task must not accept behavior because of op names,
  route ids, artifact names, script constants, ABI strings, exact intrinsic
  spellings, direct route-entry support, source-front-door residue, descriptors,
  common EmitC semantic invention, or legacy i32 authority.

## Acceptance Criteria

- [x] The PRD accurately records the module goal, boundaries, non-goals, and
      production owner proof or production fix.
- [x] Focused selected-boundary generated-bundle dry-run passes for pre-realized
      `runtime_scalar_dual_cmp_mask_and_select` with
      `route_entry_realization=false` and `pre_realized_body_consumed=true`.
- [x] Dry-run evidence extracts realized typed facts for two runtime scalar
      splats, two compares, mask-and composition, select true/false roles,
      store, runtime AVL/VL, provider route facts, target artifact mirrors, and
      runtime ABI order.
- [x] Evidence fails closed for missing second runtime scalar, swapped scalar
      bindings, missing or stale mask input, wrong compare predicate, missing
      mask-and composition, wrong true/false binding, wrong ABI order, wrong
      AVL/VL relation, stale route id or mirror metadata, artifact-name/script-
      derived acceptance, exact-intrinsic-as-authority, and common-EmitC
      semantic invention.
- [x] Direct pre-realized route-entry remains fail-closed for
      `runtime_scalar_dual_cmp_mask_and_select` with an op-specific
      selected-boundary-only diagnostic.
- [x] `ssh rvv` generated-bundle compile/run/correctness passes for counts
      including `0`, `1`, exact-vector, tail, and stress cases.
- [x] Runtime evidence uses at least two scalar-threshold pairs and mask
      patterns covering true lanes, false lanes, equality/boundary behavior,
      signed negative behavior, and tail preservation.
- [x] Non-regression for `runtime_scalar_cmp_select` and compare/select
      selected-boundary artifacts passes.
- [x] Focused script/lit/C++ checks pass for the changed or validated boundary,
      including provider route planning and target artifact ABI consumers when
      those owners are touched or needed for proof.
- [x] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      executable authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or any blocker is exact and recorded.
- [x] Final `git status --short` is clean after the coherent commit.

## Completion Evidence

- Production owner proof: no selected-body/provider/target C++ repair was
  required. Existing production lowering already materializes the selected
  dual runtime-scalar compare/mask/select body; this round tightened generated
  bundle ABI/runtime evidence extraction and added lit coverage around that
  path.
- Changed files:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed.test`
- Selected-boundary dry-run evidence:
  - command: `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_runtime_scalar_dual_cmp_mask_select_closure/probe-dry-run-v2 --run-id probe --overwrite --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  - result: `rvv_generated_bundle_abi_e2e: dry_run_success`
  - evidence: `artifacts/tmp/stage2_runtime_scalar_dual_cmp_mask_select_closure/probe-dry-run-v2/probe/runtime_scalar_dual_cmp_mask_and_select/evidence.json`
  - extracted facts: `route_entry_realization=false`,
    `pre_realized_body_consumed=true`, two `tcrv_rvv.splat` ops, two
    `tcrv_rvv.compare` ops, `tcrv_rvv.mask_and`, `tcrv_rvv.select`,
    true/false roles, output store, runtime AVL/VL, provider metadata, and ABI
    order
    `cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n`.
- Real `ssh rvv` executable evidence:
  - command: `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_runtime_scalar_dual_cmp_mask_select_closure/runtime-scalar-dual-cmp-mask-select-ssh --run-id runtime-scalar-dual-cmp-mask-select --overwrite --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
  - result: `rvv_generated_bundle_abi_e2e: success`
  - evidence: `artifacts/tmp/stage2_runtime_scalar_dual_cmp_mask_select_closure/runtime-scalar-dual-cmp-mask-select-ssh/runtime-scalar-dual-cmp-mask-select/runtime_scalar_dual_cmp_mask_and_select/evidence.json`
  - generated bundle:
    `artifacts/tmp/stage2_runtime_scalar_dual_cmp_mask_select_closure/runtime-scalar-dual-cmp-mask-select-ssh/runtime-scalar-dual-cmp-mask-select/runtime_scalar_dual_cmp_mask_and_select/generated_bundle/`
  - coverage: counts `0,1,16,23,257`; scalar threshold pairs
    `(-37,-37)`, `(-37,91)`, `(91,-37)`, `(91,91)`; output logs showed
    mask A true lanes, mask B true lanes, composed mask true lanes,
    single-mask-only lanes, and tail preservation for every case.
- Direct route-entry fail-closed evidence:
  - command: `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/stage2_runtime_scalar_dual_cmp_mask_select_closure/direct-fail-closed --run-id direct --overwrite --op-kind runtime_scalar_dual_cmp_mask_and_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  - result: exit code `1`, diagnostic:
    `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): runtime_scalar_dual_cmp_mask_and_select`.
- Non-regression evidence:
  - focused lit: `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed|rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-select-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-select-fail-closed|runtime-scalar-dual-compare-mask-and-select-dataflow|pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select|explicit-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select'`
    from `build/test`; result `7 passed`.
  - runtime scalar single/dual dry-run: `runtime_scalar_cmp_select` and
    `runtime_scalar_dual_cmp_mask_and_select`; result `dry_run_success`.
  - compare/select dry-run:
    `cmp_select`, `cmp_select_sle`, `computed_mask_select`,
    `computed_mask_select_sle` in pre-realized selected-body mode; result
    `dry_run_success`.
  - C++ smoke: `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
  - C++ selected lowering-boundary: `build/bin/tianchenrv-rvv-selected-lowering-boundary-test` passed.
- Authority scan:
  - command: `rg -n "descriptor|direct-C|source-export|source-front-door|tcrv_rvv\\.i32_|RVVI32M1|rvv-i32m1|artifact name|route id|ABI string|common EmitC.*invent|__riscv_.*authority" scripts/rvv_generated_bundle_abi_e2e.py test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed.test`
  - result: matches are the script's existing negative guardrails/self-tests and
    the new FileCheck `implicit-check-not` lines; no new positive executable
    authority depends on descriptor/source-front-door/direct-C/legacy-i32
    residue.
- Quality gate:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`: `409/409 passed`.

## Definition Of Done

- Production path is either fixed or explicitly proven to own the runtime scalar
  dual compare mask/select selected-boundary behavior.
- Generated-bundle dry-run and `ssh rvv` executable evidence are recorded.
- Relevant tests and bounded scans are run.
- Trellis task status is truthful and archived only after acceptance is met.
- One coherent commit records the round.

## Out Of Scope

- `runtime_scalar_cmp_masked_store`.
- `runtime_scalar_cmp_masked_load_store`.
- `runtime_scalar_cmp_masked_macc_add`.
- `runtime_i32_splat_store`.
- Widening dot/MAcc cleanup, segment2, standalone reductions, dtype/LMUL clone
  batches, high-level Linalg/frontend lowering, one-intrinsic wrapper dialects,
  dashboards, broad smoke matrices, or evidence-only work unrelated to this
  executable boundary.
- Direct pre-realized route-entry support for
  `runtime_scalar_dual_cmp_mask_and_select`.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant prior task:
  - `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-select-boundary/prd.md`
- Likely evidence/tooling files:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
  - `test/Dialect/RVV/runtime-scalar-dual-compare-mask-and-select-dataflow.mlir`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
