# Stage2 RVV Runtime Scalar Compare Masked Memory Side-Effect Boundary

## Goal

Close one selected-boundary executable module for
`runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store`. The round must prove that a runtime
scalar threshold binding flows from selected `tcrv.exec` RVV variants into
typed `tcrv_rvv` splat/compare/mask structure, then into masked memory side
effects, RVV provider route facts, `TCRVEmitCLowerableRoute`, neutral EmitC,
target artifact ABI, generated RVV C artifacts/harnesses, and real `ssh rvv`
correctness evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current branch is `main`; the working tree was clean before task creation.
- Latest commit before task creation is
  `3ca7ddb9 rvv: close runtime scalar dual cmp mask select boundary`.
- No `.trellis/.current-task` existed when this task was created.
- The archived previous task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-dual-cmp-mask-select-boundary`
  closed `runtime_scalar_dual_cmp_mask_and_select` through generated-bundle
  ABI/runtime evidence without touching production C++.
- `scripts/rvv_generated_bundle_abi_e2e.py` already names
  `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store` selected-body expectations. Existing
  mentions are only starting points; route/executable support must be validated
  from typed body/provider/runtime facts, not from script constants or names.
- Relevant long-term specs require:
  - `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` to remain ABI/control
    declarations, not RVV compute, dtype, policy, or route authority;
  - selected pre-realized RVV bodies to be consumed by RVV plugin-local
    realization before provider route construction;
  - RVV provider facts to own SEW/LMUL, policy, predicate, mask source, memory
    form, runtime ABI order, operand binding, and statement materialization;
  - common EmitC and target artifacts to consume provider-built routes
    neutrally and mirror metadata only after route construction;
  - runtime/correctness claims to use real `ssh rvv` evidence, with memory
    sentinel checks for inactive lanes and tail preservation.

## Requirements

- Treat `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store` as one coherent runtime-scalar
  computed-mask memory side-effect boundary.
- Use the public selected lowering-boundary path with pre-realized selected
  bodies consumed by RVV plugin realization.
- Preserve direct pre-realized route-entry unsupported behavior for both op
  kinds; do not add positive direct route-entry support.
- Validate and consume:
  - runtime ABI order for masked store:
    `lhs,rhs_scalar,src,dst,n`;
  - runtime ABI order for masked load-store:
    `lhs,rhs_scalar,src,dst,n`;
  - runtime scalar threshold binding as an ABI scalar input used by
    `tcrv_rvv.splat` and `tcrv_rvv.compare`;
  - compare predicate facts and compare-produced mask provenance;
  - mask/tail policy, runtime `n`/AVL/VL/setvl relation, selected variant
    origin, required capabilities, SEW/LMUL, and provider route-control facts;
  - masked store side effect facts: compare mask, source payload load, output
    mem_window destination, inactive false-lane preservation, and tail
    preservation;
  - masked load-store side effect facts: compare mask, source-input
    masked_load, old-destination passthrough role loaded from `dst`, final
    store to output, unchanged inactive lanes, and tail preservation;
  - provider route metadata, target artifact ABI/header/object mirrors,
    generated C signatures, argument order, RVV setvl/compare/masked
    load-store body, and harness side-effect assertions.
- If production selected-body/provider/target code is already sufficient, keep
  production stable and close the missing executable evidence boundary only.
- If validation exposes a production mismatch or stale acceptance, repair the
  owning C++ boundary rather than compensating in the script or tests.
- Evidence must fail closed when required facts are absent, stale, swapped, or
  inconsistent.

## Acceptance Criteria

- [x] PRD and Trellis context accurately record the module goal, boundaries,
      non-goals, and production-owner proof or production fix.
- [x] Focused selected-boundary generated-bundle dry-runs pass for
      `runtime_scalar_cmp_masked_store` and
      `runtime_scalar_cmp_masked_load_store` with
      `route_entry_realization=false` and `pre_realized_body_consumed=true`.
- [x] Dry-run evidence extracts realized typed facts for runtime scalar splat,
      compare mask, masked_store or masked_load plus store, runtime AVL/VL,
      provider route facts, target artifact mirrors, and ABI order.
- [x] Direct pre-realized route-entry probes fail closed for both op kinds with
      selected-boundary-only diagnostics.
- [x] Evidence checks reject missing or swapped runtime scalar binding, missing
      compare mask, wrong predicate, missing source payload role, missing old
      destination passthrough load from `dst` for load-store, wrong output
      role, wrong ABI order, wrong AVL/VL relation, stale provider mirrors,
      artifact/script/name-
      derived acceptance, exact-intrinsic-as-authority, and common-EmitC
      semantic invention.
- [x] Generated RVV C signatures and bodies prove argument order and side-effect
      behavior for both op kinds.
- [x] Real `ssh rvv` generated-bundle compile/run/correctness passes for counts
      including `0`, `1`, exact-vector, tail, and stress cases.
- [x] Runtime harness coverage includes threshold patterns for all-masked,
      none-masked, mixed active lanes, inactive lane preservation, old
      destination passthrough for load-store, and tail preservation.
- [x] Focused non-regression passes for
      `runtime_scalar_dual_cmp_mask_and_select` and
      `runtime_scalar_cmp_select`.
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

- Production owner proof: no C++ selected-body/provider/target repair was
  required. Existing production selected-boundary realization and RVV route
  provider behavior already materialized `masked_store` and `masked_load` plus
  `store` routes from typed facts; this round closed the missing workflow
  evidence boundary in `scripts/rvv_generated_bundle_abi_e2e.py` and focused
  Script tests.
- Selected-boundary dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_runtime_scalar_cmp_masked_memory_closure/probe-dry-run-v2 --run-id probe --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  passed with `route_entry_realization=false`,
  `pre_realized_body_consumed=true`, ABI order `lhs,rhs_scalar,src,dst,n`,
  provider-supported mirror metadata, and emitted RVV C evidence for both op
  kinds.
- Direct pre-realized route-entry probe failed closed for both op kinds with:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized
  op kind(s): runtime_scalar_cmp_masked_store,
  runtime_scalar_cmp_masked_load_store; these fixtures are
  selected-boundary-only and must use the public selected lowering-boundary
  producer before target bundle export`.
- Real RVV evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --ssh-target rvv --artifact-root artifacts/tmp/stage2_runtime_scalar_cmp_masked_memory_closure/ssh-rvv --run-id runtime-scalar-cmp-masked-memory --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  passed on `ssh rvv`, including inactive-lane preservation, old destination
  passthrough for load-store, source preservation, and tail preservation.
- Focused lit tests passed for the new selected-boundary dry-run and direct
  fail-closed Script tests; non-regression lit passed for
  `runtime_scalar_dual_cmp_mask_and_select` and `runtime_scalar_cmp_select`;
  Target/RVV selected-body fixture lit passed for both masked memory op kinds.
- Script checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Bounded authority scan over touched workflow/test/task files found only
  negative guards, mirror labels, existing residue checks, and PRD non-goals;
  no new executable authority depends on central ad hoc, name, metadata,
  descriptor, ABI string, script, artifact name, common EmitC, source front
  door, route id, exact intrinsic spelling, direct route-entry only,
  pre-realized fixture only, or legacy i32 facts.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 411/411.

## Definition Of Done

- Production path is either fixed or explicitly proven to own both
  runtime-scalar computed-mask memory side-effect routes.
- Generated-bundle dry-run and `ssh rvv` executable evidence are recorded.
- Direct pre-realized route-entry remains fail-closed for both op kinds.
- Relevant tests and bounded scans are run.
- Trellis task status is truthful and archived only after acceptance is met.
- One coherent commit records the completed round.

## Out Of Scope

- New compare/select variants outside the non-regression checks.
- Runtime scalar masked MAcc, widening dot/MAcc, segment2, reductions,
  conversion/dtype/LMUL expansion, high-level Linalg/frontend lowering,
  one-intrinsic wrapper dialects, dashboards, broad smoke matrices, or
  report-only work.
- Direct pre-realized route-entry positive support for either masked memory op.
- Treating artifact names, ABI strings, test names, route ids, script constants,
  mirror metadata, common EmitC output, exact intrinsic spelling, source-front
  door residue, descriptors, or legacy i32 names as semantic authority.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant prior task:
  - `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-dual-cmp-mask-select-boundary/prd.md`
- Likely evidence/tooling files:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-store.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
