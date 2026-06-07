# Stage2 RVV pre-realized composite executable artifact ABI closure

## Goal

Make the pre-realized
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` selected-body path
truthfully reach an executable generated RVV artifact bundle under the new
`RVVCompositeGatherMAccScatterRouteFamilyPlan` contract, or fail closed at the
exact stale/missing production fact. The bounded workflow is:

```text
selected pre-realized tcrv_rvv composite family bodies
  -> RVV plugin-local composite realization owner
  -> realized explicit composite tcrv_rvv body
  -> RVVCompositeGatherMAccScatterRouteFamilyPlan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> ssh rvv correctness evidence
```

## What I Already Know

* Current HEAD at session start was `b9503b2a`, and the worktree was clean.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-composite-route-family-plan-owner-extraction/`
  moved composite gather/MAcc/scatter route facts into an explicit
  plugin-owned route-family plan contract and proved the explicit selected-body
  non-dry-run `ssh rvv` path.
* The previous task left the pre-realized generated-bundle path covered as
  dry-run evidence, but this round's brief identifies executable pre-realized
  bundle/ABI closure as the bottleneck.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires this route to be
  owned by typed `tcrv_rvv` body structure, RVV plugin-local realization,
  provider route facts, and fail-closed diagnostics. It currently still says
  pre-realized composite coverage should fail closed until a realization owner
  exists, so successful implementation here likely needs a focused spec update.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC to
  consume provider-built route payloads only; it must not infer RVV semantics,
  ABI order, dtype/config, mask, index, MAcc, scatter, or intrinsic facts.

## Requirements

* Preserve or repair the production path that fuses the three pre-realized
  composite family bodies into one explicit realized `tcrv_rvv.with_vl` body.
* The realized body must carry runtime scalar compare, computed mask, indexed
  gather, masked MAcc, masked indexed scatter, accumulator/result preservation,
  source/tail preservation, dtype/SEW/LMUL/policy, runtime AVL/VL, ABI order,
  target capability, resource budget, composite plan id, typed compute chain,
  and header evidence facts through the same provider-owned plan contract used
  by explicit selected-body coverage.
* The generated bundle script must support a positive non-dry-run
  pre-realized selected-body path for this route, or reject unsupported/stale
  facts before claiming executability.
* Target artifact validation must fail closed for stale or missing composite
  plan, typed compute chain, ABI order, header evidence, exec ABI binding, or
  resource budget facts touched by this work.
* Keep Common EmitC/export neutral and avoid new route-family expansion,
  dtype/LMUL clone batches, source-front-door positives, descriptor-driven
  computation, or helper-name/test-name/route-id semantic authority.

## Acceptance Criteria

* [x] A focused production diff repairs or validates the pre-realized
      selected-body executable artifact/ABI path, or records an exact
      production fail-closed gap if execution remains unsupported.
* [x] Pre-realized generated-bundle non-dry-run `ssh rvv` evidence passes for
      representative counts/patterns/scalars and checks active-lane
      correctness, inactive accumulator preservation, source/tail
      preservation, ABI order, composite plan id, typed compute chain, and
      header evidence.
* [x] Explicit selected-body regression coverage still passes for touched
      route/provider/target behavior.
* [x] Focused fail-closed evidence covers stale composite plan id, stale typed
      compute chain, wrong ABI order, missing header evidence, stale resource
      budget, or unsupported executable claim as needed for touched code.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit filter for
      `runtime-scalar-cmp-masked-indexed-gather-macc-scatter` passes.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if the
      script is touched.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor-driven compute, source-front-door, or
      Common EmitC semantic authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly explained.

## Completion Notes

* Reproduced the current pre-realized selected-body generated-bundle path and
  found that current HEAD already closes the executable artifact/ABI path:
  `--pre-realized-selected-body` produced a generated bundle and passed
  non-dry-run `ssh rvv` correctness for counts `0,1,16,17,257`, RHS scalars
  `-37,91`, and patterns `0,1`.
* No production C++/MLIR/script change was needed. The production path already
  flows through selected lowering-boundary materialization, the
  plugin-local composite realization owner, realized `tcrv_rvv` body,
  `RVVCompositeGatherMAccScatterRouteFamilyPlan`, provider-built route,
  target artifact validation, generated header/object bundle, and the remote
  ABI harness.
* Updated `.trellis/spec/extension-plugins/rvv-plugin.md` because the route
  contract still described pre-realized composite coverage as unsupported until
  an owner exists. The spec now records the implemented owner-positive path
  and keeps fail-closed requirements for missing, duplicate, incomplete, stale,
  or unsupported family facts.
* Focused lit coverage still proves stale provider mirror, ABI order, exec ABI
  binding, composite resource budget, and missing exec binding fail closed.

## Validation Evidence

* `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-pre-composite-dry-repro --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  passed with `dry_run_success`.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-pre-composite-ssh-repro --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
  passed with `success`, `ssh_evidence=true`, `remote_compile_succeeded=true`,
  and `remote_run_succeeded=true`.
* Remote output includes active/inactive lane counts, inactive preservation,
  noncontiguous index lanes, signed product lanes, `source_preserved`,
  `payload_acc_preserved`, `tail_preserved`, and final `PASS` for counts
  `0,1,16,17,257`, RHS scalars `-37,91`, and patterns `0,1`.
* Generated header evidence includes ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`,
  `tianchenrv.rvv.composite_route_family_plan:
  rvv-composite-gather-macc-scatter-route-family-plan.v1`,
  `tianchenrv.rvv.composite_typed_compute_chain:
  tcrv_rvv.masked_indexed_load+tcrv_rvv.masked_macc+tcrv_rvv.masked_indexed_store`,
  resource budget `32`, and the exported prototype
  `void tcrv_emitc_pre_realized_composite_masked_indexed_gather_macc_scatter_kernel_rvv_pre_composite(const int32_t *cmp_lhs, int32_t rhs_scalar, const int32_t *gather_src, const int32_t *payload, const int32_t *acc, const uint32_t *index, int32_t *dst, size_t n);`.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='runtime-scalar-cmp-masked-indexed-gather-macc-scatter'`
  from `build/test` passed: 3 tests selected, 3 passed.
* Explicit selected-body generated-bundle dry-run regression passed for the
  same route and runtime counts/scalars.
* `scripts/rvv_generated_bundle_abi_e2e.py --self-test` was not rerun because
  the script was not changed in this task.
* Bounded old-authority scan over touched files found only existing negative
  spec/PRD mentions; added diff lines contain no new positive legacy authority.

## Definition Of Done

* The task ends with one coherent commit if the executable pre-realized path is
  closed or a precise fail-closed production boundary is implemented.
* Trellis context and workspace journal record the actual evidence, not just
  the intended plan.
* `.trellis/spec/` is updated if this round changes the durable pre-realized
  composite contract from fail-closed placeholder to implemented owner path.

## Technical Approach

1. Reproduce the current explicit and pre-realized generated-bundle behavior,
   including the non-dry-run pre-realized script path.
2. If pre-realized execution fails, trace whether the gap is in selected-body
   realization, composite plan facts, target validation/header metadata,
   generated bundle ABI generation, or ssh runtime checking.
3. Repair the smallest production owner/generator/validation gap that lets the
   pre-realized path execute truthfully; otherwise add a targeted fail-closed
   diagnostic for the precise stale/missing fact.
4. Run focused C++ tests, lit tests, script self-test if touched, and `ssh rvv`
   non-dry-run evidence before finishing.

## Out Of Scope

* Broad composite matrices, new route-family coverage, dtype/LMUL clone
  batches, performance tuning databases, dashboards, report-only changes, or
  high-level Linalg/Vector/StableHLO frontend work.
* Source-front-door positive routes, per-Linalg route authority, descriptor
  computation, Common EmitC RVV semantic invention, or compatibility wrappers
  preserving old route authority.
* Unrelated MAcc, mask, memory, reduction, segment, conversion, IME, offload,
  TensorExt, scalar fallback, or classroom branch work.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/testing/index.md`.
* Previous task context read:
  `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-composite-route-family-plan-owner-extraction/prd.md`.
* Primary implementation surfaces to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.
