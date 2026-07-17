# Stage2 RVV runtime-scalar-cmp composite executable ABI boundary

## Goal

Validate or repair the production executable artifact boundary for the
Stage 2 RVV runtime-scalar compare masked indexed gather-MAcc-scatter
composite route at current HEAD. The bounded workflow is:

```text
selected explicit or pre-realized tcrv_rvv composite body
  -> RVV plugin-local composite selected-body realization owner
  -> realized explicit tcrv_rvv.with_vl body
  -> RVVCompositeGatherMAccScatterRouteFamilyPlan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact object/header bundle
  -> generated external ABI harness
  -> ssh rvv correctness evidence when runtime behavior is claimed
```

If this executable seam is already complete after `9c03e2fa`, prove it with
focused non-dry-run evidence for both explicit and pre-realized selected-body
inputs, keep fail-closed executable-boundary evidence passing, and close only
that evidence gap. If current HEAD exposes a stale or missing production fact,
repair the smallest RVV-plugin/provider/target/script boundary that prevents
truthful executable claims.

## What I Already Know

* Current HEAD is `9c03e2fa` and the worktree was clean before this task was
  created.
* The immediate previous task
  `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-pre-realized-composite-family-fact-fail-closed-owner-boundary/`
  hardened the composite selected-body realization owner so incomplete and
  duplicate gather/MAcc/scatter pre-realized clusters fail closed at the named
  composite owner boundary.
* Older archived task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-pre-realized-composite-executable-abi-closure/`
  recorded non-dry-run pre-realized composite `ssh rvv` evidence at an earlier
  HEAD, so current work must re-validate after the later owner-boundary change
  instead of assuming the old evidence is still current.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines the exact composite
  route contract: runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, same compare mask
  for gather/MAcc/scatter, old-destination passthrough for inactive gather
  lanes, inactive scatter preservation, MAcc result as the indexed store value,
  and provider-derived route/header/type/resource facts.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires the common EmitC
  route to consume provider-built payload only. Common EmitC/export must not
  infer RVV semantics, ABI order, dtype/config, mask, index, MAcc, scatter, or
  intrinsic facts.
* `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  evidence for this masked route to expose mask/tail policy and composite
  resource mirrors, and requires real `ssh rvv` output before runtime
  correctness is claimed.

## Requirements

* Reproduce the current explicit selected-body and pre-realized selected-body
  generated-bundle paths for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* Prove materialized selected-boundary, provider route, emission plan, target
  artifact export, generated header/object bundle, ABI order, route operand
  binding, composite plan id, typed compute chain, provider-supported mirror,
  resource budget, runtime AVL/VL, mask/tail policy, and header/type facts
  remain aligned.
* Run non-dry-run `ssh rvv` correctness evidence before claiming executable
  behavior. The harness must check active lanes, inactive accumulator/output
  preservation, index behavior, source/payload/accumulator preservation, and
  tail sentinel preservation.
* Preserve or add focused fail-closed evidence for stale executable-boundary
  facts such as provider mirror, ABI order, exec ABI binding, header/prototype,
  composite resource budget, typed compute chain, mask/index/scatter/MAcc
  relationship, or unsupported executable claim.
* Keep Common EmitC/export neutral and route-family facts provider-owned.

## Acceptance Criteria

* [x] Current HEAD explicit selected-body generated bundle passes non-dry-run
      `ssh rvv` evidence for the composite route, or the exact fail-closed
      production blocker is implemented and reported.
* [x] Current HEAD pre-realized selected-body generated bundle passes
      non-dry-run `ssh rvv` evidence for the composite route, or the exact
      fail-closed production blocker is implemented and reported.
* [x] Focused dry-run/lit evidence still exposes materialized boundary,
      emission plan, target artifact metadata, generated C/header ABI, route
      operand binding, composite plan id, typed compute chain, provider mirror,
      resource selection, and representative stale-boundary rejection.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit filter for
      `runtime-scalar-cmp-masked-indexed-gather-macc-scatter` passes.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if the
      script is touched.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor-driven compute, source-front-door, or
      Common EmitC semantic authority.

## Completion Notes

* Reproduced the current explicit and pre-realized composite generated-bundle
  workflows at HEAD `9c03e2fa`.
* No production C++/MLIR/Python source change was needed. Current HEAD already
  carries the executable seam from selected explicit/pre-realized
  `tcrv_rvv` bodies through RVV plugin-local realization, provider-owned
  composite route facts, common EmitC materialization, target artifact
  validation, generated header/object bundle, external ABI harness, and
  `ssh rvv` correctness evidence.
* The pre-realized path explicitly records
  `selected_body_realization_producer =
  rvv-plugin-local-selected-body-realization-owner-registry`,
  `materializer = tcrv-materialize-selected-lowering-boundaries`,
  `pre_realized_body_consumed = true`, and `route_entry_realization = false`.
* The generated bundle evidence for both input modes records runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, route operand
  binding plan `rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1`,
  provider mirror
  `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated`,
  composite resource vector-register budget `32`, and legal resource
  selection for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* Focused lit coverage still checks explicit and pre-realized dry-run evidence,
  generated harness/source preservation checks, composite resource metadata,
  stale composite plan/resource rejection, missing resource rejection, and
  selected-boundary artifact/header facts.

## Validation Evidence

* `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-current-explicit-composite-dry --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  passed with `dry_run_success`.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-current-pre-composite-dry --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  passed with `dry_run_success`.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-current-explicit-composite-ssh --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
  passed with `success`, `ssh_evidence=true`, input mode
  `explicit-selected-body`, selected variant `rvv_explicit_composite`, and
  expected function
  `tcrv_emitc_explicit_composite_masked_indexed_gather_macc_scatter_kernel_rvv_explicit_composite`.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-current-pre-composite-ssh --overwrite --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
  passed with `success`, `ssh_evidence=true`, input mode
  `pre-realized-selected-body`, selected variant `rvv_pre_composite`, and
  expected function
  `tcrv_emitc_pre_realized_composite_masked_indexed_gather_macc_scatter_kernel_rvv_pre_composite`.
* Remote output for both non-dry-run commands includes cases for counts
  `0,1,16,17,257`, RHS scalars `-37,91`, patterns `0,1`, active/inactive lane
  counts, inactive lane preservation, noncontiguous index lanes, signed product
  lanes, `source_preserved`, `payload_acc_preserved`, `tail_preserved`, and
  final `PASS`.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter=runtime-scalar-cmp-masked-indexed-gather-macc-scatter`
  from `build/test` passed: 3 selected, 3 passed.
* `scripts/rvv_generated_bundle_abi_e2e.py --self-test` was not run because the
  script was not touched.
* Bounded old-authority scan over touched files and added diff lines found no
  new positive legacy authority. Mentions in this PRD are explicit negative
  guardrails.

## Out Of Scope

* Broad gather/MAcc/scatter matrices, dtype/LMUL clone batches, unrelated
  memory/MAcc/scatter/gather/reduction/segment/conversion coverage, high-level
  Linalg/Vector/StableHLO frontend routes, source-front-door positive routes,
  scalar fallback, IME/offload/TensorExt work, or tuning databases.
* Generic composite framework rewrites or new route families.
* Treating route ids, helper names, ABI strings, artifact names, test names,
  manifests, target metadata, or Common EmitC branches as semantic authority.
* Report-only closeout if current executable evidence fails or if production
  facts are stale.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous task context read:
  `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-pre-realized-composite-family-fact-fail-closed-owner-boundary/prd.md`,
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-pre-realized-composite-executable-abi-closure/prd.md`.
* Primary files to inspect:
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.
