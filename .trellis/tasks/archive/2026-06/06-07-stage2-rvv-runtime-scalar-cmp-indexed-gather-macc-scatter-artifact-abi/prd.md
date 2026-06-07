# Stage2 RVV runtime-scalar-cmp indexed-gather MAcc scatter executable artifact ABI

## Goal

Close the executable artifact/runtime ABI boundary for the existing RVV
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` selected-body route
family. The route must flow from explicit or pre-realized selected
`tcrv_rvv` structure through RVV-owned provider facts, common EmitC
materialization, target artifact export, generated object/header bundle, and
real `ssh rvv` correctness evidence when runtime behavior is claimed.

## What I Already Know

* The preceding archived task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-resource-aware-composite-fallback-dispatch-abi/`
  finished the unsupported fallback/export candidate gate in
  `TargetArtifactExport.cpp`.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines this composite as a
  Stage 2 RVV plugin-owned route contract with ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-built
  `TCRVEmitCLowerableRoute` payloads; common EmitC must not infer RVV
  semantics from route ids, artifact names, diagnostics, or metadata.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires target
  export to consume materialized EmitC and validated provider mirrors, and the
  previous selected-path candidate gate now fails closed for unsupported
  selected paths.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  evidence for runtime/correctness claims and specifically requires generated
  bundle evidence to record selected dispatch/fallback mirror boundaries.
* Current dry-run evidence already checks explicit and pre-realized generated
  bundle shape for this composite route, including ABI order, selected
  dispatch bundle metadata, provider support mirror, composite resource
  selection, inactive-lane preservation, source/tail preservation, and
  multi-pattern harness source.

## Requirements

* Produce non-dry-run generated-bundle evidence on `ssh rvv` for both explicit
  and pre-realized selected-body inputs for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* The evidence must have `dry_run=false`, `ssh_evidence=true`, and remote PASS
  output for runtime counts including boundary and nontrivial cases.
* Evidence must cover runtime scalar compare, compare-produced mask, indexed
  gather, masked MAcc, masked indexed scatter, active/inactive lane behavior,
  inactive destination preservation, source/payload/acc preservation, tail
  preservation, ABI order, selected dispatch case/fallback mirrors, provider
  support mirror, route operand binding plan, and object/header metadata
  agreement.
* If either non-dry-run path fails due to stale ABI, header/prototype mismatch,
  missing selected-path candidate facts, stale provider mirrors, missing
  runtime scalar binding, stale index/gather/scatter binding, or weak harness
  validation, repair the bounded production seam and add focused regression
  coverage.
* If both non-dry-run paths already pass without source changes, close the task
  truthfully as an evidence blocker closure, not as a new production-code
  change.

## Acceptance Criteria

* [x] Explicit selected-body generated bundle runs successfully on `ssh rvv`
      for `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` with
      `dry_run=false` and `ssh_evidence=true`.
* [x] Pre-realized selected-body generated bundle runs successfully on
      `ssh rvv` for the same route with `dry_run=false`,
      `ssh_evidence=true`, root `pre_realized_selected_body=true`, and
      per-op materializer/realization producer evidence.
* [x] Evidence JSON records selected dispatch bundle boundary fields and
      object/header agreement for both inputs.
* [x] Harness/source/evidence covers active gather-MAcc-scatter result,
      inactive-lane destination preservation, source preservation, payload/acc
      preservation, tail preservation, ABI order, runtime counts, and runtime
      RHS scalar cases.
* [x] At least one focused fail-closed boundary remains covered for stale ABI,
      stale provider mirror, stale exec binding, stale dispatch mirror, stale
      composite resource, or unsupported selected candidate.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/generated-bundle dry-run coverage passes.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor-driven compute, source-front-door, or
      common EmitC semantic authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly explained.

## Definition Of Done

* The executable artifact ABI boundary for this composite selected RVV path is
  closed with actual `ssh rvv` evidence or with a precise blocker repaired.
* Common EmitC/export remains neutral and does not acquire RVV compute,
  intrinsic, dtype, schedule, fallback, or ABI semantic authority.
* Trellis task status/context and workspace journal are updated truthfully.
* Specs are reviewed for durable knowledge; update specs only if this round
  discovers a reusable contract not already captured.
* One coherent commit is created when the task is complete.

## Out Of Scope

* Broad composite matrices, dtype/LMUL clone batches, unrelated memory,
  segment, reduction, compare/select, MAcc, dispatch, or source-front-door
  rewrites.
* Scalar fallback executable routes, scalar fallback runtime ABI, or scalar
  fallback artifact support.
* High-level Linalg/Vector/StableHLO frontend work, per-Linalg authority,
  performance databases, dashboards, or tuning reports.
* Moving RVV route facts into Common EmitC, Target export, artifact metadata,
  helper names, route ids, or tests.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and shared guides.
* Relevant implementation/evidence files:
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the focused Target/RVV and
  Scripts tests named in the direction brief.

## Completed Evidence

* Explicit selected-body non-dry-run generated-bundle evidence passed on
  `ssh rvv` with `status = success`, `dry_run = false`, and
  `ssh_evidence = true`.
* Pre-realized selected-body non-dry-run generated-bundle evidence passed on
  `ssh rvv` with `status = success`, `dry_run = false`, `ssh_evidence = true`,
  root `pre_realized_selected_body = true`,
  `materializer = tcrv-materialize-selected-lowering-boundaries`, and
  per-op `selected_body_realization_producer =
  rvv-plugin-local-selected-body-realization-owner-registry`.
* Both runs compiled on remote `riscv64` with `/usr/bin/clang` and exited `0`
  for remote compile and run.
* Runtime cases covered counts `0,1,16,17,257`, RHS scalar thresholds `-37,91`,
  and patterns `0,1`.
* Remote stdout included active/inactive lane counts, inactive lane
  preservation, noncontiguous indexed lanes, signed product lanes,
  `source_preserved`, `payload_acc_preserved`, `tail_preserved`, and final
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* Both evidence JSON records had `selected_dispatch_bundle_boundary` with
  `object_header_metadata_agree = true`, ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, provider supported
  mirror
  `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-plan-validated`,
  and route operand binding plan
  `rvv-route-operand-binding:rt_scmp_gather_macc_scatter.v1`.
* Initial command self-repair: the first run blocked before evidence because
  `--llvm-readobj llvm-readobj` was not resolvable on local `PATH`. Re-running
  with `/usr/lib/llvm-20/bin/llvm-readobj` succeeded.
* No source-code change was needed because the bounded positive path was not
  dry-run-only and no stale ABI/provider/dispatch/runtime fact was exposed.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* Focused lit dry-run test
  `rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run`
  passed: 1 selected test passed, 519 excluded.
* Focused Target/RVV lit filter
  `selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter`
  passed: 2 selected tests passed, 518 excluded.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test --op-kind
  runtime_scalar_cmp_masked_indexed_gather_macc_scatter` passed.
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-macc-scatter-artifact-abi`
  passed with 9 implement context entries and 6 check context entries.
* Bounded authority scan over the task files only found negative guardrail
  wording in the PRD. No production source changed and no positive old
  authority was added.
* Spec update review: no `.trellis/spec/` change is required. The existing RVV
  plugin, EmitC route, emission runtime, and testing contracts already state
  the executable authority chain and the `ssh rvv` evidence requirement; this
  round only supplies the missing positive evidence for the existing route.
* `git diff --check` and `git diff --cached --check` passed. Final
  `git status --short` is checked after commit.
