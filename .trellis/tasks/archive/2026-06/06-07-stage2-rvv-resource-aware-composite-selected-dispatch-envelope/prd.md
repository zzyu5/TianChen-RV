# Stage2 RVV resource-aware composite selected dispatch envelope

## Goal

Carry the existing resource-aware
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` RVV composite path
through an actual `tcrv.exec` selected dispatch/fallback envelope. The selected
RVV dispatch case, scalar fallback, runtime ABI bindings, required capability
and guard facts, imported typed `tcrv_rvv` body, RVV provider route facts, common
EmitC materialization, target artifact export, generated bundle ABI, and
`ssh rvv` evidence must agree without using route ids, artifact metadata, test
names, helper names, descriptor residue, or common EmitC semantics as authority.

## What I Already Know

* The predecessor task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-resource-aware-composite-artifact-abi/`
  closed explicit and pre-realized selected-body generated-bundle ABI evidence
  for the same composite route.
* Current specs require the authority chain:
  selected `tcrv.exec` envelope -> typed `tcrv_rvv` body -> RVV plugin legality
  and selected-body realization -> provider-built `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact -> `ssh rvv` evidence when executable
  behavior is claimed.
* `tcrv.exec` owns the envelope, selected variants, dispatch, fallback, and ABI
  role declarations. It does not own RVV compute, dtype, schedule, intrinsic
  spelling, resource candidate selection, or route support.
* Generated-bundle selected-dispatch evidence fields are mirror-only after the
  RVV provider and target artifact validator have accepted the selected route.
* The bounded negative reference
  `test/Target/RVV/selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative.mlir`
  already covers selected dispatch/fallback envelope failures for a simpler
  route class.

## Requirements

* Trace the production path from a selected `tcrv.exec.dispatch` RVV case and
  conservative fallback through selected boundary materialization, imported
  typed `tcrv_rvv` composite body, RVV provider route construction,
  `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact
  export, generated bundle ABI, and executable evidence.
* If the resource-aware composite selected envelope currently stops at
  selected-body fixtures, repair the narrow production owner that loses or fails
  to validate selected dispatch/fallback, guard, ABI, body-import, resource
  candidate, VL policy, register budget, provider mirror, or artifact export
  facts.
* Preserve plugin locality: RVV semantics, legality, resource candidate
  selection, resource mirrors, intrinsic/type/header mapping, and route support
  stay RVV-plugin-owned. Common EmitC/export may only carry and validate
  provider-built neutral payloads and mirrors.
* Add a focused positive selected-envelope fixture or update the existing
  composite fixture so that it proves the actual dispatch/fallback envelope
  reaches provider route construction, target artifact export, generated bundle
  compile, and `ssh rvv` correctness when runtime correctness is claimed.
* Add or keep focused fail-closed evidence for at least one stale or missing
  executable-boundary fact relevant to this selected envelope, such as missing
  selected RVV capability, stale dispatch guard/fallback, stale runtime ABI
  order, missing body import, stale resource candidate mirror, wrong VL policy,
  wrong register budget, wrong `provider_supported_mirror`, or unsupported
  executable route claim.
* Do not add new RVV route-family expansion, dtype/LMUL clone batches, broad
  dispatch matrices, high-level frontend work, source-front-door positive
  routes, dashboards, tuning databases, or common EmitC RVV semantic branches.

## Acceptance Criteria

* [x] A selected `tcrv.exec.dispatch` RVV envelope for
      `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` imports the typed
      composite `tcrv_rvv` body and reaches RVV provider route construction.
* [x] Runtime ABI order and exported ABI/header binding for
      `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n` propagate from the
      selected envelope/body/provider path into target artifact metadata and
      generated bundle evidence.
* [x] Selected dispatch case and conservative fallback mirrors are present only
      as validated mirror facts and are checked for object/header agreement.
* [x] Resource-aware composite facts, including selected candidate,
      `runtime-avl-single-setvl`, vector register budget `32`, target capability
      mirrors, and provider support mirror, are validated before executable
      evidence is accepted.
* [x] Focused fail-closed evidence rejects at least one stale or missing
      selected-envelope/resource/executable-boundary fact with a targeted
      diagnostic.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/generated-bundle checks for the selected envelope and the
      existing explicit/pre-realized composite artifact fixtures pass.
* [x] If `ssh rvv` runtime correctness is claimed, explicit evidence path and
      PASS output are recorded in this PRD or the workspace journal.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new legacy `i32m1` route authority, source-front-door authority,
      descriptor-driven compute, or common EmitC RVV semantic authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explained.

## Definition of Done

* Production code is changed if repository evidence shows the selected
  dispatch/fallback envelope cannot carry this resource-aware composite path.
* If no production source change is needed, the no-source-change decision is
  backed by exact code, fixture, generated bundle, and `ssh rvv` evidence.
* Trellis task context, PRD evidence, and workspace journal are updated
  truthfully.
* Specs are reviewed for durable knowledge; update specs only if this task
  discovers a reusable new contract or pitfall.
* One coherent commit is created when the task is complete. If incomplete, the
  task remains open and the next continuation point is exact.

## Technical Approach

1. Inspect the selected dispatch/fallback owners in
   `VariantSelection.cpp`, `VariantMaterialization.cpp`,
   `VariantDispatchSynthesis.cpp`, `DispatchRuntimeGuard.cpp`,
   `EmitCLowerableMaterialization.cpp`, and `ExecutionPlanningPipeline.cpp`.
2. Inspect the RVV route and artifact boundary in
   `LoweringBoundary.cpp`, `RVVEmitCRouteProvider.cpp`,
   `TargetArtifactExport.cpp`, and
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Inspect the composite fixtures and generated-bundle script to determine
   whether the current positive evidence is selected-body-only or actual
   dispatch/fallback-envelope evidence.
4. Implement the smallest production repair or fixture/evidence repair that
   proves the selected-envelope chain without moving RVV semantics into common
   code.
5. Run focused checks, self-repair, record evidence, finish/archive, and commit.

## Out of Scope

* New gather, scatter, MAcc, reduction, compare/select, dtype, LMUL, or
  low-precision route-family expansion.
* High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
* IME, Offload, TensorExt, Scalar fallback rewrites, or future plugin examples.
* Performance tuning databases, dashboards, broad smoke matrices, or report-only
  work.
* Any descriptor-driven compute/export path or common EmitC invention of RVV
  semantics.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on 2026-06-07.
* Read specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Read predecessor context:
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-resource-aware-composite-artifact-abi/prd.md`
  and `.trellis/workspace/codex/journal-25.md`.

## Completed Evidence

* No production source change was required in this round. Live repository
  evidence showed that the current HEAD already carries the resource-aware
  composite path through actual `tcrv.exec.dispatch` case/fallback envelopes:
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  both define selected RVV variants, conservative scalar fallbacks, and
  `tcrv.exec.dispatch` with explicit `tcrv.exec.case` and
  `tcrv.exec.fallback` ops.
* The explicit fixture already proves selected-envelope fail-closed behavior
  for missing dispatch case, missing fallback, missing runtime guard, stale
  dispatch case mirror, stale dispatch fallback mirror, missing dispatch case
  mirror, missing fallback mirror, stale provider mirror, stale ABI order, stale
  exec ABI binding, missing exec mirror, missing exec binding, stale composite
  resource candidate, and missing composite resource VL policy.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` validates the
  runtime-scalar indexed gather-MAcc-scatter artifact consumer from rebuilt
  provider route facts and rejects missing provider-owned route contracts,
  missing/illegal composite resource selection, resource pressure over budget,
  wrong route id, stale route headers/type mappings/ABI mappings, statement
  plan shape, and stale candidate mirrors.
* `scripts/rvv_generated_bundle_abi_e2e.py` already records
  `selected_dispatch_bundle_boundary` with authority
  `actual selected tcrv.exec dispatch/fallback envelope plus selected typed or
  realized tcrv_rvv body and RVV provider route`, and separately records
  `mask_tail_policy_boundary.composite_resource_selection` as provider-route
  mirrors.
* Focused dry-run lit evidence passed for:
  `rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run`,
  `explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter`,
  `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter`,
  and `selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative`.
* Fresh explicit selected-envelope `ssh rvv` evidence:
  `artifacts/tmp/stage2-resource-aware-composite-selected-dispatch-envelope-ssh-rvv/explicit-selected-envelope-composite-gms-ssh-rvv/evidence.json`.
* Fresh pre-realized selected-envelope `ssh rvv` evidence:
  `artifacts/tmp/stage2-resource-aware-composite-selected-dispatch-envelope-ssh-rvv/pre-selected-envelope-composite-gms-ssh-rvv/evidence.json`.
* Both fresh top-level evidence files record `ssh_evidence=true`,
  `status=success`, selected dispatch bundle boundary mirrors, runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, selected candidate
  `rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]`,
  VL policy `runtime-avl-single-setvl`, vector register budget `32`, and remote
  PASS marker
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
* Checks run:
  `python3 ./.trellis/scripts/task.py validate`,
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter`,
  focused lit under `build/test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  and `build/bin/tianchenrv-target-artifact-export-test`.
* Bounded old-authority scan over the new task files only found negative or
  forbidden-context mentions in PRD/checklist text; it found no new positive
  legacy RVV route authority, descriptor-driven compute path,
  source-front-door route, or common EmitC RVV semantic authority.
* `git diff --check` passed before the first commit. After this archived PRD
  closeout update, `git diff --check`, `git diff --cached --check`, and final
  `git status --short` were rerun before amending the commit.
