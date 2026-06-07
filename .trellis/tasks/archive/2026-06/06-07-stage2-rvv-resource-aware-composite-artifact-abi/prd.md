# Stage2 RVV resource-aware composite gather-MAcc-scatter executable artifact ABI boundary

## Goal

Close the executable artifact and runtime ABI boundary for the Stage2 RVV resource-aware
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` composite selected-body route.
The production path must line up from explicit or pre-realized typed `tcrv_rvv` bodies
through RVV plugin-local resource-aware realization, provider-owned route facts, common
EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv`
correctness evidence when executable behavior is claimed.

## Requirements

* Preserve the architecture boundary: `tcrv.exec` binds selected variants and runtime
  roles, `tcrv_rvv` owns low-level RVV body semantics, the RVV plugin owns realization,
  legality, route facts, resource validation, and intrinsic/header/type derivation, and
  common EmitC/export owns only neutral materialization mechanics.
* Inspect the existing generated-bundle path for this composite route. If it is dry-run
  only or under-validated at the executable ABI boundary, repair the narrow production
  compiler/export/runtime evidence path needed for explicit and pre-realized selected
  bodies.
* Ensure resource-aware Gearbox facts and executable-boundary facts stay coherent across
  realized `with_vl`, selected candidate, VL policy, accumulator layout, register-pressure
  budget, unroll/pipeline/prefetch intent, runtime ABI order, target capability mirrors,
  header/type mapping, mask/inactive policy, indexed gather/scatter binding, and runtime
  scalar compare binding.
* Add or update focused fail-closed evidence for at least one stale or missing executable
  boundary fact relevant to this route.
* Add harness/test updates only when they directly prove this route's compiler/export/ABI
  seam. Do not turn the task into a dashboard, report, broad matrix, or unrelated coverage
  expansion.
* Keep legacy authority out of the new path: no route-id, metadata, helper-name, test-name,
  artifact-name, source-front-door, exact intrinsic spelling, or common-EmitC semantic
  authority.

## Acceptance Criteria

* [x] Explicit selected-body input for the composite route reaches materialized selected
      boundary, emission plan, target artifact export, generated bundle compile, and real
      `ssh rvv` correctness evidence if executable behavior is claimed.
* [x] Pre-realized selected-body input for the same route reaches the same positive
      artifact/ABI evidence.
* [x] The RVV provider or target artifact boundary fails closed with targeted diagnostics
      for at least one stale or missing executable-boundary fact.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/generated-bundle checks for this route pass.
* [x] A bounded old-authority scan over touched files and added diff lines shows no new
      legacy `i32m1`/route-id/descriptor/source-front-door authority.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status --short`
      are clean or explained.

## Definition of Done

* Production compiler/export/runtime behavior is changed when evidence shows the current
  generated artifact path is dry-run-only or under-validated; otherwise the no-source-change
  decision is backed by exact repository evidence.
* Focused tests or harnesses cover the changed executable-boundary behavior.
* Specs are reviewed for whether new durable knowledge should be captured.
* Trellis task status and workspace journal are updated truthfully.
* One coherent commit is created when the task is complete.

## Technical Approach

1. Read the current specs and archived predecessor task to establish the resource-aware
   route contract.
2. Trace the current code path through:
   `RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner`,
   `RVVEmitCRoutePlanning`, target artifact route family validation,
   `RVVTargetSupportBundle`, and `scripts/rvv_generated_bundle_abi_e2e.py`.
3. Run or inspect the existing explicit/pre-realized generated-bundle tests to identify
   whether the path is still dry-run-only or missing executable-boundary validation.
4. Implement the narrow boundary repair in production C++/MLIR/TableGen/CMake/lit/Python
   harness code only where it directly supports generated bundle ABI execution evidence.
5. Add focused positive and fail-closed evidence, run the targeted checks, self-repair,
   then finish/archive and commit.

## Decision (ADR-lite)

**Context**: The previous task made resource-aware composite realization facts visible to
production route planning and target validation, but deliberately did not claim runtime,
correctness, or performance evidence.

**Decision**: This task owns only the executable artifact/runtime ABI boundary for that one
resource-aware composite route. It should repair the production path if the generated bundle
flow is dry-run-only or under-validated, and it should fail closed at the provider/artifact
boundary when executable facts are stale or missing.

**Consequences**: The task may touch compiler/export/runtime harness code for this route, but
must not expand Stage2 capability coverage, invent common EmitC semantics, or preserve old
`i32m1` route authority through compatibility wrappers.

## Out of Scope

* Additional RVV capability classes or broad Stage2 coverage expansion.
* Broad Gearbox scheduling framework, performance tuning, or benchmark claims.
* Dtype/LMUL clone batches.
* High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
* Per-Linalg route authority or one-op-per-intrinsic wrappers.
* Dashboard, index, report-only work, or broad smoke-test matrices.
* Unrelated memory, reduction, compare/select, segment, MAcc, dequant, IME,
  Offload, TensorExt, or Scalar fallback rewrites.

## Technical Notes

* Direction source: Hermes brief appended to the Codex worker base prompt on
  2026-06-07.
* Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-resource-aware-gearbox-realization/`,
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`,
  and `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.

## Completed Evidence

* Production compiler C++ did not need changes in this round: the previous
  resource-aware task had already wired composite resource facts through realized
  `with_vl`, provider route planning, target artifact mirror contracts, and
  stale/missing target artifact fail-closed lit coverage.
* This round repaired the generated-bundle executable evidence seam:
  `scripts/rvv_generated_bundle_abi_e2e.py` now checks object/header
  `tcrv_rvv.composite_resource.*` mirrors for the composite route and writes a
  `composite_resource_selection` summary under `mask_tail_policy_boundary`.
* Positive explicit `ssh rvv` evidence:
  `artifacts/tmp/stage2-resource-aware-composite-artifact-abi-ssh-rvv/explicit-resource-aware-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
* Positive pre-realized `ssh rvv` evidence:
  `artifacts/tmp/stage2-resource-aware-composite-artifact-abi-ssh-rvv/pre-resource-aware-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
* Both evidence files record `ssh_evidence=true`, selected candidate
  `rvv-composite-gather-macc-scatter-resource-candidate.v1[rt-scmp-indexed-gather-macc-scatter,e32m1,u1]`,
  `vl_policy=runtime-avl-single-setvl`, `vector_register_budget=32`, runtime ABI
  order `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, and remote PASS
  marker `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
* `.trellis/spec/testing/mlir-testing-contract.md` now records the generated-bundle
  evidence contract for `mask_tail_policy_boundary.composite_resource_selection`
  and the required object/header `tcrv_rvv.composite_resource.*` mirror checks.
