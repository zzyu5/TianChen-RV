# Stage2 RVV runtime-scalar-cmp masked segment2 load artifact ABI boundary

## Goal

Harden the existing RVV runtime-scalar compare + computed-mask + masked
segment2 load selected-body route at the executable artifact ABI boundary. The
route must line up typed `tcrv_rvv` body facts, `rhs_scalar` ABI binding,
runtime-scalar splat compare, computed mask facts, inactive-lane passthrough,
segment2 field roles, dtype/SEW/LMUL/policy, runtime AVL/VL, provider-owned
route-family facts, `TCRVEmitCLowerableRoute`, target artifact export,
generated bundle ABI, and `ssh rvv` correctness evidence.

If the route is already executable, this round must not repeat evidence as a
standalone achievement. It should either name the exact already-closed seam or
repair a focused stale/under-validated production boundary found during the
inspection.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `31992e18`.
* The Hermes brief says the next bottleneck is runtime-scalar-cmp masked
  segment2 load after computed-mask segment2 update.
* Workspace journal and archived Trellis task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi/`
  show that explicit and pre-realized runtime-scalar segment2 load/store
  generated bundles already ran on `ssh rvv` earlier on 2026-06-08.
* The current task is therefore not allowed to close as repeated evidence
  only. The bounded work is to inspect the current production seam and harden
  any stale classification/spec boundary for the load route.
* Existing target fixtures for
  `runtime_scalar_cmp_masked_segment2_load_unit_store` already cover stale
  runtime-scalar mask producer, `rhs_scalar` binding, field role, ABI order,
  header list, inactive-lane contract, and target artifact export failures.

## Requirements

* Stay on `runtime_scalar_cmp_masked_segment2_load_unit_store`; use segment2
  store only as bounded reference.
* Keep route authority in typed or realized `tcrv_rvv` body facts and
  RVV-provider-owned route-family facts. Common EmitC must remain a neutral
  materializer.
* Fix any production under-validation in route-family owner classification,
  provider-plan verification, statement-plan selection, target artifact
  validation, or generated bundle ABI if inspection shows a stale boundary.
* Runtime-scalar load must bind ABI/header parameters in order
  `lhs,rhs_scalar,src,out0,out1,n`.
* `rhs_scalar` must be a scalar ABI threshold realized as `tcrv_rvv.splat`
  before compare; vector-compare RHS facts must fail closed for this route.
* The load path must keep computed-mask segment2 facts separate from plain
  segment2 facts and from non-segment computed-mask memory facts.
* If runtime correctness is claimed in this round, use real non-dry-run
  `ssh rvv` generated-bundle evidence.
* Preserve the already-closed executable evidence from the archived task as
  context, but do not use it to hide a current production classification gap.

## Acceptance Criteria

* [x] Runtime-scalar segment2 load route-family provider-plan classification
      treats the route as a computed-mask segment2 consumer, not as a regular
      non-segment computed-mask memory route.
* [x] A focused C++ plugin test exercises the pre-realized
      `runtime_scalar_cmp_masked_segment2_load_unit_store` body through
      realization, provider-plan verification, and route construction.
* [x] Existing explicit and pre-realized runtime-scalar segment2 load dry-run
      generated-bundle tests still pass and expose ABI order, operand binding,
      runtime-scalar mask producer, inactive-lane contract, field roles,
      header/prototype, and harness oracle facts.
* [x] Existing target artifact fixtures still fail closed for stale
      runtime-scalar producer, stale `rhs_scalar` binding, stale field role,
      stale ABI order, stale header/prototype binding, and stale inactive-lane
      policy.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] If generated-bundle behavior is rerun, explicit and pre-realized load
      evidence either dry-runs successfully or runs on `ssh rvv` when runtime
      correctness is claimed.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or
      mirror-only route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly reported.

## Definition Of Done

* The runtime-scalar segment2 load route-family owner and spec lists agree with
  the production route support.
* Focused tests prove the route remains provider-owned, target-consumed, and
  fail-closed at stale executable-boundary facts.
* Trellis task context, PRD results, workspace journal, archive status, and
  one coherent commit are completed.

## Out Of Scope

* No broad segment2 matrix.
* No segment2 store owner expansion in this round.
* No dtype/LMUL clone batch.
* No computed-mask update/load/store rework except as reference.
* No MAcc, product-reduce, dequant, clamp, high-level frontend, per-Linalg
  route authority, tuning database, dashboard, source-front-door positive
  route, or common EmitC semantic invention.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous relevant archive:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi/`.
* Current inspection found a stale production split: the main route-planning
  path already classifies `RuntimeScalarComputedMaskSegment2LoadUnitStore`,
  but `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp` did not include it
  in the computed-mask segment2 route classification helpers.

## Results

* Production seam changed:
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp` now classifies
  `RuntimeScalarComputedMaskSegment2LoadUnitStore` as a computed-mask
  segment2 memory route, a computed-mask segment2 load/merge route, and maps it
  to `ComputedMaskSegment2LoadUnitStore` memory-form validation. This prevents
  the standalone computed-mask memory owner from treating the runtime-scalar
  segment2 load as regular non-segment computed-mask memory.
* Focused regression coverage added:
  `tianchenrv-rvv-extension-plugin-test` now exercises a pre-realized
  `runtime_scalar_cmp_masked_segment2_load_unit_store` body through RVV-local
  realization and route construction, checking the runtime scalar splat,
  compare, masked segment2 load, passthrough loads, field stores, source
  roles, family mirror, and provider-plan version.
* Spec sync completed:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now lists runtime-scalar
  computed-mask segment2 load in the route-family planning owner boundary,
  statement-plan boundary, target export computed-mask segment2 families, and
  statement-plan validation bullets.
* Existing focused tests passed:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and the lit filter
  `runtime-scalar-cmp-masked-segment2` all passed.
* Non-dry-run runtime correctness evidence was refreshed on `ssh rvv`:
  explicit selected body and pre-realized selected body both passed for
  `runtime_scalar_cmp_masked_segment2_load_unit_store` with
  `n=0,1,16,17,257`, patterns `0,1`, and `rhs_scalar=-37,91`.
  Evidence roots:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rtseg-load-explicit-ssh-after-owner-fix/evidence.json`
  and
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-rtseg-load-prerealized-ssh-after-owner-fix/evidence.json`.
  Both runs reported:
  `PASS op=runtime_scalar_cmp_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1`.
* Authority hygiene passed: touched-file and added-line scans found no new
  positive legacy i32, descriptor/direct-C, source-front-door, or mirror-only
  route authority.
