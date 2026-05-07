# Boundary-Aware Emission Planning Diagnostics

## Goal

Make emission-plan/readiness diagnostics consume the selected plugin-owned lowering-boundary metadata before materializing emission-plan diagnostics, so downstream diagnostics are tied to the actual RVV or scalar boundary surface selected by the generic pipeline.

## Requirements

* Preserve the existing public `--tcrv-execution-planning-pipeline` order and behavior while strengthening the emission-plan/readiness stage.
* Require a selected RVV or scalar path to have exactly one matching plugin-owned lowering boundary before emission-plan diagnostics are materialized.
* Validate missing boundaries, origin mismatch, selected-variant mismatch, required-capability mismatch, duplicate competing boundaries, stale selected surfaces, and rerun duplicates with deterministic diagnostics.
* Keep generic transforms target-neutral. Core code may inspect generic boundary-contract metadata but must not hard-code RVV/scalar operation names as selected-path policy.
* Keep RVV emission readiness unsupported and scalar fallback metadata-only. Do not add executable lowering, runtime ABI, object generation, correctness, or performance claims.
* Keep scalar and RVV boundary metadata plugin-owned and extension-local.

## Acceptance Criteria

* [ ] Valid scalar fallback-only pipeline output includes `tcrv_scalar.lowering_boundary` and emission-plan diagnostics tied to that boundary.
* [ ] Valid RVV pipeline output includes `tcrv_rvv.lowering_boundary`, emission-plan diagnostics tied to that boundary, and unsupported RVV plan status.
* [ ] RVV proposal decline plus valid scalar fallback completes through scalar boundary-aware emission planning.
* [ ] Pipeline output remains parseable by `tcrv-opt`.
* [ ] Missing boundary, origin mismatch, selected-variant mismatch, duplicate boundary, stale rerun/selection mismatch, and invalid RVV metadata are covered by lit or C++ tests.
* [ ] Existing scalar/RVV plugin, selection, lowering-boundary, emission-readiness, and execution-pipeline tests continue to pass.

## Definition of Done

* Compiler behavior remains implemented in C++ / MLIR / TableGen / CMake / lit or C++ tests.
* Specs are updated only for durable boundary-aware emission planning contracts.
* `git diff --check`, CMake configure, and `check-tianchenrv` pass locally.
* Any Trellis task state created for this work is archived before the final commit.
* A single coherent commit is created and the repo is clean.

## Technical Approach

Extend the existing emission-readiness/emission-plan helper path instead of creating a new pass. Reuse the existing selected-path traversal, collect/validate a generic boundary descriptor for each selected path, validate it against the selected variant's origin, role, selected variant symbol, and required capabilities, and materialize a minimal diagnostic reference to the boundary operation name.

## Out of Scope

* No scalar executable lowering.
* No RVV lowering, intrinsic emission, runtime ABI, object generation, hardware execution, correctness, or performance evidence.
* No new compute dialect or compute op in `tcrv.exec`.
* No target-family branches in core transforms.
* No `ssh rvv` run unless executable RVV claims are introduced, which this task explicitly does not do.

## Technical Notes

* `lib/Transforms/EmissionPlan.cpp` does not exist in current HEAD; emission-plan materialization is implemented in `lib/Transforms/EmissionReadiness.cpp`.
* Existing plugin-owned boundary ops:
  * `tcrv_rvv.lowering_boundary`: `source_kernel`, `selected_variant`, `role`, `status`, optional `capability_summary`, `unsupported_reason`.
  * `tcrv_scalar.lowering_boundary`: `source_kernel`, `selected_variant`, `origin`, `role`, `status`, `required_capabilities`, optional `fallback_reason`.
* Existing emission diagnostics are `tcrv.exec.diagnostic {reason = "emission_plan"}` with target/origin/role/status and route metadata.
