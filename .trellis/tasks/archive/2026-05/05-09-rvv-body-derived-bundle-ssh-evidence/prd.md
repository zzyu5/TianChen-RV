# rvv body-derived bundle ssh evidence

## Goal

Produce bounded runtime-correctness evidence on the `ssh rvv` host for the body-derived RVV i32-vadd microkernel target-artifact bundle route.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Expected HEAD is `d37bf7a feat: derive rvv microkernel C steps from typed body`.
* The previous round changed `lib/Target/RVV/RVVMicrokernel.cpp` so generated RVV C intrinsic emission steps are derived from the verified `tcrv_rvv.with_vl` body order, SSA chain, and `buffer_role` attributes.
* The previous round ran local compiler/export checks and `check-tianchenrv`, but did not run `ssh rvv`.
* This round should first use `scripts/rvv_microkernel_e2e.py` with `--use-target-artifact-bundle` and `--use-plan-and-export-bundle-front-door`.

## Requirements

* Run the required repository inspection before editing code or producing evidence.
* Prefer the existing front-door evidence path and the canonical MLIR fixture selected after inspecting tests.
* Run a local dry-run first.
* Verify the generated bundle includes RVV source, header, object, and typed-body provenance.
* Run the same route live on `ssh rvv`, compiling/linking/running the generated external ABI caller with the generated source/header/object.
* Keep claims scoped to bounded RVV i32-vadd microkernel external ABI runtime correctness for this route.
* If a concrete mismatch blocks the route, make the smallest necessary fix on allowed surfaces only.
* Do not introduce generic intrinsic IR, MLIR vector lowering, LLVM intrinsic lowering, core compute ops, or RVV-family branches in core orchestration.

## Acceptance Criteria

* Local dry-run succeeds and records evidence under `artifacts/tmp`.
* Dry-run evidence proves the intended artifact route and source provenance.
* Live `ssh rvv` evidence succeeds with an accepted success marker such as `tcrv_rvv_microkernel_external_abi_ok` or `tcrv_rvv_microkernel_ok`.
* Final report includes exact commands, artifact paths, selected compile flags, `ssh_evidence`, success marker, claim scope, Trellis state, repo cleanliness, and commit status.

## Out of Scope

* Performance measurement or claims.
* Generic RVV lowering claims.
* High-level linalg/stablehlo/tosa lowering prerequisites.
* Offload descriptor work unless directly blocking this RVV evidence path.
* New helper framework, broad negative matrix, dashboard, or report package.

## Technical Notes

* Relevant specs: core dialect contract, capability model, plugin protocol, RVV plugin, lowering/runtime, and MLIR testing contract.
* Python is allowed here only as evidence orchestration.

## Outcome

* Local dry-run passed with `scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --run-id rvv-microkernel-body-derived-bundle-dry --overwrite --input test/Target/TargetArtifactBundleExport/target-artifact-bundle-positive.mlir`.
* Live `ssh rvv` evidence passed with the same bundle front door and input under run id `rvv-microkernel-body-derived-bundle-ssh`.
* Evidence JSON: `artifacts/tmp/rvv_microkernel_bundle_e2e/rvv-microkernel-body-derived-bundle-ssh/evidence.json`.
* Success marker: `tcrv_rvv_microkernel_external_abi_ok`.
* Scope: bounded RVV i32-vadd target-artifact bundle external caller correctness only.
