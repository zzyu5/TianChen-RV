# Stage2 RVV runtime-scalar indexed write-side acceptance boundary

## Goal

Close one bounded production workflow submodule: runtime-scalar indexed
write-side RVV routes must carry provider/target-owned acceptance facts for
source-before, destination-before, inactive/tail preservation, unique
noncontiguous indexed writes, runtime scalar compare masks, ABI/header binding,
runtime AVL/VL, dtype/config/policy, and statement-plan shape before executable
artifact acceptance. The bounded family is the already-proven standalone
`runtime_scalar_cmp_masked_indexed_scatter_store_unit_load` path plus the
related `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` composite path.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief and set current.
* Repository inspection started from a clean `main` at commit `5331fff8 rvv:
  prove runtime scalar indexed scatter evidence`.
* The two prior archived tasks proved runtime-scalar indexed gather-MAcc-scatter
  and standalone scatter generated bundles on `ssh rvv` after fixing their
  harnesses to compute expected values from pre-call snapshots.
* The existing production path already validates much of the route boundary:
  typed body/config facts, runtime scalar compare producer, compare-produced
  mask, index load/scale, source and destination ABI roles, unique indexed
  scatter, inactive-lane policy, route operand binding, header/type facts,
  runtime AVL/VL, provider route payload, target metadata mirrors, and statement
  shape.
* Inspection found no single shared provider/target-owned contract field that
  names the write-side correctness semantics now proven by the harness:
  standalone scatter active lanes use source-before values and destination
  before/false lanes/tail preserve old destination; composite active lanes use
  gather-source-before, payload-before, accumulator-before, and
  destination-before indexed scatter semantics.

## Requirements

* Stay on the runtime-scalar indexed write-side family:
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load` and
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck. Python may only update generated-bundle evidence metadata or
  self-tests if the acceptance contract surface needs it.
* Do not make Common EmitC infer RVV semantics. The contract must be provider
  built and target consumed before candidate metadata mirrors are accepted.
* Do not choose a new route family, add dtype/LMUL clone batches, introduce
  source-front-door authority, or turn harness/report metadata into route
  authority.
* Preserve prior runtime semantics. This task should tighten the executable
  acceptance boundary and fail-closed evidence; it should not change generated
  RVV arithmetic/data movement.

## Acceptance Criteria

* [x] Provider-owned route facts and validation contracts for the two bounded
  runtime-scalar indexed write-side routes carry an explicit write-side
  semantics contract, or inspection proves an equivalent named contract already
  exists.
* [x] Target artifact validation consumes that contract before candidate mirror
  acceptance and rejects stale/missing contract facts.
* [x] Positive target/generate evidence exposes the contract for standalone
  scatter and composite GMS without treating evidence metadata as route
  authority.
* [x] Focused fail-closed evidence covers at least one stale write-side
  boundary fact, preferably stale source-before or destination-before contract
  text on a bounded runtime-scalar indexed write-side route.
* [x] Existing positive generated-bundle evidence remains valid for the
  materialized selected boundary, emission plan, target artifact export,
  generated bundle compile path, and prior `ssh rvv` runtime correctness.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit dry-run tests for standalone scatter and GMS pass.
* [x] Run generated-bundle self-test if the Python evidence generator changes.
  Not applicable: the Python evidence generator was not changed.
* [x] Run bounded old-authority scan over touched files and added diff lines.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [x] Task status, workspace journal, archive, and one coherent commit are
  completed if the task finishes.

## Out Of Scope

* No broad harness audit or gather/scatter matrix.
* No unrelated source-front-door positive route.
* No new memory family.
* No dtype/LMUL clone batch.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance database or dashboard.
* No common EmitC invention of RVV semantics.
* No mass rewrite of segment2, strided dot, gearbox, reduction,
  compare/select, or unrelated MAcc routes.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior archives read:
  `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-runtime-scalar-cmp-masked-indexed-scatter-store-abi/`
  and
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-gather-macc-scatter-abi/`.
* Primary owner files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-indexed-scatter-store-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-scatter-store.mlir`,
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.

## Completion Notes

* Added provider-owned `indexedWriteSideContract` facts for computed-mask
  indexed scatter and runtime-scalar indexed gather-MAcc-scatter.
* Standalone indexed scatter uses
  `source-before-active-indexed-write;destination-before-inactive-tail-preserve`.
* Composite gather-MAcc-scatter uses
  `gather-payload-acc-before-active-indexed-write;destination-before-inactive-tail-preserve`.
* The contract is now carried through provider route facts, family plans,
  validation contracts, route descriptions, metadata mirror contracts, target
  artifact validation, header support bundle evidence, and generated bundle
  evidence.
* Added stale write-side contract negative coverage in MLIR target fixtures and
  C++ target artifact export tests.
* `ssh rvv` evidence was refreshed for both bounded pre-realized routes:
  `build/trellis-validation/stage2-runtime-scalar-indexed-write-boundary/ssh/runtime-scalar-indexed-scatter-store`
  and
  `build/trellis-validation/stage2-runtime-scalar-indexed-write-boundary/ssh/runtime-scalar-indexed-gms`.
