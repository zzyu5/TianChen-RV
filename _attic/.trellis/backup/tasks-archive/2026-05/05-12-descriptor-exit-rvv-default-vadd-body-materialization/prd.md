# Descriptor exit from RVV default i32 add typed-body materialization

## Goal

Complete the bounded production/default RVV `i32-vadd` descriptor-exit step for
the no-frontend/no-hand-authored-body route. The default selected/export path
may request the finite `i32-vadd` family, but it must materialize and then
consume a typed `tcrv_rvv.i32_vadd_microkernel` body before selected emission or
target artifact export can claim supported code generation. The descriptor
family object must not remain the compute authority for this default path.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Branch is `main`; the worktree was clean before this task was created.
* Current HEAD before this task is
  `30dd5eb feat(rvv): derive direct vadd planning from typed body`.
* No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-12-descriptor-exit-rvv-default-vadd-body-materialization/`.
* The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-descriptor-exit-direct-rvv-variant-planning/`
  moved direct hand-authored RVV `i32-vadd` planning/export toward typed body
  authority and quarantined descriptor-only direct `i32-vadd`.
* Current specs keep `tcrv.exec` compute-free, place RVV semantics in
  extension family ops/target/plugin code, and define the main lowering route
  as extension family ops -> common EmitC -> C/C++/intrinsic/runtime ABI.
* At task start, current code still had a no-evidence default family resolution surface:
  `resolveRVVBinaryFamilyForProposal` returns the registered
  `i32-vadd` family with source kind `default-i32-vadd` when no frontend
  lowering, typed body, or descriptor candidate exists.
* Current selected lowering-boundary code already has a descriptorless default
  materialization path keyed by selected `tcrv_rvv.element_count`; this round
  should make that default request explicit as typed-body materialization
  authority and strengthen focused tests around the default route.

## Requirements

* The default RVV `i32-vadd` route with no frontend lowering attribute, no
  hand-authored `tcrv_rvv.lowering_descriptor`, and no preexisting typed body
  must materialize a typed `tcrv_rvv.i32_vadd_microkernel` body before selected
  emission/export can succeed.
* The default route source kind must not read as descriptor-family compute
  authority. If a source-kind name remains, it must mean "default typed body
  materialization requested".
* The materialized typed body must carry selected variant, role, required
  capabilities, required march, selected vector-shape/config metadata,
  `setvl`/`with_vl`, lhs/rhs load ops, `i32_add`, store op, ABI buffer roles,
  and descriptor-local `element_count` layering expected by the existing
  typed-body verifier/export path.
* Selected emission and target export for this default path must fail closed
  if the typed body is missing, stale, mismatched with the selected variant,
  mismatched with selected vector shape/config, has wrong arithmetic/dataflow,
  has missing/swapped ABI roles, or omits required march/capability metadata.
* Descriptor-only default `i32-vadd` compute authority must remain removed or
  explicitly quarantined. Tests must not preserve descriptor-driven default
  compute behavior as production behavior.
* Keep this bounded to the RVV default `i32-vadd` path. No new arithmetic
  family, dtype, backend claim, benchmark, runtime claim, or generic RVV route
  is in scope.
* Python remains tooling-only; compiler core behavior stays in
  C++/MLIR/TableGen/CMake/lit/FileCheck.

## Acceptance Criteria

* [x] Default RVV `i32-vadd` planning no longer exposes the old
      `default-i32-vadd` meaning; diagnostics/tests identify it as a default
      typed-body materialization request.
* [x] A focused default `i32-vadd` execution-planning route proves the selected
      path materializes `tcrv_rvv.i32_vadd_microkernel` without a frontend
      lowering attr, hand-authored descriptor, or preexisting typed body.
* [x] The same route proves the generated body has `setvl`, `with_vl`,
      lhs/rhs `i32_load`, `i32_add`, `i32_store`, selected vector-shape/config
      metadata, required march, required capabilities, ABI buffer roles, and
      element-count metadata.
* [x] Selected emission/export rejects descriptor-only or missing-body default
      `i32-vadd` routes instead of using descriptor metadata as compute truth.
* [x] Negative coverage exists for stale/mismatched default typed bodies:
      wrong selected variant, wrong arithmetic op, selected vector-shape/config
      mismatch, missing/swapped ABI roles, and missing required march or
      capability metadata where applicable.
* [x] A focused lit/FileCheck route proves the default `i32-vadd` selected
      export reaches the common EmitC-backed generated artifact path through
      typed body authority and emits no descriptor-only compute route claim.
* [x] `tcrv.exec` remains compute-free; RVV semantics remain in RVV extension
      family ops and target/plugin code.

## Out Of Scope

* New RVV families, dtypes, LMULs beyond the existing selected config slice,
  benchmarks, performance claims, runtime correctness claims, or broad matrix
  runs.
* Descriptor-to-C exporters, descriptor-owned compute semantics, or direct
  descriptor-to-C default `i32-vadd` export.
* MLIR vector/scalable-vector lowering.
* Compute semantics in `tcrv.exec`.
* IME, AME, Sophgo/offload, Template, Toy, scalar, or unrelated
  descriptor-exit work.
* `ssh rvv` evidence; this task is a structural compiler migration and makes
  no fresh hardware runtime/correctness/performance claim.

## Minimal Validation

* Run `git diff --check`.
* Build and run focused targets if affected:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`, and
  `tianchenrv-target-artifact-export-test`.
* Run focused lit/FileCheck coverage for the default RVV `i32-vadd`
  execution-planning/export route and its negative cases.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and local tooling/time allows.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-descriptor-exit-rvv-default-vadd-body-materialization`
  before finish/archive, and validate the archived task path if completed.

## Technical Notes

* `.trellis/spec/architecture/unified-riscv-mlir.md` defines descriptor-driven
  computation as implementation debt and keeps `tcrv.exec` compute-free.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the correct route as
  typed extension op -> generated lowerable interface -> target/plugin mapping
  -> common EmitC route.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  selected vector shape and runtime ABI layering to remain explicit and
  fail-closed before target artifacts.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines plugin materialized
  `tcrv_rvv.i32_vadd_microkernel` body shape from finite selected descriptor
  metadata and requires body/export validation before a supported emission
  path is claimed.
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` requires
  plugin-proposed variants and extension-family ops rather than generic core
  compute.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused C++ and
  lit/FileCheck coverage for RVV typed body materialization/export and
  fail-closed mismatch diagnostics.

## Round Result

Completed in this round:

* Renamed the no-evidence default RVV `i32-vadd` source kind from
  `default-i32-vadd` to `default-i32-vadd-typed-body-materialization`, making
  the default family resolution explicitly mean "typed body materialization
  requested" rather than descriptor-family compute authority.
* Kept descriptor attachment disabled for descriptorless typed-family default
  routes unless the source is explicitly descriptor-owned.
* Added focused C++ planning coverage proving the default no-body route resolves
  as a typed-body materialization request, selects `i32m1`, preserves explicit
  capability requirements, and does not reattach `tcrv_rvv.lowering_descriptor`.
* Added focused C++ selected lowering-boundary coverage proving a descriptorless
  default `i32-vadd` selected path materializes
  `tcrv_rvv.i32_vadd_microkernel` with required march, selected mabi,
  selected vector-shape metadata, `setvl`/`with_vl`, load/add/store typed RVV
  body ops, ABI buffer roles, and callable runtime ABI boundaries before
  emission/export.
* Reused existing focused lit/FileCheck coverage for default no-body
  auto-materialization/export, descriptor-only quarantine, missing-body export
  rejection, stale/mismatched body rejection, and common EmitC-backed RVV
  artifact output.

Validation completed:

* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* Focused FileCheck commands for
  `test/Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir`
  using `IR`, `READY`, and `EXPORT` prefixes.
* Focused FileCheck command for
  `test/Transforms/LoweringBoundary/rvv-i32-vadd-descriptor-only-legacy-quarantine.mlir`.
* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 206/206 lit tests.

No `ssh rvv` command was run or claimed. This round is a structural compiler
migration and makes no fresh RVV hardware runtime, correctness, or performance
claim.
