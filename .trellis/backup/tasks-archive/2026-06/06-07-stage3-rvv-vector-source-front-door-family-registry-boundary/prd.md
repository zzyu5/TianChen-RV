# Stage3 RVV vector source-front-door family registry boundary

## Goal

Introduce or repair one production compiler-interface submodule: the RVV plugin-local vector source-front-door family registry/dispatcher. The owner boundary must make the two existing opt-in families, bounded vector binary and bounded vector compare/select, consume explicit family-owned contracts from source marker validation through selected `tcrv.exec` RVV variant construction and typed `tcrv_rvv` body materialization.

## What I already know

* Commit `d38a6762` added the bounded vector compare/select source-front-door materializer after earlier binary source-front-door work.
* The repository currently has at least two RVV vector source-front-door families in `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`: binary and compare/select.
* The task is Stage 3 source-front-door boundary work. It is not Stage 1 legacy i32m1 route-authority preservation and not Stage 2 RVV coverage expansion.
* The production path must remain: opt-in bounded Vector source IR -> plugin-owned source-front-door family contract -> selected `tcrv.exec` RVV variant -> typed `tcrv_rvv` body -> RVV provider-built `TCRVEmitCLowerableRoute` -> neutral Common EmitC -> target artifact export.
* Source markers, route ids, artifact names, test names, Common EmitC, and mirror metadata must not become semantic authority.

## Assumptions

* No new source-front-door family is needed unless a tiny registry self-check is the only reasonable way to prove the boundary.
* Existing positive behavior for binary and compare/select should be preserved rather than expanded.
* If the existing structure is already sufficiently factored, the implementation will close the exact remaining duplicated authority or fail-closed gap instead of switching directions.
* Runtime behavior should not be claimed unless touched and proven with `ssh rvv`.

## Requirements

* Add or repair a plugin-local family registry/dispatcher that owns explicit per-family contracts for:
  * source marker validation;
  * source-shape parsing;
  * selected RVV variant construction;
  * typed `tcrv_rvv` body materialization;
  * runtime ABI binding expectations;
  * pass-option exposure;
  * generated-bundle op-kind selection;
  * fail-closed diagnostics.
* Make both existing families active consumers of that owner boundary.
* Preserve focused positive materialization behavior for binary and compare/select source-front-door tests.
* Preserve generated-bundle dry-run behavior for both source-front-door families.
* Add or repair fail-closed coverage for unknown marker/family, malformed source shape, stale source marker, and legacy i32m1 source-front-door.
* Keep Common EmitC/export neutral; RVV semantics must remain plugin/provider-owned.
* Keep legacy i32m1 source-front-door only as fail-closed guardrail evidence.

## Non-Goals

* No new reduction, multiply-accumulate, gather/scatter, dtype, LMUL, or frontend source family expansion.
* No high-level Linalg, Vector, or StableHLO frontend generalization.
* No default source-front-door pipeline.
* No compatibility route for legacy i32m1 source-front-door.
* No descriptor-driven compute path.
* No docs-only or spec-only completion.
* No broad smoke matrix, dashboard, global autotuning database, or readiness state machine.

## Acceptance Criteria

* [x] Production code shows binary and compare/select source-front-door families registered and dispatched through the same plugin-local family owner boundary.
* [x] Family-owned contracts explicitly cover marker validation, shape parsing, selected variant/body materialization, ABI expectations, pass option, generated-bundle entry-point evidence mirrors, and fail-closed diagnostics.
* [x] Existing positive lit coverage for binary and compare/select source materialization passes.
* [x] Focused fail-closed lit coverage proves unknown marker/family, malformed source shape, stale source marker, and legacy i32m1 source-front-door rejection.
* [x] Generated-bundle dry-run tests pass for binary and compare/select source-front-door families.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py` was not changed, so `py_compile` was not required.
* [x] Bounded old-authority scan over touched production files and added diff lines shows no new route authority from legacy i32m1/source marker/artifact metadata/Common EmitC semantics.
* [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are clean after commit.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-06/06-07-stage3-generic-vector-compare-select-source-front-door/`
* `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Transforms/RVV/rvv-vector-binary-source-front-door*.mlir`
* `test/Transforms/RVV/rvv-vector-compare-select-source-front-door*.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-vector-compare-select-source-front-door-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-fail-closed.test`
* Legacy `rvv-i32m1` source-front-door tests only as fail-closed guardrails.

## Completion Evidence

### Production Boundary

* Added the public RVV plugin-local active-family pass registration entry point:
  `registerRVVVectorSourceFrontDoorFamilyPasses(...)`.
* Introduced an active family descriptor registry in
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` for exactly:
  `bounded-vector-binary-source-front-door` and
  `bounded-vector-compare-select-source-front-door`.
* Rewired `RVVExtensionPlugin::registerSourceFrontDoorPasses(...)` so active
  bounded Vector family pass registrations come from the registry. The legacy
  i32m1 fail-closed pass remains explicit-only and outside the active family
  registry.
* Rewired marker classification through the shared dispatcher:
  missing marker -> no-op; known sibling marker -> no-op; unknown marker ->
  registry fail-closed diagnostic; matched stale `tcrv_rvv.lowering_seed` ->
  registry fail-closed diagnostic.
* Reused family descriptor fields for pass argument, pass description, selected
  variant prefix, scalar fallback symbol construction, runtime ABI purpose
  prefix, and dispatch policy mirror.
* Kept source-shape parsing and typed body materialization family-local:
  binary still owns add/sub/mul parsing and `tcrv_rvv.binary`; compare/select
  still owns predicate/select-layout parsing and `tcrv_rvv.compare/select`.
* No Common EmitC, target artifact, or provider semantic branch was added.

### Evidence Commands

* `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-vector-(binary-source-front-door|compare-select-source-front-door|source-front-door-family-registry-negative)|rvv-i32m1-vector-source-front-door|rvv-generated-bundle-abi-e2e-vector-(source-front-door|compare-select-source-front-door)"` from `build/test`
* Production old-authority scan over touched files:
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_*_i32m1`, `descriptor`, `direct-C`, `source-export`
* Added-diff-line old-authority scan over touched production files for legacy
  i32m1, descriptor/direct-C/source-export, Common EmitC, artifact-name, and
  route-id authority terms
* `git diff --check`
