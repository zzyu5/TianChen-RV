# RVV selected-body executable artifact gate

## Goal

Move the current bounded RVV selected-body path from route-supported evidence
toward executable evidence. One explicit typed `tcrv_rvv` selected body must
flow through the RVV provider-built `TCRVEmitCLowerableRoute`, common EmitC
materialization, target artifact export, and real `ssh rvv` compile/run
evidence before this task claims runtime or correctness.

## Background

- Commit `88f5457` completed selected-route artifact identity closure. Route
  IDs, target artifact IDs, runtime ABI labels, ABI labels, and retained i32m1
  specialization names are now mirrors of selected-body provider facts rather
  than independent RVV route authority.
- The current bottleneck is not more identity cleanup. The next proof is that
  the corrected route-supported path can produce artifacts that an external
  runtime caller can compile, link, and run on the real RVV host.
- The existing bounded evidence tooling around
  `scripts/rvv_generated_bundle_abi_e2e.py` is tooling only. It may be used to
  collect artifact and remote evidence, but it must not become compiler
  semantics or a descriptor/direct-C replacement path.

## Requirements

1. Use one existing bounded explicit typed RVV selected-body specialization as
   the representative path. Do not add broadcast, compare/select, reduction,
   conversion, new dtype, new LMUL, new source shape, or new intrinsic cases.
2. Verify or repair the active path:
   `selected tcrv.exec RVV variant with explicit typed tcrv_rvv body -> RVV
   provider-built TCRVEmitCLowerableRoute -> common EmitC materialization ->
   target artifact object/header/bundle export -> external caller compile/link
   and run on ssh rvv`.
3. Keep the selected typed `tcrv_rvv` body as the sole RVV semantic authority.
   Provider, construction, materialization, and target export may consume
   route IDs, artifact IDs, ABI labels, runtime parameter metadata, intrinsic
   spellings, and retained i32m1 names only as mirrors of validated
   selected-body facts.
4. Ensure common EmitC and generic target/export code remain neutral mechanics.
   They must not choose RVV operation, dtype, SEW/LMUL, VL placement, policy,
   memory form, intrinsic spelling, or runtime semantics.
5. If local artifact/export blockers appear, repair the owner code in the
   active path. Do not hide a blocker by weakening tests, adding status-only
   artifacts, restoring descriptor-driven C export, or adding compatibility
   wrappers.
6. Collect real `ssh rvv` compile/link/run evidence before making any
   runtime/correctness claim. If `ssh rvv` or its toolchain is unavailable,
   record the exact blocker and leave executable status unclaimed.
7. Keep evidence focused: local provider/materializer/target tests, one
   generated artifact or exact generation failure, real RVV compile/run if
   available, and bounded residue scans over touched provider/materializer/
   target/export surfaces.

## Acceptance Criteria

- [x] A current explicit typed RVV selected-body fixture produces a
      provider-built route and materialized MLIR EmitC route without deriving
      semantics from route IDs, artifact names, descriptor residue, test names,
      or common export code.
- [x] Target artifact export produces coherent object/header/bundle artifacts
      whose index ties the same selected variant, route, runtime ABI identity,
      ordered ABI parameters, materialized EmitC provenance, and RVV runtime
      AVL/VL metadata together.
- [x] The generated object/header bundle is consumed by an external harness on
      `ssh rvv`; remote compile/link/run succeeds for bounded runtime counts,
      or the exact remote/toolchain/artifact blocker is recorded with no
      executable claim.
- [x] Focused local checks cover the provider/materializer/target export path
      touched by this task.
- [x] If a stale metadata bypass is discovered while wiring the executable
      path, focused negative coverage proves it fails closed before artifact
      output or runtime claim.
- [x] Bounded residue scan over touched provider/materializer/target/export
      files shows common EmitC remains neutral and RVV semantics remain
      provider-owned.

## Non-Goals

- No new RVV coverage expansion: no broadcast expansion, compare/select
  expansion, reduction, conversion, new dtype, new LMUL, source-shape growth,
  intrinsic table growth, or Stage 2 coverage campaign.
- No Linalg, Vector, StableHLO, or high-level frontend generalization.
- No Scalar, IME, Offload, TensorExt, autotuning, dashboards, broad smoke
  matrices, report/status-only artifacts, or artifact index campaigns.
- No descriptor-driven computation, direct-C semantic exporter restoration, or
  compatibility wrapper that preserves old i32 route authority.
- No runtime, correctness, or performance claim without real `ssh rvv`
  compile/run evidence for the concrete generated artifact.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-rvv-selected-route-artifact-identity-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-18-rvv-source-bundle-ssh-runtime-abi-proof/prd.md`

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `lib/Target/TargetArtifactExport.cpp`
- `lib/Target/ConstructionTemplateArtifactAdapter.cpp`
- `tools/tcrv-translate/tcrv-translate.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused RVV EmitC/target/export tests and fixtures under
  `test/Conversion/EmitC`, `test/Target`, and `test/Plugin`.

## Validation Plan

- Inspect the current source artifact/bundle fixture and the generated bundle
  e2e runner to identify the exact representative selected-body path and
  current failure, if any.
- Build required tools/tests locally, at minimum `tcrv-opt`, `tcrv-translate`,
  and focused C++ tests for RVV provider/materializer/target export behavior
  touched by this task.
- Run focused lit coverage for the selected-body EmitC materialization and RVV
  target artifact/bundle export path touched by the change.
- Run the generated bundle evidence path locally first, then run the same
  bounded artifact evidence on `ssh rvv` if the artifact path succeeds.
- Run bounded residue scans over touched provider/materializer/target/export
  files for descriptor/direct-C/source-export residue, route-id or artifact
  metadata as authority, and family-specific semantic branches in common code.
- Run `python3 ./.trellis/scripts/task.py validate` for this task and
  `git diff --check`.

## Implementation Notes

- The active compiler path already generated a coherent selected-body
  object/header bundle, but the evidence runner still required the old
  `rvv_arithmetic_op` metadata key.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` so object/header bundle
  verification requires the current selected-body metadata mirrors:
  `rvv_selected_body_operation` and
  `rvv_selected_body_typed_compute_op`.
- Updated the runner fake-bundle self-test data and focused lit coverage to
  reject `rvv_arithmetic_op` and assert the selected-body metadata keys for
  add/sub/mul dry-run evidence.
- No `.trellis/spec/` update was needed: this task aligns evidence tooling to
  the existing selected-body artifact contract rather than introducing a new
  compiler or cross-layer contract.

## Evidence

- Local dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-selected-body-executable-artifact-gate-add-dry`.
- Real `ssh rvv` artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-selected-body-executable-artifact-gate-add`.
- Generated bundle index:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-selected-body-executable-artifact-gate-add/add/generated_bundle/tianchenrv-target-artifact-bundle.index`.
- Remote compile facts:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote run output:

```text
add case n=7 ok
add case n=16 ok
add case n=23 ok
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23
```

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-selected-body-executable-artifact-gate-add-dry --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-selected-body-executable-artifact-gate-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-source-family-dry-run.test` from `build/test`: 2/2 passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/vector-materialized-target-artifact-exporters.mlir ../test/Target/RVV/emitc-to-cpp-selected-boundary-negative.mlir ../test/Target/RVV/emitc-to-cpp-selected-boundary-attrs-negative.mlir ../test/Target/RVV/emitc-to-cpp-non-materialized.mlir` from `build/test`: 4/4 passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Conversion/EmitC/rvv-first-slice-materialization.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-missing-abi.mlir ../test/Conversion/EmitC/rvv-first-slice-config-vl-contract-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-vl-contract-negative.mlir` from `build/test`: 5/5 passed.
- Bounded residue scans:
  `rg -n "rvv_arithmetic_op" .` now finds only the lit negative
  `--implicit-check-not` guards; selected-body metadata checks are in the
  runner and script lit.
