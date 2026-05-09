# RVV i32 binary family descriptor registry

## Goal

Create a bounded C++ descriptor registry for the finite `i32` binary
microkernel families that already exist in the RVV/scalar target surface:
`i32-vadd` and `i32-vsub`. The registry centralizes stable family metadata
that is currently repeated across active compiler/export consumers before any
new arithmetic family or deeper RVV maturity slice is added.

## Why Now

The previous vsub dispatch runtime-evidence module finished and archived at
commit `d571d2c`. The next bottleneck is no longer vsub correctness evidence;
it is the duplicated family metadata that had to be edited across route ids,
ABI names, marker names, function stems, intrinsics, scalar operators,
dispatch expectations, and script/test surfaces.

## Requirements

- Add a C++ descriptor registry or equivalent target/plugin-owned interface
  covering exactly the existing finite families:
  - `i32-vadd`
  - `i32-vsub`
- The descriptor must record the stable values needed by active compiler/export
  surfaces, including family id, frontend lowering value, lowering descriptor,
  operation names, function/header stems, RVV intrinsic identity, scalar C
  operator, route ids, runtime ABI kind/name/id, runtime glue role, artifact
  kinds, dispatch route ids, self-check marker, and family mismatch identity.
- Migrate `lib/Target/Builtin/RVVScalarDispatch.cpp` to consume descriptor-backed
  values for dispatch family matching, route metadata, ABI metadata, function
  stems, header guard stems, scalar operator, and self-check markers.
- Migrate at least one additional active C++ consumer. This round targets
  `lib/Target/RVV/RVVMicrokernel.cpp`; `lib/Target/Scalar/ScalarMicrokernel.cpp`
  may also be migrated if it stays low-risk.
- Keep externally visible behavior stable: existing route names, ABI names,
  generated artifact names, vadd/vsub arithmetic semantics, lit behavior, and
  script behavior must continue unless a focused compatibility issue is found.
- Preserve stale-family mismatch rejection: vsub must not silently reuse vadd
  route, ABI, marker, intrinsic, or scalar arithmetic metadata.
- Add or update focused C++/lit tests proving both families are present in the
  registry and migrated consumers continue to expose descriptor-equivalent
  values.

## Staged Boundary

This round does not have to migrate every active consumer. Plugin proposal and
legality code may keep plugin-local structs where those structs still carry
plugin-only supported messages, materialization wording, or op-class checks.
The registry must still become active in at least two real C++ compiler/export
consumers before this task can finish.

## Non-goals

- No new arithmetic family beyond `i32-vadd` and `i32-vsub`.
- No `i64`/`e64`, masks, new RVV policies, dynamic-shape frontend expansion,
  StableHLO/TOSA lowering, performance tuning, or generic RVV lowering.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branches in generic core passes.
- No Python implementation of compiler registry, lowering, emission, route
  selection, runtime ABI decisions, or source generation.
- No docs-only, helper-only, report-only, smoke-only, or metadata-only closeout.
- No new remote runtime correctness claim unless a fresh focused `ssh rvv` run
  is explicitly executed and recorded.

## Acceptance Criteria

- [x] Descriptor registry exposes exactly `i32-vadd` and `i32-vsub` and tests
      verify both descriptors.
- [x] `RVVScalarDispatch.cpp` consumes descriptor-backed family values.
- [x] At least one additional active C++ consumer consumes descriptor-backed
      family values.
- [x] Existing dispatch and bundle e2e tests for add/sub remain passing.
- [x] Existing stale-family mismatch rejection remains covered.
- [x] If generated source text changes, the change is descriptor-equivalent and
      the affected focused script/lit path is rerun.
- [x] `git diff --check` passes.
- [x] CMake configure with repository LLVM/MLIR paths passes.
- [x] Focused descriptor/dispatch/translate tests pass.
- [x] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes or any blocker is recorded truthfully.
- [x] If complete, the Trellis task validates, finishes/archives, journal is
      updated, and one coherent commit is created.

## Completion Notes

- Created `include/TianChenRV/Target/I32BinaryFamilyRegistry.h` as the bounded
  C++ descriptor registry for `i32-vadd` and `i32-vsub`.
- Migrated active target/export consumers:
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
- Left plugin proposal/materialization local family structs as the staged
  boundary because they carry plugin-only supported-message and materialization
  wording. The next optional migration point is
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, then
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
- No route names, ABI names, generated artifact names, vadd/vsub arithmetic
  semantics, or external behavior were intentionally changed.
- No new `ssh rvv` runtime correctness claim was made; this round changed
  descriptor ownership and ran local compiler/export validation only.

## Validation

- `git diff --check`
- `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='I32BinaryFamilyRegistry|rvv-scalar-i32-vadd-dispatch-generic-route|rvv-scalar-i32-vsub-dispatch-generic-route|plan-linalg-i32-vadd-and-export-target-artifact-bundle|plan-linalg-i32-vsub-and-export-target-artifact-bundle|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Technical Notes

- Relevant specs:
  - `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Prior module PRD:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-scalar-vsub-dispatch-ssh-rvv-runtime-evidence/prd.md`
- Primary source surfaces:
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`

## Continuation Rule If Unfinished

Keep this Trellis task open. Record which active consumers were migrated, which
duplicate family tables remain, which tests pass, and the exact next consumer
to migrate. Do not archive or claim the registry is the source of truth until
at least two active consumers are descriptor-backed and checked.
