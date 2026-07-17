# RVV Generated Artifact Runtime ABI Closure

## Goal

Close the runtime ABI boundary for generated RVV `i32-vadd` and `i32-vsub`
artifacts so compiler-produced source, header, object, and bundle outputs can
be invoked through the selected runtime-length and scalar-dispatch contract on
real RVV hardware evidence.

## Why Now

The previous committed slice at `ab4f803` closed selected EmitC artifact
consumers for `TargetArtifactExport`, `RVVMicrokernel`, `RVVScalarDispatch`,
and focused export tests. The next bottleneck is not another artifact-format
refinement: the selected vadd/vsub route now needs to prove that generated
artifact signatures and dispatch/e2e invocation preserve selected runtime
length roles, selected config, and op-family identity into an executable
compile/run path.

## Scope

- Existing selected RVV `i32-vadd` and `i32-vsub` routes only.
- Runtime ABI signatures for generated source/header/object/bundle artifacts.
- Runtime length role preservation from `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param` through artifact export.
- Scalar dispatch or e2e harness invocation of compiler-generated artifacts,
  not descriptor-only or hand-written surrogate kernels.
- Focused `ssh rvv` compile/run evidence for the generated artifact invocation
  path, or an exact non-code blocker after local generation and compile
  succeed.

## Non-Goals

- No new dtype, i64 expansion, LMUL matrix, third operation, broad family
  matrix, broad smoke suite, or performance tuning.
- No artifact-format-only, metadata-only, FileCheck-only, negative-test-only,
  report-only, workspace-journal, or helper-only milestone.
- No descriptor-to-C production exporter and no descriptor element count or
  descriptor vector shape as authoritative compute/config/runtime control.
- No hand-written surrogate kernels as proof of compiler output.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No GCC/vendor compiler default route.
- No Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin work
  except narrow regressions caused by shared validation.
- No runtime, correctness, or performance claims beyond focused evidence
  actually run.

## Requirements

- Generated headers and source expose a stable ABI carrying lhs/rhs/out buffers
  and selected runtime length roles for `i32-vadd` and `i32-vsub`.
- The generated ABI consistently preserves selected op-family identity,
  selected SEW/LMUL/tail/mask config, runtime AVL/VL authority, and
  clang/LLVM-compatible RVV intrinsics through `riscv_vector.h`.
- Scalar dispatch or the e2e harness invokes the generated artifact path
  rather than a descriptor-only, metadata-only, or hand-written surrogate path.
- Source/header/object/bundle routes match the same validated callable
  candidate and fail closed on missing or stale ABI/runtime-role data.
- Descriptor-only compute/config/runtime authority remains quarantined.
- Core `tcrv.exec` and generic artifact/export orchestration remain
  target-neutral.

## Acceptance Criteria

- [x] vadd and vsub generated source/header prototypes preserve the same
      ordered runtime ABI parameter contract for lhs, rhs, out, and selected
      runtime element count.
- [x] Generated artifact source/header/object/bundle routes consume the same
      selected runtime length and selected config contracts before emission.
- [x] Scalar dispatch or the e2e harness calls generated vadd/vsub artifacts
      through the exported ABI rather than using a descriptor-only or
      hand-written surrogate proof.
- [x] Missing runtime length role data, stale ABI signatures, vadd/vsub ABI
      mismatch, missing EmitC body mapping, stale op identity, missing or
      conflicting selected config, and descriptor-only production attempts fail
      before source/header/object/bundle output or dispatch/e2e success.
- [x] Generated C remains clang/LLVM-compatible and includes RVV C intrinsics
      through `riscv_vector.h`.
- [x] Focused vadd/vsub TargetArtifactExport, TargetArtifactBundleExport,
      RVVMicrokernel, RVVScalarDispatch, object, selected-quarantine,
      runtime-length/config consumption, and `rvv-microkernel-bundle-e2e.test`
      regressions pass or exact blockers are recorded.
- [x] Focused `ssh rvv` compile/run evidence is produced for generated vadd
      and vsub artifact invocation, or an exact remote/toolchain blocker is
      reported after local artifact generation and compile succeed.
- [x] Bounded ABI/descriptor reference scan over touched RVV target/export and
      scalar dispatch files shows descriptor-only authority remains quarantined
      and generic core passes gained no RVV semantic branches.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean worktree, and one coherent
      commit complete the round if finished.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/architecture/design-boundaries.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-14-rvv-emitc-artifact-consumer-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-14-rvv-emitc-body-emission-production-route/prd.md`

Likely implementation surface:

- `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
- `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
- `lib/Target/RVV/RVVMicrokernel.cpp`
- `lib/Target/RVV/RVVScalarDispatch.cpp`
- `lib/Target/TargetArtifactExport.cpp`
- `scripts/rvv_microkernel_e2e.py`
- `test/Scripts/rvv-microkernel-bundle-e2e.test`
- vadd/vsub TargetArtifactExport and TargetArtifactBundleExport tests
- RVVMicrokernel and RVVScalarDispatch tests for vadd/vsub

## Evidence Plan

- Inspect selected runtime-length/config contract headers and current
  microkernel, dispatch, artifact-export, e2e, and vadd/vsub tests before
  source edits.
- Run focused C++/TableGen build for touched RVV plugin/target/export tools,
  including generated headers, RVV target/plugin libraries,
  `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, and `tcrv-translate`
  where affected.
- Run focused lit/FileCheck for vadd/vsub generated artifact signatures,
  bundle records, dispatch/e2e invocation, fail-closed ABI/runtime-role cases,
  selected quarantine, runtime-length consumption, and selected-config
  consumption.
- Run focused `ssh rvv` compile/run evidence through the generated artifact
  path, or record an exact blocker after local generation and compile.
- Run bounded ref-scans and diff whitespace checks before finish/archive.

## Definition Of Done

The finite selected `i32-vadd`/`i32-vsub` generated artifact path has a real
runtime ABI closure: compiler-produced source/header/object/bundle artifacts
carry the selected runtime-length/config/op-family contract, scalar dispatch or
e2e invokes generated artifacts through that ABI, descriptor-only authority
remains quarantined, focused checks and RVV evidence or an exact blocker are
recorded, the Trellis task is archived, and one coherent commit records the
round.

## Implementation Summary

- `scripts/rvv_microkernel_e2e.py` now constructs
  `rvv_microkernel_external_caller.c` in direct helper dry-run mode from the
  generated header function symbol and compiler-emitted runtime ABI signature.
  The same generated caller remains the non-dry-run `ssh rvv` caller used to
  compile, link, and run against the generated source-built object and the
  generated object artifact.
- Direct vadd/vsub evidence now records the generated external caller path,
  hash, runtime ABI signature, runtime counts, success marker, and
  `source_only` dry-run flag in `evidence.json`.
- `test/Scripts/rvv-microkernel-e2e.test` now checks direct vadd and vsub
  generated caller files, caller arithmetic, runtime counts, success markers,
  evidence hashes, and dry-run claim scope.

## Validation Summary

- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --run-id codex-abi-vadd-dry --overwrite --input test/Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub --run-id codex-abi-vsub-dry --overwrite`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-e2e\\.test'` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-bundle-e2e\\.test'` from `build/test`
- `python3 scripts/rvv_microkernel_e2e.py --run-id codex-abi-vadd-ssh --overwrite --input test/Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir --timeout 120`
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --run-id codex-abi-vsub-ssh --overwrite --timeout 120`
- `git diff --check`
- `git diff --cached --check`

Both `ssh rvv` runs returned `status: success`, `ssh_evidence: true`, and
runtime counts `7,16`. The vadd run observed marker
`tcrv_rvv_microkernel_external_abi_ok`; the vsub run observed marker
`tcrv_rvv_i32_vsub_microkernel_external_abi_ok`. In both runs the generated
caller compiled on RVV, the generated source compiled to an object, the caller
linked and ran against that source-built object, then linked and ran against
the generated object artifact.

## Artifact Evidence

- vadd evidence: `artifacts/tmp/rvv_microkernel_e2e/codex-abi-vadd-ssh/evidence.json`
- vsub evidence: `artifacts/tmp/rvv_microkernel_e2e/codex-abi-vsub-ssh/evidence.json`
