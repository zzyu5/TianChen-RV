# RVV callable artifact bundle with ssh rvv correctness evidence

## Goal

For the existing bounded RVV i32m1 add slice, expose a runtime-callable artifact surface that pairs the generated RISC-V object with the declared C ABI, then link and run a minimal correctness harness on the `ssh rvv` machine.

The executable path must remain:

```text
TianChen-RV MLIR
  -> selected RVV variant
  -> explicit tcrv_rvv ops
  -> EmitC
  -> MLIR C/C++ emitter
  -> RVV relocatable object
  -> runtime ABI header/bundle
  -> ssh rvv link/run correctness evidence
```

## What I already know

- Current HEAD `dc96794` made the bounded RVV i32m1 add path export a real `riscv-elf-relocatable-object`.
- The existing RVV object route is RVV-owned in `RVVTargetSupportBundle` and shares the same bounded route construction used by plugin emission planning.
- The current emission plan already records the runtime ABI kind/name/glue role and ordered ABI parameters:
  `lhs: const int32_t *`, `rhs: const int32_t *`, `out: int32_t *`, `n: size_t`.
- `TargetArtifactExport` already has generic artifact bundle machinery and recognizes `runtime-callable-c-header` plus `riscv-elf-relocatable-object` as materialized artifact kinds.
- Existing `test/Target/RVV/i32m1-add-object-artifact.mlir` proves local object export and visible ABI metadata, while `i32m1-object-unsupported-shape.mlir` proves fail-closed unsupported shape behavior.
- `.trellis/spec/` requires RVV runtime/correctness claims to use real `ssh rvv` evidence, not local compile-only or dry-run evidence.

## Requirements

1. Add a non-semantic runtime-callable header or equivalent bundle component for the bounded RVV i32m1 add route.
2. The header/bundle surface must expose the declared ABI name and ordered parameters `lhs`, `rhs`, `out`, and `n`.
3. The generated object body must still come only from explicit `tcrv_rvv` ops through the current EmitC and MLIR C/C++ emitter path.
4. The bundle/front-door metadata must pair the header and object as one runtime-callable external ABI component group when that is the chosen route.
5. Unsupported shapes such as i32m1 sub or otherwise unsupported RVV bodies must still fail closed before artifact bytes are emitted.
6. Run a minimal external C/C++ harness on `ssh rvv` that links against the generated object and validates at least one non-empty integer-vector add through the declared ABI.
7. Capture concise non-secret evidence: compiler/toolchain used, link command, run command, exit status, and expected-vs-actual vector result.
8. Perform a changed-surface scan showing descriptor/direct-C/source-export legacy authority was not restored.

## Acceptance Criteria

- [x] A selected RVV i32m1 add MLIR fixture can produce linkable artifact(s) for the callable ABI.
- [x] The callable ABI surface visibly exposes `lhs`, `rhs`, `out`, and `n` in order.
- [x] Bundle/index or equivalent packaging records a coherent header/object component group for the external ABI.
- [x] Focused lit/FileCheck or C++ coverage proves the artifact/header/bundle surface and preserves the object route behavior.
- [x] Existing unsupported RVV shape coverage remains fail-closed.
- [x] Real `ssh rvv` link/run evidence validates a non-empty integer vector add through the generated callable ABI, or the exact remote/toolchain blocker is recorded without claiming runtime correctness.
- [x] No descriptor-driven computation, direct semantic C/source exporter, source-export production route, Python compiler core, GCC-default route, compatibility wrapper, new dtype/LMUL/op family, or core RVV semantic branch is introduced.

## Out of Scope

- New RVV dtypes, LMULs, arithmetic families, generic RVV lowering, MLIR vector lowering, frontend lowering, performance benchmarks, matrix runs, broad runtime integration, descriptor/binary-family registries, direct C compute printers, or Python compiler-core logic.
- Claiming runtime performance or generic TianChen-RV/RVV correctness beyond this bounded i32m1 add external ABI run.
- Replacing clang/LLVM as the default native compiler route; GCC may only remain a compatibility detail if already present.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant current source/tests inspected:
  - `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `include/TianChenRV/Target/TargetArtifactExport.h`
  - `lib/Target/TargetArtifactExport.cpp`
  - `test/Target/RVV/i32m1-add-object-artifact.mlir`
  - `test/Target/RVV/i32m1-object-unsupported-shape.mlir`
  - `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
- Previous journal session 88 records the immediately preceding object artifact route and its local validation commands.

## Validation Plan

1. Build focused compiler/test targets touched by the RVV target support and target artifact export layers.
2. Run focused C++ tests for target artifact export and RVV plugin/route behavior.
3. Run focused lit coverage for `Target/RVV/i32m1` and any new bundle/header test.
4. Run `check-tianchenrv` if practical after focused checks pass.
5. Export the bounded bundle/artifacts locally, copy only the required generated files and external harness to `ssh rvv`, compile/link/run, and record concise evidence under `artifacts/tmp/`.
6. Run a changed-surface scan for descriptor/direct-C/source-export residue in touched files.
