# Scalar executable source-export microkernel erasure

## Goal

Delete the stale scalar dialect executable/source-export microkernel surface
that still preserves finite scalar microkernel route authority. The current
scalar fallback plugin must remain metadata-only unless a future rebuild adds a
real extension-family-to-EmitC route.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current branch is `main` and the worktree was clean before this task started.
- The previous committed round removed the active RVV executable microkernel
  wrapper path; the next residue is the scalar dialect surface.
- `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td` still defines:
  - `I32VAddMicrokernelOp`
  - `I32VSubMicrokernelOp`
  - `I32VMulMicrokernelOp`
  - `I64VAddMicrokernelOp`
  - `I64VSubMicrokernelOp`
  - `I64VMulMicrokernelOp`
  and each still attaches `TCRVEmitCLowerableOpInterface`.
- `lib/Dialect/Scalar/IR/ScalarDialect.cpp` still implements
  `verifyScalarBinaryMicrokernelOp` and per-op verify entry points for those
  six microkernel ops.
- `include/TianChenRV/Dialect/Scalar/IR/ScalarDialect.h` and the scalar
  dialect CMake files still pull in the EmitC lowerable op interface solely to
  support those ops.
- `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` remains metadata-only but still
  carries fail-closed handling for deleted `tcrv_scalar.element_count`
  selected-path metadata.
- `test/Dialect/Scalar/` still contains positive microkernel fixtures.
- `test/Plugin/ScalarExtensionPluginTest.cpp` still exercises the typed
  microkernel route and the “explicit body is inert” shape.
- `test/Target/ArtifactExport/Inputs/scalar-microkernel-source.txt` and
  `test/Target/ArtifactExport/Inputs/scalar-i64-vmul-microkernel-source.txt`
  preserve old source-export inputs.

## Assumptions

- Scalar fallback metadata-only behavior remains valid and should stay in
  place.
- The deletion target is the executable/source-export microkernel authority,
  not the generic scalar plugin identity, fallback capability metadata, or the
  `tcrv_scalar.lowering_boundary` metadata-only op itself.
- If removing the stale path breaks checks because no replacement architecture
  exists yet, that is a missing new-architecture gap to record, not something
  to patch over with compatibility code.

## Open Questions

- None. The direction brief is explicit enough to implement directly.

## Requirements

- Remove all active scalar finite microkernel dialect ops from the ODS surface.
- Remove `TCRVEmitCLowerableOpInterface` attachment from the scalar microkernel
  ops and delete the verifier helpers that exist only to validate those ops.
- Remove any scalar plugin logic that treats explicit typed scalar microkernel
  bodies as current route authority.
- Delete or rewrite tests and artifact inputs that keep
  `tcrv_scalar.*_microkernel` alive as executable source-export authority.
- Keep scalar fallback metadata-only behavior, capability metadata, and
  `tcrv_scalar.lowering_boundary` intact unless they are only reachable
  through the deleted microkernel path.

## Acceptance Criteria

- [ ] `tcrv_scalar.i32_vadd_microkernel`,
      `tcrv_scalar.i32_vsub_microkernel`,
      `tcrv_scalar.i32_vmul_microkernel`,
      `tcrv_scalar.i64_vadd_microkernel`,
      `tcrv_scalar.i64_vsub_microkernel`,
      and `tcrv_scalar.i64_vmul_microkernel` no longer exist as active
      executable/source-export dialect ops.
- [ ] No scalar dialect verifier path remains that validates those six
      microkernel ops.
- [ ] Scalar plugin code no longer depends on explicit typed scalar
      microkernel bodies as route authority.
- [ ] Focused tests in `test/Dialect/Scalar`,
      `test/Plugin/ScalarExtensionPluginTest.cpp`, and
      `test/Target/ArtifactExport` no longer assert positive support for the
      deleted microkernel route.
- [ ] Ref-scans over the touched scalar dialect/plugin/test/artifact paths no
      longer show the deleted microkernel ops as active authority.
- [ ] Focused build/test checks pass, or any remaining failures are recorded as
      expected deletion gaps caused by the missing rebuilt architecture.

## Definition of Done

- Tests added/updated for the deletion behavior.
- Focused build/lit checks run and repaired where possible.
- Task status is truthful.
- Any architecture gap exposed by the deletion is documented instead of being
  hidden by compatibility scaffolding.

## Out of Scope

- No new scalar fallback lowering.
- No new scalar loop emitter.
- No common EmitC rebuild.
- No runtime ABI path.
- No artifact route replacement.
- No compatibility mode or legacy quarantine.
- No attempt to preserve scalar microkernel ops as inert future-route IR.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant source files:
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `lib/Dialect/Scalar/IR/ScalarDialect.cpp`
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarDialect.h`
  - `lib/Dialect/Scalar/IR/CMakeLists.txt`
  - `include/TianChenRV/Dialect/Scalar/IR/CMakeLists.txt`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `test/Dialect/Scalar/`
  - `test/Plugin/ScalarExtensionPluginTest.cpp`
  - `test/Target/ArtifactExport/Inputs/scalar-microkernel-source.txt`
  - `test/Target/ArtifactExport/Inputs/scalar-i64-vmul-microkernel-source.txt`
- Focused residue patterns to remove or rewrite:
  - `tcrv_scalar.i32_vadd_microkernel`
  - `tcrv_scalar.i32_vsub_microkernel`
  - `tcrv_scalar.i32_vmul_microkernel`
  - `tcrv_scalar.i64_vadd_microkernel`
  - `tcrv_scalar.i64_vsub_microkernel`
  - `tcrv_scalar.i64_vmul_microkernel`
  - `I32VAddMicrokernelOp`
  - `I32VSubMicrokernelOp`
  - `I32VMulMicrokernelOp`
  - `I64VAddMicrokernelOp`
  - `I64VSubMicrokernelOp`
  - `I64VMulMicrokernelOp`
  - `verifyScalarBinaryMicrokernelOp`
  - `TCRVEmitCLowerableOpInterface` on scalar microkernel ops
