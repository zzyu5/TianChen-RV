# Remove descriptor authority from scalar fallback selected emission

## Goal

Complete the scalar fallback side of selected-emission descriptor exit for the
bounded finite i32/i64 add/sub/mul families already represented by typed
`tcrv_scalar` microkernel ops. Production scalar selected emission and
readiness must derive selected family, element count, route/runtime ABI
metadata, required capabilities, source/header/object route metadata, and
common EmitC source provenance from the matched typed scalar microkernel op plus
the exec-IR-backed callable ABI boundary. `tcrv_scalar.lowering_descriptor`
must no longer reconstruct or select scalar fallback compute semantics for
production selected emission/readiness.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `3efcf1c feat(rvv): remove descriptor selected emission authority`.
- Worktree was clean and `.trellis/.current-task` was absent at task start.
- The previous task
  `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/`
  completed and archived the RVV selected-emission descriptor exit.
- The scalar fallback plugin already proposes descriptorless typed-source
  variants for bounded i32/i64 add/sub/mul frontend-lowering families.
- `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` still contains descriptor
  production authority in `buildScalarMicrokernelMaterializationPlan`,
  `lookupScalarMicrokernelFamilyByDescriptor`, and
  `findMatchingExplicitMicrokernelFamily`: descriptor metadata can select or
  constrain the scalar microkernel family and descriptor-local element count
  before/while emission readiness and emission plans are built.
- `lib/Target/Scalar/ScalarMicrokernel.cpp` already validates
  `tcrv_scalar.lowering_descriptor` as mirror metadata after finding the typed
  scalar microkernel record; target/export source rendering already uses the
  typed `TCRVEmitCLowerableRoute` and exec-IR-backed callable ABI plan.
- Existing lit fixtures already assert many descriptorless scalar pipeline
  outputs through `IR-NOT: tcrv_scalar.lowering_descriptor`, typed
  `tcrv_scalar.*_microkernel` ops, and typed `emitc_source_op` selected-plan
  metadata.

## Requirements

- Rewire production scalar fallback selected emission/readiness so a supported
  plan is built only from a matched typed scalar microkernel op and matching
  `tcrv_scalar.lowering_boundary`.
- Descriptor-only scalar selected emission/readiness must fail closed: no
  descriptor-derived supported status, no descriptor-derived supported emission
  plan, and no descriptor-only microkernel materialization.
- A descriptorless typed-source scalar fallback path must remain the normal
  successful path for i32/i64 add/sub/mul families.
- Stale `tcrv_scalar.lowering_descriptor` and
  `tcrv_scalar.element_count` metadata beside a typed scalar body may only be
  validated after the typed plan is identified. They must not select or alter
  family, element count, route id, artifact kind, source rendering, header or
  object helper choice, runtime ABI identity, or callable ABI parameter plan.
- Supported scalar emission plans must continue to use typed source metadata:
  `tcrv_scalar.emitc_source_op`,
  `tcrv_scalar.emitc_lowerable_op_interface`, and runtime element-count C name
  metadata from the exec-IR callable ABI boundary.
- RVV+scalar dispatch export must consume the scalar fallback component from
  typed scalar ops/common EmitC route metadata, not from scalar descriptor
  authority.
- Keep scalar fallback plugin/target-owned behind extension family interfaces;
  do not add scalar semantic branches to core orchestration.

## Non-Goals

- No new scalar family matrix, RVV family, dtype expansion, performance
  benchmark, runtime scheduler, generic scalar backend, or generic RVV backend.
- No Python implementation of compiler internals.
- No compute semantics in `tcrv.exec`.
- No descriptor-to-C export or new route around extension family ops to common
  EmitC to C/C++ artifacts.
- No IME, AME, Sophgo/offload, Template, Toy, or unrelated RVV changes except
  dispatch/export tests required by this scalar fallback migration.
- No RVV runtime, correctness, or performance claim unless a fresh `ssh rvv`
  run is actually performed.

## Acceptance Criteria

- [x] Descriptorless typed-source scalar selected emission/readiness succeeds
  for bounded i32 add/sub/mul and i64 add/sub/mul families covered by the
  production path.
- [x] Descriptor-only scalar selected emission/readiness fails closed instead
  of producing a descriptor-derived production plan.
- [x] Stale descriptor metadata beside a typed scalar body is rejected as
  mirror mismatch or ignored as non-authoritative mirror data; it cannot alter
  the typed selected plan.
- [x] Scalar supported emission plans continue to carry typed source/common
  EmitC route metadata and exec-IR-backed runtime ABI parameters.
- [x] RVV+scalar dispatch/export tests prove scalar fallback component
  authority comes from typed scalar ops/common EmitC route and rejects stale
  scalar descriptor authority.
- [x] Focused C++ plugin and target/export tests pass.
- [x] Focused lit dispatch/export dry-run tests pass if affected.
- [x] `git diff --check` passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passes if the existing build tree is usable.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Previous task PRD read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/prd.md`
- Main implementation entry points:
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `test/Plugin/ScalarExtensionPluginTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`

## Task Status Notes

- Planning started from a clean worktree at `3efcf1c`.
- Created task and PRD before source edits.
- Rewired `ScalarExtensionPlugin` selected emission/readiness so supported
  scalar fallback plans are derived from typed `tcrv_scalar` microkernel ops,
  common EmitC route metadata, and exec-IR callable ABI metadata.
- Quarantined `tcrv_scalar.lowering_descriptor` and
  `tcrv_scalar.element_count` as optional legacy mirror metadata after typed
  plan construction; descriptor-only selected emission now fails closed as
  metadata-only and cannot materialize descriptor-derived scalar microkernels.
- Updated scalar plugin and target/export C++ tests for descriptorless
  typed-source success, descriptor-only fail-closed behavior, and stale mirror
  descriptor rejection in the RVV+scalar dispatch/export path.
- Updated affected scalar target/export and lowering-boundary lit fixtures to
  use descriptorless typed materialization and the new mirror metadata
  diagnostics.
- Validation passed:
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-runtime-abi-callable-plan-test`, focused scalar dispatch/export
  lit tests, `git diff --check`, Trellis task validation, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  with 207/207 passing.
- No `ssh rvv` run was performed, so this task makes no RVV runtime,
  correctness, or performance claim.
