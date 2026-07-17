# Remove descriptor authority from RVV selected emission planning

## Goal

Complete the descriptor exit for RVV finite binary selected emission planning.
For typed-source RVV i32/i64 add/sub/mul paths, selected emission/readiness must
derive the production selected plan from typed `tcrv_rvv` microkernel bodies,
selected vector config metadata/capabilities, and the exec-IR callable ABI
boundary. `tcrv_rvv.lowering_descriptor` must no longer reconstruct or select
the production RVV selected emission plan.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `02adee1 feat(abi): derive i64 callable params from exec IR`.
- Worktree was clean and `.trellis/.current-task` was absent at task start.
- The previous task
  `.trellis/tasks/archive/2026-05/05-12-i64-binary-ir-backed-callable-abi-production-authority/`
  completed and archived the i64 exec-IR-backed callable ABI migration with
  full `check-tianchenrv` passing.
- The prior i32 task likewise completed exec-IR-backed callable ABI authority
  for bounded i32 RVV/scalar callable paths.
- `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` still calls
  `buildDescriptorSelectedPlanForEmission` in both
  `findI32SelectedEmissionAttachment` and `findI64SelectedEmissionAttachment`.
- `buildDescriptorSelectedPlanForEmission` parses
  `tcrv_rvv.lowering_descriptor` through `buildRVVBinarySelectedPlanFromVariant`
  and can reconstruct selected family, vector shape, element count, and
  descriptor route facts before the explicit typed microkernel body has been
  accepted.
- Existing typed microkernel bodies and selected config contracts already carry
  enough bounded information to identify finite i32/i64 add/sub/mul family,
  selected vector shape, selected ABI names, and common EmitC route source
  provenance.

## Requirements

- Stop using `buildDescriptorSelectedPlanForEmission` as production authority
  for typed-source i32/i64 selected emission/readiness.
- `findI32SelectedEmissionAttachment` and `findI64SelectedEmissionAttachment`
  must construct `RVVBinarySelectedPlan` from the explicit typed RVV
  microkernel body plus selected vector config/capability metadata.
- A typed-body path with no `tcrv_rvv.lowering_descriptor` must remain the
  normal/default successful production path.
- A descriptor-only typed-source selected variant must fail closed by producing
  no selected emission/readiness plan or a bounded quarantine diagnostic; it
  must not recover selected plan facts from the descriptor.
- If stale `tcrv_rvv.lowering_descriptor` metadata exists beside a typed body,
  it may only be validated as legacy mirror metadata after the typed selected
  plan is built. It must not change selected family, vector shape, element
  count, emitted source/header/object route, or callable ABI.
- Runtime callable parameters must continue to come from the shared
  exec-IR-backed finite binary callable ABI plan over `tcrv.exec.mem_window`
  and `tcrv.exec.runtime_param`.
- RVV generated source/header/object candidates must continue to come through
  typed RVV family ops and the common EmitC route, not descriptor-to-C export.
- Keep route/family descriptor structs only as bounded route metadata,
  intrinsic spelling metadata, selected metadata mirror, and compatibility
  lookup tables where unavoidable.

## Non-Goals

- Do not implement a new RVV family matrix, generic RVV lowering, performance
  tuning, runtime scheduler, or new dtype family.
- Do not add compiler core, dialects, passes, plugin registry, capability
  model, lowering, or emission logic in Python.
- Do not add compute semantics to `tcrv.exec`.
- Do not route around extension family ops to common EmitC materialization and
  generated C/C++.
- Do not claim RVV runtime correctness, runtime success, throughput, latency,
  or performance without fresh `ssh rvv` evidence.

## Acceptance Criteria

- [x] Production typed-source i32 selected emission/readiness succeeds without
  `tcrv_rvv.lowering_descriptor`.
- [x] Production typed-source i64 selected emission/readiness succeeds without
  `tcrv_rvv.lowering_descriptor`.
- [x] Descriptor-only typed-source i32 selected emission/readiness fails closed
  instead of producing a descriptor-derived selected emission plan.
- [x] Descriptor-only typed-source i64 selected emission/readiness fails closed
  instead of producing a descriptor-derived selected emission plan.
- [x] Stale descriptor metadata beside a typed body is rejected as a mirror
  mismatch or ignored as non-authoritative mirror data; it cannot alter the
  typed-body selected plan.
- [x] Focused C++ tests cover the typed-body no-descriptor success cases and
  descriptor-only/stale descriptor fail-closed cases.
- [x] Focused target/plugin tests still prove generated RVV source/header/export
  candidates use the common EmitC route and exec-IR-backed callable ABI plan.
- [x] `git diff --check` passes.
- [x] Focused RVV selected emission planning, selected lowering-boundary,
  RVV microkernel target export, and runtime ABI callable plan checks pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  runs if the existing build directory is usable.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant archived PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-i64-binary-ir-backed-callable-abi-production-authority/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-i32-binary-ir-backed-callable-abi-production-authority/prd.md`
- Main source entry points:
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`
  - `include/TianChenRV/Support/RuntimeABICallablePlan.h`
  - `lib/Support/RuntimeABICallablePlan.cpp`
  - `test/Plugin/RVVBinaryPlanningTest.cpp`
  - `test/Plugin/RVVBinarySelectedLoweringBoundaryTest.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Target/RVVMicrokernel/`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Support/RuntimeABICallablePlanTest.cpp`

## Task Status Notes

- Planning started from a clean worktree at `02adee1`.
- Created task context and PRD before source edits.
- Removed `buildDescriptorSelectedPlanForEmission` and the typed-source
  selected emission/readiness calls to descriptor-selected plan reconstruction.
- `findI32SelectedEmissionAttachment` and `findI64SelectedEmissionAttachment`
  now construct the selected emission plan from the matched typed RVV
  microkernel op, selected vector config/capability metadata, typed body
  control/dataflow, microkernel `element_count`, and exec-IR-backed callable
  ABI plan.
- Stale `tcrv_rvv.lowering_descriptor` and stale variant
  `tcrv_rvv.element_count` are checked only after the typed selected plan is
  built, as legacy mirror metadata. They cannot choose the family, vector
  shape, route, source/header/object path, or callable ABI.
- Descriptor-only i32/i64 selected emission/readiness now fails closed with no
  selected plan/status, and descriptor-only proposal/lowering-boundary
  quarantine remains in place.
- No `.trellis/spec/` update was needed: the existing architecture,
  RVV-plugin, EmitC-route, emission-runtime, and testing specs already encode
  the descriptor-boundary rule this task implemented.
- No `ssh rvv` run was performed and no runtime correctness/performance claim
  was made.
- Focused C++ checks, focused lit checks, `git diff --check`, and full
  `check-tianchenrv` passed.
