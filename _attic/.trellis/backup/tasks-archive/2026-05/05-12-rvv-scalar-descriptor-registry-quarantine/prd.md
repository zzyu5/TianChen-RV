# rvv-scalar-descriptor-registry-quarantine

## Goal

Quarantine the finite RVV/scalar descriptor registries after the EmitC route
migration. The old finite descriptor tables may remain as route registration,
compatibility naming, legacy fixture, and selected-plan mirror validation data,
but production/default direct RVV, direct scalar, and RVV+scalar dispatch
artifact export must not use descriptor helper tables as compute semantics,
callable ABI, source rendering, selected vector config, dispatch bundle
identity, component group, external ABI, or emitted call-name authority before
typed selected-plan authority is established.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `32b3cc2 feat(dispatch): derive composite metadata from selected plans`.
- Worktree was clean and `.trellis/.current-task` was absent at task start.
- The previous dispatch descriptor-exit task is archived at
  `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-descriptor-exit/`
  and must not be reopened.
- Previous RVV and scalar selected-emission descriptor-exit tasks are archived
  at `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/`
  and `.trellis/tasks/archive/2026-05/05-12-scalar-selected-emission-descriptor-exit/`.
- Current selected RVV/scalar emission and RVV+scalar dispatch paths already
  prefer typed extension-family ops, selected plan metadata, exec-IR-backed ABI
  boundaries, and common EmitC route metadata.
- The remaining finite descriptor headers still expose production-shaped
  helpers and fields for route ids, runtime ABI names, runtime glue roles,
  function stems, operators, scalar route lookup, dispatch route manifests, and
  direct runtime ABI parameter construction.
- Specs define TianChen-RV as a unified RISC-V MLIR where `tcrv.exec` remains
  compute-free and extension families route through extension ops -> EmitC ->
  generated C/C++ artifacts. Descriptor-driven computation is bounded
  implementation debt, not the architecture.

## Requirements

- Audit production/default consumers of finite descriptor helpers in target,
  plugin, support, and tests touched by direct RVV, direct scalar, and
  RVV+scalar dispatch export.
- Rename or narrow descriptor APIs whose names make descriptor-derived values
  look like production compute, ABI, source, or bundle authority. Remaining
  helpers must be explicitly scoped as registration, compatibility, legacy
  fixture, or selected-plan mirror validation.
- Keep finite route registration tables where the artifact exporter registry
  still needs bounded route ids and artifact kinds, but make call sites
  distinguish registration metadata from selected plan authority.
- Direct RVV export, direct scalar export, and RVV+scalar dispatch export must
  continue to fail closed when only descriptor metadata is present and no
  supported typed selected emission plan exists.
- Stale descriptor mirror metadata beside valid selected plans must not alter
  emitted source function names, intrinsic/operator selection, runtime ABI
  parameters, selected vector config, bundle identity, component group, or
  external ABI fields.
- Update or delete tests that still assert descriptor-owned compute, callable
  ABI, source rendering, or dispatch bundle authority. Legacy descriptor tests
  may remain only when their names/assertions explicitly say registration,
  compatibility, legacy fixture, or mirror validation.
- Add focused regression coverage that would fail if a production export path
  starts deriving compute/ABI/source identity from descriptor helpers again.
- Keep compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
  Python may only run existing tooling, validation scripts, or artifact probes.

## Non-Goals

- No new RVV/scalar family, dtype, LMUL, operation matrix, benchmark,
  scheduler, tuning, correctness, or performance work.
- No fresh `ssh rvv` evidence as the main deliverable; this task should make
  no new RVV runtime/correctness/performance claim.
- No descriptor-to-C exporter or route around extension family ops -> common
  EmitC -> generated C/C++ artifacts.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.
- No IME, AME, Sophgo/offload, Template, Toy, or unrelated plugin work.
- No helper-only, metadata-only, report-only, broad smoke matrix, or test-only
  closeout.

## Acceptance Criteria

- [x] Descriptor registry surfaces used by RVV/scalar/dispatch target export
  are explicitly registration/compatibility/mirror scoped, not production
  compute/ABI/source authority.
- [x] Default direct RVV source/header/object export continues to use typed
  RVV selected plans, selected vector config contracts, exec-IR ABI boundaries,
  and common EmitC route metadata for production authority.
- [x] Default direct scalar source/header/object export continues to use typed
  scalar selected plans, exec-IR ABI boundaries, and common EmitC route
  metadata for production authority.
- [x] Default RVV+scalar dispatch source/header/object/bundle export continues
  to use selected RVV/scalar component plans and IR-backed dispatch ABI
  bindings for production authority.
- [x] Descriptor-only direct RVV, direct scalar, and RVV+scalar dispatch paths
  fail closed before production artifact output.
- [x] Stale descriptor mirror metadata beside valid selected plans cannot
  alter source function names, emitted intrinsic/operator/call selection,
  runtime ABI parameters, selected vector config, component group, external
  ABI, or bundle identity.
- [x] Focused C++ tests cover descriptor registry quarantine and selected-plan
  authority for changed target/export owners.
- [x] Focused lit or dry-run script checks cover touched direct RVV/scalar and
  RVV+scalar dispatch/bundle routes.
- [x] `git diff --check` and `git diff --cached --check` pass before commit.
- [x] Trellis task validation passes on the active task path, and on the
  archived path if the task is finished.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  runs if the existing build tree remains usable after focused checks pass.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant archived PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-scalar-selected-emission-descriptor-exit/prd.md`
- Initial source audit entry points:
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/TargetArtifactExport.h`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Support/RuntimeABIContract.cpp`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Plugin/RVVBinaryPlanningTest.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Plugin/ScalarExtensionPluginTest.cpp`

## Task Status Notes

- Planning started from clean worktree at `32b3cc2`.
- Created this task and PRD before source edits.
- Refactored finite RVV, scalar, i32, and RVV+scalar registry APIs from
  descriptor-shaped lookup/getter names to registration/compatibility/legacy
  mirror names.
- Renamed selected descriptor mirror metadata roles to
  `legacy-rvv-binary-descriptor-mirror` and
  `legacy-scalar-binary-descriptor-mirror`, and kept typed source metadata as
  the default production authority.
- Quarantined direct RVV descriptor-only planning/lowering-boundary diagnostics
  as legacy-registration-only fail-closed paths.
- Added target artifact export regression coverage that rejects legacy mirror
  roles when route preflight requires typed selected-plan metadata.
- Updated registry, RVV planning, RVV extension, scalar extension, artifact
  export, and focused lit expectations to assert registration/mirror scope
  instead of descriptor-owned production authority.
- Self-repair performed:
  - Rebuilt `tcrv-opt` after a focused lit failure exposed stale relinking.
  - Updated `test/Target/I32BinaryFamilyRegistry/registry.test` after full
    `check-tianchenrv` exposed a stale FileCheck string.
- Validation completed:
  - focused C++ build and runs for target artifact export, i32 registry, RVV
    extension, scalar extension, RVV binary planning, and runtime ABI callable
    plan tests.
  - focused lit over direct RVV/scalar, RVV+scalar dispatch, bundle, dry-run,
    and lowering-boundary quarantine routes, 62/62 passed after relink.
  - `git diff --check`
  - `git diff --cached --check`
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-scalar-descriptor-registry-quarantine`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
    207/207 passed after stale FileCheck self-repair.
- No `ssh rvv` runtime, correctness, or performance claim was made.
