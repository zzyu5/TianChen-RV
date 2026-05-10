# Promote RVV binary family ownership out of i32 legacy registry

## Goal

Create one target/RVV-owned binary family registry/interface for the finite
supported RVV binary families currently in the repository: `i32-vadd`,
`i32-vsub`, `i32-vmul`, and `i64-vadd`. Frontend lowering, RVV plugin
proposal/materialization/legality, RVV target artifact export, and focused
registry tests should consume that RVV-owned family interface instead of
mixing `I32BinaryFamilyRegistry` data with i64-only RVV special cases.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Starting state is clean on `main` at HEAD `18b1e28`.
* No current Trellis task existed, so this task owns the bounded structural
  refactor round.
* The previous `linalg-i64-vadd-frontend-rvv-ssh-evidence` task is complete
  and archived at commit `18b1e28`; this task must not reopen it or create
  another i64-vadd evidence task.
* Current `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h` still imports
  `include/TianChenRV/Target/I32BinaryFamilyRegistry.h` to synthesize RVV i32
  descriptors, while owning i64-vadd directly.
* Current `lib/Transforms/LowerLinalgI32BinaryToExec.cpp` has separate i32
  frontend spec getters plus an i64-vadd RVV lookup branch.
* Current `lib/Plugin/RVV/RVVExtensionPlugin.cpp` and
  `lib/Target/RVV/RVVMicrokernel.cpp` still define RVV i32 family helper specs
  from the i32 legacy registry and then branch to RVV i64 descriptor lookups.
* The i32 scalar fallback and RVV+scalar dispatch surfaces still legitimately
  depend on i32 scalar/dispatch metadata; this task should not erase that owner
  unless a bounded refactor proves it is safe.

## Boundaries

* Compiler behavior remains implemented in C++ / MLIR / LLVM / TableGen /
  CMake / lit / FileCheck.
* Python may only remain tooling/evidence orchestration; it must not implement
  descriptor lookup, compiler lowering, legality, emission, or registry logic.
* `tcrv.exec` stays compute-free. RVV binary compute/lowering metadata belongs
  under RVV plugin/target code.
* Generic plugin, selection, readiness, and target artifact orchestration must
  remain target-neutral and must not grow RVV/dtype/operation branches.
* No new semantic coverage: no i64-vsub, i64-vmul, i16/i8/fp, new LMULs,
  generic linalg lowering, or new runtime/performance claim.
* Do not commit `artifacts/tmp`, build outputs, raw ssh logs, generated
  evidence files, credentials, or secrets.

## Requirements

1. Add or refactor to one RVV-owned binary family registry/interface under
   `include/TianChenRV/Target/RVV/` and/or `lib/Target/RVV/` that exposes the
   finite RVV family set: `i32-vadd`, `i32-vsub`, `i32-vmul`, and `i64-vadd`.
2. The RVV family descriptor must own dtype, arithmetic kind, family/frontend
   lowering identifiers, lowering descriptor, descriptor noun, RVV
   microkernel/op names, arithmetic verb, function stem, header guard stem,
   intrinsic prefix, result C name, route IDs, emission kind, runtime ABI
   kind/name/glue role, external ABI component group, C operator, scalar C
   type, input/output pointer C types, and shape-composition hooks.
3. Preserve public spellings and existing behavior for all current i32 and
   i64 paths, including pass/tool spellings, frontend lowering attribute
   values, descriptor strings, route IDs, runtime ABI names, and evidence
   runner options.
4. Keep the i32 scalar/dispatch legacy registry available for scalar fallback
   and RVV+scalar dispatch ownership, but stop making
   `RVVBinaryDescriptor.h` depend on `I32BinaryFamilyRegistry.h`.
5. Update frontend lowering so RVV paths derive `FrontendBinarySpec` from the
   RVV-owned family descriptor rather than hard-coded i32 getter branches plus
   an i64-vadd special case.
6. Update RVV plugin proposal, materialization, legality, readiness, and
   emission-plan code so the RVV binary family decision is descriptor/registry
   driven for both i32 and i64 families.
7. Update RVV target artifact export/microkernel code so route, ABI, intrinsic
   spelling, C ABI types, function stems, and header/object helper matching are
   derived from the RVV binary descriptor/selected shape interface.
8. Add or update focused C++ unit tests proving RVV registry lookup by
   frontend lowering and lowering descriptor for all four families, including
   route IDs, runtime ABI names, scalar C types, pointer C types, and selected
   shape capability hooks.
9. Do not weaken existing `I32BinaryFamilyRegistry` scalar/dispatch tests;
   add RVV-owned assertions even if the old test target name remains.

## Acceptance Criteria

* [x] A RVV-owned family registry exposes exactly the four current RVV binary
      families and lookup by family id, frontend lowering, and lowering
      descriptor works for each.
* [x] `RVVBinaryDescriptor.h` no longer includes or depends on
      `I32BinaryFamilyRegistry.h`.
* [x] Frontend lowering uses the RVV-owned family descriptor for all RVV
      binary frontend markers while preserving i32 and i64 ABI output.
* [x] RVV plugin proposal/materialization/legality consumes the same
      RVV-owned family descriptor for i32-vadd, i32-vsub, i32-vmul, and
      i64-vadd.
* [x] RVV target export/microkernel code consumes descriptor-derived route,
      ABI, intrinsic, and shape metadata for existing i32 and i64 paths.
* [x] Existing i32 scalar fallback and RVV+scalar dispatch behavior remains
      compatible and still uses the legacy i32 scalar/dispatch registry where
      that is the correct owner.
* [x] Focused C++ and lit/FileCheck validation passes for touched registry,
      RVV plugin, target artifact export, and linalg frontend surfaces.
* [x] No `ssh rvv` runtime claim is made unless generated artifact or runner
      behavior changes in a way that requires fresh runtime evidence.

## Minimal Validation Plan

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
* Focused C++ tests:
  * `tianchenrv-i32-binary-family-registry-test`
  * `tianchenrv-rvv-extension-plugin-test`
  * `tianchenrv-target-artifact-export-test`
* Focused lit/FileCheck tests:
  * `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`
  * `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`
  * one existing i32 non-add family test such as i32-vsub or i32-vmul.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical after focused checks.
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-binary-family-registry-owner`

## Out Of Scope

* No new RVV arithmetic families, dtypes, LMUL shapes, or performance claims.
* No generic arbitrary linalg lowering.
* No Python implementation of compiler descriptor lookup or semantics.
* No compute semantics inside `tcrv.exec`.
* No broad smoke matrix or report-only closeout.
* No hand-written standalone RVV C path that bypasses compiler-generated
  artifacts.

## Technical Notes

* Required specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/architecture/design-boundaries.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/plugin-protocol/locality-contract.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Required archive context read:
  * `.trellis/tasks/archive/2026-05/05-10-rvv-dtype-axis-i64-vadd/prd.md`
  * `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vadd-ssh-evidence/prd.md`
  * `.trellis/tasks/archive/2026-05/05-10-linalg-i64-vadd-frontend-rvv-ssh-evidence/prd.md`
* Primary source/test surfaces inspected:
  * `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  * `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  * `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`
  * `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  * `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  * `lib/Target/RVV/RVVMicrokernel.cpp`
  * `lib/Target/TargetArtifactExport.cpp`
  * `test/Target/I32BinaryFamilyRegistryTest.cpp`
  * `test/Plugin/RVVExtensionPluginTest.cpp`
  * `test/Transforms/LinalgToExec/`
  * `test/Scripts/rvv-microkernel-e2e.test`
