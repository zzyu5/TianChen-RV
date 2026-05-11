# Descriptor exit from remaining i32 sub/mul default paths

## Goal

Move the remaining production/default i32 subtract and multiply selected/export
paths away from lowering-descriptor compute authority. The RVV direct and
scalar fallback default i32 sub/mul paths must derive arithmetic semantics from
typed TCRV extension family microkernel ops and flow through the generated
`TCRVEmitCLowerableOpInterface` plus the common EmitC materializer.

`tcrv_rvv.lowering_descriptor` and `tcrv_scalar.lowering_descriptor` may remain
only as bounded legacy/cross-check metadata for non-default paths, i64 paths
that are not part of this task, or stale/conflicting descriptor validation.
They must not select arithmetic semantics for default i32 sub/mul.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current branch is `main`; the worktree was clean before task creation.
* Current HEAD is `27c6ca9 feat(scalar): remove descriptor authority from i32 add default`.
* There was no active `.trellis/.current-task`; this task was created new as
  `.trellis/tasks/05-11-descriptor-exit-i32-sub-mul-defaults/`.
* The archived RVV add task completed the RVV default i32 add descriptor exit
  at `a093722`.
* The archived scalar add task completed the scalar default i32 add descriptor
  exit at `27c6ca9`.
* Live task brief identifies remaining descriptor-owned expectations in i32
  sub/mul fixtures and tests, including RVV/scalar linalg artifact tests,
  `RVVExtensionPluginTest.cpp`, and `TargetArtifactExportTest.cpp`.
* This round is a structural compiler/export migration. It is not an RVV
  runtime, correctness, throughput, latency, or performance evidence task.

## Requirements

* Default RVV direct i32 vsub and vmul proposals/materialized variants must no
  longer require `tcrv_rvv.lowering_descriptor` to choose subtract or multiply
  semantics.
* Default scalar fallback i32 vsub and vmul proposals/materialized variants
  must no longer require `tcrv_scalar.lowering_descriptor` to choose subtract
  or multiply semantics.
* The compute authority for default i32 sub/mul must be the typed
  `tcrv_rvv.i32_vsub_microkernel`, `tcrv_rvv.i32_vmul_microkernel`,
  `tcrv_scalar.i32_vsub_microkernel`, and
  `tcrv_scalar.i32_vmul_microkernel` body/config surfaces.
* Source export for those paths must pass through a verified
  `TCRVEmitCLowerableRoute` and common EmitC materialization before target-owned
  bounded C/C++ output.
* Generalize the existing descriptorless i32 add pattern where practical rather
  than adding isolated family-name workarounds.
* Stale or conflicting descriptors must fail closed before artifact/source
  output. A stale descriptor must not override a typed i32 sub/mul body.
* Runtime ABI parameter layering must remain intact: lhs/rhs/out buffers and
  runtime `n` come from IR-backed ABI/control surfaces and validated plan
  mirrors, not descriptor-local compute semantics.
* Obsolete lit/C++ expectations that preserve descriptor-driven computation as
  the default i32 sub/mul architecture must be updated, deleted, or quarantined
  as explicit legacy/cross-check coverage.

## Acceptance Criteria

* [x] Default RVV i32 vsub selected/export path succeeds without
      `tcrv_rvv.lowering_descriptor` selecting subtract semantics.
* [x] Default RVV i32 vmul selected/export path succeeds without
      `tcrv_rvv.lowering_descriptor` selecting multiply semantics.
* [x] Default scalar i32 vsub selected/export path succeeds without
      `tcrv_scalar.lowering_descriptor` selecting subtract semantics.
* [x] Default scalar i32 vmul selected/export path succeeds without
      `tcrv_scalar.lowering_descriptor` selecting multiply semantics.
* [x] Generated RVV i32 vsub/vmul C source still contains the family-correct
      RVV intrinsic route, for example `__riscv_vsub_vv_i32m1` or
      `__riscv_vmul_vv_i32m1`, and records bounded common EmitC route
      provenance rather than descriptor compute authority.
* [x] Generated scalar i32 vsub/vmul C source still emits family-correct
      portable C arithmetic (`lhs[index] - rhs[index]` or
      `lhs[index] * rhs[index]`) through the common EmitC route provenance.
* [x] Focused C++ tests prove descriptorless default i32 sub/mul behavior for
      RVV and scalar plugin/export paths and prove stale/conflicting descriptors
      fail closed.
* [x] Focused lit/FileCheck tests prove default linalg-to-exec,
      execution-planning, and target/export paths no longer assert
      `tcrv_rvv.lowering_descriptor` or `tcrv_scalar.lowering_descriptor` as
      compute authority for i32 sub/mul.
* [x] No compiler core, dialect, pass, plugin registry, capability model,
      lowering, or emission logic is implemented as Python data structures.

## Out Of Scope

* i64 descriptor-exit migration.
* New RVV/Scalar arithmetic families or a broad finite-family matrix.
* Benchmarks, throughput, latency, or performance claims.
* MLIR vector, LLVM scalable-vector, LLVM RVV IR, inline assembly, or generic
  RVV backend claims.
* Moving compute semantics into `tcrv.exec`.
* Descriptor-to-C replacement hidden behind comments, manifest metadata, or a
  helper that is not consumed by the default production/export path.
* SSH evidence unless a fresh runtime/correctness claim is introduced.

## Minimal Validation

* Build focused targets touched by the migration, including RVV plugin, scalar
  plugin, target artifact export, relevant dialect/interface targets if
  touched, `tcrv-opt`, and `tcrv-translate`.
* Run focused C++ tests such as `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and any lowerable-interface tests
  touched by the implementation.
* Run focused lit/FileCheck for changed `Transforms/LinalgToExec`,
  `Transforms/ExecutionPlanning`, `Target/ArtifactExport`,
  `Target/RVVMicrokernel`, and `Target/Scalar` cases as applicable.
* Run `git diff --check`.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-descriptor-exit-i32-sub-mul-defaults`.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if the local build state permits.
* If the task is archived, validate the archived task path after archive.

## Technical Notes

* `.trellis/spec/index.md` states descriptor-driven computation is bounded
  implementation debt and the current main route is extension family ops to
  EmitC to intrinsic/runtime C/C++.
* `.trellis/spec/architecture/unified-riscv-mlir.md` forbids descriptors from
  defining computation semantics or standing in for extension family ops.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the correct boundary:
  typed extension op implements generated lowerable interface, target/plugin
  maps to family-owned intrinsic/runtime call, and common route provenance is
  bounded.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  runtime ABI role/name/type/ownership metadata to mirror IR-backed
  `mem_window` and `runtime_param` boundaries.
* `.trellis/spec/extension-plugins/rvv-plugin.md` documents the current RVV
  i32 add/sub/mul microkernel slice and the typed structured body that target
  export must consume.
* `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` documents scalar
  add/sub/mul source/header/object routes and explicitly requires
  family-correct arithmetic, route ids, runtime ABI metadata, and fail-closed
  stale-family handling.
* `.trellis/spec/plugin-protocol/extension-family-plugin-template.md` requires
  extension families to own ops, verification, EmitC mapping, and tests without
  adding family-specific semantic branches to core orchestration.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused lit/C++
  coverage for RVV/scalar explicit microkernel export, default frontend
  add/sub/mul paths, common EmitC route evidence, and fail-closed descriptor
  mismatches.
* Archived add-path PRDs are reference patterns only; this task must not reopen
  those archived tasks or treat another add-path/evidence-only round as the
  deliverable.
