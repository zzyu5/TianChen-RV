# Descriptor exit from scalar i32 add fallback path

## Goal

Move the production/default scalar fallback i32 add path away from
`tcrv_scalar.lowering_descriptor` compute authority. The default scalar i32 add
fallback must use the typed `tcrv_scalar.i32_vadd_microkernel` body/config as
the source of compute semantics and must emit runtime-callable C/C++ through
the common EmitC materialization route.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current branch is `main`; the working tree was clean at task creation.
* Current HEAD is `a093722 feat(rvv): remove descriptor authority from i32 add default`.
* The archived RVV task
  `.trellis/tasks/archive/2026-05/05-11-descriptor-exit-rvv-i32-add-default/`
  completed the analogous RVV i32 add default path migration.
* Scalar fallback remains a bounded plugin-owned fallback route. It must not
  become a high-level compute IR, a new independent backend, or a descriptor
  driven architecture.
* The brief identifies live descriptor authority in
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir` expectations and
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` materialization planning.
* This task is not an RVV runtime evidence task. No `ssh rvv` evidence is
  required unless a runtime correctness or performance claim is introduced.

## Requirements

* The default scalar fallback i32 add proposal/materialization path must no
  longer require `tcrv_scalar.lowering_descriptor` to choose add semantics.
* The typed `tcrv_scalar.i32_vadd_microkernel` op/body/config must be the
  compute authority for the default scalar i32 add fallback path.
* Scalar i32 add export must use the common EmitC lowerable route/materializer
  instead of descriptor-to-C semantics.
* The default i32 add frontend/RVV+scalar pipeline must materialize a
  descriptorless scalar fallback branch where the RVV branch is already
  descriptorless.
* Descriptor data may remain only as bounded legacy/cross-check metadata on
  non-default or stale fixtures. A conflicting scalar descriptor must not
  override a typed `tcrv_scalar.i32_vadd_microkernel` body and must fail closed.
* Runtime ABI parameter layering must remain intact: lhs/rhs/out buffer roles
  and runtime `n` are IR-backed ABI/control values, not descriptor-local
  compute semantics.
* Obsolete scalar i32 add tests that encode descriptor-driven computation as
  the intended default behavior must be updated, deleted, or quarantined as
  legacy mismatch/cross-check coverage.

## Acceptance Criteria

* [x] The default scalar i32 add fallback path succeeds without
      `tcrv_scalar.lowering_descriptor` selecting add semantics.
* [x] The scalar i32 add callable C/C++ source is produced through the common
      EmitC route from a typed `tcrv_scalar.i32_vadd_microkernel` body.
* [x] Generated scalar fallback source records bounded common EmitC route
      provenance and does not contain descriptor text as compute authority.
* [x] A stale or conflicting scalar descriptor cannot override a typed
      `tcrv_scalar.i32_vadd_microkernel` body.
* [x] Focused lit/FileCheck coverage proves the default i32 add frontend
      RVV+scalar path is descriptorless for both RVV and scalar fallback compute
      branches where applicable.
* [x] Focused C++ tests prove scalar plugin/materialization/export behavior no
      longer depends on descriptor-driven scalar i32 add semantics.
* [x] No compiler core, dialect, pass, plugin registry, capability model,
      lowering, or emission logic is implemented as Python data structures.

## Out Of Scope

* i32 sub/mul migrations except preserving existing behavior.
* i64 migrations.
* RVV runtime, correctness, throughput, latency, or performance claims.
* New MLIR vector, LLVM scalable vector, high-level linalg lowering, generic
  scalar compute IR, or LLVM/RISC-V lowering path.
* Moving compute semantics into `tcrv.exec`.
* Adding a parallel helper route that is not consumed by the default scalar i32
  add production/export path in this same round.
* Preserving descriptor-driven scalar i32 add tests as default architecture.

## Minimal Validation

* Focused scalar plugin/export C++ tests covering descriptorless default i32
  add materialization/export and conflicting descriptor fail-closed behavior.
* Focused lit/FileCheck tests under `test/Transforms/LinalgToExec/`,
  `test/Transforms/ExecutionPlanning/`, or target/export tests proving the
  default frontend/RVV+scalar i32 add path uses descriptorless scalar fallback
  compute and common EmitC source export.
* Source-output check proving scalar i32 add generated source has no descriptor
  text as compute authority and records common EmitC route provenance.
* Build focused changed targets, including relevant scalar plugin/export tests
  and `tcrv-opt` / `tcrv-translate` if touched.
* Run focused lit tests for changed scalar i32 add pipeline/export cases.
* Run `git diff --check`.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` before finalizing if the local build state permits.
* Validate this Trellis task before finalizing, and validate the archived task
  if this task is archived.

## Technical Notes

* `.trellis/spec/index.md` states descriptor-driven computation is bounded
  implementation debt and the current main lowering route is extension family
  ops to EmitC to intrinsic/runtime C/C++.
* `.trellis/spec/architecture/unified-riscv-mlir.md` forbids descriptors from
  defining computation semantics or standing in for extension ops.
* `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` still documents
  the historical finite descriptor slice, but this PRD intentionally narrows
  the migration to the default scalar i32 add fallback route.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the generated
  `TCRVEmitCLowerableOpInterface` boundary and the correct route:
  typed extension op -> interface-backed route -> family-owned runtime mapping
  -> common EmitC/C source evidence.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires the
  callable ABI parameter plan to mirror real `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param` IR boundaries.
* `.trellis/spec/plugin-protocol/extension-family-plugin-template.md` requires
  extension families to contribute plugin-local ops, verification, EmitC
  mapping, and tests without adding family-specific semantic branches to core
  orchestration.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused lit/C++
  coverage for scalar explicit microkernel export, frontend-to-export
  behavior, route preflight, and fail-closed descriptor/body mismatches.
