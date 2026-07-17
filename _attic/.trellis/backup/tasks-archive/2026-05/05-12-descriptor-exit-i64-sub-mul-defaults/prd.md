# Descriptor exit from the i64 sub/mul default paths

## Goal

Move the production/default `i64-vsub` and `i64-vmul` selected/export paths away
from lowering-descriptor compute authority. The default RVV direct route and
scalar fallback route must derive subtract/multiply semantics from typed TCRV
extension family ops, flow through generated `TCRVEmitCLowerableOpInterface`
provenance, and then use the common EmitC materializer before target-owned
bounded C artifact output.

`tcrv_rvv.lowering_descriptor`, `tcrv_scalar.lowering_descriptor`, and
`selected_lowering_descriptor` may remain only as bounded legacy/cross-check
metadata. They must not choose the arithmetic for the migrated default
`i64-vsub` or `i64-vmul` paths.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current branch is `main`; the worktree was clean before task creation.
* Current HEAD before this task is
  `97c5205 feat: remove descriptor authority from i64 add default`.
* There was no active `.trellis/.current-task`; this task was created new as
  `.trellis/tasks/05-12-descriptor-exit-i64-sub-mul-defaults/`.
* The previous task
  `.trellis/tasks/archive/2026-05/05-11-descriptor-exit-i64-add-default/`
  is finished/archived and is only an implementation pattern reference.
* The archived i32 sub/mul task
  `.trellis/tasks/archive/2026-05/05-11-descriptor-exit-i32-sub-mul-defaults/`
  is also a pattern reference only and must not be reopened.
* Recent history has already removed descriptor authority from i32 add,
  i32 sub/mul, and i64 add default paths. This task covers the remaining
  default i64 binary subtract/multiply families.
* This round is a structural compiler/export migration. It is not an RVV
  runtime, correctness, throughput, latency, or performance evidence task.

## Requirements

* Add or complete generated `TCRVEmitCLowerableOpInterface` support for the
  bounded typed source ops that authority the default paths:
  `tcrv_rvv.i64_vsub_microkernel`, `tcrv_rvv.i64_vmul_microkernel`,
  `tcrv_scalar.i64_vsub_microkernel`, and
  `tcrv_scalar.i64_vmul_microkernel`.
* Rewire RVV selected-boundary, emission planning, and target artifact export so
  default `i64-vsub` and `i64-vmul` use typed RVV family op/body surfaces and
  generated lowerable-route provenance for arithmetic semantics.
* Rewire scalar fallback selected/export logic so default scalar i64 sub/mul
  uses typed scalar family op/body surfaces and generated lowerable-route
  provenance for portable `int64_t` subtraction/multiplication.
* Ensure selected-plan metadata records typed family/source-op/interface
  provenance such as selected binary family, EmitC source op, and common
  lowerable route, not descriptor compute authority.
* Keep runtime ABI layering intact: lhs/rhs/out buffers and runtime `n`/AVL/VL
  must come from IR-backed ABI/control surfaces and selected-plan mirrors, not
  descriptor-local compute semantics. Element count remains descriptor-local
  sample metadata only.
* Preserve family-correct generated source behavior. RVV export must route to
  the i64 RVV subtract/multiply intrinsic families, and scalar export must
  route to portable `int64_t` subtraction/multiplication.
* Descriptors must be bounded legacy/cross-check metadata only. A stale or
  conflicting descriptor must fail closed before artifact/source emission and
  must not override a typed i64 sub/mul body.
* Update, delete, or quarantine tests that preserve descriptor-driven i64
  sub/mul computation as the default architecture. Legacy descriptor assertions
  may remain only when explicitly testing cross-check/fail-closed behavior.

## Acceptance Criteria

* [x] Default RVV direct `i64-vsub` selected/export path succeeds without
      `tcrv_rvv.lowering_descriptor` choosing subtract semantics.
* [x] Default RVV direct `i64-vmul` selected/export path succeeds without
      `tcrv_rvv.lowering_descriptor` choosing multiply semantics.
* [x] Default scalar fallback `i64-vsub` selected/export path succeeds without
      `tcrv_scalar.lowering_descriptor` choosing subtract semantics.
* [x] Default scalar fallback `i64-vmul` selected/export path succeeds without
      `tcrv_scalar.lowering_descriptor` choosing multiply semantics.
* [x] `tcrv_rvv.i64_vsub_microkernel` and
      `tcrv_rvv.i64_vmul_microkernel` implement/query generated
      `TCRVEmitCLowerableOpInterface` provenance for the common EmitC route.
* [x] `tcrv_scalar.i64_vsub_microkernel` and
      `tcrv_scalar.i64_vmul_microkernel` implement/query generated
      `TCRVEmitCLowerableOpInterface` provenance for the common EmitC route.
* [x] Generated RVV i64 sub/mul source contains family-correct i64 RVV
      intrinsic behavior and records bounded common EmitC route provenance
      instead of descriptor compute authority.
* [x] Generated scalar i64 sub/mul source emits portable `int64_t`
      subtraction/multiplication through common EmitC route provenance.
* [x] Focused C++ tests prove typed i64 sub/mul lowerable provenance and prove
      stale or conflicting descriptors fail closed before artifact/source
      emission for both RVV and scalar fallback.
* [x] Focused lit/FileCheck tests prove default linalg-to-exec,
      selected-boundary/emission planning, RVV microkernel export, scalar
      export, artifact export, and dispatch/bundle cases touched by migration
      no longer assert descriptor authority for i64 sub/mul.
* [x] `tcrv.exec` remains compute-free and no compiler core, dialect, pass,
      plugin registry, capability model, lowering, or emission logic is
      implemented as Python data structures.

## Out Of Scope

* New arithmetic families, dtype families, LMUL expansion, broad finite-family
  matrix, generic RVV backend, MLIR vector/scalable-vector route, LLVM
  lowering, inline assembly, or performance work.
* Moving compute semantics into `tcrv.exec`.
* Extension-specific semantic branches in core/common passes where the correct
  boundary is a family op interface, plugin hook, or target-owned route.
* Descriptor-to-C replacement hidden behind comments, manifests, helper code,
  or tests not consumed by the default path.
* SSH RVV runtime/correctness/performance evidence unless this task makes a new
  fresh runtime/correctness/performance claim.
* Archiving a partial sub/mul migration as complete. If one family must remain,
  the task stays open with an exact continuation point.

## Minimal Validation

* Run `git diff --check`.
* Configure if needed with the existing LLVM/MLIR CMake paths.
* Build focused touched targets, including `tcrv-opt`, `tcrv-translate`, RVV
  plugin tests, scalar plugin tests, target artifact export tests, EmitC
  lowerable interface tests, and any dialect/interface tests touched by ODS
  changes.
* Run focused C++ tests proving typed i64 sub/mul lowerable provenance,
  family-correct selected planning, and stale/conflicting descriptor
  fail-closed behavior for both RVV and scalar fallback.
* Run focused lit/FileCheck for default `i64-vsub` and `i64-vmul`
  linalg-to-exec, RVV microkernel export, scalar export, artifact export, and
  RVV+scalar dispatch/bundle cases touched by the migration.
* Confirm generated RVV source uses expected i64 sub/mul RVV intrinsic families
  and generated scalar source uses portable `int64_t`
  subtraction/multiplication through the common EmitC route.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if local build state permits after focused checks pass.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-descriptor-exit-i64-sub-mul-defaults`.
* If the task is archived, validate the archived task path after archive.

## Round Result

Completed on 2026-05-12.

* RVV and scalar i64 sub/mul typed source ops now implement generated
  `TCRVEmitCLowerableOpInterface` where they are used as default compute
  authority.
* Default RVV planning, selected-boundary/emission planning, RVV microkernel
  export, scalar fallback export, target artifact export, and RVV+scalar
  dispatch preflight now treat all i64 binary defaults as typed source routes,
  matching the previous i64 add descriptor-exit pattern.
* Default selected-plan metadata records typed binary family, EmitC source op,
  and `TCRVEmitCLowerableOpInterface`. `selected_lowering_descriptor` is no
  longer part of default i64 sub/mul selected metadata.
* Legacy descriptor values remain bounded to stale/conflict cross-check paths
  where present and fail closed before source/artifact emission when they
  conflict with the typed body.
* No RVV runtime, correctness, throughput, latency, or performance claim was
  made; no `ssh rvv` evidence was required or collected.

Validation completed:

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-extension-plugin-test
  tianchenrv-scalar-extension-plugin-test
  tianchenrv-rvv-binary-planning-test
  tianchenrv-target-artifact-export-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* Focused lit for i64 sub/mul linalg-to-exec, RVV microkernel export,
  RVV+scalar dispatch, and target artifact bundle cases: 8/8 passed.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 206/206 lit tests.

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
* `.trellis/spec/extension-plugins/rvv-plugin.md` documents RVV microkernel
  ownership, typed structured bodies, selected routes, and descriptor debt.
* `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` documents scalar
  fallback source/header/object routes, descriptor debt, and family-correct
  source generation requirements.
* `.trellis/spec/plugin-protocol/extension-family-plugin-template.md` requires
  extension families to own ops, verification, EmitC mapping, and tests without
  adding family-specific semantic branches to core orchestration.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused lit/C++
  coverage for RVV/scalar explicit microkernel export, default frontend paths,
  common EmitC route evidence, and fail-closed descriptor mismatches.
* Archived i64 add and i32 sub/mul descriptor-exit PRDs are pattern references
  only. This task must not reopen archived tasks or report previous coverage as
  the i64 sub/mul deliverable.
