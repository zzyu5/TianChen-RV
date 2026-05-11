# Descriptor exit from the i64 add default path

## Goal

Move the production/default `i64-vadd` selected/export path away from
lowering-descriptor compute authority. The default RVV direct route and scalar
fallback route must derive add semantics from typed TCRV extension family ops,
flow through generated `TCRVEmitCLowerableOpInterface` provenance, and then use
the common EmitC materializer before target-owned bounded C artifact output.

`tcrv_rvv.lowering_descriptor`, `tcrv_scalar.lowering_descriptor`, and
`selected_lowering_descriptor` may remain only as bounded legacy/cross-check
metadata. They must not choose the arithmetic for the migrated default
`i64-vadd` path.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current branch is `main`; the worktree was clean before task creation.
* Current HEAD before this task is
  `93a95e5 feat: remove descriptor authority from i32 sub mul defaults`.
* There was no active `.trellis/.current-task`; this task was created new as
  `.trellis/tasks/05-11-descriptor-exit-i64-add-default/`.
* The previous task
  `.trellis/tasks/archive/2026-05/05-11-descriptor-exit-i32-sub-mul-defaults/`
  is finished/archived and is only an implementation pattern reference.
* Existing source evidence identifies i64 add as still descriptor-backed:
  scalar i64 ops and RVV i64 microkernel ops exist without generated lowerable
  interface authority, and tests still call out descriptor-backed scalar/RVV i64
  add paths.
* This round is a structural compiler/export migration. It is not an RVV
  runtime, correctness, throughput, latency, or performance evidence task.

## Requirements

* Add generated `TCRVEmitCLowerableOpInterface` support to the bounded typed
  i64 add source ops that authority the default path:
  `tcrv_rvv.i64_vadd_microkernel` and, if used by the production fallback,
  `tcrv_scalar.i64_vadd_microkernel`.
* Rewire RVV selected-boundary, emission planning, and target artifact export
  so default `i64-vadd` uses typed RVV family op/body surfaces and generated
  lowerable-route provenance for add semantics.
* Rewire scalar fallback selected/export logic so default scalar i64 add uses
  typed scalar family op/body surfaces and generated lowerable-route provenance
  for portable `int64_t` addition.
* Keep runtime ABI layering intact: lhs/rhs/out buffers and runtime `n` must
  come from IR-backed ABI/control surfaces and selected-plan mirrors, not
  descriptor-local compute semantics.
* Preserve family-correct generated source behavior. RVV export must route to
  the i64 RVV add intrinsic family, and scalar export must route to portable
  `int64_t` addition.
* Descriptors must be bounded legacy/cross-check metadata only. A stale or
  conflicting descriptor must fail closed before artifact/source emission and
  must not override a typed i64 add body.
* Update, delete, or quarantine tests that preserve descriptor-driven i64 add
  computation as the default architecture. Legacy descriptor assertions may
  remain only when explicitly testing cross-check/fail-closed behavior.
* Keep the deliverable bounded to i64 add. Do not claim i64 sub/mul descriptor
  exit unless those default paths are fully rewired and validated in this same
  round.

## Acceptance Criteria

* [ ] Default RVV direct `i64-vadd` selected/export path succeeds without
      `tcrv_rvv.lowering_descriptor` choosing add semantics.
* [ ] Default scalar fallback `i64-vadd` selected/export path succeeds without
      `tcrv_scalar.lowering_descriptor` choosing add semantics.
* [ ] `tcrv_rvv.i64_vadd_microkernel` implements/query generated
      `TCRVEmitCLowerableOpInterface` provenance for the common EmitC route.
* [ ] `tcrv_scalar.i64_vadd_microkernel` implements/query generated
      `TCRVEmitCLowerableOpInterface` provenance when scalar i64 add export is
      on the production fallback route.
* [ ] Generated RVV i64 add source contains family-correct i64 RVV add intrinsic
      behavior and records bounded common EmitC route provenance instead of
      descriptor compute authority.
* [ ] Generated scalar i64 add source emits portable `int64_t` addition through
      common EmitC route provenance.
* [ ] Focused C++ tests prove typed i64 add lowerable provenance and prove stale
      or conflicting descriptors fail closed before artifact/source emission.
* [ ] Focused lit/FileCheck tests prove default linalg-to-exec,
      selected-boundary/emission planning, RVV microkernel export, scalar
      export, artifact export, and dispatch/bundle cases touched by migration
      no longer assert descriptor authority for i64 add.
* [ ] `tcrv.exec` remains compute-free and no compiler core, dialect, pass,
      plugin registry, capability model, lowering, or emission logic is
      implemented as Python data structures.

## Out Of Scope

* i64 sub/mul descriptor-exit migration unless it falls out completely and is
  fully validated in this same round.
* New arithmetic families, broad finite-family matrix, generic RVV backend,
  MLIR vector lowering, LLVM scalable-vector route, inline assembly, or
  performance work.
* Moving compute semantics into `tcrv.exec`.
* Extension-specific semantic branches in core/common passes where the correct
  boundary is a family op interface, plugin hook, or target-owned route.
* Descriptor-to-C replacement hidden behind comments, manifests, or helper code
  not consumed by the default path.
* SSH RVV runtime/correctness/performance evidence unless this task makes a new
  fresh runtime/correctness/performance claim.

## Minimal Validation

* Run `git diff --check`.
* Configure if needed with the existing LLVM/MLIR CMake paths.
* Build focused touched targets, including `tcrv-opt`, `tcrv-translate`, RVV
  plugin tests, scalar plugin tests, target artifact export tests, EmitC
  lowerable interface tests, and any dialect/interface tests touched by ODS
  changes.
* Run focused C++ tests that prove typed i64 add ops implement/query generated
  lowerable provenance and stale/conflicting descriptors fail closed.
* Run focused lit/FileCheck for default `i64-vadd` linalg-to-exec,
  selected-boundary/emission planning, RVV microkernel export, scalar export,
  artifact export, and dispatch/bundle cases touched by the migration.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if local build state permits.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-descriptor-exit-i64-add-default`.
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
* Archived i32 descriptor-exit PRD is a pattern reference only. This task must
  not reopen archived i32 tasks or report i32 coverage as the i64 deliverable.
