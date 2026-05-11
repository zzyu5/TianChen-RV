# Descriptor exit from RVV i32 add default path

## Goal

Move the production/default RVV i32 add selected/export path away from
descriptor-owned compute semantics. The default i32 add export must use the
typed `tcrv_rvv` family body plus the common EmitC materializer as the compute
authority. `tcrv_rvv.lowering_descriptor` may remain only as bounded legacy or
cross-check metadata and must not choose the arithmetic operation for the
default i32 add path.

## What I already know

* The repository is `/home/kingdom/phdworks/TianchenRV`.
* The current branch is `main`; the working tree was clean at task creation.
* Recent commits already introduced generated lowerable op interfaces and real
  EmitC materialization route work.
* The Hermes brief explicitly forbids coverage-only, ssh-only, metadata-only,
  helper-only, prompt-only, and spec-only rounds.
* The active structural owner is descriptor exit for the RVV i32 add default
  path, not i32 sub/mul, i64, high-level linalg, MLIR vector, LLVM scalable
  vector, or a new Python compiler path.

## Requirements

* Identify where the default RVV i32 add export path still treats
  `lowering_descriptor`, `i32-vadd-microkernel.v1`, descriptor family registry
  data, or descriptor-owned tests as compute authority.
* Rewire the production/default i32 add selected/export path so compute
  semantics come from the typed RVV family body/config.
* Use `TCRVEmitCLowerableOpInterface`, `TCRVEmitCLowerableRoute`, and the real
  EmitC materializer when exporting the default RVV i32 add path.
* Ensure generated C/C++ for the default i32 add path comes from the EmitC
  route over typed RVV family ops.
* Keep descriptor data only as bounded non-default legacy metadata or mismatch
  cross-check input when unrelated paths still need it.
* Delete, update, or quarantine obsolete descriptor-only tests when they encode
  descriptor-driven computation as the intended default architecture.

## Acceptance Criteria

* [x] A default RVV i32 add export succeeds through typed RVV family ops and
      real EmitC materialization.
* [x] The default RVV i32 add compute authority is the typed body/config, not
      `tcrv_rvv.lowering_descriptor`.
* [x] Stale or conflicting descriptor data cannot override the typed RVV family
      body.
* [x] Generated output still contains RVV intrinsic C/C++ from the common
      EmitC route.
* [x] Focused MLIR/C++ tests cover the changed behavior.
* [x] No new compiler core, dialect, pass, registry, capability, lowering, or
      emission logic is implemented as Python data structures.

## Out Of Scope

* i32 sub/mul coverage as standalone owner.
* i64 routes.
* ssh evidence as standalone owner.
* MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, or inline assembly
  lowering.
* New high-level linalg lowering.
* New Python compiler logic.
* Prompt/spec-only changes.

## Technical Notes

* `.trellis/spec/index.md` states descriptor-driven computation is bounded
  implementation debt and the current main lowering route is extension family
  ops to EmitC to intrinsic/runtime C/C++.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the generated
  lowerable op interface boundary and explicitly marks descriptor strings
  alone choosing arithmetic intrinsics as wrong.
* `.trellis/spec/extension-plugins/rvv-plugin.md` still documents finite
  descriptor metadata for the first RVV slice, but the Hermes brief narrows
  this task to removing descriptor authority from the default i32 add compute
  export path.
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires common
  orchestration through plugin interfaces and the common EmitC route, without
  core target-family semantic branches.
* `.trellis/spec/testing/index.md` requires lit/FileCheck or C++ tests for
  compiler behavior and real `ssh rvv` evidence only for runtime correctness or
  performance claims. This task is a compiler export-path migration and should
  use focused local compiler tests unless making runtime claims.
