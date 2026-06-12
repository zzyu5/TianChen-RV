# RVV runtime AVL/VL ABI boundary

## Goal

Make the existing op-owned RVV i32 microkernel route carry an explicit
runtime-length contract from selected RVV planning into target artifact export:
runtime `n` / AVL comes from the IR-backed `tcrv.exec.runtime_param`, runtime
VL is modeled by the `tcrv_rvv.setvl` to `tcrv_rvv.with_vl` chain, and
descriptor-local `tcrv_rvv.element_count` remains bounded fixture/artifact
capacity metadata rather than production runtime-control authority.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round: clean worktree, HEAD
  `0e51c23 feat(rvv): embed op-owned object artifact evidence`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
* The previous task
  `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-object-artifact-evidence-closure/prd.md`
  closed the op-owned source/header/object artifact evidence route and embedded
  selected source/config/runtime ABI provenance in the generated object.
* The earlier task
  `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-artifact-default-emission/prd.md`
  made the default i32-vadd artifact route require selected-boundary typed
  source identity before source/header/object emission.
* Current code already has `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
  selected-vector-shape metadata, runtime ABI role validation, and
  descriptor-local count metadata. The missing module boundary is that runtime
  length is still expressed as scattered helper methods on
  `RVVBinarySelectedConfigContract`, instead of an explicit contract object
  consumed by planning/export validation.

## Requirements

* Keep the migrated slice to the already-supported RVV i32 runtime-callable
  microkernel family/config path, with no new dtype, family, LMUL, performance,
  broad smoke matrix, or scalar-dispatch expansion as the main result.
* Add a coherent RVV runtime-length contract that names:
  * the ABI-visible runtime element-count C parameter;
  * the runtime AVL source and role;
  * the runtime VL producer and scope;
  * descriptor-local element count as quarantined component-capacity metadata.
* RVV plugin planning and target export must consume this contract when
  building selected-plan metadata, source/header/object comments, route
  preflight validation, and object evidence payloads.
* Missing, stale, or descriptor-only runtime length authority must fail before
  generated source/header/object output for the migrated slice.
* The contract must keep target capability facts such as vlenb/VLEN, selected
  compile-time config such as SEW/LMUL/tail/mask, runtime AVL/VL/ABI values,
  and descriptor-local `element_count` distinct.
* Core `tcrv.exec`, generic transforms, and common construction code must not
  gain RVV semantic branches.
* Python may only remain runner/evidence tooling; compiler logic stays in
  C++/MLIR/TableGen/CMake/lit.

## Acceptance Criteria

* [x] `RVVBinarySelectedConfigContract` delegates runtime `n`/AVL/VL and
      descriptor-local count handling to an explicit RVV runtime-length
      contract instead of owning that boundary as loose fields.
* [x] Planning emits the runtime-length contract metadata for selected RVV
      paths, including `runtime_avl_source`, `runtime_avl_role`,
      `runtime_vl_source`, `runtime_vl_scope`, runtime element-count C name,
      and descriptor-local element count.
* [x] Target source/header/object export validates those selected-plan
      metadata entries against the IR-backed runtime ABI parameter and the
      structured `setvl`/`with_vl` body before artifact bytes are emitted.
* [x] Generated RVV source still uses clang-compatible `riscv_vector.h`
      intrinsics and derives `__riscv_vsetvl_*` behavior from selected config
      plus runtime AVL/VL contract, not descriptor-only metadata.
* [x] Fail-closed tests cover missing runtime element-count ABI data, stale
      runtime AVL/VL selected-plan metadata, wrong runtime element-count C
      name/type/role, and descriptor-only element-count production attempts.
* [x] Descriptor-only body or descriptor-owned length/control remains rejected,
      bypassed, or quarantined from the default production route.
* [x] Focused build/lit/C++ checks for touched RVV target/plugin/dialect paths,
      `tcrv-opt`, `tcrv-translate`, and affected artifact/export tests pass.
* [x] `git diff --check`, staged diff check, Trellis validation, archive, and
      one coherent commit complete the round if finished.

## Definition Of Done

* The runtime-length contract is used by the production selected-plan/export
  route, not only by helper tests or comments.
* Existing op-owned object artifact evidence route still passes focused
  regressions.
* The final report states the exact migrated family/config slice, runtime
  AVL/VL contract, fail-closed cases, checks, `ssh rvv` evidence or why it was
  not run, task archive status, and commit hash.

## Out Of Scope

* New RVV family, i64 expansion, vsub/vmul expansion, new LMUL work, broad
  dtype matrix, performance tuning, or broad smoke matrix as the main result.
* Descriptor-to-C production exporter or descriptor element_count as runtime
  control authority.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic transforms or common construction code.
* GCC/vendor compiler as the default route.
* Template/Toy/TensorExtLite changes except narrow regressions caused by
  shared target/export interfaces.
* Runtime/correctness/performance claims beyond focused evidence actually run.

## Technical Approach

Add a small target-owned RVV runtime-length contract near the existing RVV
selected-config contract. Then rewire `RVVBinarySelectedConfigContract` to
compose it, so plugin planning, target export, dispatch metadata, source/header
comments, and object evidence all consume one contract surface. Keep existing
external metadata names stable so current artifacts remain compatible, but make
the internal source of truth explicit and fail-closed. Add focused lit/C++
coverage for stale runtime AVL/VL metadata and contract validation.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/capability-model/capability-contract.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-object-artifact-evidence-closure/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-artifact-default-emission/prd.md`
* `.trellis/workspace/codex/journal-5.md` entries for RVV selected config,
  runtime VL boundary, selected-boundary production route, and op-owned object
  artifact closure.

Likely implementation surface:

* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* a new target-owned RVV runtime-length contract header if it keeps the
  boundary clearer than extending the existing selected-config header only
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/RVV/RVVScalarDispatch.cpp` only if dispatch metadata consumes the
  same selected-config contract surface
* focused tests under `test/Target/RVVMicrokernel/`,
  `test/Target/RVVScalarDispatch/`, and C++ target artifact export tests

## Completion Summary

This round completed the bounded RVV i32 runtime AVL/VL ABI boundary without
adding core RVV branches:

* Added `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h` as the
  target-owned runtime-length submodule for the existing microkernel route.
  It owns the runtime element-count C name, runtime AVL source/role, runtime
  VL source/scope, and descriptor-local element-count metadata helpers.
* Rewired `RVVBinarySelectedConfigContract` to compose that runtime-length
  contract instead of carrying loose runtime length fields.
* Rewired selected-emission planning to append descriptor-local
  `tcrv_rvv.descriptor_element_count` through the runtime-length contract.
* Updated target artifact C++ test helpers to build selected-plan runtime
  metadata from `RVVBinarySelectedConfigContract`, so tests exercise the same
  runtime-length source of truth as production planning/export.
* Added C++ coverage that the runtime-length contract emits the expected
  AVL/VL and descriptor-local metadata and rejects descriptor-only length
  authority outside the bounded range.
* Added lit fail-closed coverage where stale
  `tcrv_rvv.runtime_avl_source = "descriptor-element-count"` fails before RVV
  source output.
* Confirmed no changes under `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, or `lib/Dialect/Exec`.

Migrated slice:

* RVV finite binary `i32-vadd`, `i32m1`, SEW 32, LMUL m1, tail/mask agnostic.
* Runtime AVL authority: `tcrv.exec.runtime_param` role
  `runtime-element-count`, emitted as ABI C parameter `n` or fixture-selected
  `len`.
* Runtime VL authority: `tcrv_rvv.setvl` result consumed by
  `tcrv_rvv.with_vl`.
* Descriptor-local count: `tcrv_rvv.descriptor_element_count`, preserved only
  as bounded component-capacity metadata after typed source authority.

No `ssh rvv` evidence was run. This round refactors and tightens compiler-side
runtime-length contract ownership and fail-closed artifact preflight; it does
not change generated RVV intrinsic semantics, runtime execution, correctness,
or performance claims.

## Checks Run

* `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test tcrv-translate -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir` from `build/test`
* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVMicrokernel/rvv-microkernel-header.mlir Target/RVVMicrokernel/rvv-microkernel-object.mlir Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir Target/RVVMicrokernel/rvv-microkernel-stale-boundary-fails.mlir Target/RVVMicrokernel/rvv-microkernel-missing-policy-fails.mlir Target/RVVMicrokernel/rvv-microkernel-selected-shape-metadata-fails.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir Target/RVVScalarDispatch/rvv-scalar-i64-vmul-dispatch-generic-route.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'` from `build/test`
* Core neutrality scan: `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
* Runtime/descriptor boundary scan: `rg -n "descriptor-element-count|runtime_avl_source|runtime_vl_source|descriptor_compute_authority|element_count" include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp lib/Target/RVV/RVVMicrokernel.cpp test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir`
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-runtime-avl-vl-abi-boundary`
* `python3 ./.trellis/scripts/task.py finish`
* `python3 ./.trellis/scripts/task.py archive 05-13-rvv-runtime-avl-vl-abi-boundary --no-commit`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-13-rvv-runtime-avl-vl-abi-boundary`
* `git diff --cached --check`
