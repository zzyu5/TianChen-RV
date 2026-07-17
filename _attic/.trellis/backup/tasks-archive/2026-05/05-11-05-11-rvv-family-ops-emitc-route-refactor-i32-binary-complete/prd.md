# RVV family ops EmitC route i32 binary complete

## Goal

Continue the archived task
`.trellis/tasks/archive/2026-05/05-11-rvv-family-ops-emitc-route-refactor/`
under the same module owner: RVV descriptor route -> TCRV RVV family ops ->
EmitC route. The continuation closes the bounded direct i32 binary route beyond
the first i32 add slice by requiring i32 sub and i32 mul to be first-class
siblings driven by typed `tcrv_rvv` family microkernel bodies and the
plugin-local EmitC/intrinsic route.

This is not a new architecture direction. It is a continuation of migrating
descriptor-era RVV direct C computation toward:

```text
tcrv.exec envelope
  -> tcrv_rvv family microkernel body
  -> plugin-local EmitC route metadata
  -> RVV intrinsic C/C++
```

## Current Repository Evidence

- HEAD `56a1dcf` already contains a generalized RVV binary family registry
  covering i32 add/sub/mul descriptors, route ids, runtime ABI identities,
  arithmetic op names, and intrinsic prefixes.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` already defines
  `tcrv_rvv.i32_vsub_microkernel`, `tcrv_rvv.i32_vmul_microkernel`,
  `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`.
- `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp` already builds the
  selected binary VL dataflow from the family descriptor/selected-config
  contract instead of an add-only branch.
- `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp` already validates the
  typed i32 body order and rejects arithmetic op/body mismatches against the
  selected descriptor family.
- `lib/Target/RVV/RVVMicrokernel.cpp` already builds
  `RVVEmitCIntrinsicRoute` from the verified dataflow plan and selected
  intrinsic config, then emits the runtime-callable C loop from route steps.
- Existing lit and C++ tests already cover many sub/mul paths, including
  family source export, frontend lowering, object/header routes, stale route
  rejection, and dry-run evidence.

## This Round Scope

Because the implementation is already more advanced than the archived task
brief implied, this round must not duplicate or replace the working C++ route.
Instead, it will add focused acceptance coverage that makes the sub/mul closure
explicit and fail-closed:

- strengthen i32 sub and i32 mul source FileCheck coverage to prove the
  generated C reports the same `RVVEmitCIntrinsicRoute` metadata as add:
  `tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++`,
  source-op list, and arithmetic `emitc.call_opaque` mapping;
- add a negative descriptor/body mismatch case for a vsub descriptor paired
  with a typed vmul body, complementing the existing vmul descriptor + vadd
  body coverage;
- keep all source of compute semantics in typed `tcrv_rvv` microkernel bodies.
  Descriptor/config metadata may only select the bounded family/config,
  intrinsic spelling, route id, runtime ABI identity, and cross-checks.

## Requirements

- i32 sub route consumes verified
  `setvl -> with_vl -> i32_load -> i32_load -> i32_sub -> i32_store`.
- i32 mul route consumes verified
  `setvl -> with_vl -> i32_load -> i32_load -> i32_mul -> i32_store`.
- Generated source for sub includes `__riscv_vsub_vv_i32m*` only through the
  EmitC/intrinsic route metadata and generated route steps.
- Generated source for mul includes `__riscv_vmul_vv_i32m*` only through the
  EmitC/intrinsic route metadata and generated route steps.
- Mismatched descriptor/body pairs fail closed before source export, including
  vsub descriptor + vmul body.
- Selected vector shape and runtime AVL/VL layering remain preserved:
  hardware facts are capability/profile facts; SEW/LMUL/policy are
  compile-time selected config; AVL/VL are runtime SSA/control; element_count
  is descriptor-local.
- No core orchestration pass gains RVV semantic branches.
- Python remains tooling-only.

## Acceptance Criteria

- [x] `test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir` proves the
      sub body feeds `RVVEmitCIntrinsicRoute` and maps the arithmetic step to
      `__riscv_vsub_vv_i32m1`.
- [x] `test/Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir` proves the
      mul body feeds `RVVEmitCIntrinsicRoute` and maps the arithmetic step to
      `__riscv_vmul_vv_i32m1`.
- [x] Negative coverage proves a vsub descriptor paired with a vmul typed body
      fails closed before any RVV source output.
- [x] Focused build targets for the touched route/tests pass.
- [x] Focused lit tests for sub/mul source export and e2e dry-run pass.
- [x] `git diff --check` passes.
- [x] Trellis task validation passes.
- [x] `check-tianchenrv` runs if practical after focused checks pass.

## Non-Goals

- No MLIR vector, LLVM scalable-vector, inline-asm, or backend patch route.
- No new high-level lowering owner.
- No descriptor expansion as the main result.
- No generic RVV backend claim.
- No dtype/LMUL matrix expansion beyond the already bounded i32 binary direct
  route proof.
- No Python compiler-core implementation.
- No helper-only, smoke-only, spec-only, report-only, or broad matrix closeout.
- No runtime, correctness, throughput, latency, or performance claim unless
  fresh `ssh rvv` evidence is collected for the exact slice.

## Minimal Validation

- Build focused targets:
  `TianChenRVRVVTarget`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run the closest focused C++ tests:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`.
- Run focused lit:
  `Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir`,
  `Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir`,
  `Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir`,
  `Scripts/rvv-microkernel-e2e.test`.
- Run direct dry-run for i32-vmul through `scripts/rvv_microkernel_e2e.py`.
- Run `git diff --check`.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-05-11-rvv-family-ops-emitc-route-refactor-i32-binary-complete`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` if practical.

## Completion Boundary

This task may be finished when the sub/mul route is demonstrably family-shaped
at the checked surfaces above, descriptor/body mismatches fail before export,
and no source/runtime/performance claims exceed the local compiler/export
evidence collected in this round.
