# RVV Microkernel Family Generalization

## Goal

Make the selected RVV C-intrinsic executable route materially more general than the previous single hard-coded i32 vector-add slice. The RVV plugin should own a small typed microkernel-family/config layer that carries the existing i32 add case and one additional concrete i32 subtract case through dialect verification, plugin materialization, emission planning, and target-owned riscv_vector.h C source/header/object export.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD is expected to be `6f4f73e chore(task): archive rvv bundle ssh evidence`.
- Previous round was evidence closure only and did not modify compiler/source/test/spec code.
- The accepted previous runtime claim is bounded to RVV i32-vadd target-artifact bundle external C ABI correctness.
- This round must not be evidence-only, task-only, docs-only, smoke-only, wrapper-only, or i32-vadd-only.
- High-level linalg/stablehlo/tosa lowering is not a prerequisite for this RVV plugin step.
- Core `tcrv.exec` must remain compute-free and must not gain RVV-specific branches.
- Python must remain runner/evidence/tooling only, not compiler internals.

## Requirements

- Preserve the existing i32 vector-add syntax and route.
- Introduce one additional concrete supported family member: i32 vector-subtract.
- Keep concrete RVV computation plugin-local under `tcrv_rvv` and target-owned C export.
- Refactor just enough that family semantics are represented by typed/checked RVV family metadata rather than scattered single-demo i32-vadd assumptions.
- Carry the family member through RVV dialect verification, plugin proposal/materialization/emission planning, selected boundary metadata, and target export.
- Emit riscv_vector.h C intrinsics from the target exporter, including `__riscv_vadd_vv_i32m1` for add and `__riscv_vsub_vv_i32m1` for subtract.
- Preserve parameter layering: hardware facts as capabilities; SEW/LMUL/policy as compile-time RVV config; runtime n/AVL/VL as SSA/control/ABI; descriptor-local `element_count` as bounded descriptor metadata.
- Avoid changes to core passes unless required as a generic plugin interface evolution.
- Avoid weakening existing tests or evidence-producing coverage.

## Acceptance Criteria

- Existing i32-vadd RVV dialect/plugin/export tests still pass.
- i32-vsub parses/verifies as `tcrv_rvv` IR.
- At least one structurally wrong i32-vsub use is rejected by verifier/export logic.
- Plugin materialization/emission planning carries new family metadata for selected RVV paths without core RVV-specific branches.
- Target C export prints the expected riscv_vector.h intrinsic sequence for i32-sub and keeps the i32-add route compatible.
- Header/object/generic target export continue to work for existing supported routes.
- Required local commands pass:
  - `git diff --check`
  - `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Out Of Scope

- Generic RVV lowering.
- MLIR vector lowering or LLVM intrinsic lowering.
- i64 vector-add unless naturally trivial after inspection.
- Performance claims.
- Dynamic runtime integration.
- Broad RVV kernel coverage.
- New high-level frontend dependency.
- Python implementation of compiler internals.
- New `tcrv.exec` compute operations.

## Technical Notes

- Required repository inspection and required files were read before source edits.
- Chosen additional case: i32 vector-subtract. It shares the existing runtime ABI shape and RVV i32/m1 load/store/control configuration, so the change exercises a real microkernel-family generalization without broadening ABI or element-type scope.
- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
