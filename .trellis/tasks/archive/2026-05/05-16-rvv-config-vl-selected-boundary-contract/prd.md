# RVV config/VL selected-boundary contract

## Goal

Make the bounded RVV i32m1 arithmetic selected path consume one explicit,
plugin-owned config and runtime VL contract instead of relying on scattered
SEW/LMUL/policy checks in dialect verification, EmitC route construction, and
target artifact export. The existing add/sub/mul i32m1 routes must continue to
materialize through extension-family ops, common EmitC, and selected target
artifact export.

## What I already know

- Current HEAD is `c70febc target(rvv): generalize i32m1 emitc arithmetic routes`.
- Worktree was clean before task creation.
- Existing add/sub/mul i32m1 route construction lives in
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` already carry compile-time `sew`,
  `lmul`, and `policy` attributes while AVL/VL are SSA values.
- The RVV dialect currently accepts SEW32 with LMUL `m1` or `m2`; this round
  must not add new execution families or artifact routes.
- The selected artifact front door already routes object/header/bundle export
  through plugin-owned RVV route builders.

## Requirements

- Define one coherent RVV first-slice config/VL contract for the existing
  i32m1 arithmetic family.
- Keep SEW, LMUL, tail policy, and mask policy compile-time RVV config.
- Keep runtime element count, AVL, and produced VL as runtime SSA or runtime
  ABI values.
- Route construction for add/sub/mul must consume the same contract and fail
  closed for stale route/op/config combinations.
- Dialect verification must fail closed for missing policy, wrong SEW/LMUL,
  setvl/with_vl config mismatch, and AVL/VL misuse.
- Target/export code must remain a selected artifact consumer and must not
  gain RVV config semantics, intrinsic names, or header names in common target
  code.

## Acceptance Criteria

- [x] Existing add/sub/mul RVV i32m1 EmitC materialization tests still pass.
- [x] Existing add/sub/mul selected object/header/bundle export tests still
      pass through the common selected artifact front door.
- [x] New or repaired negative coverage proves mismatched setvl vs with_vl
      config, missing policy, wrong SEW/LMUL for i32m1 route export, stale
      selected route/op combinations, and AVL/VL misuse fail closed.
- [x] No new SEW, LMUL, dtype, mask, or arithmetic op family is added.
- [x] Common target/export code remains RVV-agnostic; RVV intrinsic/header
      names remain local to RVV-owned code/tests.
- [x] If generated C/header/object bytes change, rerun ssh rvv evidence for
      every affected arithmetic route. If they do not change, include an
      artifact/ABI comparison rationale.

## Out of Scope

- New RVV dtype, SEW, LMUL, mask, op family, i32m2 execution route, generic RVV
  lowering, MLIR vector lowering, high-level tensor lowering, TensorExt/IME
  implementation, performance matrices, descriptor registries, direct C
  compute printers, Python compiler-core logic, GCC-default routes,
  compatibility wrappers, or extension-specific branches in common/core
  orchestration.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Code inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Support/RuntimeABI.h`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`, and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Focus tests inspected:
  `test/Dialect/RVV/setvl.mlir`,
  `test/Dialect/RVV/with-vl.mlir`,
  `test/Dialect/RVV/dataflow.mlir`,
  `test/Conversion/EmitC/rvv-first-slice-materialization*.mlir`, and
  `test/Target/RVV/i32m1-*artifact*.mlir`.
- The implementation should prefer a shared C++ helper/model inside RVV-owned
  dialect/plugin code over repeating raw SEW32/LMUL/policy literals in each
  route.

## Definition of Done

- Focused build targets for RVV dialect/plugin/target compile.
- Focused lit/FileCheck tests for changed RVV dialect, EmitC, and target export
  behavior pass.
- `check-tianchenrv` is run if practical; otherwise the reason is recorded.
- Task status and workspace journal are updated truthfully.
- Completed work is committed as one coherent commit, or the exact blocker and
  next continuation point are recorded.

## Completion Evidence

- Build:
  `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVEmitCRouteProvider TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-target-artifact-export-test -j2`
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='(Dialect/RVV/(setvl|with-vl|dataflow)\.mlir|Conversion/EmitC/rvv-first-slice-(materialization(|-sub|-mul|-negative|-missing-abi|-missing-store)|config-vl-contract-negative|vl-contract-negative)\.mlir|Target/RVV/i32m1-(add-object-artifact|selected-dispatch-artifact|selected-path-sibling-artifact|artifact-ambiguous-selected|artifact-unselected-multivariant|object-unsupported-shape|sub-selected-dispatch-artifact|mul-selected-dispatch-artifact|object-stale-route-op)\.mlir)'`
  from `build/test`.
- C++ tests:
  `./build/bin/tianchenrv-rvv-dialect-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Full check:
  `cmake --build build --target check-tianchenrv -j2` passed 90/90 tests.
- Changed-surface scan:
  common target code scan for `riscv_vector`, `__riscv_`, `i32m1`,
  `rvv-i32m1`, and `tcrv_rvv` outside `Target/RVV` returned no matches.
- Artifact/ABI rationale:
  the target export and route payload code that adds headers, type mappings,
  ABI parameter order, EmitC call opaque steps, callable header declarations,
  and object compilation was not changed. The route provider now delegates the
  pre-payload config/VL validation to `RVVConfigContract`; generated
  C/header/object bytes are therefore unchanged, so the prior `c70febc` ssh
  rvv evidence remains applicable.
