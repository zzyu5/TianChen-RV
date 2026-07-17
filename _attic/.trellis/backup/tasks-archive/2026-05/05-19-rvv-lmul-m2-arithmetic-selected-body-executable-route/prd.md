# RVV LMUL m2 Arithmetic Selected-Body Executable Route

## Goal

Add a focused executable route for the RVV plugin-owned LMUL m2 configuration
on the existing explicit i32 arithmetic selected-body surface. The route must
start from a selected `tcrv.exec` RVV variant containing typed
`!tcrv_rvv.i32m2` vector-RHS add/sub/mul dataflow plus `lmul = "m2"` on
`tcrv_rvv.setvl` / `tcrv_rvv.with_vl`, then flow through the RVV
provider-derived `TCRVEmitCLowerableRoute`, neutral common EmitC/target bundle
export, and real `ssh rvv` runtime/correctness evidence.

## Current Facts

- Current HEAD at task creation is `79c559e`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-rhs-broadcast-memory-form-executable-route/prd.md`
  proved RHS broadcast add/sub/mul executable evidence on `ssh rvv`.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and
  `lib/Dialect/RVV/IR/RVVDialect.cpp` already expose and verify
  `!tcrv_rvv.i32m2` for load/add/sub/mul/store dataflow when the enclosing
  `setvl` / `with_vl` config is SEW32 LMUL m2.
- The current RVV provider specialization table in
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` is still proven only for
  LMUL m1. LMUL m2 reaches an unsupported route-specialization diagnostic.
- Existing target/export and plugin tests include m2 as negative coverage.
  This task must update those expectations only where the selected-body m2
  arithmetic route is genuinely supported.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently verifies generated
  bundle ABI evidence for explicit selected-body m1, pre-realized m1, RHS
  broadcast m1, and compare/select m1. It remains evidence tooling only.

## Requirements

1. Add focused explicit selected-body target fixtures for vector-RHS LMUL m2
   `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`.
2. The fixtures must use `!tcrv_rvv.i32m2` dataflow and `lmul = "m2"` config
   on both `setvl` and `with_vl`. Fixture checks must prove provider-derived
   `tcrv_rvv.lmul = "m2"` mirror metadata, typed compute op metadata, ordered
   ABI parameters, runtime AVL metadata, and target header output.
3. Extend the RVV provider so vector type, C vector type, mask type where
   relevant, `vsetvl`, load/store, arithmetic intrinsic spelling, runtime ABI
   metadata, and artifact metadata are derived from the typed selected-body
   config/body facts for LMUL m2.
4. LMUL m2 support is limited to vector-RHS add/sub/mul in this round.
   Compare/select m2, broadcast m2, reductions, conversions, and new dtypes
   remain unsupported.
5. Route labels, ABI names, artifact route names, old m1 route-family names,
   and generated artifact metadata are mirror labels only. They must not decide
   operation semantics, dtype, LMUL, policy, memory form, or intrinsic mapping.
6. Mismatched m1/m2 dataflow, missing or contradictory LMUL config, unsupported
   memory forms, stale candidate metadata, or metadata-only route attempts must
   fail closed before artifact construction.
7. Generated bundle evidence must verify selected variants, `tcrv_rvv.lmul =
   "m2"` mirror metadata, typed compute ops, vector-RHS memory form, ordered
   `lhs,rhs,out,n` ABI parameters, runtime AVL metadata, object/header
   coherence, and the external harness consuming only the generated header and
   object.
8. Common EmitC/materializer/target export must remain neutral. Any needed
   semantic repair belongs in RVV dialect/provider/fixture/evidence code.
9. Collect real `ssh rvv` compile/link/run evidence for bounded nontrivial
   counts `7,16,23`. If remote access or toolchain blocks the run, record the
   exact blocker and do not claim executable status.

## Acceptance Criteria

- [x] Explicit LMUL m2 target fixtures exercise selected `tcrv.exec` /
      `tcrv_rvv` add, sub, and mul bodies through emission planning and target
      header export.
- [x] Provider route description validates and mirrors SEW32, LMUL m2,
      agnostic policy, `!tcrv_rvv.i32m2`, `vint32m2_t`, `__riscv_vsetvl_e32m2`,
      `__riscv_vle32_v_i32m2`, `__riscv_vse32_v_i32m2`, and the matching
      add/sub/mul m2 intrinsics from selected typed body facts.
- [x] Mismatched dataflow/config or stale m1 metadata fails closed before
      artifact construction.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` supports a bounded explicit
      LMUL m2 selected-body mode for add/sub/mul without changing default m1,
      broadcast, pre-realized, compare/select, or source-seed behavior.
- [x] Generated bundle evidence records selected variants, typed compute ops,
      vector-RHS memory form, `tcrv_rvv.lmul = "m2"`, ordered runtime ABI
      parameters, runtime AVL metadata, and external C ABI harness boundaries.
- [x] Real `ssh rvv` evidence passes counts `7,16,23` for m2 add/sub/mul, or
      the exact blocker is recorded without an executable claim.
- [x] Focused build/test checks for touched RVV plugin/export/script behavior
      pass.
- [x] Bounded residue scans over touched files/artifacts show no restored
      descriptor/direct-C/source-export path, no source-front-door default
      authority, no route/artifact metadata authority, no expanded
      `RVVI32M1*` table as architecture, and no common EmitC RVV semantic
      branch.

## Implementation Result

- Added `RVVSelectedBodyConfigVLContract` as the shared selected-body
  config/VL contract surface and added the explicit
  `rvv-i32m2-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1` contract.
- Extended the RVV provider's selected-body arithmetic mapping to derive the
  m2 vector type, C vector type, vsetvl spelling, vector load/store spelling,
  and add/sub/mul intrinsics from selected `!tcrv_rvv.i32m2` body/config facts.
- Kept m2 support limited to vector-RHS add/sub/mul. RHS broadcast m2 and other
  Stage 2 classes remain fail-closed.
- Added neutral target-export support for candidate-derived dynamic metadata
  validation. Common target export still does not contain RVV, LMUL, dtype, or
  intrinsic semantics.
- Updated RVV target support bundle metadata evidence so generated headers
  carry `tcrv_rvv.lmul = "m2"` and related config metadata as mirrors of the
  provider-derived selected-body route.
- Added focused target fixtures:
  - `test/Target/RVV/explicit-selected-body-artifact-m2-add.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-m2-sub.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-m2-mul.mlir`
- Added mismatch/stale metadata coverage in RVV dialect, plugin, target export,
  and script tests.
- Added `--lmul-m2-selected-body` generated-bundle evidence mode for bounded
  m2 add/sub/mul runtime validation.

## Validation Result

- Passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- Passed:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`
- Passed:
  `./build/bin/tianchenrv-target-artifact-export-test`
- Passed focused lit from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/explicit-selected-body-artifact-m2-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-m2-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-m2-mul.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir ../test/Dialect/RVV/dataflow.mlir ../test/Scripts/rvv-generated-bundle-abi-e2e-lmul-m2-dry-run.test`
- Passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- Passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Passed local generated-bundle dry run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --lmul-m2-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id dev-lmul-m2-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Passed real `ssh rvv` generated-bundle runtime/correctness evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --lmul-m2-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-lmul-m2-selected-body-executable-route --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- Real `ssh rvv` output included:
  `PASS op=add counts=7,16,23`,
  `PASS op=sub counts=7,16,23`, and
  `PASS op=mul counts=7,16,23`.
- Runtime evidence directory:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-lmul-m2-selected-body-executable-route`
- Bounded residue scans over touched files and generated artifacts found no
  descriptor/direct-C/source-export/source-front-door restoration, no old
  `RVVI32M1*` route-table growth, no route/artifact metadata authority, and no
  RVV semantic branch in common target export.

## Non-Goals

- No new dtype/SEW beyond i32/SEW32.
- No reductions, conversions, compare/select m2, broadcast m2, pre-realized m2
  bodies, high-level Linalg/Vector/StableHLO lowering, source-front-door
  default authority, or one-intrinsic wrapper dialects.
- No common EmitC/materializer/target export RVV semantic branching.
- No dashboard, broad smoke matrix, report-only completion, helper-only
  completion, descriptor-driven computation, direct-C/source-export route
  restoration, or compatibility wrapper preserving old i32 route authority.

## Validation Plan

- Build focused targets:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-rvv-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit for new explicit m2 target fixtures and any negative
  mismatch coverage added in this task.
- Run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- Run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Run local dry-run generated bundle evidence for m2 add/sub/mul with counts
  `7,16,23`.
- Run real `ssh rvv` generated bundle ABI evidence for m2 add/sub/mul with
  counts `7,16,23`.
- Run bounded residue scans over touched runner/test files and relevant RVV
  plugin/common EmitC surfaces.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-19-rvv-lmul-m2-arithmetic-selected-body-executable-route`.
- Run `git diff --check`.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-rhs-broadcast-memory-form-executable-route/prd.md`

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
- `test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `test/Target/TargetArtifactExportTest.cpp`
