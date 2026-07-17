# RVV RHS Broadcast Memory-Form Executable Route

## Goal

Add focused executable evidence for the RVV plugin-owned RHS broadcast memory
form on the current explicit selected-body i32m1 arithmetic surface. The route
must start from a selected `tcrv.exec` RVV variant containing one explicit
`tcrv_rvv.i32_load` for lhs, one explicit `tcrv_rvv.i32_broadcast_load` for
rhs, and one typed `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, or
`tcrv_rvv.i32_mul`, then flow through the RVV provider-built
`TCRVEmitCLowerableRoute`, neutral common EmitC/target bundle export, and real
`ssh rvv` runtime/correctness evidence.

## Current Facts

- Current HEAD at task creation is `335cc1e`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-compare-select-selected-body-executable-route/prd.md`
  proved explicit compare/select selected-body object/header ABI evidence on
  real `ssh rvv`.
- `tcrv_rvv.i32_broadcast_load` already exists as a typed RVV dialect op. Its
  verifier requires an explicit RHS runtime ABI buffer operand and rejects
  dataflow-local descriptor/config authority.
- The RVV provider already has bounded support for RHS broadcast load on
  add/sub/mul: it records `RVVSelectedBodyMemoryForm::RHSBroadcastLoad`,
  rejects compare/select broadcast, and maps the RHS broadcast through the
  RVV-owned `__riscv_vmv_v_x_i32m1` intrinsic in the provider route.
- There is no focused `test/Target/RVV/explicit-selected-body-artifact-*`
  fixture or generated bundle ABI evidence path proving the RHS broadcast
  memory form is executable.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently defaults to add/sub/mul
  vector RHS fixtures and has explicit compare/select support; this task may
  extend it for explicit broadcast selected-body fixtures. The script remains
  evidence tooling only.

## Requirements

1. Add focused explicit selected-body target fixtures for RHS broadcast
   arithmetic where rhs is produced by `tcrv_rvv.i32_broadcast_load` and
   consumed by `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and
   `tcrv_rvv.i32_mul`.
2. The fixtures must produce supported emission-plan metadata and target
   header output through the existing runtime ABI names:
   `rvv-i32m1-add-callable-c-abi.v1`,
   `rvv-i32m1-sub-callable-c-abi.v1`, and
   `rvv-i32m1-mul-callable-c-abi.v1`.
3. Fixture checks must prove broadcast-specific metadata such as
   `tcrv_rvv.memory_form = "rhs-broadcast-load"` or an equivalent
   provider-derived metadata field, while still checking the typed compute op.
4. Extend the generated bundle ABI evidence runner only enough to support
   explicit selected-body RHS broadcast add/sub/mul fixtures.
5. The evidence runner must verify selected variant, route metadata, typed
   compute metadata, ordered `lhs,rhs,out,n` ABI parameters, runtime AVL
   metadata, generated object/header coherence, and remote correctness.
6. Broadcast semantics must come only from `tcrv_rvv.i32_broadcast_load`, its
   explicit RHS ABI value, the typed arithmetic consumer, config, policy,
   memory form, and runtime AVL facts. Route ids, artifact names, ABI names,
   descriptors, test names, and common EmitC/export code are mirrors only.
7. Unsupported broadcast shapes, especially compare/select broadcast and
   missing or wrongly bound RHS runtime ABI values, must fail closed before
   artifact construction. If existing negative coverage is insufficient for
   touched behavior, add focused lit coverage.
8. Common EmitC/materializer/target export must remain neutral. Any needed
   repair belongs in RVV dialect/provider/fixture/evidence code.
9. Collect real `ssh rvv` compile/link/run evidence for bounded nontrivial
   counts `7,16,23`. If remote access or toolchain blocks the run, record the
   exact blocker and do not claim executable status.

## Acceptance Criteria

- [ ] Explicit RHS broadcast target fixtures exercise selected
      `tcrv.exec` / `tcrv_rvv` bodies for add, sub, and mul through emission
      planning and target header export.
- [ ] Each fixture checks provider-derived selected-body operation,
      typed compute op, and RHS broadcast memory-form metadata.
- [ ] `scripts/rvv_generated_bundle_abi_e2e.py` supports a bounded explicit
      RHS broadcast mode for add/sub/mul without changing default vector RHS
      evidence behavior.
- [ ] Generated bundle evidence records selected variants, typed compute ops,
      route metadata, `rhs-broadcast-load` memory form, ordered
      `lhs,rhs,out,n` runtime ABI parameters, runtime AVL metadata, and the
      external C ABI harness consuming only the generated header and object.
- [ ] Real `ssh rvv` evidence passes counts `7,16,23` for broadcast add/sub/mul,
      or the exact blocker is recorded without an executable claim.
- [ ] Focused negative coverage proves unsupported compare/select broadcast or
      missing/wrong RHS broadcast binding fails before artifact construction,
      unless existing tests already cover the touched behavior.
- [ ] Focused build/test checks for touched RVV plugin/export/script behavior
      pass.
- [ ] Bounded residue scans over touched files/artifacts show no restored
      descriptor/direct-C/source-export path, no source-front-door default
      authority, no route/artifact metadata authority, no old `RVVI32M1*`
      route-table authority, and no common EmitC RVV semantic branch.

## Non-Goals

- No new dtype, new LMUL, new source shapes, reductions, conversions,
  broadcast compare/select support, high-level Linalg/Vector/StableHLO
  lowering, one-intrinsic wrapper dialects, or pre-realized broadcast bodies.
- No extension of legacy `RVVI32M1*` tables as architecture and no finite route
  ID authority. Retained i32m1 labels are allowed only as ordinary mirrors
  after typed body validation.
- No common EmitC/materializer/target export RVV semantic branching.
- No source-front-door default authority restoration.
- No dashboard, broad smoke matrix, report-only completion, or helper-only
  completion.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.

## Validation Plan

- Build focused targets:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run focused C++ tests if touched:
  `./build/bin/tianchenrv-rvv-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit for new RHS broadcast target fixtures and any negative
  broadcast coverage added in this task.
- Run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- Run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Run local dry-run generated bundle evidence for broadcast add/sub/mul with
  counts `7,16,23`.
- Run real `ssh rvv` generated bundle ABI evidence for broadcast add/sub/mul
  with counts `7,16,23`.
- Run bounded residue scans over touched runner/test files and relevant RVV
  plugin/common EmitC surfaces.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-19-rvv-rhs-broadcast-memory-form-executable-route`.
- Run `git diff --check`.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-compare-select-selected-body-executable-route/prd.md`

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`

## Implementation Notes

- Added explicit selected-body RHS broadcast target fixtures:
  - `test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-broadcast-sub.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-broadcast-mul.mlir`
- Each fixture starts from selected `tcrv.exec` / `tcrv_rvv` IR with one lhs
  `tcrv_rvv.i32_load`, one rhs `tcrv_rvv.i32_broadcast_load`, one typed
  arithmetic consumer, and one store. The emission-plan checks require
  provider-derived `tcrv_rvv.memory_form = "rhs-broadcast-load"`.
- Updated `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` so selected-body
  route artifact metadata mirrors the provider-derived memory form. This is
  derived from the selected typed body analysis, not from route IDs or artifact
  names.
- Fixed the latent RHS broadcast route by extending the neutral common EmitC
  materializer in
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` to materialize safe
  `identifier[constant]` operand expressions as `emitc.subscript` plus
  `emitc.load`. The generated broadcast source now lowers the explicit RHS ABI
  pointer to a scalar value such as `const int32_t v11 = v2[0];` before calling
  `__riscv_vmv_v_x_i32m1`.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with
  `--rhs-broadcast-selected-body` mode for add/sub/mul only. Default vector RHS
  and explicit compare/select modes remain unchanged.
- Added dry-run script lit coverage for RHS broadcast generated bundle ABI
  evidence:
  `test/Scripts/rvv-generated-bundle-abi-e2e-rhs-broadcast-dry-run.test`.
- Added fail-closed negative coverage for compare/select mixed with RHS
  broadcast:
  `test/Conversion/EmitC/rvv-first-slice-materialization-broadcast-cmp-select-negative.mlir`.
  Wrong RHS runtime binding was already covered by
  `test/Dialect/RVV/dataflow.mlir`.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  selected-body RHS broadcast memory-form contract.

## Evidence

- Local dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/dev-rhs-broadcast-dry`.
- Real `ssh rvv` artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-rhs-broadcast-selected-body-executable-route`.
- Evidence records for add/sub/mul show:
  - selected variants
    `explicit_selected_body_rvv_i32_broadcast_add`,
    `explicit_selected_body_rvv_i32_broadcast_sub`, and
    `explicit_selected_body_rvv_i32_broadcast_mul`;
  - typed compute ops `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and
    `tcrv_rvv.i32_mul`;
  - `tcrv_rvv.memory_form = "rhs-broadcast-load"`;
  - ordered runtime ABI parameters `lhs,rhs,out,n`;
  - runtime AVL metadata `runtime_abi:n`;
  - external C ABI harnesses consuming only the generated header and object.
- Generated C++ for RHS broadcast materializes the explicit RHS ABI scalar
  before the RVV broadcast intrinsic:

```text
const int32_t v11 = v2[0];
vint32m1_t v12 = __riscv_vmv_v_x_i32m1(v11, v8);
```

- Real `ssh rvv` output:

```text
add case n=7 ok
add case n=16 ok
add case n=23 ok
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23
sub case n=7 ok
sub case n=16 ok
sub case n=23 ok
tcrv_rvv_generated_bundle_abi_sub_ok counts=7,16,23
PASS op=sub counts=7,16,23
mul case n=7 ok
mul case n=16 ok
mul case n=23 ok
tcrv_rvv_generated_bundle_abi_mul_ok counts=7,16,23
PASS op=mul counts=7,16,23
```
- Bounded residue scans:
  - touched production files
    `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` and
    `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` contain no
    `descriptor`, `direct-C`, `source-export`, source-front-door, or
    `RVVI32M1Arithmetic*` route-table authority strings;
  - common EmitC materializer scan for `rvv`, `tcrv_rvv`, `__riscv`, and
    `i32_` returned no matches;
  - real RHS broadcast artifact tree contains no descriptor/direct-C/
    source-export/source-front-door residue.

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --rhs-broadcast-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id dev-rhs-broadcast-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --rhs-broadcast-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-rhs-broadcast-selected-body-executable-route --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-broadcast-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-broadcast-mul.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-broadcast-cmp-select-negative.mlir`
  passed 4/4.
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Scripts/rvv-generated-bundle-abi-e2e-rhs-broadcast-dry-run.test ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test ../test/Scripts/rvv-generated-bundle-abi-e2e-cmp-select-dry-run.test`
  passed 4/4.
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Conversion/EmitC/rvv-first-slice-materialization.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-sub.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-mul.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-broadcast-cmp-select-negative.mlir`
  passed 5/5.
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/explicit-selected-body-artifact-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir ../test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir ../test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-broadcast-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-broadcast-mul.mlir`
  passed 7/7.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-rvv-rhs-broadcast-memory-form-executable-route`
- `git diff --check`

## Status

Completed locally with real `ssh rvv` runtime/correctness evidence for
explicit selected-body RHS broadcast add/sub/mul generated artifact bundles.
Ready for Trellis archive and one coherent commit.
