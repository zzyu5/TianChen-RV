# RVV compare/select selected-body executable route

## Goal

Add a focused explicit selected-body compare/select RVV artifact fixture and
evidence path. The route must start from a selected `tcrv.exec` RVV variant
that contains an explicit typed `tcrv_rvv.i32_cmp_eq` mask producer and
`tcrv_rvv.i32_select` dataflow consumer, then flow through the RVV
provider-built `TCRVEmitCLowerableRoute`, neutral common EmitC/target bundle
export, and real `ssh rvv` runtime/correctness evidence.

## Current Facts

- Current HEAD at task creation is `7700216`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-pre-realized-arithmetic-executable-abi-gate`
  proved pre-realized add/sub/mul executable ABI evidence on real `ssh rvv`.
- `tcrv_rvv.i32_cmp_eq` and `tcrv_rvv.i32_select` already exist as typed RVV
  dialect ops with verifier checks for same-scope mask/dataflow structure.
- The RVV construction protocol and provider code already contain bounded
  compare/select route labels and provider analysis hooks, but there is no
  public `test/Target/RVV/explicit-selected-body-artifact-*` fixture or
  generated bundle ABI evidence path for compare/select.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently defaults to
  `add/sub/mul` expectations. This task may extend the evidence tooling to
  consume an explicit compare/select selected-body fixture, but the script
  remains evidence tooling only.

## Requirements

1. Add one explicit selected-body target fixture for the bounded
   compare/select route:
   `tcrv_rvv.i32_load(lhs)`, `tcrv_rvv.i32_load(rhs)`,
   `tcrv_rvv.i32_cmp_eq(lhs, rhs)`, `tcrv_rvv.i32_select(mask, lhs, rhs)`,
   and `tcrv_rvv.i32_store(out, selected)`.
2. The fixture must produce supported emission-plan metadata and target header
   output for the compare/select runtime ABI:
   `rvv-i32m1-cmp-select-callable-c-abi.v1`.
3. Extend the generated bundle ABI evidence runner only enough to support
   `--op-kind cmp_select` from the explicit selected-body fixture.
4. The evidence runner must verify selected variant, route metadata,
   typed compute metadata for `tcrv_rvv.i32_select`, generated object/header
   coherence, ordered `lhs,rhs,out,n` ABI parameters, runtime AVL metadata, and
   remote correctness.
5. Compare/select semantics must come only from the typed selected RVV body:
   `tcrv_rvv.i32_cmp_eq`, `tcrv_rvv.i32_select`, explicit mask/data operands,
   SEW/LMUL/policy config, memory form, and runtime ABI values.
6. Unsupported or malformed compare/select mask/dataflow must fail closed
   before artifact construction. If existing dialect/provider verifier
   coverage is insufficient for touched behavior, add focused negative
   coverage.
7. Common EmitC/materializer/target export must remain neutral. Any required
   repair belongs in the RVV plugin/provider/fixture/evidence path.
8. Collect real `ssh rvv` compile/link/run evidence for compare/select with
   bounded nontrivial runtime counts such as `7,16,23`. If remote access or
   toolchain blocks the run, record the exact blocker and do not claim
   executable status.

## Acceptance Criteria

- [x] `test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`
      exercises a selected explicit typed `tcrv_rvv.i32_cmp_eq` +
      `tcrv_rvv.i32_select` body through emission planning and target header
      export.
- [x] The fixture checks supported RVV emission-plan metadata including
      `rvv_selected_body_operation = "cmp_select"`,
      `rvv_selected_body_typed_compute_op = "tcrv_rvv.i32_select"`, and
      runtime ABI name `rvv-i32m1-cmp-select-callable-c-abi.v1`.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` supports
      `--op-kind cmp_select` in explicit selected-body mode and verifies the
      generated object/header bundle using the same IR-backed ABI contract as
      add/sub/mul.
- [x] Generated bundle evidence for compare/select records the selected
      variant, typed compute op, compare/select route metadata, ordered
      `lhs,rhs,out,n` runtime ABI parameters, runtime AVL metadata, and the
      external C ABI harness consuming only the generated header and object.
- [x] Real `ssh rvv` evidence for compare/select passes counts `7,16,23`, or
      the exact blocker is recorded without an executable claim.
- [x] Focused lit/self-test coverage prevents stale arithmetic metadata,
      stale route IDs, descriptor/direct-C/source-export residue, and
      source-front-door/default authority from being confused with this route.
- [x] Focused build/test checks for touched RVV plugin/export/script behavior
      pass.
- [x] Bounded residue scans over touched files/artifacts show no restored
      descriptor/direct-C/source-export path, no source-front-door default
      authority, no route/artifact metadata authority, and no common EmitC RVV
      semantic branch.

## Non-Goals

- No reductions, conversions, broadcast compare/select, new dtype, new LMUL,
  new source shapes, Linalg/Vector/StableHLO frontend work, or one-intrinsic
  wrapper dialects.
- No pre-realized compare/select body in this round unless the explicit
  selected-body route and executable gate are already complete and remain one
  coherent module.
- No common EmitC/materializer/target export RVV semantic branching.
- No source-front-door default authority restoration.
- No dashboard, broad smoke matrix, report-only completion, or helper-only
  completion.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.

## Validation Plan

- Build focused targets:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- Run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Run focused lit for:
  `test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`,
  existing explicit add/sub/mul target fixtures, and script dry-run/self-test
  coverage touched by this task.
- Run local dry-run compare/select bundle evidence with counts `7,16,23`.
- Run real `ssh rvv` compare/select bundle ABI evidence with counts
  `7,16,23`.
- Run bounded residue scans over touched runner/test files and relevant RVV
  plugin/common EmitC surfaces.
- Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-19-rvv-compare-select-selected-body-executable-route`.
- Run `git diff --check`.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-pre-realized-arithmetic-executable-abi-gate/prd.md`

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
- `test/Dialect/RVV/dataflow.mlir`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `test/Target/TargetArtifactExportTest.cpp`

## Implementation Notes

- Added `test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`.
  The fixture starts from explicit selected `tcrv.exec` / `tcrv_rvv` IR and
  uses `tcrv_rvv.i32_cmp_eq` plus `tcrv_rvv.i32_select` as the only
  compare/select authority.
- The fixture compares lhs with itself so the select true branch is observable
  in external ABI evidence while still deriving compare/select semantics from
  typed RVV operands.
- Updated `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` so compare/select
  route operands are mapped from the selected typed body SSA operands. The
  provider now permits bounded compare/select operand combinations over the
  explicit lhs/rhs vector load results and still rejects unsupported operands
  before route construction.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with explicit
  `cmp_select` expectations. Add/sub/mul remain the default op set; `cmp_select`
  must be requested explicitly.
- Added focused dry-run lit coverage in
  `test/Scripts/rvv-generated-bundle-abi-e2e-cmp-select-dry-run.test`.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the durable
  bounded i32 compare/select selected-body route contract.

## Evidence

- Local dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-cmp-select-selected-body-executable-route-dry`.
- Real `ssh rvv` artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-rvv-cmp-select-selected-body-executable-route`.
- Compare/select evidence records:
  - selected variant `explicit_selected_body_rvv_i32_cmp_select`;
  - typed compute op `tcrv_rvv.i32_select`;
  - selected-body operation metadata `cmp_select`;
  - runtime ABI name `rvv-i32m1-cmp-select-callable-c-abi.v1`;
  - ordered runtime ABI parameters `lhs,rhs,out,n`;
  - runtime AVL metadata `runtime_abi:n`;
  - external C ABI harness consuming only the generated header and object.
- Remote run output:

```text
cmp_select case n=7 ok
cmp_select case n=16 ok
cmp_select case n=23 ok
tcrv_rvv_generated_bundle_abi_cmp_select_ok counts=7,16,23
PASS op=cmp_select counts=7,16,23
```

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-cmp-select-selected-body-executable-route-dry --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-rvv-cmp-select-selected-body-executable-route --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir ../test/Target/RVV/explicit-selected-body-artifact-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
  passed 4/4.
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-cmp-select-dry-run.test ../test/Scripts/rvv-generated-bundle-abi-e2e-selected-body-dry-run.test`
  passed 3/3.
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Dialect/RVV/dataflow.mlir`
  passed 1/1.
- Real evidence artifact residue scan under the compare/select op directory
  found no `legacy-rvv-source-seed`, source-artifact front-door,
  `rvv_arithmetic_op`, descriptor, direct-C, or source-export residue.
- Touched production file scan found no `source-artifact-front-door-pipeline`,
  `rvv_arithmetic_op`, `RVVI32M1ArithmeticRouteSpec`,
  `RVVI32M1ArithmeticSlice`, or `collectRVVI32M1ArithmeticSlice`.
- Common EmitC materializer scan found no `rvv`, `tcrv_rvv`, `__riscv`, or
  `i32_` semantic branch text in
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-rvv-compare-select-selected-body-executable-route`
- `git diff --check`

## Status

Completed locally with real `ssh rvv` runtime/correctness evidence for the
explicit selected-body compare/select generated artifact. Ready for Trellis
archive and one coherent commit.
