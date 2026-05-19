# Stage2 RVV selected-body realization for generic add

## Goal

Make the bounded Stage 2 RVV plugin-local selected-body realization path
truthful for one explicit generic typed add body. The selected RVV variant must
carry explicit typed pre-realized facts for operation, config, memory form, and
runtime ABI values; the RVV plugin must consume those facts into realized
`tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, `tcrv_rvv.load`,
`tcrv_rvv.binary`, and `tcrv_rvv.store` structure before the existing RVV
provider constructs a route.

This task is not a frontend task, not a new operation-coverage expansion, and
not a legacy i32 compatibility task. It is the selected-boundary-to-typed-body
handoff for the corrected generic typed RVV surface.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `3b75ce32 rvv: add typed strided memory route`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-19-stage2-rvv-selected-body-generic-add`.
- Relevant specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Stage 1 guardrails remain active: do not reintroduce positive
  `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-seed/source-front-door,
  descriptor, artifact-name, or exact intrinsic-spelling route authority.
- Current HEAD already contains the generic pre-realized surface
  `tcrv_rvv.typed_binary_pre_realized_body`, not the older
  `tcrv_rvv.i32_binary_pre_realized_body` surface.
- Current `RVVExtensionPlugin::materializeSelectedLoweringBoundary` already
  has a plugin-local realization hook that rewrites a valid pre-realized typed
  binary body into generic realized `setvl/with_vl/load/binary/store`
  structure, then validates that realized body through the existing provider.
- Existing positive fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir` proves the
  generic add handoff reaches emission-plan metadata and target header export.
- Existing negative fixture
  `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`
  covers unsupported op kind, unsupported memory form, wrong config, mixed
  pre-realized/realized bodies, wrong runtime ABI role, missing runtime
  n/AVL definition, and stale authority metadata.

## Requirements

1. Keep the module scope to one explicit pre-realized generic add handoff.
2. The pre-realized input must carry explicit operation kind, memory form,
   SEW, LMUL, policy, lhs/rhs/out runtime ABI values, and runtime `n`/AVL.
3. RVV plugin-owned realization must produce real `tcrv_rvv` structure before
   provider route construction.
4. The existing RVV provider must remain the route authority and consume only
   the realized typed body.
5. Common EmitC/export must remain neutral and must not infer RVV semantics.
6. Missing or inconsistent explicit semantics/config/runtime facts must fail
   closed before route/artifact construction.
7. Preserve existing add/sub/mul, RHS broadcast, compare/select, reduction,
   masked add, macc, LMUL m2, strided memory, and Stage1 residue behavior.
8. If current HEAD already satisfies the bounded behavior, do not add an
   unrelated RVV coverage class. Validate and add only focused missing coverage
   if a real acceptance gap is found.

## Acceptance Criteria

- [x] PRD and Trellis task context match the bounded generic-add realization
      goal and do not drift into source frontend or broad Stage2 coverage.
- [x] Positive materialization evidence proves
      `tcrv_rvv.typed_binary_pre_realized_body` for add is consumed and the
      resulting body contains `setvl`, `with_vl`, `load`, `binary {kind =
      "add"}`, and `store`.
- [x] Positive emission-plan / artifact evidence proves the realized body is
      handed to the existing provider and reaches provider-derived generic
      route metadata.
- [x] Negative tests fail closed for missing or inconsistent op kind,
      dtype/config, memory form, runtime `n`/AVL, ABI role binding, stale
      authority metadata, and mixed pre-realized plus already-realized bodies.
- [x] Existing explicit selected-body routes remain passing.
- [x] Bounded active-authority scan confirms no positive legacy RVV route
      authority or source-front-door/source-seed path is reintroduced.
- [x] Focused build/lit/script/C++ checks for touched RVV plugin/provider,
      target artifact, and pre-realized generated-bundle paths pass.
- [x] `ssh rvv` evidence is collected only if this round makes a new
      runtime/correctness/executable claim.

## Implementation Results

- Confirmed current HEAD already has the production-path generic typed
  selected-body realization surface:
  `tcrv_rvv.typed_binary_pre_realized_body`.
- Confirmed `RVVExtensionPlugin::materializeSelectedLoweringBoundary` consumes
  the pre-realized body and realizes it into generic `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, `tcrv_rvv.load`, `tcrv_rvv.binary`, and
  `tcrv_rvv.store` operations before provider route construction.
- Confirmed the positive add fixture reaches provider-derived route metadata:
  `rvv-generic-binary-add-emitc-route`,
  `rvv_selected_body_typed_compute_op = tcrv_rvv.binary`, and runtime ABI
  `rvv-generic-binary-add-callable-c-abi.v1`.
- Added one missing focused negative fixture in
  `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`
  proving a pre-realized body with no `op_kind` attribute fails before route or
  artifact construction.
- No production C++ behavior change was needed; the bounded module behavior was
  current in HEAD and the only implementation gap found was negative coverage.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage2-rvv-selected-body-generic-add`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [x] Focused lit passed 6/6:
      `pre-realized-selected-body-artifact-add.mlir`,
      `rvv-pre-realized-selected-body-negative.mlir`,
      explicit selected-body add/sub/mul fixtures, and
      `rvv-generated-bundle-abi-e2e-pre-realized-dry-run.test`.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Add-only local dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-pre-realized-generic-add-dry --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] Dry-run evidence path:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-pre-realized-generic-add-dry`
- [x] `git diff --check`
- [x] Active-authority scan over active RVV include/lib/script/test paths found
      no newly introduced `RVVI32M1`, `rvv-i32m1`,
      `i32_binary_pre_realized_body`, positive finite `tcrv_rvv.i32_*`
      route authority, `!tcrv_rvv.i32m*` route authority, source-seed/source-
      front-door positive RVV authority, descriptor/direct-C/source-export
      authority, or common/export RVV semantic authority.

## Residue Scan Summary

Remaining high-risk scan matches are pre-existing and classified as:

- RVV provider and tests: provider-derived intrinsic/type leaves after typed
  route legality, plus stale-metadata rejection tests.
- `test/Dialect/RVV` and dialect C++ tests: deprecated Stage1 parse-only
  inventory for finite `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*`, explicitly not
  route/artifact/runtime evidence.
- `test/Transforms/RVV`: fail-closed legacy RVV source-front-door tests.
- `scripts/rvv_generated_bundle_abi_e2e.py`: unsupported `--source-seed`
  diagnostic, forbidden-residue self-tests, and residue rejection filters.
- `test/Conversion/EmitC/rvv-first-slice-*` and legacy broadcast target tests:
  fail-closed legacy selected-body negative fixtures.
- Generated add-only dry-run artifact proves the current handoff consumes
  `tcrv_rvv.typed_binary_pre_realized_body` into realized generic
  `tcrv_rvv.binary` before provider-derived route metadata.

## Out Of Scope

- No Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV route.
- No new dtype, LMUL, operation, memory-form, gather/scatter, segmented memory,
  reduction, conversion, compare/select, masked, macc, or strided coverage.
- No compatibility wrappers for legacy i32 route authority.
- No descriptor-driven computation, direct-C/source-export path, dashboard, or
  performance tuning.
- No runtime/correctness/performance claim without fresh `ssh rvv` evidence.

## Validation Plan

1. Start and validate Trellis task context.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit for:
   - `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
   - `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`
   - existing explicit selected-body add/sub/mul fixtures
   - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-dry-run.test`
4. Run the relevant C++ tests if code or route/provider behavior is touched.
5. Run `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the
   script is touched or if generated-bundle evidence needs a direct check.
6. Run `git diff --check`.
7. Run an active-authority scan over active RVV include/lib/script/test paths.
8. Run `check-tianchenrv` if shared pass/plugin/provider/export behavior is
   materially changed.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/core-dialect/index.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/testing/index.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage1-residue-source-front-door-hardening/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-realization-hook/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-arithmetic-pre-realized-realization-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-route-profile-authority-replacement/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-typed-strided-memory-route/prd.md`

Initial code surface inspected:

- `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
- `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`

## Definition Of Done

- [x] The selected-body realization handoff for generic add is either
      implemented or confirmed current in HEAD with focused evidence.
- [x] Any discovered acceptance gap is repaired with production-path code or
      focused tests, not report-only work.
- [x] Trellis task status and journal notes are truthful.
- [x] The task is finished/archived when complete.
- [ ] One coherent commit records task context, implementation/test changes,
      and validation evidence.
