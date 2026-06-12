# Stage2 RVV selected-body realization for typed strided add

## Goal

Implement one bounded Stage 2 RVV plugin-local selected-body realization path
for typed strided add. A selected `tcrv.exec` RVV variant may carry an explicit
pre-realized typed body with runtime base buffer roles, runtime `n`/AVL, and
runtime lhs/rhs/out stride roles. The RVV plugin must consume those explicit
facts into realized `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
`tcrv_rvv.strided_load`, `tcrv_rvv.binary {kind = "add"}`, and
`tcrv_rvv.strided_store` structure before the existing strided_add provider
constructs a route.

This is not a broad selected-body framework, a frontend lowering task, or a
new memory-form family. It is the selected-boundary-to-strided-body handoff for
one already route-supported typed strided add path.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `2d58dba3 rvv: verify generic selected-body realization`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as `.trellis/tasks/05-19-stage2-rvv-selected-body-strided-add`.
- Relevant specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Stage 1 guardrails remain active: do not reintroduce positive `rvv-i32m1`,
  `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-seed/source-front-door,
  descriptor, artifact-name, or exact intrinsic-spelling route authority.
- Current HEAD already supports explicit realized typed strided add through
  `tcrv_rvv.strided_load`, `tcrv_rvv.binary`, `tcrv_rvv.strided_store`,
  provider-derived `strided_add`, generated-bundle dry-run, and real `ssh rvv`
  evidence from the archived typed strided memory route task.
- Current HEAD already supports pre-realized selected-body realization for
  generic add/sub/mul unit-stride bodies through
  `tcrv_rvv.typed_binary_pre_realized_body`.
- Current `TypedBinaryPreRealizedBodyOp` carries only lhs/rhs/out/n operands
  and rejects `memory_form` other than `vector-rhs-load`; current plugin
  realization emits only unit-stride `load/load/binary/store`.
- The production gap for this round is therefore to make an explicit
  pre-realized selected body with stride ABI operands realize into the existing
  typed strided memory body and then flow through the existing provider route.

## Requirements

1. Keep scope to one pre-realized selected-body strided add path.
2. The pre-realized input must carry explicit operation kind, memory form, SEW,
   LMUL, policy, lhs/rhs/out buffer ABI values, runtime `n`/AVL, and
   lhs/rhs/out stride ABI values.
3. `memory_form = "strided-load-store"` is valid only with `op_kind = "add"`
   for this task.
4. RVV plugin-owned realization must produce real `tcrv_rvv` structure before
   provider route construction:
   `setvl`, `with_vl`, two `strided_load` ops, one `binary {kind = "add"}`,
   and one `strided_store`.
5. Missing or inconsistent op kind, dtype/config, memory form, stride runtime
   ABI, base memory roles, runtime `n`/AVL, or selected variant `requires`
   metadata must fail closed before route/artifact construction.
6. The provider must consume only the realized typed body and must not infer
   stride or compute semantics from route ids, artifact names, parameter names,
   test names, source-front-door markers, or legacy i32 surfaces.
7. Common EmitC/export must remain neutral. Any materialization changes must be
   generic mechanics for provider-built route payloads, not RVV semantic
   inference.
8. Preserve existing unit-stride pre-realized add/sub/mul, explicit strided
   add, broadcast, compare/select, masked add, reduction, macc, LMUL m2, and
   Stage1 residue fail-closed behavior.
9. `ssh rvv` evidence is required only if this round claims executable
   correctness for the new selected-boundary pre-realized strided path.

## Acceptance Criteria

- [x] PRD and Trellis context match the bounded selected-body strided add goal.
- [x] Positive materialization evidence proves a pre-realized selected strided
      add body is consumed and the resulting body contains `setvl`, `with_vl`,
      `strided_load`, `binary {kind = "add"}`, and `strided_store`.
- [x] Positive emission-plan / target artifact evidence proves the realized
      body is handed to the existing `strided_add` provider route and reaches
      provider-derived route metadata, runtime ABI, and header/artifact facts.
- [x] Negative fail-closed coverage rejects missing stride ABI, wrong stride
      type, wrong stride role, missing op kind, unsupported op kind for
      strided memory, mismatched config, missing memory role, missing runtime
      `n`/AVL, and mixed pre-realized plus already-realized bodies.
- [x] Existing explicit strided add provider/materializer/artifact coverage
      remains passing.
- [x] Existing pre-realized unit-stride add/sub/mul coverage remains passing.
- [x] Generated-bundle dry-run covers the pre-realized selected-body strided
      add handoff.
- [x] Real `ssh rvv` correctness evidence is collected if executable behavior
      is claimed for the new pre-realized strided path.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`,
      `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-seed/source-front-door positive RVV authority,
      descriptor, or common/export RVV semantic authority is reintroduced.
- [x] Focused build/lit/script/C++ checks for touched RVV dialect, plugin
      realization, provider/materializer, target artifact, and script paths
      pass.

## Implementation Results

- Extended `tcrv_rvv.typed_binary_pre_realized_body` with optional variadic
  index stride operands while preserving the existing unit-stride add/sub/mul
  selected-body surface.
- Updated the RVV dialect verifier so:
  - `memory_form = "vector-rhs-load"` requires no stride operands.
  - `memory_form = "strided-load-store"` requires exactly lhs/rhs/out stride
    operands, `op_kind = "add"`, SEW32, LMUL m1, agnostic policy, correct
    base buffer roles, and correct runtime stride ABI roles.
- Extended `RVVExtensionPlugin::materializeSelectedLoweringBoundary` so the
  pre-realized strided add body is realized into `setvl`, `with_vl`, two
  `tcrv_rvv.strided_load` ops, one `tcrv_rvv.binary {kind = "add"}`, and one
  `tcrv_rvv.strided_store` before route construction.
- Reused the existing `strided_add` provider/materializer/target path. No
  common EmitC semantic inference or new provider route family was added.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` so
  `--pre-realized-selected-body --op-kind strided_add` uses the new selected
  boundary fixture and verifies the generated-bundle ABI path.
- Added focused positive lit coverage for selected-boundary realization,
  emission-plan mirrors, header export, generated-bundle dry-run, and
  fail-closed strided selected-body errors.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage2-rvv-selected-body-strided-add`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [x] Focused lit via `/usr/lib/llvm-20/build/utils/lit/lit.py`: 11/11 passed.
      Covered new pre-realized strided add, existing pre-realized add/sub/mul,
      pre-realized negatives, explicit strided add artifact/materialization,
      and generated-bundle dry-run tests.
- [x] `build/bin/tianchenrv-rvv-dialect-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-construction-protocol-common-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Local pre-realized strided add generated-bundle dry-run:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-strided-add-dry-run`
- [x] Real `ssh rvv` correctness evidence:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-strided-add-evidence`,
      with `PASS op=strided_add counts=7,16,23`.
- [x] `git diff --check`
- [x] Diff-only active-authority scan found no newly added `RVVI32M1`,
      `rvv-i32m1`, `i32_binary_pre_realized_body`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
      descriptor/direct-C/source-export route authority, or common/export RVV
      semantic authority.
- [x] `cmake --build build --target check-tianchenrv -j2`: 168/168 lit tests
      passed.

## Self-Repair

- The pre-realized body assembly format was migrated to
  `functional-type(operands, results)` so the same op can support both
  four-operand unit-stride bodies and seven-operand strided bodies.
- Existing unit-stride pre-realized fixtures were updated to the new printed
  type form and revalidated against the same selected-body realization checks.
- The unsupported-memory-form negative expectation was updated after
  `strided-load-store` became a valid bounded pre-realized memory form.

## Spec Update Decision

No `.trellis/spec/` update was needed in this round. The existing specs already
state the long-term rule used here: code-affecting selected-body hints must be
consumed into typed `tcrv_rvv` body structure before provider route
construction, RVV semantics stay plugin-local, and common EmitC/export remains
neutral. This task implemented that existing contract for one bounded strided
add case rather than adding a new durable architecture rule.

## Non-Goals

- No Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV route or source-seed evidence mode.
- No gather/scatter, segmented memory, broad memory framework, dtype/LMUL clone
  batch, conversion, new operation class, dashboard, tuning database, or
  performance claim.
- No compatibility wrapper preserving legacy i32 route authority.
- No descriptor-driven computation or descriptor-driven C/source export.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.

## Validation Plan

1. Start and validate Trellis task context.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit for pre-realized selected-body positive/negative cases,
   explicit strided add materialization/artifact, and generated-bundle dry-run.
4. Run touched C++ tests for RVV dialect/plugin/construction/provider/target
   APIs where lit coverage is insufficient.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
6. Run a local generated-bundle dry-run for the pre-realized selected-body
   strided add path.
7. Run real `ssh rvv` correctness evidence only if the new path is claimed
   executable.
8. Run `git diff --check`.
9. Run an active-authority scan over active RVV include/lib/script/test paths.
10. Run `check-tianchenrv` if shared route/provider/runtime behavior changes
    enough to justify a broader gate.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/core-dialect/index.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/variant-pipeline/index.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-generic-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-typed-strided-memory-route/prd.md`

Initial implementation surface:

- `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Support/RuntimeABI.h`
- `include/TianChenRV/Support/RuntimeABIContract.h`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing pre-realized selected-body and strided-add tests under `test/`.

## Definition Of Done

- [x] The bounded pre-realized selected-body strided add realization path is
      implemented in production RVV plugin/dialect code.
- [x] Focused positive and fail-closed tests cover the new behavior.
- [x] Route-supported versus executable evidence is reported truthfully.
- [x] Trellis task status and journal notes are truthful.
- [x] The task is finished/archived when complete.
- [x] One coherent commit records task context, implementation, tests, and
      validation evidence.
