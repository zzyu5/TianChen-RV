# Stage2 RVV selected-body realization for macc add

## Goal

Implement one bounded Stage 2 RVV plugin-local selected-body realization path
for an explicit pre-realized `macc_add` body. A selected `tcrv.exec` RVV
variant may carry explicit pre-realized facts for operation kind, lhs/rhs
buffers, accumulator/output buffer role, accumulator/result layout,
dtype/config/policy, memory form, and runtime `n`/AVL. The RVV plugin must
consume those facts into a realized typed `tcrv_rvv.setvl`,
`tcrv_rvv.with_vl`, `tcrv_rvv.load`, `tcrv_rvv.macc`, and `tcrv_rvv.store`
body before the existing macc provider constructs a
`TCRVEmitCLowerableRoute`.

This is not a high-level matmul/contraction framework, not a frontend lowering
task, and not a new macc route skeleton. The existing explicit macc provider
route from commit `9d0fac07` is the route/materialization authority after this
round realizes the selected pre-realized boundary into typed body structure.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `11efca89 rvv: realize pre-realized masked add body`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief as
  `.trellis/tasks/05-19-05-19-stage2-rvv-selected-body-macc-add`.
- Relevant specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Stage 1 guardrails remain active: do not reintroduce positive
  `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-seed/source-front-door,
  descriptor, artifact-name, or exact intrinsic-spelling route authority.
- Current HEAD already supports explicit realized macc add through
  `tcrv_rvv.macc`, provider-derived `macc_add`, generated-bundle dry-run, and
  archived real `ssh rvv` evidence from the macc route skeleton task.
- Current HEAD already supports pre-realized selected-body realization for
  unit-stride add/sub/mul, strided add, and masked add.
- Current pre-realized surfaces do not carry an explicit accumulator role or
  macc accumulator/result layout facts. Therefore they cannot realize a
  selected macc body into the existing typed `load/load/load/macc/store`
  structure before route construction.

## Requirements

1. Keep scope to one pre-realized selected-body `macc_add` path.
2. The pre-realized input must carry explicit operation kind, memory form,
   SEW, LMUL, policy, lhs/rhs/out buffer ABI values, runtime `n`/AVL,
   accumulator role, accumulator layout, and result layout facts.
3. `op_kind = "macc_add"` and `memory_form = "vector-rhs-load"` are the only
   supported positive case for this task.
4. The accumulator role is bounded to output-buffer vector accumulator input;
   the realized body must load the accumulator from the explicit output buffer
   before `tcrv_rvv.macc`.
5. RVV plugin-owned realization must produce real `tcrv_rvv` structure before
   provider route construction:
   `setvl`, `with_vl`, lhs load, rhs load, accumulator load, `macc {kind =
   "add"}`, and output store.
6. Missing or inconsistent op kind, memory form, accumulator role/layout,
   result layout, dtype/config, policy, base memory role, runtime `n`/AVL, or
   selected variant `requires` metadata must fail closed before route/artifact
   construction.
7. The existing macc provider remains route authority and consumes only the
   realized typed body. Common EmitC/export must remain neutral.
8. Preserve existing explicit macc, unit-stride pre-realized add/sub/mul,
   pre-realized strided add, pre-realized masked add, broadcast,
   compare/select, reduction, LMUL m2, and Stage1 residue fail-closed
   behavior.
9. `ssh rvv` evidence is required only if this round claims executable
   correctness for the new selected-boundary pre-realized macc path.

## Acceptance Criteria

- [x] PRD and Trellis context match the bounded selected-body macc add goal.
- [x] Positive materialization evidence proves a pre-realized selected
      `macc_add` body is consumed and the resulting body contains `setvl`,
      `with_vl`, lhs/rhs/accumulator `load`, `macc {kind = "add"}`, and
      `store`.
- [x] Positive emission-plan / target artifact evidence proves the realized
      body is handed to the existing `macc_add` provider route and reaches
      provider-derived route metadata, runtime ABI, accumulator layout, result
      layout, and header/artifact facts.
- [x] Negative fail-closed coverage rejects missing/unsupported op kind,
      missing/unsupported memory form, missing/wrong accumulator role,
      missing/wrong accumulator layout, missing/wrong result layout,
      mismatched config, unsupported policy, missing memory role, missing
      runtime `n`/AVL, and mixed pre-realized plus already-realized bodies.
- [x] Existing explicit macc provider/materializer/artifact coverage remains
      passing.
- [x] Existing pre-realized unit-stride add/sub/mul, strided add, and masked
      add coverage remains passing.
- [x] Generated-bundle dry-run covers the pre-realized selected-body macc
      handoff.
- [x] Real `ssh rvv` correctness evidence is collected if executable behavior
      is claimed for the new pre-realized macc path.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`,
      `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-seed/source-front-door positive RVV authority,
      descriptor, or common/export RVV semantic authority is reintroduced.
- [x] Focused build/lit/script/C++ checks for touched RVV dialect, plugin
      realization, provider/materializer, target artifact, and script paths
      pass.

## Non-Goals

- No high-level Linalg/Vector/StableHLO frontend lowering.
- No matmul route authority or broad contraction algebra/framework.
- No reductions beyond this bounded accumulator use.
- No broad dtype/LMUL clone batch, source shape expansion, gather/scatter,
  conversion, dashboard, tuning database, or performance claim.
- No source-front-door positive RVV route or source-seed evidence mode.
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
3. Run focused lit for pre-realized selected-body macc positive and negative
   cases, existing explicit macc materialization/artifact, existing
   pre-realized add/sub/mul/strided/masked regression fixtures, and
   generated-bundle dry-run tests.
4. Run touched C++ tests for RVV dialect/plugin/construction/provider/target
   APIs where lit coverage is insufficient.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the
   script changes.
6. Run a local generated-bundle dry-run for
   `--pre-realized-selected-body --op-kind macc_add`.
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

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-multiply-add-route-skeleton/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-generic-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-strided-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-masked-add-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-realization-hook/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-arithmetic-pre-realized-realization-closure/prd.md`

Initial implementation surface:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing macc and pre-realized selected-body tests under `test/`.

## Implementation Results

- Added `tcrv_rvv.typed_macc_pre_realized_body` as the bounded generic macc
  pre-realized selected body. It carries explicit `op_kind = "macc_add"`,
  `memory_form = "vector-rhs-load"`, `accumulator_role = "output-buffer"`,
  accumulator layout, result layout, SEW32, LMUL m1, agnostic policy,
  lhs/rhs/out ABI values, and runtime `n`/AVL.
- Extended the RVV dialect verifier so the new pre-realized macc body rejects
  stale authority metadata, unsupported op/memory/accumulator/result facts,
  non-SEW32/LMUL m1 config, non-agnostic policy, wrong ABI roles, and missing
  runtime `n`/AVL before route construction.
- Extended `RVVExtensionPlugin::materializeSelectedLoweringBoundary` so a
  pre-realized macc body realizes into `setvl`, `with_vl`, lhs load, rhs load,
  output-buffer accumulator load, `macc {kind = "add"}`, and output store,
  then erases the pre-realized op before provider route validation.
- Reused the existing macc provider/materializer/target route. No common
  EmitC semantic inference or new provider route family was added.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` so
  `--pre-realized-selected-body --op-kind macc_add` uses the new selected
  boundary fixture and verifies the generated-bundle ABI path.
- Added focused positive target/script coverage and a macc-specific negative
  verifier file for missing/unsupported op kind, accumulator role/layout,
  result layout, output role, and mixed pre-realized plus already-realized
  body structure.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-05-19-stage2-rvv-selected-body-macc-add`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [x] `build/bin/tianchenrv-rvv-dialect-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-construction-protocol-common-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused lit set from `build/test` passed 16/16, covering new
      pre-realized macc positive/negative/script evidence, explicit macc,
      existing pre-realized add/sub/mul/strided/masked fixtures, existing
      pre-realized negatives, and macc materialization/provider negatives.
- [x] Local pre-realized macc generated-bundle dry-run:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-macc-add-dry-run`.
- [x] Real `ssh rvv` correctness evidence:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-macc-add-evidence`,
      with `PASS op=macc_add counts=7,16,23`.
- [x] `cmake --build build --target check-tianchenrv -j2`: 173/173 lit tests
      passed.
- [x] `git diff --check`
- [x] Diff-only active-authority scan found no newly added positive
      `RVVI32M1`, `rvv-i32m1`, `i32_binary_pre_realized_body`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
      descriptor/direct-C/source-export route authority, exact intrinsic
      spelling authority, or common/export RVV semantic authority. New matches
      were anti-authority text in the new op description and negative
      FileCheck guardrails.

## Self-Repair

- Re-ran the same focused lit set from `build/test` after an initial invocation
  from `build/` failed to locate the repository lit config. This was command
  placement error, not a source/test failure.

## Spec Update Decision

No `.trellis/spec/` update was needed in this round. The existing specs already
state the long-term rule used here: code-affecting selected-body accumulator
and policy facts must be consumed into typed `tcrv_rvv` body structure before
provider route construction, RVV semantics stay plugin-local, and common
EmitC/export remains neutral.

## Definition Of Done

- [x] The bounded pre-realized selected-body macc add realization path is
      implemented in production RVV plugin/dialect code.
- [x] Focused positive and fail-closed tests cover the new behavior.
- [x] Route-supported versus executable evidence is reported truthfully.
- [x] Trellis task status and journal notes are truthful.
- [x] The task is finished/archived when complete.
- [ ] One coherent commit records task context, implementation, tests, and
      validation evidence.
