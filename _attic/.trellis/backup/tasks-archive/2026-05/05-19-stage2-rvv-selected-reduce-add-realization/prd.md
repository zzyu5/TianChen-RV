# Stage2 RVV selected-body realization for reduce_add

## Goal

Implement one bounded Stage 2 RVV plugin-local selected-body realization path
for an explicit pre-realized `reduce_add` body. A selected `tcrv.exec` RVV
variant may carry explicit pre-realized facts for operation kind, input buffer,
accumulator/seed buffer role, result/output buffer role, reduction layout,
dtype/config/policy, memory form, and runtime `n`/AVL. The RVV plugin must
consume those facts into a realized typed `tcrv_rvv.setvl`,
`tcrv_rvv.with_vl`, `tcrv_rvv.load`, `tcrv_rvv.reduce`, and `tcrv_rvv.store`
body before the existing reduce-add provider constructs a
`TCRVEmitCLowerableRoute`.

This is not a broad reduction framework, high-level frontend lowering, or a new
provider route skeleton. The existing explicit generic reduce-add provider path
is the route/materialization authority after this task realizes the selected
pre-realized boundary into typed body structure.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `81d3a85f rvv: realize pre-realized macc add body`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief as
  `.trellis/tasks/05-19-stage2-rvv-selected-reduce-add-realization`.
- Relevant specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Stage 1 guardrails remain active: do not reintroduce positive
  `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-seed/source-front-door,
  descriptor, artifact-name, or exact intrinsic-spelling route authority.
- Current HEAD already supports explicit realized reduce-add through
  `tcrv_rvv.reduce`, provider-derived `reduce_add`, generated-bundle dry-run,
  and archived real `ssh rvv` evidence from the generic reduce-add executable
  closure task.
- Current HEAD already supports pre-realized selected-body realization for
  unit-stride add/sub/mul, strided add, masked add, and macc add.
- Current pre-realized surfaces do not carry explicit reduce-add accumulator
  seed role/layout or reduction result layout facts. Therefore they cannot
  realize a selected reduce-add body into the existing typed
  `load/load/reduce/store` structure before route construction.

## Requirements

1. Keep scope to one pre-realized selected-body `reduce_add` path.
2. The pre-realized input must carry explicit operation kind, memory form, SEW,
   LMUL, policy, lhs/rhs/out buffer ABI values, runtime `n`/AVL, accumulator
   role, accumulator layout, and result layout facts.
3. `op_kind = "reduce_add"` and `memory_form = "vector-rhs-load"` are the only
   supported positive case for this task.
4. The accumulator role is bounded to the RHS input buffer as the vector seed;
   the realized body must load that explicit RHS buffer before
   `tcrv_rvv.reduce`.
5. The result role is bounded to the explicit output buffer; reduction result
   layout is the existing bounded lane-0 per dynamic VL chunk layout.
6. RVV plugin-owned realization must produce real `tcrv_rvv` structure before
   provider route construction: `setvl`, `with_vl`, input load, accumulator
   load, `reduce {kind = "add"}`, and output store.
7. Missing or inconsistent op kind, memory form, accumulator role/layout,
   result layout, dtype/config, policy, input/output memory role, runtime
   `n`/AVL, or selected variant `requires` metadata must fail closed before
   route/artifact construction.
8. The existing reduce-add provider remains route authority and consumes only
   the realized typed body. Common EmitC/export must remain neutral.
9. Preserve existing explicit reduce-add, pre-realized add/sub/mul, strided
   add, masked add, macc add, broadcast, compare/select, LMUL m2, and Stage1
   residue fail-closed behavior.
10. `ssh rvv` evidence is required only if this round claims executable
    correctness for the new pre-realized reduce-add selected-boundary path.

## Acceptance Criteria

- [x] PRD and Trellis context match the bounded selected-body reduce-add goal.
- [x] Positive materialization evidence proves a pre-realized selected
      `reduce_add` body is consumed and the resulting body contains `setvl`,
      `with_vl`, input/accumulator `load`, `reduce {kind = "add"}`, and
      `store`.
- [x] Positive emission-plan / target artifact evidence proves the realized
      body is handed to the existing `reduce_add` provider route and reaches
      provider-derived route metadata, runtime ABI, accumulator layout, result
      layout, store-VL mirror, and header/artifact facts.
- [x] Negative fail-closed coverage rejects missing/unsupported op kind,
      missing/unsupported memory form, missing/wrong accumulator role,
      missing/wrong accumulator layout, missing/wrong result layout,
      mismatched config, unsupported policy, missing memory role, missing
      runtime `n`/AVL, and mixed pre-realized plus already-realized bodies.
- [x] Existing explicit reduce-add provider/materializer/artifact behavior
      remains passing.
- [x] Existing pre-realized unit-stride add/sub/mul, strided add, masked add,
      and macc add behavior remains passing.
- [x] Generated-bundle dry-run covers the pre-realized selected-body reduce-add
      handoff.
- [x] Real `ssh rvv` correctness evidence is collected if executable behavior
      is claimed for the new pre-realized reduce-add path.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`,
      `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-seed/source-front-door positive RVV authority,
      descriptor, or common/export RVV semantic authority is reintroduced.
- [x] Focused build/lit/script/C++ checks for touched RVV dialect, plugin
      realization, provider/materializer, target artifact, and script paths
      pass.

## Non-Goals

- No high-level Linalg/Vector/StableHLO frontend lowering.
- No general reduction algebra, reductions beyond this bounded reduce-add
  handoff, or additional reduction kinds.
- No broad dtype/LMUL clone batch, source shape expansion, gather/scatter,
  conversion, dashboard, tuning database, or performance claim.
- No source-front-door positive RVV route or source-seed evidence mode.
- No compatibility wrapper preserving legacy i32 route authority.
- No descriptor-driven computation or descriptor-driven C/source export.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit for pre-realized selected-body reduce-add positive and
   negative cases, existing explicit reduce-add materialization/artifact, and
   existing pre-realized add/sub/mul/strided/masked/macc regression fixtures.
4. Run touched C++ tests for RVV dialect/plugin/construction/provider/target
   APIs where lit coverage is insufficient.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
6. Run a local generated-bundle dry-run for
   `--pre-realized-selected-body --op-kind reduce_add`.
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
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-generic-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-strided-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-masked-add-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-macc-add/prd.md`

Initial implementation surface:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing reduce-add and pre-realized selected-body tests under `test/`.

## Implementation Results

- Added `tcrv_rvv.typed_reduce_pre_realized_body` as the bounded generic
  reduce-add pre-realized selected body. It carries explicit
  `op_kind = "reduce_add"`, `memory_form = "vector-rhs-load"`,
  `accumulator_role = "rhs-input-buffer"`, reduction accumulator/result
  layout, SEW32, LMUL m1, agnostic policy, lhs/rhs/out ABI values, and runtime
  `n`/AVL.
- Extended the RVV dialect verifier so the new pre-realized reduce body rejects
  stale authority metadata, unsupported op/memory/accumulator/result facts,
  non-SEW32/LMUL m1 config, non-agnostic policy, wrong ABI roles, and missing
  runtime `n`/AVL before route construction.
- Extended `RVVExtensionPlugin::materializeSelectedLoweringBoundary` so a
  pre-realized reduce body realizes into `setvl`, `with_vl`, input load,
  accumulator seed load, `reduce {kind = "add"}`, and output store, then erases
  the pre-realized op before provider route validation.
- Reused the existing reduce-add provider/materializer/target route. No common
  EmitC semantic inference or new provider route family was added.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` so
  `--pre-realized-selected-body --op-kind reduce_add` uses the new selected
  boundary fixture and verifies the generated-bundle ABI path.
- Added focused positive target/script coverage and reduce-specific negative
  verifier coverage for missing/unsupported op kind, memory form, accumulator
  role/layout, result layout, output role, config, policy, runtime `n`/AVL,
  and mixed pre-realized plus already-realized body structure.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage2-rvv-selected-reduce-add-realization`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Focused lit set from `build/test`: 11/11 passed, covering new
      pre-realized reduce-add positive/negative/script evidence, explicit
      reduce-add, existing pre-realized add/sub/mul/strided/masked/macc
      fixtures, and existing pre-realized negatives.
- [OK] Local pre-realized reduce-add generated-bundle dry-run:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-reduce-add-dry-run`.
- [OK] Real `ssh rvv` correctness evidence:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-reduce-add-evidence`,
      with `PASS op=reduce_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 176/176 passed.
- [OK] `git diff --check`
- [OK] Diff active-authority scan found no newly added positive `RVVI32M1`,
      `rvv-i32m1`, `i32_binary_pre_realized_body`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
      descriptor/direct-C/source-export route authority, exact intrinsic
      spelling authority, or common/export RVV semantic authority. New matches
      were anti-authority text in the new op description and negative
      FileCheck guardrails.

## Self-Repair

- Re-ran the focused lit set from `build/test` after the first lit invocation
  from the repository root failed to resolve the generated lit site config
  relative path.
- Tightened the pre-realized reduce-add script dry-run FileCheck expectations
  to match the actual generated harness loop for per-VL chunk accumulation.

## Spec Update Decision

No `.trellis/spec/` update was needed in this round. The existing specs already
state the long-term rule used here: code-affecting selected-body reduction,
accumulator, result-layout, runtime, and policy facts must be consumed into
typed `tcrv_rvv` body structure before provider route construction, RVV
semantics stay plugin-local, and common EmitC/export remains neutral.

## Definition Of Done

- [x] The bounded pre-realized selected-body reduce-add realization path is
      implemented in production RVV plugin/dialect code.
- [x] Focused positive and fail-closed tests cover the new behavior.
- [x] Route-supported versus executable evidence is reported truthfully.
- [x] Trellis task status and journal notes are truthful.
- [x] The task is finished/archived when complete.
- [x] One coherent commit records task context, implementation, tests, and
      validation evidence.
