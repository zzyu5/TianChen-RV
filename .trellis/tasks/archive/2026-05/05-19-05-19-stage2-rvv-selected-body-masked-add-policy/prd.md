# Stage2 RVV selected-body realization for masked add policy

## Goal

Implement one bounded Stage 2 RVV plugin-local selected-body realization path
for an explicit pre-realized masked add body. A selected `tcrv.exec` RVV
variant may carry explicit pre-realized facts for operation kind, typed
config, runtime `n`/AVL, base memory ABI roles, mask source/passthrough
policy, and tail/mask policy. The RVV plugin must consume those facts into a
realized `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, `tcrv_rvv.load`,
`tcrv_rvv.compare`, `tcrv_rvv.masked_binary`, and `tcrv_rvv.store` body before
the existing masked-add provider route constructs a `TCRVEmitCLowerableRoute`.

This is not a broad mask framework, a high-level frontend task, or a new route
family. It is the selected-boundary-to-masked-body handoff for the existing
typed masked add provider path.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `06867f9b rvv: realize pre-realized strided add body`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief as
  `.trellis/tasks/05-19-05-19-stage2-rvv-selected-body-masked-add-policy`.
- Relevant specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Stage 1 guardrails remain active: do not reintroduce positive
  `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-seed/source-front-door,
  descriptor, artifact-name, or exact intrinsic-spelling route authority.
- Current HEAD already supports explicit realized masked add through
  `tcrv_rvv.compare`, `tcrv_rvv.masked_binary`, provider-derived
  `masked_add`, generated-bundle dry-run, and archived real `ssh rvv`
  evidence from the masked-add route task.
- Current HEAD already supports pre-realized selected-body realization for
  unit-stride generic add/sub/mul and typed strided add through
  `tcrv_rvv.typed_binary_pre_realized_body`.
- Current `TypedBinaryPreRealizedBodyOp` carries only lhs/rhs/out/n plus
  optional stride facts. It cannot express a pre-realized masked-add mask
  source or passthrough layout, so the RVV plugin cannot realize masked
  selected-body facts into structural `compare + masked_binary` before route
  construction.

## Requirements

1. Keep scope to one pre-realized selected-body masked add path.
2. The pre-realized input must carry explicit operation kind, memory form, SEW,
   LMUL, policy, lhs/rhs/out buffer ABI values, runtime `n`/AVL, mask source,
   and masked passthrough layout facts.
3. `memory_form = "masked-vector-rhs-load"` is valid only with
   `op_kind = "masked_add"` for this task.
4. `mask_source` is bounded to compare-produced mask from lhs/rhs in the same
   selected VL scope.
5. `masked_passthrough` is bounded to lhs-load passthrough preserving inactive
   lanes.
6. RVV plugin-owned realization must produce real `tcrv_rvv` structure before
   provider route construction:
   `setvl`, `with_vl`, two `load` ops, one `compare {kind = "eq"}`, one
   `masked_binary {kind = "add"}`, and one `store`.
7. Missing or inconsistent op kind, mask source, passthrough layout, policy,
   dtype/config, memory role, runtime `n`/AVL, or selected variant `requires`
   metadata must fail closed before route/artifact construction.
8. The existing provider remains route authority and consumes only the
   realized typed body. Common EmitC/export must remain neutral.
9. Preserve existing explicit masked add, unit-stride pre-realized add/sub/mul,
   pre-realized strided add, broadcast, compare/select, reduction, macc, LMUL
   m2, and Stage1 residue fail-closed behavior.
10. `ssh rvv` evidence is required only if this round claims executable
    correctness for the new selected-boundary pre-realized masked add path.

## Acceptance Criteria

- [x] PRD and Trellis context match the bounded selected-body masked add goal.
- [x] Positive materialization evidence proves a pre-realized selected masked
      add body is consumed and the resulting body contains `setvl`, `with_vl`,
      `load`, `compare`, `masked_binary {kind = "add"}`, and `store`.
- [x] Positive emission-plan / target artifact evidence proves the realized
      body is handed to the existing `masked_add` provider route and reaches
      provider-derived route metadata, runtime ABI, mask source, passthrough
      layout, and header/artifact facts.
- [x] Negative fail-closed coverage rejects missing/unsupported op kind,
      missing/unsupported mask source, missing/unsupported passthrough layout,
      mismatched config, unsupported policy, missing memory role, missing
      runtime `n`/AVL, and mixed pre-realized plus already-realized bodies.
- [x] Existing explicit masked add provider/materializer/artifact coverage
      remains passing.
- [x] Existing pre-realized unit-stride add/sub/mul and strided add coverage
      remains passing.
- [x] Generated-bundle dry-run covers the pre-realized selected-body masked
      add handoff.
- [x] Real `ssh rvv` correctness evidence is collected if executable behavior
      is claimed for the new pre-realized masked path.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`,
      `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-seed/source-front-door positive RVV authority,
      descriptor, or common/export RVV semantic authority is reintroduced.
- [x] Focused build/lit/script/C++ checks for touched RVV dialect, plugin
      realization, provider/materializer, target artifact, and script paths
      pass.

## Implementation Results

- Added `tcrv_rvv.typed_masked_binary_pre_realized_body` as the bounded
  generic masked-add pre-realized selected body. It carries explicit
  `op_kind = "masked_add"`, `memory_form = "masked-vector-rhs-load"`,
  `mask_source = "compare-produced-mask-same-vl-scope"`,
  `masked_passthrough = "passthrough-vector-preserves-inactive-lanes"`,
  SEW32, LMUL m1, agnostic policy, lhs/rhs/out ABI values, and runtime
  `n`/AVL.
- Extended the RVV dialect verifier so the new pre-realized masked body
  rejects stale authority metadata, unsupported op/memory/mask/passthrough
  facts, non-SEW32/LMUL m1 config, non-agnostic policy, wrong ABI roles, and
  missing runtime `n`/AVL before route construction.
- Extended `RVVExtensionPlugin::materializeSelectedLoweringBoundary` so a
  pre-realized masked body realizes into `setvl`, `with_vl`, two generic
  `load` ops, `compare {kind = "eq"}`, `masked_binary {kind = "add"}`, and
  `store`, then erases the pre-realized op before provider route validation.
- Reused the existing masked-add provider/materializer/target route. No common
  EmitC semantic inference or new provider route family was added.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` so
  `--pre-realized-selected-body --op-kind masked_add` uses the new selected
  boundary fixture and verifies the generated-bundle ABI path.
- Added focused positive lit/script coverage and negative fail-closed cases
  for missing/unsupported mask source, passthrough layout, op kind, policy,
  and mixed pre-realized plus already-realized body structure.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-05-19-stage2-rvv-selected-body-masked-add-policy`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [x] Direct positive `FileCheck` commands for
      `test/Target/RVV/pre-realized-selected-body-artifact-masked-add.mlir`
      realization, emission-plan, and header export.
- [x] Direct negative verifier run:
      `build/bin/tcrv-opt test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries`
- [x] Focused lit filter passed 11/11 for new pre-realized masked add,
      explicit masked add, existing pre-realized add/sub/mul/strided add,
      pre-realized negative coverage, and generated-bundle dry-run tests.
- [x] `build/bin/tianchenrv-rvv-dialect-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-construction-protocol-common-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Local pre-realized masked add generated-bundle dry-run:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-pre-realized-masked-add-dry-run`
- [x] Real `ssh rvv` correctness evidence:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-pre-realized-masked-add-evidence`,
      with `PASS op=masked_add counts=7,16,23`.
- [x] `cmake --build build --target check-tianchenrv -j2`: 170/170 lit tests
      passed.
- [x] `git diff --check`
- [x] Diff active-authority scan found no newly added positive `RVVI32M1`,
      `rvv-i32m1`, `i32_binary_pre_realized_body`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
      descriptor/direct-C/source-export route authority, or common/export RVV
      semantic authority. The only new matches were negative guardrail text in
      the new pre-realized op description.

## Spec Update Decision

No `.trellis/spec/` update was needed in this round. The existing specs already
state the long-term rule used here: code-affecting selected-body mask/policy
facts must be consumed into typed `tcrv_rvv` body structure before provider
route construction, RVV semantics stay plugin-local, and common EmitC/export
remains neutral.

## Non-Goals

- No broad mask framework, mask loads/stores, mask algebra, compare/select
  expansion, reductions, macc expansion, gather/scatter, segmented memory,
  conversion, dtype/LMUL clone batch, dashboard, tuning database, or
  performance claim.
- No Linalg/Vector/StableHLO frontend lowering.
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
3. Run focused lit for pre-realized selected-body masked add positive and
   negative cases, existing explicit masked add materialization/artifact, and
   existing pre-realized add/sub/mul/strided-add regression fixtures.
4. Run touched C++ tests for RVV dialect/plugin/construction/provider/target
   APIs where lit coverage is insufficient.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
6. Run a local generated-bundle dry-run for
   `--pre-realized-selected-body --op-kind masked_add`.
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
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/core-dialect/index.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-generic-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-strided-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/check.jsonl`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-strided-add/check.jsonl`

Initial implementation surface:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing masked-add and pre-realized selected-body tests under `test/`.

## Definition Of Done

- [x] The bounded pre-realized selected-body masked add realization path is
      implemented in production RVV plugin/dialect code.
- [x] Focused positive and fail-closed tests cover the new behavior.
- [x] Route-supported versus executable evidence is reported truthfully.
- [x] Trellis task status and journal notes are truthful.
- [x] The task is finished/archived when complete.
- [x] One coherent commit records task context, implementation, tests, and
      validation evidence.
