# Stage2 RVV indexed movement operand binding adoption

## Goal

Adopt the RVV plugin-owned `RVVRouteOperandBindingPlan` contract for the
current active indexed RVV memory-movement routes, so the indexed base/index/
destination/runtime operands, provider materialization, emission-plan mirrors,
generated headers, and generated-bundle expectations all come from one route
binding authority.

This task continues the operand-binding hardening line from:

- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-coverage-closure/`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-arithmetic-select-operand-binding/`

## Direction Source

- Direction title: `Stage2 RVV indexed movement operand binding adoption`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` adoption for the
  current active indexed RVV movement routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6ffaeca0 rvv: adopt arithmetic operand binding plans`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- Current active indexed movement route-supported/executable routes are:
  - `indexed_gather_unit_store`
  - `indexed_scatter_unit_load`
- These routes already have typed/pre-realized body coverage, route
  planning/provider support, target artifact fixtures, generated-bundle
  dry-run tests, and prior `ssh rvv` evidence from their archived executable
  slice tasks.
- At current HEAD, indexed routes carry indexed metadata mirrors such as
  `indexed_memory_layout`, `index_source`, `index_eew`, `offset_unit`, and
  indexed data/destination memory form, but they do not carry
  `tcrv_rvv.route_operand_binding_plan` or
  `tcrv_rvv.route_operand_binding_operands`.
- `verifyRVVSelectedBodyEmitCRouteDescription` requires binding-plan mirrors
  for arithmetic/select/strided/masked-store routes and requires all other
  routes, including indexed movement, to carry no binding summary.
- Provider materialization still directly reads indexed route ABI operands via
  `slice.indexABI` in indexed load/store paths, while converted routes bind
  materialized operands through `RVVRouteOperandBindingPlan`.
- Specs require route authority to remain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- Common EmitC/export must remain neutral and must not infer RVV indexed
  memory semantics, base/index/destination roles, dtype, SEW/LMUL, tail
  policy, intrinsic choices, route support, or correctness from route IDs,
  artifact names, metadata, descriptors, or source-front-door markers.

## Scope

Convert this indexed movement cluster:

- `indexed_gather_unit_store`: `data[index[i]] -> out[i]`
- `indexed_scatter_unit_load`: `src[i] -> dst[index[i]]`

For every converted route, record logical operand, typed body/runtime role, C
parameter, materialized expression/use, and mirror/header field in
`RVVRouteOperandBindingPlan`, then make provider emission consume the contract
for real emitted operands.

## Requirements

1. Inventory indexed RVV movement routes that are active route-supported or
   executable today and classify them as converted, inactive/parser-only, or
   out of scope.
2. Add binding-plan IDs and logical-operand role validation for:
   - `indexed_gather_unit_store`
   - `indexed_scatter_unit_load`
3. For `indexed_gather_unit_store`, bind:
   - `data`: `lhs-input-buffer`, materialized indexed data base and header
     mirror.
   - `index`: `index-input-buffer`, materialized index vector load, index
     offset scaling, index-source mirror, and header mirror.
   - `out`: `output-buffer`, materialized unit-stride store base and header
     mirror.
   - `n`: `runtime-element-count`, setvl AVL, loop control, and header mirror.
4. For `indexed_scatter_unit_load`, bind:
   - `src`: `lhs-input-buffer`, materialized unit-stride source load and
     header mirror.
   - `index`: `index-input-buffer`, materialized index vector load, index
     offset scaling, index-source mirror, and header mirror.
   - `dst`: `output-buffer`, materialized indexed destination store base and
     header mirror.
   - `n`: `runtime-element-count`, setvl AVL, loop control, and header mirror.
5. Rewire provider emission so indexed load/store materialized operands use
   the binding contract rather than direct `slice.*ABI` lookup.
6. Require route description mirrors and generated artifact/header metadata for
   converted indexed routes to carry the same binding plan ID and summary.
7. Add or update positive structural checks proving both indexed routes carry
   binding plan IDs, binding summaries, header mirrors, and materialized
   operands from the contract.
8. Add or update negative fail-closed tests for expressible indexed movement
   errors: index/base swaps, source/destination swaps, missing or duplicate
   ABI roles, missing materialized uses, mirror/header mismatch, stale route-id
   authority, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference.
9. Keep segmented, widening, contraction, conversion, masked matrix expansion,
   source-front-door positive routes, and new operation families out of scope.

## Acceptance Criteria

- [x] Current active indexed route inventory is recorded in completion notes.
- [x] `indexed_gather_unit_store` derives `data/index/out/n` materialized
      operands and route/header mirrors from `RVVRouteOperandBindingPlan`.
- [x] `indexed_scatter_unit_load` derives `src/index/dst/n` materialized
      operands and route/header mirrors from `RVVRouteOperandBindingPlan`.
- [x] Provider emission fails closed when indexed route bindings are missing,
      duplicated, role-swapped, materialized-use mismatched, or inconsistent
      with runtime ABI mirrors.
- [x] Positive structural lit/FileCheck coverage proves converted indexed
      routes carry binding plan IDs and summaries in emission-plan metadata and
      generated headers.
- [x] Negative fail-closed coverage exists for indexed operand role swaps and
      binding-plan/mirror mismatch surfaces that the current textual IR can
      express.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized indexed
      gather/scatter routes with non-monotonic indices and tail/sentinel
      cases.
- [x] Real `ssh rvv` PASS evidence exists for representative converted indexed
      routes when runtime/correctness is claimed.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic authority, stale route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [x] Focused checks, `check-tianchenrv`, `git diff --check`, Trellis finish/
      archive, clean git status, and one coherent commit are completed if the
      task finishes.

## Non-Goals

- No segmented movement, widening, contraction, conversion, masked movement
  matrix expansion, dtype/LMUL clone batches, or new Stage2 operation
  families.
- No high-level Linalg/Vector/StableHLO/frontend lowering.
- No source-front-door positive RVV routes.
- No dashboards, report-only work, helper-only cleanup, or broad smoke tests
  as the main achievement.
- No descriptor-driven compute, direct-C/source-export route restoration, or
  compatibility wrapper preserving old route authority.
- No movement of RVV indexed semantics into common EmitC/export, target
  metadata, artifact names, route IDs, descriptors, ABI strings alone, test
  names, or exact intrinsic spelling.

## Validation Plan

1. Validate and maintain this Trellis task.
2. Run focused lit/FileCheck checks for indexed route plan mirrors, target
   artifacts, generated headers, EmitC materialization, and negative
   fail-closed cases.
3. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
4. Run generated-bundle dry-runs for explicit and pre-realized indexed
   gather/scatter routes at representative counts.
5. Run real `ssh rvv` evidence for representative converted indexed routes
   after dry-runs pass.
6. Run active-authority scans over touched active RVV include/lib/script/test
   paths.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/*indexed*gather*`
- `test/Target/RVV/*indexed*scatter*`
- `test/Scripts/*indexed*gather*`
- `test/Scripts/*indexed*scatter*`
- Existing indexed movement dialect/materialization fixtures.

## Definition Of Done

- The active indexed gather/scatter movement cluster consumes
  `RVVRouteOperandBindingPlan` for provider materialization and mirrors.
- Inactive or out-of-scope indexed-like routes, if found, are named with
  status and reason.
- Focused checks, hardware evidence, active-authority scan, and
  `check-tianchenrv` are complete or any skip is justified by concrete
  environment failure.
- Trellis task status is truthful, archived when complete, and one coherent
  commit records the work.

## Completion Notes

### Indexed Route Inventory

- Converted active executable route: `indexed_gather_unit_store`
  (`data[index[i]] -> out[i]`).
- Converted active executable route: `indexed_scatter_unit_load`
  (`src[i] -> dst[index[i]]`).
- No inactive/parser-only indexed movement route was promoted or rewritten in
  this round.
- Segmented movement, widening, contraction, conversion, masked-memory matrix
  expansion, source-front-door positive routes, and new operation coverage were
  intentionally left out of scope.

### Production Behavior Completed

- `RVVRouteOperandBindingPlan` now has indexed gather/scatter plan IDs and
  expected logical operand roles.
- Indexed gather now binds `data`, `index`, `out`, and `n` through the shared
  route operand binding contract, including materialized data-base,
  index-load/offset-scale, store-base, loop AVL, route metadata, and generated
  header mirrors.
- Indexed scatter now binds `src`, `index`, `dst`, and `n` through the same
  contract, including materialized source-load, index-load/offset-scale,
  indexed destination store-base, loop AVL, route metadata, and generated
  header mirrors.
- RVV provider emission now consumes the indexed binding contract for the
  materialized index ABI instead of directly using indexed route slice ABI
  state.
- Route description verification requires indexed binding-plan mirrors for the
  two converted memory forms and fails closed on missing, mismatched, duplicate,
  role-swapped, stale, or unsupported binding summaries.

### Evidence

- Focused script checks:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused build/checks:
  - `cmake --build build --target tcrv-opt tcrv-translate -j2`
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
- Structural and negative checks:
  - Target artifact FileCheck coverage for explicit and pre-realized indexed
    gather/scatter route metadata and generated header mirrors.
  - Negative MLIR checks for expressible indexed gather duplicate ABI role and
    indexed scatter source/destination role swap failures.
  - C++ provider/planning checks for indexed gather/scatter binding success and
    role-swap fail-closed cases.
- Generated-bundle dry-runs:
  - explicit indexed gather, counts `7,16,23`
  - pre-realized indexed gather, counts `7,16,23`
  - explicit indexed scatter, counts `7,16,23`
  - pre-realized indexed scatter, counts `7,16,23`
- Real hardware evidence:
  - `ssh rvv` PASS for explicit indexed gather, counts `7,16,23`
  - `ssh rvv` PASS for pre-realized indexed gather, counts `7,16,23`
  - `ssh rvv` PASS for explicit indexed scatter, counts `7,16,23`
  - `ssh rvv` PASS for pre-realized indexed scatter, counts `7,16,23`
- Active-authority scan:
  - Full touched-path scan found only existing guardrail/fail-closed/self-test
    text for legacy or forbidden terms.
  - Added-line scan over the diff found no new positive `RVVI32M1`,
    `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
    descriptor/direct-C/source-export/source-front-door, or public exact
    intrinsic route authority.
- Final gates:
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-stage2-rvv-indexed-movement-operand-binding`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`
    (`265/265` lit tests passed)
