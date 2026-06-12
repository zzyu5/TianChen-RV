# Stage2 RVV operand binding coverage closure

## Goal

Finish the next bounded adoption step for the RVV plugin-owned
`RVVRouteOperandBindingPlan` contract across existing executable RVV routes
that still materialize ABI operands or route mirrors outside the shared
binding contract.

This round does not add new Stage2 operation coverage. It inventories the
active executable RVV route surface, converts the current unconverted routes
called out by the Direction Brief, and leaves any intentionally unconverted
route with a precise active/non-active status and continuation point.

## Direction Source

- Direction title: `Stage2 RVV operand binding coverage closure`.
- Module owner: RVV plugin-owned `RouteOperandBindingPlan` coverage for
  remaining current executable RVV routes that still materialize operands or
  mirrors outside the shared binding contract.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `450b5e9a rvv: add route operand ABI binding contract`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- Commit `450b5e9a` introduced `RVVRouteOperandBindingPlan` and converted
  `macc_add`, `strided_load_unit_store`, and `unit_load_strided_store`.
- Current code has binding plan IDs only for:
  - `rvv-route-operand-binding:macc_add.v1`
  - `rvv-route-operand-binding:strided_load_unit_store.v1`
  - `rvv-route-operand-binding:unit_load_strided_store.v1`
- `verifyRVVSelectedBodyEmitCRouteDescription` currently requires those three
  routes to carry a binding plan and requires every other route to carry none.
- Provider emission still has ad-hoc materialized operand expressions for
  unconverted active executable routes, including the brief's minimum targets:
  `scalar_broadcast_add`, `standalone_reduce_add`, `masked_unit_store`, and
  `computed_masked_strided_store`.
- Specs require the authority chain to remain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> real
  `ssh rvv` evidence for runtime/correctness claims.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI roles only.
  RVV route operands must be derived and validated by the RVV plugin from the
  typed body/config/runtime facts, not from route IDs, helper strings,
  artifact names, source-front-door metadata, descriptors, or common EmitC.

## Requirements

1. Inventory the active executable RVV routes and classify whether each route:
   - already consumes `RVVRouteOperandBindingPlan`;
   - is converted in this round;
   - remains intentionally unconverted with an active/non-active reason and
     continuation point.
2. Convert the current unconverted routes called out by the Direction Brief if
   they are still active production/default routes:
   - `scalar_broadcast_add`
   - `standalone_reduce_add`
   - `masked_unit_store`
   - `computed_masked_strided_store`
3. For every converted route, the binding plan must record:
   - logical operand name;
   - runtime ABI role and C parameter;
   - materialized load/store/call/control operand uses;
   - mirror/header/materialized fields that must match the same contract.
4. Provider emission must use the binding plan for actual materialized operand
   expressions for converted routes, including setvl/loop-control, load bases,
   scalar/call operands, source/destination stores, accumulator/output
   preservation, and byte-stride operands where applicable.
5. Route description mirrors, generated artifact metadata/header checks, and
   generated-bundle script expectations must derive from the same binding
   summary for converted routes.
6. Converted routes must fail closed on missing logical operands, duplicate
   logical operands, duplicate ABI roles, wrong role for a logical operand,
   missing materialized use, runtime ABI order mismatch, mirror/materialized
   operand mismatch, stale route-id authority, descriptor/direct-C/source-front
   door authority, and common/export semantic inference.
7. Keep common EmitC/export neutral. The common path may consume
   provider-built route payloads and mirrors but must not infer RVV operand
   roles, dtype, policy, intrinsic choices, or memory semantics.

## Acceptance Criteria

- [ ] The PRD, implement/check context, and task metadata describe this bounded
      RouteOperandBindingPlan adoption task.
- [ ] Current active executable RVV route inventory is recorded in task notes
      or PRD completion evidence.
- [ ] `scalar_broadcast_add` materialized lhs/scalar/out/n operands and mirrors
      derive from `RVVRouteOperandBindingPlan`.
- [ ] `standalone_reduce_add` materialized lhs/acc/out/n operands and mirrors
      derive from `RVVRouteOperandBindingPlan`.
- [ ] `masked_unit_store` materialized src/mask/dst/n operands and mirrors
      derive from `RVVRouteOperandBindingPlan`.
- [ ] `computed_masked_strided_store` materialized compare lhs/rhs, payload
      source, destination, runtime destination byte-stride, and n operands and
      mirrors derive from `RVVRouteOperandBindingPlan`.
- [ ] Positive structural tests prove converted routes carry binding plan IDs,
      summaries, and materialized operands from the contract.
- [ ] Negative fail-closed tests cover role swaps, missing runtime roles,
      duplicate roles, missing materialized uses, mirror/header mismatch, stale
      route-id authority, descriptor/direct-C/source-front-door authority, and
      common/export semantic inference where the current test surfaces can
      express those failures.
- [ ] Generated-bundle dry-runs pass for converted representative routes using
      existing counts, scalar values, masks, and byte-strides.
- [ ] Real `ssh rvv` PASS evidence exists for the converted representative
      routes when runtime/correctness is claimed.
- [ ] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, route-id authority, artifact-name authority,
      or common/export RVV semantic authority.
- [ ] Focused C++/lit/script checks, `check-tianchenrv`, `git diff --check`,
      task validation, finish/archive, clean git status, and one coherent
      commit are completed if the task finishes.

## Non-Goals

- No new Stage2 operation families or route coverage.
- No dtype/LMUL clone batches.
- No high-level Linalg/Vector/StableHLO/frontend lowering.
- No broad smoke dashboard or report-only completion.
- No helper-only cleanup as the main achievement.
- No unrelated route rewrite without an active consumer.
- No source-front-door positive RVV routes.
- No descriptor-driven compute, direct-C/source-export route restoration, or
  compatibility wrapper preserving old route authority.
- No movement of RVV operand semantics into common EmitC/export, target
  metadata, artifact names, route IDs, descriptors, ABI strings alone, tests,
  or exact intrinsic spelling.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused C++ tests for route operand binding plan validation.
4. Run focused lit/FileCheck tests for converted route plan mirrors, target
   artifacts, generated headers, EmitC materialization, and negative
   fail-closed cases.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   or generated-bundle expectations change.
6. Run generated-bundle dry-runs for the converted routes using existing
   representative cases:
   - `scalar_broadcast_add`, counts `7,16,23`, RHS scalars `-37,91`;
   - `standalone_reduce_add`, existing representative counts;
   - `masked_store`, counts `7,16,23`;
   - `computed_masked_strided_store`, counts `7,16,23`, byte strides `4,8,12`.
7. Run real `ssh rvv` evidence for the converted representative routes if the
   dry-runs pass.
8. Run active-authority scans over active RVV include/lib/script/test paths.
9. Run `git diff --check`.
10. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing target artifact and EmitC fixtures for scalar broadcast,
  standalone reduction, masked store, computed-mask byte-strided store, macc,
  and byte-strided memory routes.

## Definition Of Done

- The four bounded target routes consume `RVVRouteOperandBindingPlan` for real
  provider materialization and mirrors.
- Existing converted routes remain intact.
- Any unconverted active route is named with reason and continuation point.
- Focused checks, hardware evidence, active-authority scan, and
  `check-tianchenrv` are complete.
- Trellis task status is truthful, archived when complete, and one coherent
  commit records the work.

## Route Inventory And Scope Result

Already covered before this task:

- `macc_add`
- `strided_load_unit_store`
- `unit_load_strided_store`

Converted in this task:

- `scalar_broadcast_add`
- `standalone_reduce_add`
- `masked_unit_store`
- `computed_masked_strided_store`

Intentionally not converted in this bounded round:

- Existing arithmetic / compare / select / masked arithmetic routes:
  `add`, `sub`, `mul`, `cmp_select`, `computed_mask_select`, `masked_add`,
  `reduce_add`, `strided_add`.
- Existing indexed / segmented / masked movement routes:
  `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
  `masked_unit_load_store`, `computed_masked_unit_load_store`,
  `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`.
- Existing conversion / contraction routes:
  `widen_i32_to_i64`, `widen_i16_to_i32`, `widening_macc_add`,
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`,
  `computed_masked_strided_input_widening_dot_reduce_add`.

Reason: these routes are active executable surfaces, but this round was scoped
to the Direction Brief's remaining named bottleneck routes after the initial
`macc_add` and byte-strided memory proof. They are not made non-active and
should be handled by a follow-up `Stage2 RVV RouteOperandBindingPlan follow-up
for remaining arithmetic/indexed/segmented/contraction routes`.

## Implementation Results

- Added four RVV plugin-owned route operand binding plan IDs:
  - `rvv-route-operand-binding:scalar_broadcast_add.v1`
  - `rvv-route-operand-binding:standalone_reduce_add.v1`
  - `rvv-route-operand-binding:masked_unit_store.v1`
  - `rvv-route-operand-binding:computed_masked_strided_store.v1`
- Extended logical-operand-to-runtime-role validation for the four plans,
  including RHS scalar, standalone accumulator, runtime mask input, computed
  compare operands, source payload, destination, runtime `n`, and destination
  byte stride.
- Extended provider emission so converted routes bind actual materialized
  setvl/control, load, scalar call, mask load, source load, old destination
  load, store, and byte-stride operands through `RVVRouteOperandBindingPlan`.
- Kept target artifact/header mirrors driven by the same
  `routeOperandBindingSummary`; no common EmitC/export semantic inference was
  added.
- Compressed the computed-mask strided-store binding summary so it remains
  under the repository's 512-byte single-line artifact metadata limit while
  still recording logical operand, ABI role, C parameter, and materialized
  uses.
- Updated generated-bundle evidence expectations and target artifact fixtures
  for the four converted routes.
- Added C++ unit coverage for valid new binding plans and role-swap failures.

## Validation Results

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- Generated-bundle dry-run passed:
  - mode: `--pre-realized-selected-body`
  - ops: `scalar_broadcast_add`, `standalone_reduce_add`,
    `masked_unit_store`, `computed_masked_strided_store`
  - counts: `7,16,23`
  - RHS scalars: `-37,91`
  - byte strides: `4,8,12`
  - artifact root:
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-operand-binding-coverage-dry-run`
- Real `ssh rvv` generated-bundle run passed:
  - artifact root:
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-operand-binding-coverage-ssh`
  - `PASS op=scalar_broadcast_add counts=7,16,23 rhs_scalars=-37,91`
  - `PASS op=standalone_reduce_add counts=7,16,23 seeds=-11,17`
  - `PASS op=masked_unit_store counts=7,16,23`
  - `PASS op=computed_masked_strided_store counts=7,16,23 stride_bytes=4,8,12`
  - computed-mask strided store evidence reported computed mask use,
    byte-strided store use, selected destination writes, false-lane
    preservation, sentinel preservation, and tail preservation.
- `cmake --build build --target check-tianchenrv -j2` passed `261/261`.
- `git diff --check`
- Diff-only active-authority scan over touched active RVV include/lib/script
  and test paths found no new positive `RVVI32M1`, `rvv-i32m1`, finite
  positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/
  source-export, source-front-door, public exact intrinsic route authority,
  route-id authority, artifact-name authority, or common/export RVV semantic
  authority.
