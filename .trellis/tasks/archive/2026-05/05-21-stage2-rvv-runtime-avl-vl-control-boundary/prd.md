# Stage2 RVV runtime AVL/VL control boundary

## Goal

Make the runtime ABI/VL boundary explicit for the current RVV Stage2 executable
path. This task introduces or repairs a plugin-owned Runtime AVL/VL control
plan that is consumed by the existing `scalar_broadcast_add` and
`standalone_reduce_add` selected-body routes.

The target behavior is not a new math slice. The two existing executable
consumers must continue to run, but they should derive runtime `n`/AVL, setvl,
with_vl, tail policy, runtime ABI order, route mirrors, generated artifacts,
and execution evidence from the same validated RVV-owned control contract.

## Direction Source

- Direction title: `Stage2 RVV runtime AVL/VL control boundary`.
- Module owner: RVV plugin-owned runtime AVL/VL control plan consumed by
  existing executable selected-body routes.
- Bounded first production consumers: `scalar_broadcast_add` and
  `standalone_reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f55bbd81 rvv: add standalone reduction executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- `scalar_broadcast_add` is already an executable Stage2 RVV route with
  selected-body realization, route planning/provider, generated-bundle
  dry-run, and real `ssh rvv` evidence for runtime counts `7,16,23` and RHS
  scalars `-37,91`.
- `standalone_reduce_add` is already an executable Stage2 RVV route with
  selected-body realization, route planning/provider, generated artifacts, and
  real `ssh rvv` evidence for counts `7,16,23` and seeds `-11,17`.
- Current code has `RVVSelectedBodyConfigVLContract` in the RVV dialect config
  contract and target artifact metadata validation for config/runtime-VL
  mirrors.
- Current route planning has separate
  `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan` and
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`.
- The missing behavior is a shared runtime-control authority that both
  production consumers validate and consume before selected-body realization,
  route planning, provider mirrors, and artifact mirrors.

## Requirements

1. Add one plugin-owned Runtime AVL/VL control plan or equivalent contract for
   selected RVV bodies.
2. The plan must be derived from typed body/config/runtime facts, not route ids,
   artifact names, descriptors, test names, exact intrinsic spellings, or
   common EmitC/export logic.
3. The plan must validate the runtime AVL source:
   exactly one `runtime-element-count` ABI value, ABI parameter name `n`, C type
   `size_t`, target-export ownership, and use as the `tcrv_rvv.setvl` AVL.
4. The plan must validate VL structure: one `tcrv_rvv.setvl`, one
   `tcrv_rvv.with_vl`, `with_vl` consumes the visible setvl result, setvl
   dominates with_vl in the realized body, and no fixed/stale VL authority is
   accepted.
5. The plan must validate config/control policy: SEW/LMUL from typed config,
   matching setvl/with_vl metadata, tail agnostic and mask agnostic policy for
   the bounded consumers, known runtime VL contract id, remaining AVL metadata,
   pointer-advance metadata, and multi-VL support mirror.
6. `scalar_broadcast_add` selected-body realization must derive and consume the
   runtime control plan before creating `setvl/with_vl/load/splat/binary/store`.
7. `standalone_reduce_add` selected-body realization must derive and consume
   the same runtime control plan before creating
   `setvl/with_vl/load/standalone_reduce/store`.
8. Route planning for both consumers must consume the runtime control plan and
   apply its fields to the provider route description before provider/header
   and target artifact mirrors are accepted.
9. Provider and target artifact mirrors may report runtime AVL/VL facts only
   after validation. Mirrors must remain mirrors.
10. Common EmitC/export must remain neutral and must not infer runtime n, VL,
    tail policy, operation kind, dtype, intrinsic, or route support.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe the bounded
      runtime AVL/VL control-boundary task.
- [x] A plugin-owned Runtime AVL/VL control plan or equivalent contract exists.
- [x] `scalar_broadcast_add` selected-body realization consumes the shared
      control plan before materializing the realized body.
- [x] `standalone_reduce_add` selected-body realization consumes the shared
      control plan before materializing the realized body.
- [x] Route planning for both consumers consumes the shared control plan and
      provider/artifact mirrors include validated runtime-control facts only.
- [x] Negative fail-closed coverage exists for missing/wrong runtime `n` role,
      duplicate or unused runtime `n`, wrong ABI name/type/order, fixed or
      stale VL authority, with_vl without validated setvl, invalid tail/mask
      policy, stale route-id authority, and common/export semantic inference
      where these cases are meaningful for this bounded slice.
- [x] Generated-bundle dry-run passes for `scalar_broadcast_add` and
      `standalone_reduce_add` at counts `7,16,23`.
- [x] Real `ssh rvv` generated-bundle runs pass for both representative routes,
      proving runtime `n`, multi-VL behavior, scalar RHS/seed preservation, and
      tail/output sentinel preservation.
- [x] Active-authority scan confirms no positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, or common/export RVV semantic authority is
      introduced.
- [x] Focused build/lit/C++/script checks, `check-tianchenrv`,
      `git diff --check`, task validation, archive/finish, clean git status,
      and one coherent commit are completed if the task finishes.

## Non-Goals

- No new Stage2 math coverage.
- No dtype/LMUL clone batch.
- No broad route matrix or dashboard.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive route.
- No rewrite of every RVV family; later consumers should be left as explicit
  continuation notes.
- No movement of runtime AVL/VL semantics into common EmitC/export, target
  metadata, route ids, artifact strings, descriptors, exact intrinsic
  spellings, or test names.

## Validation Plan

1. Validate task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-stage2-rvv-runtime-avl-vl-control-boundary`
2. Build focused targets:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
3. Run focused C++ tests for RVV dialect/plugin/construction/target export.
4. Run focused FileCheck tests for scalar-broadcast and standalone-reduction
   selected-body realization, route-plan/provider mirrors, target header
   export, EmitC materialization, and negative fail-closed cases.
5. Run script checks:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
6. Run generated-bundle dry-runs:
   - `scalar_broadcast_add`, counts `7,16,23`, RHS scalars `-37,91`
   - `standalone_reduce_add`, counts `7,16,23`, seeds `-11,17`
7. Run real `ssh rvv` generated-bundle correctness for the same bounded
   consumers.
8. Run active-authority scans over touched active RVV include/lib/script/test
   paths.
9. Run `git diff --check`.
10. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-standalone-reduction-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-scalar-broadcast-elementwise-family-consolidation/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice/prd.md`

Relevant journal entries read:

- `.trellis/workspace/codex/journal-12.md` scalar-broadcast family closeout.
- `.trellis/workspace/codex/journal-12.md` standalone reduction closeout.
- `.trellis/workspace/codex/journal-11.md` original vector-scalar broadcast
  executable closeout.

Initial code surface inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

## Definition Of Done

- Runtime AVL/VL control is a shared RVV plugin-owned contract for the bounded
  scalar-broadcast and standalone-reduction consumers.
- Current route-supported and executable evidence for both consumers remains
  passing.
- No legacy/source/descriptor/common-export authority is introduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
