# Stage1 legacy finite RVV i32 parse-only residue cleanup

## Goal

Clean the remaining legacy finite RVV i32 parse-only surface so it cannot look
like route, dtype, EmitC, selected-body, source-front-door, or artifact
authority. The corrected generic typed `tcrv_rvv` surface must remain the only
positive RVV production route surface for selected-body materialization.

## Direction Source

- Hermes Direction Brief: `Stage1 legacy finite RVV i32 parse-only residue cleanup`.
- Current HEAD before this task: `86ef664a rvv: realize pre-realized reduce add body`.
- Worktree before this task: clean.
- No previous current task existed; this task was created for the brief.

## Repository Facts Already Checked

- `.trellis/spec/index.md` defines the current RVV authority chain as
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level
  `tcrv_rvv` body -> RVV plugin legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` allows legacy explicit bodies
  only as parse/verify/fail-closed fixtures or rewritten generic typed surface
  instances, and forbids supported legacy `rvv-i32m1-*` compatibility routes.
- `.trellis/spec/lowering-runtime/emitc-route.md` says common EmitC must not
  treat `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m1`, exact `__riscv_*_i32m1` spellings, source-front-door
  markers, emission-plan status, or artifact metadata as route authority.
- `.trellis/spec/testing/mlir-testing-contract.md` permits positive generated
  artifact tests only for corrected generic typed `tcrv_rvv` routes and
  requires stale legacy authority tests to be negative/fail-closed.

## Initial Residue Inventory

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` still defines deprecated
  `!tcrv_rvv.i32m1`, `!tcrv_rvv.i32m1_mask`, `!tcrv_rvv.i32m2`, and
  `tcrv_rvv.i32_*` ops. These are described as deprecated parse-only
  inventory, but they still implement `TCRVEmitCLowerableOpInterface`, which
  can be misread as positive EmitC lowerable authority.
- `lib/Dialect/RVV/IR/RVVDialect.cpp` still contains legacy i32 verifier hooks
  and lowerable source-role methods for `I32LoadOp`, `I32BroadcastLoadOp`, and
  `I32StoreOp`.
- `test/Dialect/RVV/RVVDialectTest.cpp` still asserts that legacy
  `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store`
  implement the EmitC lowerable interface. This is active-looking residue and
  must be rewritten.
- `test/Dialect/RVV/dataflow.mlir` is already labeled as deprecated parser /
  verifier inventory, but its positive FileCheck surface needs to remain
  clearly non-route and non-artifact if retained.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already fail-closes any
  selected-body op whose name starts with `tcrv_rvv.i32_`.
- Exact `__riscv_*_i32m1` strings in the RVV provider and positive tests come
  from the current corrected generic typed i32/m1 route instance. They are
  target leaves derived by the RVV provider after typed body/config/runtime
  validation, not legacy route authority; this task must not delete that
  generic typed path.

## Requirements

- Remove `TCRVEmitCLowerableOpInterface` from retained legacy `tcrv_rvv.i32_*`
  ops.
- Remove obsolete lowerable interface method implementations for legacy i32 ops.
- Rewrite C++ dialect tests so positive lowerable-interface assertions target
  generic typed `tcrv_rvv.load/binary/store`, not legacy `i32_*` ops.
- Add or preserve a focused assertion that retained legacy i32 parser/verifier
  residue does not implement the EmitC lowerable interface.
- Clean misleading wording around `with_vl`, `status`, and
  `rvv_emitc_route_mapping` so those fields read as mirrors/provenance after
  selected body/provider decisions, not acceptance, dtype, compute, or route
  authority.
- Preserve RVV provider fail-closed behavior for legacy selected-body inputs.
- Preserve positive generic typed selected-body route materialization.

## Acceptance Criteria

- [x] Targeted residue scan over `include`, `lib`, `test`, and `.trellis/spec`
      is recorded before and after edits.
- [x] No retained `tcrv_rvv.i32_*` op implements
      `TCRVEmitCLowerableOpInterface`.
- [x] Remaining `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*` mentions are classified as
      deprecated parser/verifier inventory, negative/fail-closed tests, or
      historical/spec prohibition; they are not positive EmitC, selected-body,
      source-front-door, target artifact, correctness, runtime, or performance
      authority.
- [x] Generic typed RVV body paths still materialize through the RVV provider
      and common EmitC.
- [x] Common EmitC/export remains neutral; no family-specific mapping is moved
      into common code.
- [x] Focused RVV dialect / EmitC / target tests pass for touched behavior.
- [x] `git diff --check` passes.
- [x] If shared dialect/plugin/test behavior changes broadly, run
      `check-tianchenrv`.

## Out Of Scope

- No new Stage2 coverage classes.
- No dtype/LMUL expansion beyond preserving already-supported generic typed
  route instances.
- No source-front-door positive RVV route.
- No Template/Toy/TensorExtLite/future-plugin work.
- No broad dashboards, readiness states, report-only cleanup, or helper-only
  changes as the main achievement.
- No compatibility wrapper that keeps old i32 route authority alive.

## Validation Plan

- Targeted residue scan:
  `rg -n -e 'tcrv_rvv\.i32_' -e '!tcrv_rvv\.i32m' -e 'RVVI32M1' -e 'rvv-i32m1' -e 'source-front-door' -e 'source-artifact' -e 'status = "supported"' -e 'selected route' -e '__riscv_[A-Za-z0-9_]*_i32m1' include lib test .trellis/spec`
- Focused dialect/unit check for RVV dialect tests.
- Focused lit checks for legacy fail-closed and generic typed materialization.
- Positive smoke for at least one recent generic typed selected-body route.
- `git diff --check`.
- `check-tianchenrv` if the focused changes affect shared dialect/plugin/test
  behavior beyond this bounded RVV surface.

## Completion Notes

- Removed EmitC lowerable interface declarations from retained legacy
  `tcrv_rvv.i32_*` ODS ops.
- Removed obsolete `I32LoadOp`, `I32BroadcastLoadOp`, and `I32StoreOp`
  lowerable source-role method implementations.
- Reworked `RVVDialectTest.cpp` so positive lowerable-interface assertions use
  generic typed `tcrv_rvv.load`, `tcrv_rvv.binary`, and `tcrv_rvv.store`.
- Added a focused C++ assertion that retained legacy `i32_load`, `i32_add`,
  and `i32_store` parse but do not implement the EmitC lowerable interface.
- Updated RVV dialect wording around `with_vl`, `status`, and
  `rvv_emitc_route_mapping` so they read as mirrors/handoff facts after
  provider decisions, not acceptance or route authority.
- No Trellis spec update was needed in this round: the relevant specs already
  encode the Stage1 delete/fail-close policy and common EmitC neutrality.
