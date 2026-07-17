# Stage1 Gate A RVV route identity cleanup

## Goal

Close the bounded Stage1 Gate A residues in the active RVV route identity path. The selected `tcrv.exec` RVV variant must route through explicit generic typed `tcrv_rvv` body structure, while production route ids, runtime ABI names, artifact route names, config-contract names, and pre-realized selected-body entry points stop presenting `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` as route authority.

## What I Already Know

- Current HEAD is `1f48162 rvv: add generic reduction route skeleton`; the worktree was clean before task creation.
- `.trellis/.current-task` did not exist, so this task was created from the Hermes Direction Brief and started as `.trellis/tasks/05-19-stage1-gate-a-rvv-route-identity-cleanup`.
- Specs require RVV Stage1 to replace or fail-close active paths that treat bounded `i32m1`, `RVVI32M1*`, finite `tcrv_rvv.i32_*`, `rvv-i32m1-*` ids, artifact names, source-front-door metadata, descriptors, or exact `__riscv_*_i32m1` spellings as architecture authority.
- The current generic typed route surface already exists for `tcrv_rvv.load`, `tcrv_rvv.broadcast_load`, `tcrv_rvv.binary`, `tcrv_rvv.compare`, `tcrv_rvv.select`, `tcrv_rvv.reduce`, and `tcrv_rvv.store`.
- Initial scan found active production constants and positive Target/RVV fixtures still expecting `rvv-i32m1-*` route/header/ABI/bundle/config names.
- Initial scan found positive pre-realized artifact fixtures using `tcrv_rvv.i32_binary_pre_realized_body`.
- The RVV source-front-door pass is already fail-closed; its `i32m1` names are legacy negative-path identifiers, not positive artifact generation.

## Requirements

- Rename or demote active positive RVV route identities so generated route ids, runtime ABI names, target artifact route names, header route names, bundle group names, and config contract ids are generic selected-body / typed-body names rather than `rvv-i32m1-*` names.
- Replace the positive pre-realized selected-body op name with a generic typed pre-realized binary body surface, or keep any legacy `i32_binary_pre_realized_body` use negative-only.
- Keep retained i32 add/sub/mul behavior only as ordinary `!tcrv_rvv.vector<i32, "m1">` / SEW32 / LMUL / policy facts consumed by the RVV plugin provider.
- Keep source-front-door RVV paths fail-closed; do not add source-front-door positive routes.
- Do not add Stage2 coverage, new dtype/LMUL families, dashboards, performance tuning, or compatibility wrappers that preserve old i32 route authority.

## Acceptance Criteria

- [x] Positive `test/Target/RVV` generated-artifact fixtures no longer require `rvv-i32m1-*` route/header/ABI/bundle names as maturity targets.
- [x] Positive pre-realized selected-body fixtures use a generic typed pre-realized body entry point, or the old `tcrv_rvv.i32_binary_pre_realized_body` is only present in negative/fail-closed coverage.
- [x] RVV construction protocol and EmitC route provider emit generic selected-body route ids/runtime ABI names/artifact route names after provider validation.
- [x] RVV config-contract active APIs and metadata use generic selected-body naming; any remaining `RVVI32M1` names are not active production authority.
- [x] Focused RVV dialect/config/plugin/provider/target tests pass.
- [x] A targeted Stage1 Gate A scan over `include/TianChenRV`, `lib/Plugin/RVV`, `lib/Dialect/RVV`, and `test/Target/RVV` classifies remaining `rvv-i32m1`, `RVVI32M1`, and `i32_binary_pre_realized_body` matches as negative/deprecated/intrinsic leaf/mirror only, or reports exact blockers.

## Out Of Scope

- Stage2 coverage expansion beyond preserving current generic arithmetic, compare/select, broadcast, and reduction skeleton behavior.
- Source-front-door positive RVV routes.
- `ssh rvv` runtime/correctness/performance claims.
- Global dashboard/report-only work.
- New Python compiler implementation surfaces.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/index.md`.
- Main files to inspect/change: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`, `lib/Dialect/RVV/IR/RVVConfigContract.cpp`, `lib/Dialect/RVV/IR/RVVDialect.cpp`, `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`, `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`, and positive `test/Target/RVV` fixtures.
