# Stage2 RVV closure-gated binary broadcast and masked elementwise coverage expansion

## Goal

Expand the corrected typed low-level RVV route-supported surface for one
bounded Stage2 module: generic binary `add`, `sub`, and `mul` with RHS scalar
broadcast, plus mask-aware binary elementwise `add`, `sub`, and `mul`. The new
coverage must flow through typed `tcrv_rvv` body facts, RVV plugin-local
selected-body realization, RVV provider route construction, and the mandatory
`RVVRouteOperandBindingPlan` closure gate before EmitC/target artifact output.

## Direction Brief

- Direction title: `Stage2 RVV closure-gated binary broadcast and masked elementwise coverage expansion`.
- Module owner: RVV plugin-owned route-supported expansion for generic typed
  binary vector-scalar broadcast and masked elementwise routes.
- Initial repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `8f2daef3 rvv: enforce operand binding closure gate`.
- There was no `.trellis/.current-task`; this task was created from the Hermes
  direction brief.

## Current Active Inventory

Inventory source is current HEAD route planning/provider/selected-body
realization plus target/script fixtures, not old status reports:

- Generic vector-vector binary `add`, `sub`, and `mul` already route through
  `tcrv_rvv.binary` and operation-specific binding plans.
- Existing `explicit-selected-body-artifact-broadcast-{add,sub,mul}.mlir`
  files are legacy `tcrv_rvv.i32_*` fail-closed fixtures, not positive Stage2
  broadcast authority.
- Corrected RHS scalar broadcast exists as a typed/pre-realized
  `rhs-scalar-broadcast` route only for add:
  `pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`,
  `ScalarBroadcastAdd`, `rvv-route-operand-binding:scalar_broadcast_add.v1`.
- `tcrv_rvv.splat` is already the corrected generic typed scalar-broadcast
  dataflow op, but route planning and selected-body realization currently
  reject scalar-broadcast `sub` and `mul`.
- Mask-aware binary elementwise exists only for add:
  `tcrv_rvv.masked_binary {kind = "add"}`, `TypedMaskedBinaryPreRealizedBodyOp`
  `op_kind = "masked_add"`, `MaskedAdd`, and
  `rvv-route-operand-binding:masked_add.v1`.
- `tcrv_rvv.masked_binary` verifier, selected-body realization, route
  analysis, target leaf derivation, provider materialization, generated-bundle
  expectations, and fixtures currently reject or hard-code non-add masked
  binary operation kinds.

## Requirements

1. Add or repair coherent route support for RHS scalar-broadcast
   `add`, `sub`, and `mul` using the existing generic typed surface:
   runtime RHS scalar ABI value -> `tcrv_rvv.splat` -> `tcrv_rvv.binary`.
2. Add or repair coherent route support for masked binary elementwise
   `add`, `sub`, and `mul` using the existing generic typed surface:
   compare-produced mask plus passthrough vector -> `tcrv_rvv.masked_binary`.
3. Keep operation kind, scalar/vector operand roles, mask/passthrough facts,
   runtime n/AVL, dtype/config, materialized uses, header mirrors, and target
   artifact mirrors derived from typed body structure and matching
   `RVVRouteOperandBindingPlan` closure.
4. Use already-supported `add` cases as regression anchors; do not rewrite
   unrelated reduction, contraction, conversion, segmented/indexed movement,
   or masked memory movement routes.
5. Unsupported op kinds, unsupported dtype/LMUL forms, missing or swapped
   scalar/vector roles, wrong mask source, missing runtime n/AVL role, stale
   or wrong plan ID, header/mirror mismatch, materialized-use mismatch, and
   route-id/helper-string/source-front-door/descriptor fallback must fail
   closed with targeted diagnostics.
6. Keep common EmitC/export neutral: they may consume provider-built route
   payload and mirrors, but must not choose RVV operation, dtype, mask policy,
   intrinsic, or scalar-broadcast semantics.

## Acceptance Criteria

- [x] Current binary/broadcast/masked-elementwise route support is documented
      in this PRD from current code, not old artifacts.
- [x] `TypedBinaryPreRealizedBodyOp` and plugin realization accept
      `rhs-scalar-broadcast` for `add`, `sub`, and `mul` at the bounded
      supported config.
- [x] Explicit realized bodies using `tcrv_rvv.splat` and pre-realized bodies
      prove scalar-broadcast `sub` and `mul` are positive routes with correct
      plan IDs, operation mirrors, runtime ABI order, header signatures, and
      materialized operand summaries.
- [x] `tcrv_rvv.masked_binary`, `TypedMaskedBinaryPreRealizedBodyOp`, and
      plugin realization accept bounded masked `add`, `sub`, and `mul` while
      preserving compare-produced mask and passthrough-vector authority.
- [x] Explicit realized bodies and pre-realized bodies prove masked `sub` and
      `mul` are positive routes with correct plan IDs, operation mirrors,
      runtime ABI order, header signatures, mask/passthrough metadata, and
      materialized operand summaries.
- [x] `RVVRouteOperandBindingPlan` expected-plan lookup, role validation,
      route description mirrors, provider operand lookup, and target artifact
      mirror validation cover the new route kinds before lowerable-route
      construction.
- [x] Negative tests cover wrong binary kind, unsupported scalar-broadcast
      kind if any remains out of scope, missing scalar RHS role, RHS scalar
      role swap, wrong masked op kind, mask/passthrough mismatch, stale/wrong
      plan ID or summary, materialized-use mismatch, and legacy/source/front
      door fallback where touched.
- [x] Generated-bundle dry-runs cover representative explicit and pre-realized
      scalar-broadcast and masked routes with value-distinguishing inputs.
- [x] Real `ssh rvv` generated-bundle evidence passes for representative new
      scalar-broadcast and masked routes, including tail/sentinel preservation.
- [x] Active-authority scan confirms no new positive `RVVI32M1`,
      `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export/source-front-door, public exact
      intrinsic, artifact-name, or common/export RVV semantic authority.
- [x] `git diff --check` and `check-tianchenrv` pass.
- [x] Task status/journal/archive and one coherent commit are completed if the
      module is finished.

## Non-Goals

- No reductions, contractions, conversions, segmented/indexed movement, masked
  memory movement, new dtype/LMUL clone batch, high-level Linalg/Vector/
  StableHLO frontend lowering, source-front-door positive route, dashboard, or
  report-only work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor,
  direct-C, source-export, or artifact-name route authority.
- No ad-hoc provider fallback around the operand-binding closure gate.
- No common EmitC/export semantic inference for RVV operation, dtype/config,
  scalar-broadcast, mask, or intrinsic choices.

## Validation Plan

1. Build focused tools/tests if needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
2. Run focused RVV plugin C++ tests:
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
3. Run focused lit/FileCheck for new/changed RVV target and dialect tests.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
5. Run generated-bundle dry-runs for representative explicit and pre-realized
   scalar-broadcast/masked routes.
6. Run real `ssh rvv` generated-bundle checks for representative new routes.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
  plus its `implement.jsonl` and `check.jsonl`.
- Primary source surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Target/RVV/`.

## Completion Notes

- Added route-supported scalar-broadcast `sub` and `mul` as ordinary generic
  typed binary routes with RHS `tcrv_rvv.splat`, per-op binding plan IDs,
  route mirrors, materialized operand summaries, generated target artifacts,
  and explicit/pre-realized selected-body fixtures.
- Added route-supported masked elementwise `sub` and `mul` through
  `tcrv_rvv.masked_binary`, compare-produced mask authority,
  passthrough-vector inactive-lane semantics, per-op binding plan IDs, route
  mirrors, materialized operand summaries, generated target artifacts, and
  explicit/pre-realized selected-body fixtures.
- The active-authority diff scan is clean for new `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export/source-front-door, and artifact-name
  route authority. Exact RVV intrinsic spellings remain provider-owned derived
  leaf mappings/mirrors after typed body/config/operation analysis; common
  EmitC/export does not infer RVV semantics.
- Dry-run artifact roots:
  `artifacts/tmp/rvv_stage2_binary_broadcast_masked/dry-explicit` and
  `artifacts/tmp/rvv_stage2_binary_broadcast_masked/dry-pre-realized`.
- Real hardware artifact root:
  `artifacts/tmp/rvv_stage2_binary_broadcast_masked/ssh-pre-realized`.

## Definition Of Done

- The production RVV provider route path supports closure-gated scalar
  broadcast and masked elementwise `add/sub/mul` on the corrected typed
  surface.
- Positive and negative tests prove the new route behavior and fail-closed
  boundaries.
- Generated-bundle and `ssh rvv` evidence are recorded for representative new
  routes.
- Trellis task is truthful, archived when complete, and committed as one
  coherent change.
