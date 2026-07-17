# Stage2 RVV pre-realized runtime-scalar-cmp masked segment2 store ABI

## Goal

Complete one bounded Stage 2 RVV workflow submodule: a selected pre-realized
runtime-scalar-cmp masked segment2 store body must be realized by the RVV
plugin into the same typed runtime-scalar splat/compare mask, masked segment2
store, field payload, inactive-lane policy, dtype/config/policy, runtime
AVL/VL, provider route facts, EmitC route, target artifact ABI, generated
bundle, and `ssh rvv` evidence already proven for the explicit selected-body
path.

## What I Already Know

* Current HEAD is `9a611e08 rvv: add runtime scalar cmp segment2 store ABI`.
* The archived task
  `.trellis/tasks/archive/2026-06/06-07-rvv-runtime-scalar-cmp-segment2-store-abi/`
  completed the explicit selected-body path for
  `runtime_scalar_cmp_masked_segment2_store_unit_load`.
* That completed path carries runtime ABI order
  `lhs,rhs_scalar,src0,src1,dst,n`, runtime scalar splat/compare mask
  construction, field0/field1 payload roles, inactive old-destination
  preservation, provider-owned segment2 route planning, target validation,
  generated bundle ABI evidence, and `ssh rvv` correctness.
* The previous final continuation point was explicitly the pre-realized
  runtime-scalar-cmp masked segment2 store realization/evidence path.
* Specs require pre-realized selected bodies to be consumed by an RVV
  plugin-local selected-body realization owner before route-family analysis,
  statement planning, `TCRVEmitCLowerableRoute` construction, common EmitC, or
  target artifact export.
* Segment2 target export must rebuild and validate provider route facts; route
  ids, emission-plan metadata, artifact names, ABI strings, and common EmitC
  must remain mirrors or neutral carriers only.

## Requirements

* Scope is one route family only:
  `runtime_scalar_cmp_masked_segment2_store_unit_load`.
* Add or prove the production path:
  selected pre-realized RVV body -> RVV selected-body realization owner ->
  realized typed `tcrv_rvv` setvl/splat/compare/masked segment2 store body ->
  RVV route-family/provider validation -> `TCRVEmitCLowerableRoute` -> common
  EmitC materialization -> target artifact export -> generated bundle ABI.
* The pre-realized op must derive runtime scalar splat and compare mask facts
  from typed body/config/runtime facts, including `rhs_scalar` ABI binding and
  runtime AVL/VL, not from route ids, artifact names, test names, ABI strings,
  descriptor residue, source-front-door markers, or common EmitC inference.
* Preserve the explicit selected-body path and its generated ABI contract.
* Fail closed before provider route construction or artifact acceptance if any
  required pre-realized boundary fact is missing or stale, including runtime
  scalar binding, scalar splat source, compare mask producer/source, field0 or
  field1 payload role, inactive-lane policy, ABI order, header/prototype
  binding, dtype/config/policy, runtime AVL/VL, or masked segment2 store
  statement facts.
* Keep Common EmitC/export neutral: they may consume provider-built payloads
  and mirrors only, not synthesize RVV route semantics.

## Acceptance Criteria

* [x] A positive pre-realized fixture reaches materialized selected boundary,
      emission plan, target artifact export, generated bundle compile, and
      `ssh rvv` correctness when executable behavior is claimed.
* [x] The realized body removes or consumes the pre-realized body before
      provider route facts are collected.
* [x] Positive generated evidence exposes `lhs,rhs_scalar,src0,src1,dst,n`
      ABI/header participation, scalar splat/compare producer facts, compare
      predicate, mask-tail/active-inactive policy, field0/field1 payload roles,
      segment count, runtime AVL/VL, and provider-supported mirrors.
* [x] Target artifact validation consumes provider-owned route facts and
      rejects at least one stale pre-realized boundary fact relevant to this
      seam.
* [x] The existing explicit selected-body runtime-scalar-cmp masked segment2
      store fixture and generated-bundle dry-run continue to pass.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass after C++ changes.
* [x] Relevant focused lit/generated-bundle tests pass.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, or
      source-front-door route authority.
* [x] `git diff --check`, `git diff --cached --check`, and clean final
      `git status --short` are checked.

## Definition of Done

* The pre-realized runtime-scalar-cmp masked segment2 store executable
  artifact/ABI seam is implemented end to end, or the task remains open with a
  precise production blocker and exact next continuation point.
* Trellis task status/context and the codex workspace journal record the
  outcome truthfully.
* One coherent commit is created only after the bounded module is complete and
  verified.

## Out of Scope

* No broad segment2 or memory matrix.
* No dtype/LMUL clone batch.
* No runtime-scalar-cmp MAcc, product, reduction, dequant, clamp, conversion,
  compare/select, or high-level frontend rework.
* No Linalg/Vector/StableHLO source-front-door positive route.
* No performance tuning database, dashboard, index, report-only work, or
  broad smoke matrix.
* No Common EmitC invention of RVV semantics.
* No descriptor-driven, direct-C, source-export, or legacy i32 route authority.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-runtime-scalar-cmp-segment2-store-abi/prd.md`
* `.trellis/workspace/codex/journal-24.md` session 514

Likely implementation and evidence files from the direction brief:

* `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-segment2-store-dry-run.test`

Completion notes:

* Added `tcrv_rvv.typed_runtime_scalar_computed_mask_segment2_store_pre_realized_body`.
* Added RVV segment2 selected-body realization owner dispatch and owner-local
  validation for runtime scalar RHS role/type, ABI order, typed config, field
  roles, inactive-lane policy, and runtime AVL/VL.
* Added positive target fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store.mlir`
  with stale mask-producer target export rejection.
* Added generated-bundle dry-run fixture
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-segment2-store-dry-run.test`.
* Non-dry-run evidence:
  `/tmp/tianchenrv-rvv-prerealized-rtseg-store-ssh/pre-realized-runtime-scalar-cmp-segment2-store-ssh/evidence.json`.
