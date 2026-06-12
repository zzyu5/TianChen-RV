# Stage2 RVV runtime-scalar-cmp masked indexed gather-load ABI boundary

## Goal

Complete one bounded Stage 2 RVV workflow submodule: selected or
pre-realized runtime-scalar-cmp masked indexed gather-load bodies must carry
runtime scalar comparison, compare-produced mask facts, indexed gather
address/index roles, inactive-lane passthrough/result preservation,
dtype/SEW/LMUL/config/policy, runtime AVL/VL, RVV plugin-owned realization and
route validation, EmitC materialization, target artifact export, generated
bundle ABI, and executable evidence through the same provider-owned route
boundary.

## What I Already Know

* Session-start repository state was clean on `main`; recent HEAD is
  `6cfb7950 rvv: add runtime scalar cmp segment2 load ABI`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief before any source edits.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-rvv-prerealized-runtime-scalar-cmp-segment2-load-abi/`
  completed runtime-scalar-cmp masked segment2 load through RVV realization,
  provider facts, target validation, generated bundle ABI, and `ssh rvv`
  evidence.
* Existing indexed gather support includes base indexed gather and
  computed-mask indexed gather fixtures, generated-bundle dry-run tests, and a
  provider-owned computed-mask indexed route validation contract.
* Current inventory shows computed-mask indexed gather is vector-compare based;
  the runtime-scalar-cmp indexed gather variant is not yet present as a proven
  selected/pre-realized executable artifact ABI boundary.
* Specs require RVV semantics to come from typed or realized `tcrv_rvv`
  body/config/capability/runtime facts. Route ids, metadata, helper names,
  test names, ABI strings, descriptors, artifact names, and common EmitC
  behavior are mirrors or neutral carriers only.

## Requirements

* Scope is one route family only:
  `runtime_scalar_cmp_masked_indexed_gather_load_unit_store`.
* Add or repair the production path:
  selected/pre-realized RVV body -> RVV selected-body realization owner ->
  realized typed `tcrv_rvv` setvl/scalar-splat/compare/index-load/
  masked-indexed-load/old-destination-passthrough/store body ->
  RVV route-family/provider validation -> `TCRVEmitCLowerableRoute` ->
  common EmitC materialization -> target artifact export -> generated bundle
  ABI -> `ssh rvv` correctness when executable behavior is claimed.
* The selected/pre-realized body must structurally carry runtime scalar
  comparison facts, compare predicate, mask producer/source facts, data source
  role, index source role, old destination passthrough role, output role, ABI
  order, runtime AVL/VL, dtype/SEW/LMUL/config, index EEW/offset unit, and
  mask/tail/inactive-lane policy.
* The RVV plugin owner must derive or validate route/type/header/intrinsic
  facts from typed body/config/capability/runtime facts, and fail closed with
  targeted diagnostics if any executable-boundary fact is missing or stale.
* Common EmitC/export must remain neutral: they may consume provider-built
  routes and explicit mirror fields, but must not invent RVV semantics,
  intrinsic spelling, dtype, memory form, index roles, passthrough policy, or
  output ordering.
* Preserve existing base indexed gather, computed-mask indexed gather/scatter,
  and runtime-scalar segment2 behavior unless live code proves a direct shared
  bug fix is required.

## Acceptance Criteria

* [ ] Positive explicit and pre-realized runtime-scalar-cmp masked indexed
      gather-load evidence reaches materialized selected boundary, emission
      plan, target artifact export, generated bundle compile, and `ssh rvv`
      correctness when executable behavior is claimed.
* [ ] Realization, if needed, consumes the pre-realized body before provider
      route facts are collected.
* [ ] Positive generated evidence exposes runtime scalar ABI binding, compare
      predicate construction, compare-produced mask facts, mask and
      inactive-lane policy, indexed data source, index source/EEW/offset unit,
      old destination passthrough preservation, output store role, runtime
      AVL/VL, dtype/config, header/prototype binding, and explicit
      provider-supported mirrors.
* [ ] Focused fail-closed evidence rejects at least one stale or missing
      executable-boundary fact, such as runtime scalar ABI binding, compare
      producer source, index role, inactive-lane passthrough policy,
      header/prototype binding, generated C type, ABI value mapping, route
      family validation contract, or unsupported executable route claim.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [ ] Relevant focused lit/generated-bundle tests pass.
* [ ] `scripts/rvv_generated_bundle_abi_e2e.py` self-test passes if the script
      changes.
* [ ] Bounded old-authority scan over touched files and added diff lines finds
      no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, or
      source-front-door route authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are checked.

## Definition of Done

* The runtime-scalar-cmp masked indexed gather-load executable artifact/ABI
  seam is implemented end to end, or the task remains open with a precise
  production blocker and exact next continuation point.
* Trellis task status/context and the codex workspace journal record the
  outcome truthfully.
* One coherent commit is created only after the bounded module is complete and
  verified.

## Out of Scope

* No indexed scatter-store in this round.
* No broad indexed-memory matrix.
* No dtype/LMUL clone batch.
* No segment2 rework except as reference.
* No strided-memory, base-memory, reduction, MAcc, compare/select, conversion,
  or unrelated mask-route expansion.
* No high-level Linalg, Vector, StableHLO, per-Linalg route authority, or
  source-front-door positive route.
* No dashboard, index, report-only, helper-only, or broad smoke-test work as
  the main achievement.
* No common EmitC invention of RVV semantics.
* No descriptor-driven, direct-C, source-export, or legacy i32 route authority.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/validation/index.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-prerealized-runtime-scalar-cmp-segment2-load-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-prerealized-runtime-scalar-cmp-segment2-load-abi/implement.jsonl`

Likely implementation and evidence files from the direction brief and repo
inventory:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-gather-load-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-gather-load-dry-run.test`
