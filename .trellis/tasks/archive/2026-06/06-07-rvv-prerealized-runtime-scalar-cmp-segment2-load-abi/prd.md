# Stage2 RVV runtime-scalar-cmp masked segment2 load executable artifact ABI boundary

## Goal

Complete one bounded Stage 2 RVV workflow submodule: selected or
pre-realized runtime-scalar-cmp masked segment2 load bodies must carry runtime
scalar comparison, compare-produced mask facts, segment2 interleaved source
load, field0/field1 output roles, inactive-lane passthrough preservation,
dtype/SEW/LMUL/config/policy, runtime AVL/VL, RVV plugin-owned selected-body
realization and route validation, EmitC materialization, target artifact
export, generated bundle ABI, and executable evidence through the same
provider-owned route seam.

## What I Already Know

* Current HEAD is expected to include `362291d8 rvv: add pre-realized runtime scalar cmp segment2 store`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-rvv-prerealized-runtime-scalar-cmp-segment2-store-abi/`
  completed the write-side runtime-scalar-cmp masked segment2 store seam.
* Store-side completion added a pre-realized body op, RVV owner-local selected
  body realization, route-family facts, generated bundle support, dry-run lit
  coverage, and non-dry-run `ssh rvv` evidence for counts 0, 1, 16, 17, and
  257 with RHS scalars -37 and 91.
* The read-side counterpart must drive runtime scalar comparison into a
  masked segment2 load, then preserve inactive output lanes through old
  field passthrough values while writing field0 and field1 outputs.
* Existing computed-masked segment2 load fixtures provide the bounded
  comparison point for load ABI and field-order behavior.
* Specs require RVV semantics to be derived from typed or realized
  `tcrv_rvv` body/config/capability/runtime facts. Route ids, metadata,
  helper names, test names, ABI strings, descriptors, artifact names, and
  common EmitC behavior are mirrors or neutral carriers only.

## Requirements

* Scope is one route family only: runtime-scalar-cmp masked segment2 load.
* Add or repair the production path:
  selected/pre-realized RVV body -> RVV selected-body realization owner ->
  realized typed `tcrv_rvv` setvl/splat/compare/masked segment2 load body ->
  RVV route-family/provider validation -> `TCRVEmitCLowerableRoute` ->
  common EmitC materialization -> target artifact export -> generated bundle
  ABI -> `ssh rvv` correctness when executable behavior is claimed.
* The selected/pre-realized body must structurally carry runtime scalar
  comparison facts, compare predicate, mask producer/source facts,
  interleaved segment2 source role, old field passthrough roles, out0/out1
  roles, ABI order, runtime AVL/VL, dtype/SEW/LMUL/config, and mask/tail
  policy.
* The RVV plugin owner must derive or validate route/type/header/intrinsic
  facts from the typed body/config/capability/runtime facts, and fail closed
  with targeted diagnostics if any executable-boundary fact is missing or
  stale.
* Common EmitC/export must remain neutral: they may consume provider-built
  routes and explicit mirror fields, but must not invent RVV semantics,
  intrinsic spelling, dtype, memory form, or output ordering.
* Preserve existing computed-masked segment2 load behavior and store-side
  runtime-scalar-cmp behavior unless the live code proves a direct shared
  bug fix is required.

## Acceptance Criteria

* [ ] Positive selected/pre-realized runtime-scalar-cmp masked segment2 load
      evidence reaches materialized selected boundary, emission plan, target
      artifact export, generated bundle compile, and `ssh rvv` correctness
      when executable behavior is claimed.
* [ ] Realization, if needed, consumes the pre-realized body before provider
      route facts are collected.
* [ ] Positive generated evidence exposes runtime scalar ABI binding,
      compare predicate construction, compare-produced mask facts, mask and
      inactive-lane policy, interleaved source role, old field passthrough
      preservation, field0/field1 output ordering, segment count, runtime
      AVL/VL, dtype/config, header/prototype binding, and explicit
      provider-supported mirrors.
* [ ] Focused fail-closed evidence rejects at least one stale or missing
      executable-boundary fact, such as runtime scalar ABI binding, compare
      predicate construction, mask/inactive-lane policy, field0/field1
      ordering, old passthrough preservation, source/output ABI order,
      header/prototype binding, generated C type, ABI value mapping, or
      unsupported executable route claim.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [ ] Relevant focused lit/generated-bundle tests pass.
* [ ] Bounded old-authority scan over touched files and added diff lines
      finds no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, or
      source-front-door route authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are checked.

## Definition of Done

* The runtime-scalar-cmp masked segment2 load executable artifact/ABI seam is
  implemented end to end, or the task remains open with a precise production
  blocker and exact next continuation point.
* Trellis task status/context and the codex workspace journal record the
  outcome truthfully.
* One coherent commit is created only after the bounded module is complete
  and verified.

## Out of Scope

* No broad segment2 matrix.
* No dtype/LMUL clone batch.
* No update/deinterleave/interleave expansion in this round.
* No MAcc, reduction, dequant, clamp, conversion, or unrelated memory-route
  rework.
* No high-level Linalg, Vector, StableHLO, per-Linalg route authority, or
  source-front-door positive route.
* No dashboard, index, report-only, helper-only, or broad smoke-test work as
  the main achievement.
* No common EmitC invention of RVV semantics.
* No descriptor-driven, direct-C, source-export, or legacy i32 route
  authority.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-prerealized-runtime-scalar-cmp-segment2-store-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-07-rvv-prerealized-runtime-scalar-cmp-segment2-store-abi/implement.jsonl`

Likely implementation and evidence files from the direction brief:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
* `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-load.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-load-dry-run.test`
* Store-side runtime-scalar-cmp fixtures only as bounded references.
