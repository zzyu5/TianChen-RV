# Stage2 RVV Computed-Masked Segment2 Executable Artifact ABI Boundary

## Goal

Prove that the existing pre-realized computed-masked segment2 load and
computed-masked segment2 store selected-body routes are real generated RVV
artifact ABI boundaries, not dry-run-only or metadata-authorized claims. The
bounded owner is the path from selected/pre-realized typed `tcrv_rvv` segment2
memory bodies through RVV provider-owned computed-mask/segment2 facts,
`TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact
validation, generated bundle ABI, and `ssh rvv` correctness evidence.

## What I Already Know

* The live repository is on `main`, with a clean status at session start and
  recent commit `232c16e4 rvv: prove segment2 unit memory artifact abi`.
* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes/user Direction Brief.
* The relevant specs require computed-mask segment2 load/store/update target
  validation to consume provider-owned fact surfaces, not route ids, artifact
  names, fixture names, descriptors, scripts, Common EmitC, candidate metadata
  mirrors, or exact RVV intrinsic spellings.
* The computed-mask segment2 load ABI order is
  `cmp_lhs,cmp_rhs,src,out0,out1,n`; the computed-mask segment2 store ABI order
  is `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
* Common EmitC only carries provider-built route payloads. It must not infer
  segment2 direction, computed-mask provenance, field roles, inactive-lane
  behavior, ABI/header/type facts, or RVV support.
* Initial audit found production code already has computed-mask segment2
  selected-body realization, segment2 route-family provider plans, segment2
  target route validation contracts, and C++ negative tests for many stale
  provider payload, route statement, and candidate mirror facts.
* The current generated-bundle tests for pre-realized computed-mask segment2
  load/store are dry-run tests. The bounded missing evidence appears to be
  non-dry-run executable `ssh rvv` evidence for the two load/store generated
  bundles, unless focused checks expose a production seam gap.

## Requirements

* Keep the task scoped to pre-realized computed-masked segment2 load
  unit-store and computed-masked segment2 store unit-load executable artifact
  ABI boundaries.
* Preserve the production ownership chain:
  typed `tcrv_rvv` body -> RVV computed-mask/segment2 planning owner ->
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC materializer ->
  target artifact export validator -> generated bundle -> `ssh rvv` evidence.
* Positive evidence must cover both pre-realized computed-mask segment2 routes
  through materialized selected boundary, emission plan, target artifact export,
  generated bundle compile, and `ssh rvv` correctness if executable behavior is
  claimed.
* If production code is dry-run-only or under-validated, add focused
  fail-closed coverage for at least one stale or missing executable-boundary
  fact such as field order, computed-mask provenance, inactive-lane policy,
  load/store source or destination channel, header/prototype binding, type
  mapping, ABI order, runtime AVL/VL, or provider-built route statement facts.
* Do not broaden into a segment2 matrix, dtype/LMUL clones, computed-mask
  segment2 update rework, plain segment2 unit-memory rework, source-front-door
  routes, Common EmitC semantic inference, high-level frontend work, or
  unrelated memory/reduction/contraction families.

## Acceptance Criteria

* [x] PRD records the audit finding and selected contract for this round.
* [x] Pre-realized computed-masked segment2 load generated bundle executes on
      `ssh rvv` with `dry_run: false`, `ssh_evidence: true`, materialized
      selected boundary, provider-derived ABI/header/type/binding facts, mask
      active/inactive coverage, field-order coverage, and source/tail sentinel
      preservation.
* [x] Pre-realized computed-masked segment2 store generated bundle executes on
      `ssh rvv` with `dry_run: false`, `ssh_evidence: true`, materialized
      selected boundary, provider-derived ABI/header/type/binding facts, mask
      active/inactive coverage, field-order coverage, source preservation, and
      destination tail sentinel preservation.
* [x] Focused production or test changes are made if audit/execution shows the
      computed-masked segment2 load/store seam is stale, missing, or
      under-validated.
      Audit, C++ validation tests, lit tests, and non-dry-run execution found
      no stale/missing seam in the computed-masked segment2 load/store
      production path, so no production source change was required.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests for explicit/pre-realized
      computed-masked segment2 load/store and direct pre-realized fail-closed
      shortcuts pass.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new legacy `i32m1`, descriptor, source-front-door, artifact-name, or
      exact-intrinsic-as-authority route drift.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean after commit, or the exact unfinished
      continuation point is recorded.

## Out of Scope

* No broad segment2 matrix.
* No dtype/LMUL clone batch.
* No computed-mask segment2 update rework except as reference.
* No plain unit-memory interleave/deinterleave rework except as reference.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or report-only closure.
* No source-front-door positive route.
* No Common EmitC invention of RVV semantics.
* No mass rewrite of unit-stride masked memory, product/dequant, contraction,
  standalone reduction, compare/select, conversion, or unrelated route
  families.

## Technical Notes

* Direction source: Hermes/user Direction Brief on 2026-06-06.
* Relevant specs:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
* Prior reference task:
  * `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-segment2-unit-memory-artifact-abi/`
* Primary production files audited:
  * `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`
  * `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  * `scripts/rvv_generated_bundle_abi_e2e.py`
  * relevant Script and Target/RVV computed-masked segment2 fixtures

## Audit Finding And Selected Contract

The initial audit found that the current production path already routes
computed-masked segment2 load/store through the required owner chain:

```text
pre-realized typed tcrv_rvv body
  -> RVV plugin-local selected-body realization
  -> computed-mask memory and segment2 route-family planning owners
  -> provider-built TCRVEmitCLowerableRoute
  -> segment2 target artifact validation contract
  -> generated bundle ABI and harness
```

The target validator already checks provider-owned route id, runtime AVL/VL
contract, computed-mask route-family plan, mask/tail policy owner, compare
predicate, mask role/source/form, inactive-lane contract, passthrough/no-write
layout, segment layout, source/destination memory forms, field roles/names,
segment tuple/load/store/extract facts, runtime ABI order, operand-binding
summary, required headers, type mappings, ABI mappings, and statement-plan
shape. The bounded missing evidence selected for this round is non-dry-run
`ssh rvv` generated-bundle correctness evidence for computed-masked segment2
load and store, plus focused source/test changes only if those checks expose a
real seam gap.

## Current Phase

Finish.

## Evidence

* The production audit found no source seam requiring a production code change:
  computed-masked segment2 load/store already flow through the RVV
  selected-body realization owner, computed-mask/segment2 route-family plan
  owners, provider-built `TCRVEmitCLowerableRoute`, segment2 route validation
  contract, and target artifact mirror checks. The existing C++ target artifact
  test already exercises positive computed-mask segment2 load/store/update
  route validation contracts and many stale provider payload, route statement,
  ABI/header/binding, mask, field, and mirror mutations.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* From `build/test`,
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'computed-masked-segment2-(load|store)'` passed 10 selected generated-bundle
  dry-run and direct pre-realized fail-closed tests.
* The first non-dry-run generated-bundle attempt was blocked before generation
  because the bare `llvm-readobj` executable was not on `PATH`. It was rerun
  with `/usr/lib/llvm-20/bin/llvm-readobj` and `--overwrite`, replacing the
  blocked artifact evidence with successful evidence.
* Non-dry-run generated-bundle evidence for
  `computed_masked_segment2_load_unit_store` passed on `ssh rvv` with
  `dry_run: false`, `ssh_evidence: true`, `status: success`, ABI
  `rvv-generic-computed-masked-segment2-load-unit-store-callable-c-abi.v1`,
  selected variant `pre_realized_body_rvv_cmseg_load`, counts
  `0,1,7,16,23,257`, two compare-mask patterns, active/inactive lane coverage,
  inactive-lane old-field passthrough preservation, field-order distinguishing
  lanes, source preservation, and output tail preservation.
* Non-dry-run generated-bundle evidence for
  `computed_masked_segment2_store_unit_load` passed on `ssh rvv` with
  `dry_run: false`, `ssh_evidence: true`, `status: success`, ABI
  `rvv-generic-computed-masked-segment2-store-unit-load-callable-c-abi.v1`,
  selected variant `pre_realized_body_rvv_cmseg_store`, counts
  `0,1,7,16,23,257`, two compare-mask patterns, active/inactive lane coverage,
  inactive-lane interleaved-destination preservation, field-order
  distinguishing lanes, source preservation, and destination tail preservation.
* Bounded old-authority scan over this task's changed Trellis files found only
  negative/non-goal mentions of descriptor, source-front-door, and legacy
  `i32m1`; no added executable route authority was introduced.
* `git diff --check` passed before staging; `git diff --cached --check` passed
  after staging.

## Spec Update Decision

No `.trellis/spec/` update is needed. This round implements and verifies the
already documented computed-mask segment2 artifact boundary: target validation
consumes provider-owned route validation contracts and treats metadata as
mirrors only. The task contributed fresh non-dry-run `ssh rvv` executable
evidence for the existing computed-masked segment2 load/store boundary, not a
new architectural rule.
