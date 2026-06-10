# Stage2 RVV production-kernel Gearbox resource-aware selected-body realization campaign

## Goal

Create one macro owner for RVV plugin-local Gearbox/resource-aware
selected-body realization in production-kernel low-precision and contraction
paths. The campaign makes selected typed `tcrv_rvv` body facts, SEW/LMUL/VL
policy, widening-product and accumulator facts, packed/unpacked resource
schedule facts, mask/tail policy, runtime AVL/ABI facts, same-target
measurement-policy prerequisites, provider validation, target validation, and
later dispatch policy flow through one RVV-owned fail-closed boundary.

The current round completes Gate 1 only: define and connect a production
resource-planning contract plus at least one real production consumer and
diagnostic. Gates 2-4 remain future milestones unless this PRD is updated by
human steering or later evidence.

## What I Already Know

- Session start had no `.trellis/.current-task`; this task was created from the
  Hermes Direction Brief as the macro owner.
- Commit `b857b290` completed the prior same-target
  measurement/selected-dispatch campaign. That archive shows measurement
  records and selected-dispatch policy are now source-backed and separated from
  correctness execution and performance-win claims.
- The archived low-precision contraction primitive-surface campaign completed
  primitive/resource/schedule/target/policy gates. It proved existing
  production consumers in selected-body realization, Gearbox scheduling,
  provider route planning, target validation, and same-target policy, but this
  new campaign needs an explicit production resource-planning contract that can
  become the shared handoff for resource-aware realization.
- Current specs require Stage 2 work to advance resource-aware selected-body
  realization, typed low-precision coverage, or measured same-target evidence.
  q8/q4 or llama.cpp examples are pressure tests, not route authority.
- Existing source already has `RVVLowPrecisionContractionResourceCandidate`,
  `RVVLowPrecisionContractionResourceSelection`, Gearbox resource attrs,
  selected-body realization attributes, provider route-family validation, packed
  i4 schedule/remediation facts, and same-target policy inputs.
- Initial source inspection found the likely Gate 1 gap: selected-body
  realization and provider validation both consume resource facts, but the
  resource-planning handoff is implicit across candidate attrs and route
  selection fields. A named planning contract should make stale/missing
  planning facts fail before route construction.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per worker round, commit it, and leave
  this task active while remaining gates are incomplete.
- Gate 1 must change production RVV plugin/provider source unless live source
  inspection proves the production resource-planning contract already exists
  and has focused fail-closed tests.
- The resource-planning contract must be RVV-owned and must not be inferred
  from route ids, artifact names, q8/q4 or llama.cpp labels, helper names,
  reports, status fields, metadata mirrors, descriptors, or Common EmitC.
- The contract must carry or validate typed body/config facts, source/product/
  accumulator/result SEW/LMUL facts, tail/mask policy, runtime AVL source,
  runtime ABI order, resource candidate/selection facts, Gearbox realization
  schedule facts, packed/unpacked operand facts, primitive-chain facts,
  target capability mirrors, and same-target measurement-policy prerequisites
  when the selected resource is packed i4.
- The first production consumer for Gate 1 is provider route-family validation
  for the representative low-precision product-reduction dequant/dequant-clamp
  selected-body route. Selected-body realization must materialize the contract
  facts, and provider validation must fail closed when they are missing or
  stale before Common EmitC can materialize a route.
- Focused tests must cover the positive resource-planning contract path and at
  least one stale or missing planning fact diagnostic.

## Macro Campaign Gates

- [x] Gate 1: resource-planning contract plus first production
  selected-body/provider consumer and fail-closed diagnostics.
- [ ] Gate 2: selected-body realization uses the resource plan for a
  representative low-precision/contraction body without changing computation
  semantics, dtype semantics, parameter roles, runtime AVL/VL, variant origin,
  dispatch, or fallback behavior.
- [ ] Gate 3: generated artifact and same-target measurement evidence for the
  realized resource-aware path when executable correctness or performance is
  claimed.
- [ ] Gate 4: selected-dispatch/performance policy consumes resource and
  measurement facts with correctness fallback and fail-closed provenance.

## Current Slice: Gate 1

- [x] Add or repair a named RVV low-precision production resource-planning
  contract in production C++.
- [x] Materialize the planning contract from the RVV-owned Gearbox/resource
  selected-body realization path for the bounded product-reduction
  dequant/dequant-clamp representative.
- [x] Make provider route-family validation consume the planning contract and
  fail closed on missing/stale resource-planning facts before route
  construction.
- [x] Keep Common EmitC unchanged except for carrying provider-built payloads if
  already required by existing code.
- [x] Add focused tests for positive contract consumption and stale/missing
  contract rejection.
- [x] Run focused build/tests, diff whitespace checks, Trellis validation, and
  bounded authority scans.
- [x] Commit the Gate 1 slice and leave the macro task active because Gates 2-4
  remain incomplete.

## Acceptance Criteria For Gate 1

- [x] A production C++ contract or contract field names the low-precision
  resource-planning boundary and is populated from RVV-owned selected-body or
  Gearbox/resource facts.
- [x] Provider route-family validation checks that the selected resource
  planning contract matches typed/config/runtime facts, resource selection
  facts, primitive-chain facts, Gearbox schedule facts, and packed-i4
  measurement-policy prerequisites where applicable.
- [x] A missing or stale planning contract fails closed with a targeted
  diagnostic before `TCRVEmitCLowerableRoute` construction.
- [x] Positive coverage proves a representative product-reduction
  dequant/dequant-clamp selected-body path carries the contract through
  selected-body realization and provider validation.
- [x] Negative coverage proves stale/missing planning facts fail at the RVV
  selected-body/provider boundary, not in Common EmitC, artifact metadata, or a
  report-only path.
- [x] No q8/q4/llama label, route id, artifact name, helper name, descriptor,
  status field, source-front-door marker, or Common EmitC branch becomes route,
  dtype, resource, schedule, performance, or evidence authority.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Completed Gate 1 Slice

- Added the named RVV-owned low-precision production resource-planning contract
  `rvv-low-precision-production-resource-planning-contract.v1` and the
  `tcrv_rvv.low_precision_resource.planning_contract` attr.
- Connected the contract through low-precision resource candidates, Gearbox
  schedule materialization, contraction selected-body realization, provider
  route-family selection/validation, provider-built route metadata mirrors, and
  target artifact validation.
- Added fail-closed diagnostics for stale pre-realized selected-body planning
  facts, stale provider selection facts, and stale target metadata mirrors.
- Updated the RVV plugin spec with the executable contract, failure matrix, and
  required tests.

## Remaining Campaign Gates

- Gate 2 remains: selected-body realization must use the resource plan as an
  implementation driver for a representative low-precision/contraction body
  without changing computation semantics, runtime AVL/VL, ABI roles, variant
  origin, dispatch, or fallback behavior.
- Gate 3 remains: generated artifact and same-target measurement evidence for
  the realized resource-aware path when executable correctness or performance is
  claimed.
- Gate 4 remains: selected-dispatch/performance policy consumes resource and
  measurement facts with correctness fallback and fail-closed provenance.

## Out Of Scope

- New q8/q4/llama-named route ids, wrappers, artifact authority, or benchmark
  authority.
- High-level Linalg, Vector, StableHLO, source-front-door frontend work, or
  per-Linalg route authority.
- Broad dtype/LMUL clone batches, one-intrinsic wrappers, dashboards, tuning
  databases, or report-only closeout.
- Common EmitC invention of RVV dtype, primitive, resource, schedule,
  measurement, dispatch, or performance semantics.
- New generated-bundle or `ssh rvv` evidence unless Gate 1 code claims new
  executable correctness, runtime, or performance behavior.
- Selected-dispatch/performance policy changes that do not directly consume
  the new resource plan.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived predecessor tasks read:
  `.trellis/tasks/archive/2026-06/06-10-06-10-stage2-rvv-production-kernel-same-target-measurement-selected-dispatch-campaign/`
  and
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign/`.
- Initial source files inspected:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  and
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`.

## Continuation Point

After Gate 1, continue the same macro task at Gate 2: make selected-body
realization consume the resource plan as the implementation driver for a
representative low-precision/contraction body and prove it preserves semantics,
runtime AVL/VL, ABI roles, and dispatch/fallback behavior.
