# Stage2 RVV production-kernel Gearbox resource-aware selected-body realization campaign

## Goal

Create one macro owner for RVV plugin-local Gearbox/resource-aware
selected-body realization in production-kernel low-precision and contraction
paths. The campaign makes selected typed `tcrv_rvv` body facts, SEW/LMUL/VL
policy, widening-product and accumulator facts, packed/unpacked resource
schedule facts, mask/tail policy, runtime AVL/ABI facts, same-target
measurement-policy prerequisites, provider validation, target validation, and
later dispatch policy flow through one RVV-owned fail-closed boundary.

Gate 1 and Gate 2 are complete. Gate 3 is complete for both the packed-i4
`widening_product_reduce_dequantize_f32` and
`widening_product_reduce_dequant_clamp_f32` representatives, including
source-backed generated artifacts, same-target measurement records, and real
`ssh rvv` regression/no-win evidence. The current round closes Gate 4 at the
target-artifact selected-dispatch/performance policy boundary by proving
provider-owned resource and measurement facts are consumed, correctness
fallback remains available, and metadata-only route-support, correctness,
performance-selection, dispatch, or win-claim mirrors fail before artifact
acceptance.

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
- [x] Gate 2: selected-body realization uses the resource plan for a
  representative low-precision/contraction body without changing computation
  semantics, dtype semantics, parameter roles, runtime AVL/VL, variant origin,
  dispatch, or fallback behavior.
- [x] Gate 3: generated artifact and same-target measurement evidence for the
  realized resource-aware path when executable correctness or performance is
  claimed.
- [x] Gate 4: selected-dispatch/performance policy consumes resource and
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

## Current Slice: Gate 2 Handoff Planning-Contract Consumer

- [x] Keep the same macro task active and target the representative
  `widening_product_reduce_dequantize_f32` product-reduction Gearbox path.
- [x] Make selected-body realization copy the selected low-precision resource
  planning contract into the structural
  `tcrv_rvv.gearbox_cross_region_handoff`, not only the realized `with_vl`
  attrs or route metadata mirrors.
- [x] Make `tcrv_rvv.gearbox_cross_region_handoff` verifier reject missing or
  stale handoff planning contracts before provider route planning.
- [x] Make RVV route planning and route-family validation compare the handoff
  planning contract with the selected
  `RVVLowPrecisionContractionResourceSelection::planningContract` before
  `TCRVEmitCLowerableRoute` construction.
- [x] Add focused positive and fail-closed coverage for the realized handoff
  planning contract without changing computation semantics, dtype semantics,
  ABI roles, runtime AVL/VL, variant origin, dispatch, or fallback behavior.
- [x] Leave Gate 2 open after this sub-slice; remaining Gate 2 work can extend
  the same pattern to marker-level planning-contract checks or additional
  production representatives if human steering requires it.

## Current Slice: Gate 2 Marker Planning-Contract Consumer

- [x] Keep the same macro task active and continue the representative
  `widening_product_reduce_dequantize_f32` product-reduction Gearbox path.
- [x] Make selected-body realization copy the selected low-precision resource
  planning contract into each realized
  `tcrv_rvv.vsetvl_region_marker`, not only `with_vl` attrs, handoff attrs,
  or route metadata mirrors.
- [x] Make `tcrv_rvv.vsetvl_region_marker` verification reject missing or stale
  marker planning contracts for supported low-precision resource decisions.
- [x] Make Gearbox handoff verification and RVV route-family validation compare
  marker planning contracts with the selected provider resource plan before
  `TCRVEmitCLowerableRoute` construction.
- [x] Add focused positive C++/MLIR coverage and stale/missing marker
  planning-contract diagnostics without changing computation semantics, dtype
  semantics, ABI roles, runtime AVL/VL, variant origin, dispatch, or fallback
  behavior.
- [x] Leave the macro task active after this sub-slice because Gate 3 generated
  artifact/same-target measurement evidence and Gate 4 selected-dispatch/
  performance policy remain future milestones.

## Current Slice: Gate 2 Additional Representative Planning-Contract Consumer

- [x] Keep the same macro task active and target the additional
  `widening_product_reduce_dequant_clamp_f32` production representative.
- [x] Inspect production selected-body realization, dialect verification,
  route planning, and contraction route-family validation for generalized
  `selectedCandidate.planningContract` consumption.
- [x] Prove the existing production path already carries
  `rvv-low-precision-production-resource-planning-contract.v1` through the
  realized dequant-clamp `tcrv_rvv.vsetvl_region_marker` ops,
  `tcrv_rvv.gearbox_cross_region_handoff`, emission-plan provider mirrors, and
  target header metadata mirrors.
- [x] Add focused dequant-clamp positive FileCheck coverage for the structural
  marker/handoff planning contract.
- [x] Add focused dequant-clamp stale and missing marker/handoff
  planning-contract diagnostics before Common EmitC route construction.
- [x] Leave computation semantics, dtype semantics, ABI roles, runtime AVL/VL,
  variant origin, dispatch, fallback behavior, and Common EmitC neutrality
  unchanged.
- [x] Mark Gate 2 complete because both representative product-reduction
  dequant and dequant-clamp paths now have structural resource-plan consumer
  coverage.

## Current Slice: Gate 3 Same-Target Artifact Measurement Record Boundary

- [x] Keep the same macro task active and target the first resource-aware
  production-kernel representative:
  `widening_product_reduce_dequantize_f32` with the validated packed-i4
  resource-aware schedule.
- [x] Make the generated artifact / same-target measurement evidence record
  explicitly carry and validate the Gate 1/2 planning contract, selected
  resource operand form, storage/effective element width, packing layout,
  unpack intent, runtime AVL source, vsetvl region count, runtime ABI order,
  correctness evidence flags, and target capability/profile provenance.
- [x] Ensure those record fields are sourced from provider-owned resource facts
  and validated generated object/header mirrors, not from route ids, artifact
  names, fixture names, q4/q8 labels, Common EmitC, or raw stdout alone.
- [x] Add fail-closed coverage for stale or missing planning-contract,
  resource-form, runtime AVL/VL, ABI, target, measurement, or correctness facts
  before selected-dispatch/performance policy consumes the record.
- [x] Preserve Gate 4 as future policy consumption: this slice may feed the
  existing policy-input boundary, but it must not claim performance preference
  or rewrite provider maturity facts from measurement output alone.

## Current Slice: Gate 3 Dequant-Clamp Same-Target Artifact Measurement Record

- [x] Keep the same macro task active and target the packed-i4
  `widening_product_reduce_dequant_clamp_f32` representative.
- [x] Make the generated artifact / same-target measurement evidence record
  explicitly carry and validate the Gate 1/2 planning contract, packed
  resource operand form, source signedness, storage/effective element widths,
  packing layout, unpack intent, vsetvl region count, runtime AVL source,
  runtime ABI order including `lower_bound` and `upper_bound`, clamp/select
  coverage, correctness flags, and target capability/profile provenance.
- [x] Ensure those record fields are sourced from provider-owned resource facts
  and validated generated object/header mirrors, not route ids, artifact names,
  fixture names, q4/q8 labels, Common EmitC, or raw stdout alone.
- [x] Add focused fail-closed coverage for stale or missing planning/resource/
  runtime/target/measurement facts and for stale dequant-clamp sibling-route
  facts before selected-dispatch/performance policy can consume the record.
- [x] Produce real `ssh rvv` same-target measurement evidence for multiple
  runtime `n` counts. Correctness execution may be reported when guards pass;
  performance-win claims must remain gated by the measured classification and
  provider eligibility.
- [x] Preserve Gate 4 as future policy consumption. This slice may feed the
  existing policy-input boundary, but it must not claim performance preference
  or rewrite provider maturity facts from measurement output alone.

## Current Slice: Gate 4 Dequant-Clamp Source-Backed Policy Consumption

- [x] Keep the same macro task active and consume the existing packed-i4
  `widening_product_reduce_dequant_clamp_f32` Gate 3 artifact as a Gate 4
  policy input, not as route authority or artifact-name authority.
- [x] Make strict packed-i4 performance policy validation choose accepted
  same-target measurement range and record counts from the selected provider
  resource candidate, so the dequant and dequant-clamp sibling records are not
  interchangeable.
- [x] Parse the committed dequant-clamp
  `same_target_measurement_record` JSON object through the C++
  `RVVLowPrecisionSameTargetMeasurementRecord` boundary and then through the
  selected-dispatch record overload.
- [x] Prove the current dequant-clamp regression/no-win record selects
  correctness fallback, keeps correctness execution allowed, denies
  performance-preferred dispatch, and denies performance-win claims.
- [x] Add focused stale schedule-decision and correctness-disabled negative
  checks for the dequant-clamp artifact record before selected-dispatch policy
  acceptance.
- [x] Keep Common EmitC, target artifact metadata, computation semantics, dtype
  semantics, ABI roles, runtime AVL/VL, variant origin, and fallback semantics
  unchanged.

## Current Slice: Gate 4 Target-Artifact Selected-Dispatch/Performance Mirror Closure

- [x] Keep the same macro task active and close the final Gate 4 audit at the
  target-artifact selected-dispatch/performance boundary.
- [x] Make target artifact validation reject metadata-only packed-i4
  route-support, correctness-execution, correctness-fallback,
  performance-selection, dispatch-policy, route-support-effect, and win-claim
  mirrors before artifact acceptance.
- [x] Keep the positive selected-dispatch path source-backed by provider-owned
  low-precision resource facts and Gate 4 measurement policy facts, with
  correctness fallback selected and performance preference denied for the
  current regression/no-win records.
- [x] Add focused target artifact C++ coverage for metadata-only
  performance-selection, correctness-execution, correctness-fallback, and
  route-support-effect mirrors.
- [x] Keep Common EmitC, artifact names, route ids, q4/q8 labels, computation
  semantics, dtype semantics, ABI roles, runtime AVL/VL, variant origin, and
  fallback semantics unchanged.

## Completed Gate 3 Sub-Slice

- Extended `RVVLowPrecisionSameTargetMeasurementRecord` and the corresponding
  policy input to carry planning-contract, packed resource-form, signedness,
  storage/effective width, packing layout, unpack intent, vsetvl region count,
  runtime AVL source, runtime ABI order, correctness, and target-provenance
  tie-backs.
- Made C++ policy-input validation compare those record fields against the
  selected provider-owned `RVVLowPrecisionContractionResourceSelection` before
  selected-dispatch/performance policy consumption.
- Made the same-target measurement script build those fields from validated
  provider/resource metadata and generated object/header mirrors, with
  fail-closed self-tests for stale or missing planning/resource/runtime facts.
- Added focused C++ and dry-run FileCheck coverage for the packed-i4
  `widening_product_reduce_dequantize_f32` representative.
- Collected real `ssh rvv` same-target evidence for
  `artifacts/gate3-packed-i4-schedule-decision-ssh`:
  correctness guards passed for 12 cases over counts 257, 4096, and 65536;
  parsed timing produced 12 summaries; classification was `regression`;
  performance selection stayed false and performance-win claims stayed denied.

## Completed Gate 3 Dequant-Clamp Sub-Slice

- Added candidate-sensitive packed-i4 performance baseline and remediation
  evidence identities for `widening_product_reduce_dequant_clamp_f32`, so
  provider/resource policy validation rejects stale dequant sibling facts
  instead of accepting metadata that belongs to another representative.
- Extended generated-bundle ABI and same-target measurement tooling so the
  packed-i4 dequant-clamp representative carries provider-owned planning
  contract, operand form, source signedness, storage/effective element widths,
  packing layout, unpack intent, vsetvl region count, runtime AVL source,
  clamp ABI order, target profile, same-target measurement, correctness, and
  performance-claim facts through artifact/header/harness/evidence records.
- Added focused target and script coverage for the source-backed generated
  artifact, dry-run measurement record, clamp-specific ABI harness, and stale
  or missing provider/resource/provenance facts.
- Collected real `ssh rvv` same-target evidence for
  `artifacts/gate3-packed-i4-dequant-clamp-ssh`:
  correctness guards passed for 24 records over counts 257, 4096, and 65536;
  parsed timing produced 24 measurement records and 24 summaries;
  classification was `regression`; best speedup range was
  `0.693878..0.964286`; correctness execution stayed allowed and
  performance-win claims stayed denied.

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

## Completed Gate 2 Sub-Slice

- Added a structural `planning_contract` handoff attr to the realized
  product-reduction Gearbox cross-region handoff, populated from the selected
  low-precision resource candidate.
- Made the RVV dialect verifier reject handoff operations with missing or stale
  `planning_contract` values before they can reach Common EmitC.
- Made RVV route planning and route-family validation require the handoff
  planning contract to match the selected provider resource plan.
- Added positive C++/MLIR coverage and stale/missing handoff planning-contract
  diagnostics for the representative `widening_product_reduce_dequantize_f32`
  path.
- Updated the RVV plugin spec with the executable handoff planning-contract
  requirement and test obligations.
- Added a structural marker-level `planning_contract` attr to realized
  `tcrv_rvv.vsetvl_region_marker` ops, populated from the selected
  low-precision resource candidate.
- Made the RVV marker verifier, Gearbox handoff verifier, consumer-scope route
  collection, and contraction route-family validation reject missing or stale
  marker planning-contract facts before Common EmitC route construction.
- Added positive C++/MLIR coverage plus stale/missing marker planning-contract
  diagnostics for the representative product-reduction-dequant Gearbox path.
- Updated the RVV plugin spec with the executable marker planning-contract
  requirement and test obligations.
- Proved the existing generalized production path covers the additional
  `widening_product_reduce_dequant_clamp_f32` representative without new source
  changes: selected-body realization already populates marker and handoff
  `planning_contract` from the selected resource candidate, and verifier plus
  route-family validation consume the selected provider resource plan before
  route construction.
- Added dequant-clamp FileCheck evidence that marker/handoff structure,
  emission-plan provider mirrors, and target header mirrors carry
  `rvv-low-precision-production-resource-planning-contract.v1`.
- Added dequant-clamp stale and missing marker/handoff planning-contract
  negative diagnostics.

## Completed Campaign Gates

- Gate 2 is complete for the campaign's bounded representative surface:
  product-reduction dequant handoff and marker consumers are complete, and the
  additional dequant-clamp representative is covered by focused structural
  evidence using the same generalized production path.
- Gate 3 current requested representative scope is complete: both the first
  packed-i4 `widening_product_reduce_dequantize_f32` representative and the
  selected packed-i4 `widening_product_reduce_dequant_clamp_f32` representative
  now have source-backed artifact/measurement-record binding and real
  `ssh rvv` regression/no-win evidence tied back to provider resource facts.
  Gate 3 should only reopen for an additional representative if human steering
  expands the campaign surface.
- Gate 4 is complete for the current campaign scope: source-backed dequant and
  dequant-clamp measurement records feed the selected-dispatch/performance
  policy boundary, current regression/no-win evidence selects correctness
  fallback and denies performance preference, stale provenance fails closed,
  and target artifact validation rejects metadata-only performance-selection,
  dispatch, correctness, route-support, and win-claim mirrors before artifact
  acceptance.

## Out Of Scope

- New q8/q4/llama-named route ids, wrappers, artifact authority, or benchmark
  authority.
- High-level Linalg, Vector, StableHLO, source-front-door frontend work, or
  per-Linalg route authority.
- Broad dtype/LMUL clone batches, one-intrinsic wrappers, dashboards, tuning
  databases, or report-only closeout.
- Common EmitC invention of RVV dtype, primitive, resource, schedule,
  measurement, dispatch, or performance semantics.
- New generated-bundle or `ssh rvv` evidence outside the active Gate 3
  representative evidence requested by human steering.
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
- Gate 2 handoff planning-contract sub-slice additionally touched
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
- Gate 2 marker planning-contract sub-slice additionally touched
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
- Gate 2 additional representative sub-slice inspected
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, then
  updated only
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`
  for focused structural evidence.
- Gate 3 dequant-clamp same-target measurement sub-slice touched
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  RVV provider/target validation users of packed-i4 evidence identities,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `scripts/rvv_generated_bundle_same_target_measure.py`,
  focused C++ tests, focused script/target lit tests, and
  `artifacts/gate3-packed-i4-dequant-clamp-ssh`.
- Gate 4 dequant-clamp source-backed policy-consumption sub-slice touched
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp` and
  `test/Plugin/RVVExtensionPluginTest.cpp`. It consumed the existing
  `artifacts/gate3-packed-i4-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json`
  record through the C++ record overload and selected-dispatch boundary.
- Gate 4 target-artifact selected-dispatch/performance mirror closure touched
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` and
  `test/Target/TargetArtifactExportTest.cpp`. It rejects metadata-only packed-i4
  route-support, correctness, performance-selection, dispatch, route-support
  effect, and win-claim mirrors at target artifact validation.

## Continuation Point

After this Gate 4 target-artifact selected-dispatch/performance mirror closure
slice, all current macro campaign gates are complete. The next step is final
verification and Trellis wrap-up/archive unless human steering reopens Gate 4
for an additional representative or stricter same-target evidence refresh.
