# Stage2 RVV production-kernel same-target measurement and selected-dispatch campaign

## Goal

Create one macro owner for the RVV production-kernel same-target measurement
and selected-dispatch campaign. The campaign connects selected low-precision
`tcrv_rvv` bodies, RVV-owned primitive/resource facts, generated target
artifacts, source-backed same-target measurement records, target validation,
and selected-dispatch performance preference into one fail-closed compiler
workflow.

Gate 1 is complete. This round implements Gate 2: a representative
low-precision packed-i4 generated artifact must produce a structured
same-target measurement record shape that can feed the Gate 1
`RVVLowPrecisionSameTargetMeasurementPolicyInput` boundary, with fail-closed
validation for stale target, ABI, provider, primitive, schedule, and measurement
provenance fields.

## What I Already Know

- Session start had no `.trellis/.current-task`; this task was created from the
  Hermes Direction Brief as a new macro owner.
- Commit `2eebb03e` archived the previous low-precision contraction
  primitive-surface campaign after Gates 1-4. That campaign made target
  validation consume `RVVLowPrecisionSameTargetMeasurementPolicyInput`, but
  the current target validation path still obtains the accepted measurement by
  calling `getAcceptedRVVPackedI4Gate4MeasurementOutcome()`.
- Current specs require RVV performance-sensitive Stage 2 work to advance
  resource-aware selected-body realization, typed low-precision coverage, or a
  measured same-target comparison path. q8/q4 or llama.cpp examples remain
  pressure tests, not route authority.
- Existing production files already define:
  `RVVLowPrecisionPerformanceMeasurementOutcome`,
  `RVVLowPrecisionSameTargetMeasurementPolicyInput`,
  `buildRVVLowPrecisionSameTargetMeasurementPolicyInput`,
  `consumeRVVLowPrecisionSameTargetMeasurementPolicyInput`,
  `verifyRVVLowPrecisionPerformancePolicy`, and target-side policy checks.
- Existing measurement tooling
  `scripts/rvv_generated_bundle_same_target_measure.py` emits structured
  fields such as measurement evidence id, classification, outcome family,
  speedup range, record counts, same-target/ssh evidence, target profile, and
  provider policy tie-backs.
- The production gap for Gate 1 is the input boundary: policy/target tests and
  target validation can still materialize the representative Gate 4 measurement
  outcome from a fixed helper instead of from an explicit measurement-record
  source object.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per worker round, commit it, and leave
  this task active while remaining gates are incomplete.
- Gate 1 must change production source unless live source inspection proves
  the measurement-record seam is already source-backed.
- Measurement evidence is input only. It can feed policy and target validation,
  but it must not invent RVV primitive facts, resource facts, dispatch facts,
  target capability facts, maturity facts, or performance preference.
- Policy input must remain tied to provider-owned resource selection,
  primitive-chain facts, schedule decision, runtime ABI order, target
  capability mirrors, and selected-dispatch boundary facts when present.
- Wrong target, stale sibling route, wrong runtime ABI order, wrong
  primitive-chain facts, missing measurement identity, and measurement-only
  performance claims must fail closed with targeted diagnostics.
- No Common EmitC code may infer RVV low-precision measurement, dispatch,
  primitive, resource, dtype, schedule, or performance semantics.

## Macro Campaign Gates

- [x] Gate 1: source-backed measurement record ingestion/validation reaches
  target validation and `RVVLowPrecisionSameTargetMeasurementPolicyInput`
  without hardcoded outcome constants.
- [x] Gate 2: a representative low-precision contraction generated artifact can
  produce same-target correctness/performance measurement evidence on `ssh rvv`;
  q8/q4 and llama.cpp examples are pressure tests only.
- [ ] Gate 3: selected-dispatch/performance preference consumes measurement
  records plus primitive/resource facts with fail-closed stale sibling-route,
  wrong-target, wrong-ABI, and wrong-primitive-chain diagnostics.
- [ ] Gate 4: final campaign audit separates correctness execution,
  performance preference, and performance-win claims, then archives this macro
  task only when all gates are complete.

## Current Slice: Gate 2

- [x] Use the representative packed-i4
  `widening_product_reduce_dequantize_f32` generated artifact path named by
  the existing same-target measurement script and target RVV fixture.
- [x] Make `scripts/rvv_generated_bundle_same_target_measure.py` emit or
  validate a `same_target_measurement_record` object whose fields match the
  C++ `RVVLowPrecisionSameTargetMeasurementRecord` boundary.
- [x] Add a production C++ policy API that materializes
  `RVVLowPrecisionSameTargetMeasurementRecord` from that structured evidence
  object and then reuses the Gate 1 record-to-policy-input validation path.
- [x] Prove stale target, stale runtime ABI, stale provider primitive-chain, and
  missing measurement identity fail closed before selected-dispatch or
  performance policy can consume the record.
- [x] Run the measurement script self-test, focused C++ plugin/target tests, a
  representative same-target measurement command for the packed-i4 generated
  artifact, whitespace checks, and a bounded old-authority scan.
- [x] Commit the Gate 2 slice and leave `.trellis/.current-task` active because
  Gates 3-4 remain incomplete.

## Acceptance Criteria For Gate 2

- [x] The representative packed-i4 generated artifact measurement workflow
  writes a structured `same_target_measurement_record` with measurement
  evidence identity, classification, speedup range, record counts,
  same-target/ssh flags, target profile, and provider/resource/primitive/
  schedule/remediation tie-backs.
- [x] The C++ policy API can consume that record shape as an explicit
  `RVVLowPrecisionSameTargetMeasurementRecord` and build
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` through the Gate 1
  validation boundary.
- [x] A stale/cross-target/wrong-ABI/wrong-primitive-chain or
  measurement-identity-negative case fails before performance preference can be
  accepted.
- [x] Measurement evidence remains input-only: provider-owned primitive,
  resource, schedule, maturity, dispatch, target capability, and selected-body
  facts remain authoritative.
- [x] Same-target runtime/performance evidence is only claimed when real
  `ssh rvv` output exists. Dry-run evidence remains explicitly `not-measured`
  and cannot authorize a performance claim.
- [x] No q8/q4/llama label, artifact name, route id, helper name, descriptor,
  status field, source-front-door marker, or test name becomes authority.
- [x] No Common EmitC code infers RVV low-precision semantics.

## Completed Slice: 2026-06-10 Gate 1

- Added `RVVLowPrecisionSameTargetMeasurementRecord` as the explicit
  source-backed measurement-record input boundary for the low-precision
  performance policy API. The record carries the script-output-side
  measurement fields plus provider/resource/primitive/schedule/ABI/target
  tie-backs that must agree before policy input can be consumed.
- Replaced the old direct production use of
  `getAcceptedRVVPackedI4Gate4MeasurementOutcome()` in RVV provider plan
  validation, resource-remediation handoff verification, resource-selection
  policy validation, and both RVV target artifact validation paths. These paths
  now build a measurement record, convert it to
  `RVVLowPrecisionSameTargetMeasurementPolicyInput`, and only then call the
  existing policy verifier.
- Preserved the existing `stale-sibling-route-measurement` diagnosis by letting
  sibling-route records reach the policy handoff classifier, while other stale
  record fields such as wrong target, wrong runtime ABI order, missing
  provider tie-backs, and wrong primitive-chain facts still fail closed at the
  record/input boundary or policy verifier.
- Updated RVV plugin and target artifact C++ tests so accepted and measured-win
  outcomes are first materialized from measurement records. Target tests now
  mutate record fields for wrong runtime ABI and wrong target negatives,
  proving the new Gate 1 input boundary is exercised.
- No new runtime correctness/performance result was claimed in this slice; live
  `ssh rvv` measurement was not required.

## Completed Slice: 2026-06-10 Gate 2

- Added production policy APIs that materialize
  `RVVLowPrecisionSameTargetMeasurementRecord` from the structured
  same-target measurement evidence object emitted by the generated-bundle
  measurement script, then build
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` through the existing Gate 1
  fail-closed boundary.
- Updated `scripts/rvv_generated_bundle_same_target_measure.py` so packed-i4
  same-target runs emit `same_target_measurement_record` at the per-op evidence,
  per-op summary, and root aggregate levels. The record contains only the
  C++ measurement-record fields; reporting-only fields such as contract
  alignment and remediation-plan details remain outside the record.
- Added focused C++ coverage proving the generated-artifact evidence-object
  shape preserves measurement identity, runtime ABI, primitive-chain, schedule,
  and target tie-backs, and that missing measurement identity, stale target,
  stale runtime ABI, and stale primitive-chain evidence fail closed before
  policy input is accepted.
- Produced representative real same-target `ssh rvv` evidence for the packed-i4
  generated artifact:
  `artifacts/tmp/gate2-packed-i4-same-target-measurement/gate2-packed-i4-generated-artifact-measure/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`.
  The run completed with `status=success`, `ssh_evidence=true`,
  `classification=regression`, `measurement_summary_record_count=12`,
  `measurement_record_count=60`, `correctness_record_count=12`,
  `measurement_best_speedup_range=0.689567..0.705465`, and
  `performance_win_claim_allowed=false`.
- Dry-run evidence remains explicitly `not-measured` with
  `ssh_evidence=false`, `same_target_measurement=false`, and no performance
  claim allowance.

## Remaining Gates And Continuation

- Gate 3 remains open: selected-dispatch/performance preference must consume
  measurement records plus primitive/resource facts with full stale/wrong-target
  diagnostics.
- Gate 4 remains open: final campaign audit must separate correctness
  execution, performance preference, and performance-win claims before this
  macro task can be archived.
- Next continuation point: start Gate 3 by making selected-dispatch/performance
  preference consume the source-backed measurement records plus primitive and
  resource facts with full stale sibling-route, wrong-target, wrong-ABI, and
  wrong-primitive-chain diagnostics.

## Out Of Scope

- New q8/q4/llama-named route ids, wrappers, artifact authority, or benchmark
  authority.
- High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
- New global autotuning database, dashboard, or report-only workflow.
- Broad generated-bundle or `ssh rvv` test matrix.
- A performance-win claim or performance-preferred dispatch change without
  same-target measured evidence and matching provider maturity facts.
- Common EmitC invention of low-precision compute, unpack, dtype, schedule,
  measurement, dispatch, or performance semantics.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Archived predecessor task read:
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign/`.
- Initial source inspection found the target validation path at
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` still calls
  `getAcceptedRVVPackedI4Gate4MeasurementOutcome()` before building the
  policy input. This is the Gate 1 production seam.
