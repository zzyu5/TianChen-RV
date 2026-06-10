# Stage2 RVV production-kernel same-target measurement and selected-dispatch campaign

## Goal

Create one macro owner for the RVV production-kernel same-target measurement
and selected-dispatch campaign. The campaign connects selected low-precision
`tcrv_rvv` bodies, RVV-owned primitive/resource facts, generated target
artifacts, source-backed same-target measurement records, target validation,
and selected-dispatch performance preference into one fail-closed compiler
workflow.

This round implements Gate 1 only: source-backed measurement record
ingestion/validation must reach RVV target validation and
`RVVLowPrecisionSameTargetMeasurementPolicyInput` without relying on hardcoded
accepted outcome constants.

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
- [ ] Gate 2: a representative low-precision contraction generated artifact can
  produce same-target correctness/performance measurement evidence on `ssh rvv`;
  q8/q4 and llama.cpp examples are pressure tests only.
- [ ] Gate 3: selected-dispatch/performance preference consumes measurement
  records plus primitive/resource facts with fail-closed stale sibling-route,
  wrong-target, wrong-ABI, and wrong-primitive-chain diagnostics.
- [ ] Gate 4: final campaign audit separates correctness execution,
  performance preference, and performance-win claims, then archives this macro
  task only when all gates are complete.

## Current Slice: Gate 1

- [x] Add or harden a production measurement-record input boundary that builds
  `RVVLowPrecisionPerformanceMeasurementOutcome` from explicit same-target
  measurement record fields rather than from accepted outcome constants.
- [x] Thread that source-backed outcome into
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` for the representative
  packed-i4 low-precision contraction artifact target-validation path.
- [x] Keep provider/resource/primitive/ABI/schedule/target facts authoritative;
  the measurement record may only supply measurement-side fields.
- [x] Add focused C++ coverage proving the target/policy input is built from a
  measurement record and that a stale record field, such as wrong target or
  mismatched primitive-chain tie-back, fails closed.
- [x] Run focused RVV plugin and target artifact checks, the measurement script
  self-test, whitespace checks, and a bounded old-authority scan.
- [x] Commit the Gate 1 slice and leave `.trellis/.current-task` active because
  Gates 2-4 remain incomplete.

## Acceptance Criteria For Gate 1

- [x] Production target validation and policy consumption use a
  source-backed measurement record boundary for representative packed-i4
  low-precision contraction measurement input.
- [x] `getAcceptedRVVPackedI4Gate4MeasurementOutcome()` is no longer the direct
  production target-validation source for the representative accepted
  same-target measurement.
- [x] The resulting policy input still carries provider primitive-chain,
  schedule decision, runtime ABI order, target capability mirrors, maturity
  fields, remediation handoff, and measurement outcome facts.
- [x] A stale/cross-target/wrong-ABI/wrong-primitive-chain or measurement-only
  negative case fails before performance preference can be accepted.
- [x] No new runtime performance or performance-win claim is made in this
  round, so live `ssh rvv` measurement is not required unless new runtime
  evidence is actually produced.
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

## Remaining Gates And Continuation

- Gate 2 remains open: produce representative same-target correctness and
  performance measurement evidence on `ssh rvv` for a low-precision contraction
  generated artifact.
- Gate 3 remains open: selected-dispatch/performance preference must consume
  measurement records plus primitive/resource facts with full stale/wrong-target
  diagnostics.
- Gate 4 remains open: final campaign audit must separate correctness
  execution, performance preference, and performance-win claims before this
  macro task can be archived.
- Next continuation point: start Gate 2 from the representative packed-i4 or
  dequant/dequant-clamp low-precision contraction artifact, run the same-target
  measurement script on `ssh rvv` only if runtime evidence is produced, and
  feed the resulting record into the Gate 1 policy-input boundary.

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
