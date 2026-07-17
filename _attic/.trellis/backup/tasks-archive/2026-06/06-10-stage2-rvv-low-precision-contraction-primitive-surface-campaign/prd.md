# Stage2 RVV Low-Precision Contraction Primitive-Surface Campaign

## Goal

Create one macro owner for RVV Stage 2 low-precision contraction primitive
facts. The campaign makes low-precision operand element facts, typed
vector/config facts, widening product facts, accumulator/reduction facts,
selected-body realization facts, route/provider legality, target validation,
source-backed same-target measurement records, and selected-dispatch
performance policy decisions flow through production RVV-owned compiler
surfaces.

The final Gate 4 state is:

```text
typed low-precision tcrv_rvv primitive facts
  -> RVV plugin-local realization/provider facts
  -> route/provider/target artifact mirrors
  -> source-backed same-target measurement records
  -> RVV low-precision selected-dispatch/performance policy
  -> correctness fallback or performance-preferred decision
```

Primitive, resource, schedule, target, runtime, and measurement authority must
come from typed `tcrv_rvv` body/config/provider facts plus validated
source-backed measurement records. It must not come from q8/q4 labels,
llama.cpp names, packed-i4 labels, artifact names, route ids, ABI strings,
helper names, descriptor residue, raw stdout, report status, or Common EmitC
semantic inference.

## Requirements

- Keep one macro Trellis task until all campaign gates are complete.
- Complete coherent slices and record each gate truthfully in the task PRD,
  check context, journal, and commit history.
- RVV plugin/provider code owns low-precision primitive facts, legality,
  selected-body realization inputs, route facts, performance policy facts, and
  fail-closed diagnostics.
- Common EmitC/export may carry provider-built payloads and mirrors only; it
  must not infer dtype, unpack/sign/zero-extension, widening product, reduction,
  layout, schedule, measurement, dispatch, or performance preference.
- Gate 4 policy consumption must accept only fresh matching
  primitive/resource/schedule/runtime/target/measurement provenance, deny
  stale or regressed performance preference, preserve correctness fallback for
  legal routes, and fail closed for missing, stale, sibling-route, or
  metadata-only evidence.

## Macro Campaign Gates

- [x] Gate 1: typed low-precision contraction primitive facts and fail-closed
  provider/validation surface exist in production code.
- [x] Gate 2: RVV plugin-local selected-body realization consumes those facts
  for representative low-precision product-reduction/dequant paths without
  changing compute semantics.
- [x] Gate 3: route/provider/artifact export carries those facts into generated
  artifacts and source-backed same-target measurement records.
- [x] Gate 4: selected-dispatch/performance policy consumes those measurements
  fail-closed, preserving correctness fallback and denying stale/no-win
  performance claims.

## Completed Slice: Gate 1

- Hardened target artifact provider-facts validation so stale low-precision
  resource selection primitive surface fields are rejected before artifact
  export.
- Added focused target artifact coverage mutating packed-i4
  product-reduction/dequant primitive surface facts to prove stale facts fail
  closed at the provider-facts boundary.
- Verified focused target/plugin C++ tests and whitespace checks. No new
  runtime, correctness, or performance claim was made.

## Completed Slice: Gate 2

- Hardened `RVVContractionSelectedBodyRealizationOwner` so selected
  low-precision product-reduction/dequant resource candidates must match
  provider-owned primitive contract, kind, relations, intrinsics, layouts, and
  store-VL before realization accepts them.
- Preserved positive selected-body realization for typed
  setvl/with_vl/load/widening_product/standalone_reduce/handoff/dequant/store
  structure carrying provider-visible resource and primitive facts.
- Added focused RVV plugin C++ coverage for missing primitive resource facts
  and stale primitive reduction intrinsic facts failing closed before route
  construction or target artifact acceptance.

## Completed Slice: Gate 3

- Extended `RVVLowPrecisionSameTargetMeasurementRecord` and
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` so source-backed
  measurement records carry provider primitive contract/kind,
  source/product/accumulator/result dtype and SEW/LMUL facts, primitive
  widening/reduction intrinsics, scalar seed splat, accumulator/result layouts,
  reduction store-VL, resource/runtime ABI, schedule, target capability, and
  primitive-chain facts.
- Hardened measurement-record parsing and policy-input construction so missing
  or stale primitive measurement fields fail closed before selected-dispatch or
  performance policy can consume the measurement.
- Updated `rvv_generated_bundle_same_target_measure.py` and the checked-in
  dequant-clamp same-target evidence JSON so generated records mirror
  validated provider/resource/artifact metadata.
- Added focused plugin and target artifact C++ coverage for positive record
  propagation plus missing/stale primitive intrinsic fail-closed cases.

## Completed Slice: Gate 4

- Confirmed that `RVVLowPrecisionPerformancePolicy` exposes strict
  `evaluate*`/`verify*` APIs and safe `resolve*` dispatch resolvers for
  measurement outcomes, same-target policy inputs, same-target measurement
  records, and selected-dispatch boundary facts.
- Confirmed production route/provider and target-artifact validation consume
  the policy boundary for packed-i4 low-precision product-reduction candidates
  before selected-dispatch/performance preference can be accepted.
- Confirmed focused plugin coverage parses the checked-in Gate 3 dequant-clamp
  source-backed `same_target_measurement_record` JSON, feeds it through the
  selected-dispatch record overload, selects correctness fallback, denies
  performance preference, and rejects stale schedule-decision and
  correctness-disabled records.
- Confirmed positive measured-win fixtures select `performance-preferred` only
  after provider maturity, eligibility, dispatch preference, remediation facts,
  measurement identity, and source-backed tie-backs all agree.
- Confirmed negative policy coverage rejects stale measurement identity,
  speedup range, missing `ssh rvv` evidence, stale provider tie-back, stale
  primitive facts, stale schedule-decision facts, stale target/runtime facts,
  sibling-route measurements, correctness-disabled records, and measurement-only
  win promotion, while safe resolvers preserve correctness fallback for legal
  routes.

## Final Verification

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- `python3 -m json.tool artifacts/gate3-packed-i4-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json`
- `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- `git diff --check`
- `git diff --cached --check`
- Bounded current-diff scan for legacy RVV route-authority markers found no
  positive legacy-authority additions.

## Spec Update Decision

No `.trellis/spec/` update was required in the final Gate 4 slice. The RVV
plugin spec already contains the Gate 4 packed-i4 selected-dispatch/performance
policy consumption contract, including the source-backed record overload,
selected-dispatch boundary checks, resolver fallback behavior, stale provenance
rejection, and measured-win acceptance requirements.

## Status

Archived after Gates 1-4 completed. Future work should open a new task only if
human steering selects another RVV Stage 2 capability owner.
