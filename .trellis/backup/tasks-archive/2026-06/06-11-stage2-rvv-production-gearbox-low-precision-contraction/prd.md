# Stage2 RVV Production-Kernel Gearbox Resource-Aware Low-Precision Contraction Campaign

## Direction

This is a macro production-capability campaign for RVV low-precision
production kernels. The owner is the spine that lets typed `tcrv_rvv` selected
bodies, Gearbox/resource schedules, low-precision contraction primitive facts,
route planning, target mirrors, and later same-target measurements agree
without q8/q4 names, artifact names, route IDs, helper names, or common EmitC
semantics becoming authority.

## Campaign Gates

1. Production resource/primitive fact spine exists with explicit producers,
   consumers, target mirrors, and stale-fact rejection.
2. Low-precision contraction primitive/resource surface covers typed
   packed-i4/i8/u8 load/unpack, widening product, and reduction facts without
   q8/q4 route authority.
3. Gearbox selected-body realization consumes those facts to build
   resource-aware executable bodies without changing semantics, ABI roles,
   dispatch, fallback behavior, or runtime AVL.
4. Same-target measurement admits only source-backed measured wins into
   performance-preferred dispatch while preserving correctness fallback for
   no-win records.

## Repository Findings

Gate 1 is already present in production source from the prior archived
resource/primitive and selected-dispatch campaigns:

- Dequant/dequant-clamp resource candidates carry planning contract, selected
  candidate, source/product/accumulator/result facts, primitive chain facts,
  realization facts, target capability mirrors, and stale mirror rejection.
- Packed-i4 resource selections carry remediation/performance admission facts
  through provider policy and target validation.
- Target artifact validation already verifies resource/primitive mirrors when
  a low-precision resource selection is present.

The first Gate 2 slice added the plain product-reduction resource spine.
`WideningProductReduceAdd` now supports signed and unsigned low-precision
primitive facts, including unsigned u8/u16/u32 route planning, resource
selection, and target artifact validation.

The second Gate 2 slice completed the packed-i4 load/unpack resource-fact
spine. Packed-i4 candidates now carry an explicit load/unpack contract,
storage-load fact, unpack-plan fact, and unpacked-source fact. Gearbox
materializes those facts, selected-body realization consumes and mirrors them,
route planning rejects stale handoff facts before route construction, and target
artifact validation rejects stale provider or metadata mirrors. The authority
remains typed `tcrv_rvv` body/config/runtime facts plus RVV-owned resource
candidates, not q8/q4 labels, artifact names, route IDs, helper names, or common
EmitC semantics.

## Completed Gate 2 Slices

The prior rounds completed the Gate 2 widening-product/reduction
resource-candidate fact spine:

- Add explicit widening-product and widening-reduction resource-candidate facts
  to the production low-precision resource candidate/selection surface, separate
  from but derived from the provider-owned primitive chain facts.
- Produce those candidate facts in the RVV-owned Gearbox/resource candidate
  path for product-reduction, dequantize, dequant-clamp, and packed-i4-adjacent
  low-precision candidates.
- Consume the same candidate facts in selected-body realization and Gearbox
  cross-region handoff validation before route construction.
- Carry the facts through `TCRVEmitCLowerableRoute` planning metadata and target
  artifact mirrors, with stale provider and stale target rejection.
- Keep existing packed-i4 load/unpack, signed/unsigned product-reduction, and
  dequant/dequant-clamp resource paths passing.

## Current Gate 3A Slice

The current round owns selected-body realization consumption for the completed
low-precision resource fact spine. The bounded target is the
`widening_product_reduce_dequantize_f32` / product-reduce-add-adjacent spine,
including packed-i4 load/unpack and widening-product/reduction candidate facts.

Selected-body realization must consume the selected low-precision resource
candidate before route construction and materialize resource-aware realized
body structure. The realized producer/consumer `with_vl` scopes,
`tcrv_rvv.vsetvl_region_marker` ops, and
`tcrv_rvv.gearbox_cross_region_handoff` must carry provider-verifiable resource
facts derived from the selected candidate: realization decision, unroll,
region count, peak live-vector pressure, product/dequant region indices,
product/dequant phases, packed-i4 load/unpack facts when selected, and
widening-product/reduction primitive/candidate facts. Route planning must read
and compare those realized-body facts before constructing a
`TCRVEmitCLowerableRoute`; it must not reconstruct acceptance from route IDs,
artifact names, helper names, primitive strings alone, or common EmitC logic.

## Acceptance Criteria For Gate 3A

- Production source movement happens in RVV selected-body realization or its
  direct route-planning consumers, not in common EmitC semantic inference.
- Realized product/reduction/dequant producer and consumer scopes carry
  resource-aware schedule facts from the selected candidate, including
  product/dequant region index and phase facts in addition to the existing
  realized unroll, region count, peak-live, packed-i4, and primitive/candidate
  fields.
- Route planning consumes the realized-body schedule facts and rejects missing
  or stale producer/consumer `with_vl`, marker, or Gearbox handoff facts before
  `TCRVEmitCLowerableRoute` construction.
- Packed-i4 load/unpack facts and widening-product/reduction candidate facts
  remain required when selected and continue to fail closed when stale at the
  provider/handoff/target boundary.
- Focused lit/FileCheck fixtures prove realized-body fact consumption and stale
  realized-body/handoff/target fact rejection for the dequantize-f32
  product-reduction representative, with adjacent packed-i4 and dequant-clamp
  coverage kept passing.
- `tianchenrv-rvv-extension-plugin-test` and
  `tianchenrv-target-artifact-export-test` pass.

## Non-Goals

- No q8/q4-named route IDs, artifact authority, or helper-op authority.
- No new high-level Linalg/Vector/StableHLO frontend work.
- No selected-dispatch performance-preferred admission without fresh
  source-backed same-target measured-win evidence.
- No runtime `ssh rvv` claim in this slice; this is a provider/target resource
  fact-spine slice.

## Status Before Current Slice

Gate 2 is complete for the current low-precision resource fact spine:
packed-i4 load/unpack facts and widening-product/reduction candidate facts are
explicit, provider-produced, selected-body consumed, route-planning mirrored,
target-artifact mirrored, and stale-rejected in focused signed/unsigned and
packed-i4-adjacent coverage. The macro task remains active because Gate 3 and
Gate 4 campaign gates are not complete.

Continuation point: move to Gate 3 selected-body realization consumption using
the completed resource fact spine to build resource-aware executable bodies
without changing computation semantics, ABI roles, dispatch, fallback, or
runtime AVL. Do not move to Gate 4 performance-preferred dispatch without fresh
source-backed same-target measured-win evidence.

## Status After Gate 3A Slice

Gate 3A is complete for the current product-reduction/dequantize-f32
realization-consumption slice. RVV selected-body realization and Gearbox
schedule materialization now write provider-verifiable realization
producer/decision, realized unroll, realized `vsetvl` region count, realized
peak-live vector pressure, product/dequant region indexes, and product/dequant
phases onto realized producer/consumer `with_vl` scopes. Route planning reads
those realization-produced facts before constructing a
`TCRVEmitCLowerableRoute` and rejects stale product/dequant fact values before
route acceptance.

Gate 3 is not fully complete for the macro campaign. Remaining Gate 3 work is
to extend the same resource-aware realized-body consumption pattern across the
next low-precision body-family representative, keeping packed-i4 load/unpack
facts and widening-product/reduction candidate facts as producer-owned inputs.
Gate 4 remains later work and still requires fresh source-backed same-target
measured-win evidence before any performance-preferred dispatch admission.

## Current Gate 3B Slice

This round owns the next low-precision body-family representative:
`widening_product_reduce_dequant_clamp_f32`. The goal is to extend the Gate 3A
realized-body fact consumption contract from product/reduction/dequant facts to
the dequant-clamp epilogue without changing computation semantics, ABI roles,
dispatch/fallback behavior, or runtime AVL.

The dequant-clamp realized producer/consumer scopes and Gearbox handoff must
carry clamp/compare-select region and phase facts derived from the selected
low-precision resource candidate. Route planning and target artifact
validation must consume those facts before route or artifact acceptance and
must reject stale dequant-clamp realized-body, handoff, provider, or target
mirrors.

## Acceptance Criteria For Gate 3B

- Production source movement happens in the RVV Gearbox/resource schedule,
  selected-body realization, route-planning consumer, dialect handoff
  structure, or target validation spine; Common EmitC does not infer
  dequant-clamp semantics.
- Dequant-clamp producer/consumer `with_vl` scopes carry the completed Gate 3A
  realization facts plus clamp/compare-select phase and region facts derived
  from the selected resource candidate.
- `tcrv_rvv.gearbox_cross_region_handoff` carries dequant-clamp clamp region,
  clamp phase, compare/select phase, and select-layout facts when the selected
  resource candidate is a dequant-clamp candidate, and rejects those facts for
  non-clamp candidates.
- Route planning reads the dequant-clamp realization facts from the realized
  body before constructing `TCRVEmitCLowerableRoute` and rejects stale
  producer/consumer or handoff clamp facts.
- Target artifact validation mirrors the provider-selected dequant-clamp
  realization facts exactly and rejects stale target mirrors before artifact
  acceptance.
- Focused lit/FileCheck coverage proves positive dequant-clamp fact
  materialization/consumption and at least one stale realized-body or handoff
  dequant-clamp fact failure before route acceptance, while completed Gate 2
  and Gate 3A tests remain passing.

## Status After Gate 3B Slice

Gate 3B is complete for
`widening_product_reduce_dequant_clamp_f32`. RVV Gearbox schedule
materialization, selected-body realization, route planning, dialect handoff
verification, target validation, and support-bundle mirrors now carry and
consume the provider-owned dequant-clamp realization facts:
`clamp_region_index`, `clamp_phase`, `clamp_compare_select_phase`, and
`clamp_select_layout`. Stale realized-body clamp phase, stale handoff clamp
select layout, and stale target clamp select layout fail closed before route or
artifact acceptance. The macro task remains active because Gate 3 still needs a
family completion/audit pass, and Gate 4 still needs fresh source-backed
same-target measured-win evidence before any performance-preferred dispatch
admission.

## Current Gate 3C Slice

This round owns the Gate 3 family completion/audit for the low-precision
resource-aware selected-body realization representatives already present in
production tests:

- signed and unsigned plain `widening_product_reduce_add` primitive/resource
  representatives, as adjacent Gate 2 route-supported resource facts rather than
  Gearbox two-region realization owners;
- `widening_product_reduce_dequantize_f32`, including grouped/unpacked-byte and
  packed-i4 sibling resource candidates;
- `widening_product_reduce_dequant_clamp_f32`, including grouped/unpacked-byte
  and packed-i4 sibling resource candidates.

The slice must close any remaining inconsistency in how producer-owned
realization decisions derive region count, product/dequant phases, packed-i4
resource facts, clamp facts, handoff facts, route-planning selection, and target
mirrors. If the audit finds no missing source gap, it must leave a bounded
production-backed completion proof and set the next continuation point to Gate 4
measured same-target comparison, without archiving the macro task.

## Acceptance Criteria For Gate 3C

- The Gate 3 family inventory is tied to the production source files and focused
  fixtures above, not to artifact names or q8/q4 labels.
- Gearbox schedule validation, selected-body realization, dialect handoff
  verification, route planning, and target validation use the same provider-owned
  realization decision mapping for region count and product/dequant phases.
- Non-clamp product-dequant candidates reject dequant-clamp realization facts at
  the realized-body or handoff boundary before Common EmitC materialization.
- Existing Gate 3A/Gate 3B positive and stale-fact coverage remains passing for
  grouped/unpacked-byte and packed-i4 siblings.
- The PRD, task notes, and journal record whether Gate 3 is complete after this
  audit and keep Gate 4 blocked until fresh source-backed same-target measured
  wins exist.

## Status After Gate 3C Slice

Gate 3 is complete for the current campaign body-family audit. Gearbox schedule
validation, selected-body realization, dialect handoff verification, route
planning, and target validation now use the same provider-owned realization
decision mapping for supported decisions, region count, product phase, and
product/dequant handoff facts. Non-clamp product-dequant handoff coverage proves
dequant-clamp-only facts are rejected before Common EmitC materialization.

The macro task remains active because Gate 4 is not complete. Gate 4 must
consume source-backed same-target measurement evidence before dispatch
preference changes. The current accepted packed-i4 evidence is no-win/regression
for the representative paths, so production dispatch must preserve
`correctness-fallback` and `not-performance-preferred` unless a newer
source-backed measured win and matching provider-owned admission facts arrive.

## Current Gate 4 Slice

This round owns Gate 4 source-backed same-target measurement admission at the
evidence-root ingestion boundary for the packed-i4
`widening_product_reduce_dequantize_f32` representative, with
`widening_product_reduce_dequant_clamp_f32` kept as the sibling audit path.

The existing same-target measurement script already emits root-level
`measurement_harness` and `measurement_schedule_decision_evidence` objects that
mirror provider-owned schedule/resource-cost/performance-admission facts. The
production RVV low-precision performance policy must validate those root-level
mirrors against the parsed source-backed measurement record before selected
dispatch can consume the policy decision. A stale root-level performance
admission closure or reopen requirement must fail closed even if the nested
`same_target_measurement_record` still carries the accepted no-win record.

## Acceptance Criteria For Gate 4 Root Admission Slice

- Production source movement happens in the RVV low-precision performance
  policy evidence-root verifier, not in Common EmitC semantic inference or
  artifact-name logic.
- The evidence-root verifier checks root `measurement_harness` and
  `measurement_schedule_decision_evidence` performance admission closure and
  reopen requirement fields against the provider-owned measurement record.
- The accepted dequant packed-i4 source-backed same-target evidence continues to
  select `correctness-fallback`, deny performance preference, preserve route
  support/correctness execution, and populate selected-dispatch policy-output
  mirrors only from provider-owned policy decisions.
- A stale root-level performance admission closure or reopen requirement is
  rejected before policy evaluation can authorize selected dispatch.
- The dequant-clamp packed-i4 sibling evidence root continues to be parsed and
  consumed through the same policy path, preserving no-win fallback behavior.
- The macro task remains active after this slice unless Gate 4 campaign-level
  acceptance is fully satisfied; a committed slice with the task still open is
  valid.

## Status After Gate 4 Root Admission Slice

Gate 4 root-admission closure/reopen validation is complete for the current
packed-i4 source-backed evidence-root slice. The accepted measurement result
remains no-win, so dispatch stays `correctness-fallback` /
`not-performance-preferred`. The macro task remains active for remaining Gate 4
campaign-level measurement/admission audit and any future measured-win
admission only after fresh source-backed same-target evidence plus matching
provider admission facts exist.

## Current Gate 4 Policy-Output Mirror Admission Slice

This round owns the selected-dispatch / target-artifact policy-output mirror
admission boundary for packed-i4 `widening_product_reduce_dequantize_f32` and
the sibling `widening_product_reduce_dequant_clamp_f32` audit path. The goal is
to ensure the source-backed same-target no-win policy decision is carried
through provider-owned selected-dispatch policy-output fields and target
artifact mirrors without allowing artifact metadata, route IDs, helper names,
or Common EmitC logic to choose dispatch preference.

The bounded production gap is the selected-dispatch preference mirror:
`selected_dispatch_preference` is part of the provider-owned policy-output
contract and must be emitted and validated as its own exact mirror, in addition
to the existing low-precision resource `dispatch_preference` mirror. It must
come from the accepted `RVVLowPrecisionPerformancePolicyDecision`, not from
target metadata or artifact names.

## Acceptance Criteria For Gate 4 Policy-Output Mirror Admission Slice

- Production source movement happens in RVV provider-owned route metadata,
  target support-bundle mapping, and target artifact validation; Common EmitC
  still only carries provider-built mirrors.
- `selected_dispatch_preference` is emitted into target metadata only when
  `hasSelectedDispatchPolicyOutput` is true, and it mirrors the accepted
  `RVVLowPrecisionPerformancePolicyDecision::dispatchPreference` exactly.
- Target artifact validation rejects a metadata-only
  `selected_dispatch_preference` mirror when provider-owned policy-output facts
  are absent.
- Target artifact validation rejects stale `selected_dispatch_preference`
  mirrors, including attempted `performance-preferred` promotion while the
  accepted packed-i4 source-backed evidence remains no-win.
- Existing accepted packed-i4 dequant and dequant-clamp source-backed no-win
  records continue to populate selected-dispatch policy-output facts, preserve
  `correctness-fallback`, and deny performance-preferred dispatch.
- The macro task remains active after this slice unless all Gate 4
  campaign-level measured-win/no-win, provider, selected-dispatch, target, and
  stale-mirror gates are actually complete.

## Status After Gate 4 Policy-Output Mirror Admission Slice

Gate 4 policy-output mirror admission is complete for the current packed-i4
source-backed no-win selected-dispatch/target boundary. The provider route
metadata, target support bundle, and target artifact validation now carry
`selected_dispatch_preference` as an explicit provider-owned policy-output
mirror. Target validation rejects metadata-only policy-output mirrors and stale
`performance-preferred` promotion while the accepted packed-i4 same-target
records remain no-win/regression.

The macro task remains active. Gate 4 still needs the final dispatch/fallback
consumption consistency audit: source-backed same-target records, accepted
`RVVLowPrecisionPerformancePolicyDecision`, selected-dispatch case/fallback
facts, policy-output mirrors, target artifact mirrors, and conservative
fallback intent must agree before any dispatch preference or win claim can be
consumed.

## Current Gate 4 Dispatch/Fallback Consumption Slice

This round owns the selected-dispatch and fallback consumption consistency
slice for packed-i4 `widening_product_reduce_dequantize_f32` and the sibling
`widening_product_reduce_dequant_clamp_f32` audit path. The goal is to make the
accepted no-win policy decision flow through the real consumer boundary:
source-backed same-target evidence -> provider-owned
`RVVLowPrecisionPerformancePolicyDecision` -> selected-dispatch case/fallback
facts -> selected-dispatch policy-output mirrors -> target artifact mirrors ->
conservative correctness fallback. No artifact metadata, route id, helper name,
Common EmitC string, or selected-dispatch mirror may change dispatch preference
or bypass fallback consumption.

The current accepted records are no-win/regression. Therefore the production
path must keep `routeSupportAllowed = true`, `correctnessExecutionAllowed =
true`, `performanceSelectionAllowed = false`, `performanceWinClaimAllowed =
false`, `dispatchPolicyPath = correctness-fallback`,
`selected_dispatch_preference = not-performance-preferred`,
`correctness_fallback_path_selected = true`, and
`performance_preferred_path_selected = false` across provider, target, and
dispatch/fallback consumers.

## Acceptance Criteria For Gate 4 Dispatch/Fallback Consumption Slice

- Production source movement happens in the RVV policy/selected-dispatch
  boundary and target artifact consumer; Common EmitC still only carries
  provider-built route metadata.
- A packed-i4 selected-dispatch policy-output boundary is accepted only when it
  is backed by complete selected-dispatch case facts, fallback facts, and the
  accepted no-win `RVVLowPrecisionPerformancePolicyDecision`.
- Accepted source-backed dequant and dequant-clamp no-win records preserve
  route support and correctness execution while keeping correctness fallback
  selected and denying performance-preferred path selection.
- Target artifact validation rejects stale or metadata-only
  `fallback_reason`, `correctness_fallback_path_selected`,
  `performance_preferred_path_selected`, `performance_win_claim_allowed`,
  selected-dispatch case mirror, selected-dispatch fallback mirror, or
  policy-output mirrors before artifact/header export.
- The dequant-clamp packed-i4 sibling fixture exposes and validates the same
  selected-dispatch policy-output mirrors as the dequant representative.
- No measured-win path is admitted in this slice unless it is already backed by
  explicit source-backed measured-win evidence and synchronized provider/target
  mirrors; otherwise measured-win admission remains blocked.
- The macro task remains active unless the remaining Gate 4 campaign-level
  measured-win/no-win, provider, selected-dispatch, target, and
  dispatch/fallback gates are all proven complete.

## Status After Gate 4 Dispatch/Fallback Consumption Slice

Gate 4 dispatch/fallback consumption consistency is complete for the accepted
packed-i4 dequant and dequant-clamp no-win evidence paths. Target artifact
validation now rejects stale provider-owned selected-dispatch policy-output
values even when candidate metadata mirrors the stale provider value exactly.
The accepted no-win records still preserve route support and correctness
execution while denying performance-preferred path selection and performance
win claims.

The macro task remains active. Gate 4 still needs an admission-boundary cleanup
audit so stable Gearbox/resource schedule facts, provider/target mirrors,
generic selected-dispatch policy-output facts, and campaign measurement or
admission records stay in their correct owner layers.

## Current Gate 4 Admission Boundary Cleanup Slice

This round owns the authority boundary between stable RVV low-precision
resource/schedule/selected-dispatch facts and Gate 4 experiment/admission
evidence. The bounded production target is the packed-i4
`widening_product_reduce_dequantize_f32` path recently hardened in target
validation, with the packed-i4 dequant-clamp sibling preserved.

The field inventory for this cleanup is:

- Stable compiler/resource facts: candidate set, selected candidate, planning
  contract, dtype/SEW/LMUL, packed-i4 load/unpack, primitive product/reduction,
  resource cost, region count, live-vector pressure, realization decision,
  product/dequant/clamp phases, route-family plan, provider-supported mirror,
  target capability mirrors, legality and rejection reason.
- Generic policy-output facts: selected-dispatch case/fallback mirrors and the
  accepted `RVVLowPrecisionPerformancePolicyDecision` outputs
  `selected_dispatch_policy_contract`, `dispatch_policy_path`,
  `selected_dispatch_preference`, denial/fallback reason, route support,
  correctness execution, performance selection, win-claim allowance, and path
  selection booleans.
- Campaign evidence/admission facts: performance feedback/baseline/speedup
  range/action, remediation handoff and measurement evidence, remediation
  diagnosis/action/plan, performance admission closure/reopen fields,
  beyond-local repair admission fields, performance maturity fields, and
  source-backed same-target measurement records.

The cleanup must remove at least one real misplaced dependency where a stable
compiler schedule or resource mirror accepts or rejects based on campaign
evidence/admission fields. It may keep the current mirrored field names for
artifact compatibility, but the production helpers and diagnostics must make
the owner boundary explicit.

## Acceptance Criteria For Gate 4 Admission Boundary Cleanup Slice

- The packed-i4 Gearbox schedule decision is accepted from stable resource and
  schedule facts, not from Gate 4 performance admission closure/reopen or
  beyond-local repair admission fields.
- Target artifact validation separates stable resource candidate mirror
  validation from Gate 4 campaign evidence/admission mirror validation, while
  still checking both for packed-i4 routes that carry those fields.
- Accepted dequant and dequant-clamp no-win records still preserve
  `correctness-fallback`, route support, and correctness execution while
  denying performance-preferred path selection and performance win claims.
- Stale Gate 4 admission/evidence mirrors still fail closed at the target or
  policy boundary, but they are not route authority, schedule authority,
  artifact-name authority, or permanent Gearbox schedule contract inputs.
- Focused target or plugin tests prove the stable schedule-decision helper no
  longer depends on stale/missing Gate 4 admission fields, and existing stale
  selected-dispatch / no-win rejection coverage remains passing.
- The macro task remains active after this cleanup unless the full Gate 4
  measured-win/no-win, provider, selected-dispatch, target, and
  dispatch/fallback campaign gates are actually complete.

## Status After Gate 4 Admission Boundary Cleanup Slice

Gate 4 admission-boundary cleanup is complete for the first packed-i4
production slice. The packed-i4 stable Gearbox schedule-decision helper now
accepts from stable resource/schedule facts and no longer accepts or rejects
based on performance-admission closure/reopen or beyond-local repair admission
fields. Target artifact validation now consumes stable resource mirrors and
Gate 4 evidence/admission mirrors through separate helpers, with Gate 4 stale
mirror diagnostics labeled as evidence/admission ownership rather than stable
resource authority.

The accepted dequant and dequant-clamp records remain no-win/regression:
route support and correctness execution are preserved, while
performance-preferred selection and performance-win claims remain denied. The
macro task stays active for remaining Gate 4 measured-win admission and final
end-to-end provider, selected-dispatch, target mirror, and dispatch/fallback
closure.

## Current Gate 4 Final Admission/Consumer Closure Slice

This round owns the final Gate 4 admission/consumer consistency audit for the
packed-i4 low-precision representatives already used by the campaign:
`widening_product_reduce_dequantize_f32` and
`widening_product_reduce_dequant_clamp_f32`.

The current source-backed same-target evidence remains no-win/regression, so no
new performance-preferred dispatch may be admitted in this slice. The production
path must continue to prove the conservative no-win chain end to end:
source-backed same-target record -> accepted
`RVVLowPrecisionPerformancePolicyDecision` -> selected-dispatch case/fallback
facts -> selected-dispatch policy-output mirrors -> target artifact mirrors ->
correctness fallback. The same target consumer boundary must also prove that a
future measured-win path cannot be faked from metadata: synthetic
`performance-preferred` provider fields and target mirrors remain rejected
until a fresh source-backed measured-win record can be consumed through the same
provider-owned policy and target-validation chain.

Code inspection before implementation found that production source already has
the required Gate 4 owner layering for this bounded slice:

- `RVVLowPrecisionPerformancePolicy` has evidence-root, record, policy-input,
  selected-dispatch-boundary, and policy-output population overloads.
- Route analysis uses the provider-owned no-win default record for the current
  packed-i4 path, while tests can feed parsed source-backed evidence-root
  records through the explicit record overload.
- Target artifact validation already has separate stable resource mirror,
  Gate 4 evidence/admission mirror, selected-dispatch case/fallback mirror, and
  selected-dispatch policy-output mirror consumers, including a
  `performance-preferred` branch guarded by provider policy-output facts.

Therefore the intended production movement for this slice is focused coverage
at the real target consumer boundary rather than adding new compiler semantics.

## Acceptance Criteria For Gate 4 Final Admission/Consumer Closure Slice

- No fresh source-backed measured-win evidence is admitted; the current dequant
  and dequant-clamp source-backed records remain `correctness-fallback` /
  `not-performance-preferred`.
- Target artifact coverage feeds a parsed source-backed dequant evidence-root
  record through `populateRVVLowPrecisionSelectedDispatchPolicyOutput` and
  validates that no-win selected-dispatch mirrors are consumed by target
  artifact validation.
- Target artifact coverage proves a synthetic measured-win fixture remains
  blocked at the current no-win target admission boundary, even if provider
  maturity, selection eligibility, performance admission closure/reopen,
  beyond-local repair admission, selected-dispatch case/fallback facts, and
  policy-output mirrors are rewritten together without fresh source-backed
  target evidence.
- Target artifact validation rejects measured-win promotion when the target
  artifact metadata is rewritten without the provider-owned policy-output
  boundary matching it, including stale `dispatch_policy_path`,
  `selected_dispatch_preference`, or `performance_win_claim_allowed`.
- Existing stale no-win, metadata-only policy-output, selected-dispatch
  case/fallback, and evidence-root rejection coverage remains passing.
- The macro task may close Gate 4 for the current campaign if the no-win
  fallback closure and stale measured-win promotion rejection are proven across
  provider, selected-dispatch, target, and dispatch/fallback consumers. A later
  real measured-win round must update source-backed evidence and provider-owned
  admission facts before changing dispatch preference.

## Status After Gate 4 Final Admission/Consumer Closure Slice

Gate 4 final no-win admission/consumer closure is complete for the current
packed-i4 dequant and dequant-clamp campaign representatives. The source-backed
same-target evidence remains no-win/regression, so no performance-preferred
dispatch or performance-win claim was admitted. The dequant evidence-root record
continues to drive selected-dispatch policy-output mirrors through the explicit
record overload, target artifact validation consumes those no-win mirrors, and
synthetic measured-win promotion remains fail-closed at target validation unless
a future fresh source-backed measured-win evidence/admission chain replaces the
current no-win default.

All four campaign gates are complete for the current packed-i4 low-precision
Gearbox campaign scope. Future measured-win work, if new evidence appears, is a
new evidence/admission update on top of this closed no-win boundary rather than
an open blocker for this macro task.
