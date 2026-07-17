# Stage2 RVV low-precision performance remediation campaign

## Goal

Create the next macro production-capability owner for RVV low-precision
performance remediation. The compiler path must consume same-target
low-precision measurement outcomes as structured RVV-owned facts, then use
those facts to drive resource-aware selected-body realization, provider route
facts, target artifact validation, and dispatch/performance policy. A measured
win may become performance-preferred only through this structured path; a
measured no-win/regression must preserve correctness/executable support while
fail-closing performance preference and win claims with a precise reason.

## What I already know

- The previous archived macro task at
  `.trellis/tasks/archive/2026-06/06-09-rvv-production-kernel-capability-campaign/`
  completed Gates 1-5 for the earlier production-kernel capability campaign.
- Commit `89c773fe` added a production
  `RVVLowPrecisionPerformancePolicy` consumer for the accepted packed-i4
  same-target measurement. Current policy preserves executable correctness and
  denies performance preference for the accepted regression/no-win outcome.
- Current policy code verifies accepted measurement identity, target profile,
  `ssh rvv` evidence, provider maturity tie-back, primitive-chain facts,
  dispatch preference, and measurement-only win promotion.
- The new campaign starts after that policy rejection. Its first bottleneck is
  to make the measurement-result diagnosis and policy handoff explicit and
  structured, so later Gate 2 resource-aware remediation can consume a named
  diagnosis instead of interpreting raw measurement/policy fields.
- `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md` require RVV Stage 2
  performance work to stay on the typed `tcrv_rvv` body / RVV provider /
  Common EmitC / target artifact / `ssh rvv` evidence chain.

## Requirements

- Keep this as one macro Trellis task until all remediation campaign gates are
  complete or human steering redirects the campaign.
- Implement the current round as a coherent Gate 5 dispatch/performance policy
  consumption slice because live code and commit `c8f1dfac` prove Gates 1-4 are
  complete.
- Add or repair structured measurement-diagnosis and policy-handoff facts for
  the existing packed-i4 low-precision route.
- The diagnosis must distinguish at least: correctness-supported,
  no-win/regression, stale measurement, stale sibling route, and
  performance-preferred/measured-win outcomes as structured facts or
  fail-closed reasons.
- Provider route validation and target artifact validation must consume the
  structured handoff and reject stale sibling measurements before dispatch or
  performance preference can be authorized.
- Provider route facts and target artifact validation must carry the
  low-precision resource route-family plan and provider-supported mirror as
  RVV-owned facts tied to the remediation/resource/primitive chain, not as
  artifact-name, route-id, sidecar, report-text, or Common EmitC authority.
- Correctness support must remain allowed for the accepted packed-i4
  regression/no-win outcome.
- Performance-preferred dispatch and performance-win claims must remain denied
  for the accepted regression/no-win outcome.
- Do not use artifact names, route ids, q4/q8/llama labels, metadata status,
  or Common EmitC inference as authority.
- Do not run broad smoke matrices. Use focused provider/target/export and
  same-target generated-artifact measurement checks for Gate 4; do not claim a
  performance win unless real `ssh rvv` evidence classifies one through the
  structured evidence path.

## Macro Campaign Gates

- [x] Gate 1: measurement-result diagnosis and policy handoff distinguishes
  correctness-supported, no-win, regression, stale measurement, stale sibling
  route, and performance-preferred outcomes using structured facts.
- [x] Gate 2: Gearbox/resource-aware selected-body realization or
  primitive-chain planning consumes those facts or exposes the precise missing
  resource/schedule blocker.
- [x] Gate 3: provider route facts and target artifact validation mirror the
  remediated resource/measurement/policy facts without metadata authority.
- [x] Gate 4: generated artifacts are measured on the same target and accepted
  only with real `ssh rvv` evidence.
- [x] Gate 5: dispatch/performance policy enables a performance-preferred path
  for a measured win, or fail-closes performance claims while retaining
  correctness support for measured no-win/regression.

## Completed Gate 1 Slice

- [x] Add a structured RVV low-precision measurement diagnosis / policy handoff
  surface to `RVVLowPrecisionPerformancePolicy`.
- [x] Route policy decisions through the structured diagnosis rather than
  directly interpreting scattered measurement strings at the final decision
  point.
- [x] Keep the accepted packed-i4 Gate 4 measurement classified as
  same-target `regression` / `no-win`, with correctness allowed and
  performance preference denied.
- [x] Add focused provider coverage showing accepted no-win/regression,
  stale measurement, stale sibling-route measurement, and attempted
  performance-preferred promotion cannot authorize performance-preferred
  dispatch.
- [x] Add focused target-artifact coverage showing stale sibling-route
  measurement cannot authorize target acceptance or performance preference.
- [x] Leave the macro task active after this slice because Gates 2-5 remain
  open.

## Completed Gate 2 Slice

- [x] Add a provider-owned Gate 2 remediation decision to the selected
  low-precision resource planning facts for the accepted packed-i4
  representative.
- [x] Derive that remediation decision from the structured
  `RVVLowPrecisionPerformancePolicyHandoff` instead of re-reading raw
  measurement strings in resource planning.
- [x] Require selected-body realization/pass facts, route-family plans, route
  descriptions, and target artifact mirrors to agree on the handoff contract,
  diagnosis, selected measurement identity, resource action, and dispatch
  consequence.
- [x] Keep the accepted same-target regression/no-win result as a
  correctness-supported resource decision that preserves route support but
  blocks performance-preferred selection and win claims.
- [x] Add focused negative coverage for stale/missing remediation handoff,
  stale diagnosis/action, and stale target mirror facts before route or target
  acceptance.
- [x] Leave the macro task active after this slice because Gates 3-5 remain
  open.

## Completed Gate 3 Slice

- [x] Add low-precision resource route-family plan and provider-supported
  mirror facts to the provider-owned route/resource surface for the accepted
  packed-i4 representative.
- [x] Require those Gate 3 facts to match the validated contraction
  route-family plan before provider route construction.
- [x] Export the Gate 3 facts as target artifact mirrors and reject missing or
  stale candidate metadata before artifact acceptance.
- [x] Keep accepted no-win/regression correctness support unchanged and keep
  performance preference/win claims denied.
- [x] Leave the macro task active after this slice because Gates 4-5 remain
  open.

## Acceptance Criteria

- [x] Production source changes land in policy/handoff and the provider or
  target validation seam that consumes it.
- [x] Focused tests prove accepted no-win/regression preserves route support and
  correctness while denying performance preference and win claims.
- [x] Focused tests prove stale measurement identity and stale sibling-route
  measurement fail closed before performance-preferred dispatch.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Gate 2 source changes make RVV resource-aware planning consume the
  structured measurement diagnosis/policy handoff or fail closed with the
  exact missing handoff/resource/schedule fact.
- [x] Focused tests prove stale or missing Gate 2 remediation handoff facts are
  rejected before route/provider or target artifact acceptance.
- [x] Gate 3 source changes make low-precision resource/provider route facts
  carry route-family plan and provider-supported mirrors derived from the
  validated RVV provider plan.
- [x] Focused provider and target tests prove missing or stale Gate 3
  low-precision route-family/provider-supported mirrors fail closed before
  route construction or target artifact acceptance.
- [x] Gate 4 generated-artifact measurement binds accepted/rejected
  same-target evidence to provider-owned resource, route-family,
  primitive-chain, remediation, target artifact, ABI/runtime, and same-target
  facts.
- [x] Focused script/export tests prove missing or stale Gate 4
  route-family/provider/remediation/target/runtime facts fail closed before
  measurement evidence acceptance.
- [x] Real `ssh rvv` measurement was collected for the generated packed-i4
  remediated artifact and classified as accepted no-win/regression, with
  performance preference and win claims denied.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new q4/q8/llama route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, or legacy i32
  positive route.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the slice, while `.trellis/.current-task`
  remains active with a precise continuation point.
- [x] Gate 5 production policy exposes an explicit dispatch policy path:
  accepted same-target no-win/regression evidence selects correctness fallback,
  while accepted measured-win evidence can select performance-preferred only
  when provider maturity, selection eligibility, dispatch preference, target
  profile, same-target evidence, and win-claim fields all agree.
- [x] Focused provider/policy and target tests prove the current accepted
  Gate 4 regression/no-win outcome preserves route/correctness support and
  denies performance-preferred dispatch, synthetic measured-win facts enable
  the performance-preferred path only through the structured provider contract,
  and stale measurement evidence resolves to correctness fallback or fails
  strict verification before performance preference.

## Completed Gate 1 Slice

Completed this round as the first production source slice:

- [x] Added `RVVLowPrecisionPerformancePolicyHandoff` and
  `RVVLowPrecisionPerformanceMeasurementDiagnosisKind` to make accepted and
  rejected packed-i4 measurement/policy outcomes structured.
- [x] Routed `evaluateRVVLowPrecisionPerformancePolicy` through the handoff so
  accepted no-win/regression, stale measurement, stale sibling-route
  measurement, and performance-preferred/win-promotion attempts are diagnosed
  before dispatch/performance policy decisions are built.
- [x] Kept correctness/executable support enabled and performance-preferred
  dispatch/win claims denied for the accepted same-target `regression` /
  `no-win` measurement.
- [x] Added focused provider and target tests for the new structured handoff
  and stale sibling-route measurement rejection.

Slice result: Gate 1 is complete for the current packed-i4 representative.
The macro task remains active. Gates 2-5 are still open.

## Completed Gate 2 Slice Result

The Gate 2 owner now has a bounded production resource-planning consumer:
the selected packed-i4 low-precision resource selection carries the Gate 1
measurement policy handoff as a provider-owned remediation decision. The RVV
route-family planner derives the decision from
`RVVLowPrecisionPerformancePolicyHandoff`, writes it into selected resource
facts, requires selected-body realization/pass facts to preserve it, exposes it
through route descriptions, and requires target artifact mirrors to match. The
accepted decision is tied to the structured
`correctness-supported-no-win-regression` diagnosis and accepted same-target
packed-i4 measurement identity. It preserves executable route/correctness
support, requires resource/schedule repair before any performance claim, and
blocks performance-preferred dispatch.

This is intentionally not a new runtime measurement or dispatch enablement.
The target/provider mirror checks added in this slice are bounded consistency
checks for the new Gate 2 remediation facts. Gates 3-5 remain open for the
campaign-level provider/target fact surface, new same-target measurement, and
dispatch policy enablement.

## Completed Gate 3 Slice Result

The Gate 3 owner now has a production provider/target mirror surface for the
low-precision resource route-family facts. The packed-i4 low-precision resource
selection carries the validated contraction route-family plan id and
provider-supported mirror from the RVV provider plan. Route-family validation
requires those facts to match the selected provider plan and route description
before route construction can be accepted.

Target artifact export now mirrors those Gate 3 facts through selected-body
metadata and target support bundles, then rejects stale or missing candidate
metadata before artifact acceptance. The checks remain mirror checks only:
artifact metadata can confirm provider-owned facts, but it cannot invent
low-precision semantics, performance preference, or win claims.

This slice does not add new runtime evidence or dispatch enablement. Gate 3 is
complete for the current accepted packed-i4 representative. Gates 4-5 remain
open for same-target generated artifact measurement acceptance and
dispatch/performance policy consumption.

## Completed Gate 4 Slice Result

The Gate 4 owner now has same-target generated-artifact measurement acceptance
bound to the Gate 1-3 structured facts. The packed-i4 generated-bundle path
exports route-family plan, provider-supported mirror, primitive-chain,
remediation, target artifact, and runtime ABI facts through the generated
artifact metadata/header surface. The same-target measurement script ties its
accepted or rejected measurement evidence back to those provider-owned facts
and fail-closes stale or missing resource, route-family, primitive-chain,
remediation, target, or runtime facts before evidence can be accepted.

The real `ssh rvv` Gate 4 run generated and measured the remediated
low-precision artifact at:

`artifacts/tmp/codex-gate4-remediation-real/gate4-remediation-packed-i4-ssh`

Result: 12 summaries and 60 measurements were collected for
`widening_product_reduce_dequantize_f32` with classification `regression`,
best speedup range `0.691667..0.705064`, `selection_eligible=false`, and
`claim_allowed=false`. This is an accepted no-win/regression classification:
correctness/executable support remains preserved, but performance preference
and win claims remain denied.

Gate 4 is complete for the current accepted packed-i4 representative. Gate 5
remains open for dispatch/performance policy consumption of this structured
measurement classification.

## Completed Gate 5 Slice Result

The Gate 5 owner now has an explicit production dispatch/performance policy
surface for low-precision packed-i4 evidence consumption.
`RVVLowPrecisionPerformancePolicyDecision` carries the chosen dispatch policy
path and distinguishes `performance-preferred` from `correctness-fallback`.
The accepted Gate 4 same-target packed-i4 regression/no-win measurement still
preserves executable route support and correctness execution, but resolves to
`correctness-fallback` with performance preference and win claims denied.

The policy also has a strict measured-win path for future provider updates:
a measured win can select `performance-preferred` only when same-target
measurement facts, `ssh rvv` evidence, provider maturity tie-back, performance
selection eligibility, dispatch preference, remediation facts, target profile,
and win-claim fields all agree. Measurement-only win promotion and stale
measurement identity remain fail-closed before performance-preferred dispatch
can be authorized. A non-throwing resolver preserves the safe correctness
fallback path for stale evidence while strict policy verification still rejects
stale/missing facts at provider or target validation seams.

Gate 5 is complete for the current accepted packed-i4 representative. All
macro campaign gates are now complete.

## Out of Scope

- No adjacent generated-bundle ABI closeout.
- No new same-target performance claim unless this round actually runs and
  accepts real `ssh rvv` evidence.
- No one-op route-family seam, q4/q8/llama-named wrapper, artifact-name route
  authority, source-front-door positive route, dtype/LMUL clone batch, broad
  dashboard, performance database, high-level Linalg/Vector/StableHLO frontend,
  or unrelated MAcc/mask/segment/future-plugin expansion.
- No Common EmitC invention of RVV semantics.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous macro PRD read:
  `.trellis/tasks/archive/2026-06/06-09-rvv-production-kernel-capability-campaign/prd.md`.
- Production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.

## Continuation Point

No Gate 5 continuation remains for this macro campaign. Future work should
start from a new Trellis task only if it changes a different production-kernel
capability owner, repairs the packed-i4 provider/resource contract, or collects
new same-target evidence for a genuinely changed production path.
