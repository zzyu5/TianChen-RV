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
- Implement the current round as a coherent Gate 2 production slice because
  live code and commit `7829700f` prove Gate 1 is complete.
- Add or repair structured measurement-diagnosis and policy-handoff facts for
  the existing packed-i4 low-precision route.
- The diagnosis must distinguish at least: correctness-supported,
  no-win/regression, stale measurement, stale sibling route, and
  performance-preferred/measured-win outcomes as structured facts or
  fail-closed reasons.
- Provider route validation and target artifact validation must consume the
  structured handoff and reject stale sibling measurements before dispatch or
  performance preference can be authorized.
- Correctness support must remain allowed for the accepted packed-i4
  regression/no-win outcome.
- Performance-preferred dispatch and performance-win claims must remain denied
  for the accepted regression/no-win outcome.
- Do not use artifact names, route ids, q4/q8/llama labels, metadata status,
  or Common EmitC inference as authority.
- Do not run broad smoke matrices. Use focused provider/target tests and the
  already accepted Gate 4 measurement unless this round claims new runtime or
  performance evidence.

## Macro Campaign Gates

- [x] Gate 1: measurement-result diagnosis and policy handoff distinguishes
  correctness-supported, no-win, regression, stale measurement, stale sibling
  route, and performance-preferred outcomes using structured facts.
- [x] Gate 2: Gearbox/resource-aware selected-body realization or
  primitive-chain planning consumes those facts or exposes the precise missing
  resource/schedule blocker.
- [ ] Gate 3: provider route facts and target artifact validation mirror the
  remediated resource/measurement/policy facts without metadata authority.
- [ ] Gate 4: generated artifacts are measured on the same target and accepted
  only with real `ssh rvv` evidence.
- [ ] Gate 5: dispatch/performance policy enables a performance-preferred path
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
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new q4/q8/llama route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, or legacy i32
  positive route.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the slice, while `.trellis/.current-task`
  remains active with a precise continuation point.

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

Continue with Gate 3: lift the Gate 2 remediation decision into the full
provider route fact and target artifact validation surface without giving
metadata authority. Keep the same fail-closed policy for stale resource,
measurement, schedule, primitive-chain, or candidate facts. Gates 4 and 5 remain
after that: new same-target measurement evidence and dispatch/performance
policy enablement only when production facts and measurements justify it.
