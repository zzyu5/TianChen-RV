# Stage2 RVV low-precision production-kernel selected-dispatch campaign

## Goal

Create a macro Trellis owner for the RVV Stage 2 low-precision
production-kernel selected-dispatch campaign. The campaign connects the already
proven low-precision primitive surface, Gearbox/resource facts, target artifact
evidence, and same-target measurement policy to a bounded selected
`tcrv.exec` RVV production-kernel dispatch workflow. Gate 1 is complete. The
current round implements Gate 2: selected/pre-realized low-precision
production-kernel bodies must reach RVV route, export, and target-artifact
validation with selected-dispatch case/fallback mirrors and conservative
fallback policy facts still structurally connected to provider-owned
primitive/resource/measurement facts.

## What I Already Know

- There is no `.trellis/.current-task` at session start, and the worktree is
  clean.
- The previous low-precision contraction primitive-surface campaign is
  archived at
  `.trellis/tasks/archive/2026-06/06-10-rvv-low-precision-contraction-primitive-surface-campaign/`.
- Commit `f07722c3` completed the previous campaign Gate 4 by making the
  low-precision measurement policy consume same-target measurement/resource
  facts for the packed-i4 widening product-reduce-dequantize path.
- The next bottleneck is not another generated-bundle evidence closeout. It is
  a production selected-dispatch workflow that uses the completed primitive,
  resource, artifact, and measurement facts as structural policy input.
- Durable authority remains:
  selected `tcrv.exec` RVV boundary -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization and route/provider planning -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materialization -> target artifact
  -> `ssh rvv` evidence when runtime, correctness, or performance is claimed.
- A dispatch or preference decision may mirror route/artifact/evidence facts,
  but it must not derive authority from route ids, q8/q4 labels, artifact
  names, test names, helper names, ABI strings, Common EmitC inference, or
  descriptor residue.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per round and update this PRD with
  completed and remaining gates.
- Gate 1 must introduce or repair the production selected-dispatch policy
  boundary so it consumes low-precision primitive/resource/measurement facts
  structurally.
- Gate 1 must consume, at minimum for the bounded current production path:
  primitive source signedness/type facts, Gearbox selected resource and
  remediation facts, target artifact evidence tie-back, same-target
  measurement identity/range/outcome, and fallback eligibility.
- Gate 1 must fail closed with targeted diagnostics for stale or missing
  primitive, Gearbox/resource, artifact, measurement, ABI, or policy facts.
- Conservative fallback must remain available when route correctness is
  structurally supported but performance preference is not justified.
- Common EmitC/export must remain neutral mechanics. The selected-dispatch
  policy may consume provider-owned route/export/measurement payloads, but
  Common EmitC must not infer RVV semantics or policy state.
- Runtime, correctness, or performance claims require `ssh rvv` evidence only
  when the changed production path makes such a claim in the current gate.
  Gate 1 may use dry-run or focused unit evidence if it validates the changed
  policy boundary without claiming new runtime/performance behavior.

## Macro Campaign Gates

- [x] Gate 1: selected-dispatch policy boundary consumes low-precision
  primitive/resource/measurement facts structurally and fails closed for stale
  or missing facts.
- [x] Gate 2: selected/pre-realized low-precision production-kernel body reaches
  route/export/artifact validation with fallback semantics preserved.
- [ ] Gate 3: same-target measurement evidence updates only the bounded policy
  input and demonstrates preferred-route or conservative-fallback behavior.
- [ ] Gate 4: final `ssh rvv` correctness/performance evidence and spec/task
  archive after production gates are satisfied.

## Current Slice: Gate 1 Selected-Dispatch Policy Boundary

- [x] Inspect selected-dispatch, low-precision performance policy,
  Gearbox/resource handoff, route validation, target artifact validation, and
  same-target measurement consumption code to locate the smallest production
  boundary.
- [x] Add or repair a production compiler surface that represents the bounded
  selected low-precision RVV dispatch decision as provider-owned facts rather
  than route/test/artifact names.
- [x] Consume primitive signedness/type, Gearbox resource/remediation,
  artifact-evidence, same-target measurement, and fallback facts in the policy
  boundary.
- [x] Reject missing, stale, disconnected, cross-target, measurement-only, or
  metadata-only facts before a performance-preferred dispatch decision is
  accepted.
- [x] Preserve conservative fallback when structural correctness support is
  present but performance preference is denied.
- [x] Add focused positive tests for structural policy consumption.
- [x] Add focused negative tests for stale or missing primitive/resource/
  measurement/policy facts.
- [x] Run affected RVV plugin, target artifact, route/export, and script checks.
- [x] Run bounded old-authority scans over touched files and added diff lines.
- [x] Keep this macro task active after Gate 1 unless all gates are truly
  complete.

## Gate 1 Acceptance Criteria

- [x] Production source diff is in the selected-dispatch, policy, or
  route-validation seam that affects the bounded low-precision production path.
- [x] The selected-dispatch policy consumes provider-owned low-precision
  primitive signedness/type facts, Gearbox resource/remediation facts,
  target-artifact evidence tie-back, and same-target measurement facts.
- [x] Missing or stale primitive, resource, artifact, measurement, ABI, or
  policy facts fail closed with targeted diagnostics.
- [x] Conservative fallback remains structurally available when performance
  preference is denied.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, status field, or Common EmitC inference
  becomes dispatch, route, dtype/config, or evidence authority.
- [x] Focused tests exercise accepted structural policy facts and stale/missing
  fact rejection for the changed seam.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin code
  or tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code or tests are touched.
- [x] Relevant `tcrv-opt`, `tcrv-translate`, or script checks pass if
  route/export/generated-bundle paths change.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 1 slice.
- [x] The macro task remains active with Gates 2-4 unchecked and a precise
  continuation point.

## Gate 1 Completion Notes

Gate 1 is complete. The production seam now carries an explicit
`RVVLowPrecisionSelectedDispatchPolicyBoundary` from selected
`tcrv.exec.dispatch` case/fallback collection into the RVV low-precision
performance policy and contraction route-family provider validation. The
boundary requires provider-owned dispatch case origin/policy, conservative
fallback role/origin/policy, selected-dispatch mirrors, accepted primitive/
resource/measurement handoff, and conservative correctness fallback when the
same-target measurement is a no-win/regression. Missing fallback facts and
metadata-only fallback origins fail closed in focused tests.

No new runtime or performance claim is made in this gate; the accepted Gate 4
same-target measurement remains a no-win/regression policy input, so the Gate 1
selected-dispatch decision keeps the conservative fallback path.

## Current Slice: Gate 2 Route/Export/Artifact Validation

- [x] Inspect the low-precision product-reduction-dequant route planning,
  Common EmitC metadata materialization, and RVV target artifact validation
  path to determine whether the selected dispatch case/fallback boundary is
  consumed as part of target acceptance.
- [x] Ensure the selected/pre-realized packed-i4 low-precision
  production-kernel body reaches route/export/artifact validation with
  selected case, conservative fallback, primitive signedness/type, Gearbox
  resource/remediation, artifact-evidence, same-target measurement,
  runtime AVL/VL, ABI/header, and provider mirror facts preserved.
- [x] Fail closed when selected case or fallback mirrors are missing, stale, or
  disconnected from the provider-owned low-precision resource/performance
  policy handoff.
- [x] Keep conservative fallback available for the current no-win/regression
  measurement input; Gate 2 must not claim a new performance-preferred result.
- [x] Add focused positive route/export/artifact evidence for the
  selected/pre-realized packed-i4 body.
- [x] Add focused negative target artifact evidence for stale or missing
  selected dispatch case/fallback facts.
- [x] Run affected RVV plugin, target artifact, `tcrv-opt`, and
  `tcrv-translate` checks.
- [x] Keep this macro task active after Gate 2 unless Gates 3-4 are also
  genuinely complete.

## Gate 2 Acceptance Criteria

- [x] Production source diff, or exact no-source-change proof, demonstrates
  that selected/pre-realized low-precision production-kernel bodies reach
  provider route validation, Common EmitC export mirrors, and target artifact
  validation with selected dispatch case/fallback facts intact.
- [x] Target artifact validation consumes selected dispatch case/fallback
  mirrors together with provider-owned primitive/resource/measurement facts
  for the bounded packed-i4 product-reduction-dequant path.
- [x] Missing or stale selected case, fallback, primitive/resource handoff,
  artifact tie-back, measurement identity, runtime AVL/VL, ABI/header, or
  provider validation facts fail closed with targeted diagnostics.
- [x] Conservative fallback remains structurally available when the selected
  RVV route is correctness-supported but performance preference is denied by
  the accepted no-win/regression measurement.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, status field, or Common EmitC inference
  becomes dispatch, route, dtype/config, or evidence authority.
- [x] Focused tests exercise accepted structural route/export/artifact facts
  and stale/missing selected dispatch fact rejection for the changed seam.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin
  code or tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code or tests are touched.
- [x] Relevant `tcrv-opt` and `tcrv-translate` checks pass for the selected
  pre-realized packed-i4 artifact fixture.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 2 slice.
- [x] The macro task remains active with Gates 3-4 unchecked and a precise
  continuation point.

## Gate 2 Completion Notes

Gate 2 is complete. The widening product-reduction/dequantization route
validation contract now carries the provider-owned
`RVVLowPrecisionSelectedDispatchPolicyBoundary` from route planning into target
artifact validation. The target artifact consumer validates
`tcrv_rvv.selected_dispatch_case_mirror` and
`tcrv_rvv.selected_dispatch_fallback_mirror` against that provider boundary
alongside the existing primitive signedness/type, Gearbox resource/remediation,
artifact-evidence, same-target measurement, runtime AVL/VL, ABI/header, and
provider support mirrors.

Focused target artifact tests cover the accepted selected-dispatch packed-i4
product-reduction/dequantization candidate, stale case mirror rejection,
missing fallback mirror rejection, and provider boundary missing fallback facts.
The selected pre-realized fixture also has route/export evidence showing the
case/fallback mirrors in the target header artifact and stale mirror rejection
at `tcrv-translate --tcrv-export-target-header-artifact`.

No new runtime, correctness, or performance claim is made in this gate. The
accepted same-target measurement remains a no-win/regression input, so the
selected-dispatch policy keeps the conservative fallback path and denies a new
performance-preferred result.

## Out Of Scope

- New q8/q4 route ids or llama.cpp-named route authority.
- Another standalone generated-bundle ABI evidence closeout.
- One-op fixture work that does not affect the production selected-dispatch
  policy boundary.
- Broad smoke matrices unrelated to the changed Gate 1 seam.
- Global autotuning databases or readiness state machines.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority or dtype/LMUL clone batches.
- Common EmitC invention of RVV semantics.
- Unrelated MAcc, mask, broadcast, or reduction expansion outside this selected
  low-precision production-kernel dispatch campaign.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous completed campaign:
  `.trellis/tasks/archive/2026-06/06-10-rvv-low-precision-contraction-primitive-surface-campaign/prd.md`.
- Likely production entry points from the Direction Brief:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  RVV Gearbox/selected-body realization and route-provider files under
  `lib/Plugin/RVV`, target artifact validation under `lib/Target/RVV`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.
- The current slice should prefer focused dry-run/unit evidence unless the
  policy change itself makes a new runtime/correctness/performance claim.

## Continuation Point

Current continuation is Gate 3: update only the bounded same-target measurement
policy input and demonstrate preferred-route or conservative-fallback behavior
from provider-owned primitive/resource/artifact/measurement facts, without
turning measurement evidence into route authority and without claiming Gate 4
runtime/correctness/performance completion.
