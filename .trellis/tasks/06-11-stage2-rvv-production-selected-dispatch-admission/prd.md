# Stage2 RVV production-kernel selected-dispatch performance-admission campaign

## Goal

Make the RVV low-precision production dispatch workflow explicit and
fail-closed from provider-owned typed/body/resource facts through
`RVVLowPrecisionPerformancePolicy`, selected `tcrv.exec.dispatch` case/fallback
facts, and target artifact mirrors. The macro campaign keeps route support and
correctness fallback separate from performance-preferred dispatch, and admits a
performance-preferred path only when provider-owned facts and same-target
measurement evidence agree.

## Campaign Gates

- [x] Gate 1: inventory and harden the policy-to-dispatch boundary. Production
  owners that consume route support, correctness fallback, performance
  admission, source-backed measurement, schedule/resource closure, and selected
  fallback mirrors must fail closed on stale or metadata-derived handoff facts.
- [x] Gate 2: one concrete RVV low-precision or contraction pressure path carries
  policy output into selected-dispatch/fallback behavior without artifact
  metadata, route id, q8/q4 label, or test-name authority.
- [ ] Gate 3: same-target measured-win or no-win records can update
  admission/reopen facts through provider-owned contracts and target mirrors,
  with fail-closed stale-provenance diagnostics.
- [ ] Gate 4: focused executable or measurement evidence demonstrates the
  policy-to-dispatch workflow on `ssh rvv` only when the production compiler
  path claims runtime, correctness, or performance behavior.

## Current Round Slice

Implement Gate 3 for the current packed-i4 selected-dispatch pressure path. The
bounded slice is:

- Add an explicit same-target measurement-record admission path for
  selected-dispatch policy-output facts, so accepted no-win or measured-win
  records can update the `RVVLowPrecisionPerformancePolicy` decision and
  selected-dispatch mirrors through provider-owned C++ contracts.
- Keep the current no-win path as `correctness-fallback` /
  `not-performance-preferred` unless the selected provider resource facts,
  admission closure/reopen facts, maturity fields, same-target record, and real
  `tcrv.exec.dispatch` case/fallback facts all agree on a measured-win update.
- Reject stale target, capability/resource, workload/runtime ABI, selected-body,
  policy-output, mirror-handoff, or measurement-root facts before target
  artifact/header mirrors can claim dispatch preference.

## What I Already Know

- There is no active Trellis task at session start; this task owns the macro
  campaign across rounds.
- The previous archived low-precision contraction primitive campaign closed
  packed-i4 Gate 4 under a no-further-repair no-win admission boundary.
- `RVVEmitCRoutePlanning` collects selected-dispatch case/fallback facts from
  the real `tcrv.exec.dispatch` envelope into
  `RVVLowPrecisionSelectedDispatchPolicyBoundary`.
- `RVVLowPrecisionPerformancePolicy` verifies selected-dispatch shape and
  no-win/performance-preferred decisions before production pressure profile
  consumption.
- Target artifact validation compares selected-dispatch case/fallback mirrors
  against provider-owned route descriptions.

## Discovered Gate 1 Gap

The selected-dispatch boundary currently requires non-empty case/fallback mirror
strings and target artifact validation compares candidate metadata against those
provider mirrors. However, a provider boundary mirror can be stale while still
matching artifact metadata unless the policy boundary verifies that the mirror
spells the same selected variant, role, origin, policy, runtime guard, fallback
variant, and fallback role as the structured dispatch/fallback facts. That
would let a stale provider mirror masquerade as selected-dispatch authority.

## Discovered Gate 2 Gap

Gate 1 made selected-dispatch case/fallback mirrors tie back to structured
`tcrv.exec.dispatch` case/fallback facts. The remaining Gate 2 gap is that the
packed-i4 no-win policy decision itself is mostly consumed as a verifier result:
route support, correctness fallback, dispatch policy path, dispatch preference,
and denial reason are not yet carried as provider-owned selected-dispatch
policy-output facts into the route description and target artifact mirror
surface. Target validation rejects metadata-only policy-output claims, but there
is no positive provider-owned mirror path for the same facts.

## Discovered Gate 3 Gap

Gate 2 populated selected-dispatch policy-output facts from the current
packed-i4 provider selection by constructing the accepted no-win record inside
the production helper. That keeps the existing no-win path fail-closed, but the
selected-dispatch mirror seam does not yet expose a record-aware provider API
that lets an explicit source-backed same-target no-win or measured-win record
drive the policy-output update. Without that seam, tests can evaluate records
through the policy API, but the target/header mirrors still exercise the
selection-only helper rather than the measured-record admission handoff.

## Requirements

- Gate 1 behavior remains required: `RVVLowPrecisionPerformancePolicy` must
  reject selected-dispatch case and fallback mirrors that do not mirror the
  structured dispatch boundary fields.
- Gate 2 policy-output facts must be produced inside the RVV plugin/provider
  path from the accepted packed-i4 policy decision, not in Python scripts,
  reports, artifact names, route IDs, fixture names, or target-local metadata.
- Gate 3 measured-record admission must prefer explicit
  `RVVLowPrecisionSameTargetMeasurementRecord` facts when they are supplied,
  and must update selected-dispatch policy-output mirrors only after the record
  ties back to provider resource, admission, maturity, schedule, runtime,
  selected-body/source, target capability, and dispatch boundary facts.
- Existing no-win packed-i4 policy must continue to select
  `correctness-fallback`, deny `performance-preferred`, and reject
  performance-preferred markers unless a provider-owned measured-win contract
  updates all required admission facts.
- Target artifact validation must accept policy-output mirrors only when they
  exactly match provider-owned selected-dispatch policy-output facts derived
  from accepted record/provider/dispatch facts, and must reject stale or
  metadata-only policy-output mirrors.

## Acceptance Criteria

- [x] Production C++ rejects stale selected-dispatch case mirror handoff facts.
- [x] Production C++ rejects stale selected-dispatch fallback mirror handoff
  facts.
- [x] Production C++ carries accepted packed-i4 no-win policy output into
  provider-owned selected-dispatch policy-output facts.
- [x] RVV plugin tests cover positive no-win selected-dispatch policy-output
  facts and stale policy-output failures.
- [x] Target artifact tests cover accepted policy-output mirrors and stale or
  metadata-only policy-output mirror rejection.
- [x] Production C++ exposes a same-target measurement-record selected-dispatch
  policy-output update path.
- [x] RVV plugin tests cover accepted no-win and measured-win records updating
  selected-dispatch policy-output facts, plus stale record rejection.
- [x] Target artifact tests cover record-derived selected-dispatch
  policy-output mirrors and stale policy-output mirror rejection before artifact
  acceptance.
- [x] Focused build/tests for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` pass.
- [x] Focused lit for selected-dispatch policy/target tests passes.
- [x] `git diff --check`, `git diff --cached --check`, and clean final status
  are recorded.

## Out Of Scope

- No new q8/q4 route IDs or llama.cpp route authority.
- No performance-preferred dispatch claim without source-backed same-target
  measured-win evidence and provider-owned fact updates.
- No generated-bundle evidence refresh as the main achievement.
- No high-level Linalg/Vector/StableHLO frontend work.
- No Common EmitC invention of RVV semantics.
- No broad smoke matrix beyond focused verification unless a focused failure
  requires it.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and shared Trellis guides.
- Primary production files:
  `include/TianChenRV/Plugin/RVV/RVVLowPrecisionPerformancePolicy.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Primary tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVV/selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative.mlir`.

## Current Status

Gate 1 current slice is complete. Gate 2 current slice is complete. Gate 3
current slice is complete. Gate 4 remains open, so the macro task stays active.

Completed in Gate 1:

- Added production policy validation that selected-dispatch case/fallback mirror
  strings must mirror the structured `tcrv.exec.dispatch` case/fallback facts:
  case variant, role, runtime guard flag/value, origin, policy, fallback
  variant, fallback path role, fallback role, fallback origin, and fallback
  policy.
- Added target artifact validation so a stale provider selected-dispatch mirror
  is rejected even when candidate artifact metadata mirrors that stale provider
  value exactly.
- Added RVV plugin regression coverage for stale selected-dispatch case and
  fallback provider mirrors.
- Added target artifact regression coverage for stale provider case mirror
  rejection after candidate metadata matches the stale mirror.
- Updated the RVV plugin spec with the new selected-dispatch mirror tie-back
  contract and target artifact test requirement.

Self-repair performed:

- The first policy implementation reported mirror mismatch before the existing
  no-win performance-preferred diagnostic. The verifier was reordered so the
  no-win performance-preferred marker rejection remains stable, and mirror
  tie-back runs after the accepted no-win/performance path is established.
- The first mirror variant check used prefix matching, so a sibling variant
  ending in `_sibling` could still contain the expected variant token. The check
  now matches the `@variant;` field boundary.
- The first target artifact negative test still passed because target validation
  only compared candidate metadata with provider mirrors. Target validation now
  also checks provider mirror tie-back against structured boundary fields.

Checks:

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed after self-repair.
- `build/bin/tianchenrv-target-artifact-export-test` passed after self-repair.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  selected-dispatch` from `build/test` passed 1/1.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-11-stage2-rvv-production-selected-dispatch-admission`
  passed.
- `git diff --check` passed.
- Bounded old-authority scan over added diff lines found no new legacy RVV
  authority strings. A production-file scan still sees existing untouched
  `__riscv_*_i32m1` target validation strings in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.

No `ssh rvv` run was required: this slice changes policy/target validation and
diagnostics only, not generated RVV runtime behavior, correctness behavior, or
performance claims.

## Current Gate 2 Slice

In this round, the packed-i4 no-win policy output was carried into
selected-dispatch facts for the product-reduction dequant/dequant-clamp
pressure path:

- provider route analysis derives selected-dispatch policy-output facts from
  the packed-i4 resource selection, the source-backed no-win measurement record,
  and collected `tcrv.exec.dispatch` case/fallback boundary;
- target artifact metadata mirrors those facts with explicit mirror labels;
- target validation rejects stale or metadata-only policy-output mirrors before
  artifact acceptance.
- target support bundle mapping exports the same policy-output mirrors into
  generated header comments.

Completed in Gate 2:

- Added `hasSelectedDispatchPolicyOutput` and explicit policy-output fields to
  `RVVLowPrecisionSelectedDispatchPolicyBoundary`.
- Added provider helper
  `populateRVVLowPrecisionSelectedDispatchPolicyOutput` so the accepted
  packed-i4 no-win `RVVLowPrecisionPerformancePolicyDecision` becomes
  provider-owned selected-dispatch boundary state before route metadata/export.
- Added target artifact validation for policy-output mirrors and metadata-only
  rejection when provider-owned policy-output facts are absent.
- Added plugin, target, and lit coverage for positive no-win policy-output
  propagation and stale/metadata-only policy-output rejection.
- Updated the RVV plugin spec with the provider-owned selected-dispatch
  policy-output contract.

Checks for Gate 2:

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|selected-dispatch'`
  from `build/test` passed 2/2.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-11-stage2-rvv-production-selected-dispatch-admission`
  passed.
- `git diff --check` passed.
- Bounded old-authority scan over added diff lines found no new legacy RVV
  authority strings.

No `ssh rvv` run was required: this slice changes provider policy-output
facts, target/header mirrors, validation, and diagnostics only. It does not
change generated runtime behavior or make a new correctness/performance claim.

## Current Gate 3 Slice

In this round, explicit same-target measurement records can drive
selected-dispatch policy-output population for the packed-i4 pressure path:

- added a record-aware
  `populateRVVLowPrecisionSelectedDispatchPolicyOutput(selection, record,
  boundary, context)` overload;
- kept the old selection-only helper as the current accepted no-win default,
  now delegating through the record-aware path;
- proved accepted no-win records populate conservative
  `correctness-fallback` selected-dispatch policy-output facts;
- proved measured-win records populate `performance-preferred`
  selected-dispatch policy-output facts only after provider maturity,
  admission, remediation, dispatch, and measurement tie-backs agree;
- proved stale target records fail before policy-output mirrors can be
  populated or target artifact mirrors accepted;
- updated the RVV plugin spec with the record-aware policy-output contract.

Checks for Gate 3:

- `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|selected-dispatch'`
  from `build/test` passed 2/2.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-11-stage2-rvv-production-selected-dispatch-admission`
  passed.
- `git diff --check` and `git diff --cached --check` passed.
- Bounded old-authority scan over added diff lines found no new legacy RVV
  authority strings.

No `ssh rvv` run was required: this slice changes policy/admission API,
selected-dispatch policy-output mirrors, target validation coverage, and specs.
It does not change generated RVV runtime behavior or make fresh correctness or
performance claims.

## Continuation Point

Keep this macro task active. Gate 4 is the next campaign gate: focused
executable or measurement evidence must demonstrate the selected-dispatch
performance-admission behavior end to end only when the production path claims
runtime, correctness, or performance behavior.
