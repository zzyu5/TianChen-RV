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

- [ ] Gate 1: inventory and harden the policy-to-dispatch boundary. Production
  owners that consume route support, correctness fallback, performance
  admission, source-backed measurement, schedule/resource closure, and selected
  fallback mirrors must fail closed on stale or metadata-derived handoff facts.
- [ ] Gate 2: one concrete RVV low-precision or contraction pressure path carries
  policy output into selected-dispatch/fallback behavior without artifact
  metadata, route id, q8/q4 label, or test-name authority.
- [ ] Gate 3: same-target measured-win or no-win records can update
  admission/reopen facts through provider-owned contracts and target mirrors,
  with fail-closed stale-provenance diagnostics.
- [ ] Gate 4: focused executable or measurement evidence demonstrates the
  policy-to-dispatch workflow on `ssh rvv` only when the production compiler
  path claims runtime, correctness, or performance behavior.

## Current Round Slice

Implement Gate 1. The bounded slice is:

- Inventory the current production boundary among
  `RVVLowPrecisionPerformancePolicy`, selected-dispatch envelope collection,
  route/provider validation, and target artifact validation.
- Harden one real handoff where selected-dispatch mirror strings could otherwise
  be treated as valid merely because they match target artifact metadata.
- Keep the packed-i4 Gate 4 no-win closure as correctness fallback; do not
  reopen performance preference, add q8/q4-named routes, or refresh evidence as
  the main work.

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

## Requirements

- `RVVLowPrecisionPerformancePolicy` must reject selected-dispatch case and
  fallback mirrors that do not mirror the structured dispatch boundary fields.
- The check must be inside the production policy/dispatch boundary, not in
  Python scripts, reports, artifact names, or route IDs.
- Existing no-win packed-i4 policy must continue to select
  `correctness-fallback` and reject performance-preferred markers.
- Target artifact validation must reject a candidate whose metadata mirrors a
  stale provider selected-dispatch mirror, not only candidates whose metadata
  differs from the provider mirror.

## Acceptance Criteria

- [ ] Production C++ rejects stale selected-dispatch case mirror handoff facts.
- [ ] Production C++ rejects stale selected-dispatch fallback mirror handoff
  facts.
- [ ] RVV plugin tests cover positive no-win selected-dispatch policy and the
  new stale provider mirror failures.
- [ ] Target artifact tests cover stale provider selected-dispatch mirror
  rejection even when candidate metadata matches the stale provider mirror.
- [ ] Focused build/tests for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` pass.
- [ ] Focused lit for selected-dispatch policy/target tests passes.
- [ ] `git diff --check`, `git diff --cached --check`, and clean final status
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

Gate 1 current slice is complete.

Completed in this slice:

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

## Continuation Point

Keep this macro task active. Gate 1 is complete for the selected-dispatch
provider-mirror hardening slice. Gates 2-4 remain open.

Next continuation point: implement Gate 2 by carrying one concrete
low-precision/contraction pressure path's policy output into selected
dispatch/fallback behavior through provider-owned facts, without artifact
metadata, route ID, q8/q4 label, or test-name authority.
