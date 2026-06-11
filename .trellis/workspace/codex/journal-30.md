> Continuation from `journal-29.md` (archived at more than 2000 lines)

## Session 606: Provider primitive route-payload canonicalization

**Date**: 2026-06-12
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the provider primitive
route-payload canonicalization slice. This round introduced a provider-owned
primitive payload gate for product-reduction routes so signed/unsigned
product-reduction and packed-i4 dequant representatives validate the full
`RVVLowPrecisionPrimitiveRoutePayload` against provider widening-reduction
primitive facts before route construction or target artifact mirror validation.

### Main Changes

- Repaired the active macro PRD for provider primitive route-payload
  canonicalization, with field classification for compiler authority,
  mirror/test facts, and policy/evidence facts.
- Added
  `verifyRVVLowPrecisionPrimitiveRoutePayloadFromWideningReductionFacts` as the
  canonical provider helper for source dtype/signedness, load/extension,
  product/accumulator/final-result dtype, SEW/LMUL, policy, runtime AVL/VL,
  product-reduction relation, intrinsics, seed, layout, and store-VL payload
  validation.
- Wired the helper into provider plan validation and route-description
  validation before route construction.
- Changed target provider-facts validation to call the same provider payload
  helper, then compare route-description mirrors against `payload.*` rather
  than reconstructing primitive authority from target-side fields.
- Added C++ stale payload negative coverage for product-reduction chain
  relation, product-reduction runtime AVL source, and packed-i4 dequant
  reduction intrinsic.
- Updated the RVV plugin code-spec with the helper signature, payload gate
  contract, error behavior, and target provider-facts test requirements.
- Updated active task metadata/context while leaving the macro task in progress.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|pre-realized-selected-body-artifact-widening-product-reduce-add-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'` from `build/test`
- [OK] Diff-only scan over touched production/test files found no added
  admission/remediation/measurement/same-target/no-win/dispatch wording in the
  primitive payload compiler-fact path.
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] `rtk git diff --check`

### Self-Repair

- The first implementation added the shared helper but left target
  provider-facts follow-up comparisons using primitive fact fields as expected
  values. The final version keeps primitive facts inside the helper and makes
  target provider-facts validation consume the provider payload for subsequent
  route-description mirror checks.
- The first broad policy/evidence scan included existing spec and long-lived
  packed-i4 policy tests, so it was too noisy for this slice. The final scan was
  diff-only over touched production/test files.

### Status

[OPEN MACRO TASK] Provider primitive route-payload canonicalization is complete
for the signed/unsigned product-reduction and packed-i4 dequant representative
payload boundary. The macro task remains active for adjacent low-precision
primitive-surface cleanup or future measurement-disposition work only with fresh
source-backed same-target RVV evidence.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 596: Stage2 RVV support-bundle primitive payload mirror boundary cleanup

**Date**: 2026-06-12
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the bounded
support-bundle/export primitive mirror-boundary cleanup slice. The provider
payload canonicalization from the previous slice remains the compiler authority;
this round made the target support-bundle/header surface visibly mirror-only for
low-precision primitive fields.

### Boundary Classification

- Compiler authority remains selected typed `tcrv_rvv` body/config/runtime
  facts, provider primitive facts, and `RVVLowPrecisionPrimitiveRoutePayload`.
- Mirror/transport facts are emission metadata, target candidate metadata,
  support-bundle/header comments, fixture `PLAN`/`HEADER` checks, and C++ mirror
  mutation records.
- Policy/evidence facts such as admission, remediation, measurement, no-win,
  performance, and dispatch remain outside primitive support-bundle authority.

### Main Changes

- Replaced the inline support-bundle list of
  `low_precision_primitive.*` header evidence with a dedicated provider payload
  mirror helper.
- Kept candidate metadata keys as `tcrv_rvv.low_precision_primitive.*` so target
  validation still compares mirrors against the rebuilt provider payload.
- Changed generated header evidence labels to
  `low_precision_primitive.payload_mirror.*`, making the header/support-bundle
  surface explicitly mirror-only.
- Added C++ negative coverage proving a missing primitive export mirror
  (`tcrv_rvv.low_precision_primitive.runtime_control_plan`) fails at target
  mirror validation rather than being inferred from adjacent metadata.
- Updated signed/unsigned product-reduction, pre-realized product-reduction,
  and standalone widening-product header checks for the new payload-mirror
  labels.
- Updated the RVV plugin spec and active macro PRD/task metadata with the
  support-bundle/export mirror-boundary contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-widening-product\.mlir|explicit-selected-body-artifact-widening-product-unsigned-u8|pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'` from `build/test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] `git diff --check`

### Self-Repair

- The first lit invocation from the repo root did not discover the build-backed
  test suite because the generated `lit.site.cfg.py` uses relative paths. The
  focused lit command was rerun from `build/test` and passed 8 selected tests.

### Status

[OPEN MACRO TASK] The support-bundle/export primitive mirror-boundary cleanup is
complete for this slice. The macro task remains active for adjacent
low-precision primitive-surface cleanup and for future measurement-disposition
work only if fresh source-backed same-target RVV evidence is introduced.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 605: Selected-body realization admission/evidence boundary cleanup

**Date**: 2026-06-12
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the selected-body realization
admission/evidence boundary cleanup slice. This round made the provider
realization handoff explicit: stable compiler facts remain in
typed/resource/realization, Gearbox handoff, stable schedule, resource-cost,
provider payload, and target resource validation surfaces, while packed-i4
admission, remediation, performance, same-target measurement, no-win, and
dispatch facts are read only through named measurement-disposition
policy/evidence helpers.

### Main Changes

- Repaired the active macro PRD with a source-backed classification of stable
  compiler authority versus policy/evidence fields for the realization handoff.
- Split provider-side route-family validation so
  `requireRVVLowPrecisionResourceRealizationCompilerFacts` checks only stable
  realization/resource/schedule facts.
- Added a named policy/evidence string reader for packed-i4
  measurement-disposition attrs, keeping performance feedback, remediation,
  admission, maturity, same-target evidence, no-win, and dispatch fields out of
  resource, schedule, and route acceptance.
- Retained existing target artifact separation between packed-i4 resource
  provider facts and measurement-disposition evidence/admission mirrors.
- Added focused missing-policy-evidence lit coverage for the packed-i4
  dequantize fixture.
- Updated the RVV plugin spec and active task metadata to preserve the helper
  boundary as an executable convention.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'` from `build/test`
- [OK] Bounded scans confirmed the realization compiler-fact helper does not
  consume packed-i4 policy/evidence fields, and target validation keeps resource
  facts separate from measurement-disposition evidence mirrors.
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] `rtk git diff --check`

### Self-Repair

- A root-level lit attempt did not discover the configured suite. The focused
  lit command was rerun from `build/test`, where the generated lit site config
  is valid.
- The provider handoff diagnostic for remediation planning still used route
  acceptance wording. It now names measurement-disposition remediation planning
  and evidence mirror validation instead.
- The first final task validation rejected a check-log entry that named the
  task directory rather than a real file. The entry now points at `task.json`,
  and task validation passes.

### Status

[OPEN MACRO TASK] The selected-body realization admission/evidence boundary
cleanup slice is complete. The macro task remains active for adjacent
low-precision primitive-surface cleanup or future measurement-disposition work
with fresh source-backed same-target RVV evidence before any measured-win,
performance-preferred, runtime, correctness, or performance claim.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 602: Stage2 RVV outcome-language and measurement-disposition cleanup

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the bounded
outcome-language/measurement-disposition cleanup slice after the
provider/artifact carry-through payload work. The active PRD, task context, RVV
low-precision spec sections, and directly related low-precision production
helpers now use foundation, selected-body realization, provider/artifact
carry-through, and measurement-disposition language instead of numbered
milestone terms that could be mistaken for compiler authority.

### Main Changes

- Repaired the active macro PRD/task metadata and implement/check context around
  campaign outcomes, current cleanup acceptance, completed payload carry-through
  state, and the still-open measurement-disposition continuation boundary.
- Updated RVV low-precision spec wording so primitive payload, selected-body
  realization, realization-admission proof, resource-aware realized-body
  consumption, and packed-i4 policy sections keep compiler facts separate from
  measurement-disposition evidence.
- Renamed the public packed-i4 helper from
  `buildRVVPackedI4Gate4SameTargetMeasurementRecord` to
  `buildRVVPackedI4MeasurementDispositionSameTargetRecord`.
- Renamed measurement-disposition constants/helpers in
  `RVVLowPrecisionPerformancePolicy.cpp`, updated policy diagnostics, and kept
  the existing no-win measurement record values unchanged.
- Renamed target artifact evidence/admission mirror helpers and diagnostics to
  measurement-disposition wording while preserving exact provider/candidate
  mirror validation.
- Updated C++ test call sites/context strings and packed-i4 fixture comments.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'` from `build/test`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] `rtk git diff --check`
- [OK] `rtk git diff --cached --check`
- [OK] bounded uppercase scan over active task, RVV low-precision spec, touched
  production files, directly affected C++ tests, and focused lit fixtures found
  no `Gate`/`Gate1`/`Gate2`/`Gate3`/`Gate4` matches.

### Self-Repair

- The first production rename left long C++ diagnostic lines; these were wrapped
  manually before the build/check pass.
- The initial active-task scan still matched self-referential cleanup language in
  the PRD/context files. The PRD now calls those terms numbered-milestone
  terminology so the active task no longer reintroduces the searched tokens.
- Lowercase `gate4-...` strings remain only as historical same-target evidence
  paths or evidence IDs; they are measurement-disposition evidence identifiers,
  not route, schedule, dtype/config, artifact-name, or selected-body authority.

### Status

[OPEN MACRO TASK] The outcome-language and measurement-disposition boundary
cleanup milestone is complete. Foundation, selected-body realization, and the
plain signed/unsigned provider/artifact carry-through outcomes remain recorded
as complete for this representative slice. The macro task stays active for
adjacent low-precision primitive-surface gaps or a future
measurement-disposition slice backed by fresh same-target RVV evidence.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 603: Stage2 RVV packed-i4 measurement-disposition policy boundary cleanup

**Date**: 2026-06-12
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the packed-i4
measurement-disposition policy-boundary cleanup slice. This round separated
stable packed-i4 compiler/resource facts from realization-admission,
remediation, admission, maturity, dispatch-preference, and same-target evidence
facts in Gearbox diagnostics, provider planning, target artifact validation,
focused tests, and the RVV plugin spec.

### Main Changes

- Kept packed-i4 load/unpack, realization, schedule, resource-cost, runtime
  ABI, and target facts in resource validation helpers.
- Moved remediation planning and realization-admission checks into explicitly
  named measurement-disposition policy/evidence helpers.
- Updated target artifact validation so resource provider mirrors no longer
  accept or validate remediation and realization-admission mirrors as compiler
  facts.
- Split the `tcrv_rvv.gearbox_cross_region_handoff` verifier wording so
  resource-cost, resource schedule, measurement-disposition remediation, and
  measurement-disposition admission fields fail under their own labels.
- Updated stale evidence/admission diagnostics in C++ and FileCheck coverage.
- Recorded the policy/evidence-only ownership rule in the RVV plugin spec and
  active macro PRD.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|rvv-generated-bundle-same-target-measure-gate4-dry-run'`
- [OK] Bounded scans over touched production, spec, fixtures, script, and gate4
  evidence surfaces for Gate/admission/result/support/no-win/remediation wording
  that could imply compiler authority.
- [OK] `git diff --check`

### Self-Repair

- The first root-level lit invocation did not discover the built test suite, so
  the focused lit command was rerun from `build/test`.
- A stale schedule-decision assertion was initially moved under the evidence
  label too broadly; it was restored to fail first as a stable packed-i4
  resource schedule fact, while admission/remediation assertions now fail under
  measurement-disposition labels.
- A bounded scan exposed that the cross-region handoff verifier still labeled
  remediation and admission fields as resource facts. The verifier now splits
  resource-cost, resource schedule, remediation planning, and admission
  diagnostics, and the affected packed-i4 lit fixture passes with the new
  wording.

### Status

[OPEN MACRO TASK] The packed-i4 measurement-disposition policy-boundary cleanup
slice is complete. The macro task remains active for adjacent low-precision
primitive-surface cleanup or future measurement-disposition work with fresh
source-backed same-target RVV evidence before any measured-win or
performance-preferred admission claim.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 604: Gearbox low-precision resource-schedule canonicalization

**Date**: 2026-06-12
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gearbox low-precision
resource-schedule canonicalization slice. This round concentrated the accepted
packed-i4 stable schedule/resource-cost facts into a single RVV plugin-local
helper and made Gearbox candidate selection, provider route validation, handoff
schedule validation, and target artifact resource validation consume that same
compiler-owned contract.

### Main Changes

- Added `RVVLowPrecisionPackedI4StableResourceScheduleFacts` for the accepted
  packed-i4 schedule-decision, unroll/accumulator/region/live-vector budget,
  and resource-cost facts.
- Removed remediation plan, performance admission, beyond-local admission,
  dispatch preference, no-win classification, same-target evidence IDs, and
  campaign outcome strings from stable schedule acceptance.
- Reused the canonical stable schedule facts in Gearbox handoff schedule/cost
  verification, provider route planning validation, selected-body handoff
  checks, and target artifact packed-i4 resource validation.
- Split provider handoff diagnostics so performance/beyond-local admission
  fields fail as measurement-disposition admission facts, resource-cost fields
  fail as resource-cost facts, and `schedule_decision*` fields fail as stable
  resource schedule facts.
- Extended the RVV plugin C++ test so stale measurement-disposition
  admission/remediation fields no longer affect stable schedule acceptance.
- Updated the active PRD/task metadata and RVV plugin spec for the stable
  schedule/resource-cost ownership rule.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|rvv-generated-bundle-same-target-measure-gate4-dry-run'` from `build/test`
- [OK] Bounded scans confirmed the stable packed-i4 schedule helper no longer
  references remediation/admission fields and provider handoff diagnostics no
  longer route admission through resource-cost or schedule through remediation
  labels.
- [OK] `rtk git diff --check`

### Self-Repair

- The first implementation still left `schedule_decision*` checks in a
  provider-side remediation handoff helper and performance/beyond-local
  admission fields in a resource-cost helper. These were split into stable
  resource schedule, resource-cost, and measurement-disposition admission
  helpers before the final build/check pass.
- The first canonicalization pass updated provider and target consumers but
  left the Gearbox schedule owner using local schedule/resource-cost constants.
  `RVVGearboxSchedules.cpp` now consumes the same canonical stable facts.

### Status

[OPEN MACRO TASK] The Gearbox low-precision resource-schedule canonicalization
slice is complete. The macro task remains active for adjacent low-precision
primitive-surface cleanup or future measurement-disposition work with fresh
source-backed same-target RVV evidence before any measured-win or
performance-preferred admission claim.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 592: Packed-i4 dequant/dequant-clamp resource/evidence boundary cleanup

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed one packed-i4
dequantize/dequant-clamp carry-through cleanup slice. The route/resource path
now keeps stable compiler facts in typed/realized/provider/target validation
surfaces, while performance feedback, admission, remediation handoff, maturity,
and same-target measurement facts are validated through explicit
measurement-disposition evidence/policy helpers.

### Main Changes

- Split provider-side packed-i4 validation so resource selection and
  resource-cost checks are separate from measurement-disposition evidence and
  policy handoff checks.
- Split target artifact validation into packed-i4 resource provider facts and
  packed-i4 measurement-disposition evidence mirror validation.
- Tightened Gearbox diagnostics to say `resource-planning`, `resource-cost`, or
  `measurement-disposition evidence` instead of grouping admission/evidence
  facts as route-support prerequisites.
- Renamed the packed-i4 resource-cost blocker value to
  `packed-i4-loop-11-budget-5of32-resource-cost-boundary` and synchronized
  tracked dequant/dequant-clamp artifact/evidence mirrors plus the ABI evidence
  script expectation.
- Updated the RVV plugin spec to record that `resource_cost_blocker` is a
  stable resource-cost boundary diagnostic and must not encode no-win,
  admission, or campaign outcome conclusions.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|explicit-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|pre-realized-selected-body-artifact-widening-product-reduce-add|pre-realized-selected-body-artifact-widening-product-reduce-add-unsigned-u8|rvv-generated-bundle-same-target-measure-gate4-dry-run'` from `build/test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] bounded scans for old packed-i4 no-win resource-cost blocker, merged resource/evidence diagnostics, and resource_cost_blocker admission/no-win/campaign residue
- [OK] `git diff --check`

### Self-Repair

- The first C++ test run exposed stale test expectations for
  `remediation diagnosis`; assertions were updated to the new
  `measurement-disposition diagnosis` label.
- Target artifact validation initially failed because source-backed evidence
  JSON still carried the old no-win-shaped `resource_cost_blocker`; the tracked
  dequant/dequant-clamp evidence and generated-bundle mirrors were synchronized
  without changing measurement/admission conclusions.
- The first lit run failed because the ABI evidence script still expected the
  old blocker string; only that resource-cost constant was updated.

### Status

[OPEN MACRO TASK] The packed-i4 dequant/dequant-clamp carry-through boundary
cleanup slice is complete. The macro task remains active for adjacent
packed-i4 primitive-surface cleanup and any future measurement-disposition work
that has fresh source-backed same-target evidence.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 601: Stage2 RVV Gate 3 product-reduction carry-through boundary

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a Gate 3 carry-through cleanup for
the bounded signed i8 and unsigned u8 product-reduction representatives. The
provider now concentrates the low-precision primitive route facts into a single
route payload before route construction. Emission metadata and target artifact
validation consume that payload as mirrors only, instead of treating artifact
names, q8/q4 labels, result/admission wording, or Common EmitC defaults as route
authority.

### Main Changes

- Added `RVVLowPrecisionPrimitiveRoutePayload` to the selected-body contraction
  route description and widening-dot/reduce target validation contract.
- Populated and verified the payload from the provider route-family plan,
  including source/product/accumulator/result dtype, signedness, SEW/LMUL,
  policy, runtime control/AVL, widening-product relation, product-reduction
  relation, intrinsics, seed splat, layouts, and store VL.
- Made emission metadata serialize
  `tcrv_rvv.low_precision_primitive.*` from the provider route payload.
- Tightened target artifact validation so product-reduction primitive metadata
  mirrors are compared against the provider payload before candidate acceptance.
- Added focused signed and unsigned explicit artifact negative checks for
  missing primitive runtime-control and product-SEW mirrors.
- Updated the RVV plugin code-spec and macro PRD/task notes for Gate 3.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'` from `build/test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add'` from `build/test`

### Self-Repair

- Repaired the active PRD from the previous Gate 2 slice to the current Gate 3
  route/provider/artifact-export carry-through goal before implementation.
- Manually inspected and wrapped the new C++ snippets after adding the provider
  route-payload checks.
- A too-narrow final lit filter matched only 2 tests; the final verification used
  the product-reduction filename fragments above and passed the intended 4-test
  explicit/pre-realized signed/unsigned set.
- The build still reports pre-existing unused/switch warnings in target artifact
  validation and export test code; they do not come from this slice's new
  payload fields.

### Status

[OPEN MACRO TASK] Gate 3 product-reduction carry-through is complete for this
plain signed/unsigned representative slice. The macro task remains active. The
next continuation point is any remaining Gate 3 expansion that is truly needed
for adjacent low-precision product-reduction carry-through, otherwise Gate 4
fresh source-backed same-target evidence/admission only if later claimed.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 600: Stage2 RVV Gate 2 product-reduction signedness boundary

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and closed the Gate 2 plain
product-reduction selected-body signedness boundary. The signed i8
pre-realized path remains supported, and unsigned u8 now goes through the same
typed pre-realized op, RVV dialect verification, provider-side validation,
RVV plugin-local selected-body realization, route provider facts, and
target/header export path. The slice avoids a separate u8 route clone and does
not use q8/q4 labels, artifact names, target mirrors, or Common EmitC logic as
signedness authority.

### Main Changes

- Added explicit `source_signedness` to
  `tcrv_rvv.typed_widening_product_reduce_pre_realized_body`.
- Extended the RVV dialect verifier to accept only signedness-consistent signed
  i8/i16/i32 or unsigned u8/u16/u32 product-reduction configs, ABI C types,
  product relations, reduction-chain relations, SEW/LMUL facts, and policy.
- Made the RVV contraction selected-body realization owner derive unsigned
  source/product/result vector types, `unsigned_widening_product`,
  `unsigned_widening_reduce_add`, and unsigned provider primitive facts from
  the typed signedness boundary.
- Updated the provider-side pre-realized validator to validate signed or
  unsigned C ABI and provider-derived relations before route construction.
- Added focused unsigned u8 pre-realized lit coverage proving the pre-realized
  op is removed and explicit `setvl` / `with_vl` / ui8 load /
  unsigned widening product / unsigned widening standalone reduction / u32
  store structure is produced.
- Updated the RVV plugin spec, active PRD, and task notes with the completed
  Gate 2 signedness-boundary slice and Gate 3 continuation point.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'` from `build/test`

### Self-Repair

- The first focused lit attempt from the repository root with path `.` did not
  discover tests; the second attempt against `test` lacked the CMake-provided
  `tianchenrv_obj_root`. The working invocation is from `build/test`, where
  `lit.site.cfg.py` resolves the configured suite correctly.
- The first focused lit run from `build/test` failed only because the signed
  stale-source FileCheck diagnostic still expected the old signed-only wording.
  The check was updated to the new signed/unsigned verifier diagnostic and the
  same filtered lit run passed 4/4.
- The environment has no `clang-format` binary; the changed C++ snippets were
  manually inspected and line-wrapped.

### Status

[OPEN MACRO TASK] Gate 2 product-reduction selected-body signedness boundary is
complete for the bounded signed i8 and unsigned u8 representatives. The macro
task remains active. The next continuation point is Gate 3 route
provider/artifact/export carry-through beyond this plain product-reduction
realization foundation. Gate 4 remains later work and requires fresh
source-backed same-target evidence before any runtime/performance/admission
claim.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 597: Stage2 RVV Gate 4 admission boundary cleanup

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the first Gate 4 admission
boundary cleanup slice for packed-i4 low-precision resource/schedule and target
validation ownership. The cleanup separates stable compiler/resource facts from
generic selected-dispatch policy-output facts and campaign evidence/admission
facts without admitting a measured-win or changing fallback behavior.

### Main Changes

- Classified stable facts as candidate selection, planning contract,
  dtype/SEW/LMUL, packed-i4 load/unpack, primitive product/reduction,
  resource cost, schedule decision, realization/region/live-vector facts,
  target capability mirrors, legality, and rejection reason.
- Classified generic policy-output facts as selected-dispatch case/fallback
  mirrors plus `RVVLowPrecisionPerformancePolicyDecision` route, correctness,
  performance, win-claim, dispatch-path, and path-selection outputs.
- Classified campaign evidence/admission facts as performance
  feedback/baseline/speedup/action, remediation narrative/evidence,
  performance admission closure/reopen, beyond-local repair admission,
  performance maturity, and same-target measurement records.
- Renamed the packed-i4 schedule-decision predicate to a stable Gearbox
  schedule helper and removed Gate 4 admission/evidence fields from its
  acceptance predicate.
- Split target artifact validation into stable resource mirror validation and
  Gate 4 evidence/admission mirror validation. Stale Gate 4 mirrors still fail
  closed, but diagnostics now label them as evidence/admission ownership rather
  than stable resource authority.
- Added focused plugin coverage proving stale or missing Gate 4 admission
  fields do not invalidate an otherwise stable packed-i4 schedule decision.
- Updated the RVV plugin code-spec, macro PRD, task notes, and Trellis JSONL
  logs with the completed cleanup slice and remaining macro continuation.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `rtk git diff --check`
- [OK] bounded old-authority scan over added diff lines returned only
  spec/PRD negative boundary text for artifact-name authority.

### Self-Repair

- The first focused lit run failed because `tcrv-translate` had not been
  rebuilt after the target diagnostic-label split and still emitted the prior
  stable-resource diagnostic label. Rebuilding `tcrv-translate` and rerunning
  the same lit filter passed 3/3.

### Status

[OPEN MACRO TASK] Gate 4 admission-boundary cleanup is complete for the first
packed-i4 production slice. Accepted dequant and dequant-clamp records remain
no-win/regression, preserving route support and correctness execution while
denying performance-preferred selection and performance-win claims. The macro
task remains active for measured-win admission and final provider,
selected-dispatch, target mirror, and dispatch/fallback closure.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 599: Stage2 RVV Gate 2 signed product-reduction realization closure

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and closed the interrupted dirty Gate 2 signed
selected-body realization slice for the plain low-precision product-reduction
representative. The slice adds a typed pre-realized signed i8 product-reduction
body, validates it in the RVV dialect and provider boundary, and realizes it
through the RVV contraction selected-body owner before existing route/provider,
emission-plan, and target artifact validation consume the facts.

### Main Changes

- Added `tcrv_rvv.typed_widening_product_reduce_pre_realized_body` for the
  bounded signed i8 -> i16 product -> i32 reduction body.
- Added verifier checks for direct selected-variant nesting, runtime ABI roles
  and C types, signed source/product/accumulator/result SEW/LMUL, policy,
  memory form, product relation, and product-reduction chain relation.
- Added provider-side pre-realized validation and wired the RVV contraction
  selected-body realization owner to materialize `setvl`, `with_vl`, two
  loads, `widening_product`, `standalone_reduce`, and store before route
  construction.
- Added focused lit coverage for the pre-realized signed path, emission-plan
  mirrors, header export, stale source SEW rejection, and preservation of the
  existing explicit signed and unsigned product-reduction fixtures.
- Updated the RVV plugin spec and active macro PRD/task notes with the Gate 2
  signed realization contract and continuation boundary.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface`
- [OK] `git diff --check`
- [OK] bounded old-authority scan over added diff lines returned only negative
  q8/q4/source-front-door/descriptor boundary text.

### Self-Repair

- Fixed a formatting issue left in the dirty realization branch after the
  first successful build/lit pass.
- Corrected primitive fact mismatch diagnostics so provider-owned stale facts
  are reported as the found value against the selected typed body contract.
- Corrected the new RVV plugin spec signature example to use the actual
  verifier/provider attribute values for accumulator role/layout and result
  layout.

### Status

[OPEN MACRO TASK] Gate 2 is complete for the signed i8
product-reduction selected-body realization representative. The macro task
remains active. The exact continuation point is unsigned u8 selected-body
realization for the same product-reduction primitive surface, or a bounded
decision to keep unsigned u8 as explicit-body-only until the next gate. Gate 3
route/provider/artifact carry-through and Gate 4 measured same-target evidence
remain future work only when those claims are made.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 592: Stage2 RVV low-precision primitive Gate 1 mirror hardening

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Created the new macro campaign task and completed the bounded Gate 1 foundation
slice. This round hardened the existing typed low-precision product-reduction
primitive contract by carrying typed config and runtime control facts in the
`tcrv_rvv.low_precision_primitive.*` mirror namespace instead of leaving them
only as generic route-family metadata.

### Main Changes

- Added provider-emitted low-precision primitive mirrors for
  source/product/accumulator/result SEW/LMUL, tail policy, mask policy,
  runtime control plan, and runtime AVL source.
- Added support-bundle mappings so target header/export artifacts can carry
  the new primitive mirrors.
- Required target artifact validation to compare the new primitive mirrors
  against provider-owned widening-product and product-reduction contracts
  before artifact acceptance.
- Extended signed i8 and unsigned u8 product-reduction lit fixtures with
  positive mirror checks and stale runtime AVL source / stale tail policy
  fail-closed coverage.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter 'explicit-selected-body-artifact-widening-product-reduce-add'` from `build/test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] Added-diff old-authority scan for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, exact `__riscv_*_i32m1`, q8/q4,
  source-front-door, and artifact-name authority patterns returned no matches.

### Self-Repair

- The first focused lit run passed the signed fixture but failed the unsigned
  header check because FileCheck expected new primitive mirrors after
  `low_precision_resource.*` mirrors. The actual support-bundle order is
  primitive mirrors first, resource mirrors second, so the unsigned header
  checks were reordered without changing production code.

### Status

[OPEN MACRO TASK] Gate 1 foundation is complete for the current
product-reduction primitive mirror hardening slice. The task remains active for
Gate 2 selected-body realization from this primitive surface, Gate 3 broader
route/artifact carry-through, and later Gate 4 source-backed same-target
evidence/admission only if fresh measured evidence is available.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 598: Stage2 RVV Gate 4 final no-win admission consumer closure

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Completed the final Gate 4 admission/consumer closure slice for the current
packed-i4 low-precision Gearbox campaign. Current source-backed same-target
evidence remains no-win/regression, so this round did not admit
performance-preferred dispatch. The closure proves the conservative path at the
target consumer boundary and rejects synthetic measured-win promotion without a
fresh source-backed measured-win target admission chain.

### Main Changes

- Added target-artifact consumer coverage showing parsed dequant evidence-root
  records populate selected-dispatch policy-output mirrors through the explicit
  record overload before target validation consumes no-win mirrors.
- Added target validation coverage for synthetic measured-win promotion:
  provider maturity/admission fields, selected-dispatch policy-output mirrors,
  and target metadata rewritten together still fail closed because the current
  target admission boundary remains source-backed no-win.
- Added stale no-win provider policy-output cases for
  `dispatch_policy_path = performance-preferred` and
  `performance_win_claim_allowed = true` where candidate metadata mirrors the
  stale provider value.
- Recorded the durable spec rule that target validation using the default
  packed-i4 helper is a current no-win/fallback consumer, not a measured-win
  admission seam.
- Updated PRD/task notes to mark Gate 4 and the current macro campaign scope
  complete.

### Testing

- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `rtk python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `rtk git diff --check`
- [OK] bounded old-authority scan returned only archived PRD
  negative/non-authority boundary text for q8/q4 labels, artifact names, route
  IDs, helper names, and helper-op authority; production/test/spec added lines
  returned no matches.

### Self-Repair

- The first target test run failed because a measured-win fixture reused a
  boundary already populated with no-win policy-output facts. The test now
  clears provider-owned policy-output before record-derived population.
- The next target test run showed production target validation still consumes
  the default current no-win record at the target boundary. The test was
  corrected from a synthetic measured-win target-positive to an explicit
  fail-closed measured-win-promotion rejection, matching the no-fresh-win PRD.

### Status

[READY TO FINISH] Gate 1 resource/primitive fact spine, Gate 2 packed-i4
primitive/resource surface, Gate 3 resource-aware selected-body realization,
and Gate 4 same-target no-win admission/consumer closure are complete for the
current macro campaign scope. Future measured-win work requires new
source-backed same-target evidence and synchronized provider-owned admission
facts before dispatch preference may change.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 596: Stage2 RVV Gearbox Gate 4 dispatch/fallback consumption audit

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a bounded Gate 4 selected-dispatch
/ fallback consumption consistency slice. The production gap was that target
artifact validation could prove candidate metadata exactly matched provider
policy-output mirrors, but still needed to reject a stale provider
policy-output value that the candidate metadata had mirrored exactly. This
round anchored target-artifact consumption back to provider-owned resource
selection and same-target no-win/regression policy facts.

### Main Changes

- Extended the active macro PRD from policy-output mirror admission into the
  current dispatch/fallback consumption slice.
- Required target artifact validation to consume selected-dispatch
  policy-output mirrors only with complete provider-owned selected-dispatch
  case and fallback facts.
- Added packed-i4 no-win/regression consumption checks that preserve
  `correctness-fallback`, `not-performance-preferred`, route support,
  correctness execution, no performance-selection allowance, no performance-win
  claim, correctness fallback path selection, and no performance-preferred path
  selection.
- Rejected stale provider `selected_dispatch_preference` and
  `correctness_fallback_path_selected` values even when candidate metadata was
  rewritten to match those stale provider values exactly.
- Added C++ target validation coverage for the stale-provider exact-mirror
  cases.
- Added dequant-clamp lit coverage for positive policy-output plan/header
  mirrors and stale target-export rejection.
- Updated the RVV plugin code-spec with the target-artifact consumption
  contract that selected-dispatch policy-output mirrors must be anchored in
  resource-selection and same-target measurement facts, not metadata equality.

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] bounded old-authority scan over production/test/spec/PRD diff lines
  returned only PRD negative boundary text forbidding artifact metadata, route
  id, and helper-name authority.

### Self-Repair

- The first focused lit run failed because the new dequant-clamp `PLAN-DAG`
  policy-output checks were placed after stable low-precision metadata that
  appears later in the emission plan. The checks were moved to the actual plan
  order and the same focused lit filter passed 3/3.

### Status

[OPEN MACRO TASK] Gate 4 dispatch/fallback consumption consistency is complete
for the accepted packed-i4 dequant and dequant-clamp no-win evidence paths.
The macro task remains active for the remaining Gate 4 measured-win admission
and final end-to-end provider, selected-dispatch, target mirror, and
dispatch/fallback closure. No performance-preferred dispatch is admitted by
this slice.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 595: Stage2 RVV Gearbox Gate 4 policy-output mirror admission

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a bounded Gate 4 selected-dispatch
/ target artifact policy-output mirror admission slice. The production gap was
that `selected_dispatch_preference` was specified as a provider-owned
policy-output field, but was not yet emitted and target-validated as its own
mirror alongside the existing low-precision `dispatch_preference` mirror.

### Main Changes

- Extended the active macro PRD from the completed Gate 4 root-admission slice
  into the current policy-output mirror admission slice.
- Emitted `tcrv_rvv.low_precision_resource.selected_dispatch_preference` from
  RVV provider-owned route metadata only when
  `hasSelectedDispatchPolicyOutput` is true.
- Added the same key to the RVV target support-bundle metadata mapping.
- Required target artifact validation to reject metadata-only or stale
  `selected_dispatch_preference` mirrors before artifact acceptance.
- Added focused C++ target validation coverage for positive source-backed
  no-win policy output, stale `performance-preferred` preference promotion, and
  metadata-only selected-dispatch preference mirrors.
- Added lit/FileCheck plan/header checks and stale target-export rejection for
  the packed-i4 dequant representative.
- Updated the RVV plugin code-spec so future Gate 4 target tests must cover
  stale `selected_dispatch_preference` policy-output mirrors explicitly.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded old-authority scan over production/test diff lines returned no
  matches; docs-inclusive scan returned only negative PRD/spec boundary text.

### Status

[OPEN MACRO TASK] Gate 4 selected-dispatch policy-output mirror admission is
complete for the `selected_dispatch_preference` target mirror slice. The
accepted packed-i4 evidence remains no-win, so dispatch stays
`correctness-fallback` / `not-performance-preferred`. The macro task remains
active for the rest of Gate 4: full same-target measured-win/no-win provider,
selected-dispatch, target mirror, and fallback/dispatch behavior audit.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 594: Stage2 RVV Gearbox Gate 4 evidence-root admission closure

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed a bounded Gate 4 slice at the
source-backed same-target evidence-root ingestion boundary. The representative
path is the packed-i4 `widening_product_reduce_dequantize_f32` evidence root,
with packed-i4 `widening_product_reduce_dequant_clamp_f32` preserved as the
sibling audit path.

### Main Changes

- Repaired the active macro PRD from Gate 3C completion state into the current
  Gate 4 root-admission slice.
- Made `verifyPackedI4SameTargetEvidenceRoot` compare root-level
  `measurement_harness`, `measurement_schedule_decision_evidence`, and
  `packed_i4_reference_oracle` performance-admission closure/reopen mirrors
  against the parsed source-backed measurement record.
- Added C++ fail-closed coverage for stale root
  `provider_performance_admission_closure` and
  `provider_performance_admission_reopen_requirement` fields, proving the
  policy verifier does not rely only on the nested
  `same_target_measurement_record`.
- Updated the RVV plugin code-spec to record the Gate 4 root-ingestion
  closure/reopen mirror contract and required stale-root tests.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-same-target-measure-gate4-dry-run|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4'`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded old-authority scan over added diff lines returned only PRD
  negative boundary text for artifact-name logic.

### Status

[OPEN MACRO TASK] Gate 4 root-admission closure/reopen validation is complete
for the current packed-i4 source-backed evidence-root slice. The accepted
measurement result remains no-win, so dispatch stays
`correctness-fallback` / `not-performance-preferred`. The macro task remains
active for remaining Gate 4 campaign-level measurement/admission audit and any
future measured-win admission only after fresh source-backed same-target
evidence plus matching provider admission facts exist.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 590: Stage2 RVV Gearbox Gate 2 candidate facts

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 2
widening-product/reduction resource-candidate fact slice. The previous slice
made packed-i4 load/unpack resource facts explicit. This round made the
selected widening-product and widening-reduction candidate facts explicit in
the RVV-owned low-precision resource selection, consumed them during
selected-body realization and route planning, mirrored them into target
artifacts, and rejected stale provider-side and target-side facts.

### Main Changes

- Added explicit widening-product and widening-reduction candidate fact fields
  to RVV low-precision resource candidate/selection payloads.
- Derived signed i8 and unsigned u8 candidate facts from provider-owned
  primitive chain facts instead of from route ids, artifact names, helper names,
  or emitted C spelling.
- Required selected-body realization to consume pre-realized candidate facts and
  emit them on `tcrv_rvv.gearbox_cross_region_handoff`.
- Required selected-body route planning and provider family validation to
  reject stale Gearbox handoff candidate facts before constructing a
  `TCRVEmitCLowerableRoute`.
- Mirrored the candidate facts through route metadata, target support bundle
  keys, and target artifact validation with stale target-mirror rejection.
- Updated the RVV plugin code-spec to record the new resource candidate facts,
  handoff fields, target stale-mirror behavior, and focused test requirements.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] Manual FileCheck stale-provider/stale-target cases for
  `STALE-WIDENING-CANDIDATE`, `STALE-REDUCTION-CANDIDATE`,
  `STALE-HANDOFF-CANDIDATE`, `STALE-TARGET-CANDIDATE`, and
  `STALE-RESOURCE-REDUCTION-CANDIDATE`.

### Self-Repair

- Seeded C++ selected-body and target artifact fixtures with the new explicit
  candidate facts after stricter selected-body realization and target
  validation exposed missing fixture attrs.
- Updated explicit selected-body dequant and pre-realized dequant expectations
  to match the current fail-closed diagnostics and stable metadata/header order.
- Added the code-spec update after recognizing that the new candidate fields are
  a cross-layer compiler contract, not only an implementation detail.

### Status

[OPEN MACRO TASK] Gate 2 is complete for the current low-precision resource
fact spine: packed-i4 load/unpack facts plus widening-product/reduction
candidate facts are explicit, produced by RVV-owned resource paths, consumed by
selected-body realization and route planning, mirrored by target artifacts, and
stale-rejected in focused signed/unsigned and packed-i4-adjacent coverage.

Gates 3 and 4 remain open. The next continuation point is Gate 3 selected-body
realization consumption using this completed fact spine to build
resource-aware executable bodies without changing computation semantics, ABI
roles, dispatch, fallback, or runtime AVL. Do not move to Gate 4
performance-preferred dispatch without fresh source-backed same-target measured
win evidence.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 593: Stage2 RVV Gearbox Gate 3C family completion audit

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3C family
completion/audit slice for the current low-precision resource-aware
selected-body realization campaign. The audit covered signed/unsigned
plain product-reduction resource facts as adjacent Gate 2 route-supported
representatives, plus product-reduction/dequantize-f32 and
product-reduction/dequant-clamp-f32 grouped/unpacked-byte and packed-i4
Gearbox representatives.

### Main Changes

- Rewired Gearbox product-dequant handoff validation to use the canonical
  provider-owned realization-decision helpers for supported decision,
  expected `with_vl` region count, and product phase.
- Added non-clamp product-dequant stale-handoff coverage proving a
  dequant-clamp-only `clamp_phase` fact is rejected before Common EmitC
  materialization.
- Updated the macro PRD with an explicit Gate 3C inventory, acceptance criteria,
  Gate 3 completion status, and Gate 4 continuation boundary.
- Updated the RVV plugin code-spec with the Gate 3C completion/audit evidence
  contract so future slices keep Gearbox schedule validation, selected-body
  realization, dialect handoff verification, route planning, and target
  validation on the same realization-decision mapping.

### Testing

- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32'`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|rvv-gearbox-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] `git diff --check`

### Self-Repair

- The Gate 3C audit found one production-source consistency gap: Gearbox
  product-dequant handoff validation locally rederived the supported decisions,
  expected region count, and product phase instead of consuming the same
  provider-owned realization-decision mapping used by realization, route
  planning, and target validation. The fix replaced the local derivation with
  the canonical helpers.

### Status

[OPEN MACRO TASK] Gate 3 is complete for the current campaign body-family
audit. Gate 4 remains blocked until fresh source-backed same-target measured-win
evidence exists for performance-preferred dispatch admission. The Trellis task
stays active.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 592: Stage2 RVV Gearbox Gate 3B dequant-clamp realization fact consumption

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3B slice for
`widening_product_reduce_dequant_clamp_f32`. This round extended the Gate 3A
realized-body consumption pattern from product/dequant region and phase facts
to provider-derived dequant-clamp clamp/compare-select facts, while preserving
semantics, ABI roles, dispatch/fallback behavior, and runtime AVL.

### Main Changes

- Added dequant-clamp low-precision resource fields:
  `clamp_region_index`, `clamp_phase`, `clamp_compare_select_phase`, and
  `clamp_select_layout`.
- Materialized those facts on realized producer and consumer `with_vl` scopes
  and on `tcrv_rvv.gearbox_cross_region_handoff`.
- Made Gearbox schedule materialization, dialect handoff verification, route
  planning, and target artifact validation consume and stale-reject those facts
  before route or artifact acceptance.
- Mirrored the provider-selected clamp facts into route-plan metadata and target
  header metadata only after provider/route consumers have validated them.
- Added focused lit coverage for positive dequant-clamp fact materialization
  and stale realized-body, handoff, and target mirror rejection.
- Preserved the packed-i4 dequant-clamp sibling fixture with positive clamp
  fact/mirror checks.
- Updated the RVV plugin code-spec and the active macro PRD with the Gate 3B
  field contract and acceptance criteria.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4'`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-11-stage2-rvv-production-gearbox-low-precision-contraction`
- [OK] bounded old-authority scan over added diff lines returned no matches.

### Self-Repair

- `tianchenrv-target-artifact-export-test` initially failed because a synthetic
  packed-i4 sibling-candidate negative test changed `selectedCandidateID` to
  dequant-clamp without completing the new clamp facts. The test now fills the
  clamp facts from provider helpers so it still exercises the intended stale
  performance-baseline diagnostic.
- A header helper declaration was split to match the surrounding formatting
  style before the final build/check pass.

### Status

[OPEN MACRO TASK] Gate 3B is complete for
`widening_product_reduce_dequant_clamp_f32` realization-fact consumption. Gate
2 and Gate 3A facts remain preserved. The macro task stays active for remaining
Gate 3 resource-aware body-family completion/audit and later Gate 4
source-backed same-target measured-win dispatch admission.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 591: Stage2 RVV Gearbox Gate 3A realization fact consumption

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3A selected-body
realization-consumption slice for the current low-precision
product-reduction/dequantize-f32 spine. This round made the route-planning path
consume realization-produced product/dequant region and phase facts from
realized `with_vl` scopes rather than rebuilding those accepted facts locally
from candidate names or route metadata.

### Main Changes

- Added explicit `product_region_index`, `dequant_region_index`,
  `product_phase`, and `dequant_phase` low-precision resource attr constants.
- Required RVV selected-body realization to write product/dequant region and
  phase facts on realized producer and consumer `with_vl` scopes.
- Required the Gearbox schedule materialization path to materialize or
  fail-closed-validate the same realization producer/decision, realized
  unroll/region/peak-live, and product/dequant region/phase facts for explicit
  selected-body fixtures.
- Changed low-precision route-family planning to read realization-produced
  resource facts before route construction and to reject stale realized-body
  product/dequant facts.
- Added focused stale `product_phase` and `dequant_region_index` FileCheck
  coverage for the pre-realized product-reduction/dequantize-f32 artifact path.
- Updated the RVV plugin code-spec to preserve the Gate 3A realization fact
  consumption contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|rvv-gearbox-widening-product-reduce-dequantize-f32'`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|rvv-gearbox-widening-product-reduce-dequantize-f32'`

### Self-Repair

- The first explicit selected-body lit run exposed that the Gearbox schedule
  materialization path wrote provider resource facts but not the new
  realization region/phase facts. The fix moved the fact materialization into
  `materializeLowPrecisionResourceAttrs`, so both explicit and pre-realized
  paths share the same producer-owned fact surface.
- The first pre-realized FileCheck update used an over-broad unordered match
  and could bind to the consumer `with_vl`. The check was tightened to an
  ordered selected-variant anchor and `SAME` region/phase assertions.

### Status

[OPEN MACRO TASK] Gate 3A is complete for the current
product-reduction/dequantize-f32 realization-consumption slice. Gate 3 remains
open for extending the same resource-aware realized-body consumption pattern to
the next low-precision body-family representative. Gate 4 remains later work
and still requires fresh source-backed same-target measured-win evidence before
performance-preferred dispatch admission.

### Git Commits

Final coherent commit is created after this journal entry.
