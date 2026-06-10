# Stage2 RVV Production-Kernel Capability Campaign

## Goal

Create one active macro owner for the Stage 2 RVV production-kernel capability
campaign. The campaign ties Gearbox resource-aware selected-body realization,
low-precision contraction primitive facts, source-backed same-target
measurement records, and selected-dispatch/performance policy inputs into one
representative quantized contraction pressure path.

The pressure path may use packed-i4 widening product-reduce dequant-clamp as a
representative stressor, but q8/q4 names, llama.cpp labels, artifact names,
benchmark labels, route ids, ABI strings, raw stdout, or metadata-only records
must never become route, dtype, schedule, measurement, dispatch, or performance
authority.

## What I Already Know

- The previous macro task
  `06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign` is
  archived after Gates 1-4 completed.
- That completed task established typed low-precision primitive facts,
  selected-body realization consumption, source-backed same-target measurement
  records, and strict selected-dispatch/performance policy consumption.
- The new owner should not be another generated-bundle evidence seam. It must
  advance the production boundary that connects realization/resource facts to
  measurement and policy inputs.
- Specs require authority to flow through selected typed `tcrv_rvv` body/config
  facts, RVV provider-owned facts, validated target mirrors, source-backed
  measurement records, and selected-dispatch policy checks.
- Common EmitC/export may carry provider payloads and mirrors only. It must not
  infer dtype, schedule, low-precision resource identity, measurement identity,
  selected-dispatch behavior, or performance decisions.
- The immediately previous slice, commit
  `d35f9f42 rvv: add production pressure profile boundary`, completed Gate 1 by
  adding the production pressure-profile boundary and focused fail-closed
  policy tests.

## Requirements

- Keep this Trellis task active across rounds until all campaign gates are
  genuinely complete or human steering redirects the campaign.
- Complete one coherent slice per round and update this PRD with completed and
  remaining gates.
- Gate 1 made a concrete production compiler or production workflow change
  in the pressure-profile, measurement-policy input, provider validation, or
  selected-dispatch boundary.
- Pressure-profile facts must be tied to provider-owned typed body/config,
  Gearbox/resource, low-precision primitive, runtime ABI, target capability,
  same-target measurement, and selected-dispatch policy provenance.
- Fail closed when pressure-profile facts are label-only, stale, sibling-route,
  metadata-only, disconnected from typed body/config/provider provenance, or
  tied to a selected-dispatch boundary that does not match the measurement and
  policy input.
- Do not add q8/q4-named route ids, llama.cpp-named route authority, high-level
  frontend contracts, one-off generated-bundle ABI seams, evidence-only
  closeouts, dashboard/database work, source-front-door positive routes,
  Common EmitC semantic inference, dtype/LMUL clone batches, or unrelated RVV
  primitive expansion.

## Macro Campaign Gates

- [x] Gate 1: production pressure-profile boundary ties typed
  low-precision/Gearbox/provider facts to same-target measurement and
  selected-dispatch policy inputs, with stale/label-only/sibling-route/
  metadata-only fail-closed validation.
- [x] Gate 2: generated artifact and measurement workflow emits source-backed
  same-target comparison records for the representative pressure path on
  `ssh rvv`.
- [x] Gate 3: selected-dispatch/performance policy consumes those records for
  correctness fallback versus measured-win preference with stale/no-win/
  regression rejection.
- [ ] Gate 4: campaign closeout proves the full selected boundary ->
  plugin-owned realization/provider -> artifact/runtime measurement ->
  selected-dispatch policy path and archives only after all gates are complete.

## Current Round Slice: Gate 3

Implement Gate 3 only. The bounded slice should make the selected-dispatch /
performance policy seam consume the source-backed pressure-profile records
emitted by Gate 2, with decisions derived from selected-boundary,
provider-owned low-precision primitive/resource, generated artifact,
same-target measurement, and dispatch-boundary facts.

The policy may use the packed-i4 widening product-reduce dequant/dequant-clamp
records as representative pressure inputs. It must treat q8/q4-style labels,
artifact paths, route ids, benchmark names, raw stdout, and metadata-only
records as mirrors or invalid pressure facts, never as route, dtype, schedule,
measurement, dispatch, or performance authority.

Acceptance criteria:

- [x] Production C++ policy code consumes source-backed
  `RVVLowPrecisionSameTargetMeasurementRecord` / pressure-profile records
  through the selected-dispatch boundary before producing a dispatch /
  performance decision.
- [x] Positive focused coverage proves the current accepted no-win/regression
  source-backed record selects `correctness-fallback`, preserves route support
  and correctness execution, and denies performance preference / win claims.
- [x] Positive focused coverage proves a measured-win record selects
  `performance-preferred` only when provider maturity, eligibility, dispatch,
  remediation, schedule, primitive/resource, measurement, and selected-dispatch
  facts all agree.
- [x] Negative focused coverage rejects or safely resolves stale/missing
  provenance, label-only q8/q4 pressure, sibling-route measurements, stale
  selected-dispatch facts, correctness-disabled records, stale schedule /
  primitive / target facts, and measurement-only win promotion.
- [x] Relevant focused checks pass, including C++ plugin tests, script/lit checks
  only if touched, `git diff --check`, `git diff --cached --check`, and a
  bounded old-authority scan over touched files or added diff lines.
- [x] The task remains active after the Gate 3 slice unless Gate 4 is also
  genuinely complete.

## Completed Slice: Gate 1

- Added `RVVLowPrecisionProductionPressureProfile` plus build/verify APIs in
  the RVV low-precision performance policy boundary.
- The pressure-profile materializer consumes provider-owned resource/Gearbox,
  typed low-precision primitive, runtime ABI, target capability,
  source-backed same-target measurement, and selected-dispatch policy facts in
  one boundary object.
- The selected-dispatch policy-input and source-backed measurement-record
  overloads now pass through the pressure-profile boundary before returning a
  dispatch/performance decision.
- Added focused RVV plugin coverage for positive pressure-profile construction
  from provider-owned facts and source-backed records.
- Added fail-closed coverage for label-only q8/q4 pressure authority, stale
  runtime ABI, stale primitive intrinsic, stale schedule decision,
  metadata-only provider support, sibling-route measurements, and stale
  selected-dispatch origin.

## Completed Slice: Gate 2

- Extended `RVVLowPrecisionSameTargetMeasurementRecord`,
  `RVVLowPrecisionSameTargetMeasurementPolicyInput`, and
  `RVVLowPrecisionProductionPressureProfile` with source-backed artifact /
  measurement fields: selected variant, selected input, generated function,
  generated object/header path and SHA256, measurement target/provenance,
  runtime-count set/provenance, and a non-authoritative production pressure
  label/provenance pair.
- Added strict C++ validation before policy-boundary materialization for missing
  source contract, stale selected-boundary facts, missing or marker-only
  generated artifact identity, wrong measurement target, stale runtime-count
  provenance, and q8/q4 label-only pressure.
- Updated `rvv_generated_bundle_same_target_measure.py` so the generated
  artifact / same-target measurement workflow emits the source-backed record
  into per-op evidence, root evidence, and provider feedback tie-back inputs
  without rewriting provider-owned maturity/resource facts.
- Refreshed the checked-in packed-i4 dequant-clamp `ssh rvv` evidence JSONs by
  adding source-backed record fields derived from the existing validated
  artifact identity and measurement configuration. This preserves the old raw
  timing evidence and does not claim a new runtime/performance run.
- Added focused positive and negative C++ coverage plus script self-test and
  FileCheck coverage for the new source-backed fields.

## Completed Slice: Gate 3

- Hardened the selected-dispatch safe resolver for source-backed
  `RVVLowPrecisionSameTargetMeasurementRecord` inputs so it consumes the same
  production pressure-profile boundary as the strict selected-dispatch policy
  evaluator before returning an accepted decision.
- If selected-dispatch pressure facts are marker-only or otherwise stale after
  the record/resource/measurement handoff has been accepted, the resolver now
  denies performance preference and win claims, preserves correctness fallback
  for the legal route, and carries the precise pressure-profile diagnostic in
  the fallback reason.
- Added focused C++ coverage proving the selected-dispatch record resolver
  consumes the source-backed pressure-profile boundary and falls back on a
  metadata-only selected-dispatch marker instead of treating it as policy
  authority.
- Existing Gate 3/4 C++ coverage continues to prove current no-win/regression
  records select `correctness-fallback`, measured-win records select
  `performance-preferred` only after matching provider facts, parsed
  dequant-clamp evidence is consumed through the record overload, and stale
  target/runtime ABI/planning/vsetvl/primitive/schedule/correctness/win
  promotion cases fail closed.

## Verification

- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [x] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [x] Packed-i4 dequant dry-run with manual
  `FileCheck --check-prefix=PACKED-WPRD`
- [x] Packed-i4 dequant-clamp dry-run with manual
  `FileCheck --check-prefix=PACKED-CLAMP-WPRDC`
- [x] `python3 -m json.tool` on refreshed packed-i4 evidence JSON files
- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [x] `git diff --check`
- [x] `git diff --cached --check`
- [x] Bounded current-diff scan for legacy RVV route-authority markers returned
  no new route-authority matches; remaining q8/q4/metadata-only occurrences are
  negative guards, pressure labels, spec text, or evidence mirrors.
- [x] Gate 3: `cmake --build build --target
  tianchenrv-rvv-extension-plugin-test`
- [x] Gate 3: `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Gate 3: `cmake --build build --target tcrv-opt tcrv-translate`
- [x] Gate 3: `git diff --check`
- [x] Gate 3: `git diff --cached --check`
- [x] Gate 3: bounded added-diff scan for legacy RVV route-authority markers
  returned no `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `__riscv_*_i32m1`, source-front-door, or
  descriptor-driven route-authority matches. The broader scan only found PRD
  non-authority wording and the new metadata-only negative test marker.

## Spec Update Decision

`.trellis/spec/extension-plugins/rvv-plugin.md` was updated for Gate 2 because
this slice changed a cross-layer record/API contract. The spec now names the
source-backed record fields, validation matrix, bad cases, and required tests
for generated artifact identity, selected-boundary provenance, measurement
target/runtime-count provenance, and non-authoritative pressure labels.

Gate 3 did not require a spec update. The RVV plugin spec already requires the
selected-dispatch record resolver to prefer source-backed measurement records,
consume the record/resource/measurement handoff through the pressure-profile
boundary, preserve correctness fallback for stale evidence, and deny
performance preference for marker-only or mismatched pressure facts. This slice
implements that existing contract in production policy code and tests.

## Status

Open macro task. Gates 1, 2, and 3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: campaign closeout must prove the full selected
boundary -> plugin-owned realization/provider -> artifact/runtime measurement
-> selected-dispatch policy path, including a parsed generated evidence record
through the selected-dispatch record overload, before the macro task can be
finished or archived.

## Out of Scope

- Runtime/performance claims without new `ssh rvv` evidence.
- New frontend lowering contracts or high-level Linalg route support.
- New q8/q4/llama.cpp route identities.
- Common EmitC dtype, schedule, low-precision, measurement, or policy
  inference.
- Broad benchmark dashboards, global tuning databases, or performance profile
  systems.
- Archiving this macro task after only Gates 1-2.

## Technical Notes

- Read `.trellis/spec/index.md`.
- Read `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Read `.trellis/spec/lowering-runtime/emitc-route.md`.
- Read `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  selected-body and production-kernel campaign guidance.
- Read `.trellis/spec/testing/mlir-testing-contract.md` same-target measurement
  and selected-dispatch bundle-boundary contracts.
- Read archived primitive-surface campaign PRD at
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign/prd.md`.
- Initial repository state: `main`, clean worktree, latest commit
  `d35f9f42 rvv: add production pressure profile boundary`.
