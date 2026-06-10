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

## Requirements

- Keep this Trellis task active across rounds until all campaign gates are
  genuinely complete or human steering redirects the campaign.
- Complete one coherent slice per round and update this PRD with completed and
  remaining gates.
- Gate 1 must make a concrete production compiler or production workflow change
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
- [ ] Gate 2: generated artifact and measurement workflow emits source-backed
  same-target comparison records for the representative pressure path on
  `ssh rvv`.
- [ ] Gate 3: selected-dispatch/performance policy consumes those records for
  correctness fallback versus measured-win preference with stale/no-win/
  regression rejection.
- [ ] Gate 4: campaign closeout proves the full selected boundary ->
  plugin-owned realization/provider -> artifact/runtime measurement ->
  selected-dispatch policy path and archives only after all gates are complete.

## Current Round Slice: Gate 1

Implement Gate 1 only unless repository evidence proves it is already complete
in production source. The bounded slice should add or harden a production
pressure-profile boundary for the representative packed low-precision
contraction path, such as packed-i4 widening product-reduce dequant-clamp.

Acceptance criteria:

- [x] Production C++ or production workflow code constructs or validates a
  pressure-profile input from provider-owned typed low-precision primitive
  facts, Gearbox/resource facts, runtime ABI facts, target facts,
  same-target measurement identity, and selected-dispatch policy tie-backs.
- [x] Positive focused coverage proves the pressure profile can be built from
  provider-owned typed facts, not labels or artifact names.
- [x] Negative focused coverage rejects label-only q8/q4 pressure, stale
  primitive/resource/schedule/runtime facts, sibling-route measurements,
  metadata-only measurement records, and stale selected-dispatch tie-backs.
- [x] Relevant focused checks pass, including C++ plugin/target tests when
  touched, measurement script self-test when touched, lit/script tests when
  touched, `git diff --check`, `git diff --cached --check`, and a bounded
  old-authority scan over touched files or added diff lines.
- [x] The task remains active after the Gate 1 slice unless all macro gates are
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

## Verification

- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [x] `git diff --check`
- [x] `git diff --cached --check`
- [x] Bounded current-diff scan for legacy RVV route-authority markers returned
  no matches.

## Spec Update Decision

No `.trellis/spec/` update is required for Gate 1. The existing RVV plugin,
EmitC route, variant-pipeline, and testing specs already require provider-owned
typed low-precision facts, same-target measurement records, selected-dispatch
mirrors, and fail-closed stale/metadata-only rejection. This slice implements
that existing contract as a production pressure-profile boundary.

## Status

Open macro task. Gate 1 is complete. Gates 2-4 remain open. The next
continuation point is Gate 2: generated artifact and measurement workflow
emits source-backed same-target comparison records for the representative
pressure path on `ssh rvv`, using the production pressure-profile boundary as
the policy/measurement input contract.

## Out of Scope

- Runtime/performance claims without new `ssh rvv` evidence.
- New frontend lowering contracts or high-level Linalg route support.
- New q8/q4/llama.cpp route identities.
- Common EmitC dtype, schedule, low-precision, measurement, or policy
  inference.
- Broad benchmark dashboards, global tuning databases, or performance profile
  systems.
- Archiving this macro task after only Gate 1.

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
  `aacf5600 rvv: archive low-precision primitive campaign`.
