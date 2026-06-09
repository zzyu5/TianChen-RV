# Stage2 RVV low-precision contraction primitive-surface campaign

## Goal

Create one macro owner for the reusable RVV low-precision contraction
primitive surface. The campaign makes typed i8/u8 and packed-low-precision
body/config facts flow into RVV-owned primitive facts, fail-closed provider and
target validation, resource-aware Gearbox selected-body realization, generated
artifacts, and measured same-target policy inputs without letting q8/q4 names,
artifact names, route ids, ABI strings, helper names, or Common EmitC become
authority.

## What I Already Know

- Session start had no `.trellis/.current-task`; this task was created from the
  Hermes Direction Brief as the macro owner.
- Commit `5bff0e50` completed and archived the packed-i4 performance
  remediation campaign. Its Gate 4 consumes same-target measurement and
  provider schedule decisions, preserves correctness fallback for the current
  regression, and does not leave an active task.
- The archived packed-i4 campaign is
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-packed-i4-performance-remediation-campaign/`.
- Current specs require Stage 2 low-precision work to advance typed
  `tcrv_rvv` primitive coverage, provider-owned facts, selected-body
  realization, target validation, and measured evidence. Repeated generated
  bundle or `ssh rvv` evidence alone is not sufficient.
- Existing source already contains several relevant production surfaces:
  `RVVWideningProductRouteFacts`,
  `RVVLowPrecisionWideningReductionPrimitiveFacts`,
  `RVVLowPrecisionContractionResourceSelection`, Gearbox packed-i4 resource
  facts, provider validation, target artifact mirror validation, and lit/C++
  tests for signed i8, unsigned u8, and packed-i4 paths.
- A bounded Gate 1 hardening point remains useful: provider resource-selection
  validation should directly consume the full low-precision widening-reduction
  primitive facts for source/product/accumulator/reduction dtype, signedness,
  SEW/LMUL, relations, intrinsics, layouts, and store-VL instead of relying only
  on adjacent route-plan fields and selected string mirrors.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per worker round, commit it, and leave
  this task active while remaining gates are incomplete.
- Gate 1 must change production RVV plugin/provider/target source unless live
  source inspection proves the gap is already closed.
- RVV plugin/provider code owns low-precision primitive facts, legality,
  selected-body realization inputs, route facts, and fail-closed diagnostics.
- Common EmitC/export may carry provider-built payloads and mirrors only; it
  must not infer low-precision dtype, unpack/sign/zero-extension, widening
  product, reduction, layout, policy, schedule, or dispatch semantics.
- Primitive facts must be derived from typed `tcrv_rvv` body/config/runtime and
  RVV capability facts, not q8/q4 labels, llama.cpp names, route ids, artifact
  names, ABI strings, test names, descriptors, or old helper names.
- Missing, stale, disconnected, metadata-only, or measurement-only primitive
  facts must fail closed before route support, route planning, target artifact
  acceptance, or performance preference.

## Macro Campaign Gates

- [x] Gate 1: production primitive-surface contract plus fail-closed consumer
  for low-precision widening contraction facts.
- [ ] Gate 2: RVV plugin-local selected-body realization and Gearbox scheduling
  consume those primitive facts for a coherent contraction slice.
- [ ] Gate 3: generated artifact correctness plus same-target `ssh rvv`
  measurement for the changed production path when runtime/correctness or
  performance behavior is claimed.
- [ ] Gate 4: selected-dispatch/performance policy consumes measurement and
  schedule facts without promoting measurement-only wins.

## Current Slice: Gate 1 Primitive-Surface Contract Hardening

- [x] Inspect repository state, recent commits, and current Trellis pointer.
- [x] Create this macro Trellis task because no active task existed.
- [x] Read relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/index.md`.
- [x] Read the archived packed-i4 performance-remediation campaign PRD.
- [x] Inspect current RVV primitive, Gearbox/resource, provider, and target
  validation surfaces.
- [x] Harden the provider-owned low-precision primitive-surface consumer so
  product-reduction resource selection directly validates source/product/
  accumulator/reduction dtype, signedness, SEW/LMUL, widening product relation,
  product-reduction chain relation, product intrinsic, reduction intrinsic,
  scalar seed splat, layouts, and reduction store VL against
  `RVVLowPrecisionWideningReductionPrimitiveFacts`.
- [x] Add focused C++ coverage proving a stale primitive-surface resource fact
  fails closed before route acceptance.
- [x] Run focused RVV plugin and target artifact tests plus diff/authority
  checks.
- [x] Update this PRD and the workspace journal with completed Gate 1 behavior
  and the precise Gate 2 continuation point.
- [ ] Commit one coherent Gate 1 slice while leaving `.trellis/.current-task`
  active.

## Acceptance Criteria For This Round

- [x] Production source changes in the RVV plugin/provider/validation path.
- [x] The new/strengthened consumer is provider-owned and fail-closed; it is not
  documentation, generated-bundle evidence, or report-only metadata.
- [x] The accepted signed i8 product-reduction chain still validates source
  `i8/mf4`, product `i16/mf2`, accumulator/reduction `i32/m1`, signed source
  facts, `vwmul`, `vwredsum`, scalar seed splat, accumulator/result layout, and
  store-VL.
- [x] A stale primitive-surface fact fails before route acceptance with a
  targeted diagnostic.
- [x] Existing unsigned u8 widening-product target validation remains intact.
- [x] Existing packed-i4 policy and schedule-decision behavior remains
  conservative; this round does not claim a new performance result.
- [x] No Common EmitC code infers RVV low-precision semantics.
- [x] No q8/q4/llama label, artifact name, route id, helper name, descriptor,
  status field, source-front-door marker, or test name becomes authority.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] The task remains active because Gates 2-4 are not complete.

## Completed Slice: 2026-06-10 Gate 1

- Hardened the RVV provider-owned low-precision product-reduction primitive
  consumer by adding direct validation from
  `RVVLowPrecisionContractionResourceSelection` to
  `RVVLowPrecisionWideningReductionPrimitiveFacts`.
- The provider now fail-closes stale source/product/accumulator/reduction
  dtype, signedness, SEW/LMUL, and final result dtype before ordinary route
  field checks can accept a product-reduction resource selection.
- Moved primitive-chain validation earlier in both route-family plan and route
  description resource-selection validation so primitive-surface errors are
  caught before route acceptance rather than only at later target artifact
  mirror checks.
- Added focused C++ coverage for a stale product-reduction primitive product
  SEW value. The diagnostic includes `primitive product SEW`, expected `16`,
  and stale `32`.
- Rebuilt and ran `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- Manually ran signed and unsigned widening-product artifact export chains with
  `build/bin/tcrv-opt` and `build/bin/tcrv-translate`, plus an unsigned stale
  primitive signedness negative check. `llvm-lit` was not available locally.
- No new runtime/correctness/performance claim was made, so no `ssh rvv`
  evidence was required in this Gate 1 slice.

## Out Of Scope

- New q8/q4/llama-named route ids, wrappers, artifact authority, or benchmark
  authority.
- High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
- Per-Linalg route authority or broad dtype/LMUL clone batches.
- New global autotuning database, dashboard, or report-only path.
- Common EmitC invention of low-precision compute, unpack, dtype, schedule,
  policy, or intrinsic semantics.
- Standalone generated-bundle or `ssh rvv` evidence closeout without a changed
  production path.
- A performance-win claim or performance-preferred dispatch change.

## Technical Notes

- Main source seams inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Current implementation already supports signed i8 widening product,
  unsigned u8 widening product, signed i8 product-reduction/vwredsum facts, and
  packed-i4 remediation/schedule facts. This slice hardens the provider
  resource consumer rather than introducing q8/q4 authority.

## Continuation Point

After Gate 1 lands, continue with Gate 2: make RVV plugin-local
selected-body realization/Gearbox scheduling consume the reusable primitive
surface for a coherent low-precision contraction slice beyond this validation
handoff.
