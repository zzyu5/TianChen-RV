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

## Current Slice: Gate 2b Representative Resource Consumption

- [x] Inspect repository state, recent commits, and current Trellis pointer.
- [x] Read relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/index.md`.
- [x] Inspect current RVV primitive, Gearbox/resource, selected-body
  realization, statement-plan/provider, selected-dispatch policy, and target
  validation surfaces for dequant-clamp and packed-i4 representatives.
- [x] Make the RVV Gearbox schedule pass preserve the dequant-clamp resource
  operation when it re-consumes an already realized product-reduction
  cross-region handoff. The pass now derives direct dequant-store versus
  lower/upper clamp-store from realized body structure before writing resource
  attrs onto producer and consumer `with_vl` regions.
- [x] Keep packed-i4 selected-body, statement-plan, remediation, schedule, and
  target-boundary facts provider-owned. Source inspection found the packed-i4
  path already verifies handoff remediation/schedule facts and target artifact
  resource mirrors; this slice refreshed the fixture checks to match actual
  attr order and reran the representative positive and stale-fact chains.
- [x] Prevent packed-i4 Gate 4 measurement policy verification from becoming a
  blocker for non-packed dequant/dequant-clamp representative resource paths.
  Provider plan verification now enters the selected-dispatch performance
  policy gate only for provider-selected packed-i4 candidates.
- [x] Add focused coverage showing the dequant-clamp realized
  handoff/consumer path keeps the clamp resource candidate, primitive kind, and
  memory form, and fails closed when a stale dequantize memory form is injected
  before route planning.
- [x] Run focused RVV plugin/target checks plus diff/authority checks.
- [x] Update this PRD and the workspace journal with completed Gate 2 behavior
  and the precise remaining Gate 2 continuation point.
- [x] Commit one coherent Gate 2 slice while leaving `.trellis/.current-task`
  active.

## Acceptance Criteria For This Round

- [x] Production source changes stay in the RVV plugin/provider seams, not
  Common EmitC.
- [x] Dequant-clamp realized selected-body consumption derives the selected
  resource memory form from body structure and provider primitive/resource
  facts, not from mirrors, route ids, ABI strings, artifact names, or helper
  names.
- [x] A stale dequantize memory form injected into a dequant-clamp resource path
  fails before route construction with a targeted diagnostic naming the expected
  clamp memory form, stale dequantize form, and selected dequant-clamp
  candidate.
- [x] Packed-i4 representative selected-body handoff, statement planning,
  schedule/remediation facts, and target-boundary validation remain
  fail-closed for stale or disconnected provider-owned resource facts.
- [x] Packed-i4 Gate 4 measurement/policy checks still apply to provider-owned
  packed-i4 candidates, while non-packed dequant-clamp Gate 2 realization is no
  longer blocked by packed-i4 measurement evidence.
- [x] No Common EmitC code infers RVV low-precision semantics.
- [x] No q8/q4/llama label, artifact name, route id, helper name, descriptor,
  status field, source-front-door marker, or test name becomes authority.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] The task remains active because Gates 3-4 are not complete and any
  remaining Gate 2 sub-slice must stay explicit.

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

## Completed Slice: 2026-06-10 Gate 2a

- Added a Gearbox-pass local primitive-surface consumption check for bounded
  product-reduction-dequantization resource scheduling. After the pass builds
  and selects a `RVVLowPrecisionContractionResourceCandidate`, it now obtains
  the provider-owned `RVVLowPrecisionWideningReductionPrimitiveFacts` for the
  selected product-reduction operation and compares source/product/
  accumulator/result dtype, SEW/LMUL, signedness, primitive contracts, chain
  kind, widening product relation, product-reduction relation, widening product
  intrinsic, reduction intrinsic, scalar seed splat, layouts, and store-VL
  before writing schedule/resource attrs.
- Kept selected-body realization as the consumer that materializes the selected
  schedule into producer/consumer `with_vl` regions, vsetvl markers, handoff,
  and provider-verifiable primitive-chain attrs. Common EmitC remains unchanged.
- Added FileCheck coverage proving `--tcrv-rvv-materialize-gearbox-schedules`
  alone carries the primitive-chain resource facts on the pre-realized selected
  body, plus a stale schedule primitive negative chain that fails in selected
  body realization before route construction.
- Rebuilt `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
  Ran both C++ tests, the new Gearbox primitive schedule FileCheck positive
  chain, and the stale schedule primitive negative chain using
  `/usr/bin/FileCheck-20`.
- Local `FileCheck` and `not` are not on PATH; `/usr/bin/FileCheck-20` was used
  for direct checks, and the negative chain used an explicit shell exit-code
  check. `llvm-lit` was not available locally.
- No runtime/correctness/performance claim was made, so no `ssh rvv` evidence
  was required in this Gate 2a slice.

## Completed Slice: 2026-06-10 Gate 2b

- Fixed the dequant-clamp representative resource path in
  `RVVGearboxSchedules.cpp`. When the Gearbox schedules pass re-consumes an
  already realized cross-region product-reduction handoff, it now records
  whether the consumer stores the dequantized value directly or feeds a
  lower-then-upper clamp compare/select chain, and materializes the matching
  dequant or dequant-clamp resource memory form onto both producer and consumer
  `with_vl` regions.
- Narrowed selected-dispatch performance policy verification in
  `RVVEmitCContractionRouteFamilyPlanOwners.cpp` so the packed-i4 Gate 4 policy
  gate runs only for provider-selected packed-i4 resource candidates. This keeps
  packed-i4 policy validation active without letting packed-i4 measurement
  evidence become a false prerequisite for non-packed dequant-clamp Gate 2
  realization.
- Added dequant-clamp FileCheck coverage for the realized
  handoff/resource-consumption path after a second Gearbox schedules pass. The
  check proves the clamp selected candidate, clamp primitive kind, and clamp
  memory form survive provider-owned resource consumption, with no fallback to
  the sibling dequantize memory form.
- Added a stale dequant-clamp resource-memory negative chain that fails before
  emission-plan route construction when a dequantize memory form is injected
  into the dequant-clamp resource path.
- Refreshed packed-i4 representative fixture checks to match the actual
  provider-owned attr order and reran positive selected-body, statement-plan,
  target-header, CPP emission, handoff-remediation stale, and artifact schedule
  stale checks.
- No runtime/correctness/performance claim was made, so no `ssh rvv` evidence
  was required in this Gate 2b slice.

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

Gate 2 remains open after this sub-slice, but Gate 2b representative
dequant-clamp and packed-i4 resource-consumption evidence is complete. Continue
by auditing whether any remaining low-precision product-reduction
representative still lacks direct selected-body/provider/statement-plan/target
proof. If no source gap remains, close Gate 2 explicitly and move to Gate 3
generated artifact correctness plus same-target `ssh rvv` evidence for the
changed production path. Gates 3 and 4 remain future work.
