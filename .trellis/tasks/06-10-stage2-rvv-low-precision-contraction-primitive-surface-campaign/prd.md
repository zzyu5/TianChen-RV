# Stage2 RVV Low-Precision Contraction Primitive-Surface Campaign

## Goal

Create one macro owner for the RVV Stage 2 low-precision contraction
primitive surface. The campaign must make low-precision operand element
facts, typed vector/config facts, widening product facts, accumulator/reduction
facts, selected-body realization facts, route/provider legality, target
validation, and later same-target policy consumption flow through production
RVV-owned compiler surfaces.

The current round implements Gate 2 only. It must harden RVV plugin-local
selected-body realization so a representative low-precision
product-reduction/dequant path consumes the Gate 1 primitive facts before route
construction. The realization boundary must preserve compute semantics, dtype
semantics, ABI roles, runtime AVL/VL, variant origin, dispatch, and fallback
behavior while rejecting missing or stale primitive/resource facts before Common
EmitC, target artifact metadata, q8/q4 labels, packed-i4 labels, descriptor
residue, generated C strings, route ids, or artifact names can stand in for
typed body/config/provider facts.

## What I Already Know

- Session start had no `.trellis/.current-task`; this task was created from the
  Hermes Direction Brief as the new macro owner after the Gearbox
  resource-aware selected-body realization campaign was completed and archived
  by commit `229e21bc`.
- The archived Gearbox campaign closed provider-owned resource facts,
  target-artifact policy mirrors, same-target measurement facts, correctness
  fallback, and performance preference denial for the current regression/no-win
  evidence.
- Current specs require Stage 2 work to advance corrected typed `tcrv_rvv`
  coverage and RVV plugin-local selected-body realization. q8/q4 and
  llama.cpp-style kernels are pressure tests, not route, artifact, or naming
  authority.
- Existing source already has low-precision contraction route planning,
  selected-body realization, Gearbox/resource facts, target artifact validation,
  and low-precision performance policy code. Gate 1 must inspect those
  surfaces and close the first real production gap instead of adding another
  generated-bundle or evidence-only seam.
- The first likely owner is the RVV dialect/plugin/provider/target-validation
  boundary that decides whether a selected `tcrv_rvv` body structurally carries
  low-precision operand element/config facts and widening product/accumulator
  facts.
- Source inspection showed the signed i8 product-reduction primitive surface
  already exists in production provider code, selected-body realization,
  target mirror validation, and focused C++/lit tests. The Gate 1 gap for this
  round is narrower: target artifact provider-facts validation did not directly
  compare `lowPrecisionResourceSelection` source/product/accumulator/
  reduction/final primitive surface facts against
  `RVVLowPrecisionWideningReductionPrimitiveFacts` before artifact export.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Complete one coherent milestone slice per worker round, commit it, and leave
  this task active while remaining gates are incomplete.
- Gate 2 must change or harden production compiler/validation code. PRD-only,
  journal-only, generated-bundle dry-run-only, broad smoke-only, or evidence
  packaging-only work does not satisfy the slice.
- Primitive authority must come from typed `tcrv_rvv` body/config/provider
  facts. It must not come from q8/q4 labels, llama.cpp names, packed-i4
  artifact labels, descriptor residue, generated C strings, route ids,
  artifact names, Common EmitC, status fields, or metadata mirrors.
- The Gate 2 surface must prove that selected-body realization consumes the
  smallest coherent production subset of low-precision contraction facts that
  source inspection proves is missing or weak. Acceptable minimum scope is the
  signed i8 product-reduction/dequant path with source/product/accumulator/
  reduction/final element facts, SEW/LMUL, signedness, primitive chain facts,
  selected resource decision, runtime AVL/VL, and provider-visible realization
  schedule facts.
- Common EmitC must remain neutral. It may consume provider-built route payloads
  that already exist, but it must not invent RVV dtype, SEW/LMUL, widening, or
  contraction semantics.
- Focused tests must prove both the positive selected-body realization path and
  stale or missing primitive/resource facts failing closed before route or
  artifact acceptance.

## Macro Campaign Gates

- [x] Gate 1: typed low-precision contraction primitive facts and fail-closed
  provider/validation surface exist in production code.
- [x] Gate 2: RVV plugin-local selected-body realization consumes those facts
  for a representative low-precision contraction/dequant path without changing
  compute semantics.
- [ ] Gate 3: route/provider/artifact export carries those facts into generated
  artifacts and source-backed same-target measurement records.
- [ ] Gate 4: selected-dispatch/performance policy consumes those measurements
  fail-closed, preserving correctness fallback and denying stale/no-win
  performance claims.

## Current Slice: Gate 2

- [x] Inspect the RVV contraction selected-body realization owner and
  provider-facing low-precision resource selection path for the representative
  product-reduction/dequant body.
- [x] Harden selected-body realization so the selected resource candidate's
  primitive chain fields are checked directly against provider-owned
  `RVVLowPrecisionWideningReductionPrimitiveFacts` before realized `with_vl`,
  region-marker, handoff, or provider-visible resource facts are accepted.
- [x] Preserve the existing positive realization path: pre-realized
  product-reduction/dequant selected bodies become realized typed
  setvl/with_vl/load/widening_product/standalone_reduce/handoff/dequant/store
  structure with provider-visible resource and primitive facts.
- [x] Add focused negative C++ coverage for missing or stale primitive/resource
  facts failing closed at selected-body realization before route or artifact
  authority.
- [x] Run focused C++ tests, directly touched tests, whitespace checks, bounded
  old-authority scans, and Trellis validation.
- [x] Commit the Gate 2 slice and keep `.trellis/.current-task` active because
  Gates 3-4 remain incomplete.

## Completed Slice: Gate 1

- Hardened `validateRVVLowPrecisionPrimitiveChainResourceProviderFacts` so
  target artifact provider-facts validation directly rejects stale
  low-precision resource selection primitive surface fields before artifact
  export.
- Added target artifact C++ coverage mutating a packed-i4 product-reduction
  dequant resource selection's primitive product SEW to prove stale resource
  primitive facts fail closed at the target provider-facts boundary.
- Verified the slice with focused target/plugin C++ tests and whitespace
  checks. No `ssh rvv` evidence was claimed or required because this slice does
  not make a new runtime/correctness/performance claim.

## Completed Slice: Gate 2

- Hardened `RVVContractionSelectedBodyRealizationOwner` so the selected
  low-precision product-reduction/dequant resource candidate's primitive
  contract, primitive kind, chain contract/kind, product/reduction relations,
  widening/reduction/seed intrinsics, accumulator/result layout, and reduction
  store-VL must match provider-owned
  `RVVLowPrecisionWideningReductionPrimitiveFacts` before realization accepts
  the candidate.
- Preserved the positive realization path where the representative
  pre-realized product-reduction/dequant body materializes into typed
  setvl/with_vl/load/widening_product/standalone_reduce/handoff/dequant/store
  structure carrying provider-visible resource and primitive facts.
- Added focused RVV plugin C++ coverage proving missing primitive resource
  facts and stale primitive reduction intrinsic facts fail closed at selected
  body realization, before provider route construction or target artifact
  acceptance.
- Verified the slice with focused plugin/target C++ tests, Trellis validation,
  whitespace checks, and a bounded old-authority scan. No `ssh rvv` evidence was
  claimed or required because this slice does not make a new runtime,
  correctness, or performance claim.

## Remaining Macro Gates

- Gate 3: route/provider/artifact export must carry the facts into generated
  artifacts and source-backed same-target measurement records.
- Gate 4: selected-dispatch/performance policy must consume those measurements
  fail-closed while preserving correctness fallback.

## Next Continuation Point

Continue Gate 3 in this same task: inspect route/provider/artifact export for
the representative low-precision product-reduction/dequant path and make the
facts consumed by selected-body realization carry into generated artifacts and
source-backed same-target measurement records with fail-closed stale mirror
diagnostics. Do not switch to performance policy until Gate 3 is complete.

## Out Of Scope

- No generated-bundle or `ssh rvv` evidence closeout as the default milestone.
- No q8/q4/llama.cpp route ids, wrappers, artifact names, or naming authority.
- No dtype/LMUL clone batch.
- No high-level Linalg, Vector, StableHLO, or per-Linalg authority work.
- No global tuning database or broad dispatch-policy rewrite.
- No Common EmitC invention of RVV semantics.
- No resurrection of legacy `i32m1` route authority.
- No unrelated memory, mask, compare/select, reduction, or dispatch rewrites
  outside the low-precision primitive-surface campaign boundary.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  archived Gearbox campaign PRD/check context,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVLowPrecisionPerformancePolicy.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Packed-i4/dequantize/dequant-clamp fixtures are bounded pressure-test
  references only. They must not become route authority.
