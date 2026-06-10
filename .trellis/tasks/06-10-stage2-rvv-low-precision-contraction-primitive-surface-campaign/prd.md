# Stage2 RVV Low-Precision Contraction Primitive-Surface Campaign

## Goal

Create one macro owner for the RVV Stage 2 low-precision contraction
primitive surface. The campaign must make low-precision operand element
facts, typed vector/config facts, widening product facts, accumulator/reduction
facts, selected-body realization facts, route/provider legality, target
validation, and later same-target policy consumption flow through production
RVV-owned compiler surfaces.

The current round implements Gate 1 only. It must add or harden production
compiler behavior for typed low-precision contraction primitive facts and
fail-closed diagnostics before stale metadata, q8/q4 names, packed-i4 labels,
descriptor residue, generated C strings, route ids, or artifact names can stand
in for typed body/config facts.

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
- Gate 1 must change or harden production compiler/validation code. PRD-only,
  journal-only, generated-bundle dry-run-only, broad smoke-only, or evidence
  packaging-only work does not satisfy the slice.
- Primitive authority must come from typed `tcrv_rvv` body/config/provider
  facts. It must not come from q8/q4 labels, llama.cpp names, packed-i4
  artifact labels, descriptor residue, generated C strings, route ids,
  artifact names, Common EmitC, status fields, or metadata mirrors.
- The Gate 1 surface must expose and validate the smallest coherent production
  subset of low-precision contraction facts that source inspection proves is
  missing or weak. Acceptable minimum scope is i8/u8 operand element facts plus
  widening product/accumulator facts and provider/target fail-closed
  diagnostics.
- Common EmitC must remain neutral. It may consume provider-built route payloads
  that already exist, but it must not invent RVV dtype, SEW/LMUL, widening, or
  contraction semantics.
- Focused tests must prove both the positive typed-body/config fact path and at
  least one stale metadata/name/artifact-label/generated-string authority path
  failing closed before route or artifact acceptance.

## Macro Campaign Gates

- [x] Gate 1: typed low-precision contraction primitive facts and fail-closed
  provider/validation surface exist in production code.
- [ ] Gate 2: RVV plugin-local selected-body realization consumes those facts
  for a representative low-precision contraction/dequant path without changing
  compute semantics.
- [ ] Gate 3: route/provider/artifact export carries those facts into generated
  artifacts and source-backed same-target measurement records.
- [ ] Gate 4: selected-dispatch/performance policy consumes those measurements
  fail-closed, preserving correctness fallback and denying stale/no-win
  performance claims.

## Current Slice: Gate 1

- [x] Inspect the existing RVV dialect/plugin/provider/target-validation
  low-precision contraction surfaces and identify the first production consumer
  or diagnostic gap.
- [x] Add or harden production typed low-precision contraction primitive facts:
  operand element type/config facts, source signedness where relevant,
  widening product facts, and accumulator/result facts.
- [x] Make RVV-owned provider or target validation fail closed when stale
  metadata, q8/q4 labels, packed-i4 artifact labels, descriptor residue, route
  ids, artifact names, or generated strings try to stand in for typed
  body/config facts.
- [x] Add focused positive and negative tests for the changed production
  surface.
- [x] Run focused C++ tests, any directly touched lit tests, whitespace checks,
  bounded old-authority scans, and Trellis validation.
- [ ] Commit the Gate 1 slice and keep `.trellis/.current-task` active because
  Gates 2-4 remain incomplete.

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

## Remaining Macro Gates

- Gate 2: RVV plugin-local selected-body realization must consume these facts
  for a representative low-precision contraction/dequant path without changing
  compute semantics.
- Gate 3: route/provider/artifact export must carry the facts into generated
  artifacts and source-backed same-target measurement records.
- Gate 4: selected-dispatch/performance policy must consume those measurements
  fail-closed while preserving correctness fallback.

## Next Continuation Point

Continue Gate 2 in this same task: inspect selected-body realization for the
representative low-precision contraction/dequant path and make it consume the
typed primitive facts structurally before route construction, without changing
compute semantics, dtype semantics, ABI roles, runtime AVL/VL, variant origin,
dispatch, or fallback behavior.

## Out Of Scope

- No generated-bundle or `ssh rvv` evidence closeout as the default milestone.
- No q8/q4/llama.cpp route ids, wrappers, artifact names, or naming authority.
- No dtype/LMUL clone batch.
- No high-level Linalg, Vector, StableHLO, or per-Linalg authority work.
- No global tuning database or broad dispatch-policy rewrite.
- No Common EmitC invention of RVV semantics.
- No resurrection of legacy `i32m1` route authority.
- No unrelated memory, mask, compare/select, reduction, or dispatch rewrites
  outside the Gate 1 primitive-surface boundary.

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
