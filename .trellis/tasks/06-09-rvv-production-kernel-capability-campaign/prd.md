# RVV production-kernel capability campaign

## Goal

Advance the RVV Stage 2 production compiler path for low-precision contraction
kernels beyond repeated generated-bundle evidence. This macro campaign keeps
one Trellis owner for the production capability path:

```text
selected/pre-realized low-precision tcrv_rvv body
  -> RVV-owned primitive and resource facts
  -> Gearbox/resource-aware selected-body realization
  -> provider-owned route and statement planning
  -> target artifact mirror validation
  -> changed-path artifact/measurement feedback
  -> dispatch/performance policy
```

Gate 1 is complete. Gate 2 is complete for the current production source slice:
Gearbox/resource-aware selected-body realization now consumes the explicit
primitive-chain resource facts added in Gate 1 and uses them to materialize or
validate legal low-precision contraction structure before route planning,
Common EmitC, or target export can accept the body. The task remains active
until all campaign gates below are complete.

## What I already know

- The previous packed-i4 performance-maturity feedback campaign is archived at
  `.trellis/tasks/archive/2026-06/06-09-rvv-packed-low-precision-performance-maturity-feedback-campaign/`.
- That campaign completed provider/target/script maturity policy and proved the
  packed-i4 path remains executable but not performance-mature under same-target
  regression evidence.
- The archived campaign explicitly says no production source change was required
  for its final Gate 5; therefore this new campaign must not be another
  evidence-only or metadata-only closeout.
- `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require Stage 2 work to
  advance resource-aware selected-body realization, typed low-precision
  coverage, or measured same-target evidence only when the executable path
  changes.
- `RVVLowPrecisionContractionResourceSelection` already carries resource shape,
  packed/unpack, schedule, realization, performance-maturity, and target
  capability mirror fields.
- `RVVSelectedBodyContractionRouteFamilyPlan` already carries low-precision
  primitive fields, including primitive contract/kind, source/product,
  accumulator/result, product relation, reduction intrinsic, seed splat,
  accumulator layout, result layout, and store VL.
- Gate 1 repaired the first production bottleneck: low-precision resource
  selection now carries the primitive-chain contract as RVV-owned resource
  facts, and those facts flow through resource selection, Gearbox pass facts,
  selected-body realization validation, route planning, provider metadata,
  target support metadata, and target artifact validation.
- Gate 2 repaired the next production bottleneck: selected-body realization now
  consumes those primitive-chain resource facts when it materializes and
  validates low-precision contraction structure. A realized body is rejected if
  realization ignores, drops, or mismatches the resource-owned primitive chain
  that route/target mirrors later claim.

## Requirements

- Add the smallest reusable provider-owned resource surface that connects
  low-precision resource selection to the primitive chain for product-reduction
  contraction families.
- The surface must be family-level, not tied to q4, q8, llama.cpp, artifact
  names, route ids, or one fixture.
- The facts must be derived by RVV plugin/resource/plan owners from typed
  selected-body facts and selected resource candidates.
- The facts must include enough chain identity to fail closed on stale source,
  product, accumulator, result, widening-product relation, product-reduction
  relation, widening-product intrinsic, widening-reduction intrinsic, scalar
  seed splat, reduction layout, result layout, and reduction store VL claims.
- Common EmitC must only carry provider-built route payloads and metadata
  mirrors; it must not infer low-precision primitive/resource semantics.
- Target artifact validation must compare candidate mirrors against
  provider-owned facts before accepting target artifacts.
- Existing executable correctness support and current packed-i4 maturity policy
  must remain intact.
- Gearbox selected-body realization must consume the Gate 1 primitive-chain
  resource facts from the same selected resource candidate that governs the
  low-precision contraction realization.
- Realization must fail closed on missing or stale resource-owned primitive
  chain facts before route/provider/Common EmitC/export acceptance.
- The realized low-precision contraction structure must remain derived from
  typed body/config/resource facts. It must not infer legality from q4/q8/llama
  names, route ids, artifact names, exact intrinsic spellings, status fields, or
  Common EmitC behavior.

## Macro Campaign Gates

- [x] Gate 1: production low-precision primitive/resource surface is explicit
  and RVV-owned for the packed/low-precision contraction family, with
  fail-closed validation for stale dtype, pack/unpack, widening product,
  reduction, dequant, resource, schedule, or ABI facts.
- [x] Gate 2: Gearbox/resource-aware selected-body realization consumes those
  facts to materialize legal low-precision contraction structure without
  changing compute semantics or using route/artifact names as authority.
- [ ] Gate 3: route/statement planning and target artifact validation mirror
  provider-owned facts and reject stale or unsupported performance/resource
  claims.
- [ ] Gate 4: generated artifact correctness and same-target measurement run
  only for the changed production path.
- [ ] Gate 5: dispatch/performance policy consumes measurement outcome
  truthfully without promoting stale/no-win/regression evidence into route
  authority or win claims.

## Completed Gate 2 Slice

Completed Gate 2 as a production source slice:

- [x] Connect the selected resource primitive-chain contract into the
  Gearbox/resource-aware selected-body realization owner for low-precision
  product-reduction contraction families.
- [x] Make the owner materialize or validate realized low-precision contraction
  structure from typed body/config/resource facts and the selected resource
  candidate's primitive-chain facts.
- [x] Reject missing or stale realization/resource facts before route/provider
  planning, Common EmitC, or target artifact export can accept the realized
  body.
- [x] Add focused positive coverage that realization consumes the primitive-chain
  resource facts.
- [x] Add focused fail-closed coverage for a stale or missing primitive-chain
  realization/resource fact.
- [x] Leave the Trellis macro task active after the slice unless all gates are
  completed by later work.

## Completed Gate 3 First Slice

Completed the first Gate 3 production source slice:

- [x] Require low-precision product-reduction route/statement planning to
  consume the selected Gearbox realization resource facts before statement
  construction.
- [x] Require target artifact provider validation to reject missing or stale
  low-precision realization/resource facts before accepting route statements or
  candidate metadata mirrors.
- [x] Keep the facts provider-owned: selected candidate, realization producer,
  realization decision, realized unroll, realized VL-region count, realized
  peak-live estimate, product/dequant region ordering, product/dequant phases,
  primitive chain, target capability mirrors, and performance maturity mirrors
  stay RVV resource/provider facts.
- [x] Add focused positive/fail-closed C++ coverage for statement planning and
  target artifact validation. The tests must prove validation derives from
  provider resource/realization facts, not artifact names, route ids, Common
  EmitC inference, or measurement output.
- [x] Leave the Trellis macro task active after the slice unless Gates 3, 4,
  and 5 are all complete.

## Gate 1 Acceptance Criteria

- [x] Production code carries explicit low-precision primitive-chain resource
  facts in the RVV-owned resource selection surface.
- [x] Derived selection, pass fact ingestion, route-family plan validation, route
  description validation, equality comparison, and target artifact validation
  consume the new facts.
- [x] Missing or stale primitive-chain resource facts fail closed before Common
  EmitC or target artifact export can accept the route.
- [x] Focused tests cover the changed owner and at least one stale chain fact.
- [x] No q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, or one-fixture
  evidence-only work is introduced.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if provider logic
  changes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation changes.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Gate 2 Acceptance Criteria

- [x] Gearbox/resource-aware selected-body realization consumes the
  provider-owned low-precision primitive-chain resource facts from the selected
  resource candidate.
- [x] Realization-produced low-precision contraction facts are matched against
  the selected resource primitive-chain contract before route/provider/Common
  EmitC/export acceptance.
- [x] Missing, stale, or mismatched primitive-chain realization/resource facts
  fail closed with focused diagnostics at the RVV owner boundary.
- [x] Focused positive coverage proves the changed owner consumes the
  primitive-chain facts, not route ids, artifact names, status fields, or Common
  EmitC behavior.
- [x] Focused negative coverage proves stale or missing realization/resource
  facts are rejected.
- [x] No q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, or one-fixture
  evidence-only work is introduced.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if provider or
  realization logic changes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target mirror
  validation changes.
- [x] Relevant FileCheck/lit or focused manual FileCheck checks cover changed
  diagnostics or realized IR when text-visible behavior changes.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Gate 3 Acceptance Criteria

- [ ] Route-family planning, statement planning, provider mirrors, and target
  artifact validation consume the same selected primitive-chain/resource/
  realization facts for the low-precision product-reduction contraction path.
- [ ] Missing or stale selected candidate, realization producer/decision,
  realized resource budget, product/dequant region facts, primitive-chain facts,
  provider mirror, or target-validation mirror fails closed before artifact
  acceptance.
- [ ] Focused positive coverage proves route/statement planning and target
  artifact validation consume provider-owned facts, not route ids, artifact
  names, Common EmitC, status fields, or measurement scripts.
- [ ] Focused negative coverage proves stale or missing realization/resource
  facts are rejected.
- [ ] No q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, or one-fixture
  evidence-only work is introduced.
- [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if provider or
  statement-planning logic changes.
- [ ] `build/bin/tianchenrv-target-artifact-export-test` passes if target mirror
  validation changes.
- [ ] `git diff --check` and `git diff --cached --check` pass.

## Out of Scope

- No generated-bundle or same-target rerun unless this slice changes the
  executable/performance path and makes a runtime/correctness/performance claim.
- No new q4/q8/llama route ids, artifact names, helper semantics, or wrapper
  owners.
- No high-level Linalg/Vector/StableHLO frontend work.
- No one-op intrinsic wrapper or dtype/LMUL clone batch.
- No Common EmitC invention of RVV dtype, schedule, primitive, resource,
  performance, or dispatch semantics.
- No dispatch/performance preference change or performance-win claim.
- No archive/finish after this Gate 3 slice unless all macro gates are also
  complete.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Production files to inspect/change as needed:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Focused tests are expected under the existing RVV extension plugin and target
  artifact test owners, plus lit/FileCheck only where user-visible metadata or
  diagnostics change.

## Continuation Point

Gate 1 and Gate 2 are complete. The first Gate 3 source slice is complete, but
Gate 3 remains open for any follow-up broadening that Hermes selects before
artifact measurement. Gates 4 and 5 remain open, so this macro task stays
active.

Next owner: either continue Gate 3 only if another route/provider/target mirror
surface is identified, or advance to Gate 4 generated artifact correctness and
same-target measurement for the changed production path. Gate 5 dispatch/
performance policy must wait for Gate 4 measurement facts.
