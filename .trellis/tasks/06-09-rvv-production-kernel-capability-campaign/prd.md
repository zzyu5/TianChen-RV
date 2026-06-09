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

The current round completes Gate 1 only: make the first concrete production
low-precision primitive/resource bottleneck explicit and repair it in production
owners. The task remains active until all campaign gates below are complete.

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
- The current production bottleneck is that the low-precision resource
  selection does not carry the primitive-chain contract as resource-owned facts.
  Primitive facts and resource facts can therefore be validated separately
  without proving the same selected resource candidate owns the widening-product
  and widening-reduction chain that route/target mirrors later claim.

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

## Macro Campaign Gates

- [x] Gate 1: production low-precision primitive/resource surface is explicit
  and RVV-owned for the packed/low-precision contraction family, with
  fail-closed validation for stale dtype, pack/unpack, widening product,
  reduction, dequant, resource, schedule, or ABI facts.
- [ ] Gate 2: Gearbox/resource-aware selected-body realization consumes those
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

## Current Round Slice

Complete Gate 1 as a production source slice:

- Extend `RVVLowPrecisionContractionResourceSelection` and its candidate/pass
  fact surface with a provider-owned primitive chain contract for low-precision
  product-reduction contraction families.
- Populate those facts from the same typed plan/resource candidate facts that
  already select the low-precision direct-contraction resource candidate.
- Require pass-produced and selected-body-realization-produced facts to match
  the derived provider contract before route acceptance.
- Thread the facts through provider route descriptions and target artifact
  mirror validation where the low-precision resource selection is consumed.
- Add focused fail-closed coverage for a stale primitive/resource chain fact.
- Leave the Trellis macro task active after the slice unless all gates are
  completed by later work.

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
- No archive/finish after Gate 1 unless all macro gates are also complete.

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

Gate 1 is complete. This round made the primitive-chain contract part of the
RVV-owned low-precision resource surface and threaded it through Gearbox pass
facts, selected-body realization validation, route-family planning, provider
metadata, target support metadata, and target artifact validation.

Continue with Gate 2: make Gearbox/resource-aware selected-body realization
consume the explicit primitive-chain resource facts to materialize legal
low-precision contraction structure, still without making route ids, artifact
names, Common EmitC, or measurement scripts the authority.
