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

Gates 1, 2, and 3 are complete for the current production compiler path:
Gearbox/resource-aware selected-body realization now consumes the explicit
primitive-chain resource facts added in Gate 1 and uses them to materialize or
validate legal low-precision contraction structure before route planning,
Common EmitC, or target export can accept the body. Route-family planning,
statement planning, provider mirrors, and target artifact validation now all
consume the same provider-owned low-precision resource/realization facts and
reject stale mirrors before artifact acceptance. The task remains active until
all campaign gates below are complete.

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
- [x] Gate 3: route/statement planning and target artifact validation mirror
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

## Completed Gate 3 Second Slice

Completed the Gate 3 route/provider mirror completion slice:

- [x] Confirmed production route-family planning/provider verification already
  consumes the selected low-precision resource selection and rejects description
  mirrors that diverge from the validated family plan.
- [x] Confirmed production target artifact validation already compares
  provider-owned resource/realization facts against candidate metadata mirrors.
- [x] Added focused provider-level positive coverage that grouped and packed-i4
  low-precision product-reduction provider plans carry realization schedule and
  selected target capability mirrors before statement planning.
- [x] Added focused provider-level stale-fact coverage for realization decision
  mirrors, selected target capability mirrors, missing selected candidates,
  primitive-chain kind, product region facts, and resource budget pressure.
- [x] No production source change was required in this second slice because the
  inspected production owners already implemented the Gate 3 route/provider/
  target validation contract after the first slice and earlier Gate 1/2 work.
- [x] Leave the Trellis macro task active after the slice because Gate 4 and
  Gate 5 remain open.

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

- [x] Route-family planning, statement planning, provider mirrors, and target
  artifact validation consume the same selected primitive-chain/resource/
  realization facts for the low-precision product-reduction contraction path.
- [x] Missing or stale selected candidate, realization producer/decision,
  realized resource budget, product/dequant region facts, primitive-chain facts,
  provider mirror, or target-validation mirror fails closed before artifact
  acceptance.
- [x] Focused positive coverage proves route/statement planning and target
  artifact validation consume provider-owned facts, not route ids, artifact
  names, Common EmitC, status fields, or measurement scripts.
- [x] Focused negative coverage proves stale or missing realization/resource
  facts are rejected.
- [x] No q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, or one-fixture
  evidence-only work is introduced.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if provider or
  statement-planning logic changes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target mirror
  validation changes.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Gate 4 Acceptance Criteria

- [ ] Generated artifact bundle evidence for the selected low-precision
  product-reduction/dequantization path carries the Gate 1-3 primitive-chain,
  resource, realization schedule, provider route, target capability, artifact
  ABI, and target validation facts as mirrors after provider route construction.
- [ ] Same-target measurement evidence records the generated object/header
  identity, correctness-before-timing harness, scalar C baseline identity,
  timing method, target profile, measurement classification, and provider
  feedback tie-back without making measurement JSON, artifact names, q4/q8/
  llama labels, route ids, or status fields route authority.
- [ ] Packed-i4 measurement support is selected only from provider-owned
  low-precision resource mirrors and records packed input semantics, reference
  oracle, runtime `n` unit, primitive-chain facts, realization schedule facts,
  target capability mirrors, performance maturity mirrors, and policy-readiness
  denial/allowance separately from executable correctness.
- [ ] Missing or stale primitive-chain resource facts, realization schedule
  facts, provider feedback facts, target capability facts, artifact/header ABI
  facts, or measurement outcome facts fail closed before a same-target
  measurement record can be accepted.
- [ ] Focused dry-run/script coverage proves the changed evidence path carries
  these facts and rejects stale facts. Real `ssh rvv` evidence is required
  before claiming runtime correctness or performance; dry-run evidence must
  remain classified as `not-measured`.
- [ ] No q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, dispatch policy
  change, or performance-win claim is introduced.
- [ ] `scripts/rvv_generated_bundle_same_target_measure.py --self-test` passes.
- [ ] Focused generated-bundle same-target measurement dry-run passes.
- [ ] `git diff --check` and `git diff --cached --check` pass.

## Current Gate 4 Slice

This round covers the generated artifact / same-target measurement evidence
plumbing sub-slice:

- [x] Make the generated-bundle checker and same-target measurement record
  require the low-precision primitive-chain resource mirrors emitted by the
  production provider/target path.
- [x] Carry primitive-chain, resource, realization, provider, target
  capability, object/header ABI, correctness-before-timing, and measurement
  classification facts into the Gate 4 evidence JSON.
- [x] Add focused stale-fact coverage for primitive-chain and performance/
  measurement tie-back fields.
- [x] Leave Gate 4 open if real `ssh rvv` measurement is not run in this
  slice, and leave Gate 5 unopened until accepted Gate 4 measurement facts
  exist.

Slice result: completed the generated artifact and same-target measurement
evidence plumbing for dry-run/not-measured Gate 4 records. The Gate 4 macro
acceptance criteria above remain open because this slice did not run real
`ssh rvv` correctness/timing and therefore did not create accepted measured
win/no-win/regression facts for Gate 5.

## Out of Scope

- No new q4/q8/llama route ids, artifact names, helper semantics, or wrapper
  owners.
- No high-level Linalg/Vector/StableHLO frontend work.
- No one-op intrinsic wrapper or dtype/LMUL clone batch.
- No Common EmitC invention of RVV dtype, schedule, primitive, resource,
  performance, or dispatch semantics.
- No dispatch/performance preference change or performance-win claim.
- No archive/finish after this Gate 4 partial slice unless all macro gates are
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

Gates 1, 2, and 3 are complete. The first Gate 4 generated artifact /
same-target measurement evidence-plumbing sub-slice is complete. Gate 4 remains
open because real `ssh rvv` correctness/timing evidence was not run in this
slice and the dry-run records remain classified as `not-measured`. Gate 5
remains unopened, so this macro task stays active.

Next owner: continue Gate 4 by running or accepting real same-target `ssh rvv`
correctness/timing evidence for the generated low-precision primitive/resource
artifact path, then record truthful measured win/no-win/regression facts. Gate 5
dispatch/performance policy must wait for those accepted Gate 4 measurement
facts.
