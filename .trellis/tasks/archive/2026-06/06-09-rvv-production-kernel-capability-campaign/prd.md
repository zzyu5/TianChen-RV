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
- [x] Gate 4: generated artifact correctness and same-target measurement run
  only for the changed production path.
- [x] Gate 5: dispatch/performance policy consumes measurement outcome
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

- [x] Generated artifact bundle evidence for the selected low-precision
  product-reduction/dequantization path carries the Gate 1-3 primitive-chain,
  resource, realization schedule, provider route, target capability, artifact
  ABI, and target validation facts as mirrors after provider route construction.
- [x] Same-target measurement evidence records the generated object/header
  identity, correctness-before-timing harness, scalar C baseline identity,
  timing method, target profile, measurement classification, and provider
  feedback tie-back without making measurement JSON, artifact names, q4/q8/
  llama labels, route ids, or status fields route authority.
- [x] Packed-i4 measurement support is selected only from provider-owned
  low-precision resource mirrors and records packed input semantics, reference
  oracle, runtime `n` unit, primitive-chain facts, realization schedule facts,
  target capability mirrors, performance maturity mirrors, and policy-readiness
  denial/allowance separately from executable correctness.
- [x] Missing or stale primitive-chain resource facts, realization schedule
  facts, provider feedback facts, target capability facts, artifact/header ABI
  facts, or measurement outcome facts fail closed before a same-target
  measurement record can be accepted.
- [x] Focused dry-run/script coverage proves the changed evidence path carries
  these facts and rejects stale facts. Real `ssh rvv` evidence is required
  before claiming runtime correctness or performance; dry-run evidence must
  remain classified as `not-measured`.
- [x] No q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, dispatch policy
  change, or performance-win claim is introduced.
- [x] `scripts/rvv_generated_bundle_same_target_measure.py --self-test` passes.
- [x] Focused generated-bundle same-target measurement dry-run passes.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Gate 5 Acceptance Criteria

- [x] A production dispatch/performance policy consumer reads accepted Gate 4
  same-target packed-i4 measurement facts through a typed policy contract,
  not through artifact names, route ids, status fields, q4/q8/llama labels, or
  Common EmitC inference.
- [x] The policy consumer ties the accepted measurement outcome back to the
  Gate 1-3 low-precision resource selection, primitive-chain facts,
  realization facts, provider maturity mirrors, target capability mirrors, and
  `ssh rvv` same-target evidence identity.
- [x] Accepted `regression` / `no-win` evidence preserves executable route
  support and correctness execution but denies performance-preferred dispatch
  and performance-win claims.
- [x] Missing, stale, mismatched, or untrusted measurement identity, target
  profile, `ssh rvv` evidence, provider maturity tie-back, selected candidate,
  primitive-chain fact, realization/resource fact, target mirror, or win-claim
  field fails closed before provider/target artifact acceptance.
- [x] Focused provider and target artifact tests prove the policy accepts the
  accepted Gate 4 outcome and rejects stale measurement identity, missing
  `ssh rvv` evidence, stale target profile, stale provider tie-back, stale
  primitive-chain facts, unmeasured packed-i4 candidate reuse, and measurement
  win-promotion attempts.
- [x] No new q4/q8/llama-named route authority, source-front-door authority,
  descriptor-driven computation, Common EmitC semantic branch, broad dashboard,
  performance database, adjacent op family, or generated-bundle-only closeout is
  introduced.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Bounded old-authority scan over touched files and current diff lines is
  reviewed.
- [x] `git diff --check` and `git diff --cached --check` pass.

## Completed Gate 4 Evidence-Plumbing Slice

The first Gate 4 round covered the generated artifact / same-target measurement
evidence plumbing sub-slice:

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
evidence plumbing for dry-run/not-measured Gate 4 records. That first Gate 4
slice did not run real `ssh rvv` correctness/timing and therefore did not by
itself create accepted measured win/no-win/regression facts for Gate 5.

## Completed Gate 4 Real-Measurement Acceptance Slice

Completed the second Gate 4 slice for the selected signed packed-i4 /
low-precision widening product-reduce-dequantize artifact path:

- [x] Ran the non-dry-run same-target measurement path on `ssh rvv` for
  `widening_product_reduce_dequantize_f32` with the packed-i4 pre-realized
  selected-body fixture.
- [x] Preserved generated bundle identity in evidence:
  `artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`,
  `artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`,
  and `tianchenrv-target-artifact-bundle.index`.
- [x] Recorded object/header SHA-256 identity, selected input, selected variant,
  correctness-before-timing harness, packed scalar baseline
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`, timing method
  `clock_gettime(CLOCK_MONOTONIC_RAW)`, and `ssh rvv` target profile.
- [x] Carried provider-owned primitive-chain/resource/realization/target mirrors
  into the measurement evidence, including
  `rvv-low-precision-widening-reduction-primitive-facts.v1`,
  `signed-i8mf4xi8mf4-to-i16mf2-product-i32m1-vwredsum.v1`,
  `__riscv_vwmul_vv_i16mf2`,
  `__riscv_vwredsum_vs_i16mf2_i32m1`,
  `consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1`,
  and the selected RVV target capability legality mirror.
- [x] Accepted the parsed measured outcome as `classification = regression`,
  `outcome_family = no-win`, `best_speedup_range = 0.688889..0.705200`,
  `summary_record_count = 12`, `measurement_record_count = 60`, and
  `correctness_record_count = 12`.
- [x] Preserved provider-owned maturity policy in the measurement tie-back:
  `provider_maturity = executable-not-performance-mature`,
  `provider_maturity_outcome = regression`,
  `provider_performance_selection_eligible = false`,
  `provider_dispatch_preference = not-performance-preferred`,
  `performance_win_claim_allowed = false`,
  `performance_preference_denial_reason =
  same-target-measurement-no-win-or-regression`, and
  `correctness_execution_allowed = true`.
- [x] Kept the result as measurement evidence input only. It does not update
  dispatch policy, does not make a performance-win claim, and does not make
  artifact names, route ids, q4/q8/llama labels, status fields, or measurement
  JSON route authority.

Slice result: Gate 4 is complete for the current accepted packed-i4
representative. The accepted measured fact is a same-target regression/no-win
classification, so Gate 5 must consume it truthfully without promoting the
route to performance-preferred or allowing a win claim.

## Completed Gate 5 Dispatch/Performance Policy Consumption Slice

Completed Gate 5 as a production source slice:

- [x] Added an RVV-owned low-precision performance policy surface for the
  accepted packed-i4 Gate 4 measurement outcome, including measurement identity,
  target profile, record counts, provider maturity tie-back, denial reason, and
  route-support effect.
- [x] Wired the policy into provider route-family validation so selected
  packed-i4 product-reduction/dequantization resource facts must match the
  accepted Gate 4 measurement contract before statement planning can continue.
- [x] Wired the policy into RVV target artifact route-family validation so
  provider descriptions and candidate mirrors cannot accept stale,
  mismatched, or unmeasured packed-i4 performance policy facts.
- [x] Kept the accepted regression/no-win outcome separate from route authority:
  executable route support and correctness execution are preserved, while
  performance-preferred dispatch and win claims remain denied.
- [x] Added focused provider and target artifact coverage for the accepted
  outcome and fail-closed stale/missing/mismatched measurement, provider,
  primitive, target, and candidate facts.

Slice result: Gate 5 is complete for the current accepted packed-i4
representative. The macro campaign gates are complete and the task can be
finished/archived after the focused checks and commit are recorded.

## Out of Scope

- No new q4/q8/llama route ids, artifact names, helper semantics, or wrapper
  owners.
- No high-level Linalg/Vector/StableHLO frontend work.
- No one-op intrinsic wrapper or dtype/LMUL clone batch.
- No Common EmitC invention of RVV dtype, schedule, primitive, resource,
  performance, or dispatch semantics.
- No performance-win claim, performance database, or broad benchmark dashboard.
- No archive/finish before Gate 5 production consumption and focused checks are
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

Gates 1, 2, 3, 4, and 5 are complete for the current accepted packed-i4
representative. Gate 5 production policy consumption now preserves executable
correctness while denying performance-preferred dispatch and performance-win
claims for the accepted `ssh rvv` regression/no-win outcome. The macro task is
ready to finish/archive after the focused checks and final commit are recorded.
