# RVV production-kernel packed low-precision performance repair campaign

## Goal

Start a macro RVV production-kernel performance repair campaign for the packed
low-precision contraction path. The campaign begins from the measured
packed-i4 same-target no-win/regression evidence and feeds it back into
RVV-owned selected-body/resource/statement/artifact planning, so the compiler
path either carries a more resource-aware packed low-precision structure, fails
closed for stale or unsupported resource facts, or truthfully records that the
current generated RVV artifact has no same-target performance win.

The production path under repair is:

```text
selected/pre-realized packed low-precision product-reduction-dequant body
  -> RVV Gearbox/resource-aware realization facts
  -> RVV route/statement/artifact validation
  -> generated RVV artifact
  -> same-target measurement against scalar-c-reference/product-reduction-dequant-packed-i4-v1
```

## Campaign Gates

- [x] Gate 1: no-win diagnosis tied to production owners, with at least one
  machine-checkable resource/schedule/statement/artifact fact or validation
  seam identified from the measurement evidence.
- [x] Gate 2: production compiler/source change in RVV plugin-local
  Gearbox/resource-aware selected-body realization, route planning, statement
  planning, or artifact validation that addresses diagnosed packed
  low-precision overhead or fails closed for stale resource/performance facts.
- [x] Gate 3: generated artifact/header/C/correctness evidence showing
  provider-owned packed resource and performance-feedback facts are preserved
  after the production change.
- [x] Gate 4: same-target measurement against the scalar baseline with honest
  win, no-win, or regression reporting.
- [x] Gate 5: production RVV plugin/target-source repair, or a targeted
  fail-closed production diagnosis, for the measured packed-i4
  statement/resource overhead. The changed facts must be provider-owned and must
  not rely on q4/q8 names, artifact metadata, route ids, or Common EmitC
  semantics.
- [x] Gate 6: regenerated artifact/header/C/correctness evidence and
  same-target measurement using the existing Gate 4 contract. Provider feedback
  facts must be updated only when the measured result changes.

## Current Round Slice

This round is the Gate 5 production repair slice, with Gate 6 remeasurement
required if the generated executable statement path changes. Gates 1 through 4
are already complete: the measured same-target no-win/regression was mapped to
the packed-i4 resource-selection seam, provider-owned no-win feedback facts are
carried through selected-body realization and provider/statement/target
validation, generated artifact/header/C/correctness evidence proves those facts
survive as mirrors after target artifact validation, and real `ssh rvv` timing
classified the current artifact as a regression.

The current slice must make one coherent production compiler/source change in
the RVV plugin-local Gearbox/resource selected-body realization,
direct-contraction statement planning, or target validation path for the
selected packed-i4 product-reduction candidate. The preferred repair is to
reduce the generated statement/resource overhead that caused the Gate 4
regression. If no safe statement repair is possible, the compiler path must
fail closed with a precise provider-owned production reason explaining why the
current selected-body route must not claim executable performance maturity.

The generated RVV artifact under repair remains the
`widening_product_reduce_dequantize_f32` pre-realized selected-body fixture
`test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`.
The scalar comparator/oracle remains
`scalar-c-reference/product-reduction-dequant-packed-i4-v1`, selected only after
the generated object/header bundle metadata validates provider-owned
`packed-i4-nibbles` facts. If the executable statement path changes, Gate 6 must
rerun the same-target timing contract on `ssh rvv`, record target profile,
compile flags, raw timing records, parsed summaries, and classify the result as
win, no-win, or regression. Acceptance is honest evidence, not a speedup claim
by default.

## Repository Findings

- The archived resource-realization campaign completed Gate 4 and measured the
  packed-i4 generated RVV artifact on `ssh rvv` against
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`.
- The per-op evidence at
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
  records `packed_i4_resource_metadata_selected = true` and the packed scalar
  reference oracle derived from provider-owned `packed-i4-nibbles` metadata.
- Raw timing stdout records all 12 parsed `SUMMARY` lines below 1.0
  best-speedup, from `0.761006` to `0.807006`, so the current production
  artifact is no-win/regression versus the scalar C packed-i4 baseline.
- The current packed fixture explicitly selects the packed-i4 candidate through
  `tcrv_rvv.low_precision_resource.selected_candidate`; the candidate carries
  packed nibble form, sign-extension intent, unroll 1, two realized `vsetvl`
  regions, peak live vector groups 7 after the product-pair-sum repair, product
  region 1, and dequant region 2.
- `RVVLowPrecisionContractionResourceSelection` is already consumed by route
  family planning, direct contraction statement planning, and target artifact
  validation. That is the correct production owner seam for the first no-win
  feedback loop.

## Requirements

- Keep the macro task active after this slice if Gate 4 still reports no-win or
  regression and the PRD identifies the next production repair owner for Hermes
  review.
- Do not introduce q4/q8/llama.cpp, benchmark-name, route-id, artifact-name,
  descriptor, source-front-door, Common EmitC, or exact intrinsic spelling as
  route authority.
- Map the measured no-win/regression to RVV-owned packed low-precision resource
  facts, not to the scalar baseline implementation or measurement harness.
- Do not add a new production compiler owner in this measurement-only slice
  unless the measurement infrastructure exposes a focused production blocker;
  otherwise harden only the bounded evidence tooling needed for Gate 4.
- If adding performance feedback facts, they must be provider-owned mirrors
  consumed by statement and target validation, must truthfully encode no-win
  repair-required state, and must fail closed on stale/missing packed-i4
  mirrors.
- Preserve existing correctness/artifact evidence routes; do not claim a
  performance win while the measured result is regression and the provider-owned
  feedback action remains `no-win-repair-required-before-performance-claim`.
- Gate 5 must touch a production RVV plugin/target owner. A slice that only
  edits scripts, reports, generated evidence, PRD text, or measurement
  formatting is insufficient.
- If a production statement repair changes generated executable code, Gate 6
  must regenerate packed-i4 artifact/header/C evidence and rerun same-target
  measurement before changing provider-owned performance feedback fields.
- If this slice fail-closes instead of repairing statements, the fail-closed
  reason must be provider-owned, target-validated, and precise about the
  statement/resource overhead that blocks performance maturity.

## Acceptance Criteria

- [x] PRD and task context identify this as a macro campaign and this round as
  Gate 4 same-target measurement evidence, with Gates 1 through 3 already
  complete.
- [x] Generated-bundle dry-run evidence for the packed-i4 fixture proves the
  four provider-owned no-win feedback fields are preserved in bundle/object/header
  metadata mirrors after target artifact validation.
- [x] Generated header evidence explicitly shows the four no-win feedback fields
  as mirror comments and keeps the runtime-callable C prototype declaration-only.
- [x] Emitted RVV C/C++ evidence shows packed low/high signed-i4 nibble unpack,
  two widening product/reduction steps, and final dequant/store structure
  derived from the packed resource facts.
- [x] Correctness evidence is limited to the generated harness/oracle or dry-run
  scope unless a real `ssh rvv` run is performed; the packed external ABI harness
  must select the signed low/high i4 oracle from validated packed metadata.
- [x] Focused negative coverage proves stale or missing packed-i4 performance
  feedback metadata fails closed before generated-bundle evidence is accepted.
- [x] Same-target `ssh rvv` measurement evidence records the generated packed-i4
  RVV artifact, scalar packed-i4 baseline identity, shared input sizes/scales,
  correctness guards, target profile, compile flags, raw timing records, parsed
  summaries, and a structured win/no-win/regression classification.
- [x] Measurement evidence ties the classified result back to provider-owned
  packed-i4 feedback/resource facts, including the feedback action that blocks
  any performance-win claim without a new production repair and timing run.
- [x] Focused script/lit/self-test checks pass for the touched measurement
  evidence surfaces, including dry-run not-measured behavior and packed-i4
  metadata-selected baseline/oracle behavior.
- [x] Run `git diff --check`, `git diff --cached --check`, and a bounded
  old-authority scan over touched files/added diff lines.
- [x] Update the task PRD/journal with completed and remaining campaign gates.
- [x] Create one coherent commit for the Gate 4 slice. If the result is still
  no-win or regression, keep the macro task active only if the PRD identifies a
  remaining production repair milestone; otherwise close the campaign gates
  truthfully without claiming a performance win.
- [x] Gate 5 production compiler/source repair changes a provider-owned RVV
  Gearbox/resource selected-body, direct-contraction statement-plan, or target
  validation owner, or encodes a precise fail-closed production reason for the
  measured statement/resource overhead.
- [x] The packed-i4 generated statement evidence changes, or the compiler emits
  a targeted fail-closed diagnostic that prevents the current regressed route
  from being treated as performance mature.
- [x] Target validation remains a consumer of provider-owned facts and rebuilt
  route payloads. Common EmitC, artifact metadata, route ids, fixture names, and
  q4/q8 labels must not become semantic authority.
- [x] Regenerated packed-i4 artifact/header/C/correctness evidence shows the
  repaired provider-owned statement/resource facts or the fail-closed
  diagnostic.
- [x] If executable code changed, same-target `ssh rvv` measurement reruns with
  the Gate 4 contract and provider feedback facts are updated only if the
  measured result changes.

## Definition of Done

- Production compiler/source changes, if any, are implemented in the
  C++/MLIR/LLVM stack. This Gate 5 slice changes RVV plugin-local statement
  planning, resource facts, and target validation owners.
- Tests or focused checks exercise the changed evidence/tooling surface and the
  existing provider-owned feedback path.
- Trellis task files and journal record the campaign state and continuation
  point.
- Worktree is clean after commit.

## Out of Scope

- No q4/q8-named route authority.
- No llama.cpp wrapper or benchmark-name owner.
- No adjacent generated-bundle evidence closeout without production owner
  change.
- No new standalone MAcc, mask, broadcast, dtype/LMUL clone, or source-front
  door positive route.
- No broad autotuning database or dashboard/report-only milestone.
- No Common EmitC invention of RVV semantics.
- No scalar baseline optimization masquerading as RVV compiler progress.
- No performance-win claim without same-target measurement after the production
  change.

## Technical Approach

The current slice uses `scripts/rvv_generated_bundle_same_target_measure.py` as
the bounded Gate 4 measurement tool. The script first regenerates and validates
the generated RVV object/header through the generated-bundle ABI e2e path, then
builds an external C harness that calls both the generated artifact and the
scalar packed-i4 reference on the same `ssh rvv` target. The measurement harness
uses `clock_gettime(CLOCK_MONOTONIC_RAW)`, records raw `MEASURE` lines and
parsed `SUMMARY` lines, and runs correctness guards before timing.

This round repairs the provider-owned packed-i4 statement shape first, then
uses the same script to regenerate evidence and rerun same-target measurement.
The script must remain evidence tooling only: it must not infer RVV semantics,
broaden into a benchmark framework, or let q4/q8 names, benchmark names,
artifact names, route ids, or Common EmitC become route authority.

## Current Round Result

Completed:

- Gate 5 production repair: the packed-i4 direct-contraction statement planner
  no longer performs a low-nibble widening reduction before the high-nibble
  path. The repaired provider-owned statement shape is packed lhs/rhs loads,
  low/high signed nibble sign-extension for both operands, low and high
  `__riscv_vwmul_vv_i16mf2` widening products, one
  `__riscv_vadd_vv_i16mf2` product-pair sum, one
  `__riscv_vwredsum_vs_i16mf2_i32m1` reduction from the pair-sum into
  `dot_acc_vec`, and final dequant/store. This directly changes the RVV
  plugin-local statement owner and not Common EmitC.
- Gate 5 provider/resource facts: the packed-i4 realization decision is now
  `consume-low-precision-packed-i4-product-pair-sum-single-reduce-budget-7of32.v1`
  with `realized_peak_live_vector_groups = 7`. The increased peak-live mirror
  is intentional because the low and high i16 product vectors are live until the
  pair-sum statement.
- Gate 5 target validation: target artifact validation now consumes the rebuilt
  provider route payload for the packed-i4 high product, product-pair add,
  single reduction input/result, and final carry assignment. Stale metadata,
  stale pair-sum operands, stale reduction input/result, or stale carry source
  fail closed before artifact acceptance.
- Gate 6 generated evidence: regenerated dry-run evidence lives under
  `artifacts/tmp/gate5-packed-i4-product-pair-sum-repair/packed-i4-gate5-product-pair-sum`.
  The emitted C contains `__riscv_vadd_vv_i16mf2`; evidence JSON records
  `product_pair_sum_vector` and `single_reduction_vector`; bundle/index/header
  mirrors carry the provider-owned resource decision, peak-live value, packed
  metadata, and no-win feedback facts.
- Gate 6 same-target measurement: real same-target measurement reran under
  `artifacts/tmp/gate6-same-target-measurement/gate6_packed_i4_product_pair_sum_same_target_measure_ssh`
  with the existing Gate 4 contract. It produced 12 summaries and 60
  measurements, classified the repaired artifact as `regression`, and measured
  `best_speedup_range = 0.688427..0.705724`. The provider feedback range is
  updated to that value; `performance_feedback` remains
  `same-target-packed-i4-no-win.v1`, `performance_action` remains
  `no-win-repair-required-before-performance-claim`, and
  `performance_win_claim_allowed = false`.

Remaining:

- Gate 5 and Gate 6 are truthful and complete for this macro campaign slice.
  The measured result remains below 1.0, so no performance-win claim is allowed.
  Any future work should be a new direction/owner for another production repair
  or a targeted fail-closed performance-maturity policy, not a continuation of
  an unchecked Gate 5/6 item in this task.

Continuation point:

None for this Trellis task after commit/archive. The next campaign, if opened,
should start from the Gate 6 regression evidence above and choose a fresh
production owner, likely resource scheduling or a performance-maturity
fail-closed policy for packed low-precision routes.

## Technical Notes

- Direction source: Hermes Direction Brief in the user prompt for this session.
- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived campaign read:
  `.trellis/tasks/archive/2026-06/06-09-rvv-production-kernel-packed-resource-realization-campaign/prd.md`.
- Evidence read:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`
  and `remote_measure_run_stdout.txt`.
- First code owners inspected:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`.

## Open Questions

No blocking user question is needed. The Direction Brief provides the owner,
gates, boundaries, non-goals, and expected first slice.
