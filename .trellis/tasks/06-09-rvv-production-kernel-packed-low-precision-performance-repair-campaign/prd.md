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

## Current Round Slice

This round is the Gate 4 same-target measurement slice. Gates 1 through 3 are
already complete: the measured same-target no-win/regression was mapped to the
packed-i4 resource-selection seam, the production compiler path now carries
provider-owned performance feedback facts through selected-body realization and
provider/statement/target validation, and generated artifact/header/C/correctness
evidence proves those facts survive as mirrors after target artifact validation.

The current slice reruns same-target timing for the packed-i4 fixture already
used by the campaign. The generated RVV artifact under test is the
`widening_product_reduce_dequantize_f32` pre-realized selected-body fixture
`test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`.
The scalar comparator/oracle is
`scalar-c-reference/product-reduction-dequant-packed-i4-v1`, selected only after
the generated object/header bundle metadata validates provider-owned
`packed-i4-nibbles` facts. Correctness must pass before each timed case. Timing
must run generated TianChen-RV output and the scalar baseline on the same
`ssh rvv` target with the same ABI/input semantics, record target profile,
compile flags, raw timing records, parsed summaries, and classify the result as
win, no-win, or regression. Acceptance is honest evidence, not a speedup claim by
default.

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
  regions, peak live vector groups 6, product region 1, and dequant region 2.
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

## Definition of Done

- Production compiler/source changes, if any, are implemented in the
  C++/MLIR/LLVM stack. This Gate 4 slice intentionally changes no C++ production
  owner because it only hardens and executes measurement evidence.
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

This round may minimally harden that script so evidence JSON contains a
machine-checkable result classification and provider-feedback tie-back. It must
not broaden into a benchmark framework, infer RVV semantics in Python, or let
q4/q8 names, benchmark names, artifact names, route ids, or Common EmitC become
route authority.

## Current Round Result

Completed:

- Gate 1: the measured same-target no-win/regression is tied to the packed-i4
  resource-selection boundary, not to a benchmark name or artifact label. The
  production seam is `RVVLowPrecisionContractionResourceSelection` flowing
  through Gearbox attrs, selected-body realization, route-family planning,
  statement planning, route metadata, target artifact validation, and generated
  bundle metadata mirrors.
- Gate 2: the production path now carries provider-owned packed-i4 performance
  feedback facts:
  `same-target-packed-i4-no-win.v1`,
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`,
  `0.761006..0.807006`, and
  `no-win-repair-required-before-performance-claim`. Stale packed-i4 feedback
  fails closed before target artifact acceptance.
- Gate 3: generated-bundle dry-run evidence now proves those feedback facts are
  preserved in generated bundle object/header metadata mirrors, generated header
  comments, emitted RVV C/C++ packed low/high nibble statements, evidence JSON
  `widening_product_reduction_boundary.low_precision_resource` summary fields,
  and the packed external ABI correctness oracle selected from validated
  `packed-i4-nibbles` metadata. The verifier self-test rejects missing or stale
  packed-i4 performance feedback metadata before accepting evidence.
- Gate 4: same-target `ssh rvv` timing now reruns the generated packed-i4 RVV
  artifact against `scalar-c-reference/product-reduction-dequant-packed-i4-v1`
  with shared input sizes `257,4096,65536`, scales `-0.125,0.375`, warmups `2`,
  repeats `5`, iterations `8`, compile flags `-O2 -march=rv64gcv -mabi=lp64d
  -I.`, `CLOCK_MONOTONIC_RAW`, and correctness guards before timing. Evidence
  lives under
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh_current`.
  The root evidence is `evidence.json`; per-op evidence is
  `widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`;
  raw target profile is `remote_target_profile_stdout.txt`; raw timing stdout is
  `remote_measure_run_stdout.txt`.
- Gate 4 classification: all 12 parsed `SUMMARY` records are below 1.0
  best-speedup, so the structured classification is `regression` with
  `best_speedup_range = 0.761468..0.804008`. The provider feedback tie-back is
  `consistent-with-current-no-win-feedback`, the baseline identity is
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1`, and
  `performance_win_claim_allowed = false`.

Remaining:

- The campaign gates are now completed as evidence gates, but the macro task
  remains active per the continuation rule because Gate 4 still reports a
  regression. The next unfinished production milestone is a real RVV
  plugin-local repair of the selected packed-i4 product-reduction path, not
  another evidence-only closeout.

Continuation point:

Start the next slice from the measured regression evidence under
`artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh_current`.
The production repair owner is the RVV plugin-local
Gearbox/resource/statement-planning path for the selected packed-i4
product-reduction candidate. It should reduce the packed-i4 overhead in the
generated statement plan or fail closed with a specific resource/statement
reason before any future performance-win claim. After that production repair,
rerun the same Gate 4 measurement contract and update the provider-owned
performance feedback facts if the result changes.

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
