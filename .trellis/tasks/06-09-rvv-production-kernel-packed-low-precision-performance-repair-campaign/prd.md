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
- [ ] Gate 4: same-target measurement against the scalar baseline with honest
  win, no-win, or regression reporting.

## Current Round Slice

This round is the Gate 3 evidence slice. Gate 1 and Gate 2 are already complete:
the measured same-target no-win/regression was mapped to the packed-i4
resource-selection seam, and the production compiler path now carries
provider-owned performance feedback facts through selected-body realization,
route planning, statement planning, route metadata, target support bundles, and
target artifact validation.

The current slice proves those facts survive into generated artifacts. The
evidence must cover the generated bundle index, object/header metadata mirrors,
generated header comments, emitted RVV C/C++ source, packed-i4 external ABI
harness correctness oracle, and a stale or missing feedback fail-closed path.
This is not a new performance claim. Gate 4 same-target timing remains a later
milestone unless a later production compiler change attempts a real performance
repair.

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

- Keep the macro task active after this slice unless Gates 1 through 4 are all
  genuinely complete with committed evidence.
- Do not introduce q4/q8/llama.cpp, benchmark-name, route-id, artifact-name,
  descriptor, source-front-door, Common EmitC, or exact intrinsic spelling as
  route authority.
- Map the measured no-win/regression to RVV-owned packed low-precision resource
  facts, not to the scalar baseline implementation or measurement harness.
- Add or harden one production owner under RVV Gearbox/resource realization,
  route planning, statement planning, or target validation.
- If adding performance feedback facts, they must be provider-owned mirrors
  consumed by statement and target validation, must truthfully encode no-win
  repair-required state, and must fail closed on stale/missing packed-i4
  mirrors.
- Preserve existing correctness/artifact evidence routes; do not claim a
  performance win without new same-target measurement after the production
  change.

## Acceptance Criteria

- [x] PRD and task context identify this as a macro campaign and this round as
  Gate 3 generated artifact/header/C/correctness evidence, with Gate 1 and Gate
  2 already complete.
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
- [x] Relevant focused script/lit/C++ checks pass for the touched evidence
  surfaces; no same-target timing or performance win is claimed in this round.
- [x] Run `git diff --check`, `git diff --cached --check`, and a bounded
  old-authority scan over touched files/added diff lines.
- [x] Update the task PRD/journal with completed and remaining campaign gates.
- [x] Create one coherent commit for the Gate 3 slice while keeping the macro
  task active unless Gate 4 is also complete.

## Definition of Done

- Production compiler/source change is implemented in C++/MLIR/LLVM stack.
- Tests or focused checks exercise the changed owner and fail-closed path.
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

The current slice hardens the generated-bundle evidence path around the
provider-owned packed-i4 no-win feedback contract added in Gate 2. The
generated-bundle verifier and dry-run lit coverage must prove that
`same-target-packed-i4-no-win.v1`,
`scalar-c-reference/product-reduction-dequant-packed-i4-v1`,
`0.761006..0.807006`, and
`no-win-repair-required-before-performance-claim` appear as exact mirrors in
bundle/object/header evidence after RVV provider and target validation. The
same evidence must show the generated RVV C/C++ packed nibble sequence and the
packed external ABI correctness oracle. Missing or stale feedback metadata must
fail closed in the evidence verifier.

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

Remaining:

- Gate 4: rerun same-target packed-i4 timing against
  `scalar-c-reference/product-reduction-dequant-packed-i4-v1` only after a
  later production compiler path change attempts an actual performance repair.

Continuation point:

Start Gate 4 from the Gate 3 generated artifact/header/C/correctness evidence
under `artifacts/tmp/gate3-packed-i4-feedback-evidence/packed-i4-gate3-feedback`
or regenerate it with the checked dry-run command. The next slice should rerun
same-target packed-i4 timing only after deciding whether the RVV-owned
Gearbox/resource/statement path has a real performance repair to measure; if no
repair is attempted, it must continue reporting the provider-owned no-win
feedback honestly and avoid any performance-win claim.

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
