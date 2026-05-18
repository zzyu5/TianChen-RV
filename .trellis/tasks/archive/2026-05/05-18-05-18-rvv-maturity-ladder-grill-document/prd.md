# brainstorm: RVV maturity ladder grill document

## Goal

Record the current read-only grill consensus about the RVV-first maturity
scope before context compaction loses the discussion. This task only creates a
control-plane artifact note under `artifacts/`; it does not update durable
Trellis specs, prompts, compiler code, tests, or active loop steering.

## Scope Update After User Authorization

After the grill converged, the user explicitly authorized the follow-up durable
promotion step in the same working session:

- reread the full grill note and preserve important details;
- update owning specs and the canonical supervisor prompt;
- replace or remove incorrect old expressions instead of only appending new
  text;
- finish the task, keep the worktree clean, add Stage 1 one-shot steering, and
  restart the loop.

This expanded the task from artifact-note capture into spec/prompt promotion
and loop steering. Compiler implementation and tests remain out of scope for
this task.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- The Hermes/Codex loop was safely stopped with the supervisor STOP file before
  this document task started.
- Latest committed RVV work is `08dfc65 rvv: require selected boundary
  conformance facts`.
- The user clarified that the RVV-first phase should not stop at the bounded
  i32m1 add/sub/mul seed.
- The target is also not an importer for arbitrary external RVV C, RVV
  intrinsic C/C++, or assembly programs.
- Hand-authored TianChen-RV MLIR, selected `tcrv.exec` kernels, and
  hand-authored `tcrv_rvv` typed bodies remain valid low-level phase inputs.
- The proposed core phrase is:
  `Every route-supported tcrv_rvv lowering unit must legalize through its
  declared RVV-owned route.`
- The user clarified that the RVV route-supported set should be calibrated by
  Linalg-like structured computation coverage, even though high-level frontend
  conversion is not the current implementation phase.
- The user clarified that the route-supported RVV scope should not be framed as
  batches. A small "first batch" can become another permanent stop like the
  current i32m1 seed. The document should define the complete coverage matrix
  up front and treat implementation ordering only as dependency management.
- The user clarified that TianChen-RV must remain a low-level execution layer,
  not a high-level mathematical IR or kernel IR. Kernel names such as softmax or
  matmul are evidence scenarios / future frontend targets, not core IR
  semantics.
- The user agreed to rename Layer 2 to `route-supported low-level RVV body
  shapes`, with an explicit note that it is not a new high-level IR layer and
  not directly expressed by `tcrv.exec` core as matmul/reduction ops.
- Continued grill consensus from questions 16-33 clarified that the bounded
  i32m1 source path is an exemplar, not the RVV maturity unit; `tcrv_rvv` is
  TianChen-RV MLIR's selected-RVV-variant typed extension-family IR, not an
  independent plugin dialect; naked fixtures are local aids, not selected-path
  maturity evidence; and future durable updates should define a low-level RVV
  unit taxonomy rather than a progress ledger or test/dashboard plan.
- The next grill branch should inspect repo reality before answering whether
  current i32m1 units are too dtype-specific, how selected `tcrv_rvv` IR lowers
  into RVV intrinsic C/C++, and where the pass pipeline should live.

## Requirements

- Create `artifacts/grill-rvv-maturity-ladder-20260518.md`.
- Mark the note as discussion/control-plane interpretation, not spec,
  acceptance evidence, or task truth.
- Capture the corrected RVV direction:
  selected/hand-authored TianChen-RV MLIR with explicit `tcrv_rvv` typed bodies
  lowers toward RVV intrinsic C/C++ and target artifacts.
- Clarify that external RVV C/intrinsic-C/assembly import is out of scope, while
  hand-authored TianChen-RV MLIR is in scope.
- Define `route-supported tcrv_rvv lowering unit` as a TianChen-RV project term
  aligned with MLIR legalization/conversion patterns/interfaces, not an upstream
  MLIR term.
- Record that Linalg/structured-computation classes are a roadmap reference for
  RVV route-supported expansion, not current source authority.
- Record that reduction and matmul/contraction belong in the required complete
  coverage matrix, not an optional later batch.
- Record the low-level boundary: roadmap layers must be primitive RVV units,
  RVV-plugin-owned body-shape contracts, and evidence scenarios, not high-level
  kernel ops.
- Capture the test-only round concern for later prompt/spec hardening.
- Preserve the question 16-33 consensus before further compaction.
- Preserve concrete open grill questions about i32m1 granularity,
  IR-to-intrinsic-C lowering, and pass pipeline organization.

## Acceptance Criteria

- [x] New grill artifact exists under `artifacts/`.
- [x] The artifact records the agreed terminology and corrected direction.
- [x] The artifact preserves open questions for the next grill continuation.
- [x] Original artifact-note capture completed.
- [x] After user authorization, durable spec/prompt promotion completed without
      compiler implementation or test-code edits.

## Out of Scope

- Editing compiler implementation or tests.
- Resolving the entire RVV feature roadmap in this single note.
- Implementing the RVV Stage 1 route-authority refactor in compiler code.

## Technical Notes

- Prior related note: `artifacts/grill-consensus-20260515.md`.
- Current new note: `artifacts/grill-rvv-maturity-ladder-20260518.md`.
- Follow-up authorized in-session: RVV specs and canonical supervisor prompt
  were updated from the grill consensus, then Stage 1 one-shot steering was
  prepared for loop restart.
