# Descriptor Erasure Owner: lowering_descriptor metadata chain deletion

## Goal

Delete the remaining `lowering_descriptor` metadata chain as a deletion-only
Wrong Logic Deletion Campaign round. This task removes legacy RVV/scalar
lowering-descriptor constants, selected-plan mirror metadata, plugin/target
consumers, and tests that preserve descriptor-as-input or descriptor-mirror
behavior.

## Why Now

Commit `a88b87a` deleted RVV binary descriptor authority and left a clean
worktree, but bounded ref-scans still show production and test references to:

- `tcrv_rvv.lowering_descriptor`
- `tcrv_scalar.lowering_descriptor`
- `tcrv_rvv.selected_lowering_descriptor`
- `tcrv_scalar.selected_lowering_descriptor`
- plain `selected_lowering_descriptor`

Deletion before rebuild still applies. This round must continue descriptor
erasure rather than add capability legality, new RVV routes, or replacement
architecture.

## Scope

- Delete production constants and helpers that expose selected-lowering
  descriptor metadata names or legacy descriptor mirror roles/notes.
- Delete plugin and target consumers that accept, validate, quarantine,
  cross-check, or emit `tcrv_rvv.lowering_descriptor`,
  `tcrv_scalar.lowering_descriptor`, selected-lowering descriptor metadata, or
  legacy descriptor mirror selected-plan records.
- Rewrite selected-plan/export validation to require typed family/source,
  EmitC route, runtime, selected-config, and component-capacity metadata only.
- Delete or rewrite tests, fixtures, sed injections, quarantine checks, and
  stale-descriptor negative cases that protect descriptor-as-legal-input,
  descriptor mirror, selected-lowering descriptor, or legacy fail-closed
  semantics.
- Update task-local notes and any directly stale specs/comments only to align
  with the deletion boundary; do not turn documentation into the milestone.

## Acceptance Criteria

- [ ] No new feature/rebuild code is added.
- [ ] No descriptor compatibility, quarantine, optional mirror, selected-route
      token, renamed wrapper, or legacy fail-closed API is preserved.
- [ ] Production code no longer names or consumes
      `tcrv_rvv.lowering_descriptor`, `tcrv_scalar.lowering_descriptor`,
      `tcrv_rvv.selected_lowering_descriptor`,
      `tcrv_scalar.selected_lowering_descriptor`, or plain
      `selected_lowering_descriptor`, except for out-of-scope governance text
      that explicitly remains a campaign guardrail.
- [ ] Tests protecting descriptor-as-legal-input or descriptor mirror behavior
      are deleted or rewritten to descriptor-erased typed-source expectations.
- [ ] If deletion exposes missing new-architecture gaps, exact build/test
      failures are reported without restoring descriptor logic.
- [ ] Bounded ref-scan classifies every remaining hit as deleted, rewritten, or
      explicitly outside this owner.
- [ ] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      focused touched-area build/lit/C++ checks are run when the tree is
      buildable.

## Non-Goals

- No capability-gated selected-config implementation.
- No new RVV family, dtype, SEW, LMUL, tail, mask, frontend, runtime, EmitC
  route, plugin template, common interface, performance/evidence matrix, or
  hardware claim.
- No descriptor-to-C production exporter, Python compiler semantics,
  computation semantics in `tcrv.exec`, or extension-specific compute branch in
  generic core orchestration.
- No helper-only, PRD-only, journal-only, smoke-only, report-only, or
  artifact-text-only milestone.
- No restoration of descriptor paths solely to make checks pass.

## Minimal Evidence

- Focused deletion inventory over the directly relevant production files and
  tests.
- Bounded ref-scan over `include/`, `lib/`, `test/`, `.trellis/spec/`, and
  supervisor prompt/script guardrail text for:
  `lowering_descriptor`, `selected_lowering_descriptor`,
  `legacy descriptor mirror`, `descriptor-only`, `descriptor-as-legal-input`,
  `descriptor-to-C`, `RVVBinaryDescriptor`, `RVVBinaryFamilyRegistry`, and
  descriptor-shaped route authority.
- Focused build/C++ tests/lit checks for the touched RVV, scalar, transform,
  target artifact, and lowering-boundary areas when buildable.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish and after archive, final clean worktree, and one coherent commit.

## Technical Notes

- The long-term specs keep TianChen-RV as a unified RISC-V MLIR execution layer
  after high-level MLIR; descriptor-driven computation is not architecture.
- Relevant code owners include RVV/scalar selected config metadata contracts,
  RVV selected lowering boundary/planning/legality/emission planning, scalar
  plugin planning, scalar/RVV microkernel export, RVV+scalar dispatch export,
  source lowering frontends, and tests under `test/Plugin`,
  `test/Transforms`, and `test/Target`.
- Offload runtime descriptor routes and generic `CapabilityDescriptor` support
  types are not part of this bounded lowering-descriptor metadata owner unless
  they reference the RVV/scalar lowering-descriptor chain above.
