# Core RVV Source-to-Exec Deleted-Route Fixture Erasure

## Goal

Delete the active old core RVV source-to-exec pass fixture surface left by the
former high-level linalg/vector-to-`tcrv.exec` route. The round removes named
negative fixtures and durable route text that keep historical core RVV
source-to-exec public option names alive as active contracts, so later frontend
reconstruction must be plugin/interface-owned and flow through extension-family
ops, selected execution surfaces, common EmitC, and target export validation.

## Requirements

- Delete or rewrite the active `Transforms/DeletedRoutes` lit coverage that
  invokes the seven historical core RVV source-to-exec public options as named
  absence fixtures.
- Update active specs and README prose that list those historical public option
  names as durable route contracts.
- Preserve the architecture rule that high-level frontend reconstruction is a
  future plugin/interface-owned path, not a core RVV semantic branch.
- Preserve fail-closed deletion-campaign language generically, without keeping
  old public option spellings as active tests, specs, README route contracts, or
  compatibility promises.
- Do not add source-to-exec passes, aliases, wrappers, quarantine modes,
  descriptor paths, direct C exporters, RVV rebuilds, common EmitC rebuilds, or
  executable plugin templates.

## Acceptance Criteria

- [ ] Active tests no longer invoke the deleted source-to-exec options as named
  negative fixtures.
- [ ] Active specs/docs describe the deletion rule generically and do not list
  those historical public option names as durable route contracts.
- [ ] Remaining source-to-exec discussion says future high-level frontend work
  is plugin/interface-owned and not a core RVV semantic branch.
- [ ] A focused active-surface ref scan finds no remaining historical public
  option spelling hits, except explicitly justified non-authoritative
  governance if any.
- [ ] Focused build/test coverage for `tcrv-opt` and affected transform lit
  coverage is run or a precise build-environment blocker is reported.
- [ ] `check-tianchenrv` is attempted or the current status/blocker is reported
  without restoring old routes to satisfy unrelated failures.
- [ ] `git diff --check`, Trellis task validation, finish/archive state, clean
  `git status --short`, and one coherent commit are produced.

## Definition of Done

- Deletion/refactor-only changes are limited to active tests, specs, README, and
  directly exposed active registration residue if found.
- Any build or test failure is classified as either this-round regression or
  pre-existing missing-architecture/baseline gap.
- The Trellis task is started, validated, finished, archived, and recorded in a
  single commit.

## Out of Scope

- RVV rebuild.
- High-level frontend lowering.
- Common EmitC implementation.
- Executable plugin templates.
- New pass options, aliases, wrappers, compatibility modes, or deleted-route
  replacements.
- `ssh rvv` runtime evidence as the main result.
- Edits to `artifacts/tmp`, archived Trellis tasks, workspace journals, or
  supervisor prompt guardrails merely to reduce grep counts.

## Technical Approach

Use deletion-before-rebuild discipline:

1. Remove the named deleted-route lit fixture instead of preserving old public
   option names as negative contracts.
2. Rewrite active spec/README source-to-exec deletion sections to state the
   generic rule and future plugin/interface-owned reconstruction path without
   enumerating the historical option spellings.
3. Scan active repo surfaces for the historical spellings and inspect
   Passes.td/lib registration only if active code still contains residue.
4. Run focused lit/build checks, then attempt or report `check-tianchenrv`.

## Technical Notes

- Current repo root: `/home/kingdom/phdworks/TianchenRV`.
- Current baseline before edits: `main` at `e2492a3`, clean worktree.
- Relevant specs read for this round:
  `.trellis/spec/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Directly relevant active surfaces from the task brief:
  `test/Transforms/DeletedRoutes/source-rvv-pass-family-deleted.mlir`,
  `README.md`, and the three spec files above.
