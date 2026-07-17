# Descriptor Erasure Owner: Offload descriptor-only route deletion

## Goal

Delete the Offload runtime descriptor-only artifact route as a deletion-only
Wrong Logic Deletion Campaign round. This task removes the selected-plan
metadata, target exporter, built-in registration, tests, fixtures, and spec text
that still present `tcrv-export-offload-runtime-descriptor` or
`runtime-offload-handoff-descriptor` as an active compiler artifact route.

## Why Now

Commit `0c902b4` completed the RVV/scalar lowering-descriptor metadata chain
deletion and left the repository clean. The next bounded descriptor-era surface
is the Offload descriptor-only route: the Offload plugin still advertises a
supported descriptor emission plan, the target layer still registers an Offload
descriptor exporter, and tests/specs still guard descriptor artifact acceptance.

Deletion before rebuild applies. This round must remove the old route authority
without adding a new Offload runtime ABI, EmitC route, vendor runtime call,
plugin template, compatibility wrapper, quarantine mode, or replacement lowering.

## Scope

- Remove Offload selected-plan descriptor-only route metadata from the active
  plugin emission path.
- Remove the target-owned Offload runtime descriptor exporter API, source,
  CMake target, and built-in target artifact exporter registration.
- Remove route metadata requirements that make the Offload extension bundle
  publish `tcrv-export-offload-runtime-descriptor`.
- Delete or rewrite tests and fixtures that protect descriptor artifact
  acceptance, descriptor route spoofing through a registered route, descriptor
  bundle output, descriptor-only selected-plan metadata, or
  `runtime-offload-handoff-descriptor` as a legal active artifact kind.
- Update directly stale specs/comments so they no longer describe the Offload
  descriptor route as supported architecture.
- Keep the Offload plugin capability/proposal/legality/lowering-boundary slice
  only as non-executable metadata; do not rebuild executable Offload lowering.

## Acceptance Criteria

- [ ] No new Offload replacement route, runtime handoff ABI, vendor runtime call
      lowering, plugin template, capability model, common interface foundation,
      or EmitC implementation is added.
- [ ] Production code no longer registers or returns
      `tcrv-export-offload-runtime-descriptor` or
      `runtime-offload-handoff-descriptor` as an active route/artifact plan.
- [ ] Offload plugin emission planning fails closed as unsupported rather than
      producing descriptor-only selected-plan metadata.
- [ ] Target built-in exporter registration no longer depends on an Offload
      descriptor exporter bundle.
- [ ] Tests that only protected descriptor artifact acceptance are deleted, and
      remaining Offload tests expect absence or unsupported/fail-closed behavior.
- [ ] Specs no longer present the descriptor route/exporter as supported
      Offload architecture.
- [ ] If deletion exposes build/test gaps, the exact failures are reported as
      missing future Offload architecture and descriptor routes are not restored.
- [ ] Bounded ref-scan classifies remaining hits for descriptor-only route terms
      as deleted, rewritten, or explicitly outside this owner.
- [ ] `git diff --check`, Trellis validation, and focused C++/lit checks for
      touched Offload/plugin/target/export areas are run when the tree remains
      buildable.

## Non-Goals

- No Offload rebuild.
- No new runtime handoff ABI.
- No vendor runtime call emission.
- No new EmitC route.
- No common interface foundation.
- No executable plugin template.
- No RVV feature work.
- No capability-gated selected-config implementation.
- No helper/wrapper/compatibility/quarantine/legacy mode around Offload
  descriptor artifacts.
- No renaming descriptor-only semantics to a new token.
- No Python compiler semantics.
- No direct descriptor-to-C exporter.
- No computation semantics in `tcrv.exec`.
- No restoration of descriptor route support to make checks pass.

## Minimal Evidence

- Focused deletion inventory over Offload plugin selected-plan metadata, Offload
  target exporter files, built-in route registration, target artifact/bundle
  generic descriptor special-casing, tests, fixtures, and specs.
- Bounded ref-scan over directly touched Offload/plugin/target/export/test/spec
  areas for `descriptor-only`, `runtime_offload_descriptor_scope`,
  `tcrv-export-offload-runtime-descriptor`,
  `runtime-offload-handoff-descriptor`, `OffloadRuntimeDescriptor`,
  `descriptor route`, `descriptor exporter`, and `descriptor artifact`.
- Focused C++/lit checks for touched areas when buildable.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish and after archive, final clean worktree, and one coherent commit.

## Technical Notes

- `.trellis/spec/index.md` defines TianChen-RV as a unified RISC-V MLIR after
  high-level MLIR, with descriptor-driven computation explicitly invalid as a
  long-term architecture.
- `.trellis/spec/extension-plugins/offload-runtime-plugin.md` currently still
  contains first-slice descriptor route text; this task rewrites that stale
  active-route language to a deletion-era gap.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` currently
  contains Offload descriptor exporter/bundle route text; this task removes or
  rewrites that active support language.
- `.trellis/spec/testing/mlir-testing-contract.md` currently requires Offload
  descriptor artifact tests; this task rewrites the testing requirement to
  unsupported/fail-closed behavior after descriptor deletion.
- The previous archived task PRD
  `.trellis/tasks/archive/2026-05/05-15-lowering-descriptor-metadata-chain-deletion/prd.md`
  left Offload descriptor routes out of that bounded owner and identified them
  as a next deletion surface.
