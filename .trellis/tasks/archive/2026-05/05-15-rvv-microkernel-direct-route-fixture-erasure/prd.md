# RVV Microkernel Direct Route Fixture Erasure

## Goal

Delete active RVV/scalar/dispatch direct microkernel route fixtures that
preserve historical direct-C route identity after the direct C semantic exporter
deletion campaign. This round is deletion/refactor-only: it removes route-name
fixtures and route-id absence fixtures for old direct microkernel, self-check,
scalar, and RVV+scalar dispatch paths without adding any replacement emission
route.

## What I Already Know

- The repository is clean at `41a9fbb chore(rvv): erase smoke-probe route fixtures`.
- No Trellis task was active, so this task was created from the Hermes Direction Brief.
- The deletion target is active route identity in tests/spec text, not archived
  task history or temporary artifact output.
- Current architecture requires future source output to go through extension
  family ops, a materialized MLIR EmitC module route, the MLIR C/C++ emitter,
  and target export validation.
- Historical direct C route names must not remain as active negative fixtures:
  `tcrv-export-rvv-microkernel-c`,
  `tcrv-export-rvv-microkernel-self-check-c`,
  `tcrv-export-scalar-microkernel-c`, and
  `tcrv-export-rvv-scalar-i32-vadd-dispatch-c`.

## Requirements

- Remove active lit invocations of deleted direct RVV microkernel and
  self-check translate options.
- Remove or rewrite active C++ tests that look up historical RVV/scalar/dispatch
  direct route ids as named absence fixtures.
- Adjust directly related spec/testing language so it states the deletion rule
  without preserving the old route ids as active route contracts.
- Keep generic registry/front-door fail-closed behavior covered through current
  route-agnostic checks.
- Keep the change deletion-only: do not add RVV emission, source/header/object/
  bundle routes, aliases, wrappers, compatibility paths, self-check exporters,
  executable plugin templates, or descriptor compatibility.

## Acceptance Criteria

- [ ] Active tests no longer invoke deleted direct `tcrv-translate` options for
      RVV microkernel or self-check.
- [ ] Active C++ tests no longer look up historical RVV/scalar/dispatch direct
      route ids as named absence fixtures.
- [ ] Active specs/testing contracts describe route deletion without retaining
      the old route ids as active route contracts or required fixtures.
- [ ] Generic route registry/front-door fail-closed behavior remains covered.
- [ ] Focused ref scan over active repo surfaces, excluding `artifacts/tmp`,
      `.trellis/tasks/archive`, and `.trellis/workspace`, finds no remaining
      active hits for the four historical route ids, or any remaining hit is
      justified as deletion-campaign governance rather than route authority.
- [ ] Focused build/test coverage runs for `tianchenrv-target-artifact-export-test`,
      `tcrv-translate`, and affected Target/ArtifactExport or
      Target/RVVMicrokernel lit tests.
- [ ] `git diff --check` and Trellis task validation pass.
- [ ] Task is finished/archived and the round is committed.

## Out Of Scope

- RVV rebuild.
- Common EmitC implementation.
- Executable plugin template.
- New target artifact route.
- Real `ssh rvv` evidence as the main result.
- Replacement direct microkernel, dispatch, or self-check exporter.
- Descriptor compatibility, legacy mode, wrapper, alias, or quarantine.
- Edits to `artifacts/tmp`, archived Trellis tasks, or supervisor prompt
  guardrails just to reduce grep counts.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Initial target files:
  - `test/Target/ArtifactExport/target-source-artifact-routes.test`
  - `test/Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir`
  - `test/Target/TargetArtifactExportTest.cpp`
- Directly related registration/export code should only be inspected if test
  deletion exposes an actual production registration gap.
