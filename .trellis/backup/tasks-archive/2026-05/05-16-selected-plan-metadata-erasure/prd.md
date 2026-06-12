# Selected-plan metadata channel erasure

## Goal

Delete the generic `selected_plan_metadata` /
`VariantSelectedPlanMetadata` conduit as active compiler state. Selected-path
and emission-readiness decisions must remain expressible through typed generic
fields such as selected variant, role, origin, lowering boundary, runtime ABI
ownership, runtime ABI parameters, and required capabilities, not through an
arbitrary name/value/role/note dictionary.

## Why

The previous deletion round erased finite RVV vector-shape and runtime-length
metadata helpers. Bounded live evidence still shows a generic
`selected_plan_metadata` attribute on `tcrv.exec.diagnostic`, a
`VariantSelectedPlanMetadata` C++ object on `VariantEmissionPlan`, validation
for arbitrary dictionary entries, and EmissionReadiness materialization of the
array. That channel can preserve descriptor-style selected-path authority even
after finite RVV catalogs are gone.

## Scope

- Delete the `selected_plan_metadata` diagnostic attribute from active
  `tcrv.exec` ODS surfaces.
- Delete diagnostic convention constants that name this attribute.
- Delete `VariantSelectedPlanMetadata` storage, accessors, mutators, and
  validation from the extension plugin emission-plan API.
- Delete EmissionReadiness materialization of selected-plan metadata arrays.
- Delete or rewrite plugin tests that only assert the selected-plan metadata
  channel is empty or populated.
- Rewrite spec wording that protects selected-plan metadata as an active
  diagnostic, plugin-plan, manifest, exporter, or bundle contract.
- Retain typed generic fields only where they do not recreate arbitrary
  dictionary authority: `selection_kind`, `target`, `origin`, `role`,
  `lowering_boundary`, `runtime_abi_parameters`, and
  `required_capabilities`.

## Non-goals

- No replacement metadata channel.
- No new RVV lowering.
- No common lower-to-EmitC pass.
- No executable plugin template.
- No new artifact route.
- No runtime ABI feature.
- No evidence matrix.
- No finite RVV shape metadata restoration.
- No wrappers, legacy modes, compatibility paths, or descriptor tests to keep
  the old channel alive.

## Requirements

- No active source or test outside archived Trellis history defines,
  materializes, validates, or asserts `selected_plan_metadata`,
  `VariantSelectedPlanMetadata`, `getSelectedPlanMetadata`,
  `addSelectedPlanMetadata`, `clearSelectedPlanMetadata`, or selected-plan
  metadata dictionary fields.
- Emission-plan diagnostics still materialize typed fields and still fail
  closed when no materialized route exists.
- Plugin emission-plan validation still checks bounded generic route, ABI,
  glue-role, status, capability, runtime ABI parameter, diagnostic, and
  explanation fields.
- Tests that only asserted the selected metadata channel is empty are deleted
  or rewritten around typed fields instead of preserved as compatibility
  coverage.
- If deletion exposes a missing new-architecture gap, record it rather than
  restoring the metadata channel.

## Acceptance Criteria

- [ ] Focused active-surface ref-scan finds no live reference to
  `selected_plan_metadata`, `VariantSelectedPlanMetadata`,
  `getSelectedPlanMetadata`, `addSelectedPlanMetadata`,
  `clearSelectedPlanMetadata`, `selected-plan metadata`, or
  `selected plan metadata` under `include`, `lib`, `test`, `scripts`,
  `README.md`, and `.trellis/spec`, excluding archived task records,
  workspace notes, build artifacts, temporary artifacts, and git metadata.
- [ ] `include/TianChenRV/Dialect/Exec/IR/ExecOps.td` no longer exposes a
  `selected_plan_metadata` diagnostic attribute.
- [ ] `include/TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h` no longer
  names the selected-plan metadata attribute.
- [ ] `include/TianChenRV/Plugin/ExtensionPlugin.h` and
  `lib/Plugin/ExtensionPlugin.cpp` no longer define or validate
  `VariantSelectedPlanMetadata`.
- [ ] `lib/Transforms/EmissionReadiness.cpp` no longer materializes selected
  plan metadata dictionaries.
- [ ] Affected plugin tests and EmissionReadiness tests pass or are rewritten
  to assert typed fields only.
- [ ] Focused build/check commands for Exec diagnostics, plugin emission plans,
  and EmissionReadiness are run.
- [ ] `ninja -C build check-tianchenrv` is attempted after focused checks.
- [ ] `git diff --check`, Trellis validation, task finish/archive, clean
  status, and one coherent commit are completed if the task is complete.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
- `include/TianChenRV/Dialect/Exec/IR/DiagnosticConventions.h`
- `include/TianChenRV/Plugin/ExtensionPlugin.h`
- `lib/Plugin/ExtensionPlugin.cpp`
- `lib/Transforms/EmissionReadiness.cpp`
- Direct tests and specs found by focused ref-scan only.

## Technical Notes

- Initial repo state: `/home/kingdom/phdworks/TianchenRV`, clean worktree,
  HEAD `bfb4107 chore(rvv): erase finite vector-shape metadata`.
- There is no pre-existing active Trellis task; this task was created from the
  Direction Brief as a deletion-only Wrong Logic Deletion Campaign round.
- Durable project rules from `.trellis/spec/index.md`: compiler implementation
  stays in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck; Python is tooling only;
  descriptor-driven computation is deletion target/fail-closed debt; future
  executable work goes through extension family ops and common EmitC lowering.
- Memory-derived campaign guardrail: deletion before rebuild, no compatibility
  wrappers, and no mixing RVV/Common EmitC rebuild work into this deletion
  round.
