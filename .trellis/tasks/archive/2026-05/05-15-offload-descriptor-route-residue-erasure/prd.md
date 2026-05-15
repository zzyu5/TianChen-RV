# Offload Descriptor Route Residue Erasure

## Goal

Delete the remaining active Offload descriptor-route residue from production
diagnostics and directly protecting tests. The Offload plugin must still fail
closed as a plugin-owned metadata-only unsupported executable path, but it must
not preserve the deleted descriptor artifact export identity, route id, or
descriptor-deletion wording as active authority.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `c293e6c`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous completed commit `c293e6c` erased explicit
  `runtime-callable-c-source` and `standalone-c-source` source artifact-kind
  authority from active plugin/target validation and directly related tests.
- `.trellis/spec/index.md` states descriptor-driven computation is invalid and
  future executable work must go through extension-family ops and common EmitC
  lowering.
- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks and independent backend dialect framing.
- `.trellis/spec/extension-plugins/offload-runtime-plugin.md` says the current
  Offload first slice may materialize capability-gated metadata and a selected
  `tcrv_offload.lowering_boundary`, but emission planning must fail closed as
  unsupported and must not publish route ids, artifact kinds, selected-plan
  export metadata, ABI role mirrors, or target exporter bundle contributions.
- Active residue scan found descriptor-route authority in:
  `lib/Plugin/Offload/OffloadExtensionPlugin.cpp`,
  `test/Plugin/OffloadExtensionPluginTest.cpp`,
  `test/Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir`,
  `test/Target/EmissionManifest/emission-manifest-offload-pipeline.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Requirements

- Replace the Offload plugin unsupported emission-plan diagnostic so it reports
  the current truth: Offload has no active executable lowering or target
  artifact route.
- Remove active tests that require `descriptor artifact export has been
  deleted` or the older runtime-offload descriptor artifact-export explanation.
- Remove the TargetArtifactExportTest fixture that preserves the deleted
  `tcrv-export-offload-runtime-descriptor` route id.
- Keep Offload fail-closed: no supported lowering pipeline, artifact kind,
  runtime ABI parameters, selected-plan metadata, route id, bundle contribution,
  target artifact output, correctness claim, or runtime/performance claim.
- Do not add descriptor compatibility, wrappers, quarantine branches, legacy
  route aliases, replacement route ids, new Offload executable lowering, RVV
  rebuild work, source generation, or common EmitC implementation.
- If deletion exposes missing new-architecture gaps, record the gap and do not
  restore the descriptor route.

## Acceptance Criteria

- [x] Active production code no longer mentions descriptor artifact export as
      the reason Offload has no executable target route.
- [x] Active tests no longer require descriptor-deletion text for Offload
      unsupported emission plans.
- [x] Active tests no longer carry or assert the old
      `tcrv-export-offload-runtime-descriptor` route id.
- [x] Offload still reports unsupported emission planning with empty lowering
      pipeline, empty artifact kind, no runtime ABI parameters, no selected-plan
      metadata, and preserved required capability symbols.
- [x] Emission-planning and manifest lit tests still prove Offload selected
      metadata reaches an unsupported plugin-owned emission plan.
- [x] No descriptor compatibility mode, wrapper, quarantine, generic legacy
      fixture, new route, source generation, executable Offload lowering, or
      broad rebuild work is added.
- [x] Focused active ref-scan for `descriptor artifact export has been deleted`
      and `tcrv-export-offload-runtime-descriptor` is clean, excluding
      archived tasks, `artifacts/tmp`, and supervisor prompt guardrails.
- [x] Focused build/test coverage for the Offload plugin test, Target artifact
      export test, emission manifest Offload lit test, and execution planning
      Offload lit test passes or reports expected deletion gaps without
      restoring descriptor residue.
- [x] `git diff --check`, Trellis validation, finish/archive, final clean
      `git status --short`, and one coherent commit are produced if complete.

## Non-goals

- No Offload executable lowering.
- No RVV rebuild.
- No new target artifact route.
- No common EmitC implementation.
- No source generation.
- No descriptor compatibility mode.
- No edits to `artifacts/tmp`, archived Trellis tasks, or supervisor prompt
  guardrails just to reduce grep counts.
- No conversion of the deleted descriptor route into a generic legacy fixture.

## Minimal Evidence

- Focused ref-scan over active code/tests touched by this module for
  `descriptor artifact export has been deleted` and
  `tcrv-export-offload-runtime-descriptor`.
- Focused build of
  `tianchenrv-offload-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- Focused run of the Offload plugin C++ test and Target artifact export C++
  test.
- Focused lit run for `Transforms/ExecutionPlanning` and
  `Target/EmissionManifest` Offload tests.
- `git diff --check`.
- `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- The supported current Offload surface is metadata-only capability-gated
  selected boundary materialization plus unsupported emission planning.
- This round removes descriptor-route identity from the fail-closed explanation;
  it does not change Offload's selected-boundary or capability metadata model.
- Target artifact registry tests should continue proving unrelated plugin-owned
  metadata routes still register, but they should not preserve a deleted
  Offload descriptor route name as a fixture.

## Completion Evidence

- Replaced the Offload unsupported emission-plan diagnostic in
  `lib/Plugin/Offload/OffloadExtensionPlugin.cpp` with a current unsupported
  executable-route explanation that does not mention descriptor artifact export.
- Rewrote `test/Plugin/OffloadExtensionPluginTest.cpp`,
  `test/Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir`,
  and `test/Target/EmissionManifest/emission-manifest-offload-pipeline.mlir`
  to assert the descriptor-free unsupported Offload emission-plan explanation.
- Replaced the old `tcrv-export-offload-runtime-descriptor` route-id fixture in
  `test/Target/TargetArtifactExportTest.cpp` with a route-id-free assertion
  that an Offload-only plugin registry contributes no target artifact exporters
  while unrelated plugin-owned Toy metadata routes still register.
- Focused active ref-scans passed:
  `rg -n "descriptor artifact export has been deleted|tcrv-export-offload-runtime-descriptor" lib test .trellis/spec --glob '!artifacts/tmp/**' --glob '!**/archive/**'`
  returned no matches, and
  `rg -n "offload.*descriptor|descriptor.*offload|runtime-offload descriptor" lib test .trellis/spec --glob '!**/archive/**' --glob '!artifacts/tmp/**'`
  returned no matches.
- Checks passed:
  `git diff --check`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-offload-descriptor-route-residue-erasure`;
  `cmake --build build --target tianchenrv-offload-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`;
  `./build/bin/tianchenrv-offload-extension-plugin-test`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir Target/EmissionManifest/emission-manifest-offload-pipeline.mlir`
  from `build/test`.
- No self-repair was needed after focused build/test checks.
- No compatibility layer, descriptor quarantine, legacy route alias, new
  Offload lowering, new artifact route, source generation, or RVV rebuild work
  was added.
- Missing new-architecture gap: Offload still has no executable lowering or
  target artifact route. That is intentional for this deletion round; future
  Offload executable work must enter through extension-family ops, common
  EmitC lowering, and plugin-owned runtime C ABI glue.
