# Toy source-seed plugin-template proof

## Goal

Prove that the source-seed registry is a reusable extension-family plugin
construction template by adding Toy as a second plugin-owned source-seed pass
consumer. The Toy seed must flow through the common plugin source-seed
registration hook, materialize the existing Toy selected boundary surface, and
feed the already existing Toy EmitC and target artifact route without adding
Toy or RVV semantic wiring to public tools or common/core orchestration.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `36427a5`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the supplied Direction Brief.
- The previous task added `SourceSeedPassRegistration`,
  `ExtensionPlugin::registerSourceSeedPasses`, deterministic registry
  collection/validation, and moved the RVV source seed behind the RVV plugin.
- `tcrv-opt` currently collects source-seed pass registrations from enabled
  plugins and has no direct RVV seed factory include.
- Toy already has a plugin-owned construction protocol, an ODS
  `tcrv_toy.compute_skeleton` boundary op, an EmitC lowerable route
  (`toy-template-compute-emitc-route`), and a Toy target artifact/export route.
- The current missing proof is a non-RVV source-seed registration that proves
  the registry surface is not RVV-only.

## Requirements

- Add exactly one bounded Toy-owned source seed pass under the Toy plugin.
- Register that pass through `ToyExtensionPlugin::registerSourceSeedPasses`.
- Keep the public tool front door generic: `tcrv-opt` must continue to collect
  source seeds from `ExtensionPluginRegistry` and must not directly include or
  name Toy/RVV seed factories.
- The Toy source shape may be intentionally tiny, but it must be source-only,
  explicitly marked, and fail closed on unsupported markers or unsupported
  shapes.
- The pass must materialize the existing Toy execution surface:
  `tcrv.exec.kernel`, available `toy.template` capability metadata, selected
  `origin = "toy-plugin"` variant metadata, selected diagnostic, and the Toy
  `tcrv_toy.compute_skeleton` selected boundary consumed by the existing Toy
  EmitC/target route.
- Pre-existing `tcrv.exec` or `tcrv_toy` operations in seed input are stale
  selected-boundary or variant residue for this pass and must fail closed.
- Built-in plugin enablement must control Toy seed availability: available
  with built-in plugins enabled, unavailable as an unknown pass when
  `--tcrv-disable-builtin-plugins` is used.
- Common/core and tools must remain extension-neutral; no descriptor-driven
  computation, direct C semantic exporter, Python compiler-core path, generic
  vector lowering, or Toy/RVV semantic branch may be introduced.

## Acceptance Criteria

- [x] Toy source seed pass is registered by the Toy plugin through
      `registerSourceSeedPasses`.
- [x] The Toy seed pass is available when built-in plugins are enabled.
- [x] The Toy seed pass is unavailable/fail-closed when built-in plugins are
      disabled.
- [x] Common registry C++ tests cover source-seed collection with at least two
      plugin-provided registrations or a duplicate/fail-closed case involving
      the new Toy registration.
- [x] Toy plugin/unit tests cover the Toy source-seed registration metadata.
- [x] Positive lit shows source input -> Toy selected boundary -> Toy
      EmitC/route-consumable output.
- [x] Positive lit shows source input -> Toy selected boundary -> Toy target
      artifact route output.
- [x] Negative lit covers unsupported Toy seed marker/shape.
- [x] Negative lit covers stale pre-existing `tcrv.exec` or `tcrv_toy`
      residue.
- [x] `tcrv-opt` has no direct Toy/RVV seed factory wiring.
- [x] Common/core has no Toy/RVV semantic branch added for source seed logic.
- [x] Focused build/tests and `git diff --check` pass; run full
      `check-tianchenrv` if practical.

## Validation Results

- Focused build passed:
  `cmake --build build --target TianChenRVToyPlugin tcrv-opt tcrv-translate
  tianchenrv-toy-extension-plugin-test tianchenrv-plugin-registry-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-toy-extension-plugin-test` and
  `./build/bin/tianchenrv-plugin-registry-test`.
- Focused lit passed from `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'toy-template-selected-boundary-seed|toy-template-target-artifact|toy-template-materialization|rvv-i32m1-selected-boundary-seed'`,
  14/14 selected tests passed.
- Full practical check passed:
  `cmake --build build --target check-tianchenrv -j2`, 106/106 tests passed.
- `git diff --check` passed.
- Tool direct-wiring scan returned no matches:
  `rg -n
  "ToySelectedBoundarySeed|RVVSelectedBoundarySeed|createMaterializeToyTemplateSelectedBoundarySeedPass|createMaterializeRVVI32M1SelectedBoundarySeedPass|tcrv-toy-materialize-template-selected-boundary-seed|tcrv-rvv-materialize-i32m1-selected-boundary-seed"
  tools/tcrv-opt tools/tcrv-translate`.
- Common/core source-seed semantic branch scan returned no matches:
  `rg -n
  "ToySelectedBoundarySeed|createMaterializeToyTemplateSelectedBoundarySeedPass|tcrv-toy-materialize-template-selected-boundary-seed|tcrv_toy\\.lowering_seed|RVVSelectedBoundarySeed|createMaterializeRVVI32M1SelectedBoundarySeedPass"
  include/TianChenRV/Plugin/ExtensionPlugin.h lib/Plugin/ExtensionPlugin.cpp
  include/TianChenRV/Transforms lib/Transforms`.
- Hardware evidence was not run. This task adds a Toy template source seed
  with no runtime, correctness, performance, object generation, or `ssh rvv`
  claim.

## Self-Repair

- Initial Toy seed lit inputs used zero-argument `func.func @name` syntax
  without `()`. Repaired the MLIR syntax and reran focused lit.
- Initial Toy seed implementation wrote construction-protocol metadata with
  helper metadata names instead of the existing Toy legality attributes
  (`tcrv_toy.*`). Repaired the emitted variant attributes and reran focused
  build/lit.
- Initial stale Toy residue test used an unknown op in the registered
  `tcrv_toy` dialect. Replaced it with a legal `tcrv_toy.compute_skeleton`
  residue so the fail-closed diagnostic is produced by the Toy seed pass.

## Spec Update Judgment

No `.trellis/spec/` update is needed for this round. The durable rule already
exists in `plugin-protocol/interfaces-and-registry.md`: source-seed
registrations are plugin-owned pass plumbing collected by the common registry,
while semantics stay inside the concrete plugin pass. This task proves a second
consumer of that rule rather than introducing a new long-term contract.

## Definition Of Done

- PRD and Trellis context files are truthful.
- Implementation is one coherent Toy plugin-source-seed integration, not
  docs-only or helper-only work.
- Focused tests prove both registry availability and route consumption.
- Task status and workspace journal are updated.
- Task is finished/archived when complete.
- One coherent commit is created when complete. If unfinished, leave the task
  open and name the exact continuation point.

## Out Of Scope

- No RVV source-shape expansion, RVV SEW/LMUL/dtype/op-family work, generic
  MLIR vector lowering, high-level tensor lowering, TensorExt/IME/offload work,
  new artifact routes, runtime correctness, performance work, descriptor or
  binary-family registries, direct C semantic exporters, Python compiler-core
  logic, GCC-default route, compatibility wrappers, state-machine ledgers, or
  docs/tests-only round.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Prior PRDs read:
  `.trellis/tasks/archive/2026-05/05-16-plugin-owned-source-seed-registration-interface/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-16-toy-target-artifact-export-template/prd.md`.
- Relevant journal read:
  `.trellis/workspace/codex/journal-8.md` sessions 93, 94, and 95.
- Primary code surfaces inspected:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/Toy/ToyExtensionPlugin.h`,
  `lib/Plugin/Toy/ToyExtensionPlugin.cpp`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `test/Plugin/PluginRegistryTest.cpp`,
  `test/Plugin/ToyExtensionPluginTest.cpp`,
  `test/Transforms/ExecutionPlanning/execution-planning-pipeline-toy.mlir`,
  `test/Conversion/EmitC/toy-template-materialization.mlir`, and
  `test/Target/Toy/toy-template-target-artifact.mlir`.
