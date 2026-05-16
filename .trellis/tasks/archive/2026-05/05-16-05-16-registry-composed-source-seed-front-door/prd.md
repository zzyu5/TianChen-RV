# Registry-composed source-seed-to-artifact front door

## Goal

Add one explicit registry-composed front door for bounded source-seed inputs.
The front door must collect source-seed pass factories from enabled extension
plugins, run those plugin-owned materializers, and hand the resulting selected
extension-family boundary IR to the existing emission-plan/coherence and target
artifact export route.

This round should make this workflow real:

```text
source MLIR seed marker
  -> plugin-owned source-seed materialization pass
  -> selected extension-family boundary
  -> plugin legality / generic capability gates
  -> emission-plan materialization
  -> execution-plan coherence gate
  -> existing tcrv-translate target artifact route
```

The front door is explicit. It must not silently change the ordinary
`tcrv-execution-planning-pipeline`, and it must not make common/tool code
recognize RVV or Toy marker names, arithmetic, template semantics, route ids,
intrinsic names, or runtime ABI details.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `9b7e1dc`.
- No current Trellis task existed at session start; this task was created from
  the supplied Direction Brief.
- Commit `36427a5` added common `SourceSeedPassRegistration` plumbing and
  moved the RVV source seed behind `RVVExtensionPlugin`.
- Commit `9b7e1dc` proved the source-seed registry is not RVV-only by adding a
  Toy source-seed consumer through `ToyExtensionPlugin::registerSourceSeedPasses`.
- Existing public tools no longer directly include Toy/RVV seed factory
  headers or directly invoke Toy/RVV seed factory functions.
- Existing individual seed commands work, but production workflow still
  requires manual pass sequencing such as seed materialization followed by
  emission planning and target export.
- The existing `tcrv-execution-planning-pipeline` starts from execution
  anchors and proposal materialization. It is not the right prelude for current
  source-seed outputs because the bounded seed passes already materialize
  selected variants and selected extension-family boundary IR.
- The coherent bounded slice for this round is therefore a named `tcrv-opt`
  front-door pipeline that sequences registered source-seed materializers into
  selected-boundary artifact preflight. `tcrv-translate` integration can remain
  the existing target artifact consumer for this round.

## Requirements

- Add one named public `tcrv-opt` pipeline for source-seed artifact preflight.
- The pipeline must collect source-seed pass registrations from the enabled
  `ExtensionPluginRegistry` and add each registered pass in deterministic
  registry order.
- After source-seed materialization, the pipeline must run only generic/common
  gates needed to make the selected seed output artifact-front-door-ready:
  hart-parallel capability check, plugin-local variant legality routing,
  generic capability requires check, emission-plan materialization, and
  execution-plan coherence.
- The pipeline must not invoke `tcrv-materialize-plugin-variants` or
  `tcrv-select-variants` for source-seed outputs, because the bounded seed
  materializers already produce selected variant/boundary surfaces.
- The pipeline must not run selected lowering-boundary materialization
  unconditionally, because current Toy source seed already materializes its
  selected `tcrv_toy.compute_skeleton` boundary and duplicate boundaries must
  remain fail-closed.
- Keep source shape semantics inside the plugin-owned seed passes. Common/tool
  code may collect and sequence pass factories, but must not inspect seed
  marker names, source operations, dtype/shape, Toy template metadata, RVV
  arithmetic, route ids, intrinsic names, or runtime ABI fields.
- RVV and Toy source seeds must both exercise the same named pipeline.
- With built-in plugins disabled, the named pipeline must fail closed on source
  input rather than materializing a seed through hidden built-in state.
- Malformed seed input, unsupported marker values, stale pre-existing
  `tcrv.exec`/extension-boundary residue, and mixed incompatible RVV+Toy seed
  inputs must fail closed through the plugin-owned seed passes or the existing
  generic gates.
- Existing explicit selected-boundary inputs and existing explicit seed pass
  commands must continue to work.
- If the new front door is used to reach RVV target artifact export, refresh
  `ssh rvv` correctness evidence for the generated object/header path.

## Acceptance Criteria

- [x] A named public pipeline is available through `tcrv-opt` and documented in
      tests by its pass-pipeline option spelling.
- [x] The pipeline uses source-seed pass registrations collected from
      `ExtensionPluginRegistry`; it has no direct Toy/RVV seed factory includes
      and no direct Toy/RVV seed factory calls.
- [x] RVV source seed input reaches the new pipeline and produces supported
      RVV emission-plan metadata consumable by `tcrv-translate
      --tcrv-export-target-artifact`.
- [x] Toy source seed input reaches the same pipeline and produces the existing
      Toy target artifact output through `tcrv-translate
      --tcrv-export-target-artifact`.
- [x] The pipeline remains available but fail-closed when built-in plugins are
      disabled.
- [x] Negative lit coverage proves unsupported marker values, stale
      pre-existing selected-boundary/variant residue, and mixed incompatible
      RVV+Toy seed input fail closed through the new front door.
- [x] Existing individual RVV and Toy seed positive/negative tests still pass.
- [x] Existing explicit selected-boundary or target artifact tests still pass.
- [x] Common/core and public tools remain extension-neutral; changed-surface
      scans show no direct Toy/RVV seed factory wiring in `tcrv-opt` or
      `tcrv-translate`, and no Toy/RVV source-seed semantic branch in common
      transforms/plugin registry code.
- [x] No descriptor-driven computation, direct C semantic exporter, Python
      compiler-core logic, compatibility wrapper, or new artifact route is
      introduced.
- [x] Focused build/test, focused lit, `git diff --check`, and full
      `check-tianchenrv` if practical pass.
- [x] If RVV artifacts are generated through the new front door, the generated
      RVV harness is rerun on `ssh rvv`; otherwise the bounded stopping point is
      recorded explicitly.

## Validation Results

- Focused build passed:
  `cmake --build build --target TianChenRVTransforms TianChenRVPlugin
  TianChenRVRVVPlugin TianChenRVToyPlugin tcrv-opt tcrv-translate
  tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test
  tianchenrv-toy-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-plugin-registry-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`, and
  `./build/bin/tianchenrv-toy-extension-plugin-test`.
- Focused lit passed from `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'source-seed-artifact-front-door|rvv-i32m1-selected-boundary-seed|toy-template-selected-boundary-seed|i32m1-add-object-artifact'`,
  8/8 selected tests passed.
- Full practical check passed:
  `cmake --build build --target check-tianchenrv -j2`, 109/109 lit tests
  passed.
- `git diff --check` passed.
- `tcrv-opt --help-hidden` lists
  `--tcrv-source-seed-artifact-front-door-pipeline`.
- RVV front-door artifact evidence directory:
  `artifacts/tmp/rvv_selected_boundary_seed_frontdoor/20260516T143133Z`.
- Generated RVV front-door object/header:
  `seed.o` and `seed.h` from
  `tcrv-opt --tcrv-source-seed-artifact-front-door-pipeline` piped to
  `tcrv-translate --tcrv-export-target-artifact` and
  `--tcrv-export-target-header-artifact`; `file` reported a RISC-V ELF
  relocatable object.
- `ssh rvv` link/run evidence:
  `artifacts/tmp/rvv_selected_boundary_seed_frontdoor/20260516T143133Z/ssh_rvv_link_run.log`.
  Remote run printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`
  and exited with status 0.
- Tool direct-wiring scan returned no matches:
  `rg -n
  "ToySelectedBoundarySeed|RVVSelectedBoundarySeed|createMaterializeToyTemplateSelectedBoundarySeedPass|createMaterializeRVVI32M1SelectedBoundarySeedPass|tcrv-toy-materialize-template-selected-boundary-seed|tcrv-rvv-materialize-i32m1-selected-boundary-seed"
  tools/tcrv-opt tools/tcrv-translate`.
- Common/core source-seed semantic branch scan returned no matches:
  `rg -n
  "ToySelectedBoundarySeed|createMaterializeToyTemplateSelectedBoundarySeedPass|tcrv-toy-materialize-template-selected-boundary-seed|tcrv_toy\\.lowering_seed|RVVSelectedBoundarySeed|createMaterializeRVVI32M1SelectedBoundarySeedPass|tcrv-rvv-materialize-i32m1-selected-boundary-seed|tcrv_rvv\\.lowering_seed"
  include/TianChenRV/Transforms lib/Transforms
  include/TianChenRV/Plugin/ExtensionPlugin.h lib/Plugin/ExtensionPlugin.cpp`.

## Self-Repair

- The first design sketch considered composing source-seed passes directly with
  `tcrv-execution-planning-pipeline`. A live probe showed this fails because
  source-seed outputs already contain selected materialized variants and the
  proposal stage correctly rejects them as no-viable-proposals. The
  implemented front door therefore composes source-seed materialization with
  selected-boundary artifact preflight gates instead.
- A probe also showed unconditional selected lowering-boundary materialization
  duplicates the Toy seed's already materialized `tcrv_toy.compute_skeleton`.
  The implemented pipeline leaves selected-boundary creation inside the
  plugin-owned seed passes.
- The disabled-builtins negative test initially expected the emission-plan
  diagnostic, but the final pipeline reports the later coherence gate's
  generic "requires at least one tcrv.exec.kernel" fail-closed diagnostic.
  Updated the test to the actual generic gate.

## Spec Update Judgment

Updated `.trellis/spec/variant-pipeline/generation-selection-tuning.md` with
the durable public contract for
`--tcrv-source-seed-artifact-front-door-pipeline`, including pass order,
extension-neutrality, why it does not run proposal/selection or unconditional
lowering-boundary materialization, and disabled-plugin fail-closed behavior.

## Definition Of Done

- Task context files are curated and truthful.
- Implementation is a coherent front-door integration, not docs-only,
  helper-only, or a manual pass-order report.
- Focused tests prove both source-seed families share the same front door.
- Task status and workspace journal are updated.
- Task is finished/archived when complete.
- One coherent commit is created when complete. If unfinished, leave the task
  open and name the exact next continuation point.

## Out Of Scope

- No new RVV source shapes, Toy source shapes, RVV sub/mul seeds, new
  SEW/LMUL/dtype/op families, generic MLIR vector lowering, high-level tensor
  lowering, TensorExt/IME/offload work, new artifact routes, performance
  tuning, descriptor or binary-family registries, direct C semantic exporters,
  Python compiler-core logic, GCC-default routes, compatibility wrappers,
  state-machine ledgers, or docs/tests-only work.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior PRDs read:
  `.trellis/tasks/archive/2026-05/05-16-plugin-owned-source-seed-registration-interface/prd.md`,
  `.trellis/tasks/archive/2026-05/05-16-toy-source-seed-plugin-template-proof/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-16-05-16-bounded-mlir-to-rvv-selected-boundary-seed/prd.md`.
- Relevant journal read:
  `.trellis/workspace/codex/journal-8.md` sessions 93, 94, and 96.
- Primary code surfaces inspected:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `include/TianChenRV/Transforms/Passes.h`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Plugin/Toy/ToySelectedBoundarySeed.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/TargetTranslateRegistration.cpp`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed*.mlir`,
  `test/Transforms/Toy/toy-template-selected-boundary-seed*.mlir`,
  `test/Target/RVV/`, and
  `test/Target/Toy/`.
