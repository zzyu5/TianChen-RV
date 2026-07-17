# Common source-artifact bundle front door

## Goal

Add one common production front door that takes supported source MLIR through
plugin-registered source-front-door passes, shared planning/coherence,
plugin-owned EmitC route materialization, MLIR C/C++ emission, and target
object/header/bundle export without requiring a manual `tcrv-opt |
tcrv-translate` chain.

## What I already know

- The repository is `/home/kingdom/phdworks/TianchenRV`; HEAD before this task
  is `ddc9409 docs: erase emission-runtime direct route residue`; the worktree
  was clean before task creation.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- `tcrv-opt` already registers plugin-owned source-front-door passes from the
  enabled extension plugin registry and exposes
  `--tcrv-source-artifact-front-door-pipeline`.
- The existing source pipeline composes registered source-front-door passes with
  shared legality/capability/emission-plan/coherence checks.
- `tcrv-translate --tcrv-export-target-artifact-bundle` exports bundles only
  from already planned/selected MLIR.
- `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` currently runs
  the execution planning pipeline before bundle export, but does not collect or
  run source-front-door pass registrations.
- Existing RVV, Toy, and TensorExtLite tests prove the manual chain from source
  front door to target artifact output for selected routes.

## Boundaries

- This task is a common workflow bridge, not a new RVV/Toy/TensorExt family and
  not an Offload/IME/scalar implementation.
- Core/front-door code may compose registered plugin passes and common
  interfaces only. It must not infer source semantics from family names, route
  IDs, descriptors, artifact names, or old source-export metadata.
- Source materialization remains plugin-owned.
- Target artifact materialization must continue to use selected emission-plan
  candidates, plugin-owned `TCRVEmitCLowerableRoute` builders, the common EmitC
  materializer, MLIR EmitC C/C++ emission, and target exporter registrations.
- No descriptor-driven computation, direct-C semantic exporter,
  source-export route, compatibility route ID, Python compiler-core behavior,
  or extension-specific semantic branch may be introduced.

## Requirements

- Add or repair a `tcrv-translate` production path that directly composes:
  source MLIR -> plugin source-front-door pipeline -> selected emission-plan
  coherence -> target artifact bundle export.
- The path must collect source-front-door pass registrations from the same
  enabled plugin registry used by other built-in planning/export routes.
- The path must fail closed when no source-front-door pass is registered, when
  no source-front-door pass materializes a selected execution surface, when
  selected supported candidates are absent or ambiguous, and when target bundle
  output cannot be written.
- The path must preserve existing target artifact exporter behavior for
  materialized EmitC object/header/bundle output.
- At least one RVV vector-source add/sub/mul case must prove the one-shot front
  door produces the same selected variant identity, route provenance, runtime
  ABI records, object/header bundle records, and unmangled callable symbol as
  the existing manual chain.
- At least one non-RVV plugin-registered source front door, preferably Toy
  because it already has source-front-door and object/header bundle coverage,
  must prove the bridge consumes plugin registrations rather than hard-coding
  RVV.

## Acceptance Criteria

- [x] A positive RVV vector-source case invokes the new common source-artifact
      bundle front door in one `tcrv-translate` command and produces a complete
      object/header bundle.
- [x] The RVV bundle index records the selected variant, route, origin plugin,
      runtime ABI kind/name, ordered runtime ABI parameters, materialized EmitC
      provenance, and no descriptor/direct-C/source-export residue.
- [x] The RVV object contains the expected unmangled runtime-callable C symbol.
- [x] A positive non-RVV plugin source-front-door case reaches bundle export
      through the same common front door.
- [x] Negative coverage fails closed for disabled/unregistered plugins or empty
      source-front-door registrations.
- [x] Negative coverage fails closed when no source-front-door pass matches or
      no selected supported target artifact route is produced.
- [x] Targeted scans over touched core/translate/target/plugin surfaces show no
      new descriptor route authority, direct C semantic exporter,
      source-export route, or core RVV/Toy/TensorExt semantic branch.
- [x] Focused lit/C++ checks for the changed route pass.

## Out of Scope

- New arithmetic families, generic RVV lowering, IME lowering, Offload
  executable lowering, scalar direct C export, descriptor adapters, old route
  compatibility aliases, Python compiler-core logic, source-shape scanners in
  common/core code, and standalone evidence/report tooling.
- Refreshing `ssh rvv` runtime evidence unless the generated RVV bundle ABI or
  object/header output changes relative to the existing proved route.

## Technical Notes

- Relevant specs:
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`.
- Likely implementation surface:
  `tools/tcrv-translate/tcrv-translate.cpp` plus focused lit tests under
  `test/Target/TargetArtifactBundleExport/`.
- Existing reusable pipeline builder:
  `transforms::buildSourceArtifactFrontDoorPipeline`.
- Existing manual positive evidence lives in RVV/Toy/TensorExtLite target tests
  that pipe `tcrv-opt --tcrv-source-artifact-front-door-pipeline` into
  `tcrv-translate --tcrv-export-target-artifact-bundle`.
