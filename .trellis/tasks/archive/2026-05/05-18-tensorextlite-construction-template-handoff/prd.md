# TensorExtLite construction-template production handoff

## Goal

Turn TensorExtLite into the production non-RVV proof that the
Extension-Family Plugin Construction Protocol is executable and reusable. The
selected TensorExtLite typed role sequence must reach materialized EmitC,
relocatable object export, declaration-only header export, and object/header
bundle packaging through the common construction-template surfaces rather than
TensorExtLite-only target preflight logic.

## What I already know

- Repository state before edits: `/home/kingdom/phdworks/TianchenRV`, clean
  worktree, HEAD `633ecf0 test: prove rvv source bundle ssh abi`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
- The previous archived RVV task proved the generated RVV source bundle ABI on
  real `ssh rvv`, and this round must not become another RVV evidence-only
  task.
- The relevant specs define TensorExtLite as a bounded fragment-MMA
  construction-template proof. It is not a general TensorExt frontend, runtime
  execution path, source-export route, correctness claim, or performance path.
- Current TensorExtLite already has source-front-door materialization,
  construction manifest/role graph metadata, ordered typed role ops,
  materialized EmitC route provenance, object/header/bundle target tests, and
  fail-closed checks for stale source-front-door metadata and missing
  lowering-boundary markers.
- The remaining production handoff gap is that TensorExtLite target export
  performs selected lowering-boundary lookup and conformance as private
  TensorExtLite target preflight logic. The common construction-template
  artifact adapter should own this reusable validation responsibility, and
  TensorExtLite should consume it in its default object/header/bundle/EmitC
  export path.
- TensorExtLite target support also must publish its selected lowering-boundary
  op through the ExtensionBundle target-support manifest, matching the common
  handoff shape already used by RVV, Toy, and Template.

## Requirements

1. Keep the production path:
   TensorExtLite source marker or already materialized selected role sequence
   -> selected variant + ordered TensorExtLite role ops
   -> materialized TensorExtLite lowering-boundary marker
   -> TensorExtLite-owned EmitC route provenance
   -> common construction-template adapter
   -> relocatable object, declaration-only header, and coherent bundle.
2. Move selected materialized lowering-boundary validation into the common
   construction-template artifact adapter. The common adapter must validate the
   selected variant, origin plugin, role, status, required capabilities,
   lowering boundary op name, and route-local extra attributes before object,
   header, bundle, or EmitC-to-C++ output.
3. TensorExtLite production export must configure and consume that common
   adapter validation instead of using a TensorExtLite-only target preflight for
   lowering-boundary conformance.
4. TensorExtLite must register its construction route lowering-boundary op on
   the ExtensionBundle target-support surface.
5. The object and header artifacts in a TensorExtLite bundle must continue to
   share one selected variant, origin plugin, runtime ABI name, ordered ABI
   parameters, materialized EmitC route, component group, object handoff kind,
   and construction evidence metadata.
6. Missing or stale archetype, role graph, common-interface realization,
   EmitC route mapping, runtime ABI metadata, bundle mapping, selected
   lowering-boundary metadata, route provenance, source-front-door consumption,
   or forbidden descriptor/direct-C/source-export residue must fail closed.
7. Keep common/core orchestration family-neutral. Do not add core or translate
   branches on TensorExtLite, RVV, Toy, Template, dtype, shape, runtime, or
   vendor semantics.

## Acceptance Criteria

- [x] TensorExtLite object/header/bundle/EmitC-to-C++ export validates the
      selected `tcrv_tensorext_lite.lowering_boundary` through the common
      construction-template artifact adapter.
- [x] TensorExtLite target support publishes
      `tcrv_tensorext_lite.lowering_boundary` on its ExtensionBundle
      target-support manifest.
- [x] Focused C++ tests prove TensorExtLite consumes the common adapter
      selected-boundary validation and that stale boundary attributes fail
      closed through the common path.
- [x] Focused lit tests continue proving TensorExtLite source-front-door bundle
      export produces one coherent materialized EmitC object/header bundle and
      missing boundary fails before output.
- [x] Template/RVV/Toy regression coverage still shows the common construction
      and target artifact surfaces remain extension-family neutral.
- [x] Targeted residue scans over touched common/plugin/target/test surfaces
      show no descriptor-driven compute authority, direct C semantic exporter,
      source-export route, Python compiler-core path, compatibility wrapper, or
      family-specific common/core branch was introduced.

## Non-Goals

- No new RVV dtype, LMUL, op-family, source-front-door expansion, or new `ssh`
  proof.
- No TensorExtLite runtime execution, correctness, performance, general
  frontend lowering, linalg lowering, scalar fallback compute, descriptor
  adapter, direct C semantic exporter, source-export route, compatibility
  wrapper, or legacy mode.
- No new independent backend dialect or TensorExtLite/RVV/Toy/Template semantic
  branch in common/core orchestration.
- No docs-only/template-only/metadata-only completion.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-tensorextlite-construction-template-handoff`
- Focused build targets for touched tests and tools, especially
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-tensorext-lite-extension-plugin-test`, `tcrv-opt`, and
  `tcrv-translate`.
- Focused C++ tests:
  `./build/bin/tianchenrv-construction-protocol-common-test`,
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`, and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit tests under `test/Target/TensorExtLite`,
  `test/Target/TargetArtifactBundleExport`, and relevant construction
  protocol / source-front-door tests.
- `git diff --check`.
- Targeted residue scans over touched common/plugin/target/translate/test
  surfaces.
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Definition of Done

- TensorExtLite's default production artifact export path consumes the common
  construction-template selected-boundary validation.
- The selected TensorExtLite artifact handoff remains materialized-EmitC backed,
  bundle-coherent, and fail-closed for missing/stale construction protocol
  pieces.
- Trellis task status, journal/archive notes, and one coherent commit truthfully
  reflect the completed scope and any remaining rebuild gaps.

## Completion Notes

- Common construction-template artifact adapter now owns selected
  lowering-boundary validation before header, object, and EmitC-to-C++ output.
- TensorExtLite removed its private selected-boundary target preflight and
  configures the common adapter with its fragment ABI and object handoff
  boundary expectations.
- TensorExtLite target-support ExtensionBundle now publishes the selected
  lowering-boundary op alongside the required dialect and target artifact
  exporter bundle.
- TemplateConsumer C++ coverage proves stale selected-boundary attributes fail
  closed through the common adapter; TensorExtLite, Template, Toy, and RVV
  focused tests continue passing.
- Remaining rebuild gap: TensorExtLite remains a bounded construction-template
  artifact proof, not a runtime correctness or performance proof.

## Technical Notes

- Specs read for PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous archived PRD read:
  `.trellis/tasks/archive/2026-05/05-18-rvv-source-bundle-ssh-runtime-abi-proof/prd.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/ExtensionBundle.h`,
  `lib/Plugin/ExtensionBundle.cpp`,
  `include/TianChenRV/Plugin/TensorExtLite/`,
  `lib/Plugin/TensorExtLite/`,
  `include/TianChenRV/Target/TensorExtLite/`,
  `lib/Target/TensorExtLite/`,
  `include/TianChenRV/Plugin/Template/`,
  `lib/Plugin/Template/`,
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp`,
  `test/Target/TensorExtLite/`, and
  `test/Target/TargetArtifactBundleExport/`.
