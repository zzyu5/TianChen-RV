# Template Materialized EmitC Object Bundle Packaging Bridge

## Goal

Complete the Template construction-template artifact workflow by extending the
already selected Template materialized EmitC route from generated C++ and a
declaration-only header into a coherent generated source, declaration header,
relocatable object, and bundle manifest packaging example.

Template remains a construction template. The object is only a local
relocatable packaging proof for generated C++ from the selected materialized
EmitC route. The bundle is only a manifest and file packaging bridge tying
source/header/object provenance to the same selected Template candidate.

## What I Already Know

- Current HEAD `1788519` already materializes the Template route through:
  Template capability MLIR, generic selection, `tcrv_template.compute_skeleton`,
  Template-owned `TCRVEmitCLowerableRoute`, common materialized EmitC module,
  MLIR EmitC C/C++ emitter, generated C++ compile evidence, and a
  declaration-only header artifact.
- `.trellis/spec/plugin-protocol/extension-family-plugin-template.md` requires
  extension-family plugin construction through archetype, semantic role graph,
  family declaration, common interface realization, EmitC route mapping, and
  evidence profile.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires selected artifact
  export to derive source/object/header/bundle output from a selected
  supported emission-plan candidate and the common materialized EmitC handoff,
  not descriptor metadata, direct-C printers, source-export routes, or
  family-name branches in core orchestration.
- The common target layer already provides
  `MaterializedEmitCObjectBundleArtifactConfig` and
  `registerMaterializedEmitCObjectBundleArtifactExporters`.
- RVV and TensorExtLite already use that common construction helper for object
  route registration, object-backed header composite registration, and bundle
  metadata. Template currently registers only a standalone header artifact and
  a generated C++ translate route.

## Requirements

- Preserve Template as a plugin-local construction template in the unified TCRV
  RISC-V MLIR system.
- Keep all compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/
  FileCheck. Python may not become compiler-core logic.
- Reuse the existing selected Template materialized EmitC route. Do not add new
  Template computation semantics beyond the current `compute_skeleton`
  sentinel route.
- Make the Template target support bundle register a relocatable object route
  plus an object-backed declaration-only header composite through the common
  materialized EmitC object/header bundle helper.
- The object route must emit generated C++ through the MLIR EmitC C/C++ emitter
  and compile that generated C++ into a relocatable object when local native
  clang support is available.
- The header route must remain declaration-only and must be validated against
  the same selected object candidate, route identity, runtime ABI identity,
  lowering boundary, and ordered runtime ABI parameter signature.
- Bundle export must produce object and header records tied to the same
  selected variant, origin plugin, route id, runtime ABI kind/name, lowering
  boundary, and construction protocol evidence.
- Candidate validation must fail closed for non-materialized modules, stale
  route metadata, fallback-only or unsupported candidates, multiple supported
  candidates, missing runtime ABI parameters, mismatched header/object route
  identity, and descriptor/source-export/direct-C/compute-body residue.

## Acceptance Criteria

- [x] Template construction protocol exposes a current object route identity,
      header route identity, component group, and object handoff kind without
      replacing the selected materialized EmitC route or adding new Template
      semantics.
- [x] Template target support registers one standalone relocatable object route
      and one object-backed header composite through the common
      `registerMaterializedEmitCObjectBundleArtifactExporters` helper.
- [x] Template generated C++ route remains available through
      `tcrv-template-emitc-to-cpp` and continues to require selected
      materialized EmitC route provenance before output.
- [x] Focused C++ target artifact tests prove Template object exporter shape,
      header composite shape, bundle metadata, runtime ABI signature reuse,
      duplicate/ambiguous/mismatch fail-closed behavior, and common route reuse.
- [x] Focused lit tests under `test/Target/Template` prove generated C++,
      header artifact, object artifact when local object clang support is
      available, and bundle manifest contents.
- [x] Negative coverage fails closed for non-materialized input, stale route
      metadata, fallback-only or unsupported candidates, duplicate supported
      candidates, missing runtime ABI parameters, mismatched header/object
      route identity, and descriptor/source-export/direct-C residue.
- [x] Bounded residue scans over touched Template plugin/target/tests show no
      descriptor-driven compute authority, no direct C semantic exporter, no
      source-export route, and no extension-specific core branch was introduced.

## Definition Of Done

- Task status and journal entries are truthful.
- Focused tests for Template plugin/target behavior pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the exact blocker is
  recorded.
- The completed task is finished/archived according to Trellis convention.
- One coherent commit records PRD, implementation, tests, and task finish if
  the task is complete.

## Out Of Scope

- New RVV, TensorExt, IME, Offload, scalar, or frontend lowering behavior.
- Runtime execution, hardware correctness, correctness harness, performance
  claim, or `ssh rvv` evidence for Template.
- Descriptor compatibility paths, direct C semantic exporters, source-export
  routes, metadata-only checklists, helper-only achievements, broad smoke
  matrices, or legacy wrappers.
- Making Template object packaging the default for real target plugins unless a
  selected materialized EmitC route and explicit target support opt in.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/*`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`,
  `lib/Plugin/Template/TemplateConstructionProtocol.cpp`,
  `lib/Plugin/Template/TemplateExtensionPlugin.cpp`,
  `include/TianChenRV/Target/Template/TemplateTargetSupportBundle.h`,
  `lib/Target/Template/TemplateTargetSupportBundle.cpp`,
  `lib/Target/TargetArtifactExport.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Lit surfaces:
  `test/Target/Template/`,
  `test/Conversion/EmitC/template-compute-skeleton-materialization*.mlir`.

## Completion Summary

- Template construction protocol now carries object route identity, header
  route identity, header artifact kind, bundle component group, and object
  handoff kind, with a verifier that keeps these fields tied to the current
  selected Template EmitC route.
- Template target support now registers the selected materialized EmitC object
  route and object-backed declaration header composite through the common
  materialized EmitC object/header bundle helper. The object callback emits
  generated MLIR EmitC C++ and locally compiles it with native `clang++` into a
  relocatable object when available.
- The bundle manifest packages the selected object and header records under the
  same selected variant, origin plugin, route/runtime ABI identity, component
  group, lowering boundary, construction protocol evidence, and object handoff
  kind. This remains a construction-template packaging proof only; it does not
  claim Template runtime correctness, performance, or hardware execution.
- Durable spec updates were added to
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md` and
  `.trellis/spec/lowering-runtime/emitc-route.md` to capture the Template
  object/header/bundle opt-in contract.

## Validation

- `cmake --build build --target tianchenrv-template-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target check-tianchenrv -j2`
- `git diff --check`
- Bounded residue scans over Template plugin/target/tests and common/core
  target surfaces found no descriptor-driven compute authority, direct C
  semantic exporter, source-export route, or extension-specific core branch.
- `clang-format` was not available in the local toolchain; C++ formatting was
  manually checked, and `git diff --check` passed.
