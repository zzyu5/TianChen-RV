# Toy Materialized EmitC Object-Bundle Bridge

## Goal

Make Toy the next concrete extension-family consumer of the materialized EmitC
object/header/bundle construction pattern. The Toy selected source-front-door
or plugin-selected compute_skeleton path must move from a header-only target
artifact surface to a coherent object, declaration header, and bundle route
through the common target artifact exporter helpers.

This is a Toy target artifact bridge. It proves that a concrete non-RVV
extension family can consume the common materialized EmitC object-bundle route
without descriptor authority, direct-C/source-export behavior, or a
family-specific semantic branch in common target code.

## What I Already Know

- Current HEAD `9b4c218` already makes Template publish a materialized EmitC
  object route, object-backed declaration header composite, and bundle through
  `registerMaterializedEmitCObjectBundleArtifactExporters`.
- Toy already has a capability/source front door, construction protocol,
  typed `tcrv_toy.compute_skeleton` boundary, plugin-owned
  `TCRVEmitCLowerableRoute`, runtime ABI parameter, and declaration-only
  header exporter.
- `TargetArtifactExportTest` currently treats Toy object export as unsupported;
  this is now the intentional surface to replace.
- `.trellis/spec/plugin-protocol/extension-family-plugin-template.md` requires
  extension-family plugin construction through archetype, semantic role graph,
  family declaration, common interface realization, EmitC route mapping, and
  evidence profile.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires target artifact
  object/header/bundle output to derive from a selected supported
  emission-plan candidate and verified materialized EmitC handoff, not from
  descriptor metadata, direct-C printers, source-export routes, or family-name
  branches.

## Requirements

- Preserve Toy as one extension family inside unified TianChen-RV MLIR; do not
  turn it into an independent backend.
- Keep compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
  Do not add Python compiler-core logic.
- Reuse the existing Toy source front door, Toy construction protocol,
  typed `tcrv_toy.compute_skeleton` boundary, Toy EmitC route provider, and
  common materialized EmitC target artifact helpers.
- Change the Toy selected materialized EmitC route to publish a
  `riscv-elf-relocatable-object` candidate as the object handoff authority.
- Add Toy construction route metadata for the object-backed header route,
  header artifact kind, bundle component group, object handoff kind, and any
  verification required to keep those fields coherent.
- Register Toy target support through the common materialized EmitC
  object/header bundle helper, with one standalone object exporter and one
  object-backed declaration-only header composite.
- The object exporter must emit generated C++ through the MLIR EmitC C/C++
  emitter and compile it locally into a nonempty relocatable object when
  native `clang++` is available.
- The header exporter must remain declaration-only and validate against the
  same selected object candidate, selected variant, origin plugin, object route
  id, header route id, runtime ABI kind/name, ordered runtime ABI parameter,
  component group, lowering boundary, construction protocol, source-op
  provenance, semantic role graph, typed-role evidence, and object handoff.
- Existing registry/composite exporter tests must no longer assert that Toy
  object export is unsupported.
- Negative coverage must fail closed for fallback-only selection, missing
  materialized EmitC provenance, stale header/object candidate metadata,
  mismatched route/runtime ABI fields, multiple selected supported Toy
  candidates, and descriptor/source-export/direct-C route residue.

## Acceptance Criteria

- [x] Toy construction protocol exposes object route identity, header route
      identity, header artifact kind, component group, object handoff kind,
      and runtime ABI parameter identity through verified C++ route metadata.
- [x] Toy target support registers a standalone relocatable object exporter
      and an object-backed declaration header composite through
      `registerMaterializedEmitCObjectBundleArtifactExporters`.
- [x] A positive Toy source-front-door path emits a nonempty relocatable object
      from the materialized EmitC route and MLIR EmitC C/C++ emitter.
- [x] The Toy header route remains declaration-only and records the Toy
      selected variant, runtime ABI parameter, origin plugin, selected route,
      source-op/interface provenance, construction protocol, semantic role
      graph, and typed-role evidence.
- [x] The Toy bundle index ties object and header records to the same selected
      candidate, origin plugin, object/header routes, materialized EmitC
      runtime ABI identity, runtime ABI parameter, component group, and object
      handoff.
- [x] Focused C++ target artifact tests prove Toy exporter shape, header
      composite shape, bundle metadata, runtime ABI parameter reuse,
      duplicate/ambiguous/mismatch fail-closed behavior, and common helper
      reuse.
- [x] Focused lit tests under `test/Target/Toy` prove Toy generated object,
      declaration header, bundle manifest, and stale-provenance rejection.
- [x] Bounded residue scans over Toy target/plugin/tests and common translate
      surfaces show no descriptor-driven compute authority, no direct-C
      semantic exporter, no source-export route, and no extension-specific
      common target branch introduced.

## Definition Of Done

- Task status and journal entries are truthful.
- Focused Toy/target artifact tests pass.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; if it is blocked, the exact blocker
  is recorded.
- The completed task is finished/archived according to Trellis convention.
- One coherent commit records PRD, implementation, tests, and task finish if
  the task is complete.

## Out Of Scope

- RVV, TensorExtLite, Template, IME, Offload, scalar, or frontend rewrite work,
  except compile fallout from shared declarations.
- New extension families, new arithmetic families, descriptor compatibility
  paths, direct C semantic exporters, source-export routes, metadata-only
  target artifacts, broad smoke matrices, performance claims, or RVV runtime
  claims.
- Restoring deleted RVV descriptor/direct-C routes.
- Adding a Toy target translate route unless compile fallout proves it is
  required for this object/header/bundle bridge.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task reference:
  `.trellis/tasks/archive/2026-05/05-17-template-emitc-object-bundle/prd.md`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/Toy/ToyConstructionProtocol.h`,
  `lib/Plugin/Toy/ToyConstructionProtocol.cpp`,
  `lib/Plugin/Toy/ToyEmitCRouteProvider.cpp`,
  `lib/Plugin/Toy/ToySourceFrontDoor.cpp`,
  `include/TianChenRV/Target/Toy/ToyTargetSupportBundle.h`,
  `lib/Target/Toy/ToyTargetSupportBundle.cpp`,
  `lib/Target/Template/TemplateTargetSupportBundle.cpp`,
  `lib/Target/TargetArtifactExport.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Lit surfaces:
  `test/Target/Toy/`.

## Completion Summary

- Toy construction protocol now treats
  `toy-template-compute-emitc-route` as a materialized EmitC object route with
  a separate `.header` route, `runtime-callable-c-header` header kind,
  Toy-specific bundle component group, object handoff kind, and generated C++
  compile evidence in the construction profile.
- Toy EmitC route construction now provides the runtime callee declaration
  needed by the MLIR EmitC C/C++ emitter and local `clang++` relocatable object
  packaging.
- Toy target support now registers through
  `registerMaterializedEmitCObjectBundleArtifactExporters`, producing one
  standalone object exporter and one object-backed declaration header
  composite from the same selected Toy object candidate.
- Toy object/header/bundle tests no longer assert that the object front door is
  unsupported. They now prove source-front-door object export, declaration
  header output, bundle index coherence, stale provenance rejection,
  fallback-only rejection, ambiguous candidate rejection, and descriptor /
  source-export / direct-C residue rejection.
- The lowering-runtime spec now records the durable rule that Toy consumes the
  common declaration-header helper through an object-backed composite when the
  selected object candidate is the handoff authority.

## Validation

- `cmake --build build --target tianchenrv-toy-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Target/Toy|Transforms/Toy|ExecutionPlanning/execution-planning-pipeline-toy|Conversion/EmitC/toy-template|Dialect/Toy'` from `build/test`
- `cmake --build build --target check-tianchenrv -j2`
- `git diff --check`
- Targeted residue scans over Toy plugin/target/tests and common
  `TargetArtifactExport` surfaces found only negative guard/check strings and
  common fail-closed metadata rejection; no Toy-specific branch exists in
  common target export code.
