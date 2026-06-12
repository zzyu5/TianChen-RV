# Toy Extension Bundle Target Artifact Bridge

## Goal

Add a Toy plugin-local target artifact bridge that proves the existing Toy
construction protocol and Toy EmitC route provider can reach a target artifact
surface through the `ExtensionBundleRegistry` default front door.

The intended route is:

```text
Toy extension-family selected boundary
  -> Toy plugin-owned EmitC route provider
  -> common materialized EmitC handoff
  -> Toy target-support artifact exporter bundle
  -> built-in ExtensionBundleRegistry default target artifact registration
  -> Toy runtime ABI artifact evidence
```

## Why Now

Commit `69aaca3` made the built-in extension bundle registry the default front
door for tools, target artifact exporters, and translate routes. The remaining
bounded proof is not another registry helper: Toy already has plugin-local
construction and EmitC route surfaces, but its extension bundle does not publish
a target artifact exporter bundle. The default front door therefore proves RVV
and TensorExtLite target artifact routes, but not a non-RVV construction-template
plugin route.

## Current Repository Facts

- `.trellis/spec/index.md` requires compiler behavior to stay in the
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck stack and rejects descriptor-driven
  computation.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires common
  orchestration to go through plugin interfaces and registries, not
  family-specific core branches.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` and
  `.trellis/spec/lowering-runtime/emitc-route.md` require selected target
  artifact handoff to consume selected emission-plan candidates, runtime ABI
  metadata, and materialized EmitC provenance rather than descriptor/direct-C
  source authority.
- `ToyExtensionPlugin` already owns capability metadata, selected-boundary
  materialization, lowering-boundary validation, and
  `buildVariantEmitCLowerableRoute`.
- `ToyEmitCRouteProvider` already requires exactly one selected
  `tcrv_toy.compute_skeleton`, verifies `TCRVEmitCLowerableOpInterface`, and
  builds a bounded `TCRVEmitCLowerableRoute`.
- `ToyExtensionPlugin::buildVariantEmissionPlan` currently returns unsupported,
  so Toy cannot produce a supported target artifact candidate.
- `ToyExtensionPlugin::configureTargetSupportExtensionBundle` currently only
  adds `tcrv_toy`; it does not publish lowering-boundary or target artifact
  exporter bundle metadata.
- `lib/Target/` has RVV and TensorExtLite target-support bundle modules, but no
  Toy target support module.
- `TargetArtifactExport` already has a common selected EmitC artifact front
  door capable of materializing a selected route and verifying route/call
  source-op provenance before artifact packaging.
- The previous task
  `.trellis/tasks/archive/2026-05/05-17-extension-bundle-registry-default-plugin-front-door/prd.md`
  made the bundle registry the default consumer path for built-in target
  artifact exporters.

## Requirements

- Add a Toy target-support module under `include/TianChenRV/Target/Toy/` and
  `lib/Target/Toy/` that owns Toy target artifact exporter registration.
- Rewire `ToyExtensionPlugin::configureTargetSupportExtensionBundle` so the Toy
  extension bundle publishes Toy required dialect, Toy selected lowering
  boundary op, and Toy target artifact exporter bundle registration.
- Change the Toy emission plan from unsupported to a supported materialized
  EmitC target artifact route when the selected Toy variant is legal and the Toy
  route mapping is valid.
- The Toy target artifact exporter must validate the selected candidate against
  Toy construction protocol metadata: origin plugin, selected route id,
  selected variant, emission kind, artifact kind, lowering boundary, runtime
  ABI kind/name/value, runtime glue role, source-op provenance, and
  construction-protocol role metadata.
- The exported Toy artifact may be a bounded runtime ABI evidence/header surface;
  object/native compilation is not required in this round.
- Artifact evidence must carry origin plugin, selected variant, route id,
  runtime ABI name/kind/parameters, source op provenance, and construction
  protocol role information.
- Negative coverage must fail closed for absent/disabled Toy bundle routes,
  stale or missing EmitC route provenance, missing runtime ABI fields, multiple
  ambiguous Toy candidates, and descriptor/direct-C/source-export authority.
- Existing RVV and TensorExtLite artifact routes must keep flowing through the
  bundle registry unchanged.
- `tcrv-opt`, `tcrv-translate`, `lib/Target/Builtin`, and
  `TargetArtifactExport.cpp` must not gain Toy-specific compute branches.

## Acceptance Criteria

- [ ] The built-in `ExtensionBundleRegistry` publishes a Toy target artifact
      exporter bundle via `ToyExtensionPlugin::configureTargetSupportExtensionBundle`.
- [ ] Default built-in target artifact registration registers Toy through the
      bundle front door; RVV and TensorExtLite route counts/shape remain
      unchanged except for the intentional Toy addition.
- [ ] A positive Toy selected-boundary fixture reaches materialized EmitC
      handoff and a Toy target artifact output through the default
      `tcrv-export-target-header-artifact` route.
- [ ] Toy artifact output includes origin plugin, selected variant, route id,
      runtime ABI kind/name, runtime ABI parameter signature, source op
      provenance, and construction-protocol role metadata.
- [ ] Negative tests reject missing Toy bundle/plugin registration, disabled
      built-ins, stale or missing EmitC route provenance, missing runtime ABI
      fields, ambiguous Toy candidates, and descriptor/direct-C/source-export
      metadata.
- [ ] No descriptor-driven compute, direct C/source-export authority, Python
      compiler-core behavior, new independent backend, or Toy-specific
      core/tool semantic branch is introduced.

## Out Of Scope

- No new extension family or dynamic plugin loader.
- No RVV broadening and no hardware performance/correctness claim.
- No object/native compilation requirement for Toy in this round.
- No source-export route or descriptor/direct-C compatibility wrapper.
- No movement of target artifact semantics into plugin/common.
- No broad smoke matrix unless focused checks expose a build-system issue.

## Minimal Evidence

- Focused build/test for Toy plugin construction, Toy EmitC materialization,
  Toy target artifact export, and tcrv-opt/tcrv-translate front doors.
- Focused C++ coverage in `test/Target/TargetArtifactExportTest.cpp` for Toy
  bundle-front-door registration shape, candidate validation, disabled/missing
  plugin fail-closed behavior, and existing RVV/TensorExtLite preservation.
- lit coverage for Toy positive target artifact export and key fail-closed
  cases.
- Targeted scans over tools, built-in target registration, common target export,
  and Toy/target files proving no Toy-specific core/tool branch and no
  descriptor/direct-C/source-export authority.
- `git diff --check`.

## Technical Notes

- Read specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Main implementation files:
  `include/TianChenRV/Plugin/Toy/ToyConstructionProtocol.h`,
  `include/TianChenRV/Plugin/Toy/ToyEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/Toy/ToyExtensionPlugin.h`,
  `lib/Plugin/Toy/`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Conversion/EmitC/toy-template-materialization*.mlir`,
  `test/Transforms/Toy/`, and
  `test/Target/TargetArtifactExportTest.cpp`.
