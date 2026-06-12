# Template-consumer materialized artifact bridge

## Goal

Carry the bounded `template-consumer-plugin` workflow proven by commit
`c15a8cc` from registry-time executable construction conformance and common
EmitC materialization into the target artifact boundary: generated C++ through
the MLIR EmitC emitter, a locally compile-checked relocatable object, a
declaration-only callable C header, and coherent object/header bundle metadata.

The consumer remains test-local. The purpose is to prove the construction
template can reach the current common materialized EmitC artifact APIs without
turning TemplateConsumer into a production hardware family or adding
descriptor/direct-C/source-export authority.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- The worktree was clean at session start.
- Current HEAD before this task is
  `c15a8cc plugin: add executable template consumer proof`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-18-executable-extension-family-construction-template-consumer/prd.md`
  ended with a test-local `template-consumer-plugin` in
  `test/Plugin/ConstructionProtocolCommonTest.cpp`.
- That consumer already owns a construction manifest, semantic role graph,
  family declaration, typed-role realization, executable role step, route
  mapping, construction artifact metadata, selected boundary fixture, and
  plugin-owned `TCRVEmitCLowerableRoute`.
- The current production Template target path already shows the intended
  common helper shape:
  `registerMaterializedEmitCObjectBundleArtifactExporters(...)` over a
  `MaterializedEmitCObjectBundleArtifactConfig`.
- The relevant long-term spec now includes `Scenario: Template Object And
  Bundle Packaging Bridge`, which explicitly requires object-backed header and
  bundle packaging from a selected materialized EmitC route.

## Requirements

- Extend the same bounded `template-consumer-plugin` fixture, not a different
  Template/Toy/RVV/TensorExtLite route, to prove target artifact packaging.
- Keep the consumer test-local unless a small reusable adapter is needed.
- Use the real common target artifact APIs:
  `SelectedEmitCArtifactRouteConfig`,
  `emitSelectedEmitCArtifactCppSource`,
  `exportMaterializedEmitCHeaderArtifact`,
  `registerMaterializedEmitCObjectBundleArtifactExporters`, and bundle record
  collection/export helpers.
- The target path must be selected by real `tcrv.exec` selected-path and
  emission-plan diagnostics, not by scanning arbitrary metadata.
- The plugin-owned route builder must validate the selected consumer boundary
  before producing the `TCRVEmitCLowerableRoute`.
- The object route must compile MLIR-EmitC-emitted C++ locally with `clang++`
  into a non-empty relocatable object when local `clang++` is available.
- The header route must be declaration-only and derived from the same selected
  object candidate.
- The bundle metadata must tie object and header records to the same selected
  variant, origin plugin, route id, runtime ABI, construction protocol,
  source-op provenance, typed role realization, component group, and handoff
  kind.
- Negative coverage must fail closed for stale route/artifact metadata,
  fallback-only selection, ambiguous selection, missing materialized EmitC
  provenance, object/header identity mismatch, runtime ABI mismatch, and
  descriptor/direct-C/source-export/compute-body residue.
- Do not add common/core semantic branches for the consumer family name.
- Do not add descriptor adapters, source-export routes, direct C semantic
  exporters, handwritten C compute bodies, Python compiler-core behavior,
  compatibility wrappers, production hardware features, runtime/correctness/
  performance claims, or `ssh rvv` evidence.

## Acceptance Criteria

- A valid consumer selected-boundary fixture produces a verified
  materialized EmitC module through the common selected EmitC artifact front
  door.
- The same selected fixture emits C++ text through the MLIR EmitC C/C++
  emitter.
- The emitted C++ is locally compiled with `clang++` into a non-empty
  relocatable object artifact.
- A declaration-only `runtime-callable-c-header` artifact is generated from
  the same selected object candidate and contains no descriptor/direct-C/
  source-export/compute-body residue.
- The registered common object/header bundle helper produces one object
  exporter and one object-backed header composite for the consumer.
- Bundle records or bundle output preserve selected variant, role, origin
  plugin, object route id, header route id, runtime ABI kind/name, component
  group, object handoff kind, construction protocol, source-op provenance,
  source role, source op interface, semantic role graph, and typed role
  realization.
- Negative coverage rejects stale route metadata, stale source-op/interface
  metadata, missing metadata, metadata-only artifact kind, fallback role,
  ambiguous candidates, runtime ABI parameter mismatch, header/object route
  identity mismatch, missing route provenance, and forbidden descriptor/direct-C/
  source-export/compute-body metadata.
- Existing built-in construction and target artifact tests remain green.
- Focused scans show no descriptor/direct-C/source-export authority and no new
  consumer-specific common/core branch.
- `git diff --check` passes.

## Out Of Scope

- Production TemplateConsumer plugin, dialect, source front door, target
  translate route, or built-in bundle registration.
- New RVV/Toy/Template/TensorExtLite/Offload/Scalar feature semantics.
- New SEW/LMUL/dtype/op families, high-level frontend lowering, or production
  hardware backend.
- Runtime execution, correctness, performance, link success, or `ssh rvv`
  claims.
- Descriptor-driven computation, direct C/source-export semantic exporters,
  handwritten C compute bodies, source-only generic front doors, compatibility
  wrappers, or legacy modes.

## Definition Of Done

- PRD and task context describe this round truthfully.
- The test-local consumer reaches materialized EmitC C++ emission, local object
  packaging, declaration-only header packaging, and bundle metadata through
  common target artifact APIs.
- Focused positive and fail-closed C++ coverage exercises the artifact bridge.
- Existing relevant C++ tests and lit/FileCheck target tests pass or any
  blocker is recorded precisely.
- Residue scans prove no descriptor/direct-C/source-export authority or
  extension-specific common/core branch was introduced on touched surfaces.
- Trellis task is finished/archived when complete.
- One coherent commit is created if the task is complete.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-18-executable-extension-family-construction-template-consumer/prd.md`.
- Main source surfaces inspected before implementation:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`,
  `lib/Plugin/Template/TemplateConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateEmitCRouteProvider.h`,
  `lib/Plugin/Template/TemplateEmitCRouteProvider.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/Template/TemplateTargetSupportBundle.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`, and
  `test/Target/Template/*`.
- Implementation choice: keep the consumer test-local and extend its existing
  C++ fixture to call common target artifact APIs directly. This avoids
  introducing a production family while proving the real artifact boundary.

## Completion Notes

- Extended the existing test-local `template-consumer-plugin` fixture in
  `ConstructionProtocolCommonTest.cpp`; no production TemplateConsumer plugin,
  dialect, bundle, source front door, or target translate route was added.
- Added a selected artifact fixture with real `tcrv.exec.kernel`,
  selected-path diagnostic, supported emission-plan diagnostic, and the
  existing `tcrv_template_consumer.compute_sentinel` selected boundary.
- Added a plugin-owned target route builder that resolves exactly one selected
  consumer boundary from the `VariantEmitCLowerableRequest`, validates the
  selected role/boundary metadata through the common construction conformance
  helpers, and then returns the same `TCRVEmitCLowerableRoute` used by the
  prior consumer materialization proof.
- Registered the consumer object/header route through the real
  `registerMaterializedEmitCObjectBundleArtifactExporters(...)` helper.
- Positive coverage now proves:
  materialized EmitC module handoff, MLIR EmitC C++ emission, local
  `clang++ -c` object packaging, declaration-only header output, bundle record
  collection, and complete bundle index output.
- Fail-closed coverage now rejects missing/stale construction artifact
  metadata, stale source interface metadata, `metadata-diagnostic` artifact
  kind, fallback-only selected role, stale runtime ABI signature, direct-C/
  descriptor/source-export/compute-body residue, object/header route identity
  mismatch, ambiguous header candidates, and missing route source-op
  provenance in the materialized EmitC handoff.
- Added `test/Plugin/construction-protocol-common.test` so the construction
  common C++ binary is executed by lit and included in `check-tianchenrv`.
- Spec-update judgment: no `.trellis/spec/` edit was needed. The already-read
  `Template Object And Bundle Packaging Bridge`, selected EmitC artifact
  handoff, and emission runtime contracts already describe the behavior
  implemented here.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-template-consumer-materialized-artifact-bridge`
- `cmake --build build --target tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target check-tianchenrv -j2` passed 123/123 tests.
- `git diff --check`
- Touched-surface forbidden-residue scan:
  `rg -n "descriptor-driven|descriptor|direct-C|direct C|source-export|source exporter|compute-body|Python compiler-core|python compiler-core" test/Plugin/ConstructionProtocolCommonTest.cpp test/Plugin/construction-protocol-common.test test/CMakeLists.txt test/lit.cfg.py`
  returned only fail-closed rejection strings and negative assertions.
- Production consumer residue scan:
  `rg -n "template_consumer|template-consumer|TemplateConsumer" include lib tools`
  returned no matches.
- Common/core family-branch scan:
  `rg -n "RVV|TensorExt|Toy|Template|Offload|Scalar|template_consumer|template-consumer|TemplateConsumer|rvv|tensorext|toy|template|offload|scalar|consumer" include/TianChenRV/Plugin/ConstructionProtocol.h lib/Plugin/Construction/ConstructionProtocol.cpp lib/Target/TargetArtifactExport.cpp`
  returned no matches.
