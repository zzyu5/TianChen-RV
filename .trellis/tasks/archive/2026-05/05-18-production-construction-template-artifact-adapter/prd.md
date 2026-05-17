# Production construction-template artifact adapter consumption

## Goal

Promote the test-local construction-template object/header/bundle consumption
shape into reusable production include/lib code, then make the existing
production Template target-support path consume that adapter. The adapter must
bridge construction manifest evidence, selected-boundary validation,
plugin-owned EmitC route construction, runtime ABI contract, materialized EmitC
C++ emission, relocatable object export, declaration-only header export, and
object/header bundle registration without learning Template semantics.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Session-start `git status --short` was clean.
- Current HEAD at session start is
  `36403d8 plugin: bridge template consumer artifacts`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-18-template-consumer-materialized-artifact-bridge/prd.md`
  explicitly left the construction-template consumer object/header/bundle proof
  test-local.
- The current test-local `template_consumer` fixture in
  `test/Plugin/ConstructionProtocolCommonTest.cpp` owns the only reusable-looking
  construction-template artifact bridge shape.
- The production Template target support path in
  `lib/Target/Template/TemplateTargetSupportBundle.cpp` already uses the common
  target object/header bundle helper, but it still hand-builds the
  construction-template adapter logic locally.

## Requirements

- Add a production C++ construction-template artifact adapter in include/lib
  code.
- The adapter must be reusable by production extension-family target support
  code and must not live only in tests.
- Make the existing production Template target-support artifact path consume
  the adapter.
- The adapter may validate route identity, selected candidate shape,
  construction metadata, source-op/source-role/interface provenance, runtime
  ABI identity and ordered parameters, header/object route coherence, bundle
  records, and forbidden descriptor/direct-C/source-export/compute-body
  residue.
- The adapter must delegate family semantics to caller-supplied manifest,
  route config, selected candidate validation, route builder, header evidence,
  object exporter, and header exporter callbacks.
- The adapter must not synthesize compute C/C++ bodies from metadata, create a
  production TemplateConsumer plugin/dialect, add family-specific branches in
  common/core code, or infer Template/RVV/Toy/TensorExtLite semantics.
- The Template target support path must remain plugin-local and must still
  materialize through:
  selected Template role op -> Template-owned EmitC route -> common
  materialized EmitC module -> MLIR EmitC C++ emitter -> local object callback
  -> object-backed declaration header and bundle registration.
- The test-local TemplateConsumer fixture should be reduced toward consumer
  coverage of the production adapter instead of owning the only implementation.

## Acceptance Criteria

- Production changed files include reusable include/lib code, not only tests or
  Trellis metadata.
- Existing Template object/header/bundle registration goes through the new
  construction-template artifact adapter or a clearly factored production
  helper.
- The adapter is generic over construction-template config and does not branch
  on concrete extension family names for semantics.
- Template object exporter and object-backed header composite preserve the
  same route ids, artifact kinds, origin plugin, emission kind, runtime ABI
  identity, zero-argument runtime ABI signature, component group, external ABI
  name, handoff kind, source-op provenance, semantic role graph, and typed role
  evidence already tested today.
- Fail-closed behavior covers stale or missing construction metadata, missing
  materialized EmitC provenance, fallback-only or ambiguous selected paths,
  runtime ABI mismatch, header/object route mismatch, and forbidden
  descriptor/direct-C/source-export/compute-body metadata.
- Existing Template target artifact/export behavior remains green.
- Focused scans show no production `TemplateConsumer` leakage, no
  descriptor/direct-C/source-export authority, and no new common/core
  family-name semantic branch.
- `git diff --check` passes.

## Out Of Scope

- Production TemplateConsumer plugin, dialect, source front door, target
  translate route, or built-in bundle registration.
- RVV/IME/TensorExt/Offload/scalar feature semantics.
- New frontend lowering, high-level tensor/tile IR, hardware/runtime/
  correctness/performance claims, or `ssh rvv` evidence.
- Descriptor adapters, direct C semantic exporters, source-export routes,
  handwritten compute bodies, Python compiler-core behavior, compatibility
  wrappers, legacy modes, or common/core extension-specific semantic branches.

## Definition Of Done

- PRD and task context describe this round truthfully.
- Production adapter and Template consumer are implemented in C++ include/lib.
- The test-local proof consumes the production adapter where appropriate.
- Focused build/tests for construction protocol common coverage, target
  artifact export coverage, and relevant lit/FileCheck target surfaces pass or
  any blocker is recorded precisely.
- Residue scans prove no descriptor/direct-C/source-export authority,
  production TemplateConsumer leakage, or common/core family branch was
  introduced on touched surfaces.
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
  `.trellis/tasks/archive/2026-05/05-18-template-consumer-materialized-artifact-bridge/prd.md`.
- Main source surfaces inspected before implementation:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`,
  `lib/Plugin/Template/TemplateConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateEmitCRouteProvider.h`,
  `lib/Plugin/Template/TemplateEmitCRouteProvider.cpp`,
  `lib/Target/Template/TemplateTargetSupportBundle.cpp`,
  `test/Plugin/ConstructionProtocolCommonTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Completion Notes

- Added production adapter surface
  `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h` and
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp`.
- The adapter owns construction-template artifact consumption wiring over
  caller-supplied route config: config validation, object-backed header config
  construction, object/header bundle config construction, selected candidate
  validation, generated C++ export through the selected EmitC artifact front
  door, object packager callback invocation, declaration-header export, and
  object/header bundle exporter registration.
- The adapter remains generic over construction-template object/header
  artifact contracts. It does not branch on RVV, Toy, Template, TensorExtLite,
  Offload, Scalar, TemplateConsumer, dtype, shape, intrinsic spelling, runtime
  target, or hardware semantics.
- Rewired `lib/Target/Template/TemplateTargetSupportBundle.cpp` so production
  Template target support now supplies only Template-local config, validation,
  route builder, and object packager callback, then consumes the production
  adapter for object/header/bundle registration and export.
- Rewired the test-local TemplateConsumer artifact proof in
  `test/Plugin/ConstructionProtocolCommonTest.cpp` to consume
  `registerConstructionTemplateArtifactAdapterExporters`,
  `exportConstructionTemplateObjectArtifact`, and
  `exportConstructionTemplateHeaderArtifact` instead of owning the only
  object/header/bundle helper implementation.
- No production TemplateConsumer plugin, dialect, target route, source front
  door, compatibility wrapper, descriptor adapter, direct-C/source-export path,
  or Python compiler-core behavior was added.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-production-construction-template-artifact-adapter`
- `cmake --build build --target tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Target/Template|construction-protocol-common'` from `build/test` passed 5/123 selected tests.
- `cmake --build build --target check-tianchenrv -j2` passed 123/123 tests.
- `git diff --check`
- Production TemplateConsumer leakage scan:
  `rg -n "TemplateConsumer|template_consumer|template-consumer" include lib tools`
  returned no matches.
- Touched-surface forbidden-residue scan:
  `rg -n "descriptor-driven|descriptor|direct-C|direct C|source-export|source exporter|compute-body|Python compiler-core|python compiler-core|metadata-diagnostic" include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h lib/Target/ConstructionTemplateArtifactAdapter.cpp lib/Target/Template/TemplateTargetSupportBundle.cpp test/Plugin/ConstructionProtocolCommonTest.cpp`
  returned only fail-closed rejection strings and negative assertions in
  `test/Plugin/ConstructionProtocolCommonTest.cpp`.
- New production adapter family-branch scan:
  `rg -n "RVV|TensorExt|Toy|Offload|Scalar|rvv|tensorext|toy|offload|scalar|TemplateConsumer|template_consumer|template-consumer" include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h lib/Target/ConstructionTemplateArtifactAdapter.cpp`
  returned no matches.
