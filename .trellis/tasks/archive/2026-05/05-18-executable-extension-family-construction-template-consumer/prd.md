# Executable extension-family construction template consumer

## Goal

Instantiate one minimal bounded executable extension-family construction
template consumer that proves the construction protocol can add a new plugin
instance through the common registry gate and common EmitC materializer without
adding common/core semantic branches.

This round uses a test-local/example plugin consumer rather than a production
hardware family. The consumer must still be real enough to own a construction
manifest, semantic role graph, family declaration, common-interface
realization, executable role step, route mapping, artifact metadata/evidence
profile, selected boundary fixture, and plugin-owned
`TCRVEmitCLowerableRoute`.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean before this task.
- Current HEAD before this task was
  `07931ab plugin: gate remaining construction builtins`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-18-05-18-executable-construction-gate-remaining-builtins/prd.md`
  closed the registry-time construction conformance gate for RVV,
  TensorExtLite, Toy, and Template.
- Live code shows `ExtensionPluginRegistry::registerPlugin()` calls
  `ExtensionPlugin::verifyExecutableConstructionConformance()` before
  accepting plugins.
- Live common construction code already validates manifests, typed-role graph
  realization, executable role steps, selected role sequences, selected
  lowering-boundary conformance, and construction artifact metadata in an
  extension-family-agnostic way.
- Live common EmitC code materializes a verified in-memory MLIR EmitC module
  from `TCRVEmitCLowerableRoute`.

## Requirements

- Add a bounded template-consumer plugin fixture that is a new plugin instance,
  not an RVV/Toy/Template/TensorExtLite reuse disguised as success.
- Keep the new consumer test/example-local. It must not become a production
  hardware backend, runtime path, target exporter, object/header/bundle route,
  or runtime/correctness/performance claim.
- Give the consumer an explicit construction manifest, semantic role graph,
  extension family declaration, common-interface realization, typed-role
  realization, executable role step, EmitC route mapping, construction artifact
  metadata, and evidence profile.
- Register the valid consumer through the same registry-time executable
  construction conformance hook used by builtins.
- Prove stale manifest, stale typed-role/interface realization, stale route
  mapping, and stale artifact metadata variants fail before proposal or route
  use.
- Materialize one selected boundary/source-front-door fixture into a
  plugin-owned `TCRVEmitCLowerableRoute`.
- Prove the common materializer produces a verified MLIR EmitC module from the
  consumer route.
- Do not add RVV, Toy, Template, TensorExtLite, Scalar, Offload, or consumer
  family-name semantic branches in common construction, registry, target
  export, or core orchestration code.
- Do not add descriptor tables, descriptor adapters, direct C/source-export
  semantic exporters, Python compiler-core behavior, or metadata as compute
  authority.

## Acceptance Criteria

- A valid executable construction template-consumer plugin registers
  successfully through `ExtensionPluginRegistry::registerPlugin()`.
- Registry-time conformance rejects a stale manifest before plugin acceptance.
- Registry-time conformance rejects stale typed-role/interface realization
  before plugin acceptance.
- Registry-time conformance rejects stale route mapping before plugin
  acceptance.
- Registry-time conformance rejects stale artifact metadata before plugin
  acceptance.
- The selected boundary fixture passes the common selected-boundary/role
  conformance checks and builds a plugin-owned `TCRVEmitCLowerableRoute`.
- The route verifies and `verifyTCRVEmitCLowerableRouteMaterializesToEmitC()`
  succeeds for the consumer route.
- Existing RVV, TensorExtLite, Toy, and Template construction-capable builtin
  registration coverage remains intact.
- Focused changed-surface scans show no descriptor/direct-C/source-export
  authority and no family-name semantic branch in common construction or
  registry code.
- `git diff --check` passes.
- Focused C++ build/test targets pass. Run `check-tianchenrv` if practical;
  otherwise record the exact blocker.

## Out Of Scope

- New RVV SEW/LMUL/dtype/op coverage.
- TensorExt, IME, Offload, Scalar, Toy, Template, or RVV feature expansion.
- Target object/header/bundle packaging for the new consumer.
- Runtime execution, correctness, performance, or `ssh rvv` evidence for the
  new consumer.
- Descriptor-driven computation, descriptor adapters, direct C/source-export
  semantic exporters, generated source authority, or Python compiler-core
  behavior.
- Common/core extension-family semantic branches.
- Documentation-only manifest checklists that do not pass through registry,
  route construction, and common EmitC materialization.

## Definition Of Done

- PRD and task context describe this round truthfully.
- Code changes implement the coherent submodule through registry gate plus
  materialized EmitC module.
- Focused C++ tests cover valid consumer registration, stale-input failures,
  selected boundary route construction, and common EmitC materialization.
- Existing built-in construction plugin registration tests remain green.
- `git diff --check` passes.
- Residue scans prove no descriptor/direct-C/source-export authority or
  extension-specific common semantic branch was introduced on touched
  surfaces.
- Trellis task is finished/archived when complete.
- One coherent commit is created when the task is complete.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/index.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-18-05-18-executable-construction-gate-remaining-builtins/prd.md`.
- Main source surfaces inspected before implementation:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  Template plugin construction/route code,
  `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableInterface.h`,
  `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableMaterializer.h`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`, and
  `test/Plugin/ConstructionProtocolCommonTest.cpp`.
- Implementation choice: use a C++ test-local/example plugin consumer in
  `ConstructionProtocolCommonTest.cpp`. This proves the construction workflow
  for a new bounded plugin instance while avoiding production plugin registry
  expansion, target exporter registration, hardware claims, and family-specific
  common branches.

## Completion Notes

- Added a test-local `template-consumer-plugin` fixture in
  `ConstructionProtocolCommonTest.cpp` as a new bounded plugin instance.
- The consumer owns its own construction manifest, single-role semantic role
  graph, family declaration, common-interface realization, typed-role
  realization, executable role step, EmitC route mapping, construction artifact
  metadata, evidence profile, selected boundary fixture, and plugin-owned
  `TCRVEmitCLowerableRoute`.
- Valid consumer registration now passes through
  `ExtensionPluginRegistry::registerPlugin()` and the same executable
  construction conformance gate as builtins.
- Added fail-closed registry coverage for stale consumer manifest,
  stale typed-role/interface realization, stale route mapping, and stale
  artifact metadata.
- Added route construction coverage proving the selected boundary fixture is
  checked through common selected role-sequence and selected-boundary
  conformance before a plugin-owned `TCRVEmitCLowerableRoute` is returned.
- Added common materializer coverage proving the consumer route verifies and
  materializes to a verified MLIR EmitC module.
- Updated the common construction test target to link
  `TianChenRVConversionEmitC` and `MLIRIR`, because the test now directly
  exercises the common EmitC materializer.
- Self-repair performed: the first fail-closed table used nested
  `initializer_list<StringRef>` temporaries that produced an invalid fragment
  during the test. Rewrote those cases as explicit scoped assertions.
- No production plugin, target exporter, object/header/bundle packaging path,
  RVV runtime path, descriptor path, direct-C/source-export path, Python
  compiler-core behavior, or common/core family semantic branch was added.
- Spec-update judgment: no `.trellis/spec/` change was needed because the
  existing executable construction template and EmitC route specs already
  describe this behavior.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-executable-extension-family-construction-template-consumer`
- `cmake --build build --target tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `cmake --build build --target tianchenrv-toy-extension-plugin-test tianchenrv-template-extension-plugin-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- Common construction/registry family-name scan:
  `rg -n "RVV|TensorExt|Toy|Template|Offload|Scalar|template-consumer|TemplateConsumer|rvv|tensorext|toy|template|offload|scalar|consumer" include/TianChenRV/Plugin/ConstructionProtocol.h lib/Plugin/Construction/ConstructionProtocol.cpp lib/Plugin/ExtensionPlugin.cpp`
  returned no matches.
- Touched-source descriptor/direct-C/source-export scan:
  `rg -n "descriptor-driven|descriptor|direct-C|direct C|source-export|source exporter|Python compiler-core|python compiler-core" test/Plugin/ConstructionProtocolCommonTest.cpp test/CMakeLists.txt`
  returned no matches.
- Production-surface consumer scan:
  `rg -n "template_consumer|template-consumer|TemplateConsumer" include lib tools`
  returned no matches.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 122/122 lit
  tests.
