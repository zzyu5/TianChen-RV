# Executable construction protocol registry gate

## Goal

Make executable construction protocol conformance a reusable production gate
at the plugin registry / bundle setup boundary before a plugin can publish
selected executable construction artifacts. The gate must validate the common
construction shape shared by RVV, TensorExtLite, Toy, Template, and future
families while leaving concrete extension semantics inside each owning plugin.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean before this task.
- Current HEAD is `c2a4bd1 plugin: adopt rvv construction conformance`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Direction Brief before source edits.
- The previous RVV task moved the bounded i32m1 add/sub/mul path onto common
  selected role-sequence, selected lowering-boundary, and construction
  artifact-metadata conformance APIs.
- RVV remains the first real executable plugin path, but its construction
  conformance is still mostly hand-called from RVV code.
- Existing non-RVV construction-template surfaces already share the common
  manifest / typed-role model; TensorExtLite is the best reusable non-RVV proof
  because it has selected role sequence, materialized EmitC route, and object
  artifact metadata.

## Requirements

- Add a common construction-conformance gate that production code can call from
  plugin registration, extension-bundle setup, emission readiness, or emission
  plan construction before executable construction artifacts are published.
- Keep the gate extension-agnostic. It may validate construction protocol
  version, archetype, family declaration, semantic role graph, typed-role
  realization, common-interface realization, selected role-sequence
  completeness/order, selected lowering-boundary conformance inputs, and
  construction artifact metadata shape.
- Do not move RVV/TensorExtLite/Toy/Template semantics into common code. The
  common gate must not learn RVV SEW/LMUL/VL, RVV intrinsic names, arithmetic
  meaning, TensorExt fragment semantics, IME/offload semantics, or
  target-specific packaging rules.
- RVV must remain the first real user. The existing RVV i32m1 add/sub/mul path
  must still reach its materialized EmitC object/header/bundle route.
- Add at least one lightweight non-RVV proof that the same gate is reusable
  without RVV branches. Prefer TensorExtLite or Toy construction-template
  coverage using existing plugin surfaces.
- Invalid manifest, missing typed role, stale interface realization,
  duplicate/out-of-order selected role step, missing selected boundary
  conformance input, stale artifact metadata, and unsupported artifact kind
  must fail closed before selected emission plan or target exporter
  publication.
- The common API must be a production-consumed hook/helper with explicit
  callers, not a report-only checklist or metadata dump.
- Implementation stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only support Trellis/tooling/probes.
- Do not add new RVV dtype/op families, high-level source lowering, TensorExt
  or IME executable semantics, new EmitC route kinds, direct C semantic
  exporters, descriptor adapters, compatibility routes, Python compiler-core
  behavior, or family-name semantic branches in common/core code.

## Acceptance Criteria

- A common gate API exists under the plugin construction/common surface and is
  callable through the production plugin registry or extension-bundle setup
  path.
- Plugin registration or built-in bundle setup consumes the gate before a
  plugin is accepted as an executable construction publisher.
- RVV overrides or wires the production gate so registration and the selected
  emission path still validate RVV manifest, typed role realization, selected
  role sequence, selected boundary conformance, and construction artifact
  metadata before publishing a supported plan.
- TensorExtLite, Toy, or Template proves the same gate works for a non-RVV
  construction-template path without adding common RVV/TensorExt/Toy branches.
- Focused C++ tests cover valid gate success and failure for invalid manifest,
  missing role, stale interface realization, unsupported artifact kind, stale
  artifact metadata, and production registry rejection.
- Focused RVV tests still pass for the existing i32m1 selected boundary,
  readiness, emission plan, and materialized EmitC target artifact path.
- Existing target artifact export tests still pass if target/export candidate
  surfaces are touched.
- Focused residue scans over touched common/plugin/test files show no
  descriptor-driven route authority, direct-C/source-export semantic path,
  Python compiler-core behavior, or extension-specific semantic branch added
  to common construction code.

## Out Of Scope

- New extension-family runtime semantics or new plugin coverage beyond the
  smallest reusable gate proof.
- New RVV SEW/LMUL/dtype/op families or broader RVV lowering.
- New high-level source lowering or frontend adapters.
- Direct C/source exporters, descriptor-driven computation, source skeleton
  generation, legacy compatibility wrappers, or historical route aliases.
- Fresh RVV runtime/correctness/performance claims unless emitted object bytes
  or runtime package behavior changes.
- Broad experiment or benchmark matrices.

## Definition Of Done

- PRD and task context describe this round truthfully.
- Code changes rewire a production/default plugin registration, bundle, or
  emission path to consume the common gate.
- Focused C++ tests pass for construction common gate, RVV plugin adoption, and
  the chosen non-RVV reusable proof.
- Focused target artifact tests and lit are run if the touched surface reaches
  target/export or textual pass behavior.
- `git diff --check` passes.
- `check-tianchenrv` is run if practical; otherwise the exact blocker is
  recorded.
- Trellis task is finished/archived when complete.
- One coherent commit is created when the task is complete.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/index.md`, and guides under `.trellis/spec/guides/`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-18-rvv-executable-construction-conformance-adoption/prd.md`.
- Main likely source surfaces:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`,
  RVV construction/plugin code, and one non-RVV construction-template plugin.
- Main likely tests:
  `test/Plugin/ConstructionProtocolCommonTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  one non-RVV plugin test, and target artifact export tests if exporter
  publication gates are touched.

## Completion Notes

- Added `ConstructionConformanceGateSpec` and
  `ConstructionArtifactMetadataConformanceSpec` to the common construction
  protocol API, with `verifyConstructionConformanceGate` validating manifest,
  typed-role realization, optional executable role steps, and construction
  artifact metadata through existing extension-agnostic conformance helpers.
- Added `ExtensionPlugin::verifyExecutableConstructionConformance` and wired
  `ExtensionPluginRegistry::registerPlugin` to call it before accepting a
  plugin. This makes plugin registration and built-in extension bundle setup a
  production consumer of the construction gate.
- RVV now overrides the production hook and routes
  `verifyRVVConstructionProtocolReady` through the common gate for its
  manifest, typed role realization, and construction artifact metadata before
  the existing i32m1 route can publish supported emission artifacts.
- TensorExtLite now overrides the same production hook and uses the same common
  gate for the non-RVV construction-template proof, including role-step and
  artifact metadata conformance.
- RVV selected role-sequence validation remains on the selected EmitC route
  construction path, where it already consumes common selected role-sequence
  APIs. RVV config/VL, ABI, intrinsic, arithmetic, and target packaging
  semantics remain RVV-owned.
- No object bytes, generated C/C++, runtime package, or hardware execution
  behavior were intentionally changed. Fresh `ssh rvv` evidence was not
  refreshed because this round changed plugin construction gating and
  fail-closed checks, not emitted RVV artifacts or runtime claims.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-executable-construction-protocol-registry-gate`
- `cmake --build build --target tianchenrv-construction-protocol-common-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `git diff --check`
- Common construction family-name scan:
  `rg -n "RVV|TensorExt|Toy|Template|Offload|Scalar|rvv|tensorext|toy|template|offload|scalar" include/TianChenRV/Plugin/ConstructionProtocol.h lib/Plugin/Construction/ConstructionProtocol.cpp`
  returned no matches.
- Changed plugin/test residue scan:
  `rg -n "descriptor-driven|descriptor|direct-C|direct C|source-export|source exporter|Python compiler-core|python compiler-core" <changed include/lib/test plugin files>`
  returned no matches.
- `cmake --build build --target check-tianchenrv -j2` passed 122/122 lit
  tests.
