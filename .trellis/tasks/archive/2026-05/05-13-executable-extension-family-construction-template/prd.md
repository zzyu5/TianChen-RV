# Executable extension-family construction template

## Goal

Move the Extension-Family Plugin Construction Protocol from specification and
prompt material into a code-consumable construction template for one minimal
extension-family archetype.

The slice must prove the full protocol sequence through compiler-owned C++ APIs
and focused tests:

```text
extension archetype
  -> semantic role graph
  -> extension family declaration
  -> common interface realization
  -> EmitC route mapping
  -> evidence profile
```

## Current State

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD is `73a4f95 chore(supervisor): define extension family construction protocol`.
* The previous worker left an untracked clarification task because the worker
  prompt had no concrete Direction Brief.
* That stale clarification task has been removed from the active task set and
  replaced by this concrete Trellis task; it is not module progress.
* Long-term specs already define the construction protocol and manifest role,
  but existing compiler evidence is still primarily a metadata-only Template
  plugin route.

## Module Goal

Repair the existing `Template` extension path into a construction-template
entry point that code and tests consume. The Template path remains a minimal
future-extension archetype; it must not claim new hardware support, RVV runtime
success, correctness, or performance.

## Boundaries

* Compiler functionality must stay in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Python may only run Trellis/tooling checks; it must not implement compiler
  model behavior.
* Use one minimal extension-family archetype only.
* RVV may be used as a reference for interface/route shape, but this task must
  not expand RVV families, dtype coverage, LMUL policy, runtime claims, or
  performance evidence.
* The Template path may produce compiler handoff/construction artifacts, but it
  must state clearly that these are not executable runtime artifacts.
* Core orchestration must remain extension-neutral. If core files change, they
  must only expose or validate generic interfaces, not add Template/RVV/IME
  semantic branches.

## Requirements

* Add or repair a code-consumable construction manifest for the minimal
  Template extension archetype.
* Represent and validate:
  * archetype;
  * semantic role graph;
  * extension family declaration;
  * common-interface realization mapping;
  * EmitC route mapping;
  * evidence profile.
* Make the Template plugin and/or target artifact path consume that
  construction manifest rather than duplicating disconnected strings.
* Add focused C++ and lit/FileCheck coverage proving the manifest is consumed
  by code or generated output.
* Preserve the boundary between common TCRV surfaces and extension-specific
  hooks/mappings.
* Prove that core orchestration does not gain extension-specific semantic
  branches.
* Keep stale clarification-task handling truthful in final notes.

## Acceptance Criteria

* [x] Stale `.trellis/tasks/05-13-clarify-worker-direction-brief` is resolved
      and not committed as module progress.
* [x] Template construction entry point exposes archetype, role graph, family
      declaration, interface realization, EmitC route mapping, and evidence
      profile through C++.
* [x] Plugin proposal / legality / emission-plan path consumes construction
      manifest values or validates against them.
* [x] Target/generated artifact output includes construction-protocol fields
      and is validated by FileCheck.
* [x] C++ tests assert the manifest shape and its agreement with the plugin
      registry/capability/emission surfaces.
* [x] Existing Template/RVV plugin route activation remains passing if touched.
* [x] No core orchestration pass gains a concrete extension semantic branch.
* [x] `git diff --check` and focused build/test checks pass.
* [x] Trellis validation passes before finish/archive.
* [x] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Non-goals

* No documentation-only template.
* No checklist-only PRD.
* No metadata-only manifest as the main result.
* No helper-only cleanup as the main result.
* No new RVV vmul, i64, LMUL, dtype, performance, or hardware coverage.
* No descriptor cleanup unless it is the single blocker for this construction
  template.
* No Python compiler implementation.
* No direct descriptor-to-C exporter.
* No computation semantics in `tcrv.exec`.
* No extension-specific semantic branch in core passes.

## Focused Evidence

Expected local checks:

```bash
cmake --build build --target TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2
./build/bin/tianchenrv-template-extension-plugin-test
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline | ./build/bin/tcrv-translate --tcrv-export-target-artifact
llvm-lit -sv test/Plugin/template-extension-plugin.test test/Target/TemplateMetadataArtifact
git diff --check
git diff --cached --check
python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-executable-extension-family-construction-template
```

`ssh rvv` is not expected because this task must not change emitted RVV runtime
artifacts or make RVV runtime/correctness/performance claims.

## Completion Notes

Implemented a C++ construction manifest for the Template extension path:

* New `TemplateConstructionProtocol` API exposes and validates:
  * protocol version;
  * minimal future-extension archetype;
  * `configure -> load -> compute -> store` semantic role graph;
  * Template family declaration;
  * common-interface realization mapping;
  * plugin-owned EmitC route mapping;
  * focused evidence profile.
* Template plugin proposal materialization now attaches construction protocol,
  archetype, role graph, interface realization, EmitC route, and evidence
  profile metadata to the materialized variant.
* Template legality validates those construction fields against the C++
  manifest before readiness/planning.
* Template emission planning records construction selected-plan metadata.
* Template target artifact validation and output consume the same manifest and
  print archetype, role graph, family, interface, EmitC route, and evidence
  fields in generated artifact output.
* Plugin-protocol spec now records the executable construction-template
  manifest contract with signatures, validation matrix, test requirements, and
  wrong/correct cases.
* Stale clarification task was removed after this real task was created; it is
  not committed as module progress.

Checks run:

```bash
python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-executable-extension-family-construction-template
cmake --build build --target TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2
./build/bin/tianchenrv-template-extension-plugin-test
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline | ./build/bin/tcrv-translate --tcrv-export-target-artifact
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='template-extension-plugin|TemplateMetadataArtifact'
cmake --build build --target tianchenrv-target-artifact-export-test -j2
./build/bin/tianchenrv-target-artifact-export-test
cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2
./build/bin/tianchenrv-rvv-extension-plugin-test
git diff --check
git diff --cached --check
```

The first direct `llvm-lit` invocation from the repository root failed because
`llvm-lit` was not on `PATH`; the successful lit run used the configured LLVM
20 lit entry from `build/test`.

No `ssh rvv` evidence was run because this task did not change RVV emitted
runtime artifacts and makes no RVV runtime/correctness/performance claim.
