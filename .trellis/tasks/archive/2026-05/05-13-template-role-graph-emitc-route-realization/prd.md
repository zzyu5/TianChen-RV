# Template role-graph EmitC route realization

## Goal

Make the Template extension-family construction path consume the executable
construction manifest as the entry point for a concrete generated-output route.
The route must derive a deterministic source-like skeleton from the manifest's
semantic role graph, common-interface realization, and EmitC role-to-call
mapping, instead of only echoing construction metadata fields.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD is `1be2ea5 feat(template): add executable construction manifest`.
* There was no active Trellis task; this task was created from the Hermes
  Direction Brief.
* The previous archived task added `TemplateConstructionProtocol`, attached
  manifest-derived metadata to Template proposal/legalization/emission-plan,
  and made `TemplateMetadataArtifact` validate and print construction fields.
* The remaining bottleneck is the generated artifact surface: it still centers
  on manifest metadata printing rather than a role-graph-to-EmitC generated
  output skeleton.

## Module Goal

Within the Template plugin/target boundary, realize one minimal
`configure -> load -> compute -> store` construction route that emits a
deterministic source-like function section derived from the construction
manifest. This route remains a compiler construction artifact only and makes no
runtime, correctness, hardware, or performance claim.

## Boundaries

* Compiler behavior must stay in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Python may only run Trellis/tooling checks; it must not implement compiler
  behavior.
* Scope is Template plugin/target only plus focused tests.
* Use one minimal Template extension archetype.
* Do not expand RVV, IME, scalar, dtype, LMUL, runtime, or performance paths.
* Do not add descriptor-driven computation or direct descriptor-to-C exporter
  architecture.
* Do not add Template/RVV/IME/offload semantic branches to core orchestration
  passes.

## Requirements

* Add a C++ construction-output representation derived from
  `TemplateConstructionManifest`.
* Validate that semantic roles are ordered, have common-interface realization,
  and have matching EmitC calls before generated output is accepted.
* Make Template target artifact output include a deterministic source-like
  generated section derived from role-to-EmitC mapping.
* Keep role graph, common-interface realization, EmitC route mapping, and
  evidence profile validation fail-closed for stale/missing/mismatched data.
* Add focused C++ coverage for the generated-output route and negative
  manifest validation cases.
* Add lit/FileCheck coverage proving generated output is derived from the
  role-to-EmitC mapping, including compute-role output.
* Preserve existing Template metadata behavior only as construction context;
  the main artifact evidence must include generated role-to-call output.

## Acceptance Criteria

* [x] Template construction manifest is the entry point for generated output.
* [x] Role graph entries map to common-interface realizations and EmitC call
      skeleton lines in C++.
* [x] Generated artifact contains a deterministic source-like function or
      section derived from role-to-EmitC mapping, not only printed metadata.
* [x] Missing, reordered, stale, or mismatched role/interface/EmitC mapping
      fails closed in focused tests.
* [x] Evidence profile validates generated output.
* [x] Tests prove common TCRV surfaces are reused and Template-specific logic
      stays in Template plugin/target files.
* [x] `lib/Transforms`, core `tcrv.exec`, and common orchestration passes do
      not gain Template semantic branches.
* [x] Focused C++ build/tests and lit/FileCheck pass.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis validation passes before finish/archive.
* [ ] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Definition Of Done

* Focused Template/plugin/target build targets pass.
* Focused Template C++ test passes.
* Focused Template lit/FileCheck tests pass.
* The final artifact output demonstrates role-graph-to-EmitC generated output.
* No RVV hardware evidence is claimed or required because RVV artifacts do not
  change.

## Out Of Scope

* Documentation-only template work.
* Metadata-only artifact additions as the main result.
* Helper-only cleanup without production/default Template route consumption.
* RVV vmul, i64, LMUL, dtype, performance, or hardware expansion.
* Descriptor cleanup unless it becomes the single blocker for this route.
* Python compiler implementation.
* Direct descriptor-to-C exporter architecture.
* Moving computation semantics into `tcrv.exec`.

## Technical Notes

* Relevant specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/plugin-protocol/index.md`
  * `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  * `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-13-executable-extension-family-construction-template/prd.md`.
* Initial code surfaces inspected:
  * `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`
  * `lib/Plugin/Template/TemplateConstructionProtocol.cpp`
  * `lib/Plugin/Template/TemplateExtensionPlugin.cpp`
  * `lib/Target/Template/TemplateMetadataArtifact.cpp`
  * `test/Plugin/TemplateExtensionPluginTest.cpp`
  * `test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir`

## Completion Notes

Implemented a manifest-driven Template generated-output route:

* Added `TemplateGeneratedOutputRoute` and `TemplateGeneratedOutputStep`.
* Added `buildTemplateGeneratedOutputRoute`, which validates the construction
  manifest and derives ordered source-like call skeleton steps from the
  semantic role graph and EmitC role-to-call mapping.
* Tightened Template manifest validation for exact role order, role-specific
  common interfaces, common-interface summary agreement, ordered EmitC mapping,
  required header, duplicate/missing role mapping, and invalid call names.
* Updated `TemplateMetadataArtifact` validation and output so the target route
  emits `generated_emitc_step[...]` records and a deterministic
  `generated_source` function:
  `tcrv_template_generated_template_zero_core_first_slice`.
* Added C++ coverage for positive generated-output route construction and
  fail-closed reordered role graph, missing EmitC call, reordered EmitC call,
  mismatched common-interface realization, and missing `generated_output`
  evidence profile.
* Added lit/FileCheck coverage for the generated role-to-EmitC source skeleton
  and a stale selected-plan EmitC mapping negative route.
* Updated the existing plugin-protocol Template scenario with the new
  generated-output API and validation/test contract.

Checks run:

```bash
cmake --build build --target TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2
./build/bin/tianchenrv-template-extension-plugin-test
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline | ./build/bin/tcrv-translate --tcrv-export-target-artifact
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='template-extension-plugin|TemplateMetadataArtifact'
git diff --check
git diff --cached --check
python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-template-role-graph-emitc-route-realization
```

The first direct `lit.py -sv test/...` invocation failed because the source
`test/lit.cfg.py` lacks CMake-generated `tianchenrv_obj_root`; the successful
lit run used `build/test` as the working directory.

No `ssh rvv` evidence was run because this task did not change RVV emitted
runtime artifacts and makes no RVV runtime/correctness/performance claim.
