# Template typed-role op construction path

## Goal

Make the Template construction protocol consume a minimal typed role/interface
realization for the selected semantic role graph, so generated construction
output is validated against concrete extension-family role surfaces rather
than only manifest strings.

This round extends the existing Template construction manifest and generated
role-graph EmitC route. It remains a construction-template proof for future
extensions, not a runtime, hardware, correctness, or performance claim.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD before this task is
  `5ce4513 feat(template): generate role graph emitc route`.
* The worktree was clean before creating this task.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
* The previous Template construction tasks added:
  * a C++ `TemplateConstructionManifest`;
  * Template proposal, legality, emission-plan, and target-artifact
    consumption of manifest-derived metadata;
  * a deterministic generated role-graph-to-EmitC source-like route with
    `generated_emitc_step[...]` and `generated_source`.
* The remaining bottleneck is that the generated route still primarily
  validates manifest-derived strings. This task must add a minimal typed
  role/interface realization consumed by planning/artifact tests.

## Module Goal

Within the Template plugin/target construction path, introduce one minimal
typed role/interface realization for the selected
`configure -> load -> compute -> store` role graph. The generated output route
must cross-check that typed realization against the construction manifest,
common-interface realization, EmitC role-to-call mapping, and evidence profile
before artifact export.

## Boundaries

* Scope is Template plugin/target/construction code and focused tests.
* Compiler behavior must stay in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Python may only run Trellis/tooling checks; it must not implement compiler
  model behavior.
* Use one minimal Template archetype and one coherent role slice.
* Prefer typed compute-role validation plus enough configure/load/store
  context to prove ordered role-graph agreement.
* If real TableGen/dialect op definitions are too large for this round,
  finish a C++ interface-realized role model consumed by planning/artifact
  tests and stage real TableGen op definitions next.
* Do not expand RVV, IME, scalar, dtype, LMUL, runtime, hardware, or
  performance paths.
* Do not add descriptor-driven computation, descriptor-to-C export, or
  computation semantics in `tcrv.exec`.
* Do not add Template/RVV/IME/offload semantic branches to core orchestration
  passes, `lib/Transforms`, or core `tcrv.exec`.

## Requirements

* Add a C++ typed role/interface realization model for the minimal Template
  selected role graph.
* Make selected Template planning or artifact input carry typed role identity
  or interface-realized role objects for the role graph.
* Validate typed roles against:
  * construction protocol version and archetype;
  * semantic role graph ordering;
  * Template family declaration and family operation names;
  * common TCRV interface realization;
  * role-specific common interfaces;
  * EmitC role-to-call mapping;
  * evidence profile.
* Fail closed before artifact export for missing, stale, reordered,
  duplicated, or mismatched typed role/interface realization.
* Keep generated output deterministic and derived from the role-to-EmitC
  mapping, including `generated_emitc_step[...]` and `generated_source`.
* Keep Template-specific logic in Template plugin/target/construction code.
* Preserve the existing generated route regression from
  `Template role-graph EmitC route realization`.
* Add focused C++ and lit/FileCheck coverage for positive and fail-closed
  typed role/interface realization cases.
* Preserve RVV plugin tests if shared plugin interfaces are touched.

## Acceptance Criteria

* [x] Template construction input carries a concrete typed role/interface
      realization for the selected role graph.
* [x] Generated output route cross-checks typed roles against the manifest,
      common-interface realization, role-specific interfaces, EmitC mapping,
      and evidence profile.
* [x] Missing typed role realization fails closed.
* [x] Reordered typed role realization fails closed.
* [x] Stale or mismatched typed role identity fails closed.
* [x] Stale or mismatched interface realization fails closed.
* [x] Generated output still includes deterministic `generated_emitc_step[...]`
      and `generated_source` text derived from role-to-EmitC mapping.
* [x] Common TCRV interface names are reused; Template semantic logic remains
      in Template plugin/target/construction code.
* [x] `lib/Transforms`, core `tcrv.exec`, and common orchestration passes do
      not gain Template semantic branches.
* [x] Focused C++ build/tests and lit/FileCheck checks pass.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis validation passes before finish/archive and after archive.
* [x] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Definition Of Done

* Focused Template/plugin/target build targets pass:
  `TianChenRVTemplatePlugin`, `TianChenRVTemplateTarget`,
  `tianchenrv-template-extension-plugin-test`, `tcrv-opt`, `tcrv-translate`,
  and any new construction-template test target.
* Focused C++ tests prove typed role/interface validation against archetype,
  role graph, family declaration, common interfaces, EmitC mapping, and
  evidence profile.
* lit/FileCheck proves generated output is derived from typed role/interface
  realization and fails closed on stale or mismatched typed role/interface
  data.
* Existing Template generated route still emits `generated_emitc_step[...]`
  and `generated_source`.
* RVV plugin regression passes if shared plugin interfaces are touched.
* No `ssh rvv` evidence is required unless emitted RVV artifacts change.

## Out Of Scope

* Documentation-only template work.
* Checklist-only task progress.
* Manifest-only metadata addition as the main result.
* Smoke-only coverage or helper-only cleanup as the main result.
* New RVV vmul, i64, LMUL, dtype, performance, or hardware expansion.
* Descriptor cleanup unless it becomes the single blocker for this Template
  construction path.
* Python compiler implementation.
* Direct descriptor-to-C exporter architecture.
* Moving computation semantics into `tcrv.exec`.
* Runtime correctness or performance claims for Template output.

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
* Previous PRDs read:
  * `.trellis/tasks/archive/2026-05/05-13-template-role-graph-emitc-route-realization/prd.md`
  * `.trellis/tasks/archive/2026-05/05-13-executable-extension-family-construction-template/prd.md`
* Workspace journal read:
  * `.trellis/workspace/codex/journal-5.md` recent Template construction
    entries.
* Source surfaces to inspect before implementation:
  * `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`
  * `lib/Plugin/Template/TemplateConstructionProtocol.cpp`
  * `lib/Plugin/Template/TemplateExtensionPlugin.cpp`
  * `lib/Target/Template/TemplateMetadataArtifact.cpp`
  * `test/Plugin/TemplateExtensionPluginTest.cpp`
  * `test/Target/TemplateMetadataArtifact/`

## Completion Notes

Implemented a C++ typed role/interface realization for the Template
construction path:

* Added `TemplateTypedRoleInterfaceRealization` and
  `TemplateTypedRoleGraphRealization`.
* Added `getTemplateTypedRoleGraphRealization`,
  `verifyTemplateTypedRoleGraphRealization`, and a generated-output builder
  overload that consumes the typed realization after manifest cross-checking.
* Extended `TemplateGeneratedOutputStep` with typed role identity,
  role-specific interface, and EmitC-lowerable interface fields.
* Template proposal materialization now carries
  `tcrv_template.typed_role_realization`; Template legality requires it before
  readiness or emission planning.
* Template emission planning serializes `template_typed_role_realization`
  selected-plan metadata.
* Template target artifact route registration and candidate validation require
  the typed role realization metadata before export.
* Generated artifact output now prints `typed_role_realization`,
  `typed_role[...]`, and typed role fields inside `generated_emitc_step[...]`.
* Generated source remains deterministic and derived from the checked
  role-to-EmitC mapping.
* Added C++ fail-closed coverage for missing typed role, reordered typed
  roles, stale typed role operation identity, mismatched role-specific
  interface, and mismatched typed EmitC call.
* Added lit/FileCheck coverage for typed role output and stale selected-plan
  typed role metadata rejection.
* Updated the plugin-protocol Template construction scenario to record the
  typed-role realization API, validation matrix, and test expectations.

Real TableGen Template role operation definitions remain staged as the next
larger step; this round intentionally finishes the C++ interface-realized role
model allowed by the task boundary.

Checks run:

```bash
cmake --build build --target TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2
./build/bin/tianchenrv-template-extension-plugin-test
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline
./build/bin/tcrv-opt test/Target/TemplateMetadataArtifact/template-metadata-artifact-route.mlir --tcrv-execution-planning-pipeline | ./build/bin/tcrv-translate --tcrv-export-target-artifact
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='template-extension-plugin|TemplateMetadataArtifact'
./build/bin/tianchenrv-target-artifact-export-test
cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2
./build/bin/tianchenrv-rvv-extension-plugin-test
rg -n "template-plugin|tcrv_template|Template" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec || true
git diff --check
python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-05-13-template-typed-role-op-construction-path
python3 ./.trellis/scripts/task.py archive 05-13-05-13-template-typed-role-op-construction-path --no-commit
python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-13-05-13-template-typed-role-op-construction-path
git diff --cached --check
```

The first target-artifact C++ test invocation used `./bin/...` from
`build/test` and failed with `No such file or directory`; the test was rerun
successfully from the repository root as
`./build/bin/tianchenrv-target-artifact-export-test`.

No `ssh rvv` evidence was run because emitted RVV runtime artifacts did not
change and this task makes no RVV runtime, correctness, or performance claim.
