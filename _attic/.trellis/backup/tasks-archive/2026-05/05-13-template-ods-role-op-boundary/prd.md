# Template ODS role-op boundary materialization

## Goal

Materialize the Template extension-family role surface as the smallest real
TableGen/ODS-backed MLIR role-op boundary, then wire that boundary into the
existing Template construction protocol, planning metadata validation, EmitC
role mapping, and Template artifact export path.

This task continues after the C++ typed-role realization layer from
`a528403`. The current bottleneck is that Template construction proves typed
role identity from C++ metadata strings, but does not yet prove that a future
extension family can declare concrete MLIR role ops that implement the common
generated interface boundary and have the route consume those ops.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD before this task is
  `a528403 feat(template): validate typed role construction path`.
* The worktree was clean before task creation.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
* The previous completed Template tasks added:
  * an executable Template construction manifest;
  * a role-graph-to-EmitC generated output route;
  * a C++ typed role/interface realization layer;
  * Template proposal, legality, selected-plan metadata, target preflight, and
    generated artifact validation against manifest, role graph, common
    interfaces, EmitC mapping, and evidence profile.
* The remaining omitted stage is a real Template TableGen/ODS role-op
  boundary. Existing `tcrv_template.lowering_boundary` is selected-boundary
  metadata; it is not the role-op surface requested by this task.

## Module Goal

Add one coherent minimal Template role-op slice through TableGen/ODS:

```text
Template construction manifest
  -> Template ODS compute role op
  -> generated TCRVEmitCLowerableOpInterface provenance
  -> typed role realization cross-check
  -> selected-plan/artifact candidate validation
  -> role-to-EmitC generated output
```

The preferred MVP is one `tcrv_template.compute_skeleton` op implementing the
generated `TCRVEmitCLowerableOpInterface`, plus enough construction validation
to preserve the existing configure/load/compute/store role order checks and to
fail closed when the IR-backed compute role is missing, stale, wrong-role, or
wrong-interface.

## Boundaries

* Scope is Template extension-family only.
* Compiler behavior stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Python is allowed only for Trellis/tooling checks.
* Add the role-op boundary in Template dialect/plugin/target/construction
  files, not in core `tcrv.exec`.
* Do not expand RVV, IME, scalar, dtype, LMUL, runtime, hardware, correctness,
  or performance behavior.
* Do not add descriptor-driven computation or direct descriptor-to-C export.
* Do not add Template/RVV/IME/offload semantic branches to core transforms,
  core `tcrv.exec`, or generic orchestration.
* If configure/load/store ODS roles are too large for this round, keep them as
  explicit staged follow-up and finish the compute role op with tests proving
  the remaining roles are still manifest-ordered and not silently replaced by
  metadata-only fields.

## Requirements

* Declare a real Template role op or minimal role-op family through ODS.
* The role op must implement the appropriate generated common interface for
  the current slice: at minimum `TCRVEmitCLowerableOpInterface` for
  interface-backed EmitC provenance.
* The Template construction protocol must be able to build or validate a typed
  role realization from the ODS op/interface identity, not only from C++
  metadata strings.
* Template planning or artifact validation must consume the real op/interface
  boundary before generated artifact export.
* Construction validation must cross-check the op/interface identity against:
  * construction protocol version and archetype;
  * semantic role graph and compute role order;
  * Template family declaration and operation name;
  * existing typed role realization summary;
  * generated `TCRVEmitCLowerableOpInterface` source op and source role;
  * EmitC role-to-call mapping;
  * evidence profile.
* Stale, missing, duplicate, wrong-op, wrong-role, or missing-interface
  role-op data must fail closed before target artifact export.
* Generated output must still derive from the role-to-EmitC route and keep
  deterministic `generated_emitc_step[...]` and `generated_source` output.
* Existing Template generated route regressions must remain covered.
* Template-specific semantics must remain in Template dialect/plugin/target/
  construction files.
* Shared plugin/interface changes, if any, must preserve RVV plugin tests.

## Acceptance Criteria

* [x] `tcrv_template.compute_skeleton` or an equivalent minimal Template role
      op is declared through TableGen/ODS.
* [x] The op implements `TCRVEmitCLowerableOpInterface` and exposes bounded
      source-op/source-role provenance through the generated interface.
* [x] Template construction code validates the real op/interface identity
      against the manifest, typed role graph, EmitC mapping, and evidence
      profile.
* [x] Template artifact export consumes the validated op/interface boundary
      before printing generated output.
* [x] Valid path proves generated artifact output is still derived from the
      checked role-to-EmitC mapping.
* [x] Missing role op fails closed before export.
* [x] Wrong operation name fails closed before export.
* [x] Wrong generated source role fails closed before export.
* [x] Missing or unsupported generated interface fails closed before export.
* [x] Stale selected-plan typed-role or EmitC mapping metadata still fails
      closed.
* [x] lit/FileCheck covers parsing/verifying the Template role op and at least
      one export-path positive and negative case.
* [x] Focused C++ tests cover op/interface identity against archetype, role
      graph, family declaration, typed role realization, EmitC mapping, and
      evidence profile.
* [x] No core-pass Template semantic branch is added.
* [x] Focused build/tests pass.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis validation passes before finish/archive and after archive.
* [x] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Definition Of Done

* Focused targets build, including generated Template dialect headers and:
  `TianChenRVTemplateDialect`, `TianChenRVTemplatePlugin`,
  `TianChenRVTemplateTarget`, `tianchenrv-template-extension-plugin-test`,
  `tcrv-opt`, and `tcrv-translate`.
* Focused C++ Template/plugin/target tests pass.
* Focused lit/FileCheck tests for Template dialect/op and target artifact
  export pass.
* Existing Template generated route still emits `generated_emitc_step[...]`
  and `generated_source`.
* RVV plugin tests pass if shared plugin/interface code is touched.
* No `ssh rvv` evidence is required unless RVV emitted artifacts change.

## Completion Notes

Implemented the first real Template ODS role-op boundary:

* Added `tcrv_template.compute_skeleton` in TableGen/ODS.
* Wired it to the generated `TCRVEmitCLowerableOpInterface` with bounded
  source-op/source-role provenance.
* Added Template dialect verifier checks for selected variant, required
  capability mirror, typed role id, role order, source role, role-specific
  interface, EmitC call, and forbidden unknown/generic-compute attributes.
* Added `verifyTemplateComputeRoleOpInterface` to cross-check the real MLIR
  op/interface identity against the Template construction manifest, typed role
  graph, EmitC route mapping, and evidence profile.
* Changed Template selected-boundary materialization to emit both
  `tcrv_template.lowering_boundary` and `tcrv_template.compute_skeleton`.
* Changed Template artifact candidate validation to require exactly one
  matching `tcrv_template.compute_skeleton` before generated output export.
* Added artifact output fields:
  `validated_role_op`, `validated_role_op_interface`,
  `validated_role_op_source`, and `validated_role_op_source_role`.
* Preserved deterministic `generated_emitc_step[...]` and `generated_source`
  output from the existing role-to-EmitC mapping.
* Added C++ coverage for the generated interface identity, stale source-role
  mutation, stale typed-role mutation, and missing-interface/wrong-op
  validation.
* Added lit/FileCheck coverage for Template ODS parse/verify, positive
  pipeline/export output, and missing compute role-op export failure.
* Updated the plugin-protocol construction-template code-spec with the ODS
  role-op/interface validation contract.
* Verified no Template semantic branches were added to `lib/Transforms` or
  core `tcrv.exec`.

Checks run:

```bash
cmake --build build --target TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2
./build/bin/tianchenrv-template-extension-plugin-test
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Template|template'
cmake --build build --target tianchenrv-target-artifact-export-test -j2
./build/bin/tianchenrv-target-artifact-export-test
cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2
./build/bin/tianchenrv-rvv-extension-plugin-test
git diff --check
python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-template-ods-role-op-boundary
python3 ./.trellis/scripts/task.py archive 05-13-template-ods-role-op-boundary --no-commit
python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-13-template-ods-role-op-boundary
rg -n "tcrv_template|template-plugin|Template" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec
```

The local environment does not provide `clang-format` or
`/usr/lib/llvm-20/bin/clang-format`, so no automatic format command was run.
The edited C++ follows the nearby style and `git diff --check` passes.

No `ssh rvv` evidence was run because this task changes Template construction
and artifact validation only; it does not change emitted RVV runtime artifacts
or make RVV runtime, correctness, or performance claims.

## Out Of Scope

* Documentation-only template work.
* Checklist-only or manifest-only metadata addition as the main result.
* Helper-only cleanup without production/default path consumption.
* New RVV vmul, i64, LMUL, dtype, runtime, performance, or hardware expansion.
* Descriptor cleanup unless it becomes the single blocker for this Template op
  boundary.
* Python compiler implementation.
* Direct descriptor-to-C exporter architecture.
* Moving computation semantics into `tcrv.exec`.
* Template runtime correctness, hardware execution, or performance claims.

## Technical Notes

* Relevant specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/plugin-protocol/index.md`
  * `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  * `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/core-dialect/index.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Prior Template PRDs read:
  * `.trellis/tasks/archive/2026-05/05-13-template-role-graph-emitc-route-realization/prd.md`
  * `.trellis/tasks/archive/2026-05/05-13-05-13-template-typed-role-op-construction-path/prd.md`
* Initial code surfaces inspected:
  * `include/TianChenRV/Dialect/Template/IR/TemplateOps.td`
  * `lib/Dialect/Template/IR/TemplateDialect.cpp`
  * `include/TianChenRV/Conversion/EmitC/TCRVEmitCLowerableOpInterface.td`
  * `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`
  * `lib/Plugin/Template/TemplateConstructionProtocol.cpp`
  * `lib/Plugin/Template/TemplateExtensionPlugin.cpp`
  * `lib/Target/Template/TemplateMetadataArtifact.cpp`
  * `test/Plugin/TemplateExtensionPluginTest.cpp`
  * `test/Target/TemplateMetadataArtifact/`
