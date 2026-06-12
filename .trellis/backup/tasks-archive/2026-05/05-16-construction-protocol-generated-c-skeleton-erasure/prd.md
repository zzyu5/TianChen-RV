# Construction-protocol generated C skeleton erasure

## Goal

Erase construction-protocol authority that derives generated C/source skeletons
from plugin construction metadata. The common construction protocol and the
Template/Toy/TensorExtLite family wrappers may continue to describe extension
families, semantic role graphs, typed role/interface realization, and evidence
labels, but they must not build generated output routes, source lines, fake
intrinsic headers, or direct `__tcrv_*` call skeletons from manifest data.

## What I already know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- The initial worktree was clean and HEAD was
  `179c655 chore(rvv): erase deleted metadata-route diagnostics`.
- No `.trellis/.current-task` existed before this task was created.
- This is a Wrong Logic Deletion Campaign round: delete/refactor obsolete
  direct-C/source-authority residue first, without rebuilding a replacement
  EmitC/source/export architecture in the same round.
- Current specs say the durable route is explicit extension-family ops to a
  materialized EmitC module and then target/native compilation; descriptor and
  direct-C source exporters are deletion targets or fail-closed debt.
- The active construction surface still contains common generated-output types
  and functions, family-level generated-output wrappers, manifest fields naming
  direct headers/calls, and tests that assert generated source skeleton strings.

## Requirements

- Remove common construction-protocol generated-output structs and functions:
  `GeneratedOutputStep`, `GeneratedOutputRoute`,
  `buildGeneratedOutputRoute`, and `emitGeneratedOutputRoute`.
- Remove Template/Toy/TensorExtLite family aliases and public wrappers for
  generated-output routes.
- Remove manifest fields that only exist to name generated direct-C/source
  skeleton behavior, including fake required headers and role-to-call maps.
- Remove typed-role fields and role-op validation inputs that only exist to
  validate direct `__tcrv_*` call names for generated source skeletons.
- Preserve non-semantic construction checks: protocol version, archetype,
  semantic role graph, family declaration, ordered role/interface validation,
  typed role/interface realization, and evidence labels.
- Rewrite or delete tests that protect generated-output route construction,
  generated function names, `sourceLine`, `roleToCallMap`, fake intrinsic
  headers, direct `__tcrv_*` call strings, or source skeleton output as route
  behavior.
- Update durable specs that still require construction-protocol generated C
  skeleton authority, without adding a rebuild route.

## Acceptance Criteria

- [ ] No active common construction code can build or emit a generated C/source
  route from construction metadata.
- [ ] No active family wrapper exposes `buildTemplateGeneratedOutputRoute`,
  `buildToyGeneratedOutputRoute`, or `buildTensorExtLiteGeneratedOutputRoute`.
- [ ] No active tests assert generated output function names, generated output
  source lines, source skeleton text, fake intrinsic headers, `roleToCallMap`,
  or direct `__tcrv_*` call strings as route behavior.
- [ ] Remaining construction protocol checks are non-semantic plugin-family and
  typed-role/interface declaration checks only.
- [ ] Focused ref-scan is run for the requested stale tokens, excluding
  `.trellis/tasks/archive`, `.trellis/workspace`, `artifacts/tmp`, `build`, and
  `.git`, with any remaining hits classified as valid non-route residue or
  removed.
- [ ] Targeted build passes for existing affected targets: `tcrv-opt`,
  `tcrv-translate`, `tianchenrv-template-extension-plugin-test`,
  `tianchenrv-toy-extension-plugin-test`, and
  `tianchenrv-tensorext-lite-extension-plugin-test`.
- [ ] `ninja -C build check-tianchenrv` is attempted; deletion-caused
  missing-architecture gaps are reported without restoring old generated
  source authority.
- [ ] `git diff --check` and Trellis validation pass.
- [ ] The task is finished/archived and one coherent commit records the round,
  unless a real blocker prevents completion.

## Out of Scope

- No Common EmitC rebuild.
- No new executable plugin template.
- No new EmitC route mapping replacement.
- No target artifact exporter.
- No runtime ABI implementation.
- No compatibility or deprecated generated-output API.
- No wrapper around the old generated route.
- No broad deletion of valid non-semantic extension-family declarations merely
  because they are not executable yet.

## Technical Notes

- Required specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Initial code/test surfaces:
  - `include/TianChenRV/Plugin/ConstructionProtocol.h`
  - `lib/Plugin/Construction/ConstructionProtocol.cpp`
  - `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`
  - `lib/Plugin/Template/TemplateConstructionProtocol.cpp`
  - `include/TianChenRV/Plugin/Toy/ToyConstructionProtocol.h`
  - `lib/Plugin/Toy/ToyConstructionProtocol.cpp`
  - `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`
  - `lib/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.cpp`
  - `lib/Plugin/Template/TemplateExtensionPlugin.cpp`
  - `lib/Plugin/Toy/ToyExtensionPlugin.cpp`
  - `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`
  - `test/Plugin/ConstructionProtocolCommonTest.cpp`
  - `test/Plugin/TemplateExtensionPluginTest.cpp`
  - `test/Plugin/ToyExtensionPluginTest.cpp`
  - `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`
  - Template/Toy/TensorExtLite dialect skeleton tests if direct call-name
    attributes remain as source skeleton authority.
