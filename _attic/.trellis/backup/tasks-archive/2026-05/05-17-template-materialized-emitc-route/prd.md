# Template Materialized EmitC Construction-Template Route

## Goal

Make Template a minimal executable extension-family construction template
rather than a metadata-only checklist. A hand-written Template-capability
TianChen-RV MLIR input must flow through generic variant selection into a
Template-owned selected role boundary, then through a Template-owned
`TCRVEmitCLowerableRoute`, a materialized MLIR EmitC module, the common MLIR
EmitC C/C++ emitter, and focused generated C++ compile evidence.

This task is bounded to Template plugin/dialect/EmitC-route/target-support
surfaces and directly necessary registration hooks. It must not add high-level
frontend lowering, real hardware execution, descriptor adapters, direct C
semantic exporters, source-export authority, compatibility wrappers, or core
family-specific Template semantic branches.

## Requirements

- Preserve TianChen-RV as a unified RISC-V MLIR: Template remains one extension
  family plugin, not a separate backend.
- Keep compiler behavior in C++/MLIR/TableGen/CMake/lit/FileCheck. Python may
  not become a compiler-core path for this route.
- Consume the Template construction manifest as the source of archetype,
  semantic role graph, family declaration, common interface realization, EmitC
  route mapping, and evidence profile.
- Materialize a selected Template role boundary from generic selected-path
  handling. The selected role must be plugin-owned Template IR and must realize
  `TCRVEmitCLowerableOpInterface`.
- Build a Template-owned EmitC lowerable route from that selected role, with
  bounded source-op/source-role/source-interface provenance and no descriptor,
  source-export, direct-C, benchmark, correctness, or performance authority.
- Produce a supported emission plan only after the Template route can be built
  from the selected role boundary. Missing/stale role-interface realization,
  stale selected metadata, or stale route metadata must fail closed.
- Provide a Template target-support route sufficient for generated C++ evidence:
  common materialized EmitC module -> MLIR EmitC C/C++ emitter -> local
  compile/FileCheck proof. Object/header/bundle packaging is not required for
  this round unless it falls out naturally from the minimal header route.
- Negative coverage must reject metadata-only artifact authority,
  descriptor/direct-C/source-export residue, stale selected metadata, missing
  Template role interface realization, and core extension-specific Template
  semantic branches.

## Acceptance Criteria

- [x] Template construction manifest names a current materialized route rather
      than the old `no-active-route` metadata-only diagnostic route, and C++
      tests prove the manifest verifies.
- [x] Template selected-path materialization creates exactly one selected
      Template role boundary for the selected variant, and C++ tests prove
      stale/missing selected metadata fails closed.
- [x] Template `buildVariantEmitCLowerableRoute` is implemented through a
      plugin-owned route provider that consumes the selected
      `tcrv_template.compute_skeleton` role op and verifies the generated
      `TCRVEmitCLowerableOpInterface` provenance.
- [x] Template emission readiness and emission planning become supported only
      when the selected role route can be built; the supported plan carries
      route id, runtime ABI identity, lowering boundary, construction protocol,
      semantic role graph, typed role realization, and source-op/interface
      artifact metadata.
- [x] A lit/FileCheck route from hand-written Template-capability TianChen-RV
      MLIR through `tcrv-opt` produces selected Template role IR, a supported
      emission plan, and materialized EmitC route evidence.
- [x] A lit/FileCheck route through `tcrv-translate` emits generated C++ via
      the common MLIR EmitC C/C++ emitter, and local compile/syntax evidence is
      included when the local C++ compiler is available.
- [x] Negative tests fail closed for missing selected Template role IR, stale
      route/variant metadata, descriptor/direct-C/source-export residue, and
      metadata-only artifact authority.
- [x] Focused build targets, Template plugin tests, focused lit filters,
      bounded residue scans, `git diff --check`, and `check-tianchenrv` if
      practical pass.

## Definition Of Done

- Task status and journal notes are truthful.
- No runtime correctness, hardware correctness, or performance claim is made.
- One coherent commit records the PRD, implementation, tests, task finish, and
  archive if the task is complete.

## Out Of Scope

- Another TensorExtLite evidence-only round.
- RVV expansion, IME/TensorExt real semantics, performance matrices, broad
  smoke dashboards, report/status plumbing, compatibility layers, legacy modes,
  descriptor adapters, source-export routes, direct C semantic exporters,
  Python compiler-core behavior, or core `if Template` / `if TensorExtLite`
  semantic branches.
- Claiming Template runtime correctness, hardware correctness, or performance.
- Building a full object/header/bundle route if the coherent generated-C++
  submodule is the achievable bounded slice.

## Completion Summary

- Replaced Template's metadata-only `no-active-route` authority with a
  construction-protocol-backed `compute_skeleton` materialized EmitC route,
  generated C++ translate route, and declaration-only runtime-callable header
  artifact route.
- Rewired Template selected-path materialization to create
  `tcrv_template.compute_skeleton` as the selected role boundary, and made
  emission readiness/planning supported only when the Template route provider
  can consume that role op through `TCRVEmitCLowerableOpInterface`.
- Added Template target support registration for the plugin-owned header
  exporter and `tcrv-template-emitc-to-cpp` translate route without adding a
  core `if Template` semantic branch.
- Added C++ and lit coverage for manifest consumption, selected role
  materialization, route materialization, generated C++ emission and syntax
  compile, header artifact evidence, stale route/missing boundary failures,
  descriptor/direct-C/source-export rejection, and metadata-only artifact
  rejection.
- No runtime correctness, hardware correctness, or performance claim is made.

### Status

[OK] Completed

### Next Steps

- None - task complete

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/index.md`.
- Primary Template surfaces:
  `include/TianChenRV/Dialect/Template/IR/TemplateOps.td`,
  `lib/Dialect/Template/IR/TemplateDialect.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`,
  `lib/Plugin/Template/TemplateConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/Template/TemplateExtensionPlugin.h`,
  `lib/Plugin/Template/TemplateExtensionPlugin.cpp`, and
  `test/Plugin/TemplateExtensionPluginTest.cpp`.
- Reference shapes only:
  Toy selected compute-skeleton -> EmitC route -> header route, and
  TensorExtLite selected role sequence -> materialized EmitC -> generated C++
  / object/header route. Do not copy-paste TensorExtLite proof packaging as the
  main deliverable.
