# Generic metadata-only plugin route contract erasure

## Goal

Erase metadata-only plugin emission/artifact plans as active compiler routes.
The selected compiler path must fail closed unless the selected variant has a
real materialized extension-family / EmitC route. This is a deletion/refactor
round only.

## Context

- The previous RVV-specific round ended at `e5b00bf` and removed the active RVV
  metadata-only first-slice route.
- The same old route shape still exists generically in common plugin emission
  planning and in Template, Toy, TensorExtLite, and Offload plugin/target
  surfaces.
- Current long-term specs require compiler implementation in C++ / MLIR /
  LLVM / TableGen / CMake / lit / FileCheck.
- Direct descriptor-to-C export and metadata-only computation authority are
  invalid architecture. Future executable work must go through extension
  family ops, materialized EmitC, C/C++ emission, and real target evidence.

## Module Goal

Remove the common contract that treats metadata-only plugin emission plans or
metadata artifacts as supported route ownership, and update directly dependent
Template, Toy, TensorExtLite, Offload, execution-planning, emission-readiness,
execution-plan-coherence, target artifact, and plugin protocol tests so they no
longer protect metadata-only selected emission or artifact handoff as valid
compiler evidence.

## Boundaries

- Common plugin orchestration must no longer accept metadata-only emission
  plans as supported compiler routes.
- Metadata-only artifacts must be deleted or rewritten as fail-closed/no-route
  diagnostics.
- Template, Toy, TensorExtLite, and Offload surfaces that depended on
  metadata-only route/export/test behavior are in scope.
- Existing tests that encode metadata-only selected emission, runtime ABI,
  runtime glue, artifact kind, or handoff manifests as valid route evidence
  should be removed or rewritten.

## Non-goals

- No Common EmitC rebuild.
- No executable plugin construction template.
- No new Template, Toy, TensorExtLite, or Offload implementation.
- No new runtime ABI, artifact exporter, direct C route, descriptor route,
  helper wrapper, compatibility mode, or legacy quarantine.
- Do not restore RVV or scalar metadata routes.
- Do not turn this into broad unrelated comment cleanup while active route
  code/tests remain.

## Acceptance Criteria

- [x] Common plugin emission-plan validation treats metadata-only plans as
      unsupported/fail-closed rather than supported route ownership.
- [x] Template, Toy, TensorExtLite, and Offload plugin/target surfaces no
      longer publish metadata-only artifact routes as active compiler routes.
- [x] Tests no longer assert metadata-only selected emission, runtime ABI,
      runtime glue, artifact kind, handoff manifest, or target artifact route
      as valid evidence.
- [x] Focused active-surface ref-scan for the specified metadata-only route
      strings returns no active code/test/spec hits except task/archive or
      explicit current-task documentation.
- [x] `ninja -C build tcrv-opt tcrv-translate` passes.
- [x] Affected C++ tests and targeted lit/FileCheck tests pass or any failures
      are reported as deletion-exposed missing architecture gaps without
      restoring metadata-only route authority.
- [x] `ninja -C build check-tianchenrv` is attempted.
- [x] `git diff --check` passes.
- [x] Trellis task validation passes, task is finished/archived, worktree is
      clean after commit.

## Technical Notes

- Required specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Required active code surfaces:
  - `lib/Plugin/ExtensionPlugin.cpp`
  - `include/TianChenRV/Transforms/Passes.td`
  - `lib/Plugin/Template/TemplateExtensionPlugin.cpp`
  - `lib/Plugin/Toy/ToyConstructionProtocol.cpp`
  - `lib/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.cpp`
  - `lib/Plugin/Offload/`
  - `lib/Target/Template/TemplateMetadataArtifact.cpp`
  - `lib/Target/Toy/ToyMetadataArtifact.cpp`
  - `lib/Target/TensorExtLite/TensorExtLiteMetadataArtifact.cpp`
  - directly related tests under `test/Transforms/ExecutionPlanning`,
    `test/Transforms/EmissionReadiness`,
    `test/Transforms/ExecutionPlanCoherence`, and `test/Target/`.
