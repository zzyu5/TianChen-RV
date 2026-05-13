# First concrete extension-family template instantiation

## Goal

Instantiate the first minimal non-RVV extension-family consumer of the
Extension-Family Plugin Construction Protocol. The concrete family for this
round is `Toy`, because the repository already has a plugin-local Toy
dialect/plugin/target surface that currently stops at metadata-only planning.
This task upgrades that surface into a protocol-backed construction consumer
with a semantic role graph, a minimal ODS role op, plugin-local planning, and a
generated artifact route without adding extension-specific semantic branches to
core orchestration.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD at task creation is
  `d601379 feat(template): materialize ods role op boundary`.
* The worktree was clean before creating this task.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief and then started as the current Trellis task.
* The previous Template tasks added:
  * a construction manifest;
  * a role-graph-to-EmitC generated output route;
  * a typed role/interface realization layer;
  * a minimal ODS role-op boundary `tcrv_template.compute_skeleton`;
  * selected-boundary materialization, plugin legality, emission-plan metadata,
    target artifact preflight, and focused fail-closed tests.
* Existing `Toy` files already exist under `include/TianChenRV/Dialect/Toy/`,
  `lib/Dialect/Toy/`, `include/TianChenRV/Plugin/Toy/`,
  `lib/Plugin/Toy/`, `include/TianChenRV/Target/Toy/`,
  `lib/Target/Toy/`, `test/Plugin/`, and `test/Target/ToyMetadataArtifact/`.
* Existing Toy behavior is intentionally metadata-only:
  `tcrv_toy.lowering_boundary`, `toy.template` capability,
  `toy_template_first_slice` variant, and `toy_metadata_artifact` output. It
  does not yet declare a construction manifest/archetype, typed role graph,
  minimal compute role op, role-op interface validation, or generated
  role-graph-to-EmitC output.

## Module Goal

Make `Toy` the first concrete extension-family instantiation of the protocol:

```text
Toy construction manifest / archetype
  -> Toy semantic role graph
  -> Toy family declaration
  -> Toy minimal ODS compute role op
  -> generated TCRVEmitCLowerableOpInterface provenance
  -> Toy plugin-local proposal / legality / selected boundary
  -> Toy emission-plan selected metadata
  -> Toy target artifact preflight
  -> generated role-graph-to-EmitC source-like output
```

The generated output remains source-like construction evidence only. It must
not claim runtime ABI glue, linked code, hardware execution, correctness, or
performance.

## Boundaries

* Scope is the existing Toy extension family plus focused CMake/tests.
* Compiler behavior stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Python may only run Trellis/tooling checks.
* Toy remains a TCRV extension family inside the unified system, not an
  independent backend dialect.
* Toy-specific logic belongs in Toy dialect/plugin/target/construction code.
* Shared edits are allowed only for generic registration/build surfaces needed
  to wire the Toy family; do not add Toy semantic branches to core passes.
* Do not move computation semantics into `tcrv.exec`.
* Do not add descriptor-driven computation, descriptor-to-C export, hardware
  runtime, correctness, or performance behavior.
* Do not expand RVV, scalar, offload, IME, dtype, LMUL, runtime, or benchmark
  behavior except for regression checks if a shared surface is touched.

## Requirements

* Define a Toy construction protocol surface that records protocol version,
  archetype, semantic role graph, family declaration, common-interface
  realization, EmitC route mapping, typed role realization, and evidence
  profile.
* The Toy semantic role graph must include enough ordered role metadata to
  validate the extension pattern, with at least `configure -> load -> compute
  -> store`.
* Declare a minimal Toy ODS compute role op, such as
  `tcrv_toy.compute_skeleton`, implementing
  `TCRVEmitCLowerableOpInterface`.
* Materialize or validate the Toy compute role op as part of the selected Toy
  path before target artifact export.
* The Toy plugin proposal, legality, readiness, and emission-plan hooks must
  consume protocol-derived metadata instead of ad hoc metadata-only strings.
* The Toy target artifact route must validate:
  * selected variant origin and required capability;
  * construction protocol version and archetype;
  * semantic role graph and ordered role metadata;
  * common-interface realization and typed role realization;
  * generated op-interface source op/source role;
  * EmitC role-to-call mapping;
  * evidence profile;
  * exactly one matching Toy compute role op before export.
* Generated output must include deterministic construction evidence fields and
  a source-like role-graph-to-EmitC skeleton derived from Toy typed role data.
* Missing, stale, duplicate, wrong-role, wrong-op, wrong-interface, stale
  selected-plan metadata, or malformed capability/profile data must fail closed
  before generated artifact output.
* Existing Template ODS role-op boundary and generated route regressions must
  still pass.
* RVV plugin regressions must pass if shared plugin/interface/target registry
  code is touched.
* Confirm core-pass neutrality by scanning common transform/core dialect paths
  for accidental Toy-specific branches.

## Acceptance Criteria

* [x] Toy declares a construction manifest/archetype/role graph through C++
      compiler code, not through passive documentation.
* [x] Toy has a minimal ODS role op implementing
      `TCRVEmitCLowerableOpInterface`.
* [x] Toy selected-boundary materialization creates or requires the minimal
      Toy role op for the selected variant.
* [x] Toy plugin-local legality/readiness/emission-plan hooks validate the
      protocol-derived fields.
* [x] Toy target artifact preflight validates the manifest, typed role graph,
      role op/interface identity, selected-plan metadata, and EmitC mapping.
* [x] Toy generated artifact output includes construction protocol,
      archetype, role graph, family declaration, interface realization, EmitC
      route, evidence profile, ordered generated steps, and deterministic
      source-like calls.
* [x] Positive lit/FileCheck proves the valid Toy path reaches generated
      artifact output.
* [x] Negative lit/FileCheck or C++ coverage proves stale/missing/wrong-role
      Toy cases fail closed before export.
* [x] Focused C++ tests cover construction manifest, role graph, family
      declaration, plugin registration/planning, role op/interface validation,
      selected metadata, and target artifact validation.
* [x] Template generated route and Template ODS role-op regressions still pass.
* [x] RVV plugin tests pass if shared plugin/interface code is touched.
* [x] No Toy semantic branch is added to `tcrv.exec`, `lib/Transforms`, or
      common orchestration passes.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis validation passes before finish/archive and after archive.
* [x] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Definition Of Done

* Focused build targets pass, including generated Toy dialect headers,
  `TianChenRVToyDialect`, `TianChenRVToyPlugin`, `TianChenRVToyTarget`,
  Template regression targets, `tcrv-opt`, and `tcrv-translate`.
* Focused C++ plugin/target tests pass for Toy construction and artifact
  validation.
* Focused lit/FileCheck tests pass for Toy dialect/op and generated artifact
  route output/failure cases.
* `git diff --check` passes before commit and `git diff --cached --check`
  passes after staging.
* Trellis task status, context, and archive state are truthful.

## Out Of Scope

* Documentation-only template work.
* Checklist-only, manifest-only, helper-only, or broad smoke-only progress as
  the main result.
* New RVV vmul/i64/LMUL/dtype/runtime/performance work.
* Descriptor cleanup unless it becomes the single blocker for Toy construction
  protocol validation.
* Python compiler implementation.
* Direct descriptor-to-C exporter architecture.
* Moving computation semantics into `tcrv.exec`.
* Hardware runtime, correctness, or performance claims.

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
  * `.trellis/tasks/archive/2026-05/05-13-template-ods-role-op-boundary/prd.md`
  * `.trellis/tasks/archive/2026-05/05-13-05-13-template-typed-role-op-construction-path/prd.md`
* Workspace journal read:
  * `.trellis/workspace/codex/journal-5.md` recent Template/Toy-related
    entries.
* Initial source surfaces inspected:
  * `include/TianChenRV/Dialect/Toy/IR/ToyOps.td`
  * `lib/Dialect/Toy/IR/ToyDialect.cpp`
  * `include/TianChenRV/Plugin/Toy/ToyExtensionPlugin.h`
  * `lib/Plugin/Toy/ToyExtensionPlugin.cpp`
  * `include/TianChenRV/Target/Toy/ToyMetadataArtifact.h`
  * `lib/Target/Toy/ToyMetadataArtifact.cpp`
  * `include/TianChenRV/Plugin/Template/TemplateConstructionProtocol.h`
  * `lib/Plugin/Template/TemplateConstructionProtocol.cpp`
  * `test/Plugin/ToyExtensionPluginTest.cpp`
  * `test/Target/ToyMetadataArtifact/`

## Completion Notes

Implemented the first concrete extension-family consumer of the construction
protocol by upgrading the existing Toy family from metadata-only integration
to protocol-backed generated artifact evidence:

* Added `ToyConstructionProtocol` with a C++ construction manifest, archetype,
  semantic role graph, family declaration, common-interface realization,
  typed role realization, EmitC route mapping, generated output route, and
  fail-closed verifiers.
* Added `tcrv_toy.compute_skeleton` as a minimal ODS-backed Toy role op
  implementing `TCRVEmitCLowerableOpInterface`.
* Wired Toy proposal/materialized variant metadata through protocol-derived
  fields for construction protocol, archetype, role graph, common interfaces,
  typed roles, EmitC route, and evidence profile.
* Updated Toy selected-boundary materialization to emit both
  `tcrv_toy.lowering_boundary` and `tcrv_toy.compute_skeleton`, then validate
  the compute op against the Toy construction manifest and typed role graph.
* Updated Toy emission planning and target artifact route metadata to require
  the protocol selected-plan metadata before export.
* Updated Toy target artifact preflight to validate the manifest, typed role
  graph, selected metadata, lowering boundary, exactly one matching compute
  role op, generated op-interface provenance, and EmitC role mapping.
* Extended Toy generated artifact output with construction protocol,
  archetype, semantic role graph, family declaration, semantic roles,
  interface realization, validated role-op provenance, typed role graph,
  EmitC route, evidence profile, ordered `generated_emitc_step[...]`, and a
  deterministic source-like `generated_source` skeleton.
* Added focused C++ coverage in `tianchenrv-toy-extension-plugin-test` for the
  Toy construction manifest, typed role graph, generated route, proposal
  metadata, selected compute role materialization, op-interface validation,
  and emission-plan selected metadata.
* Added lit/FileCheck coverage for `tcrv_toy.compute_skeleton` parse/verify,
  Toy generated artifact positive output, and missing compute role-op
  fail-closed export.
* Confirmed no Toy semantic branches were added to `lib/Transforms` or core
  `tcrv.exec`.

Checks run:

```bash
cmake --build build --target TianChenRVToyDialect TianChenRVToyPlugin TianChenRVToyTarget tianchenrv-toy-extension-plugin-test tcrv-opt tcrv-translate -j2
./build/bin/tianchenrv-toy-extension-plugin-test
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Toy|toy'
cmake --build build --target TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tianchenrv-target-artifact-export-test -j2
./build/bin/tianchenrv-template-extension-plugin-test
./build/bin/tianchenrv-target-artifact-export-test
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Template|template'
cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2
./build/bin/tianchenrv-rvv-extension-plugin-test
rg -n "tcrv_toy|toy-plugin|Toy" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec || true
git diff --check
python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-first-concrete-extension-family-template-instantiation
python3 ./.trellis/scripts/task.py archive 05-13-first-concrete-extension-family-template-instantiation --no-commit
python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-13-first-concrete-extension-family-template-instantiation
git diff --cached --check
```

`clang-format` was not available as `clang-format` or
`/usr/lib/llvm-20/bin/clang-format`; formatting was kept manually consistent
with nearby code and `git diff --check` passed.

No `ssh rvv` evidence was run because this task changes Toy construction and
artifact validation only. It does not change RVV emitted runtime artifacts and
makes no RVV runtime, correctness, or performance claim.
