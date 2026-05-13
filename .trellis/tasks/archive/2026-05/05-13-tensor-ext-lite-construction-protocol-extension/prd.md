# TensorExtLite construction-protocol extension instantiation

## Goal

Instantiate one richer non-RVV extension family, `TensorExtLite`, through the
Extension-Family Plugin Construction Protocol. This task proves that the
Template/Toy construction protocol is reusable beyond a minimal Toy consumer
and can carry a more meaningful typed IR boundary to generated source-like
artifact output without adding TensorExtLite semantic branches to core
orchestration.

The intended slice is:

```text
TensorExtLite construction manifest / archetype
  -> TensorExtLite semantic role graph
  -> TensorExtLite family declaration
  -> minimal ODS role op or role-op family
  -> common TCRV interface realization
  -> plugin-local proposal / legality / selected planning
  -> EmitC/generated-output mapping
  -> evidence profile and fail-closed artifact validation
```

Generated output is construction evidence only. It must not claim hardware
runtime, linked code, correctness, performance, or vendor availability.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD at task creation is
  `f0e4b31 feat(toy): instantiate construction protocol extension`.
* The worktree was clean before creating this task.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief and must be started before implementation.
* Recent Template/Toy tasks established the reference construction route:
  * Template construction manifest, role graph, typed role realization,
    generated EmitC route, and `tcrv_template.compute_skeleton` role op.
  * Toy construction manifest, role graph, plugin-local planning, target
    artifact validation, generated source-like output, and
    `tcrv_toy.compute_skeleton` role op.
* TensorExtLite should be richer than Toy but bounded: one coherent
  compute-like role path may be completed this round if a full config/load/
  compute/store role-op family is too large.

## Module Goal

Add a concrete `TensorExtLite` extension family under TCRV that reaches
generated artifact output from typed extension-family IR:

```text
tcrv_tensorext_lite role op(s)
  -> TensorExtLite construction protocol validation
  -> TensorExtLite plugin-local proposal / legality / planning
  -> selected boundary and route metadata
  -> target-owned generated-output route
  -> deterministic source-like artifact evidence
```

The preferred MVP is one minimal compute-like ODS role op, such as
`tcrv_tensorext_lite.tile_mma_skeleton`, plus manifest-level configure/load/
store role metadata to validate role ordering. If the code shape makes one
role-op family cheap and localized, add config/load/compute/store role ops;
otherwise finish the compute-role path fully and leave remaining roles staged
truthfully.

## Boundaries

* Scope is the new TensorExtLite extension family plus narrowly required
  CMake, builtin registration, and tests.
* Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Python may only run Trellis/tooling validation.
* TensorExtLite is a TCRV extension family in the unified RISC-V MLIR system,
  not an independent backend dialect.
* TensorExtLite-specific logic belongs in TensorExtLite dialect/plugin/target/
  construction files.
* Shared edits are allowed only for generic registration/build surfaces or
  narrowly extracted helpers that have active Template/Toy/TensorExtLite
  consumers in the same round.
* `tcrv.exec`, generic transforms, and common orchestration passes must not
  receive TensorExtLite semantic branches.
* Do not add descriptor-driven computation, direct descriptor-to-C export, a
  vendor runtime path, linked artifacts, RVV changes, or performance behavior.

## Requirements

* Declare a TensorExtLite construction protocol surface that records protocol
  version, archetype, semantic role graph, family declaration, common-interface
  realization, typed role realization, EmitC/generated-output route mapping,
  and evidence profile.
* The semantic role graph must be richer than Toy by carrying TensorExtLite
  fragment-style ordering, at least `configure -> load_frag -> tile_mma ->
  store_frag`.
* Define a minimal ODS role op or role-op family implementing the appropriate
  generated common interface. At minimum the compute-like role op must
  implement `TCRVEmitCLowerableOpInterface`.
* Register TensorExtLite as a plugin-local family through the existing plugin
  registry/builtin registration conventions.
* Plugin-local proposal, legality, readiness, selected-boundary, and
  emission-plan behavior must consume TensorExtLite protocol-derived metadata
  rather than passive checklist strings.
* Map the TensorExtLite role graph to a generated-output/EmitC-like route
  owned by TensorExtLite target code.
* Target artifact validation must fail closed for stale, missing, duplicate,
  wrong-role, wrong-op, wrong-interface, wrong-route, malformed selected-plan,
  and stale role-realization cases before generated output.
* Generated output must include deterministic construction evidence fields and
  source-like role-to-call steps derived from TensorExtLite typed role data.
* Existing Toy and Template construction routes must continue to pass.
* RVV plugin tests must pass if shared plugin/interface/registry code is
  touched.
* Core-pass neutrality must be checked by scanning core `tcrv.exec` and common
  transform paths for accidental TensorExtLite branches.

## Acceptance Criteria

* [x] TensorExtLite declares a construction manifest/archetype/semantic role
      graph through C++ compiler code, not documentation-only metadata.
* [x] TensorExtLite defines a minimal ODS role op or role-op family under its
      own dialect namespace.
* [x] The TensorExtLite compute-like role op implements
      `TCRVEmitCLowerableOpInterface` and exposes bounded source-op/source-role
      provenance.
* [x] TensorExtLite registers/proposes/plans through plugin-local hooks.
* [x] TensorExtLite selected path materializes or validates the typed role op
      before target artifact export.
* [x] TensorExtLite generated artifact output is derived from validated role
      graph and EmitC/generated-output mapping.
* [x] Positive lit/FileCheck proves a valid TensorExtLite path reaches
      generated artifact output.
* [x] Negative lit/FileCheck or C++ coverage proves stale/missing/wrong-role/
      wrong-interface/wrong-route cases fail closed before export.
* [x] Focused C++ tests cover construction manifest, role graph, family
      declaration, common-interface realization, plugin registration/planning,
      generated-output mapping, and evidence-profile validation.
* [x] Existing Toy and Template construction route regressions pass.
* [x] RVV plugin tests pass if shared plugin/interface code is touched.
* [x] No TensorExtLite semantic branch is added to `tcrv.exec`,
      `lib/Transforms`, or common orchestration passes.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis validation passes before finish/archive and after archive.
* [x] Work is finished, archived, and committed as one coherent commit if the
      task completes.

## Definition Of Done

* Focused build targets pass for TensorExtLite dialect/plugin/target, generated
  headers, `tcrv-opt`, and `tcrv-translate`.
* Focused C++ TensorExtLite plugin/target/construction tests pass.
* Focused lit/FileCheck tests pass for TensorExtLite dialect/op and generated
  artifact route output/failure cases.
* Template and Toy construction regressions still pass.
* No `ssh rvv` evidence is required unless emitted RVV artifacts change.
* Trellis task status, context, archive state, and git commit are truthful.

## Out Of Scope

* Documentation-only templates or checklists as the main result.
* Metadata-only manifest copy without production/default path consumption.
* Helper-only refactors without active TensorExtLite consumption.
* Additional Toy-only polishing unless it is the named blocker for
  TensorExtLite.
* RVV vmul, i64, LMUL, dtype, descriptor cleanup, hardware runtime,
  correctness, or performance expansion.
* Treating TensorExtLite as an independent backend dialect outside the TCRV
  extension-family model.
* Python compiler implementation.
* Direct descriptor-to-C exporter architecture.
* Moving computation semantics into `tcrv.exec`.

## Completion Summary

Implemented `TensorExtLite` as a construction-protocol extension family with:

* `tcrv_tensorext_lite.lowering_boundary` and
  `tcrv_tensorext_lite.tile_mma_skeleton` ODS role-op surface.
* TensorExtLite construction manifest, fragment-MMA-like archetype, semantic
  role graph `configure -> load_frag -> tile_mma -> store_frag`, typed role
  realization, common interface realization, generated EmitC/source-like route
  mapping, and evidence profile validation.
* Plugin-local capability proposal, lowering-boundary validation, selected
  planning metadata, and emission-plan metadata for the
  `tensorext_lite.tile_mma` capability.
* TensorExtLite-owned target artifact exporter for the
  `none-executable-tensorext-lite-fragment-mma-metadata` route.
* Deterministic generated artifact output with no runtime, hardware,
  correctness, or performance claims.
* Fail-closed C++ and lit/FileCheck coverage for stale, missing, wrong-route,
  wrong-role, wrong-interface, malformed selected-plan, and stale role
  realization cases.
* Builtin plugin and target bundle registration without TensorExtLite branches
  in `tcrv.exec`, `lib/Transforms`, or common orchestration passes.

## Checks Run

* `cmake --build build --target TianChenRVTensorExtLiteDialect TianChenRVTensorExtLitePlugin TianChenRVTensorExtLiteTarget tianchenrv-tensorext-lite-extension-plugin-test TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test TianChenRVToyDialect TianChenRVToyPlugin TianChenRVToyTarget tianchenrv-toy-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
* `./build/bin/tianchenrv-template-extension-plugin-test`
* `./build/bin/tianchenrv-toy-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='TensorExtLite|tensorext-lite|tensorext_lite|Toy|toy|Template|template'` from `build/test`
* `rg -n "tensorext|TensorExtLite|tcrv_tensorext_lite" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
* `git diff --check`
* `git diff --cached --check`
* Trellis validation before finish/archive and after archive.

No `ssh rvv` evidence was run because this task changes TensorExtLite
construction/plugin/target metadata evidence only and makes no RVV runtime,
correctness, or performance claim.

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
* Prior reference PRDs read:
  * `.trellis/tasks/archive/2026-05/05-13-first-concrete-extension-family-template-instantiation/prd.md`
  * `.trellis/tasks/archive/2026-05/05-13-template-ods-role-op-boundary/prd.md`
* Workspace journal read:
  * `.trellis/workspace/codex/journal-5.md` recent Template/Toy-related
    entries.
* Initial reference source surfaces to inspect before implementation:
  * `include/TianChenRV/Dialect/Toy/`
  * `lib/Dialect/Toy/`
  * `include/TianChenRV/Plugin/Toy/`
  * `lib/Plugin/Toy/`
  * `include/TianChenRV/Target/Toy/`
  * `lib/Target/Toy/`
  * `include/TianChenRV/Dialect/Template/`
  * `lib/Dialect/Template/`
  * `include/TianChenRV/Plugin/Template/`
  * `lib/Plugin/Template/`
  * `lib/Target/Template/`
