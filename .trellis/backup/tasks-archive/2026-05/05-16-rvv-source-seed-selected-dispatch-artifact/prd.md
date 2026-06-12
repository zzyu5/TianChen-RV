# RVV source-seed selected-dispatch artifact path

## Goal

Add one bounded source-seed dispatch proof for the existing RVV i32m1 add source
pattern. The accepted source body must materialize a source-derived RVV selected
variant, reference that variant through a coherent `tcrv.exec`
dispatch/case/fallback envelope, and feed the existing emission-readiness,
execution-plan coherence, EmitC, and target artifact export path.

The proof path for this round is:

```text
source MLIR func/vector/scf/arith i32 add shape
  -> source-derived RVV selected variant
  -> tcrv.exec.dispatch / case / fallback organization
  -> plugin-owned RVV selected boundary / EmitC route
  -> target object/header/bundle artifact
  -> ssh rvv evidence for the RVV case
```

## What I Already Know

- The previous task validated the bounded RVV i32m1 add source function shape
  before selected-boundary materialization and preserved source-derived runtime
  ABI provenance through `purpose = "source-arg-N:<role>"`.
- `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp` currently materializes one
  `rvv-plugin` variant plus a direct `variant-selected` diagnostic using
  `selection_kind = "fallback-only"`.
- `lib/Transforms/EmissionReadiness.cpp` already traverses dispatch cases and
  fallback targets, but dispatch cases currently require a separately
  materialized lowering-boundary candidate before emission-plan routing.
- `lib/Transforms/ExecutionPlanCoherence.cpp` already validates selected
  dispatch surfaces, selected emission-plan diagnostics, runtime ABI metadata,
  and target artifact candidates.
- `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir` proves a hand-authored
  selected dispatch surface can export RVV object/header/bundle artifacts from
  the selected RVV case while ignoring an unsupported scalar fallback candidate.
- The task must remain bounded to the existing i32m1 add source shape and the
  existing RVV extension-family ops/config/runtime-VL contract.

## Requirements

- Update the RVV source-seed materialization so the accepted i32m1 add source
  shape produces a selected dispatch envelope instead of only a direct selected
  marker.
- The dispatch envelope must contain:
  - the source-derived RVV i32m1 add variant as a `tcrv.exec.case` target;
  - a conservative scalar fallback envelope as `tcrv.exec.fallback`;
  - explicit available `rvv` and `scalar.fallback` capabilities;
  - stable source-derived runtime ABI values for `lhs`, `rhs`, `out`, and `n`;
  - no scalar fallback compute semantics.
- Keep RVV arithmetic semantics inside the existing `tcrv_rvv` extension-family
  ops. Common/core code may traverse dispatch/fallback and registry interfaces,
  but it must not branch on RVV family semantics.
- Ensure emission-plan materialization can consume the selected dispatch case
  for a plugin route that already validates explicit typed extension-family IR
  and reports the bounded `tcrv_rvv.with_vl` lowering-boundary metadata.
- Ensure target artifact export selects from the selected dispatch case's
  supported emission-plan candidate, not by scanning for a sole variant.
- Keep the implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only be used for evidence tooling if needed.

## Acceptance Criteria

- A positive source-seed dispatch fixture proves:
  - source-derived ABI provenance for `lhs`, `rhs`, `out`, and `n`;
  - selected RVV variant origin `rvv-plugin`;
  - `tcrv.exec.dispatch` with `tcrv.exec.case @<rvv variant>`;
  - `tcrv.exec.fallback @<scalar fallback variant>`;
  - fallback/reference coherence without scalar compute semantics;
  - supported RVV emission-plan metadata for the dispatch case;
  - unsupported scalar fallback emission-plan metadata for the dispatch
    fallback;
  - target artifact candidate selection for the RVV dispatch case;
  - RVV object/header/bundle export.
- Negative tests fail closed for:
  - missing or stale selected variant references;
  - dispatch case targeting a non-direct or wrong-origin variant;
  - missing fallback envelope where required;
  - mismatched runtime ABI/order/metadata between selected path and artifact
    candidate;
  - unrelated source body plus dispatch marker;
  - manually pre-existing `tcrv.exec` or `tcrv_rvv` residue.
- Existing direct RVV selected-boundary/source-seed tests and existing
  hand-authored RVV selected-dispatch/object/header/bundle target tests continue
  to pass.
- A changed-surface scan shows no descriptor-driven compute route, no direct C
  semantic exporter, no Python compiler-core logic, and no extension-specific
  semantic branch added to common/core orchestration.
- Real `ssh rvv` evidence is refreshed for the source-to-dispatch artifact path.

## Non-Goals

- Generic dispatch runtime selection.
- Scalar fallback compute semantics.
- New RVV arithmetic/source families.
- i32m1 sub/mul source lowering.
- i32m2 artifact routes.
- High-level tensor lowering.
- New EmitC or target artifact routes.
- Performance tuning.
- Descriptor or binary-family registries.
- Direct C semantic exporters.
- Python compiler-core logic.
- Core RVV semantic branches.
- Compatibility wrappers.
- Broad smoke matrices.
- Generic RVV backend maturity claims.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  and `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-16-05-16-bounded-source-vector-to-rvv-selected-boundary-lowering/prd.md`.
- Current implementation files inspected:
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Transforms/EmissionReadiness.cpp`,
  `lib/Transforms/ExecutionPlanCoherence.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  and `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
- Current tests inspected:
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`,
  `test/Transforms/SourceSeed/source-seed-artifact-front-door-pipeline-mixed.mlir`,
  `test/Target/RVV/i32m1-selected-dispatch-artifact.mlir`,
  and `test/Target/RVV/i32m1-add-object-artifact.mlir`.

## Minimal Evidence Plan

- Build touched compiler/plugin/target libraries plus `tcrv-opt` and
  `tcrv-translate`.
- Run focused lit for:
  - RVV source-seed selected-boundary/source-dispatch path;
  - source-seed artifact front door;
  - selected-dispatch source positives and negatives;
  - existing RVV selected-dispatch/object/header/bundle target tests.
- Run relevant RVV/plugin/target C++ tests.
- Run `check-tianchenrv` if practical after focused checks pass.
- Refresh `ssh rvv` evidence for the source-selected-dispatch artifact path.
- Run `git diff --check` and changed-surface scans.

## Implementation Summary

- `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp` now materializes the bounded
  source-seed i32m1 add path as a coherent selected-dispatch envelope:
  `@rvv`, `@scalar_fallback`, source-derived `@seed_rvv_i32_add`,
  conservative `@seed_scalar_fallback`, `tcrv.exec.case`, and
  `tcrv.exec.fallback`.
- The scalar fallback remains an envelope/selection fallback only. It has
  fallback identity and policy metadata, but no scalar compute body.
- `lib/Transforms/EmissionReadiness.cpp` now distinguishes references that
  accept optional lowering-boundary materialization from references that require
  it. Dispatch cases can be planned directly through plugin-owned typed
  extension ops, while direct static selected markers still require a boundary.
- `lib/Transforms/EmitCLowerableMaterialization.cpp` now prefers a supported
  selected emission-plan diagnostic when multiple direct variants exist in the
  module. This lets the source-derived RVV dispatch case drive the EmitC
  materialization and artifact route while unsupported fallback diagnostics stay
  non-selected.
- `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir` now proves the
  source body reaches dispatch/case/fallback organization, emission-plan
  metadata, EmitC route materialization, object/header export, and bundle index
  selection for the RVV dispatch case.

## Accepted And Rejected Cases

- Accepted: the validated source `func.func` with `vector.load`, `arith.addi`,
  `vector.store`, runtime `n`, and `tcrv_rvv.lowering_seed = "i32m1_add"` now
  produces a source-derived RVV dispatch case and target artifacts for
  `@seed_rvv_i32_add`.
- Accepted: the conservative scalar fallback variant is referenced through
  `tcrv.exec.fallback` and gets an unsupported dispatch-fallback emission
  diagnostic, but it is not exported as an object/header artifact.
- Rejected by existing and preserved negative coverage: unrelated source body,
  stale or pre-existing `tcrv.exec`/`tcrv_rvv` residue, unsupported source
  marker shapes, mismatched source argument structure, stale selected routes,
  ambiguous or unsupported selected artifact candidates, and selected dispatch
  artifacts with missing/mismatched contract metadata.

## Validation

- Built `tcrv-opt`, `tcrv-translate`, and the touched RVV/plugin/target test
  executables.
- Focused lit:
  `rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door|i32m1-(add-object-artifact|selected-dispatch-artifact|sub-selected-dispatch-artifact|mul-selected-dispatch-artifact)|rvv-first-slice-materialization|emitc-to-cpp-handoff|toy-template-selected-boundary-seed`,
  18/18 selected tests passed.
- Relevant C++ tests passed:
  `tianchenrv-emission-readiness-test`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-dialect-test`, and
  `tianchenrv-target-artifact-export-test`.
- Full check passed:
  `cmake --build build --target check-tianchenrv -j2`, 110/110 lit tests
  passed.
- `git diff --check` passed.
- Changed-surface scan:
  - no Python files changed;
  - changed common/core transforms contain no RVV/scalar/descriptor/direct-C/
    source-export/Python compiler-core terms;
  - changed RVV seed/test diff has no descriptor/direct-C/source-export/Python
    compiler-core route terms. The only intrinsic/header term is the expected
    `riscv_vector.h` EmitC route assertion in the RVV fixture.
- Refreshed source-selected-dispatch artifacts under
  `artifacts/tmp/source_seed_selected_dispatch_artifact/20260516T161432Z`.
- Real RVV run passed:
  `artifacts/tmp/source_seed_selected_dispatch_artifact/20260516T161432Z/ssh_rvv_link_run.log`
  records
  `tcrv_rvv_i32m1_source_selected_dispatch status=PASS n=4 add=[12,6,16,12]`
  with `ssh_status=0`.

## Self-Repair

- Fixed an initial link failure caused by making the RVV selected-boundary seed
  object depend on scalar plugin helper definitions. The final implementation
  uses local scalar fallback identity constants and does not add an RVV plugin
  static dependency on the scalar plugin library.
- Fixed FileCheck ordering in the updated RVV source-seed fixture so it matches
  the actual emission-plan attribute order.
- Renamed the temporary RVV harness status string from the previous
  selected-boundary label to this round's source-selected-dispatch label before
  recording the final `ssh rvv` evidence.

## Spec Update Judgment

No spec update was needed. The existing variant-pipeline, EmitC route,
plugin-protocol, and MLIR testing specs already cover this bounded dispatch
envelope, plugin-owned emission-plan, target artifact, and RVV evidence policy.

## Definition Of Done

- The accepted source-derived RVV variant survives selected dispatch/case/fallback
  organization into artifact export.
- Unsupported or stale source/dispatch combinations fail closed before artifact
  output.
- Focused checks and practical full checks are reported truthfully.
- Trellis task status/context/journal/archive state is updated.
- One coherent commit records the completed round, or the task remains open with
  an exact continuation point.
