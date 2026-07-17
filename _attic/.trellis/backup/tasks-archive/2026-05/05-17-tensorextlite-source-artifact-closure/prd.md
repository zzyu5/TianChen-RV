# TensorExtLite Source-To-Artifact Production Path Closure

## Goal

Close the current TensorExtLite source-to-artifact production/default path, or
prove from current repository evidence that the path is already closed and
name any exact remaining blocker instead of adding another evidence harness.

The production route under review is:

```text
TensorExtLite source-only MLIR marker
  -> TensorExtLite-owned source front-door pass
  -> selected TensorExtLite variant
  -> ordered TensorExtLite role ops plus lowering-boundary marker
  -> TensorExtLite-owned EmitC-lowerable route provenance
  -> common source-artifact front-door pipeline
  -> supported emission-plan candidate
  -> target artifact exporter registry
  -> materialized EmitC through the MLIR C/C++ emitter
  -> TensorExtLite object, callable header, and coherent bundle artifacts
```

## Current Repository Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean at task creation.
- Starting HEAD was `151c2cb test(rvv): prove vector source arithmetic family on ssh rvv`.
- No `.trellis/.current-task` existed, so this task was created from the
  TensorExtLite source-to-artifact direction brief as
  `05-17-tensorextlite-source-artifact-closure`.
- Current specs already define the TensorExtLite construction-template source
  front door and the correct route shape: source marker -> plugin-owned role
  ops -> EmitC route provenance -> common emission/coherence gates -> object
  and declaration-only header/bundle export.
- Current code already contains TensorExtLite role skeleton ops implementing
  `TCRVEmitCLowerableOpInterface`, a TensorExtLite source-front-door pass,
  a TensorExtLite EmitC route provider, and TensorExtLite target object/header
  bundle support.
- The task must inspect and validate whether those surfaces are wired through
  the default/common production path. If that is already true, this round must
  not invent another script or evidence harness.

## Requirements

- Keep TensorExtLite as a bounded non-RVV extension-family proof.
- Limit the behavior to the existing fragment-MMA-like first slice and its
  `configure -> load_frag -> tile_mma -> store_frag` role sequence.
- Ensure the default/common source-artifact pipeline can consume a
  TensorExtLite source-only input and produce a supported selected artifact
  candidate owned by `tensorext-lite-plugin`.
- Ensure the selected candidate is materialized through TensorExtLite role ops
  implementing the common EmitC-lowerable provenance interface.
- Ensure target artifact exporter registration can export TensorExtLite
  relocatable object, declaration-only callable header, and coherent
  object/header bundle artifacts from the same selected route identity and
  runtime ABI metadata.
- Ensure fail-closed coverage exists for stale `lowering_seed` metadata,
  pre-existing selected-boundary residue, missing role ops, stale EmitC route
  mapping, missing materialized EmitC provenance, object/header route mismatch,
  and RVV/Toy/direct-C/descriptor/source-export residue.
- Keep common/core code extension-neutral. Family-specific behavior must remain
  in TensorExtLite plugin/target-owned code or common interfaces.

## Acceptance Criteria

- [x] Focused TensorExtLite source-front-door lit proves source-only MLIR
      reaches selected role ops, lowering-boundary marker, selected diagnostic,
      and supported emission-plan metadata through
      `--tcrv-source-artifact-front-door-pipeline`.
- [x] Focused TensorExtLite EmitC materialization lit proves the selected role
      sequence materializes a route with route/call source-op provenance and no
      RVV intrinsic, descriptor, direct-C, or source-export residue.
- [x] Focused TensorExtLite target artifact lit proves object, header, and
      bundle export from the source-front-door pipeline.
- [x] Focused C++ target artifact checks prove TensorExtLite exporter registry
      shape, route metadata validation, header/object bundle coherence, and
      fail-closed negative cases.
- [x] If a production/default path blocker is found, repair code or tests in
      the smallest TensorExtLite/common-interface surface that owns the issue.
- [x] If no production/default path blocker is found, record that the current
      path is already complete and do not add helper-only scripts or unrelated
      harnesses.
- [x] `git diff --check` passes.
- [x] Focused build and lit/C++ checks pass; run `check-tianchenrv` if
      practical.

## Definition Of Done

- The task state and journal truthfully report whether this round added a
  missing production-path repair or validated an already-complete path.
- No descriptor-driven computation, direct C semantic exporter, source-export
  route, compatibility wrapper, Python compiler-core behavior, RVV-specific
  common branch, or TensorExtLite-as-independent-backend wording is introduced.
- One coherent commit records the task/validation/repair work if complete. If
  validation exposes a larger blocker, leave the task open with the exact next
  continuation point.

## Out Of Scope

- New RVV arithmetic cases, SEW/LMUL expansion, performance matrices, broad
  smoke/report/status artifacts, TensorExtLite runtime correctness claims,
  real tensor/tile performance semantics, new extension families, descriptor
  adapters, direct C semantic exporters, compatibility aliases, or Python
  compiler-core behavior.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-vector-source-arithmetic-family-runtime-closure/prd.md`.
- Primary implementation surfaces inspected:
  `include/TianChenRV/Dialect/TensorExtLite/IR/TensorExtLiteOps.td`,
  `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`, and
  `lib/Transforms/ExecutionPlanningPipeline.cpp`.
- Primary evidence surfaces inspected:
  `test/Transforms/TensorExtLite/`,
  `test/Conversion/EmitC/tensorext-lite-first-slice-materialization*.mlir`,
  `test/Target/TensorExtLite/`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Closure Summary

Current HEAD already contains the requested production/default TensorExtLite
source-to-artifact path. This round did not add source code, helper scripts, or
new evidence harnesses because focused inspection and validation showed the
route is already implemented through existing common interfaces:

- `TensorExtLiteOps.td` defines the lowering-boundary marker and ordered
  TensorExtLite role skeleton ops with generated
  `TCRVEmitCLowerableOpInterface` provenance.
- `TensorExtLiteSourceFrontDoor.cpp` consumes source-only TensorExtLite module
  markers, rejects stale seed/residue inputs, and materializes the selected
  variant, role sequence, lowering-boundary marker, and selected diagnostic.
- `TensorExtLiteEmitCRouteProvider.cpp` builds the TensorExtLite
  `TCRVEmitCLowerableRoute` from the ordered role ops and fails closed for
  missing or reordered role ops.
- `TensorExtLiteExtensionPlugin.cpp` registers the source front-door pass and
  builds a supported emission plan containing TensorExtLite route provenance,
  runtime ABI metadata, and artifact metadata.
- `TensorExtLiteTargetSupportBundle.cpp` registers object/header/bundle target
  artifact exporters, validates selected candidate coherence, requires the
  materialized lowering-boundary marker, invokes the common materialized EmitC
  C/C++ emitter handoff, and packages the object/header bundle.

No remaining production/default-path blocker was found in this round. The only
initial command failure was a lit invocation from the repo root; the same tests
passed when run from `build/test`, which is this repository's established lit
working directory.

## Validation

- Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`.
- Focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`, and
  `./build/bin/tianchenrv-construction-protocol-common-test`.
- Focused lit from `build/test`: TensorExtLite source-front-door,
  TensorExtLite EmitC materialization, TensorExtLite target object/header/
  bundle, unsupported/non-materialized negatives, and target artifact registry
  coverage; 13/13 passed.
- Full project check:
  `cmake --build build --target check-tianchenrv -j2`, 115/115 lit tests
  passed.
- `git diff --check` passed.

## Spec Update Review

No `.trellis/spec/` update was needed. The relevant long-term TensorExtLite
source-front-door, EmitC route, target artifact handoff, and descriptor/direct-C
fail-closed rules already exist in the plugin-protocol and lowering-runtime
specs. This task verified the current implementation against those rules and
did not introduce a new compiler contract.

## Status

Complete. The TensorExtLite source-only MLIR -> plugin-owned role ops ->
common materialized EmitC -> target object/header/bundle production path is
already closed in current HEAD and is covered by focused lit/C++ checks plus
the full `check-tianchenrv` target.
