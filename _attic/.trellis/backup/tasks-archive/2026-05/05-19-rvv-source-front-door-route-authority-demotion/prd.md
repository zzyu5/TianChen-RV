# RVV source-front-door route-authority demotion

## Goal

Keep the bounded RVV vector source front door only as a source-to-typed-body
adapter. Route support, runtime ABI names, lowering pipeline route ids, and
artifact metadata must be derived after RVV provider validation of the explicit
selected `tcrv_rvv` body, not from source patterns, stale lowering seeds,
`with_vl` route metadata, or construction manifest route ids stamped at the
selected boundary.

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo checks for this round:
  * `pwd` -> `/home/kingdom/phdworks/TianchenRV`
  * `git status --short` -> clean
  * `git log --oneline -8` begins with
    `f43baae rvv: select body construction protocol boundary`
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-construction-protocol-boundary/prd.md`
  moved construction and target artifact naming toward selected-body terms.
* Specs require the authority chain:
  selected `tcrv.exec` RVV variant -> explicit typed `tcrv_rvv` body -> RVV
  plugin validation/route description -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC/export mechanics.
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` still stamps
  `rvv_emitc_route_mapping` onto generated `tcrv_rvv.with_vl` using
  `getRVVConstructionManifest().emitcRoute.routeID`.
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp` still requires that route mapping as
  selected-boundary conformance metadata and reports/plans with manifest route
  ids before or alongside provider-selected route descriptions.
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already has
  `describeRVVSelectedBodyEmitCRoute` and
  `buildRVVSelectedBodyEmitCLowerableRoute`, which are the correct provider
  authority APIs for this round.

## Requirements

* Preserve the explicit public RVV source-front-door pass as a bounded,
  explicit-only source adapter for the current vector i32 add/sub/mul slice.
* The source adapter must materialize explicit selected `tcrv_rvv` body
  structure and runtime ABI bindings, but must not stamp route id or
  `rvv_emitc_route_mapping` metadata as selected-boundary authority.
* RVV selected-boundary validation must not require a manifest route id or
  `rvv_emitc_route_mapping` attribute on `tcrv_rvv.with_vl` before provider
  analysis.
* RVV emission readiness must derive the supported route id from
  `describeRVVSelectedBodyEmitCRoute` or
  `buildRVVSelectedBodyEmitCLowerableRoute` after validating the selected typed
  body.
* RVV emission planning must use the provider-derived selected-body route
  description for route id and runtime ABI name. Artifact metadata must be
  rebuilt for that provider-derived route id after body validation.
* Stale source/front-door/lowering-seed/route metadata must not make an
  otherwise unsupported or missing selected typed body route-supported.
* Keep common EmitC/export semantic-neutral. RVV operation, dtype, SEW/LMUL,
  policy, memory form, runtime ABI use, and intrinsic mapping stay in RVV
  plugin/provider code.

## Acceptance Criteria

* [x] Source-front-door materialization output no longer contains
      `rvv_emitc_route_mapping` on `tcrv_rvv.with_vl`.
* [x] RVV selected-boundary validation succeeds for a provider-supported typed
      body without route mapping metadata on `with_vl`.
* [x] `checkVariantEmissionReadiness` reports the provider-derived selected-body
      route id, not the construction manifest family route id.
* [x] `buildVariantEmissionPlan` uses provider-derived route id/runtime ABI and
      provider-route-derived artifact metadata after body validation.
* [x] Negative coverage proves stale route metadata on `with_vl` cannot
      authorize route/export when the explicit body is missing or unsupported.
* [x] Focused source-front-door lit coverage still proves the retained bounded
      adapter produces a provider-supported typed body.
* [x] A bounded ref-scan of `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` and
      `lib/Plugin/RVV/RVVExtensionPlugin.cpp` shows no remaining
      `getRVVConstructionManifest().emitcRoute.routeID` authority use.

## Implementation Notes

* `RVVVectorSourceFrontDoor.cpp` now materializes `with_vl` without
  `rvv_emitc_route_mapping`; it still materializes explicit runtime ABI,
  `setvl`, `with_vl`, load/compute/store body structure, and bounded source
  provenance.
* `RVVExtensionPlugin.cpp` no longer requires route mapping metadata as an
  extra selected-boundary attribute. Readiness and emission planning call
  `describeRVVSelectedBodyEmitCRoute` after selected body validation and use
  its target route/runtime/artifact fields.
* `RVVSelectedBodyEmitCRouteDescription` now carries the target artifact route
  id and artifact kind after provider body analysis, so plugin/target callers
  do not consult manifest route ids before selected body validation.
* RVV target candidate preflight now checks candidate route id/artifact kind
  after rebuilding the selected-body route description, and the construction
  template boundary adapter no longer requires `rvv_emitc_route_mapping`.
* Focused tests updated:
  * source-front-door lit checks assert no `rvv_emitc_route_mapping` appears
    on the materialized selected boundary.
  * RVV plugin C++ coverage compares readiness/plan route fields against the
    provider description and rejects stale route metadata on an empty
    `with_vl` body.

## Checks Run

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `cmake --build build --target tcrv-opt -j2`
* Source-front-door FileCheck commands for add/sub/mul boundary and emission
  plan paths.
* `build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir --split-input-file --verify-diagnostics --tcrv-rvv-materialize-i32m1-vector-source-front-door`
* `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build build --target tcrv-translate -j2`
* RVV `emitc-to-cpp` handoff FileCheck commands.
* Focused target artifact object/stale-route checks for
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.
* `rg -n "getRVVConstructionManifest\\(\\)\\.emitcRoute\\.routeID|rvv_emitc_route_mapping" lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp lib/Plugin/RVV/RVVExtensionPlugin.cpp || true`
* `git diff --check`

## Out Of Scope

* No new RVV coverage: no broadcast, compare/select, reduction, conversion,
  dtype, LMUL, source-shape, or intrinsic expansion.
* No Linalg/Vector/StableHLO frontend generalization.
* No Scalar, IME, Offload, TensorExt, autotuning, dashboards, broad smoke
  matrices, report/status machinery, or artifact-index-only evidence.
* No compatibility wrapper preserving source-pattern or route-id authority.
* No common EmitC/export RVV semantic branch.
* No descriptor-driven computation or direct-C semantic export.
* No runtime, correctness, or performance claim without real `ssh rvv`
  evidence. This task does not require such a claim.

## Definition Of Done

* Code implements the source-front-door and RVV plugin authority demotion.
* Focused lit/C++ checks for touched source-front-door and plugin-boundary
  behavior pass.
* `git diff --check` passes.
* Trellis task status is truthful; if finished, archive using Trellis task
  convention.
* One coherent commit records the completed bounded round.

## Technical Notes

Relevant specs and prior task:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-construction-protocol-boundary/prd.md`

Relevant implementation areas:

* `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* Focused tests under `test/Transforms/RVV`, `test/Plugin`, and `test/Target`
  that cover source-front-door materialization, selected-boundary validation,
  and artifact export.
