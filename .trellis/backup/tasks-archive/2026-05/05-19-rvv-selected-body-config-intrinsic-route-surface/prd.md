# RVV selected-body config and intrinsic route surface

## Goal

Replace the active RVV provider / extension boundary route surface so the
current supported RVV path is selected from an explicit typed selected-body
route descriptor. The retained i32m1 add/sub/mul/compare-select behavior must
remain supported, but only as an ordinary specialization derived from typed
`tcrv_rvv` body, config, memory, runtime ABI, and boundary facts.

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo checks for this round:
  * `pwd` -> `/home/kingdom/phdworks/TianchenRV`
  * `git status --short` -> clean
  * `git log --oneline -8` begins with
    `b9f410e rvv: demote source front-door route authority`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-source-front-door-route-authority-demotion/prd.md`
  removed source-front-door route metadata authority and moved readiness /
  planning toward provider-derived route descriptions.
* Specs require the authority chain:
  selected RVV variant -> explicit typed `tcrv_rvv` body -> RVV-owned route
  description -> provider-built `TCRVEmitCLowerableRoute` -> neutral common
  EmitC / target export mechanics.
* Current code still exposes Stage 1 debt in the provider / extension boundary:
  * `lib/Plugin/RVV/RVVExtensionPlugin.cpp` uses
    `findSelectedRVVI32M1WithVLBoundary` and
    `validateSelectedRVVI32M1WithVLBoundary` as the selected-body gate.
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` derives body shape, but
    intrinsic/type mapping is still keyed by operation-only
    `RVVSelectedBodyIntrinsicMapping` entries with exact i32m1 intrinsic
    spellings.
  * `RVVSelectedBodyEmitCRouteDescription` carries operation/memory/SEW/LMUL
    but not the full config policy, boundary structure, C type, setvl/load/
    store, and mapping fields needed to make the descriptor the visible route
    surface.
  * `buildVariantEmissionPlan` still appends
    `getRVVI32M1ArithmeticArtifactMetadata()` directly rather than making the
    plan mirror provider-derived descriptor/config metadata.

## Requirements

* Provider analysis must assemble an explicit selected-body route descriptor
  from typed body facts, including operation kind, memory form, SEW, LMUL,
  tail/mask policy, selected boundary op structure, runtime ABI parameter
  roles, vector/mask C types, setvl/load/broadcast/store intrinsic choices, and
  compute/compare intrinsic choices.
* Intrinsic/type/EmitC mapping must be resolved from that descriptor and fail
  closed for unsupported dtype/SEW, LMUL, policy, memory form, operation, or
  boundary combinations.
* Existing typed i32m1 add/sub/mul/compare-select bodies must still build
  provider-derived routes and materialize through common EmitC as the retained
  specialization.
* RVV extension readiness, planning, and lowering-boundary validation must use
  typed selected-body route description / provider route validation as the
  family gate. The extension boundary must not expose `RVVI32M1Arithmetic` or
  `findSelectedRVVI32M1...` naming as the route-support protocol.
* Construction/artifact metadata used by emission plans must mirror the
  provider-derived route/config description after body validation; stale route
  ids, artifact names, helper names, or old metadata must not authorize a route.
* Keep common EmitC / target export semantic-neutral. RVV operation, dtype,
  SEW/LMUL, policy, memory form, ABI use, and intrinsic selection remain in
  RVV provider/plugin code.

## Acceptance Criteria

* [x] `RVVSelectedBodyEmitCRouteDescription` or an equivalent provider-owned
      descriptor explicitly records config policy, boundary structure, runtime
      ABI roles, C type, and intrinsic mapping facts derived from the selected
      typed body.
* [x] Provider route construction uses descriptor-resolved mapping, not a bare
      operation-only intrinsic lookup, and reports targeted errors for
      unsupported config/memory/operation combinations before route/export.
* [x] Existing explicit typed i32m1 selected bodies still produce supported
      readiness, emission plans, and EmitC lowerable routes.
* [x] Negative provider/plugin coverage proves unsupported or mismatched
      selected-body config facts fail closed before route/export.
* [x] RVV extension boundary helpers and diagnostics no longer present
      `findSelectedRVVI32M1...` / `validateSelectedRVVI32M1...` as the
      selected-body gate where typed RVV selected-body route support is meant.
* [x] Emission-plan artifact/config metadata is rebuilt from provider-derived
      selected-body description after validation.
* [x] A bounded ref-scan of
      `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
      `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
      `lib/Plugin/RVV/RVVExtensionPlugin.cpp` shows the remaining i32m1 strings
      are implementation specialization labels, not public route-authority
      gates.

## Implementation Notes

* Added selected-body config/VL structure helpers so provider analysis can
  derive SEW, LMUL, policy, and visible VL structure before checking whether a
  concrete specialization is route-supported.
* Expanded `RVVSelectedBodyEmitCRouteDescription` with config contract,
  boundary, loop, runtime ABI, C type, and intrinsic mapping fields.
* Replaced the operation-only intrinsic lookup with descriptor-keyed mapping.
  Current SEW32/LMUL m1/agnostic add/sub/mul/compare-select remains supported;
  unsupported descriptor combinations fail closed before route/export.
* Renamed RVV extension plugin internal selected-boundary helpers and
  diagnostics to selected-body terminology, and removed the lmul-specific
  conformance check from the extension gate so unsupported LMUL reaches the
  provider descriptor rejection path.
* `buildVariantEmissionPlan` now adds runtime ABI parameters and
  `tcrv_rvv.*` config metadata from the provider-derived description rather
  than appending the old fixed config metadata directly.
* Focused plugin tests now assert descriptor fields on positive i32m1 bodies,
  keep stale route metadata negative coverage, and add an unsupported LMUL m2
  selected-body rejection test.

## Out Of Scope

* No new RVV coverage: no broadcast expansion, reduction, conversion, new
  dtype, new LMUL, new source-shape, or new intrinsic family.
* No high-level Linalg/Vector/StableHLO frontend generalization.
* No Scalar, IME, Offload, TensorExt, autotuning, dashboards, broad smoke
  matrices, report/status machinery, or artifact-index-only evidence.
* No compatibility wrapper preserving old i32 route authority.
* No common EmitC/export RVV semantic branch.
* No descriptor-driven computation or direct-C semantic export.
* No runtime, correctness, or performance claim without real `ssh rvv`
  evidence. This task does not require such a claim.

## Focused Checks

* Build and run `tianchenrv-rvv-extension-plugin-test`.
* Build and run focused target artifact export coverage if touched behavior
  crosses target exporter preflight.
* Run focused lit/FileCheck cases under `test/Transforms/RVV` and
  `test/Target/RVV` only when their expected provider-derived metadata changes.
* Run `git diff --check`.
* Run bounded ref-scans for old route-authority residue in the touched provider
  and extension files.

## Checks Run

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build build --target tcrv-opt -j2`
* `build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir --tcrv-rvv-materialize-i32m1-vector-source-front-door --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir --check-prefix=PLAN`
* `rg -n "RVVI32M1Arithmetic|findSelectedRVVI32M1|validateSelectedRVVI32M1|route-id|artifact-name" include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp lib/Plugin/RVV/RVVExtensionPlugin.cpp || true`
* `rg -n "__riscv_" include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp lib/Plugin/RVV/RVVExtensionPlugin.cpp || true`
* `git diff --check`

## Definition Of Done

* Code implements one coherent provider/extension-boundary route-surface
  correction.
* Focused C++ / lit checks for touched behavior pass.
* Trellis task status, PRD, and workspace journal are truthful.
* If complete, archive the task using repo Trellis convention.
* Create one coherent commit for the round.

## Technical Notes

Specs and prior task read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/tasks/archive/2026-05/05-19-rvv-source-front-door-route-authority-demotion/prd.md`

Relevant implementation files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
* `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* Focused tests under `test/Transforms/RVV` and `test/Target/RVV`.
