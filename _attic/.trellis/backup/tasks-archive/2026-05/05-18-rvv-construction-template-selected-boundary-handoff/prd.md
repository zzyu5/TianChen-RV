# RVV construction-template selected-boundary handoff

## Goal

Require the existing bounded RVV i32m1 materialized EmitC target artifact path
to consume the common construction-template selected-lowering-boundary adapter
before object, declaration-header, bundle, or RVV EmitC-to-C++ output. This
continues the TensorExtLite handoff from commit `a779db3` and proves the common
adapter is a production target-support boundary for RVV, not only a
construction-template example.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree is clean and HEAD is
  `a779db3 target: route tensorextlite through construction template boundary`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
- TensorExtLite now validates its selected lowering boundary through
  `ConstructionTemplateArtifactAdapterConfig`.
- RVV target support already uses `ConstructionTemplateArtifactAdapter` for
  object/header/bundle registration, but its adapter config does not yet
  require common selected-lowering-boundary validation.
- RVV still exposes a public `tcrv-rvv-emitc-to-cpp` translate route that calls
  the materialized EmitC C/C++ emitter directly.
- The RVV selected boundary is the existing nested `tcrv_rvv.with_vl` operation
  in the selected variant body. It is not the deleted RVV lowering-boundary op
  and not a direct `*.lowering_boundary` marker.
- RVV-specific validation may remain only for RVV-owned semantics: typed
  i32m1 body/config/VL/AVL policy, route/source-op provenance,
  arithmetic-op/route agreement, runtime ABI parameter mapping, and RISC-V
  object packaging.

## Requirements

1. Keep the production path:
   selected RVV path -> explicit typed `tcrv_rvv` i32m1 add/sub/mul body ->
   selected `tcrv_rvv.with_vl` boundary -> RVV-owned EmitC route -> common
   construction-template adapter -> object/header/bundle or C++ emitter.
2. Add RVV adapter config for the selected `tcrv_rvv.with_vl` boundary:
   selected variant, source kernel, origin plugin, role, status, required
   capabilities, and RVV-local route attributes such as LMUL must be validated
   through the common adapter before output.
3. The common adapter may be extended only generically, with plugin-supplied
   configuration, to support selected boundaries that live inside the selected
   variant body. It must not branch on RVV names or interpret RVV computation.
4. The RVV target object, header, bundle, and public RVV EmitC-to-C++ front
   door must all use the common construction-template adapter boundary check.
5. Keep RVV route-local validation for body/route/ABI/object-package semantics
   that the common selected-boundary contract cannot express.
6. Missing boundary, duplicate boundary, stale selected variant, wrong origin,
   wrong role/status, missing required capabilities, stale RVV route-local
   attrs, mismatched route/source-op provenance, and forbidden descriptor,
   direct-C, source-export, or compute-body metadata must fail closed before
   artifact or C++ bytes become authoritative.
7. Keep common/core orchestration family-neutral. Do not add a common/core
   branch on RVV, TensorExtLite, Toy, Template, dtype, shape, runtime,
   toolchain, or microarchitecture semantics.

## Acceptance Criteria

- [x] RVV `ConstructionTemplateArtifactAdapterConfig` requires common selected
      boundary validation for `tcrv_rvv.with_vl`.
- [x] The common adapter supports a plugin-configured nested selected-variant
      boundary without hardcoding RVV semantics.
- [x] RVV object, header, bundle, and `tcrv-rvv-emitc-to-cpp` output all pass
      through the common adapter selected-boundary validation before output.
- [x] RVV-specific duplicate/preflight logic is shrunk or retained only where
      it checks RVV-owned config/VL/body/ABI/packaging semantics.
- [x] Focused C++ coverage proves common-adapter selected-boundary failure for
      missing/stale nested RVV boundary conditions.
- [x] Focused lit coverage proves RVV source and materialized object/header/
      bundle paths still emit the relocatable object, declaration-only header,
      and coherent bundle through the materialized EmitC route.
- [x] Focused negative lit or C++ coverage proves stale/missing boundary and
      descriptor/direct-C/source-export residue fail closed before RVV output.
- [x] No new RVV ops, dtype/SEW/LMUL families, generic RVV lowering, direct
      source exporters, descriptor adapters, compatibility routes, Python
      compiler-core behavior, or common/core RVV semantic branches are added.

## Non-Goals

- No new RVV operation families, dtype expansion, SEW/LMUL expansion,
  sub/mul source-expansion work, or generic RVV lowering.
- No new direct source exporters, descriptor adapters, compatibility routes,
  Python compiler-core behavior, or direct-C semantic printers.
- No TensorExtLite feature expansion.
- No new common/core family semantic branch.
- No runtime, correctness, or performance claim beyond preserving or
  refreshing existing `ssh rvv` object/harness evidence when practical.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-construction-template-selected-boundary-handoff`
- Focused build targets for touched code and tests, especially
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-construction-protocol-common-test`, `tcrv-opt`, and
  `tcrv-translate`.
- Focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-construction-protocol-common-test`.
- Focused lit tests under `test/Target/RVV`, relevant
  `test/Transforms/RVV`, and RVV EmitC-to-C++ handoff coverage.
- `git diff --check`.
- Targeted residue scans over touched common/plugin/target/test surfaces.
- `cmake --build build --target check-tianchenrv -j2` if practical.
- `ssh rvv` generated bundle ABI evidence may be preserved from the most recent
  unchanged object/harness proof; if artifact semantics change, refresh it and
  report the artifact directory.

## Definition of Done

- The RVV materialized EmitC target artifact path consumes the same common
  construction-template selected-boundary adapter as TensorExtLite.
- The selected RVV artifact handoff remains explicit-RVV-IR backed,
  materialized-EmitC backed, bundle-coherent, and fail-closed for stale
  boundary or route metadata.
- Trellis task status, journal/archive notes, and one coherent commit
  truthfully reflect the completed scope and any remaining rebuild gaps.

## Completion Notes

- Common construction-template artifact adapter now supports a
  plugin-configured selected boundary inside the selected variant body and can
  temporarily synthesize only missing generic conformance attributes before
  calling the common selected-boundary verifier.
- RVV config now requires the selected `tcrv_rvv.with_vl` boundary, using RVV
  construction-protocol attr names, status `selected-lowering-boundary`, and
  route-local `lmul = "m1"` expectation through the common adapter.
- RVV object, header, bundle, and `tcrv-rvv-emitc-to-cpp` paths all invoke the
  common adapter boundary validation before output. RVV-specific candidate
  checks remain for body/route agreement, config/VL/ABI mapping, forbidden
  metadata, and RISC-V object packaging.
- Added C++ coverage for missing and stale nested RVV selected boundaries
  before C++ output, and added lit coverage for the RVV EmitC-to-C++
  selected-boundary negative path.
- Preserved prior `ssh rvv` evidence from
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-source-bundle-ssh-runtime-abi-proof-add`;
  this round changed validation order/front-door handoff only, not generated
  RVV object semantics or harness behavior.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-construction-template-selected-boundary-handoff`
- `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- RVV EmitC-to-C++ focused FileCheck commands for help, positive, missing
  selected-boundary, and non-selected input behavior.
- RVV materialized and source target artifact focused FileCheck commands for
  object/header/bundle output.
- `git diff --check`
- Targeted residue scan over touched common/RVV target/test surfaces.
- `cmake --build build --target check-tianchenrv -j2` passed 126/126.

## Technical Notes

- Specs read for PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous archived PRD read:
  `.trellis/tasks/archive/2026-05/05-18-tensorextlite-construction-template-handoff/prd.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h`,
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `test/Target/RVV/vector-source-target-artifact-exporters.mlir`,
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`,
  `test/Target/TargetArtifactExportTest.cpp`, and
  `test/Plugin/ConstructionProtocolCommonTest.cpp`.
