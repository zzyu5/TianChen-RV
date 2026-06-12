# RVV selected-boundary conformance without exporter synthesis

## Goal

Make the bounded RVV i32m1 selected construction-template boundary carry the
conformance facts required by artifact export before export runs, then make the
RVV object/header/bundle and EmitC-to-C++ paths validate those IR-carried facts
without synthesizing missing boundary truth.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree was clean; HEAD was
  `9e55a8e rvv: require construction template boundary`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
- The previous archived task made `tcrv_rvv.with_vl` mandatory as the selected
  construction-template boundary before RVV artifact export.
- The previous task explicitly left a temporary adapter path that can synthesize
  missing generic conformance attributes before verification.
- The active RVV adapter config still sets
  `synthesizeMissingConformanceAttributes = true`.
- `tcrv_rvv.with_vl` currently accepts bounded config attrs only
  (`sew`, `lmul`, `policy`), so source/materialized RVV IR cannot yet carry
  `source_kernel`, `selected_variant`, `origin`, `selected_path_role`,
  `status`, or `required_capabilities` on the selected boundary.
- RVV source-front-door materialization currently creates `tcrv_rvv.with_vl`
  with config attrs only.
- Hand-authored RVV materialized target/export fixtures also currently omit
  selected-boundary conformance attrs and pass only because export can synthesize
  the missing generic attrs.

## Requirements

1. Keep the production route unchanged in shape:
   selected RVV path -> explicit typed `tcrv_rvv` i32m1 add/sub/mul body ->
   selected `tcrv_rvv.with_vl` boundary -> RVV-owned EmitC route -> common
   construction-template adapter -> object/header/bundle or C++ emitter.
2. Allow the selected `tcrv_rvv.with_vl` boundary to carry construction-template
   conformance attrs: source kernel, selected variant, origin plugin, selected
   path role, status, required capabilities, and RVV route/config attributes
   needed by the adapter.
3. Make RVV source-front-door materialization populate those conformance attrs
   on the selected `with_vl` op from the same kernel/variant/requires/path-role
   state that drives the selected dispatch and materialized EmitC route.
4. Update focused hand-authored RVV materialized fixtures so the boundary op is
   the IR source of truth rather than relying on export-time synthesis.
5. Turn off RVV selected-boundary synthesis in the production adapter config.
6. If the common `synthesizeMissingConformanceAttributes` option has no
   remaining legitimate consumer after RVV stops using it, delete the option and
   its helper implementation/tests instead of preserving a compatibility bypass.
7. Keep adapter verification generic: it may validate plugin-supplied attribute
   names and expected values, but it must not branch on RVV or interpret RVV
   compute semantics.
8. Keep RVV-specific validation limited to RVV-owned config/VL/body/ABI/route
   agreement, construction metadata, forbidden residue, and RISC-V object
   packaging.

## Acceptance Criteria

- [x] Positive RVV source-front-door evidence shows `tcrv_rvv.with_vl` itself
      carries source kernel, selected variant, origin plugin, selected path role,
      status, required capabilities, and bounded RVV config attrs.
- [x] Positive RVV hand-authored materialized evidence carries the same
      selected-boundary conformance attrs and still produces object, declaration
      header, bundle, and EmitC-to-C++ output through the materialized EmitC
      route.
- [x] RVV object/header/bundle and `tcrv-rvv-emitc-to-cpp` exports validate the
      IR-carried selected boundary attrs and do not synthesize missing facts.
- [x] Missing selected-boundary attrs fail closed with targeted diagnostics
      before generated C++ source, header, object, or bundle bytes become
      authoritative.
- [x] Stale selected-boundary attrs, including stale source kernel, selected
      variant, origin, role/status, required capabilities, or RVV extra attrs,
      fail closed with targeted diagnostics.
- [x] Existing selected candidate, runtime ABI, materialized EmitC route, and
      object/header/bundle coherence remain tied to the same selected variant
      and route.
- [x] No descriptor-driven computation, direct C/source-export route,
      compatibility wrapper, Python compiler-core logic, new RVV op family, or
      common/core RVV semantic branch is added.

## Non-Goals

- No descriptor-driven route authority, descriptor tables, descriptor-to-C
  exporters, source-export routes, direct C semantic exporters, scalar fallback
  compute, or compatibility/legacy wrappers.
- No new RVV SEW/LMUL/op families, new performance matrices, broad smoke/report
  work, Python compiler core, or new generic lowering framework.
- No new `tcrv.exec` compute semantics.
- No runtime/correctness/performance claim unless emitted object or generated
  package semantics change enough to require fresh `ssh rvv` evidence.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-selected-boundary-conformance-no-synthesis`
- Focused build targets for touched code and tests, especially `tcrv-opt`,
  `tcrv-translate`, and `tianchenrv-target-artifact-export-test`.
- Focused lit/FileCheck coverage under `test/Transforms/RVV` and
  `test/Target/RVV` for positive boundary attrs and missing/stale attr
  negatives.
- Focused C++ coverage in `test/Target/TargetArtifactExportTest.cpp` if needed
  to prove target artifact export validation rejects missing/stale IR-carried
  boundary facts.
- Targeted scan over RVV target/plugin/tests and common adapter surfaces for
  descriptor/direct-C/source-export residue and selected-boundary synthesis.
- `git diff --check`.
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Definition of Done

- RVV selected-boundary conformance facts are materialized into IR before
  export.
- RVV artifact export and EmitC-to-C++ paths validate those facts and fail
  closed when they are missing or stale.
- The common synthesis bypass is deleted if unused.
- The Trellis task status, journal/archive notes, and one coherent commit
  truthfully reflect the completed scope and any remaining rebuild gaps.

## Completion Notes

- `tcrv_rvv.with_vl` now accepts and verifies bounded selected-boundary
  conformance attributes for the RVV i32m1 materialized route:
  `source_kernel`, `selected_variant`, `origin`, `selected_path_role`,
  `status`, `required_capabilities`, `rvv_construction_protocol`, and
  `rvv_emitc_route_mapping`.
- RVV source-front-door materialization writes the selected boundary facts from
  the same kernel, variant, capability, and path-role state consumed by the
  selected materialized EmitC route.
- Hand-authored RVV materialized artifact fixtures now carry the same facts on
  the IR boundary op instead of relying on target/export-time completion.
- The common `synthesizeMissingConformanceAttributes` adapter option and helper
  path were deleted. RVV object/header/bundle and EmitC-to-C++ export now fail
  closed when required selected-boundary facts are missing or stale.
- Added missing/stale selected-boundary negative coverage for C++ target export
  and a focused EmitC-to-C++ FileCheck negative.
- The first full `check-tianchenrv` run exposed an older
  `test/Transforms/LoweringBoundary/rvv-with-vl-selected-boundary.mlir`
  fixture that still used bare `with_vl`; that fixture was repaired to carry
  direct-variant selected-boundary facts and the full check was rerun cleanly.
- No fresh `ssh rvv` proof was produced because this round did not change the
  generated C++ body, object ABI, or runtime package semantics. The prior
  generated bundle ABI proof under
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-source-bundle-ssh-runtime-abi-proof-add`
  remains the current runtime evidence for the unchanged object path.

## Checks Run

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-selected-boundary-conformance-no-synthesis`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-18-rvv-selected-boundary-conformance-no-synthesis`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused lit/FileCheck filter for RVV dialect, source-front-door,
  EmitC-to-C++, source/materialized target artifact exporters, and
  selected-boundary lowering fixtures passed.
- [OK] Targeted scan found no remaining
  `synthesizeMissingConformanceAttributes` or `setMissingBoundaryAttr`
  consumer. Remaining descriptor/direct-C/source-export hits are spec deletion
  rules, fail-closed guards, or FileCheck negative assertions.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 127/127 lit
  tests after self-repair.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous archived PRD read:
  `.trellis/tasks/archive/2026-05/05-18-rvv-construction-template-selected-boundary-handoff/prd.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h`,
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `test/Target/RVV/emitc-to-cpp-handoff.mlir`,
  `test/Target/RVV/emitc-to-cpp-selected-boundary-negative.mlir`,
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.
