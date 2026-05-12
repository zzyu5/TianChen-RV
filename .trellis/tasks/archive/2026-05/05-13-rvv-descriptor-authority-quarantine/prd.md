# RVV descriptor-authority quarantine

## Goal

Quarantine legacy descriptor-oriented RVV binary microkernel emission surfaces
from production artifact routes. Production RVV source/header/object artifacts
must be authorized by selected typed `tcrv_rvv` family/body ops, validated
selected-config metadata, generated EmitC lowerable op-interface provenance,
and the common EmitC source route. Legacy descriptor metadata may remain only
as selected-config mirror or mismatch validation data.

## What I Already Know

* Starting repo state is clean on `main` at `497392a test(rvv): pin typed emitc artifact authority`.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-rvv-typed-family-emitc-artifact-authority/prd.md`
  added a focused regression proving descriptor-only RVV i32 production input
  fails before exact artifact export and before RVV C source emission.
* The current RVV plugin already materializes typed i32/i64 binary
  microkernel bodies and validates structured `setvl -> with_vl ->
  load/load/arithmetic/store` dataflow before source export.
* The current RVV target source path uses `TCRVEmitCLowerableInterface`,
  generated `TCRVEmitCLowerableOpInterface` provenance on compute ops, and
  `lowerTCRVEmitCLowerableToEmitCSource`.
* Remaining debt appears in legacy generic RVV microkernel route/API surfaces:
  `tcrv-export-rvv-microkernel-*` still shares a legacy generic route shape
  for the i32-vadd default, while family-specific vsub/vmul/i64 routes are
  exact.

## Requirements

* Production RVV source/header/object export must not choose computation from
  `tcrv_rvv.lowering_descriptor` or descriptor-only variants.
* Exact target artifact route selection must continue to require typed
  source metadata, EmitC op-interface metadata, runtime ABI metadata, and
  selected vector-shape metadata.
* Any retained `tcrv_rvv.lowering_descriptor` handling must be limited to
  mirror metadata or mismatch validation after typed body authority is known.
* At least one obsolete legacy route/API surface must be deleted, bypassed,
  renamed, or hard-gated so it cannot operate as a broad production authority.
* Default auto-planned RVV artifact routes must still materialize typed
  family/body ops and emit common EmitC provenance.
* Keep changes scoped to RVV plugin/target artifact emission. Do not rewrite
  `tcrv-translate`, common pass orchestration, or unrelated artifact systems.

## Acceptance Criteria

* [x] Descriptor-only RVV production input still fails closed before source,
      header, object, or exact artifact output.
* [x] The legacy generic RVV microkernel route/API surface is no longer a
      broad production fallback; it is exact, harness-only, or otherwise
      explicitly quarantined.
* [x] Auto-planned i32-vadd RVV artifact export still emits typed family/body
      source with common EmitC source-authority and generated op-interface
      provenance.
* [x] Family-specific vsub/vmul and i64 RVV artifact routes continue to reject
      stale route or stale descriptor mirror metadata.
* [x] Source/header/object target artifact exporters still pass focused
      C++ and lit coverage.
* [x] Trellis validation, `git diff --check`, and `git diff --cached --check`
      pass before finish/archive.

## Definition Of Done

* Source changes are in C++/MLIR/TableGen/CMake/lit surfaces only.
* No compiler core, dialect, pass, plugin registry, lowering, or emission
  logic is implemented in Python.
* No new arithmetic family, dtype family, runtime correctness claim,
  performance claim, or standalone `ssh rvv` claim is introduced.
* Stale descriptor-preserving tests are updated or removed if they encode the
  old architecture.
* The task is archived and a coherent commit is created if the round completes.

## Completion Notes

* Removed the broad production RVV microkernel source/header/object API surface
  (`exportRVVMicrokernelC`, `exportRVVMicrokernelHeader`,
  `exportRVVMicrokernelObject`) and replaced it with exact
  `*ForBinaryFamily` entry points.
* Deleted the legacy i32-vadd generic direct-route exception and the now-dead
  i32-only `buildModuleRecordForFamily` helper. Direct RVV artifact export now
  requires an exact manifest-backed typed binary family and calls
  `buildModuleRecordForRVVBinaryFamily`.
* Registered per-family source/header/object exporter function pointers for
  i32-vadd, i32-vsub, i32-vmul, i64-vadd, i64-vsub, and i64-vmul, so the target
  artifact registry cannot fall back to a broad descriptor-shaped exporter.
* Changed RVV scalar dispatch embedded source generation to export through the
  selected dispatch pair's exact RVV binary family.
* Extended the descriptor-only production rejection lit test to cover the
  legacy-named `tcrv-export-rvv-microkernel-c` route as well as the explicit
  i32-vmul route.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|RVVScalarDispatch|target-source-artifact-routes|target-artifact-export-registry'`
  from `build/test`
* `git diff --check`
* `git diff --cached --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-descriptor-authority-quarantine`

## Out Of Scope

* New RVV arithmetic or dtype expansion.
* Broad `tcrv-translate` route rewrite.
* Generic compute semantics in `tcrv.exec`.
* RVV-specific branches in shared/common passes.
* Python compiler internals.
* Broad smoke matrices or standalone RVV hardware evidence.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
* Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-typed-family-emitc-artifact-authority/prd.md`.
* Implementation surfaces inspected:
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/RVVScalarDispatch/`.
