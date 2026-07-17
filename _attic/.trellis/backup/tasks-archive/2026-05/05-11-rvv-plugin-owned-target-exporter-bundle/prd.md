# RVV selected artifact routes through plugin-owned exporter registration

## Goal

Move the existing bounded selected RVV binary microkernel target artifact
exporter group through the generic plugin-owned target exporter bundle boundary
introduced by the previous Toy registration task. The selected RVV front-door
artifact route must continue to consume real selected emission-plan /
`TargetArtifactCandidate` records through `tcrv-translate`, while RVV route
semantics, candidate validation, selected config, runtime ABI layering, and
family checks remain in RVV target-owned C++ code.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round is clean on `main` at
  `9dd83bd feat(plugin): register target exporters via extension pipeline`.
- No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-11-rvv-plugin-owned-target-exporter-bundle`.
- Commit `9dd83bd` added `PluginTargetArtifactExporterBundle` /
  `PluginTargetArtifactExporterRegistry` and moved Toy metadata artifact
  registration behind that generic plugin-owned boundary.
- Current built-in target exporter composition still registers
  `rvv::registerRVVMicrokernelTargetExporters` directly in the non-plugin
  target route set. That is the main registration boundary to migrate.
- The RVV selected binary microkernel exporter group already owns source,
  header, and object routes through `Target/RVV/RVVMicrokernel`, including
  route-local runtime ABI role requirements and candidate preflight callbacks.
- The selected RVV binary family scope stays finite: i32 add/sub/mul and i64
  add/sub/mul microkernel routes. This is not a generic RVV backend.

## Requirements

- Add or extend a narrow RVV target/exporter bundle that registers the existing
  selected RVV binary microkernel source/header/object target artifact
  exporters through `PluginTargetArtifactExporterRegistry`.
- Keep RVV route identity, selected binary family validation, source/header/
  object formatting, intrinsic/source/object details, and route-local candidate
  preflight in `Target/RVV` code.
- Change built-in target artifact exporter composition so enabled
  `rvv-plugin` contributes the selected binary microkernel routes through the
  generic plugin-owned exporter boundary, not through the central non-plugin
  route registration list.
- Preserve unrelated non-plugin route ownership: RVV smoke-probe, scalar,
  offload, RVV+scalar dispatch, and Toy behavior must not be redesigned.
- Preserve parameter layering explicitly: VLEN/vlenb/ELEN remain target
  capability facts; SEW/LMUL/tail/mask/op/dtype remain selected compile-time
  RVV config; AVL/VL remain runtime SSA/control values; descriptor-local
  element counts remain descriptor-local and must not become runtime VL.
- Prove fail-closed behavior for duplicate RVV plugin-owned route registration
  and for missing/disabled `rvv-plugin` registration where the generic API can
  detect it.
- Prove that the selected RVV front-door artifact route still works after the
  migration through `tcrv-opt --tcrv-execution-planning-pipeline` and
  `tcrv-translate --tcrv-export-target-artifact`.
- Keep generic target export, generic plugin registry, generic transforms, and
  `tcrv.exec` free of RVV semantic branches.
- Update durable specs only if current specs do not already cover real
  plugin-owned RVV target exporter bundles.

## Non-Goals

- No generic RVV backend, arbitrary RVV vector lowering, new high-level tensor
  or tile IR, new family expansion, or performance tuning module.
- No RVV semantic branches in generic core passes, generic target export logic,
  or `tcrv.exec`.
- No compute semantics in `tcrv.exec`.
- No Python implementation of compiler internals, plugin registry, target
  artifact registration, lowering, or emission.
- No RVV runtime/correctness/performance claim without real `ssh rvv`
  evidence.
- No docs-only, registration-only, wrapper-only, broad smoke-matrix, or helper
  script closeout.
- No IME, AME, Sophgo/offload, or unrelated scalar migration unless a tiny
  compile fix is required by the RVV migration.

## Acceptance Criteria

- [x] RVV selected binary microkernel source/header/object target exporters
      can be contributed through a `rvv-plugin` plugin-owned target exporter
      bundle.
- [x] Built-in target exporter registration no longer registers the RVV
      selected binary microkernel routes through the central non-plugin route
      list; it gets them from enabled plugin-owned bundles.
- [x] The selected RVV route still appears in the default built-in registry
      because the default front door enables builtin extension plugins.
- [x] A registry populated without enabled `rvv-plugin` does not expose the
      selected RVV binary microkernel routes, while unrelated non-plugin routes
      remain available.
- [x] Duplicate RVV plugin-owned bundle registration and duplicate selected
      RVV route registration fail closed through the generic registry errors.
- [x] Focused C++ coverage proves the RVV plugin-owned exporter bundle,
      duplicate-route behavior, and disabled/missing plugin fail-closed
      behavior where practical.
- [x] Focused lit/FileCheck coverage proves at least one selected RVV artifact
      route reaches `tcrv-translate --tcrv-export-target-artifact` after the
      migration.
- [x] Generic core/export logic avoids RVV semantic branches and `tcrv.exec`
      remains compute-free.
- [x] No RVV runtime/correctness/performance claim is made unless fresh real
      `ssh rvv` evidence is collected.

## Minimal Validation

- `git diff --check`
- Focused build of `tcrv-opt`, `tcrv-translate`, the RVV target/plugin test
  surfaces, and `tianchenrv-target-artifact-export-test`.
- Run focused C++ tests for target artifact export and RVV plugin/exporter
  registration behavior.
- Run focused lit/FileCheck tests for the selected RVV front-door artifact
  route after plugin-owned registration migration.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` after focused checks pass if feasible.
- Run `ssh rvv` evidence only if this round claims emitted RVV runtime
  correctness or runtime behavior. Registration and front-door compiler
  artifact routing alone do not require fresh RVV runtime evidence.
- Validate this Trellis task path before finish/archive.

## Definition of Done

- Source changes are implemented in C++/MLIR/CMake/lit/FileCheck as
  appropriate.
- PRD acceptance criteria and focused validation are satisfied, or the task
  remains open with a precise continuation point.
- Trellis task context and workspace journal truthfully record the outcome.
- The task is finished/archived only after focused validation,
  `check-tianchenrv` if feasible, Trellis validation, and one coherent commit.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-plugin-owned-target-exporter-registration/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-config-vl-boundary-model/prd.md`.
- Key code surfaces:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
