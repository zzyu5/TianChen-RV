# RVV+scalar dispatch artifact routes through plugin-owned exporter registration

## Goal

Move the existing bounded RVV+scalar dispatch source/header/object target
artifact routes through the plugin-owned target exporter bundle boundary. The
generic target artifact front door must consume active plugin-owned exporter
bundles and still produce a coherent dispatch bundle with selected dispatch
surface metadata, component roles, runtime ABI parameter roles, selected RVV
vector-shape metadata, and scalar family-derived route matching.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round is clean on `main` at
  `2f58e67 feat(scalar): add finite header object artifact routes`.
- No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-11-rvv-scalar-dispatch-plugin-owned-exporter-bundle`.
- Commit `fea02ff` moved selected RVV binary microkernel source/header/object
  routes behind the `rvv-plugin` plugin-owned exporter bundle.
- Commit `2f58e67` completed scalar finite source/header/object route parity
  for i32/i64 add/sub/mul, including family-derived non-add route ids.
- Current specs require disabled or missing plugins to fail closed rather than
  publishing plugin-owned target routes from central built-in code.
- Current dispatch artifacts are target-owned RVV+scalar composite exports and
  must preserve component roles plus runtime ABI roles for lhs, rhs, out,
  runtime element count, and the dispatch availability guard.

## Requirements

- Add or extend a narrow RVV+scalar dispatch target/exporter bundle that
  registers the existing dispatch source/header/object target artifact routes
  through `PluginTargetArtifactExporterRegistry`.
- Change built-in target artifact exporter composition so enabled relevant
  plugin-owned bundle registration contributes the RVV+scalar dispatch route
  group; central built-in code must not directly publish these
  extension-specific dispatch routes.
- Preserve target-owned dispatch route semantics in RVVScalarDispatch target
  code: selected surface `dispatch`, dispatch case role, scalar fallback role,
  component source/header/object roles, external ABI identity, runtime ABI
  parameter role ordering, route-local candidate preflight, object compilation
  behavior, and bundle index component-group validation.
- Use `i32-vmul` as the main non-legacy acceptance family. Add only a small
  `i32-vadd` legacy compatibility assertion if needed to prove existing route
  IDs remain usable.
- Preserve selected RVV vector-shape/config metadata where already modeled and
  preserve scalar fallback family-derived route matching from the previous
  scalar module.
- Disabled or missing relevant plugins must fail closed as unknown or
  unavailable RVV+scalar dispatch artifact routes. They must not silently
  publish dispatch routes from central built-in code.
- Keep generic target export, generic plugin registry, execution-plan
  coherence, and `tcrv.exec` free of RVV/scalar semantic branches.
- Update durable specs only where this task changes the exporter bundle
  contract.

## Non-Goals

- No broad family smoke matrix.
- No standalone evidence packaging as the main result.
- No RVV runtime, correctness, throughput, latency, or performance claim
  without fresh real `ssh rvv` evidence.
- No movement of RVV or scalar semantic route checks into generic core/export
  registry code.
- No compute semantics in `tcrv.exec`.
- No Python implementation of compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or target artifact export.
- No replacement of existing finite selected-boundary/exporter logic with a
  generic unsupported backend claim.

## Acceptance Criteria

- [x] RVV+scalar dispatch source/header/object target exporters are contributed
      through active plugin-owned target exporter bundles.
- [x] Built-in central target exporter composition no longer directly registers
      the RVV+scalar dispatch route group.
- [x] Default built-in front doors still expose legal RVV+scalar dispatch
      routes when the relevant builtin extension plugins are enabled.
- [x] A registry populated without the relevant enabled plugin bundle does not
      expose the RVV+scalar dispatch source/header/object routes.
- [x] Focused C++ tests prove plugin-owned dispatch route contribution,
      duplicate/fail-closed behavior, and disabled/missing plugin behavior.
- [x] A focused lit/FileCheck target artifact bundle test proves an `i32-vmul`
      dispatch bundle through `tcrv-translate`, checking index contents,
      source/header/object emission, component roles, runtime ABI parameter
      roles, selected surface, route ids, selected RVV config metadata, and no
      unrelated family route leakage.
- [x] Generic core/export logic avoids extension-specific semantic branches and
      `tcrv.exec` remains compute-free.
- [x] No RVV runtime/correctness/performance claim is made unless fresh real
      `ssh rvv` evidence is collected and recorded.

## Minimal Validation

- `git diff --check`
- Build focused touched targets, including `tcrv-opt`, `tcrv-translate`, and
  `tianchenrv-target-artifact-export-test`.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck coverage for RVV+scalar dispatch bundle export,
  including the new non-legacy `i32-vmul` acceptance path and fail-closed route
  publication checks where available.
- Run a focused dry-run script/lit path only if it validates the same
  front-door dispatch bundle behavior.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  after focused checks pass.
- Validate this Trellis task path before finish/archive.

## Definition of Done

- Source changes are implemented in C++/MLIR/CMake/lit/FileCheck as
  appropriate.
- PRD acceptance criteria and focused validation pass, or the task remains open
  with a precise failing check and continuation point.
- Trellis task context and workspace journal truthfully record the outcome.
- The task is finished/archived only after focused validation,
  `check-tianchenrv`, Trellis validation, and one coherent commit.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-plugin-owned-target-exporter-bundle/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-05-11-scalar-fallback-header-object-artifact-routes/prd.md`.
- Key implementation surfaces to inspect before code changes:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  RVV, scalar, and RVVScalarDispatch target/plugin files under
  `include/TianChenRV` and `lib`, `test/Target/RVVScalarDispatch/`,
  `test/Target/TargetArtifactBundleExport/`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`, and
  `scripts/rvv_scalar_dispatch_e2e.py`.
