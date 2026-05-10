# Offload descriptor exporter through plugin-owned bundle

## Goal

Move the existing runtime-offload descriptor target artifact exporter behind
the generic plugin-owned target exporter bundle boundary. The default public
front doors must still expose the same offload descriptor route when
`offload-plugin` is enabled, but central built-in target exporter composition
must no longer directly publish the offload descriptor route as a non-plugin
route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round is clean on `main` at
  `663002b feat(scalar): register exporters via plugin bundle`.
- No `.trellis/.current-task` existed at session start.
- The latest supervisor audit/input is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0070-20260510T184232Z/`.
- The latest review tail explicitly named offload runtime descriptor route
  migration as a natural follow-up, not as an unfinished blocker from the prior
  scalar task.
- Current `registerBuiltinNonPluginTargetArtifactExporters` still directly
  registers `offload::registerOffloadRuntimeDescriptorTargetExporters`.
- RVV selected microkernel, Toy metadata, scalar fallback standalone routes,
  and RVV+scalar dispatch routes are already contributed through
  plugin-owned target exporter bundles.
- The offload descriptor exporter body, candidate preflight, runtime ABI role
  contract, and descriptor text generation already live in `Target/Offload`.
  This task is a registration-boundary migration, not new offload runtime
  execution semantics.

## Requirements

- Add an offload plugin-owned target exporter bundle registration function that
  contributes the existing runtime-offload descriptor route through
  `PluginTargetArtifactExporterRegistry`.
- Keep route semantics, descriptor formatting, selected-plan metadata
  validation, runtime ABI role contract, and candidate preflight validation
  inside `Target/Offload` code.
- Change built-in target artifact exporter composition so enabled
  `offload-plugin` contributes the descriptor route through the plugin-owned
  bundle boundary.
- Remove direct offload descriptor route publication from the central
  non-plugin built-in target exporter list.
- Preserve existing default front-door behavior when built-in extension
  plugins are enabled: legal selected offload descriptor paths must still be
  exportable through `tcrv-translate`.
- A registry populated without enabled `offload-plugin` must not expose the
  `tcrv-export-offload-runtime-descriptor` route, while unrelated non-plugin
  routes such as RVV smoke-probe remain available.
- Duplicate offload plugin-owned bundle registration and duplicate offload
  route registration must fail closed through existing generic registry
  errors.
- Keep generic target export, generic plugin registry, generic transforms, and
  `tcrv.exec` free of offload/vendor semantic branches.
- Update durable specs only where the offload exporter ownership boundary
  changes.

## Non-Goals

- No vendor runtime integration, DMA, accelerator kernel emission, object
  generation, linked runtime glue, correctness claim, or performance claim.
- No new offload dialect operation, offload capability schema, or runtime ABI
  descriptor fields beyond existing bounded descriptor behavior.
- No RVV runtime/correctness/performance claim and no required fresh `ssh rvv`
  evidence for this registration-boundary task.
- No movement of offload descriptor policy into generic core/export registry
  code.
- No compute semantics in `tcrv.exec`.
- No Python implementation of compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or target artifact export.
- No smoke/probe/guardrail/test-harness-only closeout as the main deliverable.

## Acceptance Criteria

- [x] Runtime-offload descriptor target exporter is contributed through an
      `offload-plugin` plugin-owned target exporter bundle.
- [x] Built-in central target exporter composition no longer directly registers
      the offload descriptor route in the non-plugin route list.
- [x] Default built-in front doors still expose the legal offload descriptor
      route when builtin extension plugins are enabled.
- [x] A built-in registry populated without enabled `offload-plugin` does not
      expose the offload descriptor route.
- [x] Unrelated non-plugin routes, at minimum RVV smoke-probe, remain registered
      when offload is missing/disabled.
- [x] Focused C++ coverage proves offload plugin-owned exporter contribution,
      duplicate bundle/route fail-closed behavior, and missing/disabled
      `offload-plugin` behavior.
- [x] Focused lit/FileCheck coverage proves an existing legal offload
      descriptor front-door route still reaches `tcrv-translate` after the
      registration migration.
- [x] Generic core/export logic avoids offload/vendor semantic branches and
      `tcrv.exec` remains compute-free.
- [x] No runtime/correctness/performance claim is made.

## Minimal Validation

- `git diff --check`
- Build focused touched targets:
  `tcrv-opt`, `tcrv-translate`, and `tianchenrv-target-artifact-export-test`.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck tests for offload descriptor target artifact
  export, including the pipeline/front-door route if available.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  after focused checks pass if feasible.
- Validate this Trellis task path before finish/archive.

## Definition of Done

- Source changes are implemented in C++/MLIR/CMake/lit/FileCheck as
  appropriate.
- PRD acceptance criteria and minimal validation pass, or the task remains open
  with a precise failing check and continuation point.
- Trellis task context and workspace journal truthfully record the outcome.
- The task is finished/archived only after focused validation,
  `check-tianchenrv` if feasible, Trellis validation, and one coherent commit.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/offload-runtime-plugin.md`,
  and `.trellis/spec/testing/mlir-testing-contract.md`.
- Latest audit/input read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0070-20260510T184232Z/repo_audit.md`
  and
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0070-20260510T184232Z/review_input.md`.
- Prior scalar exporter migration PRD read:
  `.trellis/tasks/archive/2026-05/05-11-05-11-scalar-fallback-plugin-owned-exporter-bundle/prd.md`.
- Key implementation surfaces:
  `include/TianChenRV/Target/Offload/OffloadRuntimeDescriptor.h`,
  `lib/Target/Offload/OffloadRuntimeDescriptor.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  and `test/Target/TargetArtifactExportTest.cpp`.

## Result

- Added
  `offload::registerOffloadRuntimeDescriptorPluginTargetExporterBundle`, which
  contributes the existing runtime-offload descriptor route through the
  `offload-plugin` plugin-owned target exporter bundle while preserving
  descriptor generation and preflight in `Target/Offload`.
- Removed direct offload descriptor route publication from
  `registerBuiltinNonPluginTargetArtifactExporters`; default built-in route
  registration now receives the descriptor through the active plugin bundle.
- Added focused C++ coverage for offload plugin-owned route contribution,
  duplicate bundle/route failure, missing/disabled `offload-plugin`, and
  built-in registration without offload.
- Updated offload/lowering-runtime specs to record the offload descriptor
  exporter ownership boundary.
- No runtime execution, correctness, throughput, latency, ratio, or
  performance claim was made, and no fresh `ssh rvv` evidence was required for
  this registration-boundary task.
