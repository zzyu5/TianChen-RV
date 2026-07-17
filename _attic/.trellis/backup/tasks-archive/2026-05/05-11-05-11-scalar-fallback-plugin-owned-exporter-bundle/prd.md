# Scalar fallback artifact exporters through plugin-owned bundle

## Goal

Move the existing finite scalar fallback source/header/object target artifact
exporters behind the generic plugin-owned target exporter bundle boundary.
The default public front doors must still expose the same scalar artifact routes
when `scalar-plugin` is enabled, but central built-in target exporter
composition must no longer publish scalar microkernel routes directly.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round is clean on `main` at
  `2d5751b chore(trellis): record rvv scalar dispatch ssh evidence`.
- No `.trellis/.current-task` existed at session start.
- The latest supervisor audit/input is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0069-20260510T182130Z/`.
- The prior round recorded real `ssh rvv` evidence for an `i32-vmul`
  RVV+scalar dispatch bundle, but it did not change compiler source code.
- Current `registerBuiltinNonPluginTargetArtifactExporters` still directly
  registers `scalar::registerScalarMicrokernelTargetExporters`.
- Toy, selected RVV microkernel, and RVV+scalar dispatch target exporters have
  already moved behind plugin-owned target exporter bundles.
- Scalar fallback source/header/object route behavior is already finite and
  family-derived for i32/i64 add/sub/mul; this task is a registration-boundary
  migration, not new scalar arithmetic semantics.

## Requirements

- Add a scalar plugin-owned target exporter bundle registration function that
  contributes the existing finite scalar fallback source/header/object target
  artifact routes through `PluginTargetArtifactExporterRegistry`.
- Keep scalar route semantics, source/header/object formatting, object
  compilation, family-derived route ids, runtime ABI metadata, and candidate
  preflight validation inside `Target/Scalar` code.
- Change built-in target artifact exporter composition so enabled
  `scalar-plugin` contributes scalar fallback source/header/object routes
  through the plugin-owned bundle boundary.
- Remove direct scalar microkernel route publication from the central
  non-plugin built-in target exporter list.
- Preserve existing default front-door behavior when built-in extension plugins
  are enabled: scalar source/header/object routes for finite i32/i64 add/sub/mul
  must still be available through `tcrv-translate`.
- A registry populated without enabled `scalar-plugin` must not expose scalar
  fallback microkernel source/header/object routes, while unrelated non-plugin
  routes such as RVV smoke-probe and offload descriptor routes remain available.
- Duplicate scalar plugin-owned bundle registration and duplicate scalar route
  registration must fail closed through existing generic registry errors.
- Keep generic target export, generic plugin registry, generic transforms, and
  `tcrv.exec` free of scalar-family semantic branches.
- Update durable specs only where the scalar exporter ownership boundary
  changes.

## Non-Goals

- No generic scalar backend, arbitrary scalar lowering, linker/runtime
  integration, scalar runtime correctness claim, or performance claim.
- No new arithmetic families beyond the existing finite i32/i64 add/sub/mul
  scalar fallback descriptors.
- No RVV runtime/correctness/performance claim and no required fresh `ssh rvv`
  evidence for this registration-boundary task.
- No movement of scalar semantic route checks into generic core/export registry
  code.
- No compute semantics in `tcrv.exec`.
- No Python implementation of compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or target artifact export.
- No smoke/probe/guardrail/test-harness-only closeout as the main deliverable.
- No offload exporter migration in this task; offload remains a separate
  registration-boundary follow-up.

## Acceptance Criteria

- [x] Scalar fallback source/header/object target exporters are contributed
      through a `scalar-plugin` plugin-owned target exporter bundle.
- [x] Built-in central target exporter composition no longer directly registers
      scalar fallback microkernel routes in the non-plugin route list.
- [x] Default built-in front doors still expose legal scalar fallback routes
      when builtin extension plugins are enabled.
- [x] A built-in registry populated without enabled `scalar-plugin` does not
      expose scalar fallback source/header/object routes.
- [x] Unrelated non-plugin routes, at minimum RVV smoke-probe and offload
      runtime descriptor, remain registered when scalar is missing/disabled.
- [x] Focused C++ coverage proves scalar plugin-owned exporter contribution,
      duplicate bundle/route fail-closed behavior, and missing/disabled
      scalar-plugin behavior.
- [x] Focused lit/FileCheck coverage proves at least one scalar non-add
      source/header/object route still reaches `tcrv-translate` through the
      selected-plan/front-door path after the registration migration.
- [x] Generic core/export logic avoids scalar semantic branches and
      `tcrv.exec` remains compute-free.
- [x] No RVV or scalar runtime/correctness/performance claim is made unless
      fresh real runtime evidence is collected and recorded.

## Minimal Validation

- `git diff --check`
- Build focused touched targets:
  `tcrv-opt`, `tcrv-translate`, and `tianchenrv-target-artifact-export-test`.
- Run `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck tests for scalar source/header/object target
  artifact export, including a non-add route.
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
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  and `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Latest audit/input read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0069-20260510T182130Z/repo_audit.md`
  and
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0069-20260510T182130Z/review_input.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-plugin-owned-target-exporter-bundle/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-scalar-dispatch-plugin-owned-exporter-bundle/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-11-05-11-scalar-fallback-header-object-artifact-routes/prd.md`.
- Key implementation surfaces:
  `include/TianChenRV/Target/Scalar/ScalarMicrokernel.h`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  and `test/Target/TargetArtifactExportTest.cpp`.

## Result

- `PluginTargetArtifactExporterRegistry` now supports multiple bundles owned by
  the same extension plugin. Enabled-plugin population registers each bundle
  whose declared dependencies are enabled, so scalar standalone routes can be
  published without forcing RVV-dependent RVV+scalar dispatch routes to publish.
- Added
  `scalar::registerScalarMicrokernelPluginTargetExporterBundle`, which
  contributes the existing scalar source/header/object route group through the
  `scalar-plugin` bundle while leaving scalar route semantics and preflight in
  `Target/Scalar`.
- Removed direct scalar route publication from
  `registerBuiltinNonPluginTargetArtifactExporters`; default built-in route
  registration now receives scalar fallback routes through the active plugin
  bundle.
- Added focused C++ coverage for scalar plugin-owned route contribution,
  duplicate bundle/route failure, missing/disabled `scalar-plugin`, and
  built-in registration without scalar.
- Updated scalar/lowering-runtime specs to record scalar source/header/object
  exporter ownership and the multiple-bundle-per-plugin dependency boundary.
- No runtime, correctness, throughput, latency, ratio, or performance claim was
  made, and no fresh `ssh rvv` evidence was required for this registration
  boundary task.
