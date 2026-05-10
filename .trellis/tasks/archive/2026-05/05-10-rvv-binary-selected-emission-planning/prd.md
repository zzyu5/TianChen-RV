# Plugin-local RVV binary selected emission planning

## Goal

Extract finite RVV binary selected-emission readiness and emission-plan
construction from `RVVExtensionPlugin.cpp` into an explicit plugin-local C++
planning API. `RVVExtensionPlugin` should delegate the selected direct/callable
finite binary path to that planner while preserving all public plugin behavior,
diagnostics, route ids, artifact kinds, runtime ABI metadata, selected vector
metadata, capacity metadata, and target-owned manifest ownership.

## Why Now

The previous archived task
`.trellis/tasks/archive/2026-05/05-10-rvv-binary-proposal-legality-planning`
completed proposal planning extraction into `RVVBinaryPlanning`. The next
remaining concentration in the same modularization path is selected emission:
`RVVExtensionPlugin.cpp` still owns the finite binary selected-path branch
logic for i32 and i64 microkernels and constructs `VariantEmissionStatus` /
`VariantEmissionPlan` inline.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree for this round was clean at
  `6758862 refactor(rvv): extract binary proposal planning`.
- There was no active `.trellis/.current-task`; this task was created from the
  Hermes brief and started as the current task.
- The prior task
  `rvv-binary-proposal-legality-planning` is finished and archived and must not
  be reopened.
- Existing selected-emission responsibility in `RVVExtensionPlugin.cpp`
  includes `findMatchingExplicitMicrokernelFamily`,
  `findMatchingExplicitI64MicrokernelDescriptor`,
  `checkVariantEmissionReadiness`, `buildVariantEmissionPlan`,
  runtime ABI parameter attachment, selected vector-shape metadata propagation,
  and selected capacity metadata propagation.
- `RVVBinaryPlanning` already exposes descriptor-backed selected/proposal facts
  and target-owned finite family identity: route id, artifact kind, runtime ABI
  kind/name, runtime glue role, intrinsic spelling, and supported message.
- `RVVBinaryMicrokernelMaterialization` already owns descriptor-derived
  materialization plans and IR-backed callable runtime ABI parameter collection.
- Finite family scope remains exactly `i32-vadd`, `i32-vsub`, `i32-vmul`,
  `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- No new runtime/correctness/performance claim is planned for this round; if no
  generated runtime artifact semantics change, no fresh `ssh rvv` run is
  required.

## Requirements

- Add or extend a plugin-local C++ API, preferably in `RVVBinaryPlanning` or a
  closely named `Plugin/RVV` module, that builds structured selected-emission
  planning data for finite RVV binary selected variants from
  `VariantEmissionRequest` and existing typed inputs.
- The planner must cover both representative finite classes already present:
  an i32 family path and an i64 family path, including `i64-vmul`.
- The planner must keep typed facts distinguishable: selected family,
  descriptor, selected vector shape, runtime ABI parameters, required
  capability symbols, route/artifact identity, selected vector metadata, and
  optional capacity metadata.
- `RVVExtensionPlugin::checkVariantEmissionReadiness` and
  `RVVExtensionPlugin::buildVariantEmissionPlan` must delegate the finite
  binary direct/callable selected path to the new planner instead of locally
  owning the i32/i64 branch and plan-construction logic.
- Do not leave a wrapper where the real finite binary emission decisions still
  live in `RVVExtensionPlugin.cpp`.
- Preserve target-owned manifest ownership: route ids, artifact kinds, runtime
  ABI names, runtime glue roles, self-check/source artifact facts, and
  descriptor-derived ABI specs must continue to come from target descriptors or
  existing target-owned helpers.
- Preserve parameter layering: hardware/profile capability facts,
  compile-time selected vector shape, runtime SSA/control values, runtime ABI
  descriptors, descriptor-local element count, artifact route properties, and
  selected-plan metadata must remain separate.
- Keep smoke-probe and unsupported diagnostic paths stable and separate from
  the extracted finite binary planner.
- Add focused C++ tests for the selected-emission planner API and update
  extension plugin tests as needed to prove public behavior is unchanged.

## Acceptance Criteria

- [x] A named plugin-local selected-emission planner API exists for finite RVV
      binary selected paths.
- [x] The planner can return no-match, supported i32 selected emission, and
      supported i64 selected emission without plugin-file branch ownership.
- [x] The planner builds or exposes `VariantEmissionStatus` /
      `VariantEmissionPlan` facts with descriptor-owned route id, artifact
      kind, runtime ABI, runtime ABI kind/name, runtime glue role, callable
      runtime ABI parameters, required capability symbols, selected vector
      metadata, and capacity metadata.
- [x] `RVVExtensionPlugin::checkVariantEmissionReadiness` delegates finite
      binary selected readiness to the planner.
- [x] `RVVExtensionPlugin::buildVariantEmissionPlan` delegates finite binary
      selected plan construction to the planner.
- [x] Focused C++ tests cover at least one i32 selected-emission planner path
      and the `i64-vmul` selected-emission planner path.
- [x] Existing RVV extension plugin public tests continue to prove behavior is
      stable.
- [x] No new RVV runtime/correctness/performance claim is made unless backed by
      fresh real `ssh rvv` evidence.

## Non-goals

- No new RVV family, dtype, vector shape, mask/tail policy, generic vector
  dialect route, or generic RVV backend claim.
- No performance benchmark or speedup claim.
- No Python implementation of compiler IR, dialects, passes, plugin registry,
  capability model, lowering, emission planning, route selection, or runtime
  ABI decisions.
- No compute semantics in `tcrv.exec`.
- No RVV-specific branches in generic core passes.
- No broad smoke matrix, dashboard/status/report-only work, route-count-only
  cleanup, or evidence-schema-only change as the main result.
- No rewrite of target manifests or exporter route ownership unless a concrete
  duplicate table is found and removed.
- No credentials, tokens, passwords, or connection strings in source, tasks,
  artifacts, or reports.

## Minimal Validation Plan

- `git diff --check`
- Build affected tools/tests at minimum: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and finite family/runtime ABI tests
  if touched.
- Run focused C++ binaries covering RVV binary planning, RVV extension plugin
  behavior, target artifact export, and finite family/runtime ABI behavior if
  touched.
- Run focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-binary-planning`, `rvv-extension-plugin`,
  `rvv-scalar-dispatch-e2e`, and `rvv-scalar-dispatch-bundle-e2e`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  because this round changes public plugin planning behavior.
- Do not run Python runner checks unless runner files change.
- Do not run `ssh rvv` unless generated runtime artifacts change or a fresh
  runtime/correctness claim is made.

## Completion Evidence

- Added `RVVBinarySelectedEmissionPlanning` as the plugin-local selected
  emission planner.
- `RVVExtensionPlugin::checkVariantEmissionReadiness` and
  `RVVExtensionPlugin::buildVariantEmissionPlan` now delegate finite RVV binary
  selected paths to that planner.
- Focused C++ coverage was added in `RVVExtensionPluginTest.cpp` for an i32
  selected-emission planner path and all i64 binary family helpers, including
  `i64-vmul`.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test
  tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter
  'rvv-(binary-planning|extension-plugin|scalar-dispatch-e2e|scalar-dispatch-bundle-e2e)' .`
  passed 4/4 selected tests.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 194/194 lit tests.
- No generated runtime artifact semantics changed and no new RVV
  runtime/correctness/performance claim was made; no fresh `ssh rvv` run was
  required.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-proposal-legality-planning/prd.md`
  and `.trellis/workspace/codex/journal-2.md`.
- Primary code surfaces inspected:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `test/Plugin/RVVBinaryPlanningTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.
