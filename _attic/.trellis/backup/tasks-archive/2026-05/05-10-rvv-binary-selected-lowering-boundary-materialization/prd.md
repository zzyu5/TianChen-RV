# Plugin-local RVV selected lowering-boundary materialization

## Goal

Extract finite RVV binary selected lowering-boundary materialization and
validation from `RVVExtensionPlugin.cpp` into a focused plugin-local C++ module.
The selected `tcrv.exec.variant` plus existing selected emission/binary planning
facts should produce `tcrv_rvv.lowering_boundary` and optional callable RVV
microkernel ops without moving RVV semantics into generic core passes.

## Why Now

The previous archived task
`.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-emission-planning`
completed selected emission planning extraction at commit `fb977ca`. The next
remaining selected-path concentration is lowering-boundary ownership:
`RVVExtensionPlugin.cpp` still owns boundary materialization/validation,
capability summary construction, capacity metadata copying/validation,
duplicate-boundary rejection, runtime ABI mem_window/runtime_param ensuring,
and i32/i64 callable microkernel materialization orchestration.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree for this round was clean at
  `fb977ca refactor(rvv): extract selected emission planning`.
- There was no active `.trellis/.current-task`; this task was created from the
  Hermes brief and must not reopen the archived
  `rvv-binary-selected-emission-planning` task.
- Existing selected-emission ownership already lives in
  `RVVBinarySelectedEmissionPlanning`.
- Existing descriptor-backed microkernel operation construction and IR-backed
  callable ABI parameter collection live in
  `RVVBinaryMicrokernelMaterialization`.
- Finite family scope remains exactly `i32-vadd`, `i32-vsub`, `i32-vmul`,
  `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- No new runtime/correctness/performance claim is planned; if generated runtime
  artifact semantics are unchanged, no fresh `ssh rvv` evidence is required.

## Requirements

- Add a plugin-local C++ module/API named along the lines of
  `RVVBinarySelectedLoweringBoundary` that owns finite RVV binary selected
  lowering-boundary materialization and validation.
- `RVVExtensionPlugin::materializeSelectedLoweringBoundary` and
  `RVVExtensionPlugin::validateSelectedLoweringBoundary` must delegate the
  selected RVV boundary path to the new module.
- The real selected-boundary behavior must move out of
  `RVVExtensionPlugin.cpp`: i32/i64 descriptor branch handling, boundary
  attribute construction, capability summary, boundary capacity metadata
  copying/validation, duplicate boundary rejection, runtime ABI mem_window and
  runtime_param ensuring, and callable microkernel materialization orchestration.
- Preserve public behavior for dispatch-fallback no-boundary handling,
  duplicate boundary rejection, `tcrv_rvv.lowering_boundary` metadata,
  capability summary, selected vector-shape metadata, capacity metadata,
  runtime ABI boundary ensuring, and callable microkernel op materialization.
- Cover both finite selected callable classes already proven by the repo: at
  least one i32 path and one i64 path, including `i64-vmul`.
- Reuse existing selected emission/binary planning facts rather than rebuilding
  a separate descriptor/shape/route decision tree.
- Preserve target-owned descriptor and manifest ownership: route ids, artifact
  kinds, runtime ABI kind/name, runtime glue role, buffer/window specs, and
  element-count parameter specs remain descriptor/helper owned.
- Keep parameter layering explicit: hardware capability facts, selected vector
  shape, runtime SSA/control values, runtime ABI descriptors, descriptor-local
  element count, selected-plan metadata, and boundary-local metadata stay
  distinct.
- Keep smoke-probe and unsupported diagnostic paths stable and clearly
  separated from finite callable binary handling.

## Acceptance Criteria

- [x] A named plugin-local selected lowering-boundary C++ module exists.
- [x] `RVVExtensionPlugin` delegates selected lowering-boundary materialization
      and validation to that module.
- [x] Positive i32 selected callable materialization produces the same boundary,
      selected metadata, ABI mem_windows/params, and microkernel behavior.
- [x] Positive i64 selected callable materialization, including `i64-vmul`,
      produces the same boundary, selected metadata, ABI mem_windows/params,
      and microkernel behavior.
- [x] Duplicate selected RVV boundaries are rejected through the extracted
      module.
- [x] Boundary/variant capacity metadata consistency is validated through the
      extracted module.
- [x] Existing RVV extension plugin behavior remains unchanged.
- [x] The C++/MLIR/TableGen/CMake/lit stack is preserved, `tcrv.exec` remains
      compute-free, RVV semantics remain plugin/target-local, and no new RVV
      runtime/correctness/performance claim is made without real `ssh rvv`
      evidence.

## Non-goals

- No new RVV family, dtype, vector shape, tail/mask policy, generic vector
  dialect route, or generic RVV backend claim.
- No performance benchmarking or speedup claim.
- No Python implementation of compiler IR, dialects, passes, plugin registry,
  capability model, lowering, emission, route selection, or runtime ABI
  decisions.
- No compute semantics in `tcrv.exec`.
- No RVV-specific branches in generic core passes.
- No broad smoke matrix, dashboard/status/report-only work, route-count-only
  cleanup, or evidence-schema-only change as the main result.
- No credential, token, password, connection string, raw log, or secret-like
  metadata in source, tasks, artifacts, or reports.

## Minimal Validation Plan

- `git diff --check`
- Build affected tools/tests at minimum: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and any new selected
  lowering-boundary test binary.
- Run focused C++ tests covering the new selected lowering-boundary module, RVV
  extension plugin behavior, RVV binary planning, and target artifact export
  behavior if touched.
- Run focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-extension-plugin`, `rvv-scalar-dispatch-e2e`, and
  `rvv-scalar-dispatch-bundle-e2e`; include dedicated selected-boundary lit if
  added.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` because public plugin/lit surfaces change.
- Do not run `ssh rvv` unless generated runtime artifacts change or a fresh
  runtime/correctness claim is made.

## Completion Evidence

- Added `RVVBinarySelectedLoweringBoundary` as the plugin-local selected
  lowering-boundary module.
- `RVVExtensionPlugin::materializeSelectedLoweringBoundary` and
  `RVVExtensionPlugin::validateSelectedLoweringBoundary` now delegate selected
  RVV boundary materialization/validation to that module.
- Moved selected boundary capability-summary construction, boundary op
  construction, selected vector-shape metadata copying, boundary capacity
  metadata copying/validation, duplicate-boundary rejection, callable ABI
  mem_window/runtime_param ensuring, and i32/i64 microkernel materialization
  orchestration out of `RVVExtensionPlugin.cpp`.
- Added focused `tianchenrv-rvv-selected-lowering-boundary-test` coverage for
  i32 selected boundary materialization, `i64-vmul` selected boundary
  materialization, selected metadata preservation, ABI mem_window/runtime_param
  ensuring, duplicate-boundary rejection, and ratio-valid
  boundary/variant capacity mismatch validation.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter
  'rvv-(extension-plugin|scalar-dispatch-e2e|scalar-dispatch-bundle-e2e)' .`
  passed 3/3 selected tests.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 194/194 lit tests.
- No generated runtime artifact semantics changed and no new RVV
  runtime/correctness/performance claim was made; no fresh `ssh rvv` run was
  required.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-emission-planning/prd.md`.
