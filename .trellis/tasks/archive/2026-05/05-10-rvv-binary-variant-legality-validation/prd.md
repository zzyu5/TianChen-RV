# Plugin-local RVV binary variant legality validation

## Goal

Extract finite RVV binary variant legality and shared selected-variant metadata
validation from `RVVExtensionPlugin.cpp` into a focused plugin-local C++
module. `RVVExtensionPlugin` should keep the public plugin interface and
delegate legality checks to the module while selected emission planning,
selected lowering-boundary materialization, smoke-probe diagnostics, and
unsupported metadata-only paths preserve current behavior.

## Why Now

The previous archived task
`.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-lowering-boundary-materialization`
completed selected lowering-boundary ownership at commit `cce4182`. The next
remaining RVV selected-path concentration in `RVVExtensionPlugin.cpp` is
variant legality: origin/capability ownership, required finite vector-shape
requires metadata, typed RVV policy validation, selected vector-shape metadata
validation, required march validation, smoke-probe descriptor validation,
capacity metadata validation, and finite i32/i64 microkernel selected-plan
validation.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree for this round was clean at
  `cce4182 refactor(rvv): extract selected lowering boundary`.
- There is no active `.trellis/.current-task`; this task was created from the
  Hermes malformed-review fallback after reading the latest
  `repo_audit.md` and `review_input.md`.
- Existing selected emission ownership lives in
  `RVVBinarySelectedEmissionPlanning`.
- Existing selected lowering-boundary ownership lives in
  `RVVBinarySelectedLoweringBoundary`.
- Existing descriptor-backed microkernel operation construction lives in
  `RVVBinaryMicrokernelMaterialization`.
- Finite family scope remains exactly `i32-vadd`, `i32-vsub`, `i32-vmul`,
  `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- This task is a C++/MLIR plugin-local ownership extraction; it should not add
  a new RVV runtime, correctness, or performance claim.

## Requirements

- Add a plugin-local C++ module/API named along the lines of
  `RVVBinaryVariantLegality` that owns finite RVV binary variant legality
  validation.
- `RVVExtensionPlugin::verifyVariantLegality` must delegate to the new module.
- Move the real legality behavior out of `RVVExtensionPlugin.cpp`: origin
  validation, available `rvv` capability validation, `requires` capability
  validation, required finite vector-shape resolution, descriptor dtype/shape
  consistency, typed `tcrv_rvv.policy` validation, selected vector-shape
  metadata validation, required march validation, smoke-probe descriptor
  validation, capacity metadata validation, and selected i32/i64 microkernel
  plan validation.
- Keep smoke-probe emission readiness/plan diagnostics behavior stable while
  avoiding a private legality-helper island in `RVVExtensionPlugin.cpp`.
- Preserve existing selected emission planning and selected lowering-boundary
  APIs; they may call through the plugin legality delegate unchanged.
- Preserve target/plugin-owned descriptor boundaries: route ids, runtime ABI
  names/kinds, runtime glue roles, artifact kinds, buffer specs, and element
  count specs remain descriptor/helper owned.
- Keep parameter layering explicit: hardware capability facts, selected vector
  shape, runtime SSA/control values, runtime ABI descriptors,
  descriptor-local element count, selected-plan metadata, and boundary-local
  metadata remain distinct.
- Do not add RVV-specific behavior to generic core passes.

## Acceptance Criteria

- [x] A named plugin-local RVV binary variant legality C++ module exists.
- [x] `RVVExtensionPlugin::verifyVariantLegality` delegates to that module.
- [x] Positive finite i32 selected variants validate through the extracted
      module.
- [x] Positive finite i64 selected variants, including `i64-vmul`, validate
      through the extracted module.
- [x] Existing malformed legality diagnostics remain covered for missing
      origin/capability, malformed selected vector-shape metadata, malformed
      required march, smoke-probe descriptor misuse, capacity metadata
      mismatch, and invalid descriptor/shape pairing where applicable.
- [x] Smoke-probe readiness/plan behavior remains unchanged and does not turn
      into a supported runtime/correctness/performance claim.
- [x] Existing selected emission and selected lowering-boundary tests keep
      passing.
- [x] The C++/MLIR/TableGen/CMake/lit stack is preserved, `tcrv.exec` remains
      compute-free, RVV semantics remain plugin/target-local, and no new RVV
      runtime/correctness/performance claim is made without real `ssh rvv`
      evidence.

## Completion Evidence

- Added `RVVBinaryVariantLegality` as the plugin-local C++ module for finite
  RVV binary variant legality validation.
- `RVVExtensionPlugin::verifyVariantLegality` now delegates to
  `verifyRVVBinaryVariantLegality`.
- Moved RVV variant legality checks out of `RVVExtensionPlugin.cpp`: origin
  validation, available `rvv` capability validation, `requires` capability
  validation, required finite vector-shape resolution, descriptor dtype/shape
  consistency, typed policy validation, selected vector-shape metadata
  validation, required march validation, smoke-probe descriptor validation,
  capacity metadata validation, and selected i32/i64 microkernel selected-plan
  validation.
- Smoke-probe readiness and plan construction now reuse
  `verifyRVVBinarySmokeProbeVariantMetadata` instead of private helper logic in
  `RVVExtensionPlugin.cpp`.
- Added `tianchenrv-rvv-binary-variant-legality-test` with direct module
  coverage for finite i32 selected legality, finite `i64-vmul` selected
  legality, smoke-probe metadata validation, origin mismatch rejection, and
  selected vector-shape metadata mismatch rejection.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-binary-variant-legality-test
  tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter
  'rvv-(extension-plugin|scalar-dispatch-e2e|scalar-dispatch-bundle-e2e)' .`
  passed 3/3 selected tests.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 194/194 lit tests.
- No generated runtime artifact semantics changed and no new RVV
  runtime/correctness/performance claim was made; no fresh `ssh rvv` run was
  required.

## Non-goals

- No new RVV family, dtype, vector shape, tail/mask policy, descriptor, route,
  runtime ABI, artifact kind, or backend claim.
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
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`, and any new focused
  variant-legality test binary.
- Run focused C++ tests covering the extracted legality module, RVV extension
  plugin behavior, RVV binary planning, and selected lowering-boundary behavior.
- Run focused lit from `artifacts/tmp/tianchenrv-build/test` covering
  `rvv-extension-plugin`, `rvv-scalar-dispatch-e2e`, and
  `rvv-scalar-dispatch-bundle-e2e`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if public plugin/lit surfaces changed.
- Do not run `ssh rvv` unless generated runtime artifacts change or a fresh
  runtime/correctness claim is made.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-lowering-boundary-materialization/prd.md`.
- Current evidence read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0059-20260510T141537Z/repo_audit.md`,
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0059-20260510T141537Z/review_input.md`,
  `.trellis/workspace/codex/journal-2.md`.
