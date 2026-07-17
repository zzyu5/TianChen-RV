# Extract RVV binary planning from the monolithic RVV plugin

## Goal

Extract a real plugin-local C++ RVV binary planning component from
`RVVExtensionPlugin.cpp` and make the live RVV plugin consume it for the
bounded binary selected-plan path. The component must own selected family and
vector-shape planning, selected metadata, runtime ABI/route identity, required
intrinsic spelling descriptors, and selected vector-shape metadata validation
for existing i32/i64 add/sub/mul routes without changing runtime evidence
claims.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree is clean on `main` at `0525f37`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes/user brief as
  `05-10-rvv-binary-planning-component-extraction`.
- The previous i64 sub/mul scalar fallback dispatch task is archived and must
  not be reopened.
- Current bounded RVV binary routes cover i32 add/sub/mul and i64 add/sub/mul
  through `RVVBinaryFamilyRegistry`, `RVVBinaryDescriptor`, RVV plugin
  materialization/readiness/plan paths, target microkernel exporters, and
  RVV+scalar dispatch exporters.
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp` still owns too much in one file:
  frontend family lookup, selected vector-shape choice, selected metadata
  emission, descriptor materialization plans, explicit microkernel matching,
  route/runtime ABI identity, and selected-boundary validation helpers.
- This round should remove a meaningful planning slice from that monolith while
  preserving existing artifact semantics and evidence scope.

## Boundaries

- Compiler behavior remains C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only remain runner/probe/artifact tooling; it must not own
  compiler planning, capability legality, descriptors, lowering, or emission.
- `tcrv.exec` remains compute-free. RVV binary details stay in RVV plugin /
  RVV target-owned code, not in core passes.
- The extracted result is not a generic RVV backend, not full MLIR vector
  lowering, and not broader runtime/correctness/performance evidence.
- Existing i32/i64 add/sub/mul emitted RVV C intrinsic spelling must remain
  `riscv_vector.h` RVV C intrinsics, with arithmetic intrinsic prefixes from
  the family registry and selected vector-shape suffixes from selected-shape
  metadata.
- Existing selected vector-shape metadata remains compile-time selected config,
  distinct from hardware facts, runtime SSA/control values, AVL/VL, descriptor
  element counts, correctness, and performance claims.
- No new arithmetic family, dtype, LMUL, tail/mask policy, high-level frontend,
  broad smoke matrix, report-only artifact, or `ssh rvv` runtime claim is in
  scope unless emitted semantics or route IDs actually change.

## Requirements

1. Add one focused plugin-local C++ component, expected path
   `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h` and
   `lib/Plugin/RVV/RVVBinaryPlanning.cpp`, unless code evidence forces a
   better target-owned split.
2. The component must expose a meaningful RVV binary selected-plan abstraction
   consumed by the live RVV plugin, not only by tests.
3. The selected plan must represent the bounded RVV binary family descriptor,
   selected vector-shape config, descriptor-local element count, required
   march/mabi metadata, intrinsic descriptor, route/emission/runtime ABI
   identifiers, and selected metadata entries used by emission plans.
4. The component must own validation of the complete selected vector-shape
   metadata group for both variant attributes and boundary/microkernel
   attributes.
5. Move or delete real duplicated planning logic from
   `RVVExtensionPlugin.cpp`; the plugin should remain an orchestrator that
   calls the component during proposal, legality, selected-boundary
   materialization, readiness, emission planning, and selected-boundary
   validation.
6. Keep existing i32/i64 add/sub/mul artifact semantics and route identities
   stable unless a narrowly justified bug is found.
7. Add direct C++ tests for the planning component covering positive and
   negative selected metadata behavior, including at least one i32 family and
   one i64 family.
8. Add a lit wrapper or integrate the new focused test target into the existing
   test suite so `check-tianchenrv` executes it.
9. Update Trellis task state and journal truthfully. Archive and commit only if
   the module deliverable and focused checks complete.

## Acceptance Criteria

- [x] `RVVExtensionPlugin.cpp` consumes an extracted C++ planning component for
      at least one live proposal/materialization/readiness/emission/selected
      boundary path.
- [x] The component exposes a selected-plan abstraction that carries family,
      selected vector-shape config, required intrinsic/route/runtime ABI
      metadata, descriptor-local element count, required march, optional mabi,
      and selected metadata entries.
- [x] The component validates selected vector-shape metadata as a complete
      group and fails closed for stale/mismatched metadata.
- [x] Direct C++ tests cover positive i32 and i64 selected plans and negative
      incomplete/mismatched selected vector-shape metadata.
- [x] Existing i32/i64 add/sub/mul RVV route IDs, runtime ABI identifiers, and
      generated RVV C intrinsic naming contracts remain stable.
- [x] No extension-specific semantic branch is added to core passes.
- [x] No new RVV runtime/correctness/performance claim is made unless fresh
      `ssh rvv` evidence is collected and recorded.
- [x] `git diff --check` passes.
- [x] Focused build/test targets for the changed RVV plugin/planning component
      pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
      -j2` passes before archive when feasible.
- [x] Trellis task validation passes before archive.

## Completion Evidence

- Added plugin-local RVV binary planning component:
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h` and
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`.
- `RVVBinarySelectedPlan` now owns the bounded binary family descriptor,
  selected vector-shape config, descriptor-local element count, required march,
  optional selected mabi, RVV intrinsic descriptor, emission readiness path,
  route id, runtime ABI id/kind/name, and runtime glue role.
- `RVVExtensionPlugin.cpp` now consumes the extracted component in live RVV
  proposal construction, descriptor-backed i32/i64 materialization planning,
  selected vector-shape metadata emission, emission-plan selected metadata, and
  selected-boundary validation.
- Added direct C++ test `test/Plugin/RVVBinaryPlanningTest.cpp` plus lit
  wrapper `test/Plugin/rvv-binary-planning.test`; the test covers i32-vsub
  i32m2 selected plan, i64-vmul i64m1 selected plan, dtype/shape mismatch,
  incomplete metadata, and stale selected-shape metadata.
- No generated RVV source semantics, route IDs, runtime correctness, or
  performance claim changed; no new `ssh rvv` evidence was produced.
- Final validation passed:
  `git diff --check`;
  `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test -j2`;
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`;
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`;
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`;
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 192/192; and
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-binary-planning-component-extraction`
  passed.
- `clang-format` was attempted but no `clang-format` binary, including common
  versioned names, is installed in this environment; whitespace validation still
  passed via `git diff --check`.

## Minimal Validation Plan

- `git diff --check`
- If needed, configure:
  `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- Build focused changed targets, including `tianchenrv-rvv-extension-plugin-test`
  and the new RVV binary planning test target if added.
- Run the new focused C++ test through its executable or lit wrapper.
- Run `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` if
  dispatch/export planning behavior is touched.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
- Run
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-binary-planning-component-extraction`.

## Out Of Scope

- No new RVV arithmetic family, dtype, shape, LMUL, policy, frontend lowering,
  backend, performance benchmark, broad matrix run, or standalone evidence
  package.
- No compute semantics in `tcrv.exec`.
- No scalar fallback ownership move into RVV plugin or core passes.
- No Python-owned compiler internals.
- No claim that this is a generic RVV backend or full MLIR vector lowering
  route.

## Technical Notes

- Required specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Required archive PRDs read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vadd-scalar-fallback-dispatch-artifact-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-sub-mul-scalar-fallback-dispatch-artifact-ssh-evidence/prd.md`
- Primary implementation surfaces:
  - `include/TianChenRV/Plugin/RVV/RVVExtensionPlugin.h`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVCapabilityProfile.h`
  - `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVLoweringBoundary.h`
  - `lib/Plugin/RVV/RVVLoweringBoundary.cpp`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - relevant tests under `test/Plugin`, `test/Target`, `test/Scripts`, and
    `test/Target/TargetArtifactBundleExport`

## Continuation Rule If Unfinished

Keep this task open and record exactly which component exists, which live RVV
plugin call sites consume it, which call sites still retain planning logic in
the monolith, which checks passed, and the next continuation point. Do not
archive an incomplete extraction.
