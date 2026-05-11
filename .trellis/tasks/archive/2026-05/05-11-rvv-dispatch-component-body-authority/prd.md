# RVV+scalar dispatch component body authority

## Goal

Make the composite RVV+scalar dispatch artifact exporter fail closed unless the
selected dispatch case, selected scalar fallback, callable artifact candidates,
and typed executable component bodies all describe the same bounded binary
family and ABI surface. The first completed slice is frontend-selected
`i32-vmul`.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state was clean on `main` at
  `cd7e4fd test(rvv): enforce structured body export authority`.
- No `.trellis/.current-task` existed before this task was created.
- The previous task
  `.trellis/tasks/archive/2026-05/05-11-rvv-structured-body-export-authority/`
  is completed context, not an active task.
- That previous task closed direct RVV microkernel body authority in
  `lib/Target/RVV/RVVMicrokernel.cpp`: descriptor family and typed
  `tcrv_rvv.*_microkernel` body family must agree before direct RVV source,
  header, or object export.
- The remaining module boundary is the composite dispatcher in
  `lib/Target/Builtin/RVVScalarDispatch.cpp`, which embeds selected RVV and
  scalar callable artifacts and then emits source/header/object/self-check
  dispatch artifacts.
- The exporter already validates candidate route metadata, selected dispatch
  case linkage, scalar fallback target linkage, IR-backed callable ABI
  parameters, runtime element-count, and dispatch availability guard parameter
  layering.
- The risk for this task is stale component truth: dispatch metadata may still
  claim `i32-vmul` while an embedded RVV or scalar component body/candidate has
  drifted to `i32-vadd` or `i32-vsub`.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only run existing tooling, probes, artifact parsing, and support
  scripts. It must not implement compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or route selection.
- Preserve TianChen-RV boundaries: `tcrv.exec` stays compute-free; RVV and
  scalar executable behavior remains extension/target-owned.
- The composite RVV+scalar dispatch exporter must fail before emitting
  dispatch source, header, object, self-check source, or self-check object when
  selected dispatch case/fallback metadata disagree with typed component body
  authority.
- The RVV component must agree across selected dispatch case, callable artifact
  candidate, direct RVV exporter record/body authority, route id, role, origin,
  lowering descriptor family, dtype/arithmetic family, selected SEW/LMUL/policy
  and vector shape metadata, runtime AVL/VL layering metadata, runtime ABI
  kind/name/glue role, and artifact kind.
- The scalar fallback component must agree across selected fallback target,
  callable artifact candidate, scalar exporter record/body authority, route id,
  role, origin, lowering descriptor family, runtime ABI kind/name/glue role,
  artifact kind, and selected fallback variant.
- The composite exporter must derive component identity from validated
  target/exporter records or verified callable candidates, not from duplicated
  dispatch-only string assumptions.
- Add fail-closed coverage where dispatch metadata still says `i32-vmul` but
  either the RVV component body/candidate or the scalar fallback body/candidate
  is stale `i32-vadd` or `i32-vsub`.
- Add positive coverage proving valid `i32-vmul` dispatch source still emits
  the RVV case call, scalar fallback call, runtime guard parameter, runtime
  element-count parameter, and self-check marker.
- Preserve parameter layering: VLEN/vlenb/ELEN are target capability facts;
  SEW/LMUL/tail/mask/vector type/setvl suffix are selected compile-time RVV
  config; runtime AVL/VL/availability guard are ABI-visible runtime/control
  values; `element_count` remains descriptor-local metadata.

## Acceptance Criteria

- A valid frontend-selected `i32-vmul` RVV+scalar dispatch source export still
  emits the expected dispatcher with RVV branch, scalar branch, runtime
  element-count parameter, and dispatch availability guard parameter.
- A valid `i32-vmul` self-check dispatch source export still contains the
  `tcrv_rvv_scalar_i32_vmul_dispatch_self_check_ok` marker and exercises both
  scalar and RVV guard branches.
- Dispatch export rejects an `i32-vmul` dispatch pair whose RVV callable
  candidate or typed RVV body resolves to a stale add/sub family before source
  bytes are emitted.
- Dispatch export rejects an `i32-vmul` dispatch pair whose scalar fallback
  callable candidate or typed scalar body resolves to a stale add/sub family
  before source bytes are emitted.
- Header, object, and self-check helper paths reuse the same component
  authority validation rather than bypassing it.
- Focused C++ target/export tests cover at least one positive `i32-vmul`
  dispatch path and at least two stale-component fail-closed paths.
- Focused lit/FileCheck coverage for affected dispatch source/bundle routes
  still passes for the bounded `i32-vmul` route.
- `git diff --check` passes.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  is run before final report if the local build remains available.
- Fresh `ssh rvv` is only run if the runtime artifact behavior changes and the
  final report makes a runtime correctness claim. If run, the claim is bounded
  to this single `i32-vmul` dispatch slice.

## Non-Goals

- Do not implement a generic RVV backend or broad family matrix.
- Do not add new RVV, IME, offload, hardware, performance, or runtime maturity
  claims.
- Do not move compiler/exporter internals into Python.
- Do not add compute semantics to `tcrv.exec`.
- Do not hard-code new extension-specific branches in core passes.
- Do not make prompt/report/helper/status/smoke-only work the main result.
- Do not expand to unrelated i64 behavior unless a small existing generic
  helper naturally stays correct while completing `i32-vmul`.
- Do not hand-author final artifact truth in tests while bypassing
  compiler/exporter output.

## Validation Plan

- Build focused targets:
  `tianchenrv-target-artifact-export-test`, relevant RVV/scalar target tests if
  available, `tcrv-opt`, and `tcrv-translate`.
- Run focused C++ target/export tests covering the new dispatch component
  authority checks.
- Run focused lit/FileCheck for affected dispatch source and bundle routes,
  including `test/Scripts/rvv-scalar-dispatch-e2e.test` and
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test` when impacted.
- Run `git diff --check`.
- Run Trellis task validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-dispatch-component-body-authority`.
- Run full local project check:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if the build tree remains usable.
- Do not run `ssh rvv` unless runtime behavior changes and the report needs a
  fresh bounded runtime claim.

## Technical Notes

- Required specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Prior completed context:
  `.trellis/tasks/archive/2026-05/05-11-rvv-structured-body-export-authority/prd.md`.
- Primary implementation surfaces:
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`,
  `include/TianChenRV/Target/RVVScalarDispatch.h`,
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`, and
  `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`.
- Primary tests:
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Scripts/rvv-scalar-dispatch-e2e.test`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`, and affected
  `test/Transforms/LinalgToExec/` dispatch-related frontend tests.

## Definition Of Done

The task is done when the bounded `i32-vmul` RVV+scalar dispatch artifact path
fails closed on stale RVV and scalar component body/candidate authority,
continues to emit the valid dispatch source/header/object/self-check paths,
focused C++ and lit checks pass, Trellis task validation passes, the task is
finished and archived, and one coherent commit records the implementation.

If unfinished, leave this task open and record the exact continuation point:
PRD/context repair, RVV component authority validation, scalar component
authority validation, source/header/object/self-check reuse, focused C++ tests,
lit route checks, full `check-tianchenrv`, Trellis validation, archive, or
commit.

## Completion Notes

- Added target-owned RVV component authority validation in
  `validateRVVMicrokernelSourceAuthority`, reusing the direct RVV
  `buildModuleRecordForRVVBinaryFamily` body-authority path.
- Added target-owned scalar component authority validation in
  `validateScalarMicrokernelSourceAuthority`, reusing the scalar
  `buildModuleRecord` source/body-authority path.
- Updated `RVVScalarDispatch.cpp` so embedded callable source construction
  validates both component authorities before any dispatch source, header,
  object, self-check source, or self-check object route can emit bytes.
- Added scalar embedded-source consistency checks for function name, typed
  microkernel op, selected variant, role, runtime ABI kind/name, and runtime
  glue role after scalar source export.
- Added C++ target/export coverage for one valid `i32-vmul` dispatch component
  authority path plus stale RVV and stale scalar body mismatches.
- Added lit fail-closed coverage for stale RVV and stale scalar component
  bodies in the frontend-selected `i32-vmul` dispatch source route while
  preserving the existing positive source/header/self-check checks.
- Focused checks passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`,
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`,
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir Scripts/rvv-scalar-dispatch-e2e.test Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  from `artifacts/tmp/tianchenrv-build/test`, and `git diff --check`.
- Full project check passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  with 205 lit tests passing.
- Trellis validation passed:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-11-rvv-dispatch-component-body-authority`.
- No fresh `ssh rvv` evidence was produced because this round changes export
  preflight authority and does not claim new runtime correctness, performance,
  or generic RVV backend maturity.
