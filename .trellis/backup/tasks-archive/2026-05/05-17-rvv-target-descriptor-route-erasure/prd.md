# RVV target descriptor-route authority erasure

## Goal

Delete the RVV target-support descriptor/table authority that publishes finite
i32m1 add/sub/mul object, header, and artifact-backed translate routes as if
they were the production target export path. This round is deletion-only: keep
the already-real typed RVV op to materialized EmitC handoff, but remove the
Target/RVV route table and tests that protect descriptor-shaped artifact-route
ownership.

## What I Already Know

- Current HEAD is `6b92bca` and the worktree was clean before this task.
- The previous task proved the selected RVV i32m1 add source dispatch can reach
  a verified materialized MLIR EmitC module before MLIR EmitC C/C++ emission.
- `.trellis/spec/index.md` says descriptor-driven computation and direct-C
  artifact paths are historical residue, deletion targets, or fail-closed debt.
- `.trellis/spec/lowering-runtime/emitc-route.md` preserves the route order
  `extension family ops -> EmitC ops -> C/C++ emitter`, and explicitly rejects
  descriptor strings choosing arithmetic intrinsics or direct source export.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says the current correct
  state for deleted RVV target artifact routes is an unsupported/deleted-route
  diagnostic, with future rebuild through explicit extension-family ops to
  materialized MLIR EmitC.
- Live code inspection shows `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  currently owns `RVVI32M1ArithmeticTargetRouteDescriptor` and
  `kRVVI32M1ArithmeticTargetRoutes`, which publish object/header artifact route
  ids, translate route ids, export callbacks, composite grouping, and
  add/sub/mul registration.
- `RVVExtensionPlugin` currently reports supported RVV emission plans by
  calling Target/RVV route-id accessors, so target route erasure must also make
  selected RVV artifact emission fail closed instead of naming deleted routes.

## Requirements

- Remove `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, and equivalent Target/RVV descriptor/table
  route-authority structures.
- Stop registering RVV i32m1 add/sub/mul object artifact exporters, callable
  header composite exporters, and artifact-backed target translate routes.
- Keep the materialized RVV EmitC module to MLIR EmitC C/C++ translate route if
  it does not preserve descriptor-driven compute or descriptor-owned artifact
  route authority.
- Make selected RVV artifact emission readiness/planning fail closed with an
  unsupported/deleted-route diagnostic instead of a supported object/header
  route.
- Remove or rewrite tests that protect descriptor-selected add/sub/mul
  object/header/translate routes as compatibility behavior.
- Retain tests that prove the typed RVV body can materialize into EmitC and
  that deleted target artifact routes are absent/fail closed.
- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Do not add Python compiler-core logic.

## Acceptance Criteria

- [ ] `lib/Target/RVV`, `include/TianChenRV/Target/RVV`,
  `tools/tcrv-translate`, and `test/Target/RVV` no longer contain the RVV
  target descriptor/table route authority or tests protecting descriptor-based
  object/header/translate route behavior.
- [ ] Built-in RVV target-support route publication exposes only the
  descriptor-free materialized EmitC to C/C++ translate route.
- [ ] Built-in target artifact exporter registration no longer publishes RVV
  i32m1 arithmetic object/header artifact routes.
- [ ] Selected RVV emission planning produces an unsupported/deleted-route
  diagnostic and does not name a supported RVV object/header artifact route.
- [ ] Focused C++ and lit checks for touched target/translate/RVV surfaces pass,
  or any failure is reported as an intentional missing-rebuild gap without
  restoring descriptor logic.
- [ ] Targeted scans show no `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, direct RVV object/header translate route
  tests, or descriptor-to-C/source-export route authority in the touched
  surfaces.

## Non-Goals

- No replacement artifact route, executable plugin template, compatibility
  wrapper, descriptor adapter, legacy mode, new source-seed shape, new RVV
  operation coverage, direct C semantic exporter, generated object/header
  rebuild, or `ssh rvv` evidence as the main result.
- No broad test matrix beyond focused validation of the changed module
  behavior.
- No changes to the provider-owned typed RVV op to EmitC route except where
  required to stop selected target artifact route publication.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-source-dispatch-emitc-handoff/prd.md`
  and `.trellis/workspace/codex/journal-8.md`.
- Primary code surfaces:
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/RVV/`,
  `test/Target/TargetArtifactExportTest.cpp`, and
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`.

## Definition Of Done

- Deleted descriptor/table route authority is not replaced by another finite
  target-route table or compatibility path.
- Focused checks and scans are recorded.
- Trellis task state and workspace journal are updated truthfully.
- One coherent commit records the deletion round if complete; otherwise the
  task remains open with an exact continuation point.

## Implementation Summary

- Deleted the RVV Target/RVV i32m1 add/sub/mul descriptor table and the object,
  callable-header, and artifact-backed translate route registration it owned.
- Kept only `tcrv-rvv-emitc-to-cpp`, which requires an already materialized
  EmitC module and does not own descriptor-selected artifact export.
- Removed RVV construction-protocol target artifact route fields so provider
  construction checks no longer preserve object/header/translate route IDs as
  route authority.
- Changed RVV selected emission readiness/planning to fail closed with an
  unsupported deleted-route diagnostic while leaving explicit typed RVV body to
  EmitC materialization available in the retained EmitC tests.
- Deleted the old `test/Target/RVV/i32m1-*artifact*.mlir` object/header/bundle
  fixtures and rewrote target/plugin tests to assert route absence.
- Updated the relevant lowering-runtime and RVV plugin specs so they no longer
  describe finite RVV object/header target artifact routes as current
  production behavior.

## Validation

- Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tcrv-opt tcrv-translate -j2`.
- Focused C++ tests:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-construction-protocol-common-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized|rvv-first-slice-materialization|source-seed-artifact-front-door'`
  from `build/test`, 13/13 selected tests passed.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`, 100/100 lit tests
  passed.
- `git diff --check` passed.
- Targeted scans found no
  `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`,
  `getRVVI32M1Arithmetic*RouteID`, old `tcrv-rvv-i32m1-{add,sub,mul}` target
  route strings, or RVV construction object/header/translate route fields in
  Target/RVV, RVV plugin construction, related tests, or translate surfaces.
- Direct source/export scan found only non-RVV Toy/Offload guard messages and
  retained RVV EmitC-to-C++ fixture evidence for `riscv_vector.h` and
  `__riscv_vadd_vv_i32m1`; no RVV target artifact direct exporter remains.

## Self-Repair

- The first RVV source-seed lit rewrite tried to materialize EmitC after
  deleting the supported emission-plan route. That correctly failed because
  the source-seed module still contains both RVV case and scalar fallback and
  now has no supported non-fallback artifact plan. The test was narrowed to
  prove deleted-route diagnostics and artifact-export fail-closed behavior;
  explicit RVV EmitC materialization remains covered by the dedicated retained
  EmitC tests.

## Status

- Complete. The task can be archived after Trellis finish bookkeeping and one
  coherent commit.
