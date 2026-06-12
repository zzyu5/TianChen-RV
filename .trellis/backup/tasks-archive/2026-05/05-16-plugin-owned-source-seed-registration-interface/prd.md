# Plugin-owned source seed registration interface

## Goal

Introduce one common, plugin-owned source-seed pass registration surface and
migrate the existing RVV i32m1 add selected-boundary seed to it. The public
tool flow should become:

```text
source seed declaration
  -> extension plugin registry
  -> tool pass registration
  -> RVV-owned selected-boundary materialization pass
  -> existing RVV EmitC / target artifact route
```

This is a registration and ownership refactor for the existing bounded seed. It
must not broaden the RVV source language or restore a core RVV source frontend.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `a444682`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the supplied Direction Brief.
- The previous task added
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed` and proved the seed
  reaches selected RVV boundary IR, emission-plan/EmitC route consumption,
  target artifact generation, and `ssh rvv` correctness.
- `.trellis/spec/index.md` requires compiler implementation in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck, keeps computation out of
  `tcrv.exec`, and rejects descriptor/direct-C/Python compiler-core paths.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires public
  tool passes that need plugins to receive a registry from the tool/plugin
  loader and route through generic interfaces rather than target-family
  branches.
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md` says the
  old core RVV source frontend slice is deleted, but explicitly allows the
  bounded RVV-owned i32m1 add seed as a plugin-owned entry point.
- The current problem is architectural integration: `tcrv-opt` directly
  includes `RVVSelectedBoundarySeed.h` and registers the RVV seed factory,
  bypassing the plugin registry ownership boundary.

## Requirements

- Add a small common C++ plugin interface for source-seed pass registration.
  The common surface may carry only bounded pass descriptor/factory plumbing:
  pass option, description, pass factory, and owner/plugin metadata if useful.
- Add registry support for collecting source-seed pass registrations from
  enabled plugins in deterministic plugin registration order.
- Move the existing RVV i32m1 selected-boundary seed factory behind
  `RVVExtensionPlugin` ownership.
- Update `tcrv-opt` so it registers source-seed passes through the common
  `ExtensionPluginRegistry` instead of including or directly invoking
  `RVVSelectedBoundarySeed.h`.
- Preserve the public RVV seed option spelling:
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed`.
- Preserve the existing RVV seed semantics and fail-closed contract exactly:
  explicit marker `tcrv_rvv.lowering_seed = "i32m1_add"`, the same ABI/source
  shape checks, the same selected `origin = "rvv-plugin"` variant form, and
  the same selected-boundary output consumed by the RVV EmitC/artifact route.
- Built-in plugin enablement must control source-seed pass availability:
  with built-ins enabled, the RVV seed pass is available; with
  `--tcrv-disable-builtin-plugins`, the RVV seed option is not registered and
  should fail as an unknown pass/option.
- Keep a default empty-registry behavior for embedded/C++ users: no plugin
  source-seed pass registrations are collected, and no common layer invents
  target-family semantics.

## Acceptance Criteria

- [x] `tools/tcrv-opt/tcrv-opt.cpp` no longer includes
      `TianChenRV/Plugin/RVV/RVVSelectedBoundarySeed.h`.
- [x] `tcrv-opt` no longer directly registers
      `createMaterializeRVVI32M1SelectedBoundarySeedPass`; source-seed pass
      registration flows through `ExtensionPluginRegistry`.
- [x] The RVV seed pass remains available under built-in plugins with the same
      public option and unchanged selected-boundary / emission-plan / EmitC
      output checked by existing positive lit coverage.
- [x] The RVV seed pass is not registered when built-in plugins are disabled,
      with focused coverage using `--tcrv-disable-builtin-plugins`.
- [x] Common plugin interfaces carry only pass descriptor/factory plumbing and
      no RVV arithmetic, dtype, shape, route-id, target artifact, descriptor,
      direct-C, or Python compiler-core semantics.
- [x] Existing RVV seed positive and negative lit tests still pass.
- [x] RVV first-slice selected-boundary/materialization tests and RVV plugin
      tests affected by registry ownership still pass.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` runs if practical; if not, the narrower focused
      checks and reason are recorded.
- [x] Hardware evidence is not required for a registration-only refactor if
      selected-boundary output and target route semantics are unchanged; if any
      output or target route semantics change, rerun the RVV artifact path and
      `ssh rvv`.

## Validation Results

- Focused build passed:
  `cmake --build build --target TianChenRVPlugin TianChenRVRVVPlugin tcrv-opt
  tcrv-translate tianchenrv-plugin-registry-test
  tianchenrv-rvv-extension-plugin-test -j2`.
- Self-repair: the first focused build failed because `RVVExtensionPlugin.cpp`
  returned `std::unique_ptr<mlir::Pass>` from the new pass factory lambda
  without including the complete `mlir::Pass` type. Added
  `mlir/Pass/Pass.h` and reran the same focused build successfully.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-plugin-registry-test` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Initial lit invocation from repo root against `build/test` failed because
  this repo's generated `lit.site.cfg.py` uses relative paths. Reran from
  `build/test`, matching project convention.
- Focused RVV lit passed:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary'`
  from `build/test`, 19/19 passed.
- Full practical check passed:
  `cmake --build build --target check-tianchenrv -j2`, 104/104 passed.
- `git diff --check` passed.
- `rg -n "RVVSelectedBoundarySeed|createMaterializeRVVI32M1SelectedBoundarySeedPass" tools/tcrv-opt tools/tcrv-translate`
  returned no matches, confirming public tools no longer directly wire the RVV
  seed factory.
- Hardware evidence was not rerun because this is a registration-only refactor:
  selected-boundary, emission-plan, EmitC, and target route semantics are
  unchanged and covered by focused lit plus the previous task's `ssh rvv`
  evidence.

## Definition Of Done

- Task context files are curated and truthful.
- PRD boundaries and acceptance criteria are reflected by focused tests.
- Implementation is one coherent module-level refactor, not a helper-only
  addition beside the old direct path.
- Task status and journal are updated truthfully.
- Task is finished/archived when complete.
- One coherent commit is created when complete. If unfinished, leave the task
  open and name the exact next continuation point.

## Out Of Scope

- New RVV source shapes, sub/mul source seeds, new SEW/LMUL/dtype/op families,
  generic MLIR vector lowering, scalable-vector lowering, linalg/stablehlo/tosa
  frontend lowering, TensorExt/IME/offload work, new artifact routes,
  performance tuning, descriptor or binary-family registries, direct C
  semantic exporters, Python compiler-core logic, GCC-default routes,
  compatibility wrappers, state-machine/checkpoint ledgers, docs-only work,
  and broad smoke matrices.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-16-05-16-bounded-mlir-to-rvv-selected-boundary-seed/prd.md`.
- Relevant journal read:
  `.trellis/workspace/codex/journal-8.md` sessions 93 and 94.
- Primary code surfaces inspected:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVSelectedBoundarySeed.h`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVExtensionPlugin.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `test/Plugin/PluginRegistryTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`, and
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed-negative.mlir`.
