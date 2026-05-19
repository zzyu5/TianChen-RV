# Stage2 RVV selected-body realization module extraction

## Goal

Extract the active RVV selected-body realization owner out of the monolithic
`RVVExtensionPlugin.cpp` implementation into a plugin-local module consumed by
`RVVExtensionPlugin::materializeSelectedLoweringBoundary`. The extracted owner
must continue to find the current selected pre-realized bodies, validate their
typed facts, fail closed with targeted diagnostics, and build realized
`tcrv_rvv` bodies for the existing generic unit-stride, strided, masked, macc,
and reduce paths before route-provider construction.

This is an architecture/ownership extraction over existing Stage2 behavior. It
is not a new RVV coverage task and not a helper-only cleanup.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`, `git status --short`, and `git log --oneline -8` showed a
  clean worktree at `e47ff870 rvv: demote legacy i32 parse surface`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-20-stage2-rvv-selected-body-realization-extraction`.
- `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Current `RVVExtensionPlugin.cpp` owns too much: pre-realized body discovery,
  selected-body fact validation, fail-closed diagnostics, realized body
  construction, dispatcher logic, and plugin orchestration live in one file.
- Current selected-body realization paths cover:
  `tcrv_rvv.typed_binary_pre_realized_body` for add/sub/mul and strided add,
  `tcrv_rvv.typed_masked_binary_pre_realized_body` for masked add,
  `tcrv_rvv.typed_macc_pre_realized_body` for macc add, and
  `tcrv_rvv.typed_reduce_pre_realized_body` for reduce add.
- Existing route authority remains in `RVVEmitCRouteProvider.cpp`. The route
  provider must continue to consume realized typed body structure only.
- Common EmitC/materialization and target export must remain neutral consumers
  of provider-built routes.
- Stage1 guardrails remain active: do not reintroduce positive `RVVI32M1`,
  `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor, artifact-name, status, or exact
  intrinsic spelling as route authority.

## Requirements

1. Create a plugin-local selected-body realization module owned by RVV, with a
   public API narrow enough for `RVVExtensionPlugin.cpp` to consume.
2. Move body-specific selected-body realization decisions out of
   `RVVExtensionPlugin.cpp`: pre-realized body discovery, supported
   pre-realized body dispatch, typed fact validation, fail-closed diagnostics,
   and realized `tcrv_rvv` body construction.
3. Preserve the existing positive behavior for generic add/sub/mul,
   pre-realized strided add, masked add, macc add, and reduce add.
4. Preserve existing fail-closed behavior for missing or inconsistent op kind,
   memory form, policy/config, runtime ABI roles, stride facts, accumulator and
   result facts, mixed pre-realized/realized bodies, missing selected variant
   `requires`, stale legacy authority, and source-front-door residue.
5. `RVVExtensionPlugin::materializeSelectedLoweringBoundary` must explicitly
   consume the extracted realization module when no realized `with_vl` boundary
   exists and a pre-realized body exists.
6. Route-provider authority must remain provider-owned. The extraction must not
   move route construction, intrinsic mapping, dtype inference, SEW/LMUL
   derivation, or ABI mapping into common code.
7. Common EmitC/export must remain neutral and must not branch on RVV semantics.
8. Do not add new RVV op coverage, dtype/LMUL expansion, frontend lowering,
   source-front-door positive routes, compatibility wrappers, descriptors,
   dashboards, performance claims, or report-only achievements.

## Acceptance Criteria

- [x] PRD, implement context, and task metadata truthfully describe selected
      body realization extraction and do not drift into coverage expansion.
- [x] `RVVExtensionPlugin.cpp` no longer owns the body-specific construction
      dispatcher/planner for the existing pre-realized selected bodies.
- [x] A consumed RVV plugin-local module owns selected-body realization
      decisions for the existing generic unit-stride, strided, masked, macc,
      and reduce paths, or a single coherent complete slice with an explicit
      continuation point.
- [x] `materializeSelectedLoweringBoundary` calls the extracted module before
      selected-boundary validation and before route-provider construction.
- [x] Existing route-supported and artifact behavior remains unchanged for at
      least one unit-stride selected body and one nontrivial selected body such
      as masked, macc, reduce, or strided.
- [x] Negative fail-closed selected-body tests still pass.
- [x] Active-authority scan confirms no active `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door /
      source-seed, descriptor/direct-C/source-export, or common/export RVV
      semantic authority is reintroduced.
- [x] Focused build, lit, C++ unit, script, `git diff --check`, and task
      validation checks pass. Run `check-tianchenrv` if shared plugin behavior
      changes enough to justify the broader gate.

## Non-Goals

- No new RVV operation coverage or dtype/LMUL expansion.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV route or source-seed evidence mode.
- No Template/Toy/TensorExtLite/IME/Offload/future-plugin work.
- No descriptor-driven computation or descriptor-driven C/source export.
- No compatibility wrapper preserving legacy i32 route authority.
- No runtime, correctness, or performance claim without fresh `ssh rvv`
  evidence. This extraction should preserve existing behavior and does not need
  new hardware evidence unless it changes executable behavior.

## Validation Plan

1. Validate task context with `python3 ./.trellis/scripts/task.py validate`.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit for:
   - one unit-stride pre-realized selected body;
   - strided, masked, macc, and reduce pre-realized selected-body fixtures;
   - existing selected-body negative fixtures;
   - representative explicit selected-body provider/materializer fixtures.
4. Run touched C++ tests, especially `tianchenrv-rvv-extension-plugin-test`.
5. Run `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` and
   representative dry-run tests if script-facing paths are touched or if lit
   script tests cover the handoff.
6. Run `git diff --check`.
7. Run an active-authority scan over active RVV include/lib/script/test paths.
8. Run `check-tianchenrv` if focused validation suggests shared plugin
   behavior changed beyond the extraction.

## Implementation Results

- Added `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h` and
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` as the RVV plugin-local owner
  for selected-body realization.
- Moved pre-realized selected-body detection, unique body lookup, typed fact
  validation, fail-closed diagnostics, and realized typed-body construction out
  of `RVVExtensionPlugin.cpp`.
- The extracted module now owns the current pre-realized selected-body paths:
  generic add/sub/mul, strided add, masked add, macc add, and reduce add.
- `RVVExtensionPlugin::materializeSelectedLoweringBoundary` now keeps only the
  plugin orchestration: legality, existing realized-boundary lookup, call into
  the extracted realization module when needed, selected-boundary validation,
  and materialized-boundary result construction.
- Route authority remains in `RVVEmitCRouteProvider.cpp`; no route construction,
  intrinsic mapping, dtype/config derivation, or ABI mapping moved into common
  code.
- Common EmitC/export was not modified.
- No new RVV coverage class, dtype/LMUL expansion, source-front-door positive
  route, descriptor path, runtime behavior, correctness claim, or performance
  claim was added.

## Validation Results

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-selected-body-realization-extraction`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [x] `build/bin/tianchenrv-rvv-dialect-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-construction-protocol-common-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] Focused lit from `build/test`: 12/12 passed for pre-realized
      add/strided/masked/macc/reduce positive target fixtures, selected-body
      fail-closed negatives, and representative explicit provider/materializer
      fixtures.
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused generated-bundle lit dry-run tests: 5/5 passed for pre-realized
      unit-stride, strided, masked, macc, and reduce paths.
- [x] `git diff --check`
- [x] Diff-added active-authority scan over `include/TianChenRV/Plugin/RVV` and
      `lib/Plugin/RVV` found no newly added positive `RVVI32M1`,
      `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-seed, descriptor/direct-C/source-export, or
      default source-front-door eligibility authority.
- [x] `cmake --build build --target check-tianchenrv -j2`: 176/176 lit tests
      passed.
- [x] `clang-format` was not available in the environment; formatting was
      checked with `git diff --check`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-generic-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-body-strided-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-masked-add-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-macc-add/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-reduce-add-realization/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage1-legacy-rvv-i32-residue-cleanup/prd.md`

Initial code surface inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- selected-body lit/script tests under `test/Target/RVV`,
  `test/Transforms/LoweringBoundary`, `test/Conversion/EmitC`, and
  `test/Scripts`.

## Definition Of Done

- [x] Extracted module is consumed by the production/default RVV plugin path.
- [x] Focused checks pass and failures are repaired.
- [x] Task status and journal notes are truthful.
- [x] Task is finished/archived if complete.
- [x] One coherent commit records the PRD, extraction, validation, and task
      closeout, or the task remains open with an exact continuation point.
