# Stage2 RVV Computed-Mask Segment2 Selected-Body Realization Migration

## Goal

Move the production path for `computed_masked_segment2_load_unit_store`,
`computed_masked_segment2_store_unit_load`, and
`computed_masked_segment2_update_unit_load` fully behind the RVV
plugin-local selected-body realization producer, instead of leaving the
computed-mask segment2 realization path as an effectively direct route-entry
shortcut with only harness-level retirement checks.

The intended production chain is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv computed-mask segment2 body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving compare/mask facts, segment2
     field roles, source/destination memory forms, update arithmetic facts,
     runtime ABI binding, runtime n/AVL/VL control, RVV policy, and artifact
     ABI order
  -> computed-mask memory route-family facts
  -> segment2 route-family facts / statement-plan boundary
  -> route materialization facts
  -> memory operand-binding facts
  -> route-control provider plan
  -> RVV-owned segment2 statement-plan owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

The generated artifact path must not continue to rely on direct
pre-realized route-entry authority, route id mirroring, artifact name,
script-option authority, descriptor residue, or any stale metadata path that
short-circuits the selected-body producer.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- `19dbe2a8` demoted segment2 generated-bundle direct entry and added the
  no-route-entry fence, but that only hardened the harness path.
- The current RVV specs already require pre-realized selected bodies to flow
  through the public selected lowering-boundary producer before route facts
  are collected, and they reject route-entry authority from the selected-body
  path.
- The segment2 provider-plan owner boundary is already split out of central
  route planning, so this task should build on that owner boundary instead of
  reopening provider-plan ownership.
- `RVVSelectedBodyRealization.cpp` still contains the production realization
  branches for computed-mask segment2 load/store/update, so the selected-body
  producer path still needs an explicit migration step rather than only
  validation or harness cleanup.
- The generated-bundle script already records `route_entry_realization: false`
  and `selected_body_realization_producer` evidence for pre-realized selected
  bodies, so the remaining work is production code alignment, not a new
  harness field.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python may only be used for generated-bundle tooling, harnesses,
   diagnostics, and artifact parsing.
2. Migrate the computed-mask segment2 selected-body realization path so the
   public selected lowering-boundary producer, not direct route-entry
   authority, is the production path for the three bounded op kinds.
3. Preserve the realized body facts end to end:
   compare lhs/rhs, segment field roles, source/destination memory forms,
   masked-passthrough behavior, update arithmetic kind where applicable,
   runtime ABI binding, runtime n/AVL/VL control, policy, and artifact ABI
   order.
4. Keep unsupported or inconsistent inputs fail-closed with targeted
   diagnostics before provider facts, route construction, common EmitC, or
   target export.
5. Preserve the existing segment2 route-family/provider ownership boundary and
   keep plain segment2 and other RVV families out of scope.
6. Preserve direct route-entry retirement for the computed-mask segment2
   family; any direct probe must fail closed before route-entry materialization
   or bundle generation.

## Non-goals

- Do not start plain segment2 deinterleave/interleave migration.
- Do not start other contraction families, other selected-body families, or
  broad route-composition cleanup.
- Do not do report-only work, broad smoke matrices, or script-only fence
  cleanup without production code movement.

## Completion Criteria

- The selected-body realization path for the three computed-mask segment2
  op kinds is produced through the RVV plugin-local selected-body boundary.
- The production diff touches selected-body realization and any directly
  affected route/provider or target-artifact boundary if needed.
- Focused tests prove the pre-realized selected-body path still reaches
  provider-built route facts and emits the expected provider/header metadata.
- Generated-bundle dry-run evidence records `route_entry_realization: false`
  and `selected_body_realization_producer`.
- Direct pre-realized route-entry probes for the three bounded op kinds fail
  closed with targeted diagnostics.
- `git diff --check` passes and the worktree is clean after a coherent commit.

## Validation Plan

1. Validate the task context with `python3 ./.trellis/scripts/task.py validate`
   after the PRD and context files are in place.
2. Implement the computed-mask segment2 selected-body realization migration
   in the RVV plugin production path.
3. Run the focused RVV plugin C++ tests that cover the segment2 route/provider
   boundary and the selected-body realization path.
4. Run the computed-mask segment2 generated-bundle dry-run evidence and the
   direct fail-closed probe.
5. Run `git diff --check`.
6. Run `check-tianchenrv` or record the exact blocker.
7. Archive the task and record the session journal once the change is complete.

## Files to Inspect First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-segment2-*.test`
