# Stage2 RVV selected-boundary-only route-construction API closure

## Goal

Close the remaining direct route-entry API and naming residue in the RVV
selected-body production interface after the direct fallback retirement. RVV
route construction must expose the selected-boundary-only contract as the
production API shape: selected pre-realized bodies are consumed only by public
selected lowering-boundary materialization, provider facts are collected only
from realized typed `tcrv_rvv` bodies, and any remaining route-entry wording is
limited to explicit retired/fail-closed diagnostics or negative CLI inventory.

## Direction Source

- Direction title: `Continue: Stage2 RVV selected-boundary-only
  route-construction API closure`.
- Module owner: RVV plugin production interface between selected-body
  realization and provider route construction.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e3826f14 rvv: retire direct route-entry realization fallback`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The predecessor task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-production-direct-route-entry-fallback-retirement`
  removed production calls from `RVVExtensionPlugin` to the direct route-entry
  realization helper, made selected-boundary materialization the required path,
  and passed focused generated-bundle, `ssh rvv`, and `check-tianchenrv`
  evidence.
- `RVVExtensionPlugin.cpp` already requires an existing `setvl/with_vl`
  selected boundary before emission-plan or EmitC route construction, and
  fails closed if a pre-realized selected body reaches route construction.
- `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h` still exposes
  `RVVSelectedBodyRealizationOwner::isRouteEntryConsumer`, the segment2
  `RVVSelectedBodySegment2RouteEntryFamilyOwner` query surface,
  `variantContainsPreRealizedRVVRouteEntrySelectedBody(...)`, and
  `realizePreRealizedRVVRouteEntrySelectedBody(...)`.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` still implements an empty
  segment2 route-entry family registry, a route-entry owner lookup, a
  route-entry variant query, and a helper returning `Expected<WithVLOp>` even
  though the helper now always fails closed.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` has at least one production
  segment2 diagnostic that refers to a `registered route-entry family`, even
  though the active planning owner boundary is selected-body route-family
  planning.
- `test/Plugin/RVVExtensionPluginTest.cpp` intentionally contains direct
  route-entry negative coverage, but it also still compiles against the old
  route-entry owner fields and segment2 route-entry registry query APIs.
- `scripts/rvv_generated_bundle_abi_e2e.py` keeps
  `--direct-pre-realized-route-entry` as fail-closed CLI inventory. This is
  acceptable if it remains negative-only and no production materialization path
  depends on it.

## Requirements

1. Remove `isRouteEntryConsumer` from the production selected-body realization
   owner API and update owner registry initializers/tests accordingly.
2. Remove the empty public segment2 route-entry family owner registry/query
   surface from production headers and implementation. Segment2 production
   planning must be described through selected-body route-family provider
   planning, not route-entry family ownership.
3. Remove the production/public
   `variantContainsPreRealizedRVVRouteEntrySelectedBody(...)` query. Tests that
   need to prove a pre-realized body exists should use the selected-body query;
   tests that need direct route-entry behavior should assert the explicit
   retired diagnostic.
4. Reframe the retired direct route-entry helper so it cannot be mistaken for a
   materialization API. It may remain only as an explicit diagnostic function
   used by negative tests, and it must not return `WithVLOp` or create route
   structure.
5. Reword route planning/provider diagnostics that still say `registered
   route-entry family` when they describe the active selected-body route-family
   planning owner boundary.
6. Keep production route construction selected-boundary-only:
   `selected tcrv.exec RVV variant -> public selected lowering-boundary
   materializer -> realized typed tcrv_rvv body -> provider route facts ->
   TCRVEmitCLowerableRoute`.
7. Keep common EmitC/export neutral and avoid moving RVV realization, dtype,
   operation, route, or intrinsic authority into common code.
8. Do not delete useful negative direct-route tests merely because filenames or
   CLI flags contain route-entry wording; update only the tests required by the
   production API shape change.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      production API cleanup, specs, non-goals, and validation plan.
- [x] `RVVSelectedBodyRealizationOwner` no longer has an
      `isRouteEntryConsumer` field and no production registry entry exposes a
      route-entry predicate.
- [x] The public segment2 route-entry family registry/query API is removed from
      production headers and implementation.
- [x] The route-entry variant query is removed from production/public API; tests
      no longer use it as authority.
- [x] The retained direct-route diagnostic helper is explicitly named and typed
      as a retired diagnostic (`llvm::Error`), not as a `WithVLOp`
      materialization API.
- [x] Route planning/provider diagnostics touched by this task refer to
      selected-body route-family planning instead of registered route-entry
      family support.
- [x] Focused C++ plugin tests still prove selected-boundary-only
      fail-closed behavior, absent direct-route production predicates, and
      provider construction preconditions.
- [x] Representative selected-boundary generated-bundle dry-run still reports
      `tcrv-materialize-selected-lowering-boundaries` and
      `route_entry_realization: false` for at least one arithmetic or memory
      case and one segment2 or computed-mask case if touched.
- [x] Direct generated-bundle route-entry fail-closed subset still passes.
- [x] Build targets `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, and
      `tcrv-translate` pass.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] If no executable materialization path changes, final report states that no
      new runtime/correctness/performance claim was made and no new `ssh rvv`
      run was required.
- [x] Task status, journal, and archive state are truthful; a coherent commit is
      created only when the task is complete.

## Out of Scope

- Do not add new RVV op coverage, dtype/LMUL clone batches, compare/select,
  reduction, conversion, contraction, or memory route families.
- Do not add source-front-door routes, descriptor-driven C/source export, or
  exact-intrinsic-derived route authority.
- Do not start high-level Linalg/Vector frontend lowering.
- Do not introduce one-intrinsic wrapper dialects, Stage3 plugin generality,
  dashboards, broad smoke matrices, or report-only closure.
- Do not rewrite the RVV spec broadly beyond the specific selected-boundary-only
  API contract needed by this cleanup.

## Validation Plan

1. Focused C++ plugin test build and execution:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build production tools touched by route construction:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Direct generated-bundle route-entry fail-closed subset:
   lit filter `rvv-generated-bundle-abi-e2e-direct-pre-realized`.
4. Representative selected-boundary dry-run with at least one arithmetic or
   memory case and one segment2 or computed-mask case, checking materializer
   and `route_entry_realization: false`.
5. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and script
   self-test if the script is touched; otherwise record that it was not changed.
6. Bounded authority scan over touched RVV production files proving no
   production caller, registry field, owner predicate, fallback helper, or
   route-family diagnostic still grants direct route-entry materialization
   authority.
7. `git diff --check`.
8. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Production files inspected:
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` by route-entry scan.
- Negative direct-route CLI inventory in
  `scripts/rvv_generated_bundle_abi_e2e.py` is not production route authority.
  It may remain if checks prove it still fails closed.

## Spec Update Judgment

Required and completed. This task changed a production/public C++ API surface,
so `.trellis/spec/extension-plugins/rvv-plugin.md` was updated to remove the
old route-entry consumer field, segment2 route-entry registry/query surface,
route-entry variant query, and `WithVLOp`-returning direct helper from the
durable selected-body realization contract. The retained direct-route wording is
now limited to `diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(...)`,
negative generated-bundle CLI inventory, fail-closed diagnostics, and historical
wrong/correct examples.

## Completion Evidence

- Production API closure:
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h` no longer
  exposes `isRouteEntryConsumer`, segment2 route-entry family owner/query APIs,
  `variantContainsPreRealizedRVVRouteEntrySelectedBody(...)`, or a
  `WithVLOp`-returning direct route-entry helper.
- Implementation closure:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` removes the empty segment2
  route-entry registry and route-entry variant query, simplifies all selected
  realization owners to `{familyName, isConsumer, realize}`, and retains only
  `diagnoseRetiredPreRealizedRVVRouteEntrySelectedBody(...)` as an
  `llvm::Error` fail-closed diagnostic.
- Segment2 wording closure:
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` now reports selected-body
  route-family mismatch wording instead of `registered route-entry family`.
- C++ test closure:
  `test/Plugin/RVVExtensionPluginTest.cpp` compiles against the selected-body
  owner API only, keeps retired direct-route diagnostic coverage through the
  diagnostic helper, and still exercises selected-boundary realization/provider
  preconditions.
- Focused checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `cmake --build build --target tcrv-opt tcrv-translate -j2`,
  and the lit subset
  `rvv-generated-bundle-abi-e2e-direct-pre-realized` passed 30/30.
- Selected-boundary evidence:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body` passed for `scalar_broadcast_add` and
  `computed_masked_segment2_update_unit_load`; both evidence files report
  materializer `tcrv-materialize-selected-lowering-boundaries` and
  `route_entry_realization: false`.
- Script checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Authority scans passed:
  old production API names are absent from `include/TianChenRV/Plugin/RVV`,
  `lib/Plugin/RVV`, and `test/Plugin/RVVExtensionPluginTest.cpp`; common
  conversion/target code has no route-entry fallback references. The only
  production route-entry wording left in RVV headers/implementation is the
  explicit retired diagnostic function/string.
- Full quality gate:
  `git diff --check` passed and
  `cmake --build build --target check-tianchenrv -j2` passed 464/464 tests.
- Runtime evidence:
  No executable materialization path changed and no new
  runtime/correctness/performance claim was made, so no new `ssh rvv` run was
  required for this cleanup.
