# Stage2 RVV provider facts-only operand-binding closure

## Goal

Make the selected-body RVV EmitC route provider facts-only for production
operand binding before `TCRVEmitCLowerableRoute` statement construction. This
round closes the remaining ordinary `Add` / `Sub` / `Mul` fallback branch that
still locally binds `lhs`, `rhs`, `out`, and required materialized uses from
`RouteOperandBindingPlan` when RVV-owned elementwise/select operand-binding
facts are absent.

## Direction Source

- Direction title: `Stage2 RVV provider facts-only operand-binding closure`.
- Module owner: selected-body RVV EmitC route provider operand-binding gate for
  ordinary elementwise arithmetic fallback.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `d666bd1d rvv: own residual operand binding facts`.
- `.trellis/.current-task` was absent, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- Specs require the active RVV route authority chain to flow from selected
  `tcrv.exec` variant through typed low-level `tcrv_rvv` body, RVV-owned
  legality/materialization/operand-binding facts, RVV provider-built
  `TCRVEmitCLowerableRoute`, and common EmitC materialization.
- Common EmitC/export must not infer RVV compute, dtype, ABI roles, operands,
  route ids, or intrinsic spelling.
- Prior completed tasks added RVV-owned operand-binding facts for
  elementwise/select, memory, math, and residual clusters.
- The archived residual task explicitly completed masked elementwise arithmetic,
  strided elementwise add, and runtime scalar splat-store, and left the ordinary
  Add/Sub/Mul fallback as the unrelated remaining local provider fallback.
- Current code evidence in
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` shows the ordinary
  `Add`/`Sub`/`Mul` branch consumes
  `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` when present, but
  otherwise falls back to provider-local `bindOperand` / `requireOperandUse`
  lookups for `lhs`, `rhs`, and `out`.
- Existing elementwise/select facts already validate ordinary elementwise
  `lhs` load, `rhs` load, output store/header, runtime loop-control/header, and
  the elementwise arithmetic route-family plan before returning
  `bindsOrdinaryElementwiseArithmetic`.

## Requirements

1. Remove the ordinary `Add` / `Sub` / `Mul` provider-local fallback that binds
   `lhs`, `rhs`, `out`, and materialized uses directly from
   `RouteOperandBindingPlan`.
2. The production provider must accept ordinary elementwise arithmetic only
   when `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` has
   `bindsOrdinaryElementwiseArithmetic` and exposes the authoritative
   `lhs`, `rhs`, `out`, and runtime count ABI bindings.
3. Missing or stale ordinary elementwise operand-binding facts must fail closed
   before route statement construction with a targeted diagnostic.
4. If a selected-body route has no RVV-owned operand-binding facts cluster for
   runtime AVL/control, route construction must fail closed instead of falling
   back to direct local binding-plan lookup.
5. Remove now-unused provider-local logical operand binding helpers if the
   fallback deletion makes them dead.
6. Preserve emitted semantics, route ids, ABI order, target leaf/header facts,
   operand order, diagnostics style, and generated artifacts for valid ordinary
   elementwise selected-body routes.
7. Do not add route coverage, new operations, dtype/LMUL clone batches,
   source-front-door routes, descriptor/direct-C/source-export paths, artifact
   dashboards, broad smoke matrices, or statement-emission rewrites unrelated
   to facts-only operand binding.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin, EmitC route, testing
      specs, and the previous residual operand-binding task.
- [x] The production `RVVEmitCRouteProvider.cpp` ordinary `Add` / `Sub` /
      `Mul` branch no longer has a local `bindOperand` /
      `requireOperandUse` fallback.
- [x] Ordinary elementwise provider construction fails closed when
      `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts` is missing,
      stale, or not marked as binding ordinary elementwise arithmetic.
- [x] Focused C++ plugin/provider tests prove a valid ordinary elementwise
      route still exposes RVV-owned facts and a stale/missing ordinary
      materialized-use path is rejected before statement construction.
- [x] Representative ordinary Add/Sub/Mul FileCheck coverage under
      `test/Target/RVV` still passes for selected-body artifacts.
- [x] Selected-boundary negative coverage still passes.
- [x] A bounded scan over `RVVEmitCRouteProvider.cpp` shows no remaining
      production local `bindOperand` / `requireOperandUse` logical-operand
      table in selected-body route construction.
- [x] Active-authority scan over touched RVV planning/provider/test files finds
      no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/source-export,
      or mirror-only route authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No route-family expansion, new ops, dtype/LMUL clone batches, high-level
  frontend lowering, legacy i32 authority, source-front-door positive routes,
  descriptor/direct-C/source-export paths, artifact dashboards, broad smoke
  matrices, or unrelated statement emission changes.
- No remigration of memory, math, residual, or scalar-broadcast/select facts
  except direct integration needed to enforce facts-only binding.
- No change to emitted target sequence, ABI, operands, runtime correctness, or
  performance claims. `ssh rvv` evidence is not required unless those claims
  change.
- No movement of RVV semantics into common EmitC/export.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed for focused lit coverage.
5. Run focused lit/FileCheck coverage for ordinary Add/Sub/Mul selected-body
   artifacts and selected-boundary negative routes.
6. Run bounded provider scan for remaining local `bindOperand` /
   `requireOperandUse` logical operand tables in selected-body construction.
7. Run active-authority scan over touched RVV planning/provider/test files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-residual-operand-binding-closure/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative ordinary elementwise fixtures under `test/Target/RVV`

## Definition Of Done

The selected-body RVV provider has no production provider-local
logical-operand/materialized-use fallback table for selected-body operand
binding; ordinary Add/Sub/Mul construction consumes RVV-owned
elementwise/select facts or fails closed; focused and full checks pass; the
task is finished/archived using repo convention; and one coherent commit
records the work.

## Implementation Summary

- Removed the provider-local `getRequiredBinding`, `bindOperand`, and
  `requireOperandUse` fallback table from
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Rewired ordinary `Add` / `Sub` / `Mul` construction so `lhs`, `rhs`, and
  `out` are accepted only through
  `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts`.
- Made route construction fail closed when no RVV-owned operand-binding facts
  cluster provides runtime AVL/control facts.
- Extended the elementwise/select operand-binding facts consumer to cover the
  existing selected-body `RHSBroadcastLoad` Add path so that this legitimate
  selected-body broadcast-load route no longer depends on the provider-local
  fallback table.
- Added focused C++ coverage for stale ordinary elementwise `binary-lhs-call`
  materialized-use rejection and positive broadcast-load facts binding.

## Verification

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-provider-facts-only-operand-binding-closure`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(generic-selected-body-artifact-arithmetic|explicit-selected-body-artifact-(add|sub|mul)|pre-realized-selected-body-artifact-(add|sub|mul)|selected-boundary)' .`
- Bounded provider scan:
  `rg -n "bindOperand|requireOperandUse|getRequiredBinding|RouteOperandBindingPlan &bindingPlan|routeOperandBindingPlan" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  showed no local logical-operand binding table remains; the only hit is the
  shared route operand binding closure verification.
- Active-authority scan over touched RVV planning/provider/test diff found no
  newly introduced legacy i32, source-front-door, descriptor, direct-C,
  source-export, or provider-supported mirror authority. The only `status`
  hits were capability / selected-boundary mirror attributes inside the added
  MLIR fixture string, matching existing fixture style and not consumed as
  authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` (363/363)

## Self-Repair Notes

- The first focused C++ plugin test failed because a selected-body
  `RHSBroadcastLoad` Add route was still relying on the provider-local fallback
  and had no elementwise route-family plan. I did not preserve the fallback.
  Instead, I moved that existing selected-body route's operand binding into the
  RVV-owned elementwise/select facts boundary while leaving route-family
  expansion out of scope.

## Spec Update Judgment

No `.trellis/spec/` update was needed. The existing RVV plugin and EmitC route
specs already require plugin-owned facts before provider-built
`TCRVEmitCLowerableRoute` construction and common EmitC neutrality; this task
enforced that existing rule in code.

## Continuation Point

No continuation remains for the ordinary Add/Sub/Mul provider-local
operand-binding fallback closure. The provider is now facts-only for selected
body operand binding at the bounded scan surface covered by this task.
