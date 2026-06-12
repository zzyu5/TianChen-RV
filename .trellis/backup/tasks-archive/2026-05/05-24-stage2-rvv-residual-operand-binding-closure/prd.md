# Stage2 RVV residual operand-binding closure

## Goal

Close the remaining production RVV provider-local logical operand and
materialized-use binding branches for masked elementwise arithmetic, strided
elementwise add, and runtime scalar splat-store. The production
`RVVEmitCRouteProvider` should consume one RVV-owned residual operand-binding
facts boundary after route-family provider verification and materialization
facts selection, before building `TCRVEmitCLowerableRoute` statements.

## Direction Source

- Direction title: `Stage2 RVV residual operand-binding closure for masked/strided/splat routes`.
- Module owner: RVV plugin-local selected-body route operand-binding surface for
  masked elementwise arithmetic, strided elementwise add, and runtime scalar
  splat-store.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `64052c82 rvv: own math operand binding facts`.
- `.trellis/.current-task` was absent, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- Specs require RVV route authority to flow from selected `tcrv.exec` variant
  through typed low-level `tcrv_rvv` body/config/runtime facts, RVV-owned
  legality/realization/provider, provider-built `TCRVEmitCLowerableRoute`,
  common EmitC materialization, and target artifact/evidence.
- Common EmitC/export must stay neutral. RVV ABI mapping, logical operands,
  materialized uses, route payload, selected-body realization, and fail-closed
  diagnostics remain RVV-plugin-owned.
- Prior tasks added RVV-owned operand-binding facts for elementwise/select,
  memory movement, and math clusters.
- The archived math task explicitly left masked elementwise arithmetic, strided
  elementwise add, and runtime scalar splat-store outside its included owner.
- Current provider evidence still has local `bindOperand` and
  `requireOperandUse` tables for `MaskedAdd/Sub/Mul`, `StridedAdd`, and
  `RuntimeScalarSplatStore` before statement construction.

## Requirements

1. Add a named RVV planning/provider residual operand-binding facts boundary
   for the included route families.
2. The boundary must consume the verified `RouteOperandBindingPlan` and route
   description closure before statement construction.
3. The boundary must expose route-shape booleans for masked elementwise
   arithmetic, strided elementwise add, and runtime scalar splat-store.
4. The boundary must expose bound runtime ABI parameters for masked lhs/rhs/out,
   strided lhs/rhs/out plus lhs/rhs/out stride roles, runtime scalar splat
   scalar/out roles, and runtime element count.
5. The boundary must preserve plan id/order, runtime AVL/header closure,
   mask/compare roles, active arithmetic roles, inactive passthrough roles,
   stride address roles, scalar splat roles, load/compute/store materialized
   uses, and required header/loop-control mirrors.
6. The boundary must fail closed before statement construction for missing
   family plans, stale route-shape markers, missing logical operands, missing
   materialized uses, and stale runtime/header closure.
7. Rewire the production `RVVEmitCRouteProvider.cpp` residual binding branches
   to consume the new facts instead of locally enumerating logical operands and
   materialized-use strings.
8. Preserve emitted semantics, route ids, ABI order, target leaf/header facts,
   operand order, diagnostics, and generated artifacts. Do not add route
   coverage or new operations.
9. Keep the boundary RVV-local. Do not move RVV semantics into common
   EmitC/export or target metadata.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin, EmitC route, testing
      specs, and the previous math operand-binding task.
- [x] A named residual operand-binding facts boundary exists in the RVV
      planning API.
- [x] The production provider obtains residual operand-binding facts after
      provider-plan verification, materialization facts, elementwise/select
      facts, memory facts, and math facts.
- [x] The provider consumes residual facts for masked elementwise arithmetic,
      strided elementwise add, and runtime scalar splat-store.
- [x] The provider no longer locally enumerates the included residual cluster's
      logical operands and materialized-use strings in the central binding
      prelude.
- [x] Focused C++ plugin/provider tests cover masked elementwise arithmetic,
      strided elementwise add, and runtime scalar splat-store binding facts.
- [x] Focused negative coverage includes at least one missing or stale
      materialized-use diagnostic for the new residual binding facts boundary.
- [x] Representative FileCheck coverage for existing explicit/pre-realized
      residual artifact paths affected by the refactor still passes.
- [x] Selected-boundary negative coverage still passes.
- [x] A bounded scan over `RVVEmitCRouteProvider.cpp` shows no remaining
      production local `bindOperand` / `requireOperandUse` logical-operand table
      for the included residual paths.
- [x] Active-authority scan over touched RVV planning/provider/test files finds
      no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/source-export, or
      mirror-only route authority.
- [x] Focused build/tests, `git diff --check`, and `check-tianchenrv` pass, or
      an exact blocker is documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No route coverage expansion, new operations, dtype/LMUL clone batches,
  source-front-door routes, high-level frontend lowering, legacy i32 authority,
  descriptor/direct-C/source-export paths, artifact dashboards, broad smoke
  matrices, or statement-emission rewrites unrelated to consuming the residual
  boundary.
- No remigration of elementwise/select, memory, or math clusters except direct
  integration needed to call the residual facts boundary in the same provider
  prelude.
- No emitted target sequence, runtime ABI, operand order, correctness, runtime,
  or performance claim changes.
- No migration of RVV semantics into common EmitC/export.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed for focused lit coverage.
5. Run focused lit/FileCheck coverage for masked add, strided add, runtime
   scalar splat-store, and selected-boundary negative routes included in the
   refactor.
6. Run bounded provider scan for remaining local binding tables in the included
   residual paths.
7. Run active-authority scan over touched RVV planning/provider/test files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-math-operand-binding-surface-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- representative fixtures under `test/Target/RVV`

## Definition Of Done

The selected-body RVV provider consumes a named RVV-owned residual
operand-binding facts boundary for masked elementwise arithmetic, strided
elementwise add, and runtime scalar splat-store; central provider code no
longer owns the included residual logical operand/materialized-use decision
tables; representative route families and fail-closed diagnostics remain
covered; focused and full checks pass; the task is finished/archived using repo
convention; and one coherent commit records the work.

## Implementation Summary

- Added `RVVSelectedBodyResidualRouteOperandBindingFacts` and
  `getRVVSelectedBodyResidualRouteOperandBindingFacts(...)` to the RVV planning
  API.
- The new residual boundary verifies route operand binding closure, runtime
  AVL/header closure, elementwise arithmetic or runtime splat-store family plan
  presence, masked versus strided route markers, scalar splat-store memory
  form, active arithmetic uses, inactive passthrough uses, stride address uses,
  scalar splat roles, output store roles, and materialized header/loop mirrors
  before provider statement construction.
- Rewired `RVVEmitCRouteProvider.cpp` so masked elementwise arithmetic,
  strided elementwise add, and runtime scalar splat-store branches consume
  residual operand-binding facts instead of locally querying
  `RouteOperandBindingPlan`.
- Added focused C++ coverage for masked add, strided add, runtime scalar
  splat-store, and one missing masked passthrough materialized-use diagnostic.
- Added a durable residual operand-binding facts section to the RVV plugin spec.

## Verification

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-residual-operand-binding-closure`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(masked-(add|sub|mul)|strided-add|runtime-i32-splat-store|selected-boundary)' .`
- Bounded provider scan:
  `rg -n "MaskedArithmetic|MaskedAdd|MaskedSub|MaskedMul|StridedAdd|RuntimeScalarSplatStore|runtime scalar splat-store|strided_add|masked elementwise|bindOperand|requireOperandUse" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- Active-authority scan over touched RVV planning/provider/test diff for
  legacy i32/source-front-door/descriptor/direct-C/source-export and bare
  status/provider-supported drift.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

All focused and full checks passed. No emitted target sequence, ABI order,
correctness, runtime, or performance claim was changed, so `ssh rvv` evidence
was not required for this refactor.

## Self-Repair Notes

- The first focused lit command was run from the repository root against
  `build/test`; the generated lit site config expected its relative
  `../../test/lit.cfg.py` path from `build/test`. Re-running the same filtered
  lit invocation from `build/test` passed.

## Continuation Point

No continuation remains for the included residual operand-binding cluster. The
provider still keeps local fallback binding helpers for unrelated legacy/base
cases not included in this task, but the masked elementwise arithmetic,
strided elementwise add, and runtime scalar splat-store production branches no
longer own their logical operand/materialized-use tables locally.
