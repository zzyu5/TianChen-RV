# Stage2 RVV elementwise arithmetic statement plan ownership

## Goal

Introduce an RVV-owned elementwise arithmetic route statement-plan boundary and
rewire the selected-body RVV EmitC route provider to consume it for the bounded
production-active elementwise arithmetic routes in this round: ordinary
`Add`/`Sub`/`Mul`, scalar-broadcast `Add`, masked `Add`/`Sub`/`Mul`, and
strided `Add`.

The provider remains the owner that instantiates `TCRVEmitCLowerableRoute`, adds
neutral route headers, type mappings, ABI mappings, and source-boundary
provenance. The provider must not locally recreate the included elementwise
arithmetic statement sequence from operation names, memory-form branches, ABI
strings, or intrinsic mirrors.

## Direction Source

- Direction title: `Stage2 RVV elementwise arithmetic route statement-plan ownership`.
- Module owner: RVV plugin-local selected-body route statement-plan boundary for
  elementwise arithmetic routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `17b6c1cf rvv: close provider operand binding fallback`.
- `.trellis/.current-task` was absent, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV-owned legality/materialization/operand
  binding/route planning -> provider-built `TCRVEmitCLowerableRoute` -> common
  EmitC materialization.
- Common EmitC/export must stay neutral: no RVV operation, dtype, SEW/LMUL,
  intrinsic, ABI, or schedule choices in common code.
- Commit `17b6c1cf` closed the remaining provider-local operand-binding
  fallback. Current provider construction already obtains materialization
  facts plus elementwise/select, memory, math, and residual operand-binding
  facts before statement construction.
- Current code in `RVVEmitCRouteProvider.cpp` still assembles the included
  elementwise arithmetic route statement sequence locally via
  `addCallStepFromSource`, provider-local `addLoopStep`, operation/memory-form
  branches, hand-built address expressions, and direct callee/operand/result
  construction.
- Existing planning surfaces already own route-family plans and operand-binding
  facts for ordinary elementwise arithmetic, scalar-broadcast elementwise,
  masked elementwise arithmetic, and strided add. This round should build on
  those facts rather than adding route coverage.

## Requirements

1. Add or regularize an RVV planning API for an elementwise arithmetic route
   statement plan. The plan must be derived from verified
   `RVVSelectedBodyRouteAnalysis`, `RVVSelectedBodyRouteMaterializationFacts`,
   and the relevant RVV-owned operand-binding facts.
2. The statement plan must cover ordinary `Add`/`Sub`/`Mul`, scalar-broadcast
   `Add`, masked `Add`/`Sub`/`Mul`, and strided `Add` where those routes are
   already production-active.
3. The plan must carry source operation provenance, callee/intrinsic names,
   operands, results, full-chunk `setvl`, loop/VL placement, address
   expressions, compute/merge steps, and store steps needed by the included
   route sequence.
4. `RVVEmitCRouteProvider` may still instantiate
   `TCRVEmitCLowerableRoute`, add neutral route-level metadata, and attach the
   returned statements, but it must not locally recreate the included
   elementwise arithmetic statement sequence from operation names, memory forms,
   ABI strings, or intrinsic mirrors.
5. Missing or stale dependencies for the statement plan must fail closed before
   common EmitC materialization with targeted diagnostics.
6. Preserve emitted semantics, ABI order, operand order, VL/control behavior,
   source provenance, intrinsic spelling, route ids, and generated artifacts
   for valid included elementwise arithmetic selected-body routes.
7. Do not add route coverage, new operations, dtype/LMUL clone batches,
   reductions, compare/select expansion, memory route rewrites,
   source-front-door routes, high-level frontend lowering, legacy i32
   authority, descriptor/direct-C/source-export paths, dashboards, broad smoke
   matrices, or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and the previous provider operand-binding
      closure task context.
- [x] A production RVV planning API exists for the elementwise arithmetic
      statement plan and exposes an empty/default plan for unrelated route
      families.
- [x] Positive C++ plugin/provider tests prove statement-plan construction for
      ordinary `Add`/`Sub`/`Mul`, scalar-broadcast `Add`, masked arithmetic, and
      strided `Add`.
- [x] At least one focused C++ negative test proves missing/stale statement-plan
      dependencies fail closed before route statement construction.
- [x] `RVVEmitCRouteProvider.cpp` consumes the returned plan for the included
      elementwise arithmetic routes before the older generic provider-local
      statement assembly path.
- [x] Bounded provider scan shows the included elementwise arithmetic statement
      sequence is no longer locally assembled by the provider beyond neutral
      route instantiation and attaching the RVV-owned plan statements.
- [x] Representative FileCheck coverage for existing explicit, pre-realized,
      and/or generic selected-body elementwise arithmetic artifacts still
      passes.
- [x] Selected-boundary negative coverage still passes.
- [x] Active-authority scan over touched RVV planning/provider/test files finds
      no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/source-export, or
      mirror-only route authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No route-family expansion and no new route operation coverage.
- No scalar, IME, Offload, TensorExt, future plugin, source-front-door, or
  high-level frontend work.
- No changes to emitted target sequence, ABI, runtime correctness, or
  performance claims. `ssh rvv` evidence is not required unless those claims
  change.
- No movement of RVV semantics into common EmitC/export.
- No compatibility wrapper that preserves old i32m1 route authority.

## Validation Plan

1. Validate this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build `tcrv-opt` and `tcrv-translate` if needed for focused lit coverage.
5. Run focused lit/FileCheck coverage for representative selected-body
   elementwise arithmetic artifacts and selected-boundary negative fixtures.
6. Run bounded provider scan for included elementwise arithmetic statement
   assembly residue in `RVVEmitCRouteProvider.cpp`.
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
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-provider-facts-only-operand-binding-closure/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Representative selected-body elementwise arithmetic fixtures under
  `test/Target/RVV`

## Definition Of Done

The bounded elementwise arithmetic statement sequence is owned by RVV route
planning, consumed by the selected-body provider before common EmitC route
construction, covered by focused positive and fail-closed C++ tests plus
representative FileCheck coverage, checked for authority drift, archived using
repo convention, and committed as one coherent change.

## Implementation Summary

- Added `RVVSelectedBodyElementwiseArithmeticRouteStatementPlan` and
  `getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan()` as the
  RVV-local statement-plan boundary for ordinary Add/Sub/Mul,
  scalar-broadcast Add, masked Add/Sub/Mul, and strided Add.
- Extended route materialization facts with a strided-store leaf so the
  statement plan can own the strided Add store step without asking the provider
  to rebuild that sequence.
- Rewired `RVVEmitCRouteProvider.cpp` to request the elementwise arithmetic
  statement plan after provider verification, materialization facts,
  elementwise/select operand-binding facts, and residual operand-binding facts.
  When the plan is present, the provider attaches its pre-loop `setvl` step and
  loop and returns before the older generic statement assembly path.
- Added focused C++ coverage for positive statement-plan construction and
  provider route consumption across ordinary Add/Sub/Mul, scalar-broadcast Add,
  masked Add/Sub/Mul, and strided Add; unrelated route empty-plan behavior; and
  a missing verified family-plan dependency diagnostic before statement
  construction.
- Documented the durable elementwise arithmetic statement-plan boundary in
  `.trellis/spec/extension-plugins/rvv-plugin.md`.

## Verification

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-05-24-stage2-rvv-elementwise-arithmetic-statement-plan-ownership`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- From `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='(generic-selected-body-artifact-arithmetic|explicit-selected-body-artifact-(add|sub|mul|scalar-broadcast-add|masked-(add|sub|mul)|strided-add)|pre-realized-selected-body-artifact-(add|sub|mul|scalar-broadcast-add|masked-(add|sub|mul)|strided-add)|selected-boundary)' .`
  passed 21/21 focused tests.
- Bounded provider scan:
  `rg -n "getRVVSelectedBodyElementwiseArithmeticRouteStatementPlan|plansElementwiseArithmeticRoute|addCallStepFromSource|auto addLoopStep|RVVSelectedBodyOperationKind::(Add|Sub|Mul|ScalarBroadcastAdd|MaskedAdd|MaskedSub|MaskedMul|StridedAdd)|RHSScalarBroadcast|RHSBroadcastLoad|StridedLoadStore" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  showed the new statement-plan call and early return before the older generic
  `addCallStepFromSource` / `addLoopStep` assembly. Remaining operation and
  memory-form references are in the generic provider path for other route
  families or in dead fallback code reached only when the statement plan is not
  a consumer.
- Active-authority scan over touched RVV planning/provider/test diff found no
  newly introduced `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/source-export, or
  provider-supported mirror authority. Added `status` hits are MLIR fixture
  selected-boundary/capability mirror attrs; added exact intrinsic hits are
  expected RVV leaf callees derived from typed/provider facts, not route
  authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` (363/363)

## Self-Repair Notes

- The first continuation build failed because the new statement-plan API took a
  `const RVVSelectedBodyRouteAnalysis &` but needed operation pointers for
  source provenance. The API now takes a mutable analysis reference, matching
  provider construction use.
- The first continuation build also missed the `tcrv::rvv::` namespace on the
  remaining AVL helper; the statement plan now uses the same qualified helper
  call as the provider.
- Strided Add should not require unit-stride vector load/store leaves. The
  statement plan now requires unit-stride leaves only for non-strided routes
  and requires strided load/store leaves for strided Add.

## Spec Update Judgment

`.trellis/spec/extension-plugins/rvv-plugin.md` was updated because this task
introduced a durable planning/provider API and boundary contract. The update
records scope, signature, contracts, fail-closed behavior, good/base/bad cases,
required tests, and wrong-vs-correct provider ownership shape.

## Continuation Point

No continuation remains for the bounded elementwise arithmetic statement-plan
ownership slice. The included elementwise arithmetic statement sequence is now
owned by RVV planning and consumed by the selected-body provider before common
EmitC route materialization.
