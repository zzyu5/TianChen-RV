# Stage2 RVV Elementwise Compare-Select Selected-Body Owner Cleanup

## Goal

Move the elementwise/compare-select selected-body validation and realization
authority out of the central `RVVSelectedBodyRealization.cpp` materialization
branches and into owner-local production code adjacent to the elementwise
route-family planning owner.

The intended production chain is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv elementwise/compare-select body
  -> RVV owner-local selected-body validation and realization
  -> realized tcrv_rvv body with setvl/with_vl/load/strided_load/splat/
     compare/mask/select/binary/masked_binary/store/strided_store facts
  -> elementwise route-family plan owners
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materializer
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

The central selected-body realization file may retain the registry, shared
construction primitives, and neutral dispatch. It must not remain the
family-specific authority for elementwise arithmetic, masked arithmetic,
scalar-broadcast elementwise, compare/select, computed-mask select,
runtime-scalar compare/select, or runtime-scalar dual-compare mask-and-select
semantics.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- The worktree was clean before task creation and `HEAD` was
  `31c8d577 rvv: move segment2 selected-body realization owner-side`.
- The previous archived task
  `05-30-05-30-stage2-rvv-computed-mask-segment2-load-store-update-selected-body-realization-migration`
  completed the computed-mask segment2 selected-body owner-side migration and
  preserved the public selected lowering-boundary producer as the production
  path.
- The generated-bundle direct pre-realized route-entry shortcut is already
  retired for the recently completed segment2 family.
- Specs require RVV Stage2 selected bodies to start from selected
  `tcrv.exec` RVV variants plus typed low-level `tcrv_rvv` bodies; provider
  facts and target artifacts must consume realized structure and treat
  metadata as mirrors only.
- The task brief identifies these pre-realized bodies as in scope:
  `TypedBinaryPreRealizedBodyOp`, `TypedMaskedBinaryPreRealizedBodyOp`,
  `TypedCompareSelectPreRealizedBodyOp`,
  `TypedComputedMaskSelectPreRealizedBodyOp`,
  `TypedRuntimeScalarCompareSelectPreRealizedBodyOp`, and
  `TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp`.
- The expected owner-local code should be near
  `RVVEmitCElementwiseRouteFamilyPlanOwners`, not in a central ad hoc
  materialization branch.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
   lit / FileCheck. Python may only be used for tooling, probes, harnesses,
   and artifact parsing.
2. Preserve operation kind, predicate kind, mask source, select layout,
   scalar/runtime operands, memory form, strides, SEW, LMUL, policy, runtime
   `n`/AVL/VL, required capabilities, provider route facts, and artifact ABI
   order.
3. Move family-specific elementwise/compare-select realization validation and
   construction behind an owner-local component. Central code may call an
   owner registry or neutral dispatcher but must not encode the family
   semantics itself.
4. Keep the selected-body registry and common construction helpers neutral.
   They may classify and dispatch, but not derive missing RVV semantics from
   names, route ids, artifacts, ABI strings, exact intrinsic spelling,
   descriptors, scripts, or common EmitC.
5. Unsupported mismatches must fail closed with targeted diagnostics before
   route-family facts, provider route construction, common EmitC
   materialization, or target artifact export.
6. Preserve recently completed segment2 selected-body behavior and do not
   reopen segment2, contraction, MAcc, base memory, computed-mask memory,
   reduction, or conversion owners except for a narrow shared interface needed
   by this extraction.

## Acceptance Criteria

- [ ] Production code for elementwise/compare-select selected-body
      validation and realization lives in owner-local code outside the central
      materialization branch.
- [ ] Central selected-body realization dispatch remains thin and neutral,
      retaining only registry/shared helper responsibilities with a clear
      justification for anything left there.
- [ ] Owner-local validation rejects wrong body family, wrong op kind,
      predicate/layout mismatch, mask source mismatch, config/policy mismatch,
      missing runtime ABI binding, wrong AVL/VL relation, wrong memory form or
      stride binding, stale route id, mirror metadata misuse,
      direct-route-entry-only authority, exact-intrinsic-as-authority, and
      common-EmitC semantic invention.
- [ ] Representative selected-boundary dry-runs and generated artifacts pass
      for plain elementwise, masked elementwise, scalar-broadcast elementwise,
      compare/select, computed-mask select, and runtime-scalar compare/select
      with `route_entry_realization: false`.
- [ ] Focused `ssh rvv` correctness evidence covers a small representative
      subset that exercises elementwise, mask/select, and runtime-scalar ABI
      behavior, or records an exact infrastructure blocker.
- [ ] Recent segment2 selected-body paths have focused non-regression
      coverage.
- [ ] Bounded authority scan shows no route or executable claim for this
      family depends on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes, `check-tianchenrv` passes or an exact blocker
      is recorded, and final git status is clean after one coherent commit.

## Non-goals

- Do not add new RVV op families, dtype/LMUL clone batches, high-level
  Linalg/frontend lowering, source-front-door routes, one-intrinsic wrapper
  dialects, dashboards, broad smoke matrices, or artifact/index bookkeeping as
  the milestone.
- Do not resurrect direct route-entry support for pre-realized selected bodies.
- Do not rewrite completed segment2, contraction, MAcc, base memory,
  computed-mask memory, reduction, or conversion owners unless required by a
  narrow shared owner interface.
- Do not treat prompt edits, reports, helper-only changes, or broad tests as
  the main achievement.

## Validation Plan

1. Validate task context with `python3 ./.trellis/scripts/task.py validate`
   after PRD/context files are in place.
2. Inspect relevant specs and code around selected-body realization,
   elementwise route-family plan owners, generated bundle ABI flow, and focused
   tests.
3. Implement owner-local selected-body realization for the bounded
   elementwise/compare-select family cluster.
4. Run focused C++/lit/script tests covering selected-boundary realization,
   fail-closed diagnostics, generated-bundle evidence, and segment2
   non-regression.
5. Run a small focused `ssh rvv` correctness subset for elementwise,
   mask/select, and runtime-scalar ABI behavior if reachable.
6. Run bounded authority scans, `git diff --check`, and `check-tianchenrv` or
   record the exact blocker.
7. Finish/archive the Trellis task, record the workspace journal, and create
   one coherent commit if the task is complete.

## Files to Inspect First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-computed-mask-segment2-load-store-update-selected-body-realization-migration/`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- Directly related elementwise and compare/select generated-bundle tests.
