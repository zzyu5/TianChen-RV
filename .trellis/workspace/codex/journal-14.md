# Journal 14 - codex

> Continuation from `journal-13.md` (archived at ~2000 lines)

---

## Session 172: Stage2 RVV contraction runtime and binding closure

**Date**: 2026-05-23
**Task**: Stage2 RVV contraction route-family runtime and binding closure
**Branch**: `main`

### Summary

Closed contraction provider validation for active `widening_macc_add`,
`widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
`computed_masked_widening_dot_reduce_add`, and
`computed_masked_strided_input_widening_dot_reduce_add`.

### Main Changes

- Added contraction-owned `RVVRuntimeAVLVLControlPlan` derivation, description
  application, provider mirror validation, and `RouteOperandBindingPlan` closure.
- Added focused plugin tests for consumer isolation, missing/stale plans,
  runtime-control, mirror, ABI-order, and binding-closure mismatches.
- Updated contraction explicit/pre-realized target fixtures with runtime-control
  plan mirror checks.

### Testing

- [OK] RVV plugin C++ smoke test; focused contraction PLAN/HEADER FileChecks.
- [OK] Generated-bundle dry-runs and real `ssh rvv` runs for explicit and
  pre-realized contraction routes at counts `7,16,23`; extra evidence covers
  computed-mask plus strided-input contraction.
- [OK] Added-line authority scan, `git diff --check`, and `check-tianchenrv`
  361/361 passed.

### Status

[OK] **Completed and archived**. Commit is created after this journal entry.

---

## Session 175: Stage2 RVV scalar-broadcast elementwise runtime binding closure

**Date**: 2026-05-23
**Task**: Stage2 RVV scalar-broadcast elementwise runtime binding closure
**Branch**: `main`

### Summary

Closed scalar-broadcast elementwise provider runtime AVL/VL mirror validation
and `RouteOperandBindingPlan` closure for active explicit and pre-realized
`scalar_broadcast_add`, `scalar_broadcast_sub`, and `scalar_broadcast_mul`
routes.

### Main Changes

- Required
  `verifyRVVSelectedBodyScalarBroadcastElementwiseRouteFamilyProviderPlans` to
  compare full runtime AVL/VL control mirrors from the validated scalar
  broadcast family plan before provider materialization.
- Added family-provider `RouteOperandBindingPlan` closure validation so plan
  id, runtime ABI order, parameter mirrors, logical operand roles,
  materialized uses, and summary mirrors fail closed for scalar-broadcast
  routes.
- Added focused RVV plugin C++ coverage for consumer isolation,
  missing/stale plans, runtime-control mismatch, scalar RHS ABI and binding
  mismatch, intrinsic/type/result mirror mismatch, and add/sub/mul operation
  isolation.
- Updated explicit/pre-realized scalar-broadcast target fixtures to check
  runtime-control, runtime-VL, AVL-source, bounded-slice, binding, and family
  plan mirrors.
- Tightened generated-bundle scalar-broadcast runtime evidence so
  memory-writing routes check sentinel tail preservation across runtime `n`.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused materialization/header export for all six scalar-broadcast
  target fixtures: explicit add/sub/mul and pre-realized add/sub/mul.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  scalar-broadcast add/sub/mul, counts `7,16,23`, RHS scalars `5,-3`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  scalar-broadcast add/sub/mul, counts `7,16,23`, RHS scalars `5,-3`, with
  signed RHS behavior and `tail_preserved`.
- [OK] Added-line active-authority scan found no new legacy i32m1,
  source-front-door, source-artifact, descriptor, direct-C, or exact i32m1
  intrinsic authority matches; exact intrinsic additions are C++ mirror and
  stale-mirror tests only.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2`, 361/361 passed.

### Spec Update Judgment

Updated `.trellis/spec/testing/mlir-testing-contract.md` with a durable
evidence rule: RVV generated-bundle runtime evidence over runtime `n` for
memory-writing routes must check active lanes and guard/tail sentinel
preservation. This is evidence quality guidance only, not route authority.

### Status

[OK] **Completed and archived**. Commit is created after this journal entry.

---

## Session 173: Stage2 RVV widening conversion runtime and binding closure

**Date**: 2026-05-23
**Task**: Stage2 RVV widening conversion runtime binding closure
**Branch**: `main`

### Summary

Closed widening conversion provider validation for active `widen_i32_to_i64`
and `widen_i16_to_i32` routes.

### Main Changes

- Required widening conversion provider materialization to compare full
  runtime AVL/VL control mirrors from the validated family plan, not only the
  runtime control plan id.
- Added family-provider `RouteOperandBindingPlan` closure validation so plan id,
  runtime ABI order, parameter mirrors, logical operand roles, materialized
  uses, and summary mirrors fail closed for widening conversion routes.
- Added focused RVV plugin C++ coverage for consumer isolation, missing/stale
  plans, runtime-control mismatch, source/type/intrinsic/relation mirror
  mismatch, runtime ABI mismatch, binding role mismatch, binding summary
  mismatch, and both active conversion routes.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit from `build/test`: explicit `widen_i32_to_i64` and
  pre-realized `widen_i32_to_i64` / `widen_i16_to_i32` target artifact tests,
  3/3 passed.
- [OK] Generated-bundle dry-runs: explicit `widen_i32_to_i64`; pre-realized
  `widen_i32_to_i64` and `widen_i16_to_i32`; counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle runs for the same representative
  widening conversion routes and counts; `widen_i16_to_i32` reported
  `sign_extension_checked tail_preserved`.
- [OK] Added-line authority scan, `git diff --check`, and `check-tianchenrv`
  361/361 passed.

### Spec Update Judgment

No `.trellis/spec/` update is needed. The durable rule already exists in the
RVV plugin and unified EmitC route specs: plugin-owned typed body facts and
provider-built routes are authority, while route descriptions and artifacts are
mirrors. This round applies that existing contract to the widening conversion
family provider boundary.

### Status

[OK] **Completed and archived**. Commit is created after this journal entry.


## Session 174: Stage2 RVV plain compare-select runtime and binding closure

**Date**: 2026-05-23
**Task**: Stage2 RVV plain compare-select runtime and binding closure
**Branch**: `main`

### Summary

Closed plain `cmp_select` provider runtime AVL/VL mirror validation and
`RouteOperandBindingPlan` closure for active explicit and pre-realized
`cmp_select` / `cmp_select_sle` routes.

### Main Changes

- Required
  `verifyRVVSelectedBodyPlainCompareSelectRouteFamilyProviderPlans` to compare
  full runtime AVL/VL control mirrors from the validated plain compare-select
  family plan before provider materialization.
- Added family-provider `RouteOperandBindingPlan` closure validation so plan id,
  runtime ABI order, parameter mirrors, logical operand roles, materialized
  uses, and summary mirrors fail closed for plain compare-select routes.
- Added focused RVV plugin C++ coverage for consumer isolation, missing/stale
  plans, runtime-control mismatch, predicate/intrinsic/layout mirror mismatch,
  runtime ABI mismatch, binding role mismatch, binding summary mismatch, and
  `cmp_select_sle` signed predicate facts.
- Updated explicit/pre-realized target fixtures to check runtime-control,
  runtime-VL, AVL-source, bounded-slice, binding, and plain family-plan mirrors.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused materialization/header export for explicit and pre-realized
  `cmp_select` / `cmp_select_sle` target artifact fixtures.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `cmp_select` / `cmp_select_sle`, counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `cmp_select` / `cmp_select_sle`, counts `7,16,23`, with true/false lane
  distributions and signed `sle` predicate coverage.
- [OK] Added-line active-authority scan found no new legacy i32m1,
  source-front-door, source-artifact, descriptor, direct-C, or exact i32m1
  intrinsic authority matches.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2`, 361/361 passed.

### Spec Update Judgment

No `.trellis/spec/` update is needed. The RVV plugin and unified EmitC route
specs already require plugin-owned typed route facts, runtime/binding closure,
mirror-only metadata, common EmitC neutrality, and real `ssh rvv` evidence for
runtime/correctness claims. This round applies that existing contract to the
plain compare-select family provider boundary.

### Status

[OK] **Completed and archived**. Commit is created after this journal entry.


## Session 176: Stage2 RVV runtime scalar splat-store runtime binding closure

**Date**: 2026-05-23
**Task**: Stage2 RVV runtime scalar splat-store runtime binding closure
**Branch**: `main`

### Summary

Closed RuntimeI32SplatStore provider runtime AVL/VL mirror and RouteOperandBindingPlan closure; added focused provider tests and refreshed generated-bundle plus ssh rvv evidence.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
