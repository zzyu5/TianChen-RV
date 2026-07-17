# Stage2 RVV closure-gated reduction and accumulation coverage expansion

## Goal

Add one bounded Stage2 ordinary reduction/accumulation subcluster that is not
already covered by the archived reduction tasks: a computed-mask standalone
horizontal add-reduction route.

The new route computes:

```text
out_scalar_i32[0] = seed_i32 + sum(input_i32[i] where cmp_lhs_i32[i] <= cmp_rhs_i32[i])
```

It must start from a selected RVV variant containing typed low-level
`tcrv_rvv` body facts, realize any pre-realized form into explicit
compare/mask/reduction structure, derive route support through RVV
plugin-owned planning/provider code, enforce `RVVRouteOperandBindingPlan`
closure for materialized operands and mirrors, and prove correctness with real
`ssh rvv` evidence.

## Direction Source

- Direction title: `Stage2 RVV closure-gated reduction and accumulation coverage expansion`.
- Module owner: RVV plugin-owned route-supported expansion for generic typed
  horizontal reduction/accumulation routes on the corrected `tcrv_rvv` surface.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `83d86561 rvv: expand closure-gated compare select sle`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Active Inventory

Inventory source is current code and archived task PRDs, not status strings:

- Existing ordinary chunk-wise `reduce_add` route:
  - explicit and pre-realized fixtures exist;
  - route uses `tcrv_rvv.reduce`;
  - `RVVRouteOperandBindingPlan` ID is
    `rvv-route-operand-binding:reduce_add.v1`;
  - behavior reduces each dynamic-VL chunk and stores lane 0 at the chunk base.
- Existing `macc_add` route:
  - explicit and pre-realized fixtures exist;
  - route uses `tcrv_rvv.macc`;
  - `RVVRouteOperandBindingPlan` ID is
    `rvv-route-operand-binding:macc_add.v1`;
  - behavior is vector multiply-accumulate with explicit accumulator input.
- Existing standalone scalar-output `standalone_reduce_add` route:
  - pre-realized fixture and generated-bundle evidence exist;
  - route uses `tcrv_rvv.standalone_reduce`;
  - `RVVRouteOperandBindingPlan` ID is
    `rvv-route-operand-binding:standalone_reduce_add.v1`;
  - behavior accumulates across runtime `n` into `out[0]`.
- Existing widening/dot reduction routes:
  - widening macc, widening dot, strided-input widening dot, computed-mask
    widening dot, and computed-mask strided-input widening dot routes exist;
  - those belong to contraction-supporting widening/dot coverage, not ordinary
    standalone scalar reduction.
- Existing compare/select `sle` support from HEAD can be reused as a typed
  compare predicate fact, but this task is not another predicate expansion.

The remaining bounded increment for this brief is therefore:

```text
computed mask + ordinary source vector + scalar seed + scalar output
  -> masked standalone_reduce_add
```

This is distinct from existing standalone reduction because it must structurally
carry a compare-produced mask and prove inactive source lanes are ignored.

## Requirements

1. Add or repair one coherent ordinary computed-mask standalone reduction
   route:
   - compare lhs/rhs input buffers;
   - source vector input buffer;
   - scalar accumulator seed input;
   - scalar output buffer;
   - runtime `n`/AVL;
   - SEW32, LMUL m1, policy, compare predicate, mask source/form, reduction
     kind, accumulator layout, result layout, and inactive-lane zeroing facts.
2. The selected body must use generic typed `tcrv_rvv` structure. It must not
   use `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1` route
   IDs, descriptor/direct-C/source-front-door authority, or artifact-name
   authority.
3. If a pre-realized body is introduced, RVV selected-body realization must
   consume it into explicit `setvl`, `with_vl`, compare loads, source load,
   compare, masked standalone reduction, and scalar-output store structure
   before route planning.
4. RVV route planning must derive route support, ABI order, typed compute op,
   target leaf profile, provider support mirror, header leaves, compare leaf,
   masked merge/zeroing facts, reduction leaf, accumulator/result layouts, and
   metadata mirrors from typed body/config/runtime facts.
5. `RVVRouteOperandBindingPlan` must bind all materialized operands and mirrors:
   `cmp_lhs`, `cmp_rhs`, `src`, `acc`, `out`, and `n`.
6. Provider emission must use only the binding plan for materialized compare,
   source load, mask, zero-inactive-lane, accumulator seed, scalar output state,
   store, setvl, loop-control, and header operands.
7. Unsupported or malformed cases must fail closed with targeted diagnostics:
   unsupported reduction kind, missing source vector, missing accumulator seed,
   missing scalar output, wrong runtime role, vector/scalar role swaps,
   compare/source role swaps, mask/source mismatches, dtype/config mismatches,
   stale route or mirror authority, materialized-use mismatch, route-id/helper
   fallback, descriptor/direct-C/source-front-door authority, and common/export
   semantic inference.
8. Common EmitC/export may carry provider-built payloads and mirrors only. It
   must not infer reduction kind, mask semantics, dtype/config, policy,
   intrinsic choice, or ABI roles.

## Acceptance Criteria

- [ ] PRD records current reduction/accumulation route inventory and why the
      bounded increment is computed-mask standalone reduction.
- [ ] `tcrv_rvv` has a generic typed body surface for the masked standalone
      reduction, without dtype-prefixed helper ops.
- [ ] Pre-realized selected-body realization, if present, materializes explicit
      typed `tcrv_rvv` compare/mask/source/reduction/store structure.
- [ ] Route planning/provider derive route support, route metadata, headers,
      materialized operands, and mirrors from typed body/config/runtime facts
      under `RVVRouteOperandBindingPlan` closure.
- [ ] Positive structural tests prove explicit and/or pre-realized routes carry
      compare predicate, mask source/form, source vector, accumulator seed,
      scalar result, runtime `n`/AVL, binding plan ID, binding summary, and
      header mirrors.
- [ ] Negative fail-closed tests cover unsupported reduction kind, missing
      source vector, missing accumulator seed, missing scalar output, runtime
      role mismatch, compare/source role swap, mask/vector mismatch,
      dtype/config mismatch, stale route/mirror authority, materialized-use
      mismatch, descriptor/direct-C/source-front-door authority, and no
      common/export semantic inference where current test surfaces can express
      those failures.
- [ ] Generated-bundle dry-runs pass for representative computed-mask
      standalone reduction routes with counts `7,16,23`, value-distinguishing
      `sle` masks, mixed positive/negative source data, and at least two seed
      values.
- [ ] Real `ssh rvv` generated-bundle runs pass for the same representative
      route(s), proving seed use, active-lane accumulation, inactive-lane
      skipping, runtime `n`/AVL variation, scalar output correctness, and tail
      sentinel preservation.
- [ ] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, artifact-name authority, or common/export RVV
      semantic authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task finish/
      archive, clean git status, and one coherent commit complete if the task
      finishes.

## Non-Goals

- No new compare/select predicates beyond using already supported `sle` as a
  typed predicate fact.
- No dtype/LMUL clone batch, min/max/and/or/floating-point reductions,
  segmented reductions, broad contraction/matmul expansion, high-level
  Linalg/Vector/StableHLO frontend lowering, source-front-door positive route,
  global autotuning, dashboards, or report-only work.
- No rewrite of already completed `reduce_add`, `macc_add`,
  `standalone_reduce_add`, widening macc, or widening dot routes except for
  shared helper changes required by the new bounded route.
- No compatibility wrapper preserving legacy i32 route authority.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, route planning, lowering, or emission.

## Validation Plan

1. Validate and start this Trellis task.
2. Build focused targets needed for RVV dialect/plugin/target/script checks.
3. Run focused dialect verifier tests for new typed ops and negative cases.
4. Run focused target FileCheck tests for selected-body realization, route
   plan, generated header, and artifact metadata.
5. Run focused RVV plugin C++ tests for route operand binding plan validation
   and provider fail-closed materialized-use checks.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
7. Run generated-bundle dry-runs for the new route at counts `7,16,23` and
   multiple seed values.
8. Run real `ssh rvv` generated-bundle correctness for the new route.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Dialect/RVV/`
- `test/Target/RVV/`
- `test/Plugin/RVVExtensionPluginTest.cpp`

## Definition Of Done

- One computed-mask standalone scalar-output add-reduction route is implemented
  on the typed RVV selected-body/provider path.
- Route-supported evidence and executable `ssh rvv` evidence are current to
  this task.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
