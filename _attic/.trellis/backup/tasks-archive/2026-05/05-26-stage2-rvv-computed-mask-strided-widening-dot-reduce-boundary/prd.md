# Stage2 RVV computed-mask strided-input widening dot-reduce boundary

## Goal

Close one representative executable Stage2
`computed_masked_strided_input_widening_dot_reduce_add` boundary from a selected
`tcrv.exec` RVV variant with explicit or pre-realized typed `tcrv_rvv` body
through RVV-owned compare-mask, runtime-strided source memory, signed widening
dot-product, scalar reduction, accumulator seed/result, runtime ABI/VL policy,
provider-built `TCRVEmitCLowerableRoute`, neutral common EmitC materialization,
generated RVV artifact/harness, and real `ssh rvv` correctness evidence.

The selected route is:

```text
cmp_lhs[i], cmp_rhs[i]: signed i32 / LMUL m1
  -> slt compare-produced predicate mask in the same runtime VL scope
lhs[i * lhs_stride], rhs[i * rhs_stride]: signed i16 / LMUL mf2
  + acc[0]: signed i32 scalar seed
  + n: runtime AVL
  + lhs_stride/rhs_stride: runtime element strides
  -> runtime VL from n / remaining AVL
  -> unit-stride compare loads and element-strided dot source loads
  -> signed i16xi16 widening multiply under the compare-produced mask
  -> inactive lanes zeroed before reduction
  -> horizontal dot-reduction into i32 accumulator
  -> scalar result stored to out[0]: i32
```

This is a bounded runtime-strided computed-mask contraction closure. It is not a
stride matrix, unmasked strided-input cleanup, segment/interleave memory route,
widening MAcc route, scalar-broadcast MAcc route, high-level frontend route, or
source-front-door positive route.

## What I Already Know

- The repository started this round at clean `main`, HEAD
  `18d69089 rvv: validate computed-mask widening dot-reduce boundary`, with no
  `.trellis/.current-task`.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-computed-mask-widening-dot-reduce-boundary/`
  closed the unit-stride computed-mask widening dot-reduce boundary.
- Current code already contains many pieces for this exact strided boundary:
  explicit and pre-realized artifact fixtures, dry-run generated-bundle script
  tests, Python harness logic, route-planning/provider references, selected-body
  realization references, construction protocol entries, and C++ plugin tests.
- Existing artifact fixtures already expose `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,
  lhs_stride,rhs_stride`, `rvv-route-operand-binding:masked_strided_wdot.v1`,
  `computed-mask-strided-input-widening-dot-reduce`,
  `runtime_abi:lhs_stride`, `runtime_abi:rhs_stride`, compare-produced mask
  source, signed i16mf2-by-i16mf2-to-i32 relation, scalar accumulator/result
  layouts, `__riscv_vlse16_v_i16mf2`, and
  `__riscv_vwmul_vv_i32m1_m` mirror facts.
- The current dry-run script tests use counts `7,16,23`; the direction brief
  explicitly requires `n=0` evidence as well as small/exact/tail counts.
- The C++ direct contraction owner currently requires same-analysis contraction
  family facts, route-control provider plan, math operand-binding facts,
  compare/dot/acc/out/n/stride ABI facts, materialized leaves for setvl/store/
  contraction/source load, and for computed-mask strided dot routes compare,
  masked product, masked merge, scalar seed, and strided source load leaves
  before statement construction.

## Scope

- Prove and, where needed, repair the production owner path for
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Keep implementation focused on the missing module-level closure. If existing
  provider logic already satisfies a requirement, cite the exact planning,
  provider, selected-body realization, artifact, or test consumer and avoid
  churn.
- Preserve RVV provider authority for operation kind, predicate kind,
  compare-produced same-VL mask, mask role/source, signed i16mf2-by-i16mf2
  widening dot relation, accumulator i32 seed lane0 layout, scalar i32 result
  layout, runtime-strided lhs/rhs payload memory form, ABI roles, runtime
  `n`/AVL/VL, stride runtime values, VL/tail/mask policy, selected capability,
  contraction family plan, route-control plan, operand bindings,
  intrinsic/type/header facts, and provider-supported mirrors.
- Include explicit and pre-realized generated-bundle evidence. The pre-realized
  path must prove the typed pre-realized body is consumed into realized
  `tcrv_rvv` structure before route construction.
- Run real `ssh rvv` generated-bundle executions for explicit and pre-realized
  modes over counts including `0`, a small case, an exact/vector-friendly case,
  and a tail case.
- Preserve non-regression for recently closed RVV boundaries:
  `computed_masked_widening_dot_reduce_add`, `widening_dot_reduce_add`,
  `widen_i32_to_i64`, `macc_add`, and `computed_masked_macc_add`.

## Out of Scope

- Stride matrices, unmasked strided-input refactors beyond regression needs,
  segment/interleave memory movement, widening MAcc, scalar-broadcast MAcc,
  runtime-scalar computed-mask MAcc, high-level frontend/Linalg routes,
  source-front-door positive routes, dashboards, reports-only work, broad smoke
  matrices, or spec/test-only churn.
- New dtype-prefixed helper op families such as `tcrv_rvv.i32_*`,
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`,
  `tcrv_rvv.i32_macc`, or one-op-per-intrinsic wrappers.
- Any descriptor-derived, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, operation-name-derived,
  common-EmitC-derived, source-front-door-derived, central ad hoc,
  metadata-only, status-only, or legacy-i32 route authority.
- Weakening the already landed computed-mask unit-stride dot-reduce, plain
  widening dot-reduce, widening conversion, MAcc, computed-mask MAcc, standalone
  reduction, or memory movement boundaries.

## Requirements

- `computed_masked_strided_input_widening_dot_reduce_add` must be derived from
  typed or realized `tcrv_rvv` body/config/runtime facts and selected capability
  facts, not from names, metadata, scripts, ABI strings, or common EmitC logic.
- RVV provider planning must validate and mirror the route only after proving:
  operation kind, `slt` predicate kind, compare-produced mask in the same VL
  scope, mask role/source, unit-stride compare loads, strided dot source loads,
  signed i16mf2-by-i16mf2 widening product to i32m1, inactive-lane zeroing
  before reduction, scalar accumulator seed from `acc[0]`, scalar store to
  `out[0]`, runtime `n`/AVL/VL, `lhs_stride`/`rhs_stride` runtime ABI values,
  policy, selected capability, family/route-control plans, math operand
  bindings, intrinsic/type/header facts, and provider-supported mirrors.
- Generated harness correctness must distinguish the true signed masked
  strided widening multiply-plus-reduce from unit-stride fallback, unmasked
  reduction, wrong predicate, wrong stride use, add-only, mul-only/no-seed,
  non-widening arithmetic, wrong sign extension, vector-store-to-output,
  inactive-lane contribution, skipped-stride element contribution, metadata-only
  behavior, and tail clobbering.
- Artifact metadata and evidence JSON may mirror provider facts only after route
  construction and must remain visibly mirror/evidence data.
- Focused fail-closed or provider tests must cover the important ownership
  points already present or newly repaired: stride ABI roles, computed-mask
  facts, dot source dtypes/config, runtime/control facts, materialized leaves,
  math operand binding facts, and direct contraction owner consumption.

## Acceptance Criteria

- [x] Bounded source inspection identifies the active C++ owner path from
  selected-body realization through route-family analysis, materialization
  facts, math operand-binding facts, route-control provider plan, direct
  contraction owner, `RVVEmitCRouteProvider`, common EmitC materialization, and
  target artifact mirrors.
- [x] Focused provider/lit/C++ checks pass for provider-owned stride, mask,
  dtype/config, runtime, operand-binding, accumulator/result, route-family,
  route-control, direct-owner, and fail-closed diagnostics relevant to
  `computed_masked_strided_input_widening_dot_reduce_add`.
- [x] Explicit selected-body artifact checks expose provider-owned
  compare/predicate/mask facts, source/destination/accumulator dtype, SEW,
  LMUL, dot relation, strided memory form, runtime ABI order, stride ABI facts,
  route-family plan, operand bindings, selected capability, accumulator seed
  layout, scalar result layout, intrinsic/header facts, and provider-supported
  mirrors.
- [x] Pre-realized selected-body artifact checks expose the same facts and prove
  the pre-realized body is consumed into realized typed `tcrv_rvv` structure
  before provider route construction.
- [x] Generated-bundle dry-run evidence for explicit and pre-realized modes
  includes runtime counts `0`, small, exact, and tail cases; harness checks for
  signed masked strided dot-reduce, seed contribution, scalar output,
  predicate-controlled exclusion, inactive-lane exclusion, non-unit stride use,
  skipped source sentinel preservation, output sentinel preservation, and
  tail preservation.
- [x] Non-dry-run `ssh rvv` generated-bundle execution passes for explicit and
  pre-realized modes with the same counts and correctness oracle.
- [x] Focused non-regression checks pass for
  `computed_masked_widening_dot_reduce_add`, `widening_dot_reduce_add`,
  `widen_i32_to_i64`, `macc_add`, and `computed_masked_macc_add`.
- [x] A bounded authority scan over touched RVV planning/provider/target/script/
  fixture/test files finds no executable claim depending on central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, operation-name-derived,
  common-EmitC-derived, source-front-door-derived, route-id-derived, or
  legacy-i32-derived authority.
- [x] `git diff --check` passes. `check-tianchenrv` passes, or the exact blocker
  is recorded. The task is finished/archived and one coherent commit is created
  if acceptance is met.

## Technical Approach

Start from the existing exact route surfaces instead of adding a parallel path.
First run focused artifact and script checks to reveal the live gap, especially
the missing `n=0` dry-run/ssh evidence. If the provider path already validates
the route, keep source changes in tests/harness/evidence assertions. If a real
provider validation gap appears, fix it in RVV planning/provider or selected-body
realization, not in common EmitC, target artifact metadata, route ids, ABI
string parsing, or script-only logic.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-computed-mask-widening-dot-reduce-boundary/prd.md`,
  `implement.jsonl`, and `check.jsonl`.
- Primary code/test surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  the matching `test/Scripts` dry-run tests, and related `test/Plugin` coverage.

## Completion Evidence

- Production code change was not needed: existing RVV provider path already
  consumed the direct contraction owner for this route. The implementation
  change strengthens the explicit and pre-realized generated-bundle dry-run lit
  tests so `n=0` is part of checked evidence, alongside `7`, `16`, and `23`.
- Active C++ owner path inspected:
  `RVVSelectedBodyRouteAnalysis` captures `lhsStrideABI`/`rhsStrideABI`;
  contraction family derivation marks computed-mask + strided-input dot-reduce;
  math operand-binding facts bind `cmp_lhs`, `cmp_rhs`, `dot_lhs`, `dot_rhs`,
  `acc`, `out`, `n`, `lhs_stride`, and `rhs_stride`;
  `buildDirectContractionRouteStatementPlan` requires same-analysis
  route-family/materialization/route-control/math facts and emits compare,
  strided dot source loads, masked widening product, masked merge, scalar seed,
  horizontal reduction, and scalar store steps before
  `RVVEmitCRouteProvider` attaches the plan to `TCRVEmitCLowerableRoute`.
- Explicit `ssh rvv` evidence:
  `rvv_generated_bundle_abi_e2e: success`, artifact root
  `artifacts/tmp/stage2-msdot-ssh-explicit/explicit`, and
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,7,16,23 lhs_stride=2 rhs_stride=3`.
- Pre-realized `ssh rvv` evidence:
  `rvv_generated_bundle_abi_e2e: success`, artifact root
  `artifacts/tmp/stage2-msdot-ssh-pre/pre`, and
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,7,16,23 lhs_stride=2 rhs_stride=3`.
- Both `ssh rvv` runs reported per-case success for `n=0`, `7`, `16`, and
  `23` with
  `compare_masked_strided_signed_horizontal_dot seed_added inactive_lanes_skipped source_strides=2,3 skipped_source_elements_ignored scalar_output tail_preserved`.
- Focused checks:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-strided-input-widening-dot-reduce'`
  passed 4/4 selected tests;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-widening-dot-reduce|widening-dot-reduce|widen-i32-to-i64|macc-add|computed-masked-macc'`
  passed 40/40 selected tests;
  `./build/bin/tianchenrv-rvv-extension-plugin-test` passed;
  `git diff --check` passed;
  `cmake --build build --target check-tianchenrv -j2` passed 381/381.
- Authority scan:
  touched dry-run tests keep `descriptor`, `direct-C`, `source-export`, and
  `tcrv_rvv.i32_` only in negative `implicit-check-not` assertions. Existing
  planning code contains a fail-closed `tcrv_rvv.i32_` rejection check; no
  positive executable claim depends on legacy i32/source-front-door/descriptor/
  ABI-string/script/artifact-name/common-EmitC/metadata/route-id authority.

## Spec Update Decision

No `.trellis/spec/` update is needed for this round. The only durable lesson
was to include `n=0` in executable RVV generated-bundle evidence for masked
runtime-count routes, and `.trellis/spec/testing/mlir-testing-contract.md`
already states that runtime RVV claims must include `ssh rvv` output over
multiple runtime counts including `n=0`, at least one tail case, and mixed
active/inactive mask cases. This task implements that existing contract for the
computed-mask strided-input widening dot-reduce dry-run fixtures.
