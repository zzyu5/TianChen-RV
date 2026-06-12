# Stage2 RVV computed-mask widening dot-reduce boundary

## Goal

Close one representative executable Stage2
`computed_masked_widening_dot_reduce_add` contraction boundary from a selected
`tcrv.exec` RVV variant with an explicit or pre-realized typed
`tcrv_rvv` body through RVV-owned compare-mask, signed widening dot-product,
reduction, accumulator, runtime, ABI, and artifact validation, provider-built
`TCRVEmitCLowerableRoute`, neutral common EmitC materialization, generated RVV
artifact/harness, and real `ssh rvv` correctness evidence.

The selected route is:

```text
cmp_lhs[i], cmp_rhs[i]: signed i16 / LMUL mf2
  -> compare-produced predicate mask in the same runtime VL scope
lhs[i], rhs[i]: signed i16 / LMUL mf2
  + acc[0]: signed i32 scalar seed
  + n: runtime AVL
  -> runtime VL from n / remaining AVL
  -> unit-stride compare and dot source loads
  -> signed i16xi16 widening multiply under the compare-produced mask
  -> inactive lanes excluded from the product/reduction
  -> masked widening dot-reduction into i32 accumulator
  -> scalar result stored to out[0]: i32
```

This task is a bounded computed-mask contraction runtime/artifact closure. It
is not a strided-input expansion, not a computed-mask strided-input expansion,
not widening MAcc, not scalar-broadcast MAcc, not runtime-scalar computed-mask
MAcc, not a dtype/LMUL clone batch, and not a high-level frontend route.

## What I Already Know

- Current HEAD is `d1bd073b rvv: validate widening dot-reduce contraction boundary`.
- The repository started this task with a clean worktree and no current
  Trellis task file.
- The immediately previous archived task closed the signed unit-stride
  `widening_dot_reduce_add` boundary with generated-bundle and `ssh rvv`
  evidence and reported that existing provider logic already covered the plain
  contraction path.
- Existing Stage2 contraction infrastructure names
  `computed_masked_widening_dot_reduce_add` alongside plain and strided
  widening dot-reduction routes. This task selects only the non-strided
  computed-mask representative.
- The RVV plugin spec requires direct-provider contraction routes to be
  selected exactly once by an RVV plugin-owned direct contraction
  route-provider owner after route-family plan, materialization facts, math
  operand-binding facts, and route-control provider-plan validation.
- The computed-mask route must prove that compare-produced mask facts coexist
  with signed widening dot operands, scalar i32 accumulator seed/layout,
  scalar i32 output/result layout, runtime n/AVL/VL, unit-stride memory form,
  ABI roles, selected capability, provider-supported mirrors, and generated
  artifact ABI under RVV provider authority.
- Authority must remain structural and provider-owned: typed or realized
  `tcrv_rvv` body/config/runtime facts and RVV provider plans own operation
  kind, predicate kind, mask source/role, signed i16-by-i16 widening
  dot-product relation, source and accumulator dtypes, SEW/LMUL, memory form,
  runtime AVL/VL, ABI roles, intrinsic/header facts, route support, and
  fail-closed diagnostics. Route ids, operation names, artifact names,
  metadata, scripts, ABI strings, common EmitC, source-front-door paths, and
  legacy i32 helper names remain mirrors or negative inventory only.

## Scope

- Close the representative signed unit-stride
  `computed_masked_widening_dot_reduce_add` boundary only.
- Start with bounded evidence of the active C++ owner path. If existing
  provider logic already satisfies the route, cite exact
  planning/provider/realization consumers and keep implementation focused on
  the missing evidence or validation gaps.
- Preserve and verify provider-owned derivation of operation kind, predicate
  kind, compare-produced mask in the same VL scope, mask role/source, signed
  i16mf2-by-i16mf2 widening dot-product relation, accumulator i32 seed lane0
  layout, scalar i32 output/result layout, computed-mask unit-stride memory
  form, `cmp_lhs/cmp_rhs/lhs/rhs/acc/out/n` ABI roles, runtime n/AVL,
  VL/tail/mask policy, selected capability, contraction route-family plan,
  operand bindings, intrinsic/type/header facts, and provider-supported
  mirrors before route construction.
- Include explicit and pre-realized selected-body artifact evidence so the
  pre-realized body is consumed into realized typed `tcrv_rvv` structure before
  provider route construction.
- Include non-dry-run `ssh rvv` generated-bundle execution over `n = 0`, small,
  and tail counts with signed masked widening dot-reduce correctness,
  accumulator seed contribution, scalar `out[0]` result, predicate-controlled
  exclusion, inactive-lane exclusion, and sentinel preservation.
- Keep common EmitC/export neutral; common code may consume provider-built
  routes and mirrors after construction, but must not infer RVV dot semantics,
  predicate semantics, dtype/config, ABI, policy, intrinsic, scalar-result
  layout, or support state.

## Out of Scope

- Strided-input widening dot-reduce, computed-mask strided-input widening
  dot-reduce, widening MAcc, scalar-broadcast MAcc, runtime-scalar
  computed-mask MAcc, conversion dtype/LMUL clone batches, high-level
  frontend/Linalg routes, source-front-door positive routes, dashboards,
  reports-only work, broad smoke matrices, or spec/test-only churn.
- New dtype-prefixed helper op families such as `tcrv_rvv.i32_*`,
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`,
  `tcrv_rvv.i32_macc`, or one-op-per-intrinsic wrappers.
- Any descriptor-derived, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, operation-name-derived,
  common-EmitC-derived, source-front-door-derived, central ad hoc,
  metadata-only, or legacy-i32 route authority.
- Weakening the existing plain widening dot-reduce, widening conversion, MAcc,
  computed-mask MAcc, or standalone reduction boundaries.

## Requirements

- `computed_masked_widening_dot_reduce_add` must be derived from typed
  `tcrv_rvv` body/config/runtime facts and selected capability facts.
- The RVV provider path must validate operation kind, predicate kind,
  compare-produced mask source and same-VL scope, signed i16-by-i16 widening
  dot-product relation, source vector dtype/LMUL, accumulator dtype/layout and
  seed-from-acc role, scalar output/result layout, unit-stride computed-mask
  memory form, `cmp_lhs/cmp_rhs/lhs/rhs/acc/out/n` ABI roles, runtime AVL/VL
  control, tail/mask policy, selected capability, contraction route-family
  plan, route-control provider plan, operand bindings, intrinsic/type/header
  facts, and provider-supported mirrors before route construction.
- The direct contraction owner must consume same-analysis route-family,
  materialization, route-control, runtime, capability, and math
  operand-binding facts before constructing provider-ready compare, mask,
  masked widening product, merge/exclusion, dot-reduction, seed, and scalar
  store statements.
- Generated harness correctness must distinguish true masked signed widening
  multiply-plus-reduce from unmasked reduction, wrong predicate, add-only,
  mul-only/no-seed, non-widening, wrong sign-extension, vector-store-to-output,
  inactive-lane contribution, metadata-derived behavior, and tail clobbering.
- Artifact metadata and evidence JSON may mirror provider facts only after
  route construction and must label mirror roles where relevant.
- Focused fail-closed coverage must remain in provider/C++ or lit tests for
  stale or missing contraction route-family facts, compare-mask facts,
  runtime/control facts, policy/capability facts, materialized leaves, and math
  operand-binding facts.

## Acceptance Criteria

- [ ] Bounded code inspection identifies the active C++ owner path for
  `computed_masked_widening_dot_reduce_add` from selected body realization
  through route-family analysis, materialization facts, math operand-binding
  facts, route-control provider plan, direct contraction owner, provider-built
  route, common EmitC materialization, and target artifact mirrors.
- [ ] Provider-owned route-family, route-control, direct-contraction owner,
  materialization, math-binding, target-leaf/profile, and route-consumption
  checks prove the `computed_masked_widening_dot_reduce_add` operation,
  predicate kind, compare-mask source/same-VL scope, signed
  i16mf2xi16mf2-to-i32m1 dot-reduction relation, accumulator/result layout,
  runtime `cmp_lhs/cmp_rhs/lhs/rhs/acc/out/n` ABI roles, runtime AVL/VL,
  policy, selected capability, operand bindings, intrinsic/type/header facts,
  and provider-supported mirrors before common EmitC.
- [ ] Explicit selected-body artifact checks expose provider-owned
  compare/predicate/mask facts, source/destination/accumulator dtype, SEW,
  LMUL, dot-product relation, memory form, runtime ABI order,
  route-family plan, operand bindings, selected capability, accumulator seed
  layout, scalar result layout, and provider-supported mirror fields.
- [ ] Pre-realized selected-body artifact checks expose the same facts and
  prove the pre-realized body is consumed into realized typed `tcrv_rvv`
  structure before provider route construction.
- [ ] Generated-bundle dry-run evidence for explicit and pre-realized
  `computed_masked_widening_dot_reduce_add` includes runtime counts `0`, a
  small count, and a tail count; contraction/SEW/LMUL/mask policy evidence;
  materialized selected-body facts; emitted C/C++ compare-load, predicate,
  source-load, masked widening product, dot-reduction, scalar store, and
  runtime AVL/VL facts; mirror-only artifact metadata; and harness oracle
  checks for signed masked widening dot-reduce, seed contribution, scalar
  output, predicate-controlled exclusion, inactive-lane exclusion, and
  sentinel preservation.
- [ ] Non-dry-run `ssh rvv` generated-bundle execution passes for explicit and
  pre-realized `computed_masked_widening_dot_reduce_add` over counts including
  `0`, small, and tail cases, with signed masked widening dot-reduce
  correctness, accumulator seed contribution, scalar `out[0]` result,
  predicate-controlled exclusion, inactive-lane exclusion, and output/tail
  sentinel preservation.
- [ ] Focused non-regression checks pass for `widening_dot_reduce_add`,
  `widen_i32_to_i64`, `macc_add`, and `computed_masked_macc_add`.
- [ ] A bounded authority scan over touched RVV planning/provider/target/
  script/fixture/test files finds no executable claim depending on central
  ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived,
  operation-name-derived, common-EmitC-derived, source-front-door-derived,
  route-id-derived, or legacy-i32-derived authority.
- [ ] `git diff --check` passes. `check-tianchenrv` passes or the exact
  blocker is recorded. The task is finished/archived and one coherent commit
  is created if acceptance is met.

## Technical Approach

Inspect the existing computed-mask dot-reduce route first. If the current C++
owner path already validates the computed-mask contraction boundary, keep C++
changes narrow and strengthen the generated-bundle script, fixtures, and
focused tests to expose the provider-owned facts and executable oracle. If the
inspection finds provider validation gaps, fix them in the RVV planning/provider
or selected-body realization owner, not in common EmitC, artifacts, route ids,
ABI-string parsing, or harness-only logic.

## Decision (ADR-lite)

**Context**: The previous plain widening dot-reduce task closed the non-masked
unit-stride contraction boundary. The next missing executable representative is
the computed-mask form where compare-produced predicate facts must be validated
with widening dot-reduction and scalar accumulator/result facts.

**Decision**: Treat `computed_masked_widening_dot_reduce_add` as a direct
contraction owner route. Use the existing provider-owned route-family,
route-control, materialization, math operand-binding, and direct contraction
owner path when valid; add or repair only the missing production validation,
fixture evidence, harness oracle, and focused checks needed for the computed
mask boundary.

**Consequences**: This keeps Stage2 coverage advancing through the corrected
typed RVV provider surface and avoids broadening into strided-input,
dtype/LMUL clone, source-front-door, or common EmitC semantic work. If existing
provider logic is already complete, the implementation may be evidence-heavy,
but it must still prove the active C++ owner path and executable mask/dot
semantics rather than repeating a small dry-run-only closure.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-widening-dot-reduce-contraction-boundary/prd.md`,
  `task.json`, `implement.jsonl`, and `check.jsonl`.
- Primary code surfaces to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`,
  directly related `test/Scripts` dry-run coverage, and focused
  `test/Plugin` coverage.

## Open Questions

- None blocking. The Direction Brief, specs, previous archived task, and
  existing code surfaces identify a bounded module: signed unit-stride
  computed-mask widening dot-reduce runtime/artifact and `ssh rvv` evidence
  closure.
