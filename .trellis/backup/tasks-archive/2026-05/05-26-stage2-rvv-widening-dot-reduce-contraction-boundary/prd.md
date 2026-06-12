# Stage2 RVV widening dot-reduce contraction boundary

## Goal

Close one representative Stage2 signed `widening_dot_reduce_add` executable
contraction boundary from a selected `tcrv.exec` RVV variant with typed
`tcrv_rvv.widening_dot_reduce` body through RVV-owned dot/reduction/
accumulator/runtime validation, provider-built `TCRVEmitCLowerableRoute`,
neutral common EmitC materialization, generated RVV artifact/harness, and real
`ssh rvv` correctness evidence.

The selected route is:

```text
lhs[i], rhs[i]: signed i16 / LMUL mf2
  + acc[0]: signed i32 scalar seed
  + n: runtime AVL
  -> runtime VL from n / remaining AVL
  -> unit-stride lhs/rhs loads
  -> signed i16xi16 widening multiply plus reduction into i32 accumulator
  -> scalar result stored to out[0]: i32
```

This task is a bounded contraction-supporting scalar-result/runtime artifact
closure. It is not a computed-mask widening dot-reduce expansion, not a
strided-input expansion, not widening MAcc, not a dtype/LMUL clone batch, and
not a high-level frontend route.

## What I already know

- Current HEAD is `215bfc5f rvv: validate widening conversion artifact boundary`.
- The repository started this task with a clean worktree and no current Trellis
  task file.
- The immediately previous archived task closed the signed `widen_i32_to_i64`
  conversion artifact boundary with provider-owned dtype/runtime facts, dry-run
  generated bundles, `ssh rvv` correctness over counts including `0`, MAcc
  non-regression, and full `check-tianchenrv`.
- Existing Stage2 contraction infrastructure already names
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`; this task selects
  only the non-masked unit-stride representative.
- Existing RVV plugin spec requires direct-provider contraction routes to be
  selected exactly once by an RVV plugin-owned direct contraction
  route-provider owner after route-family plan, materialization facts,
  math operand-binding facts, and route-control provider-plan validation.
- Existing script and fixture surfaces already include explicit and
  pre-realized `widening_dot_reduce_add` generated-bundle paths; this task must
  inspect whether they already prove signed widening multiply-plus-reduce,
  accumulator seed contribution, scalar out[0] layout, runtime n/AVL/VL, and
  sentinel preservation strongly enough.
- Authority must remain structural and provider-owned: typed
  `tcrv_rvv` body/config/runtime facts and RVV provider plans own operation
  kind, signed i16-by-i16 widening dot-product relation, source and accumulator
  dtypes, SEW/LMUL, memory form, runtime AVL/VL, ABI roles, intrinsic/header
  facts, route support, and fail-closed diagnostics. Route ids, artifact names,
  metadata, scripts, ABI strings, common EmitC, source-front-door paths, and
  legacy i32 helper names remain mirrors or negative inventory only.

## Scope

- Close the representative signed unit-stride `widening_dot_reduce_add`
  boundary only.
- Preserve existing provider-owned validation when it already satisfies the
  acceptance criteria; cite exact bounded code evidence rather than duplicating
  production logic.
- Strengthen generated-bundle dry-run evidence and harness oracle only where
  the current route does not prove the boundary sharply enough.
- Include explicit and pre-realized selected-body artifact evidence so the
  pre-realized body is consumed into realized typed `tcrv_rvv` structure before
  provider route construction.
- Include non-dry-run `ssh rvv` generated-bundle execution over `n = 0`, small,
  and tail counts with negative inputs, mixed magnitudes, accumulator seed
  contribution, scalar `out[0]` result, and output/tail sentinel preservation.
- Keep common EmitC/export neutral; common code may consume provider-built
  routes and mirrors after construction, but must not infer RVV dot semantics,
  dtype/config, ABI, policy, intrinsic, scalar-result layout, or support state.

## Out of Scope

- Computed-mask widening dot-reduce, strided-input widening dot-reduce,
  computed-mask strided-input widening dot-reduce, widening MAcc,
  scalar-broadcast MAcc, runtime-scalar computed-mask MAcc, conversion
  dtype/LMUL clone batches, high-level frontend/Linalg routes,
  source-front-door positive routes, dashboards, reports-only work, or
  spec/test-only work.
- New dtype-prefixed helper op families such as `tcrv_rvv.i32_*`,
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`,
  `tcrv_rvv.i32_macc`, or one-op-per-intrinsic wrappers.
- Any descriptor-derived, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, common-EmitC-derived, source-front-door,
  central ad hoc, metadata-only, or legacy-i32 route authority.

## Requirements

- `widening_dot_reduce_add` must be derived from typed `tcrv_rvv` body/config/
  runtime facts and selected capability facts.
- The RVV provider path must validate operation kind, signed i16-by-i16
  widening dot-product relation, source vector dtype/LMUL, accumulator
  dtype/layout and seed-from-acc role, scalar output/result layout, unit-stride
  memory form, `lhs/rhs/acc/out/n` ABI roles, runtime AVL/VL control,
  tail/mask policy, selected capability, contraction route-family plan,
  route-control provider plan, operand bindings, intrinsic/type/header facts,
  and provider-supported mirrors before route construction.
- The direct contraction owner must consume same-analysis route-family,
  materialization, route-control, runtime, capability, and math operand-binding
  facts before constructing provider-ready setvl/load/dot-reduction/store
  statements.
- Generated harness correctness must distinguish true signed widening
  multiply-plus-reduce from add-only, mul-only, non-widening, wrong sign
  extension, vector-store-to-output, metadata-derived behavior, and tail
  clobbering.
- Artifact metadata and evidence JSON may mirror provider facts only after
  route construction and must label mirror roles where relevant.
- Focused fail-closed coverage must remain in provider/C++ tests for stale or
  missing contraction route-family facts, runtime/control facts, policy/
  capability facts, materialized leaves, and math operand-binding facts.

## Acceptance Criteria

- [ ] Provider-owned route-family, route-control, direct-contraction owner,
  materialization, math-binding, target-leaf/profile, and route-consumption
  tests prove the `widening_dot_reduce_add` operation, signed
  i16mf2xi16mf2-to-i32m1 dot-reduction relation, accumulator/result layout,
  runtime `lhs/rhs/acc/out/n` ABI roles, runtime AVL/VL, policy, selected
  capability, operand bindings, intrinsic/type/header facts, and
  provider-supported mirrors before common EmitC.
- [ ] Explicit and pre-realized selected-body artifact checks expose the
  provider-owned source/destination/accumulator dtype, SEW, LMUL,
  dot-product relation, memory form, runtime ABI order, route-family plan,
  operand bindings, selected capability, accumulator seed layout, scalar result
  layout, and provider-supported mirror fields.
- [ ] Generated-bundle dry-run evidence for explicit
  `widening_dot_reduce_add` includes runtime counts `0`, a small count, and a
  tail count; contraction/SEW/LMUL policy evidence; materialized selected-body
  facts; emitted C/C++ source-load, dot-reduction, scalar store, and runtime
  AVL/VL facts; mirror-only artifact metadata; and harness oracle checks for
  signed widening dot-reduce, seed contribution, scalar output, and sentinel
  preservation.
- [ ] Generated-bundle dry-run evidence for pre-realized
  `widening_dot_reduce_add` includes the same runtime counts and proves the
  pre-realized body is consumed into realized typed `tcrv_rvv` structure before
  provider route construction.
- [ ] Non-dry-run `ssh rvv` generated-bundle execution passes for
  `widening_dot_reduce_add` over counts including `0`, small, and tail cases,
  with signed widening dot-reduce correctness, accumulator seed contribution,
  scalar `out[0]` result, and output/tail sentinel preservation.
- [ ] Focused non-regression checks pass for `widen_i32_to_i64`,
  plain `macc_add`, and `computed_masked_macc_add`.
- [ ] A bounded authority scan over touched RVV planning/provider/target/script/
  fixture/test files finds no executable claim depending on central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, or legacy-i32-derived authority.
- [ ] `git diff --check` passes. `check-tianchenrv` passes or the exact blocker
  is recorded. The task is finished/archived and one coherent commit is created
  if acceptance is met.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-widen-i32-to-i64-dtype-runtime-artifact-boundary/prd.md`.
- Primary code surfaces to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run.test`,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.

## Open Questions

- None blocking. The Direction Brief, specs, previous archived task, and
  existing code surfaces identify a bounded module: signed unit-stride
  `widening_dot_reduce_add` contraction runtime/artifact and `ssh rvv`
  evidence closure.
