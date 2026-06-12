# Stage2 RVV widening conversion dtype/runtime artifact boundary

## Goal

Close one representative Stage2 signed `widen_i32_to_i64` executable boundary
from a selected `tcrv.exec` RVV variant with typed `tcrv_rvv.widening_convert`
body through RVV-owned conversion dtype/config/runtime validation,
provider-built `TCRVEmitCLowerableRoute`, neutral common EmitC materialization,
generated RVV artifact/harness, and real `ssh rvv` correctness evidence.

The selected route is:

```text
lhs[i]: i32 / LMUL m1, n
  -> runtime VL from n / remaining AVL
  -> source load as i32m1
  -> signed widening conversion to i64m2
  -> unit-stride store to out[i]: i64 / LMUL m2
```

This task is a bounded conversion/dtype/SEW/LMUL policy closure. It is not a
conversion matrix, not a dtype clone batch, and not a widening MAcc or reduction
expansion.

## What I already know

- Current HEAD is `2e45ca80 rvv: validate computed-mask macc artifact boundary`.
- The repository started this task with a clean worktree and no current Trellis
  task file.
- The previous archived computed-mask MAcc task closed its own artifact and
  `ssh rvv` evidence boundary; this task must not weaken that MAcc boundary.
- Existing conversion code already exposes `WidenI32ToI64` and
  `WidenI16ToI32` operation kinds, `UnitStrideConversion` memory form,
  widening conversion route-family plan state, route-control provider plan
  state, math operand-binding facts, migrated statement-plan ownership, and
  provider consumption.
- Existing C++ tests cover provider-owned route-family validation,
  route-control joining, stale runtime/type/policy/capability diagnostics,
  conversion statement-plan construction, migrated statement-plan registry
  ownership, and `widen_i32_to_i64` provider route consumption.
- Existing explicit and pre-realized widening conversion fixtures already
  expose source i32/m1, result i64/m2, runtime `lhs,out,n` ABI order,
  provider-supported mirrors, route operand binding mirrors, conversion
  relation, and header metadata.
- Existing generated-bundle dry-run tests for `widen_i32_to_i64` use runtime
  counts `7`, `16`, and `23`; they do not yet include `n = 0`.
- Existing `widen_i32_to_i64` generated harness checks converted lanes, but it
  does not explicitly check tail sentinel preservation beyond runtime `n` or
  sign-extension coverage the way the newer `widen_i16_to_i32` harness does.
- Authority must remain structural and provider-owned: typed body/config/runtime
  facts and RVV provider plans own conversion kind, signed relation, source and
  result element types, SEW, LMUL, memory form, runtime AVL/VL, ABI roles,
  intrinsics, and route support. Route ids, artifact names, metadata, scripts,
  ABI strings, common EmitC, source-front-door paths, and legacy i32 helper
  names remain mirrors or negative inventory only.

## Scope

- Close the representative signed `i32m1 -> i64m2` unit-stride widening
  conversion artifact/evidence boundary only.
- Preserve existing provider-owned validation and cite it where it already
  satisfies the acceptance criteria instead of duplicating production logic.
- Strengthen generated-bundle evidence for `widen_i32_to_i64` so dry-run and
  non-dry-run cases include `n = 0`, small, full-vector-ish, and tail counts.
- Strengthen the generated harness oracle for `widen_i32_to_i64` so it proves:
  signed widening of negative i32 inputs, positive input coverage, converted
  lane correctness, runtime `n`/AVL behavior, and output sentinel preservation
  beyond `n`.
- Keep the generated evidence JSON focused on conversion/SEW policy facts:
  source/result type policy, conversion relation, materialized body, emitted
  C/C++ intrinsic flow, runtime counts as execution cases, and mirror-only
  metadata.
- Keep common EmitC/export neutral; common code may consume provider-built
  routes and mirrors after construction, but must not infer RVV conversion,
  dtype/config, ABI, policy, intrinsic, or support semantics.

## Out of Scope

- Broad conversion matrices, dtype/LMUL clone batches, narrowing conversion,
  unsigned conversion, floating conversion, high-level frontend/Linalg routes,
  source-front-door positive routes, dashboards, reports-only work, or
  spec/test-only work.
- Widening MAcc, scalar-broadcast MAcc, runtime-scalar computed-mask MAcc, new
  reduction variants, or any MAcc/reduction behavior changes.
- New dtype-prefixed helper op families such as `tcrv_rvv.i32_*` conversion
  helpers or one-op-per-intrinsic wrappers.
- Any descriptor-derived, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, common-EmitC-derived, source-front-door,
  central ad hoc, metadata-only, or legacy-i32 route authority.

## Requirements

- `widen_i32_to_i64` must be derived from typed `tcrv_rvv` body/config/runtime
  facts and selected capability facts.
- The RVV provider path must validate operation kind, source element type and
  LMUL, result element type and LMUL, source/result vector types, signed
  widening relation, unit-stride memory form, `lhs/out/n` ABI roles, runtime
  AVL/VL control, tail/mask policy, route-family plan, route-control provider
  plan, operand bindings, intrinsic/type/header facts, and provider-supported
  mirrors before route construction.
- Statement construction must be RVV-owned and must include setvl, source load,
  widening conversion, and output store with runtime loop VL operands.
- Generated harness correctness must distinguish signed widening from
  truncation, zero-extension, add-only behavior, metadata-derived behavior, and
  tail clobbering.
- Artifact metadata and evidence JSON may mirror provider facts only after
  route construction and must label mirror roles where relevant.
- Focused fail-closed coverage must remain in provider/C++ tests for stale or
  missing conversion route-family, runtime/type/policy/capability, operand
  binding, conversion relation, intrinsic, and statement-plan dependencies.

## Acceptance Criteria

- [ ] Provider-owned route-family, route-control, materialization,
  math-binding, statement-plan, migrated-owner, and route-consumption tests
  prove the `widen_i32_to_i64` operation, signed conversion relation, source
  i32/m1 facts, result i64/m2 facts, runtime `lhs,out,n` ABI roles, runtime
  AVL/VL, policy, selected capability, operand bindings, intrinsic/type/header
  facts, and provider-supported mirrors before common EmitC.
- [ ] Explicit and pre-realized selected-body artifact checks expose the
  provider-owned source/destination dtype, SEW, LMUL, conversion relation,
  memory form, runtime ABI order, route-family plan, operand bindings, selected
  capability, and provider-supported mirror fields.
- [ ] Generated-bundle dry-run evidence for explicit `widen_i32_to_i64`
  includes runtime counts `0`, `7`, `16`, and `23`, conversion/SEW policy
  evidence, materialized selected-body facts, emitted C/C++ source-load /
  conversion / store facts, runtime AVL/VL facts, mirror-only artifact
  metadata, and harness sentinel/sign-extension checks.
- [ ] Generated-bundle dry-run evidence for pre-realized
  `widen_i32_to_i64` includes the same runtime counts and proves the
  pre-realized body is consumed into realized typed `tcrv_rvv` structure before
  provider route construction.
- [ ] Non-dry-run `ssh rvv` generated-bundle execution passes for
  pre-realized `widen_i32_to_i64` over counts including `0`, a small count, a
  full-vector-ish count, and a tail case, with signed widening correctness and
  output sentinel preservation.
- [ ] `computed_masked_macc_add` and plain `macc_add` focused non-regression
  checks pass.
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
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-computed-mask-macc-accumulator-mask-artifact-boundary/prd.md`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-widen-i32-to-i64-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widen-i32-to-i64-dry-run.test`,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.

## Open Questions

- None blocking. The Direction Brief, specs, existing provider tests, and
  generated-bundle gaps identify a bounded module: signed `i32m1 -> i64m2`
  conversion dtype/runtime artifact and `ssh rvv` evidence closure.
