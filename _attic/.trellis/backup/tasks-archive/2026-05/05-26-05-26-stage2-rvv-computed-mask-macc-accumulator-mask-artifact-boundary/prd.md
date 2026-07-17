# Stage2 RVV computed-mask MAcc accumulator/mask artifact boundary

## Goal

Close one representative Stage2 vector-compare `computed_masked_macc_add`
executable boundary from a selected `tcrv.exec` RVV variant with typed
`tcrv_rvv.masked_macc` body through RVV-owned mask/accumulator/runtime
validation, provider-built `TCRVEmitCLowerableRoute`, neutral common EmitC
materialization, generated RVV artifact/harness, and real `ssh rvv`
correctness evidence.

The selected route is:

```text
cmp_lhs[i], cmp_rhs[i], lhs[i], rhs[i], acc[i], n
  -> mask[i] = cmp_lhs[i] < cmp_rhs[i]
  -> out[i] = acc[i] + lhs[i] * rhs[i] where mask[i] is active
  -> inactive lanes preserve the required accumulator/pass-through behavior
```

This task is not a new MAcc family and not a runtime-scalar MAcc expansion. It
finishes the artifact/evidence boundary for the existing vector-compare
computed-mask MAcc path after previous rounds added route-family validation,
route-control ownership, math operand binding, migrated statement planning, and
plain `macc_add` accumulator/runtime/artifact evidence.

## What I already know

- Current HEAD is `4867c227 rvv: validate plain macc accumulator boundary`.
- The repository started this task with a clean worktree and no current Trellis
  task.
- The previous archived plain `macc_add` task closed provider-owned
  accumulator/runtime/artifact evidence for unmasked MAcc, including generated
  bundle dry-run and `ssh rvv` evidence.
- Current code already exposes `ComputedMaskedMAccAdd` and
  `RuntimeScalarComputedMaskedMAccAdd` operation kinds, computed-mask MAcc
  memory forms, computed-mask accumulation route-family plan state,
  route-control consumption, math operand-binding facts, and a migrated
  computed-mask accumulation statement-plan owner.
- Existing explicit selected-body computed-mask MAcc fixtures already exercise
  `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`, `out`, and `n` ABI roles, plus
  `tcrv_rvv.compare` -> `tcrv_rvv.masked_macc` -> `tcrv_rvv.store`.
- Existing generated-bundle dry-run coverage for `computed_masked_macc_add`
  uses runtime counts `7`, `16`, and `23`; it does not yet prove the required
  `n = 0` boundary and real `ssh rvv` masked MAcc correctness in this task.
- The route must remain authority-clean: typed body/config/runtime facts and
  RVV provider plans own mask, accumulator, operation, policy, ABI, and
  intrinsic facts. Route ids, artifact names, metadata, scripts, ABI strings,
  common EmitC, source-front-door paths, and legacy i32 helper names remain
  mirrors or negative inventory only.

## Scope

- Strengthen the vector-compare `computed_masked_macc_add` production boundary
  only where current code lacks executable artifact/evidence closure.
- Validate or expose, before route construction, provider-owned facts for:
  operation kind `computed_masked_macc_add`, compare predicate `slt`,
  compare-produced mask source/role/form, inactive-lane pass-through/merge
  behavior, lhs/rhs payload roles, accumulator input role/layout, output/result
  layout, runtime `n`/AVL, VL/tail/mask policy, typed SEW/LMUL config,
  selected capability, operand bindings, route-family plan, route-control plan,
  and provider-supported mirrors.
- Ensure generated artifact/harness evidence distinguishes active and inactive
  lanes and checks vector MAcc semantics only for active computed-mask lanes.
- Add `n = 0`, small, full-vector-ish, and tail runtime counts to dry-run and
  non-dry-run evidence for the selected route.
- Keep plain `macc_add` and computed-mask standalone reduction non-regression
  green.
- Keep common EmitC/export neutral; common code may consume provider-built
  routes and mirrors after construction, but must not infer RVV mask, MAcc,
  accumulator, dtype/config, ABI, policy, intrinsic, or support semantics.

## Out of Scope

- Runtime-scalar computed-mask MAcc support expansion.
- Scalar-broadcast MAcc, widening MAcc, dot reductions, new reduction variants,
  dtype/LMUL clone batches, new source-front-door positive routes, high-level
  frontend/Linalg routes, dashboards, reports-only work, or status-only work.
- Any new dtype-prefixed helper op such as `tcrv_rvv.i32_macc` or new
  one-op-per-intrinsic wrapper family.
- Any descriptor-derived, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, common-EmitC-derived, source-front-door,
  central ad hoc, metadata-only, or legacy-i32 route authority.
- Weakening the plain `macc_add` boundary or the computed-mask standalone
  reduction boundary that already landed.

## Requirements

- The selected route must be derived from typed `tcrv_rvv` body/config/runtime
  facts and selected capability facts.
- The computed-mask accumulation family plan, route materialization facts,
  math operand-binding facts, and route-control provider plan must be consumed
  before computed-mask MAcc statement construction and before
  `TCRVEmitCLowerableRoute` construction.
- Statement construction must be RVV-owned and must include setvl, compare
  producer loads, payload loads, accumulator load, compare mask, active MAcc,
  masked merge/pass-through, and output store.
- The generated harness must check `out[i] = acc[i] + lhs[i] * rhs[i]` exactly
  where the computed mask is active, inactive-lane accumulator/pass-through
  preservation where required, and tail sentinel preservation beyond runtime
  `n`.
- Artifact metadata and evidence JSON may mirror provider facts only after
  route construction and must label those fields as mirrors where relevant.
- Focused fail-closed coverage must reject missing or stale provider-owned
  facts before common EmitC, especially stale route-control/materialization
  facts, missing math binding, invalid mask producer/classification, missing
  accumulator/output/runtime ABI roles, and missing materialization leaves.

## Acceptance Criteria

- [ ] `computed_masked_macc_add` has provider-owned route-family/materialization/
  math-binding/route-control/statement-plan validation or equivalent existing
  validated surfaces proving operation, predicate, mask, accumulator, result,
  runtime, typed config, capability, and provider-supported mirror facts before
  route construction.
- [ ] The route provider reaches vector-compare computed-mask MAcc through the
  RVV-owned migrated computed-mask accumulation statement-plan boundary, not
  through central provider reconstruction from operation names, ABI strings,
  route ids, artifact metadata, scripts, or common EmitC logic.
- [ ] Missing or stale computed-mask MAcc route-family, materialization,
  route-control, math operand-binding, accumulator/output/runtime ABI,
  typed-config/policy, selected-capability, mask-producer, or materialized leaf
  facts fail closed with targeted diagnostics before common EmitC.
- [ ] Explicit selected-body target artifact checks expose provider-owned mask,
  accumulator, result, runtime-control, route-family, route-control, operand
  binding, provider-supported mirror, and artifact-mirror fields for
  `computed_masked_macc_add`.
- [ ] Generated-bundle dry-run evidence for `computed_masked_macc_add` includes
  runtime counts `0`, `7`, `16`, and `23`, typed body/config/runtime authority,
  selected ABI roles, compare predicate/source, accumulator/result layout,
  inactive-lane behavior, provider-supported mirror, RVV-owned statement plan,
  emitted masked MAcc/merge/store facts, and mirror-only artifact metadata.
- [ ] Non-dry-run `ssh rvv` generated-bundle execution passes for
  `computed_masked_macc_add` over counts including `0`, a small count,
  one full-vector-ish count, and a tail case, with active-lane MAcc correctness,
  inactive-lane preservation, and tail sentinel preservation.
- [ ] Plain `macc_add` and `computed_mask_standalone_reduce_add` focused
  non-regression pass.
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
  `.trellis/spec/implementation-stack/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-macc-accumulator-runtime-artifact-boundary/prd.md`
  plus its `implement.jsonl` and `check.jsonl`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-macc-add.mlir`,
  and `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-macc-add-dry-run.test`.

## Open Questions

- None blocking. The brief, specs, previous plain MAcc closure, and current code
  identify a bounded module: vector-compare `computed_masked_macc_add`
  mask/accumulator/runtime artifact and ssh evidence closure.
