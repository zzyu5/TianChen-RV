# Stage2 RVV runtime-scalar compare-masked standalone min/max scalar channel

## Goal

Extend the corrected typed standalone-reduction scalar accumulator/result
channel to runtime-scalar compare-masked standalone min/max reductions:

- `runtime_scalar_cmp_masked_standalone_reduce_min`
- `runtime_scalar_cmp_masked_standalone_reduce_max`

The production module spans RVV dialect verification, RVV plugin-local
selected-body realization, route planning/provider facts, target artifact ABI
mirrors, generated-bundle evidence, and focused `ssh rvv` correctness. The
route must carry a runtime scalar threshold into compare-produced mask
construction and then into a min/max scalar-channel reduction without deriving
dtype, reduction kind, inactive neutral, mask policy, route support, or ABI
order from names, descriptors, artifacts, scripts, exact intrinsic spellings, or
common EmitC.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean before task creation.
- Latest commit before task creation is
  `81364872 rvv: add standalone minmax scalar channel`.
- No `.trellis/.current-task` existed when this task was created.
- The immediate predecessor
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-standalone-minmax-scalar-channel`
  completed standalone and computed-mask min/max scalar-channel reductions.
- The prior runtime-scalar compare-masked standalone reduction task completed
  the add-shaped boundary for `runtime_scalar_cmp_masked_standalone_reduce_add`.
- Current code inspection shows the runtime-scalar compare-masked standalone
  reduction verifier/realization/profile/script surface is still add-shaped:
  it accepts `runtime_scalar_cmp_masked_standalone_reduce_add` but not min/max.
- Plain and computed-mask min/max already provide structural patterns for
  reduction-kind preservation, inactive neutral handling, scalar result channel
  mirrors, generated-bundle evidence, and direct route-entry fail-closed tests.

## Requirements

- Preserve the RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized `tcrv_rvv` body ->
  RVV plugin-local selected-body realization -> provider-derived route-family
  facts -> `TCRVEmitCLowerableRoute` -> neutral EmitC -> target artifact.
- Treat `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` as ABI/runtime role
  declarations only. The typed `tcrv_rvv` body must carry compare operand use,
  runtime scalar threshold, source vector channel, accumulator/result scalar
  channel, runtime `n`/AVL/VL, policy, mask production, and reduction kind.
- Extend the runtime-scalar computed-mask standalone reduction pre-realized body
  verifier to accept min/max only when predicate, mask role/source/form,
  memory form, accumulator/result layout, SEW/LMUL relation, scalar threshold C
  type, pointer C types, ABI roles, policy, and runtime count binding are
  structurally consistent.
- Extend RVV selected-body realization so min/max are consumed into realized
  `tcrv_rvv` structure with `setvl`, source/compare load, runtime scalar splat,
  compare mask, inactive-lane min/max neutralization, masked standalone reduce,
  scalar accumulator/result channel, lane-0 scalar output store, and selected
  ABI order.
- Extend route planning/provider facts so runtime-scalar compare-masked min/max
  are ordinary standalone-reduction family consumers with provider-derived
  operation kind, source vector type/C type, scalar result vector type/C type,
  mask type/C type, inactive neutral contract, runtime AVL/VL, and ABI order.
- Keep direct pre-realized route-entry unsupported for these selected-boundary
  op kinds unless an explicit later owner task changes the route-entry owner
  predicate and adds matching provider/runtime evidence.
- Keep target artifact/header/object metadata mirror-only after rebuilding and
  validating the provider route.
- Generated-bundle tooling may parse and verify evidence, but it must not become
  route authority. It must dry-run through selected-boundary realization with
  `route_entry_realization=false`.
- Keep focused non-regression for `runtime_scalar_cmp_masked_standalone_reduce_add`
  and the already completed standalone/computed-mask min/max scalar-channel
  family.

## Acceptance Criteria

- [ ] Dialect verifier and selected-body realization accept
      `runtime_scalar_cmp_masked_standalone_reduce_min/max` only through typed
      runtime-scalar compare-masked standalone reduction facts and fail closed on
      unsupported op kind, predicate, mask binding, memory form, scalar threshold
      type, pointer C type, accumulator/result layout, SEW/LMUL relation,
      runtime ABI order, policy, or runtime `n` binding mismatch.
- [ ] Realized bodies structurally carry runtime scalar compare parameter
      binding, compare-produced mask, source work vector SEW/LMUL, inactive
      min/max neutral semantics, scalar accumulator/result LMUL m1 channel,
      reduction kind, setvl/VL placement, policy, required capabilities, and
      selected ABI order.
- [ ] Route planning/provider facts consume the realized typed body/config/
      runtime facts and expose runtime-scalar compare-masked min/max operation
      kind, source vector type/C type, scalar result vector type/C type, mask
      type/C type, inactive neutral contract, route operand binding closure,
      provider-supported mirror, and target leaf profile.
- [ ] Target artifact and generated-bundle boundary validate provider-derived
      ABI/header/type mirrors for both min and max and do not infer support from
      op names, route ids, artifact names, ABI strings, exact intrinsic
      spellings, script expectations, status fields, or common EmitC.
- [ ] Generated-bundle dry-runs pass for min and max with
      `route_entry_realization=false`, `pre_realized_body_consumed=true`, ABI
      order `cmp_lhs,rhs_scalar,src,acc,out,n`, non-empty mask/reduction
      boundaries, scalar-channel fields, provider mirrors, and route-control
      facts.
- [ ] Direct pre-realized route-entry dry-run remains fail-closed for the new
      min/max op kinds with a selected-boundary-only diagnostic.
- [ ] Real `ssh rvv` generated-bundle runs pass for min and max over counts
      including `0`, `1`, exact-VL, tail, and stress cases, with at least two
      runtime scalar thresholds, signed input patterns, active/inactive/
      all-inactive masks, min and max oracles, seed accumulation, and tail
      preservation.
- [ ] Non-regression passes for
      `runtime_scalar_cmp_masked_standalone_reduce_add` and the existing
      standalone/computed-mask min/max scalar-channel routes.
- [ ] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] Task state is truthful; if complete, task is finished/archived and a
      coherent commit is created.

## Non-goals

- No segment2, widening/conversion, unrelated compare/select expansion, high-
  level frontend lowering, one-intrinsic wrapper dialects, tuning databases,
  dashboards, broad smoke matrices, or proof-only tests for already completed
  standalone/computed-mask min/max.
- No dtype/LMUL clone batch. Include only minimal LMUL m1/m2 scalar-channel
  coverage needed to prove the runtime-scalar mask plus min/max reduction
  contract.
- No descriptor-driven computation, source-front-door positive route, direct-C
  exporter authority, route-id authority, artifact-name authority, exact-
  intrinsic-as-authority, or common EmitC semantic invention.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/guides/index.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-standalone-minmax-scalar-channel/prd.md`

## Technical Notes

- Read first production surfaces:
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
- Initial inspection found add-only guards in runtime-scalar standalone
  reduction op-kind predicates and verifier diagnostics; min/max patterns are
  already present for plain and computed-mask standalone reductions.
- The generated-bundle evidence should reuse the predecessor scalar-channel
  harness shape, but must add runtime scalar threshold coverage and min/max
  inactive neutral oracles.
