# Stage2 RVV standalone min/max scalar-channel reduction family

## Goal

Extend the standalone-reduction scalar accumulator/result channel from the
already-proven reduce-add path to the existing low-level RVV min/max standalone
reduction surface. The production owner is the RVV selected-body realization,
route-family planning/provider facts, target artifact mirror validation, and
generated-bundle ABI/evidence boundary for:

- `standalone_reduce_min`
- `standalone_reduce_max`
- `computed_mask_standalone_reduce_min`
- `computed_mask_standalone_reduce_max`

The work must keep direct pre-realized route-entry authority fail-closed where
that family is selected-boundary-only, and must not make common EmitC, route
ids, artifact names, exact intrinsic spellings, ABI strings, scripts, or mirror
metadata decide reduction semantics.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean before task creation.
- Latest commit before task creation is
  `74ff9c06 rvv: add scalar channel for standalone reductions`.
- No `.trellis/.current-task` existed when this task was created.
- The archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-standalone-reduction-scalar-channel`
  proved the standalone reduction scalar accumulator/result channel for
  `runtime_scalar_cmp_masked_standalone_reduce_add` LMUL m2.
- Existing dialect support already recognizes plain standalone min/max and
  computed-mask standalone min/max operation names.
- Existing script and target tests already contain standalone and computed-mask
  min/max consumers, but the Direction Brief identifies non-m1 scalar-channel
  reduction materialization as add-specialized.

## Requirements

- Preserve the RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized `tcrv_rvv` body ->
  RVV plugin-local selected-body realization -> verified route-family facts ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral EmitC -> target artifact.
- Derive min/max operation kind from typed body/config/runtime facts.
- Derive the source vector channel from typed source/work facts:
  source vector type, C type, load/mask/runtime-scalar facts when present,
  SEW/LMUL, policy, memory form, runtime AVL/VL relation, and ABI role order.
- Derive the scalar accumulator/result channel from typed scalar-channel facts:
  seed behavior, scalar result vector type/C type, scalar result LMUL, lane-0
  scalar result layout, store intrinsic, store VL, and output ABI role.
- Use reduction kind to derive the correct reduction intrinsic and inactive
  lane neutral value for min/max, including computed-mask cases.
- Keep target artifact/header/object metadata as mirror-only after provider
  route reconstruction and validation.
- Keep non-regression for the completed reduce-add scalar-channel path.
- If part of the family is too large for one round, complete one coherent
  production submodule and leave task state truthful.

## Acceptance Criteria

- [x] Selected-body realization preserves min/max reduction kind, source
      vector channel, scalar accumulator/result channel, mask policy when
      present, runtime `n`/AVL/VL, and selected ABI order.
- [x] Route planning/provider facts expose provider-derived min/max source
      vector type/C type and scalar result vector type/C type, not add-only
      defaults or intrinsic-name shortcuts.
- [x] Target artifact consumers validate provider-derived source/scalar-result
      channel mirrors for min/max and fail closed on stale or missing mirrors.
- [x] Generated-bundle dry-run evidence passes for plain min/max and computed-
      mask min/max with `route_entry_realization=false` where applicable,
      `pre_realized_body_consumed=true`, explicit scalar-channel facts, ABI
      order, route-control facts, and mirror-only labels.
- [x] Fail-closed checks cover wrong reduction kind, wrong source/scalar
      channel relation, wrong SEW/LMUL relation, wrong seed/result store type,
      wrong mask binding, wrong AVL/VL relation, stale metadata/route id, and
      direct-route-entry-only authority attempts.
- [x] Real `ssh rvv` generated-bundle evidence covers min/max counts `0`, `1`,
      exact-VL, tail, and stress cases with signed payloads, seed values, and
      mask patterns for computed-mask routes included in this round.
- [x] Focused reduce-add scalar-channel non-regression passes.
- [x] Bounded authority scan over touched production, target, script, and test
      files finds no new central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] The task is finished/archived and committed if complete.

## Non-goals

- No high-level Linalg/frontend lowering.
- No Stage3 tuning, dashboards, readiness reports, or broad smoke matrices.
- No one-intrinsic wrapper dialects or dtype/LMUL clone batches.
- No new dtype-prefixed RVV helper op families.
- No descriptor-driven computation, source-front-door positive route, direct-C
  exporter authority, route-id authority, artifact-name authority, exact-
  intrinsic-as-authority, or common EmitC semantic invention.
- No unrelated revisit of widening dot, MAcc, compare/select, segment2, or
  standalone reduce-add demotions except focused non-regression.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-standalone-reduction-scalar-channel/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-boundary/prd.md`

## Technical Notes

- The standalone reduction scalar-channel spec requires distinct provider
  fields such as `sourceVectorTypeName`, `sourceVectorCType`,
  `scalarResultVectorTypeName`, and `scalarResultVectorCType`.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/control
  roles only; typed `tcrv_rvv` body/config/runtime facts must carry compute,
  dtype, SEW/LMUL, policy, mask, reduction kind, and runtime AVL/VL authority.
- The provider must consume route-family provider plans and route
  materialization facts before constructing `TCRVEmitCLowerableRoute`.
- Common EmitC may only materialize provider-built route payloads.
