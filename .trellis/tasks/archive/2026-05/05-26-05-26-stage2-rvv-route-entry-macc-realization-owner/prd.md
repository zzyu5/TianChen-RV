# Stage2 RVV route-entry MAcc realization owner

## Goal

Make direct route-entry MAcc selected-body realization a production RVV-owned
path, centered on `computed_masked_macc_add` and including the base
`macc_add` route-entry consumer. The route must start from a selected
`tcrv.exec` RVV variant containing a typed pre-realized MAcc body, consume that
body through the RVV selected-body realization owner registry, validate the
realized typed `tcrv_rvv` body through existing MAcc provider facts and
statement-plan owners, build a provider-owned `TCRVEmitCLowerableRoute`, and
produce generated-bundle plus `ssh rvv` evidence for masked accumulator
runtime semantics.

## Direction Source

- Direction title: `Stage2 RVV route-entry MAcc realization owner`.
- Module owner: RVV plugin-local direct route-entry selected-body realization
  and provider boundary for vector multiply-accumulate bodies.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7f5c56be rvv: validate direct route-entry contraction execution`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## Current Repository Facts

- The previous archived direct route-entry contraction task closed
  `computed_masked_strided_input_widening_dot_reduce_add` executable evidence
  through direct route-entry realization, generated bundle, and `ssh rvv`.
- Current `RVVSelectedBodyRealization.cpp` has selected-body realization
  owners for `MAcc` and `computed-mask MAcc`.
- Current direct route-entry owner eligibility includes scalar-broadcast MAcc
  under the `MAcc` owner, but not plain `macc_add`.
- Current direct route-entry owner eligibility does not include the
  `computed-mask MAcc` owner, so a pre-realized `computed_masked_macc_add`
  cannot enter direct route-entry realization through the production owner
  registry.
- Current realization code already validates and materializes:
  - plain/scalar-broadcast `TypedMAccPreRealizedBodyOp` into `setvl`,
    `with_vl`, loads, `tcrv_rvv.macc`, and store;
  - `TypedComputedMaskMAccPreRealizedBodyOp` into `setvl`, `with_vl`,
    compare loads, payload loads, accumulator load, compare mask,
    `tcrv_rvv.masked_macc`, and store;
  - runtime-scalar computed-mask MAcc into its own route-control-aware
    realized structure.
- Current provider planning already has computed-mask accumulation family
  validation, route-control provider plans, math operand-binding facts, a
  computed-mask accumulation statement plan, a migrated statement-plan owner,
  and generated-bundle evidence surfaces for explicit and pre-realized
  `computed_masked_macc_add`.
- Current `scripts/rvv_generated_bundle_abi_e2e.py` supports
  `--direct-pre-realized-route-entry` for selected pre-realized families, but
  not yet for plain `macc_add` or `computed_masked_macc_add`.
- The relevant specs require route-entry support to be owner-scoped through
  the selected-body realization registry, then route-family/materialization/
  math-binding/route-control/statement-plan facts before provider route
  construction. Common EmitC/export must remain neutral.

## Requirements

1. Add bounded route-entry eligibility for the production MAcc realization
   owners:
   - plain `macc_add` as an ordinary base MAcc route-entry consumer;
   - vector-compare `computed_masked_macc_add` as a computed-mask MAcc
     route-entry consumer.
2. Route-entry eligibility must remain owner-scoped through
   `RVVSelectedBodyRealizationOwner::isRouteEntryConsumer`, not through a
   script, route id, artifact name, ABI string, metadata field, or separate
   central allowlist.
3. Direct route-entry `macc_add` must consume the pre-realized MAcc body into a
   realized typed `tcrv_rvv` body and then reach the verified plain MAcc
   route-family plan, materialization facts, math operand-binding facts,
   route-control plan, migrated statement-plan owner, and provider-built route.
4. Direct route-entry `computed_masked_macc_add` must consume the pre-realized
   computed-mask MAcc body into realized typed `tcrv_rvv` structure and then
   reach the computed-mask accumulation family plan, route-control provider
   plan, math operand-binding facts, computed-mask accumulation statement plan,
   migrated statement-plan owner, and provider-built route.
5. The computed-mask MAcc path must validate/carry operation kind, predicate,
   computed mask role/source/form, same-VL scope, lhs/rhs/acc/out/n ABI roles,
   accumulator input/result layout, runtime `n`/AVL, VL/tail/mask policy,
   dtype/SEW/LMUL, selected capability, operand bindings, provider-supported
   mirrors, and header/type/intrinsic mirrors only after route construction.
6. Unsupported or inconsistent mask, accumulator, dtype/config, runtime ABI,
   stale route-control/materialization/math facts, or MAcc shape must fail
   closed with targeted diagnostics before common EmitC.
7. Generated-bundle direct route-entry evidence must be added for
   `computed_masked_macc_add`, and should include plain `macc_add` where it
   proves the base route-entry owner path without broadening scope.
8. Runtime/correctness claims for `computed_masked_macc_add` require real
   `ssh rvv` generated-bundle evidence over counts including `0`, small,
   full-vector-ish/exact, tail, and stress cases.
9. Preserve recent direct route-entry contraction non-regression and avoid
   weakening existing plain MAcc, computed-mask MAcc, or contraction provider
   boundaries.

## Acceptance Criteria

- [x] `MAcc` and `computed-mask MAcc` route-entry eligibility is explicit in
      the selected-body realization owner registry for the bounded plain and
      vector-compare MAcc bodies.
- [x] C++ tests prove direct route-entry `macc_add` and
      `computed_masked_macc_add` are route-entry eligible through their owners,
      consume the pre-realized body before provider facts are collected, and
      expose realized `setvl` / `with_vl` / load / compare / MAcc / merge-or-
      passthrough / store structure as appropriate.
- [x] C++ tests prove the realized `macc_add` path reaches plain MAcc provider
      facts and the realized `computed_masked_macc_add` path reaches computed-
      mask accumulation family, route-control, math-binding, statement-plan,
      migrated statement-plan, and provider-built route facts.
- [x] C++ fail-closed coverage covers targeted MAcc route-entry errors for at
      least mask mismatch, accumulator/result mismatch, dtype/config mismatch,
      runtime `n` ABI mismatch, and unsupported MAcc shape.
- [x] A direct route-entry dry-run succeeds for `computed_masked_macc_add`
      using `--pre-realized-selected-body --direct-pre-realized-route-entry`
      and does not run `--tcrv-materialize-selected-lowering-boundaries`.
- [x] The direct route-entry dry-run evidence labels the materializer as
      `rvv-route-entry-selected-body-realization`, sets
      `route_entry_realization` to `true`, proves pre-realized body
      consumption, and mirrors computed-mask MAcc/provider facts only after
      provider route construction.
- [x] Real `ssh rvv` generated-bundle execution passes for direct route-entry
      `computed_masked_macc_add` over counts including `0`, `7`, `16`, `23`,
      and `257`, with active-lane MAcc correctness, inactive accumulator/
      pass-through preservation, and tail sentinel preservation.
- [x] Focused non-regression covers plain `macc_add`, existing pre-realized
      selected-boundary `computed_masked_macc_add`, and recent direct
      route-entry contraction evidence.
- [x] Bounded touched-file authority scan finds no positive route/executable
      authority from central ad hoc code, names, route ids, metadata,
      descriptors, ABI strings, scripts, artifact names, common EmitC,
      source-front-door paths, or legacy i32 helper surfaces.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

- No new contraction variant, dtype/LMUL clone batch, broadcast/reduction
  expansion, high-level frontend/Linalg route, source-front-door positive
  route, dashboard, report-only task, or broad smoke matrix.
- No common EmitC RVV semantics and no target/export inference of MAcc,
  computed-mask, accumulator, dtype/config, ABI, policy, intrinsic, or route
  support.
- No new dtype-prefixed helper op or legacy `RVVI32M1`/`rvv-i32m1` positive
  path.
- No weakening of the direct route-entry contraction provider boundary that
  just landed.

## Technical Approach

1. Start and validate this Trellis task, with implement/check context limited
   to specs.
2. Add owner-scoped route-entry consumer predicates for plain MAcc and
   computed-mask MAcc in `RVVSelectedBodyRealization.cpp`.
3. Update the route-entry diagnostic text so it names the expanded bounded
   families truthfully.
4. Extend C++ route-entry owner tests to include `macc_add` and
   `computed_masked_macc_add`, including realized IR shape, provider facts,
   statement-plan/route facts, and fail-closed diagnostics.
5. Extend generated-bundle direct route-entry support and dry-run tests for
   `computed_masked_macc_add`, with plain `macc_add` as focused base MAcc
   coverage if needed.
6. Run focused dry-runs and `ssh rvv` generated-bundle evidence for
   `computed_masked_macc_add`, plus focused non-regression for plain MAcc and
   direct route-entry contraction.
7. Run authority scan, `git diff --check`, and `check-tianchenrv`.
8. Finish/archive the task and create one coherent commit if acceptance is
   met.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-route-entry-macc-realization-owner`
2. Focused C++ plugin test target or full C++ plugin test runner as available.
3. Direct route-entry dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind computed_masked_macc_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --artifact-root artifacts/tmp/stage2_rvv_route_entry_macc_realization_owner --run-id direct-route-entry-computed-masked-macc-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
4. Representative `ssh rvv` generated-bundle run for the same direct route-entry
   path and counts.
5. Focused non-regression dry-runs for plain `macc_add`, pre-realized
   selected-boundary `computed_masked_macc_add`, and recent direct route-entry
   contraction.
6. Bounded authority scan over touched production/test/script files.
7. `git diff --check`
8. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived task context read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-direct-route-entry-contraction-executable-boundary/prd.md`,
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-macc-accumulator-runtime-artifact-boundary/prd.md`,
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-computed-mask-macc-accumulator-mask-artifact-boundary/prd.md`,
  and `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-computed-mask-macc-provider-validation/prd.md`.
- Primary code surfaces:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related generated
  bundle dry-run tests under `test/Scripts`.

## Open Questions

None blocking. Current code evidence identifies a bounded production gap:
plain and computed-mask MAcc realization owners exist, but their direct
route-entry eligibility and generated-bundle direct-route evidence are not yet
explicit for the representative MAcc route-entry workflow.

## Definition Of Done

The representative direct route-entry `computed_masked_macc_add` workflow, with
plain `macc_add` as the base MAcc route-entry owner case, is implemented and
evidenced through production RVV owner-scoped selected-body realization,
provider facts and statement-plan construction, generated bundle, and real
`ssh rvv` correctness. Trellis state is truthful, authority scans are clean,
checks pass or exact blockers are recorded, and one coherent commit records the
completed task.
