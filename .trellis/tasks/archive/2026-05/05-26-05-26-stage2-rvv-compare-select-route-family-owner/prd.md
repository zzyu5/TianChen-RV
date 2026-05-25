# Stage2 RVV compare/select route-family owner

## Goal

Make compare/select a first-class RVV route-family owner for one
representative executable path: a selected `tcrv.exec` RVV variant with a
pre-realized typed `tcrv_rvv` computed-mask vector select body, realized
through the RVV selected-body route-entry owner, validated by RVV-owned
compare/select family facts, routed through provider-owned
`TCRVEmitCLowerableRoute`, materialized by neutral common EmitC, and executed
on the real `ssh rvv` target.

The representative path is `computed_mask_select`: compare LHS/RHS produce a
same-VL predicate mask, true/false payload vectors are selected under that
mask, and the result is stored without depending on MAcc, contraction, or any
masked-accumulation consumer.

## Direction Source

- Direction title: `Stage2 RVV compare/select route-family owner`.
- Module owner: RVV plugin-owned generic compare/select route-family boundary
  on the corrected typed `tcrv_rvv` surface.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f670ca0d rvv: route direct macc entry realization`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## Current Repository Facts

- `RVVSelectedBodyRealization.cpp` already has an
  `elementwise/compare-select` selected-body realization owner and the owner
  currently marks the whole elementwise/compare-select cluster as
  route-entry-capable.
- The current route-entry predicate is structural by pre-realized op class but
  does not name the bounded compare/select route-entry cases or distinguish
  plain `cmp_select` from computed-mask vector select at the route-entry
  owner boundary.
- Route planning already contains:
  - plain compare-select route-family plans;
  - computed-mask select route-family plans;
  - route-control provider plan adoption for plain and computed-mask select;
  - elementwise/select operand-binding facts;
  - compare/select statement-plan construction;
  - migrated statement-plan owner registry entry `compare/select`.
- `scripts/rvv_generated_bundle_abi_e2e.py` supports pre-realized
  `computed_mask_select` fixtures through the public selected lowering-boundary
  materializer, but `--direct-pre-realized-route-entry` is currently bounded
  to `cmp_select` / `cmp_select_sle` and other non-select families. It does
  not yet admit direct route-entry `computed_mask_select`.
- Existing dry-run tests cover pre-realized `computed_mask_select` and direct
  route-entry `cmp_select`; they do not prove the representative
  compare-produced mask plus true/false vector select route-entry workflow.
- Existing provider tests cover computed-mask select family plan validation,
  stale runtime ABI/order/mirror rejection, route-control, and statement-plan
  construction for explicit realized bodies.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-route-entry-macc-realization-owner`
  closed direct route-entry MAcc and computed-mask MAcc without weakening the
  compare/select family. This task must preserve those routes.

## Requirements

1. Make the compare/select route-entry owner boundary explicit and bounded for
   the representative route-entry families:
   - plain `cmp_select` / `cmp_select_sle`;
   - computed-mask vector `computed_mask_select` /
     `computed_mask_select_sle`.
2. Route-entry eligibility must remain owner-scoped through
   `RVVSelectedBodyRealizationOwner::isRouteEntryConsumer`, not through a
   script, route id, artifact name, ABI string, metadata field, or central
   provider allowlist.
3. Direct route-entry `computed_mask_select` must consume the pre-realized
   typed body into realized `setvl` / `with_vl` / loads / compare / select /
   store before route facts are collected.
4. The realized route must reach RVV-owned computed-mask select family facts,
   route-control provider facts, elementwise/select operand-binding facts,
   compare/select statement-plan facts, migrated statement-plan owner
   consumption, and provider-built route construction.
5. The route facts must validate and carry compare predicate kind, source
   dtype/SEW/LMUL, mask type and same-VL scope, true/false payload dtypes and
   layouts, output layout, runtime `n`/AVL/VL, tail/mask policy, selected
   capability, ABI roles, operand bindings, route-family plan,
   intrinsic/type/header facts, and provider-supported mirrors only after
   provider route construction.
6. Unsupported or inconsistent compare kind, mask source/form/scope, select
   payload dtype/layout, runtime `n` ABI, route-control policy, selected
   capability, or stale operand/materialization facts must fail closed with
   targeted diagnostics before common EmitC.
7. Generated-bundle direct route-entry dry-run evidence must be added for
   `computed_mask_select`, and it must show:
   - `materializer = rvv-route-entry-selected-body-realization`;
   - `route_entry_realization = true`;
   - pre-realized body consumption;
   - provider-owned computed-mask select and compare/select route facts;
   - no `--tcrv-materialize-selected-lowering-boundaries` step.
8. Real `ssh rvv` evidence must execute the representative direct route-entry
   generated bundle over counts including `0`, small, exact/vector-like, tail,
   and stress cases.
9. The runtime oracle must distinguish true compare/select from unmasked copy,
   inverted predicate, wrong comparison, wrong true/false arm, stale mask,
   missing runtime `n` control, and tail clobbering.
10. Preserve focused non-regression for recent direct route-entry MAcc and
    contraction owners.
11. Do not introduce route or executable authority from central ad hoc code,
    route ids, metadata, descriptors, ABI strings, scripts, artifact names,
    common EmitC, source-front-door fixtures, or legacy i32 helper surfaces.

## Acceptance Criteria

- [x] The `elementwise/compare-select` selected-body realization owner has an
      explicit bounded route-entry predicate for plain compare-select and
      computed-mask vector select.
- [x] C++ tests prove direct route-entry `computed_mask_select` is eligible
      through the owner registry, consumes the pre-realized body before route
      facts are collected, realizes compare/load/select/store structure, and
      reaches the computed-mask select route-family plan.
- [x] C++ tests prove the realized computed-mask select path reaches
      route-control provider facts, elementwise/select operand-binding facts,
      compare/select statement-plan facts, migrated statement-plan owner
      consumption, and provider-built route construction.
- [x] C++ fail-closed coverage covers at least wrong compare kind, wrong
      mask/source/layout, payload or output ABI/layout mismatch, runtime `n`
      ABI mismatch, and stale route-control or operand-binding facts before
      route statement construction.
- [x] A direct route-entry dry-run succeeds for `computed_mask_select` using
      `--pre-realized-selected-body --direct-pre-realized-route-entry` and does
      not run `--tcrv-materialize-selected-lowering-boundaries`.
- [x] The dry-run evidence carries explicit compare/select authority and
      mirror-only fields after provider route construction.
- [x] Real `ssh rvv` generated-bundle execution passes for direct route-entry
      `computed_mask_select` over counts including `0`, `7`, `16`, `23`, and
      `257`, with true/false lane coverage and tail sentinel preservation.
- [x] Focused non-regression covers direct route-entry `cmp_select`,
      direct route-entry `computed_masked_macc_add`, and recent direct
      route-entry contraction.
- [x] Bounded touched-file authority scan finds no new positive route or
      executable authority from legacy i32, source-front-door, descriptors,
      ABI strings, route ids, artifact names, scripts, common EmitC, or
      mirror-only metadata.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

- No dtype/LMUL clone batch or broad compare matrix.
- No broadcast, reduction, MAcc, contraction, memory, or frontend/Linalg
  coverage expansion beyond focused non-regression.
- No source-front-door positive route, dashboard, report-only task, broad smoke
  matrix, or evidence-only task.
- No common EmitC RVV semantics and no target/export inference of compare,
  select, mask, dtype/config, ABI, policy, intrinsic, or route support.
- No weakening of the direct route-entry MAcc or contraction owners that have
  already landed.

## Technical Approach

1. Start and validate this Trellis task, with implement/check context limited
   to specs.
2. Add an explicit RVV-selected-body route-entry predicate for compare/select
   that admits plain compare-select and computed-mask vector select and keeps
   unsupported compare/select cluster bodies bounded or fail-closed.
3. Extend direct route-entry C++ coverage to include pre-realized
   `computed_mask_select`, including realized IR shape, family/provider facts,
   statement-plan ownership, route construction, and fail-closed mutations.
4. Extend generated-bundle direct route-entry support and dry-run tests for
   `computed_mask_select`.
5. Run focused direct route-entry dry-runs for `computed_mask_select` and
   non-regression cases.
6. Run one representative `ssh rvv` generated-bundle execution for
   `computed_mask_select` over the accepted counts.
7. Run authority scan, `git diff --check`, focused C++/lit/script tests, and
   `check-tianchenrv`.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-compare-select-route-family-owner`
2. Focused C++ plugin test target or full C++ plugin test runner as available.
3. Direct route-entry dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind computed_mask_select --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --artifact-root artifacts/tmp/stage2_rvv_compare_select_route_family_owner --run-id direct-route-entry-computed-mask-select-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
4. Representative `ssh rvv` generated-bundle run for the same direct
   route-entry path and counts.
5. Focused non-regression dry-runs for direct route-entry `cmp_select`,
   direct route-entry `computed_masked_macc_add`, and recent direct
   route-entry contraction.
6. Bounded authority scan over touched production/test/script files.
7. `git diff --check`
8. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/guides/index.md`.
- Archived task context read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-route-entry-macc-realization-owner/prd.md`,
  `task.json`, `implement.jsonl`, and `check.jsonl`.
- Primary code surfaces:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related script/target
  compare-select tests.

## Open Questions

None blocking. Current code evidence identifies a bounded production gap:
compare/select route-entry ownership exists, but computed-mask vector select is
not explicit at the route-entry owner boundary and is not admitted by direct
route-entry generated-bundle evidence.

## Definition Of Done

The representative direct route-entry `computed_mask_select` workflow is
implemented and evidenced through RVV owner-scoped selected-body realization,
computed-mask select route-family/provider facts, compare/select statement
plan construction, generated bundle, and real `ssh rvv` correctness. Trellis
state is truthful, authority scans are clean, checks pass or exact blockers are
recorded, and one coherent commit records the completed task.

## Completion Evidence

- Production owner changed in `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`:
  the `elementwise/compare-select` route-entry predicate is now explicit for
  plain compare-select and computed-mask vector select route entries.
- Generated-bundle evidence changed in `scripts/rvv_generated_bundle_abi_e2e.py`:
  direct pre-realized route-entry support now admits computed-mask select,
  emits compare/select predicate boundaries, and validates generated EmitC
  compare/select structure.
- Focused C++ coverage changed in `test/Plugin/RVVExtensionPluginTest.cpp`:
  direct route-entry `computed_mask_select` reaches the computed-mask select
  family plan, provider route facts, operand bindings, migrated statement-plan
  owner, route construction, and targeted fail-closed diagnostics.
- Added dry-run lit coverage in
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-mask-select-dry-run.test`.
- Checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  focused direct route-entry dry-runs for `computed_mask_select`,
  `cmp_select`, `computed_masked_macc_add`, and contraction;
  real `ssh rvv` generated-bundle execution for `computed_mask_select` with
  counts `0,7,16,23,257`;
  bounded authority scan;
  `git diff --check`;
  `cmake --build build --target check-tianchenrv -j2` with 383/383 passing.
- Spec update judgment: no `.trellis/spec/` change was needed because the
  existing RVV plugin and EmitC route specs already describe the compare/select
  route-entry, provider-owned statement-plan, mirror metadata, fail-closed, and
  common EmitC neutrality contracts implemented here.
