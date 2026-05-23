# Stage2 RVV contraction explicit selected-body realization

## Goal

Enable the active RVV contraction route family to execute from explicit
selected bodies, not only pre-realized fixtures. A selected `tcrv.exec` RVV
variant with typed contraction body/config/runtime facts must flow through RVV
plugin-local selected-body realization, the already validated contraction
family plan, provider materialization, `TCRVEmitCLowerableRoute`, generated
bundle artifact production, and representative real `ssh rvv` evidence.

## Direction Source

- Direction title: `Stage2 RVV contraction explicit-selected-body realization path`.
- Module owner: RVV plugin-local explicit selected-body realization and
  generated-bundle artifact boundary for the validated contraction route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `07122ba8 rvv: own contraction route family`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- Long-term specs require the RVV executable path to be:
  `tcrv.exec` envelope, selected RVV variant, typed low-level `tcrv_rvv` body,
  RVV plugin-owned legality/selected-body realization/provider,
  provider-built `TCRVEmitCLowerableRoute`, common EmitC materialization,
  target artifact, and real `ssh rvv` evidence for runtime/correctness claims.
- Previous task
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/`
  completed plugin-owned contraction route-family planning/provider ownership
  for:
  - `widening_macc_add`;
  - `widening_dot_reduce_add`;
  - `strided_input_widening_dot_reduce_add`;
  - `computed_masked_widening_dot_reduce_add`;
  - `computed_masked_strided_input_widening_dot_reduce_add`.
- That previous task proved pre-realized generated-bundle and `ssh rvv`
  evidence, and preserved route ids, ABI names/order, parameter roles, runtime
  `n`/AVL behavior, selected-body realization semantics, generated artifact
  contracts, and common EmitC/export neutrality.
- The immediate reported blocker is that the same contraction op kinds are
  rejected before bundle generation in `--explicit-selected-body` mode, with
  the script diagnostic that `widening_macc_add` is not supported in
  explicit-selected-body mode.
- This task should wire explicit selected-body construction/realization into
  the already validated family plan path. It must not change pre-realized route
  semantics or create a script-only allowlist unless the production RVV
  realization/planning/provider path proves the same typed-body-derived facts.

## Scope

1. Inventory only the explicit-selected-body rejection path and directly
   related contraction realization/planning/provider/generated-bundle code.
2. Enable explicit selected-body generation for one coherent contraction
   submodule, preferably the validated active family beginning with
   `widening_macc_add` and `widening_dot_reduce_add`.
3. Require the realized body to structurally carry:
   - operation kind;
   - memory form;
   - dot LHS/RHS roles;
   - accumulator role;
   - widening MAcc or dot-reduce relation;
   - strided-input facts where active;
   - computed-mask facts where active;
   - result layout;
   - runtime AVL/VL;
   - dtype/config;
   - target leaf/header facts;
   - `RouteOperandBindingPlan` closure before provider materialization.
4. Preserve existing pre-realized route behavior, ABI order, route ids,
   runtime `n`/AVL behavior, dispatch/fallback behavior, target artifact
   contracts, and common EmitC/export neutrality.

## Requirements

1. The explicit-selected-body harness and production RVV selected-body
   realization path must accept supported contraction op kinds only when their
   typed body/config/runtime facts can be realized into the same contraction
   family plan consumed by provider materialization.
2. `widening_macc_add` and `widening_dot_reduce_add` are the minimum coherent
   explicit-realization set for this round. Strided-input and computed-mask
   representatives may be included only if their typed explicit-body facts and
   evidence fit cleanly in the same round.
3. Provider materialization must still require the validated
   `rvv-contraction-route-family-plan.v1` plan and `RouteOperandBindingPlan`
   closure. Explicit-selected-body mode must not bypass planning/provider
   validation through op-kind strings, artifact names, metadata mirrors, or
   script defaults.
4. Generated-bundle explicit-selected-body evidence must check the same
   materialized facts as pre-realized fixtures: family plan id, operation and
   memory form, contraction relation, dot operand roles, accumulator/result
   contract, runtime count binding, target header/profile facts, and provider
   support mirrors after route construction.
5. FileCheck/lit coverage must prove explicit selected-body realization reaches
   the validated contraction family plan and binding closure, and must preserve
   existing pre-realized fixture behavior.
6. Real `ssh rvv` evidence must be collected for at least
   `widening_macc_add` and `widening_dot_reduce_add` in explicit-selected-body
   mode, with counts `7,16,23` and correctness checks for widened product or
   dot reduction, accumulator behavior, runtime `n` variation, and tail
   preservation.
7. If computed-mask or strided-input explicit contraction is enabled in this
   round, it must include evidence for active/inactive lane accounting or
   strided source facts as applicable.
8. No positive legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor/direct-C/source-front-door, helper-string,
   artifact-name, or mirror-only authority may be introduced.

## Acceptance Criteria

- [ ] The PRD and task context reference the RVV plugin, EmitC route, plugin
      protocol, testing, variant-pipeline specs, and previous contraction
      ownership task.
- [ ] The explicit-selected-body rejection path is inventoried without
      unrelated route-family expansion.
- [ ] Explicit-selected-body mode reaches the validated contraction family plan
      and provider route path for `widening_macc_add` and
      `widening_dot_reduce_add`.
- [ ] Any additional strided-input or computed-mask contraction variant enabled
      in this task has the same typed-body-derived realization, family-plan,
      binding-closure, generated-bundle, and `ssh rvv` evidence.
- [ ] Existing pre-realized generated-bundle behavior for all active
      contraction routes remains passing.
- [ ] Focused lit/FileCheck coverage proves selected-body realization,
      contraction family plan id, operation/memory facts, dot operand roles,
      accumulator/result contract, route operand binding closure, header/profile
      mirrors, and fail-closed unsupported cases.
- [ ] Generated-bundle explicit-selected-body dry-runs pass across counts
      `7,16,23` for the enabled explicit contraction routes.
- [ ] Real `ssh rvv` explicit-selected-body runs pass for `widening_macc_add`
      and `widening_dot_reduce_add`, plus a computed-mask or strided-input
      representative if enabled.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new route coverage beyond the existing validated contraction family.
- No dtype/LMUL clone batches, memory route work, standalone reductions,
  plain/computed-mask MAcc ownership changes, computed-mask select work,
  high-level Linalg/frontend routes, dashboards, broad test matrices,
  helper-only cleanup, or report-only evidence packaging.
- No computation semantics change, pre-realized behavior change, ABI order
  change, route id change, runtime `n`/AVL change, dispatch/fallback change,
  or target artifact contract change.
- No common EmitC/export ownership of contraction semantics, dtype/config,
  mask policy, intrinsic spelling, support state, runtime ABI, or evidence
  authority.

## Validation Plan

1. Validate and start the Trellis task.
2. Inventory explicit-selected-body rejection in
   `scripts/rvv_generated_bundle_abi_e2e.py`, focused contraction tests, RVV
   selected-body realization, planning/provider, and target support bundle.
3. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit/FileCheck tests for explicit selected-body and pre-realized
   contraction route target/header/script fixtures.
5. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning,
   provider helpers, binding helpers, realization helpers, or target metadata
   helpers are changed.
6. Run generated-bundle explicit-selected-body dry-runs for enabled contraction
   routes with counts `7,16,23`.
7. Run generated-bundle pre-realized dry-runs for existing active contraction
   routes to prove preservation.
8. Run real `ssh rvv` generated-bundle evidence for explicit
   `widening_macc_add` and `widening_dot_reduce_add`; add a computed-mask or
   strided-input representative if enabled.
9. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`

Relevant prior task read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/task.json`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/implement.jsonl`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/check.jsonl`

Initial source surfaces to inspect:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused contraction tests under `test/Target/RVV` and `test/Scripts`.

## Definition Of Done

The explicit selected-body path for the enabled contraction routes constructs
or realizes typed `tcrv_rvv` body facts into the same validated contraction
family plan required by the provider, emits generated bundles without
script-only authority, passes focused lit/FileCheck and generated-bundle dry
runs, has real `ssh rvv` correctness evidence, preserves pre-realized
behavior, leaves common EmitC/export neutral, archives the task truthfully, and
records one coherent commit.

## Completion Evidence

Production changes:

- Added explicit selected-body fixtures for `widening_macc_add` and
  `widening_dot_reduce_add` using already-realized generic `tcrv_rvv` body
  structure, not pre-realized helper ops.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` so
  `--explicit-selected-body --op-kind widening_macc_add` and
  `--explicit-selected-body --op-kind widening_dot_reduce_add` select those
  fixtures and verify generated-bundle evidence.
- Strengthened generated-bundle metadata verification for all contraction
  expectations to require
  `tcrv_rvv.contraction_route_family_plan =
  rvv-contraction-route-family-plan.v1`.
- Added focused generated-bundle dry-run tests for both explicit contraction
  routes, including plan, route binding, harness, and correctness markers.

Explicit routes enabled:

- `widening_macc_add`
- `widening_dot_reduce_add`

Variants intentionally left out:

- `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` remain pre-realized
  only in this round.
- Reason: the coherent submodule completed here is the base explicit
  contraction path for widening MAcc and plain widening dot-reduce. Extending
  strided-input and computed-mask explicit selected bodies should be a direct
  continuation using the same generic body/provider plan pattern, with their
  additional stride and mask evidence.

Selected-body realization and family-plan evidence:

- The explicit fixtures already contain generic `tcrv_rvv.setvl`,
  `tcrv_rvv.with_vl`, typed i16mf2 loads, `tcrv_rvv.widening_macc` or
  `tcrv_rvv.widening_dot_reduce`, and final store structure.
- The existing C++ RVV planning/provider path derives
  `rvv-contraction-route-family-plan.v1` and operation-specific
  `RouteOperandBindingPlan` closure from that generic typed body.
- The new PLAN and HEADER FileCheck coverage proves family plan id, operation
  kind, memory form, source/result dtype/config, accumulator/result layout,
  relation, runtime ABI order, route operand binding closure, and target header
  mirrors for both explicit routes.

Checks:

- Focused PLAN/HEADER FileCheck passed for
  `test/Target/RVV/explicit-selected-body-artifact-widening-macc-add.mlir`.
- Focused PLAN/HEADER FileCheck passed for
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`.
- Explicit generated-bundle dry-runs passed for `widening_macc_add` and
  `widening_dot_reduce_add` with counts `7,16,23`.
- Pre-realized preservation dry-run passed for all five active contraction
  op kinds with counts `7,16,23`.
- Real `ssh rvv` explicit generated-bundle run passed for `widening_macc_add`
  with counts `7,16,23`, checking signed products, accumulation, and tail
  preservation.
- Real `ssh rvv` explicit generated-bundle run passed for
  `widening_dot_reduce_add` with counts `7,16,23`, checking signed horizontal
  dot, scalar seed, scalar output, and tail preservation.
- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Active-authority scan over touched script/test/task files found only
  expected negative guardrail and FileCheck forbidden-token references.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 355/355.

Spec update judgment:

- No `.trellis/spec/**` update was needed. This task implemented the existing
  RVV plugin ownership, selected-body, common EmitC neutrality, and testing
  contracts without creating a new long-term rule.

Continuation point:

- If continuing the same family, add explicit selected-body fixtures and
  generated-bundle/ssh evidence for
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`, preserving the same
  contraction family plan and binding-closure requirements while adding
  stride/mask-specific structural checks.
