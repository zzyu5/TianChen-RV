# Stage2 RVV contraction explicit strided/masked selected-body realization

## Goal

Enable the remaining validated RVV contraction routes to run from explicit
selected bodies, not only pre-realized fixtures. A selected `tcrv.exec` RVV
variant with typed contraction body/config/runtime facts must flow through RVV
plugin-local selected-body realization, the existing validated contraction
family plan, provider materialization, `TCRVEmitCLowerableRoute`, generated
bundle artifact production, and real `ssh rvv` evidence for the three remaining
routes:

- `strided_input_widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

## Direction Source

- Direction title: `Stage2 RVV contraction explicit strided/masked selected-body realization`.
- Module owner: RVV plugin-local explicit selected-body realization and
  generated-bundle artifact boundary for the remaining validated contraction
  routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `3e9a0c96 rvv: enable explicit contraction selected bodies`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## What I Already Know

- Long-term specs require the RVV executable path to be:
  `tcrv.exec` envelope, selected RVV variant, typed low-level `tcrv_rvv` body,
  RVV plugin-owned legality/selected-body realization/provider,
  provider-built `TCRVEmitCLowerableRoute`, common EmitC materialization,
  target artifact, and real `ssh rvv` evidence for runtime/correctness claims.
- Previous task
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-contraction-explicit-selected-body-realization/`
  enabled explicit selected-body generated-bundle and real `ssh rvv` evidence
  for `widening_macc_add` and `widening_dot_reduce_add`.
- That previous task left the exact continuation point as explicit selected
  body fixtures and evidence for strided-input and computed-mask widening
  dot-reduce variants, while pre-realized preservation for all five active
  contraction routes remained validated.
- Earlier contraction ownership work already requires
  `rvv-contraction-route-family-plan.v1` and operation-specific
  `RouteOperandBindingPlan` closure for all five active contraction routes.
- This task must extend the explicit path into the already validated family
  plan/provider boundary. It must not create a script-only allowlist or infer
  stride/mask/contraction semantics from op-kind strings, route ids, artifact
  names, metadata mirrors, helper names, or common EmitC/export logic.

## Scope

1. Inventory only the explicit-selected-body rejection/expectation path and
   directly related RVV selected-body realization, contraction planning,
   provider, generated-bundle, target/header, and focused test code for the
   three named routes.
2. Add explicit selected-body fixtures and generated-bundle expectations for
   the three named routes in dependency order:
   - strided input widening dot-reduce;
   - computed-mask widening dot-reduce;
   - computed-mask strided input widening dot-reduce.
3. Require the realized body and provider evidence to structurally carry:
   - dot LHS/RHS roles;
   - accumulator/result contracts;
   - strided source facts where active;
   - computed-mask producer facts where active;
   - inactive-lane behavior where active;
   - runtime AVL/VL;
   - dtype/config;
   - target leaf/header facts;
   - `RouteOperandBindingPlan` closure before provider materialization.
4. Preserve base explicit contraction behavior, pre-realized route behavior,
   ABI order, route ids, runtime `n`/AVL behavior, dispatch/fallback behavior,
   target artifact contracts, and common EmitC/export neutrality.

## Requirements

1. Explicit-selected-body mode must support the three named routes only when
   typed body/config/runtime facts can be realized into the same validated
   contraction family plan consumed by provider materialization.
2. The explicit fixtures must use generic typed `tcrv_rvv` body structure and
   selected `tcrv.exec` RVV variant boundaries. They must not use legacy
   `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
   descriptor/direct-C/source-front-door, helper-string, artifact-name, or
   mirror-only authority.
3. Provider materialization must still require
   `rvv-contraction-route-family-plan.v1` and `RouteOperandBindingPlan`
   closure. Explicit-selected-body mode must not bypass planning/provider
   validation through script defaults or op-kind labels.
4. Generated-bundle explicit-selected-body evidence must check family plan id,
   operation and memory form, dot operand roles, accumulator/result contract,
   runtime count binding, target header/profile facts, provider support mirrors
   after route construction, and stride/mask distinctions where active.
5. Computed-mask explicit routes must prove the mask is structurally produced
   and consumed by the body/provider path, and that inactive lanes are skipped
   or preserved according to the route contract.
6. Strided-input explicit routes must prove source stride facts are structural
   in the body/provider path and that skipped source elements do not contribute
   to the widened dot.
7. FileCheck/lit coverage must prove explicit selected-body realization reaches
   the validated contraction family plan and binding closure, while preserving
   existing base explicit and pre-realized fixture behavior.
8. Real `ssh rvv` evidence must be collected for each enabled route with counts
   `7,16,23`, checking widened dot correctness, accumulator behavior, scalar
   output, runtime `n` variation, tail preservation, and stride/mask-specific
   behavior where applicable.

## Acceptance Criteria

- [ ] The task context references the RVV plugin, EmitC route, plugin protocol,
      testing, variant-pipeline specs, the previous explicit contraction task,
      and the previous contraction ownership task.
- [ ] The explicit-selected-body rejection/expectation path is inventoried
      without unrelated route-family expansion.
- [ ] Explicit-selected-body mode reaches the validated contraction family plan
      and provider route path for
      `strided_input_widening_dot_reduce_add`.
- [ ] Explicit-selected-body mode reaches the validated contraction family plan
      and provider route path for
      `computed_masked_widening_dot_reduce_add`.
- [ ] Explicit-selected-body mode reaches the validated contraction family plan
      and provider route path for
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [ ] Existing base explicit generated-bundle behavior for `widening_macc_add`
      and `widening_dot_reduce_add` remains passing.
- [ ] Existing pre-realized generated-bundle behavior for all active
      contraction routes remains passing.
- [ ] Focused lit/FileCheck coverage proves selected-body realization,
      contraction family plan id, operation/memory facts, dot operand roles,
      accumulator/result contract, strided source facts, computed-mask producer
      facts, mask inactive-lane behavior, route operand binding closure,
      header/profile mirrors, and fail-closed unsupported cases.
- [ ] Generated-bundle explicit-selected-body dry-runs pass across counts
      `7,16,23` for all enabled routes.
- [ ] Real `ssh rvv` explicit-selected-body runs pass for all enabled routes
      with stride/mask/correctness/tail/runtime count checks.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy or mirror-only route authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No new route families, dtype/LMUL clone batches, memory routes, standalone
  reductions, plain/computed-mask MAcc ownership changes, high-level
  Linalg/frontend routes, dashboards, broad smoke matrices, helper-only
  cleanup, or report-only evidence packaging.
- No script-only allowlist that lacks production selected-body
  realization/planning/provider evidence for the same typed-body-derived stride
  and mask facts.
- No change to base explicit contraction behavior, pre-realized route behavior,
  ABI order, route ids, runtime `n`/AVL behavior, dispatch/fallback behavior,
  target artifact contracts, or common EmitC/export neutrality.

## Validation Plan

1. Start the Trellis task after PRD/context repair.
2. Inventory explicit-selected-body fixtures and generated-bundle expectations
   in `scripts/rvv_generated_bundle_abi_e2e.py`, focused contraction tests, RVV
   selected-body realization, planning/provider, and target support bundle.
3. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit/FileCheck tests for explicit selected-body and pre-realized
   contraction route target/header/script fixtures.
5. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning,
   provider helpers, binding helpers, realization helpers, or target metadata
   helpers are changed.
6. Run generated-bundle explicit-selected-body dry-runs for the three named
   routes with counts `7,16,23`.
7. Run generated-bundle preservation dry-runs for existing base explicit routes
   and all pre-realized contraction routes with counts `7,16,23`.
8. Run real `ssh rvv` generated-bundle evidence for each enabled route with
   counts `7,16,23`.
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
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant prior tasks read:

- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-contraction-explicit-selected-body-realization/task.json`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-contraction-explicit-selected-body-realization/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-contraction-explicit-selected-body-realization/implement.jsonl`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-contraction-explicit-selected-body-realization/check.jsonl`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/task.json`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-contraction-widening-dot-reduce-route-family-ownership/prd.md`

Initial source surfaces to inspect:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Focused explicit and pre-realized contraction tests under `test/Target/RVV`
  and `test/Scripts`.

## Definition Of Done

The explicit selected-body path for the enabled strided/computed-mask
contraction routes constructs or realizes typed `tcrv_rvv` body facts into the
same validated contraction family plan required by the provider, emits
generated bundles without script-only authority, passes focused lit/FileCheck
and generated-bundle dry runs, has real `ssh rvv` correctness evidence,
preserves base explicit and pre-realized behavior, leaves common EmitC/export
neutral, archives the task truthfully, and records one coherent commit.

## Completion Evidence

Module behavior completed:

- Enabled explicit selected-body generated-bundle support for
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Added explicit `tcrv.exec` RVV selected variants with generic typed
  `tcrv_rvv` bodies for all three routes.
- The new explicit bodies structurally carry dot LHS/RHS roles,
  accumulator/result contracts, runtime AVL/VL, dtype/config, target leaf/header
  facts, and provider-validated `RouteOperandBindingPlan` closure.
- The strided route carries runtime `lhs_stride` and `rhs_stride` facts from
  explicit ABI values into strided source loads and generated bundle evidence.
- The computed-mask routes carry compare-produced mask facts, mask role/source,
  inactive-lane zeroing requirement, accumulator behavior, scalar output, and
  tail preservation evidence.
- Existing base explicit contraction routes and all five pre-realized
  contraction routes remain preserved.
- No common EmitC/export semantic branch, descriptor-driven compute path, new
  route family, new dtype/LMUL clone, high-level frontend route, or script-only
  allowlist was added.

Explicit routes enabled:

- `strided_input_widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

Variants intentionally left out:

- None. All three named continuation routes were enabled in this round.

Files changed:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/explicit-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-strided-input-widening-dot-reduce-add-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-widening-dot-reduce-add-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`
- This Trellis task record and workspace journal.

Selected-body realization and boundary evidence:

- Focused PLAN/HEADER FileCheck passed for the three new explicit selected-body
  fixtures.
- The PLAN checks prove selected-body operation kind, typed compute op,
  memory form, runtime ABI order, contraction family plan
  `rvv-contraction-route-family-plan.v1`, binding-plan closure, stride facts,
  mask facts, inactive-lane behavior, widening-dot relation, target leaf
  intrinsic metadata, provider-supported mirror, and target variant.
- The HEADER checks prove generated header ABI order and target metadata mirror
  for each route without making common export choose RVV semantics.
- Focused lit filter passed 6/6 selected tests for the three new target fixtures
  and three new generated-bundle script fixtures.

Generated-bundle evidence:

- Explicit dry-run passed for the three new routes with runtime counts
  `7,16,23`:
  `rvv_generated_bundle_abi_e2e: dry_run_success`.
- Base explicit preservation dry-run passed for `widening_macc_add` and
  `widening_dot_reduce_add` with counts `7,16,23`.
- Pre-realized preservation dry-run passed for `widening_macc_add`,
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` with counts
  `7,16,23`.

Real RVV evidence:

- Real `ssh rvv` explicit generated-bundle run passed for
  `strided_input_widening_dot_reduce_add` with counts `7,16,23`,
  `lhs_stride=2`, and `rhs_stride=3`, checking
  `strided_signed_horizontal_dot`, `seed_added`,
  `skipped_source_elements_ignored`, `scalar_output`, and `tail_preserved`.
- Real `ssh rvv` explicit generated-bundle run passed for
  `computed_masked_widening_dot_reduce_add` with counts `7,16,23`, checking
  `compare_masked_signed_horizontal_dot`, `seed_added`,
  `inactive_lanes_skipped`, `scalar_output`, and `tail_preserved`.
- Real `ssh rvv` explicit generated-bundle run passed for
  `computed_masked_strided_input_widening_dot_reduce_add` with counts
  `7,16,23`, `lhs_stride=2`, and `rhs_stride=3`, checking
  `compare_masked_strided_signed_horizontal_dot`, `seed_added`,
  `inactive_lanes_skipped`, `skipped_source_elements_ignored`,
  `scalar_output`, and `tail_preserved`.

Checks:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `git diff --check` passed.
- Active-authority scan over touched script/test paths found no
  `RVVI32M1`, `rvv-i32m1`, `!tcrv_rvv.i32m*`, or source-artifact authority.
  Remaining descriptor/source-export mentions are existing denylist/self-test
  guardrails; target intrinsic spellings in tests are provider-derived leaf
  metadata checks, not route authority.
- `cmake --build build --target check-tianchenrv -j2` passed 361/361.

Self-repair:

- Shortened explicit MLIR kernel and variant names after generated bundle export
  rejected overlong C/EmitC function identifiers.
- Reordered two PLAN FileCheck assertions to match the actual single-line
  diagnostic metadata order while preserving the stride/mask facts being
  checked.

Spec update judgment:

- No `.trellis/spec/**` update was needed. This round implements existing RVV
  plugin selected-body, route-provider, common EmitC neutrality, and testing
  contracts without adding a new long-term rule.

Continuation point:

- None. The three named explicit selected-body strided/computed-mask
  contraction routes are enabled and validated.
