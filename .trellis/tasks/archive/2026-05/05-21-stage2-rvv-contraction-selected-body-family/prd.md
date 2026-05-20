# Stage2 RVV contraction selected-body realization family boundary

## Goal

Consolidate RVV plugin-local selected-body realization for the existing
contraction route family. The selected `tcrv.exec` RVV boundary to typed
`tcrv_rvv` body path must materialize the five existing pre-realized contraction
variants through one family owner or equivalent shared realization boundary
before route planning.

Covered routes:

- `widening_macc_add`
- `widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `strided_input_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

This is production-path selected-body realization consolidation. It is not a
new coverage slice, high-level frontend task, descriptor/source-front-door
task, metadata-only cleanup, or report-only guardrail.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV contraction selected-body realization family boundary`.
- Module owner: RVV plugin-local selected-body realization family for the
  existing contraction route family.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c0a17c7a rvv: validate contraction target leaf profiles`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.

## Current Repository Facts

- Commit `1d1c96fa` introduced a shared contraction route-family plan for the
  five named routes.
- Commit `c0a17c7a` moved target leaf/profile validation, provider support
  mirrors, abstract C type mapping, RVV type facts, load/product/reduction/store
  leaves, and inactive-lane zeroing validation into the plugin-owned family
  plan before provider emission.
- The remaining bottleneck is earlier: selected-body realization still risks
  accumulating per-route branches while route planning now expects structurally
  realized typed `tcrv_rvv` facts.
- `tcrv.exec` declares envelope, ABI roles, selected variant, and runtime
  values only. It must not infer RVV compute, dtype/config, mask provenance,
  stride semantics, accumulator/result layout, route support, or intrinsic
  spelling from ABI strings, route ids, artifact names, descriptors, or status
  mirrors.
- RVV selected-body realization may materialize legal generic typed RVV
  structure from pre-realized selected-body facts, but must not change
  computation semantics, dtype semantics, parameter roles, variant origin,
  dispatch/fallback behavior, or runtime `n` / AVL values.
- Route planning/provider remains authoritative after realization:
  selected-body realization creates typed body structure; the existing family
  plan validates route facts and provider emission consumes the family plan.

## Requirements

1. Refactor the active RVV selected-body realization path so the five named
   contraction pre-realized selected-body variants share a plugin-local
   contraction-family realization owner or equivalent shared boundary.
2. The family boundary must derive realized structure from typed
   body/config/runtime facts, including operation kind, memory form, optional
   compare-produced mask, optional lhs/rhs runtime strides, accumulator vector
   input or scalar seed/result layout, source/result configs, runtime `n`/AVL,
   ABI roles, and mask/tail policy.
3. Realization must materialize legal typed `tcrv_rvv` structure:
   `setvl`/`with_vl` placement, unit or strided source loads, optional compare
   mask use, widening product or masked-product shape, reduction or macc
   layout, scalar/vector store, and explicit runtime value uses.
4. Realized bodies must remain suitable inputs for the existing
   `RVVSelectedBodyContractionRouteFamilyPlan`; route planning still receives
   structural typed facts and remains the authority for provider emission.
5. Unsupported or incomplete pre-realized contraction bodies must fail closed
   with targeted diagnostics for missing/invalid mask provenance, stride roles,
   seed/accumulator/result roles, `n`/AVL, dtype/config, memory form, and stale
   route-id authority.
6. Existing route-supported and executable behavior for all five named routes
   must be preserved.
7. Common EmitC/export and target metadata must remain neutral mirror
   consumers. They must not infer RVV contraction semantics, dtype, stride,
   mask, accumulator, intrinsic, or route support.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe one bounded
      contraction selected-body realization family consolidation task.
- [x] Existing pre-realized selected-body paths for the five named contraction
      routes realize through a plugin-local family owner or equivalent shared
      boundary.
- [x] Realization derives setvl/with_vl placement, source loads, optional
      strided loads, optional compare-mask use, widening product or masked
      product, reduction or macc layout, scalar/vector store, and runtime value
      use from typed body/config/runtime facts.
- [x] Route planning still receives structurally realized typed `tcrv_rvv`
      facts, and the existing contraction family plan remains authority for
      provider emission.
- [x] Unsupported or incomplete pre-realized contraction bodies fail closed
      with targeted diagnostics for invalid mask provenance, stride roles,
      seed/accumulator/result roles, `n`/AVL, dtype/config, memory form, and
      stale route-id authority.
- [x] Positive FileCheck or equivalent evidence proves unmasked, masked,
      strided, and masked-strided pre-realized selected bodies realize through
      the family boundary into typed `tcrv_rvv` structure.
- [x] Generated-bundle dry-runs pass for all five named contraction routes.
- [x] If realized IR, route plans, provider emission, or generated C changes,
      real `ssh rvv` evidence remains PASS for `widening_macc_add`,
      `widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [x] Focused build/lit/C++/script checks pass for touched selected-body
      realization, RVV dialect/config if touched, construction protocol, route
      planning/provider if touched, scripts, and artifact paths.
- [x] Active-authority scan confirms old i32 route authority,
      source-front-door/source-seed authority, descriptor/direct-C/source-export
      authority, exact intrinsic public authority, and common/export RVV
      semantic authority are not reintroduced.
- [x] `git diff --check`, Trellis validation, task finish/archive, clean git
      status, and one coherent commit are completed if this task finishes.

## Non-Goals

- No new RVV operations or coverage, matmul lowering, high-level
  Linalg/Vector/StableHLO frontend work, gather/scatter/indexed memory,
  dtype/LMUL clone batches, unsigned variants, source-front-door positive
  routes, one-intrinsic wrapper dialects, dashboards, broad reports, or
  helper-only cleanup.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machines, Template/Toy examples, or performance claim.
- No move of RVV compute, dtype, stride, mask, accumulator, route support, or
  intrinsic semantics into common EmitC/export or target metadata.
- No inference from route ids, helper names, artifact strings, descriptors,
  exact intrinsic spelling, tests, ABI names, or status/result mirrors.
- No new dtype-prefixed helper op families such as
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or
  `tcrv_rvv.i32_macc`.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission. Python changes, if any, are limited
  to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Inspect current contraction selected-body realization, RVV dialect/config,
   construction protocol, route planning/provider, generated-bundle script, and
   focused route tests.
3. Refactor production selected-body realization so the five named contraction
   pre-realized paths share the family boundary.
4. Run focused build targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
5. Run focused lit/FileCheck tests for selected-body realization and
   fail-closed diagnostics touched by this consolidation.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for all five named contraction routes.
8. Run real `ssh rvv` correctness for representative contraction routes if
   realized IR, route plans, provider emission, or generated C changes.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-contraction-route-family-planning/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-contraction-target-leaf-profile-validation/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-masked-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-input-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-strided-widening-dot-reduction/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused artifact tests for the existing contraction routes.

## Completion Evidence

Implementation completed in this round:

- Added `RVVSelectedBodyContractionRealizationPlan` inside
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Added one plugin-local contraction realization helper that materializes
  `setvl`, `with_vl`, compare loads, unit or strided dot source loads,
  compare-mask production, widening macc / widening dot reduce /
  masked widening dot reduce compute, and scalar/vector store structure from
  typed pre-realized body/config/runtime facts.
- Rewired the five existing contraction pre-realized selected-body branches to
  validate their typed body facts, build the shared family realization plan,
  and call the shared helper.
- Preserved the downstream route-planning/provider authority boundary: route
  planning still consumes structurally realized typed `tcrv_rvv` facts and the
  existing `RVVSelectedBodyContractionRouteFamilyPlan` remains provider
  emission authority.
- Did not modify common EmitC/export, target metadata, RVV dialect ops/config,
  construction protocol, or generated-bundle script behavior.

Validation completed in this round:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Focused FileCheck passed for the five positive pre-realized contraction
  artifact tests:
  `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Focused fail-closed diagnostics passed for
  `generic-widening-macc-dataflow.mlir`,
  `strided-input-widening-dot-reduction-negative.mlir`,
  `computed-mask-widening-dot-reduction-negative.mlir`, and
  `computed-mask-strided-input-widening-dot-reduction-negative.mlir`.
- `cmake --build build --target tianchenrv-rvv-dialect-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-construction-protocol-common-test
  tianchenrv-target-artifact-export-test -j2` passed.
- `build/bin/tianchenrv-rvv-dialect-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-construction-protocol-common-test`, and
  `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Generated-bundle dry-run passed for all five named contraction routes at
  runtime counts `7,16,23` under
  `artifacts/tmp/20260521-contraction-realization-family-dryrun/contraction-realization-family-routes`.
- Real `ssh rvv` representative generated-bundle evidence passed for
  `widening_macc_add`, `widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` at counts `7,16,23`;
  the masked-strided route passed with `lhs_stride=2` and `rhs_stride=3`.
- Active-authority diff scan passed with no new `RVVI32M1`, `rvv-i32m1`,
  positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or public exact-intrinsic authority in
  this diff. Broader scan hits were existing negative/parser inventory and
  guardrail text.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 245/245 tests.

Self-repair performed:

- First focused build failed because `mlir::Location` cannot be default
  constructed inside a default-initialized plan. The plan was changed to keep
  operation/value facts only and derive location from the pre-realized body at
  materialization time.
- First focused FileCheck failed because the new shared helper materialized the
  computed mask before dot source loads, violating the existing construction
  order contract. The helper was corrected to preserve the previous order:
  compare input loads, dot source loads, compare, contraction compute, store.

## Definition Of Done

- The named contraction pre-realized selected bodies share a bounded
  RVV-plugin-owned realization family boundary or equivalent owner.
- Fresh focused positive, negative, generated-bundle, and runtime evidence is
  current to this task when claimed.
- No legacy/source/descriptor/common-export route authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.
