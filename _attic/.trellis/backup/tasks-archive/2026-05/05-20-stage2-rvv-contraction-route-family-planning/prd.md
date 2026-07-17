# Stage2 RVV contraction route-family planning consolidation

## Goal

Consolidate the existing Stage2 RVV contraction-style executable routes behind
one bounded RVV plugin-owned route-family planning owner. The owner must derive
typed contraction facts from selected/realized `tcrv_rvv` body/config/runtime
structure and build `TCRVEmitCLowerableRoute` plans for the existing widening
macc and widening dot-reduction variants.

The task is a production-path refactor, not a new coverage matrix. Existing
routes must remain route-supported and executable:

- `widening_macc_add`
- `widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `strided_input_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

## Direction Source

- Hermes Direction Brief: `Stage2 RVV contraction route-family planning consolidation`.
- Module owner: RVV plugin-local contraction/memory route-family planning for
  existing widening macc, widening dot-reduction, computed-mask
  dot-reduction, strided-input dot-reduction, and masked strided-input
  dot-reduction executable slices.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b9b8d2cb rvv: add masked strided input widening dot slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- The previous five contraction-style slices are functionally closed and were
  added as bounded executable slices through typed RVV body/config/runtime
  facts, RVV selected-body realization, RVV route planning/provider emission,
  generated-bundle dry-run, and real `ssh rvv` evidence where executable
  correctness was claimed.
- Each slice currently carries substantial one-off logic through
  `RVVSelectedBodyRealization`, `RVVEmitCRoutePlanning`,
  `RVVEmitCRouteProvider`, construction protocol, target metadata, and
  generated-bundle support.
- The production bottleneck is not another clone slice. The next owner must
  make mask, stride, reduction, accumulator, dtype/config, tail policy,
  runtime count/AVL, and ABI-order facts flow through one coherent typed
  planning family instead of duplicating branch authority per operation.
- `tcrv.exec` declares ABI/runtime roles only. It does not own contraction
  semantics, dtype/config, mask provenance, stride semantics, reduction
  identity, route support, intrinsic spelling, or acceptance state.
- RVV route planning/provider owns contraction family validation, C/RVV vector
  type mapping, intrinsic-leaf selection, ABI ordering, header/artifact mirror
  metadata, and fail-closed diagnostics. Common EmitC/export remains neutral.

## Requirements

1. Factor the active RVV plugin route-planning path so the five existing
   contraction routes share an explicit bounded family-level planning
   structure or equivalent plugin-local owner.
2. The family owner must consume typed facts for:
   source operand configs, optional compare-produced mask, optional lhs/rhs
   element strides, accumulator vector input or scalar seed/result roles,
   runtime `n` / AVL, memory form, reduction/accumulation kind, tail/mask
   policy, ABI order, and target/header mirror data.
3. The provider must consume the family plan and remain the only layer that
   selects exact RVV intrinsic leaves after typed validation.
4. Unsupported combinations must fail closed with targeted diagnostics,
   including at least invalid mask, stride, accumulator/seed, dtype/config, and
   stale route-id authority cases.
5. Target/header metadata must stay mirror-only. Route ids, helper names,
   artifact strings, ABI strings, exact intrinsic spellings, descriptors, and
   common EmitC/export code must not become authority.
6. Existing selected-body realization and construction protocol behavior may
   be reorganized only to feed the shared family owner; it must not alter
   computation semantics, dtype semantics, parameter roles, variant origin,
   dispatch/fallback behavior, or runtime `n`/AVL values.

## Acceptance Criteria

- [x] The task PRD, implement/check context, and task metadata describe one
      bounded contraction route-family consolidation task.
- [x] `RVVEmitCRoutePlanning` exposes a bounded contraction-family plan
      structure or equivalent plugin-local owner that covers the five named
      production routes.
- [x] Production planning for `widening_macc_add`,
      `widening_dot_reduce_add`, `computed_masked_widening_dot_reduce_add`,
      `strided_input_widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add` is migrated to
      the shared owner without changing route semantics.
- [x] Provider emission consumes the family plan, while exact RVV intrinsic
      leaves remain provider-owned outputs derived after typed validation.
- [x] Common EmitC/export remains neutral and does not infer contraction
      semantics, dtype/config, masks, strides, reduction layout, accumulator
      roles, or intrinsic choices.
- [x] Negative fail-closed coverage exists for at least one invalid mask,
      stride, accumulator/seed, dtype/config, and stale route-id case after
      consolidation.
- [x] Generated-bundle dry-runs pass for representative unmasked, masked,
      strided, and masked-strided contraction routes.
- [x] If provider emission or generated C changes, real `ssh rvv` PASS exists
      for representative `widening_macc_add`, `widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add` counts.
- [x] Focused build/lit/unit/script checks pass for touched RVV dialect/config,
      selected-body realization, route planning, provider, construction
      protocol, target metadata, script, and artifact paths.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, `check-tianchenrv`, Trellis task validation,
      finish/archive, and one coherent commit are completed if the task is
      finished.

## Non-Goals

- No new RVV operation class, matmul lowering, high-level Linalg/Vector/StableHLO
  lowering, broad contraction framework, gather/scatter/indexed memory
  variants, dtype/LMUL clone batches, unsigned variants, source-front-door
  positive routes, one-intrinsic wrapper dialects, dashboards, reports, or
  metadata-only churn.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machines, Template/Toy examples, or performance claim.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, exact-intrinsic spelling
  authority, or compatibility wrapper preserving old i32 authority.
- No new dtype-prefixed helper op families such as
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or
  `tcrv_rvv.i32_macc`.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission. Python changes, if any,
  are limited to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Inspect the current one-off contraction planning/provider/realization
   branches and identify the common family inputs.
3. Refactor production planning so the named contraction routes use the shared
   family owner.
4. Run focused build targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
5. Run focused lit/FileCheck tests for route planning/provider, selected-body
   realization, generated artifacts, and negative fail-closed cases touched by
   this consolidation.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle paths are used.
7. Run generated-bundle dry-runs for representative unmasked, masked,
   strided, and masked-strided contraction routes.
8. Run real `ssh rvv` correctness for representative contraction routes if
   provider emission or generated C changes.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-masked-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-input-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-strided-widening-dot-reduction/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing widening macc, widening dot, masked dot, strided dot, and masked
  strided dot generated-bundle tests.

## Definition Of Done

- The named contraction routes share one bounded RVV plugin-owned planning
  family or equivalent owner.
- Fresh focused positive, negative, generated-bundle, and runtime evidence is
  current to this task when claimed.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.

## Completion Evidence

Implementation completed:

- Added `RVVSelectedBodyContractionRouteFamilyPlan` to
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`.
- Added contraction route-family derivation/application in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` for the five named
  contraction routes.
- Rewired `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` emission branches to
  consume `analysis.contractionRouteFamilyPlan` for widening macc, dot
  reduction, computed-mask dot reduction, strided-input dot reduction, and
  computed-mask strided-input dot reduction emission decisions.
- Common EmitC/export, RVVDialect, selected-body realization, construction
  protocol, target metadata, and generated-bundle script were not changed.

Self-repair:

- First focused build found a const-accessor compile error in the new family
  derivation helper. The helper was corrected to take mutable route analysis
  because the ODS op accessors used by the planner are non-const.

Checks and evidence:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-contraction-route-family-planning`
  passed before implementation.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed after self-repair.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-dialect-test` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-construction-protocol-common-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Generated-bundle dry-run passed for
  `widening_macc_add`, `widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` with counts
  `7,16,23` under
  `artifacts/tmp/20260520-contraction-family-dryrun/contraction-family-routes`.
- Real `ssh rvv` generated-bundle run passed for `widening_macc_add`,
  `widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` with counts
  `7,16,23`, including `lhs_stride=2` and `rhs_stride=3` for the masked
  strided dot case, under
  `artifacts/tmp/20260520-contraction-family-ssh/contraction-family-representative-ssh`.
- `git diff --check` passed.
- Active-authority scan over the current diff found only provider-side exact
  intrinsic leaves. A broader RVV active-path scan found existing fail-closed
  source-front-door code, RVV dialect parse-only legacy diagnostics, and
  negative test coverage, not new positive legacy/source/descriptor/common
  authority from this task.
- `cmake --build build --target check-tianchenrv -j2` passed:
  `245/245` tests.
