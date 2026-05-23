# Stage2 RVV computed-mask strided-load route-family interface

## Goal

Migrate the existing computed-mask strided-load memory consumer onto the
shared RVV plugin-local computed-mask memory route-family interface. The active
repository route is not named exactly `computed_masked_strided_load`; the
nearest active production route is:

- `computed_masked_strided_load_unit_store`
- `RVVSelectedBodyOperationKind::ComputedMaskStridedLoadUnitStore`
- `RVVSelectedBodyMemoryForm::ComputedMaskStridedLoadUnitStore`

The route must reuse the corrected producer-source-aware computed-mask memory
family mechanics from the previous task, while preserving its distinct
strided-load/load-merge/unit-store semantics. This is bounded strided-load
support only; indexed and segmented memory are explicitly out of scope.

## Direction Source

- Direction title: `Stage2 RVV computed-mask strided-load route-family interface`.
- Module owner: RVV plugin-local computed-mask strided-load memory
  route-family support for the existing strided-load memory consumer.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1870d18d rvv: share computed-mask memory producer sources`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes/Codex Direction Brief before source edits.

## Current Inventory

- The previous completed task introduced
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` and applied it to:
  - `runtime_scalar_cmp_masked_store`
  - `runtime_scalar_cmp_masked_load_store`
  - `computed_masked_unit_load_store`
  - `computed_masked_strided_store`
- The active strided-load consumer already exists as
  `computed_masked_strided_load_unit_store`; it structurally uses compare
  lhs/rhs unit loads, a compare-produced mask, an old output passthrough load,
  a source byte-strided masked load, and a final unit store.
- The shared computed-mask memory family currently records producer source,
  runtime control, target leaf/header/mirror facts, mask role/source/form,
  inactive-lane and passthrough contracts, source/destination memory forms, and
  the destination-stride facet for `computed_masked_strided_store`.
- The family interface is missing strided-load-specific family facts needed by
  `computed_masked_strided_load_unit_store`: source stride source, masked
  strided-load leaf, strided-load layout, source byte-stride ABI closure, and
  plan validation that the route consumes `tcrv_rvv.masked_strided_load`.
- Existing non-family description fallback code still sets some strided-load
  fields after route analysis. This task should move active authority for the
  strided-load consumer into the family plan/provider path and demote or remove
  duplicated one-off authority where it is covered by the family.

## Scope

1. Extend or repair the shared computed-mask memory route-family plan so it can
   represent the existing `ComputedMaskStridedLoadUnitStore` route.
2. Keep the already migrated scoped consumers intact:
   - `RuntimeScalarComputedMaskStore`
   - `RuntimeScalarComputedMaskLoadStore`
   - `ComputedMaskUnitLoadStore`
   - `ComputedMaskStridedStore`
3. Make `ComputedMaskStridedLoadUnitStore` consume the shared family for:
   - vector compare RHS producer source;
   - runtime AVL/VL and ABI order;
   - source byte-stride binding;
   - masked strided-load leaf and source memory form;
   - old output passthrough and final unit-store semantics;
   - provider-supported mirror, target leaf profile, header declarations, and
     C type mapping;
   - route operand binding closure.
4. Keep common EmitC/export neutral. The common path may materialize provider
   output only; it must not infer stride semantics, compare/mask source,
   passthrough behavior, dtype/config, runtime roles, or support state.

## Requirements

1. The family plan must carry or validate:
   - family plan id;
   - mask producer source: vector compare RHS load for this route;
   - store-only versus load-merge/store facet;
   - runtime AVL/VL control plan and runtime ABI order;
   - provider-supported mirror, target leaf profile, headers, and C type
     mapping;
   - SEW32 LMUL m1 vector and mask type facts from typed config;
   - setvl, vector load, compare, masked strided load, final store, and any
     passthrough load leaves needed by the route;
   - mask role/source/memory form;
   - source byte-strided masked-load memory form;
   - unit-stride output store memory form;
   - inactive-lane and passthrough layout contracts;
   - source byte-stride source and ABI binding.
2. `computed_masked_strided_load_unit_store` must require explicit typed
   structure: compare lhs load, compare rhs load, old output passthrough load,
   compare-produced mask, `tcrv_rvv.masked_strided_load`, final
   `tcrv_rvv.store`, runtime `n/AVL`, and runtime `src_stride_bytes`.
3. The provider must derive route support from typed body/config/runtime facts
   and `RVVRouteOperandBindingPlan` closure, not from route ids, helper
   strings, artifact names, mirrors, descriptors, source-front-door metadata,
   or common/export semantic inference.
4. Fail-closed diagnostics must cover missing source stride, missing compare
   input, missing mask source, missing passthrough/output role, missing
   `n/AVL`, wrong memory form, wrong producer source, bad dtype/config, stale
   route id, helper-string fallback, mirror-only authority, and common/export
   semantic inference where the current test surface can express them.
5. Existing computed-mask memory routes must keep their current semantics:
   runtime scalar store, runtime scalar load-store, vector unit load-store, and
   vector strided-store must remain supported through the same shared family.

## Acceptance Criteria

- [ ] `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` or an equivalent
      plugin-local interface carries strided-load-specific facts:
      masked-strided-load intrinsic, source stride source, strided memory
      layout, and source byte-stride ABI closure.
- [ ] `ComputedMaskStridedLoadUnitStore` is included in the shared
      computed-mask memory family validation and derivation path, not handled
      only by post-analysis one-off description fallback.
- [ ] The route-family plan validates that the active route has compare
      lhs/rhs loads, a compare-produced mask, old-output passthrough, a
      `tcrv_rvv.masked_strided_load`, final unit store, runtime `n/AVL`, and
      runtime `src_stride_bytes`.
- [ ] The provider emits or validates family-derived mirror/header/type fields
      for `computed_masked_strided_load_unit_store`, including
      `provider_supported_mirror`, `target_leaf_profile`, route-family id,
      mask producer source, source memory form, destination memory form,
      strided memory layout, and source stride source.
- [ ] The route operand binding plan remains the materialized operand
      authority for source, source stride, output, old passthrough, runtime
      count, compare lhs/rhs, and final store operands.
- [ ] Focused lit/FileCheck coverage proves family-derived stride source,
      closure, materialized operands, mirrors, headers, masked strided-load
      semantics, load-merge/final-store semantics, and fail-closed diagnostics.
- [ ] Generated-bundle dry-runs pass for migrated explicit and pre-realized
      `computed_masked_strided_load_unit_store` routes if both forms exist, at
      counts `7,16,23` with multiple stride values.
- [ ] Real `ssh rvv` evidence passes for the migrated strided-load route with
      multiple counts and stride values, checking active lanes, inactive
      passthrough, skipped stride slots, sentinel/tail preservation, and
      runtime `n` variation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, route-id, helper-string, artifact-name, mirror-only,
      or common/export RVV semantic authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No indexed memory, segmented memory, new store forms, new select or
  accumulation routes, new reduction kinds, mask algebra, dtype/LMUL clones,
  frontend lowering, source-front-door positive routes, dashboards,
  report-only work, helper-only cleanup, or broad all-memory rewrite.
- No collapse of strided-load, strided-store, and unit-load-store into
  ambiguous memory semantics. Only common computed-mask memory-family mechanics
  are shared.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  descriptor, direct-C, source-front-door, helper-string, route-id,
  artifact-name, or mirror-only route authority.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test` if touched.
3. Run focused dialect/verifier lit tests for computed-mask strided-load
   positives and negatives.
4. Run focused route planning/provider, EmitC, target artifact, and generated
   bundle FileCheck tests for `computed_masked_strided_load_unit_store`.
5. Run focused C++ tests where route-family validation or provider helpers are
   covered.
6. Run generated-bundle dry-runs for explicit and pre-realized
   `computed_masked_strided_load_unit_store`, counts `7,16,23`, with multiple
   stride values.
7. Run real `ssh rvv` correctness for the migrated route if executable
   correctness is claimed.
8. Run active-authority scans over touched RVV include/lib/script/test paths.
9. Run `git diff --check`.
10. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Implementation Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused tests for computed-mask strided-load/store and the shared
  computed-mask memory route-family.

## Definition Of Done

The active `computed_masked_strided_load_unit_store` production route consumes
the producer-source-aware computed-mask memory family for route support,
stride/load/passthrough/output facts, provider mirrors, headers, binding
closure, and emitted artifact evidence. Duplicated one-off strided-load
authority is removed or demoted where the family now owns it; indexed and
segmented memory remain untouched; focused tests, generated-bundle evidence,
real `ssh rvv` evidence, active-authority scan, `check-tianchenrv`, task
archive, clean status, and one coherent commit complete the round.

## Implementation Results

- Added `ComputedMaskStridedLoadUnitStore` to the shared
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` membership.
- Added strided-load family facts for source byte-stride source, strided memory
  layout, target leaf profile, provider-supported mirror, header declarations,
  C type mapping, and runtime ABI order.
- Made family validation require the active typed body to carry
  `tcrv_rvv.masked_strided_load`, old-output passthrough load, final
  `tcrv_rvv.store`, runtime `n/AVL`, and runtime `src_stride_bytes`.
- Made the route description verifier and provider require the shared
  computed-mask memory family plan for `computed_masked_strided_load_unit_store`.
- Preserved existing runtime-scalar store/load-store, vector unit load-store,
  and vector strided-store family behavior.
- Updated explicit and pre-realized target/header FileCheck coverage to prove
  the strided-load route now emits family plan id, mask producer source,
  provider-supported mirror, target leaf profile, headers, C type mapping,
  source memory form, source stride source, and binding closure facts.
- Updated generated-bundle evidence tooling metadata expectations for the new
  strided-load family facts.

## Validation Results

- [x] Task context validation:
      `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-23-stage2-rvv-computed-mask-strided-load-route-family-interface`.
- [x] Focused build:
      `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [x] Focused explicit/pre-realized FileCheck runs for
      `explicit-selected-body-artifact-computed-masked-strided-load.mlir` and
      `pre-realized-selected-body-artifact-computed-masked-strided-load.mlir`.
- [x] Focused negative FileCheck:
      `explicit-selected-body-computed-masked-strided-load-negative.mlir`.
- [x] C++ tests:
      `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Script checks:
      `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [x] Generated-bundle dry-runs passed for explicit and pre-realized
      `computed_masked_strided_load_unit_store`, counts `7,16,23`, stride
      bytes `8,12`:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-strided-load-family-explicit-dry-run`
      and
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-strided-load-family-prerealized-dry-run`.
- [x] Real `ssh rvv` evidence passed for explicit and pre-realized
      `computed_masked_strided_load_unit_store`, counts `7,16,23`, stride
      bytes `8,12`; both runs reported active-lane checks, inactive
      passthrough preservation, source gap preservation, tail preservation, and
      `PASS op=computed_masked_strided_load_unit_store counts=7,16,23 stride_bytes=8,12`:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-computed-mask-strided-load-family-explicit`
      and
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-computed-mask-strided-load-family-prerealized`.
- [x] Active-authority scan over touched RVV include/lib/script/test paths
      found no newly added positive legacy i32/RVVI32M1/source-front-door/
      descriptor/common-export authority. Existing matches in touched files
      are pre-existing negative/fail-closed/support-tool strings.
- [x] Added-line forbidden-authority scan over the diff found no forbidden
      additions.
- [x] `git diff --check`.
- [x] `cmake --build build --target check-tianchenrv -j2` passed 349/349.

## Spec Update Decision

No `.trellis/spec/**` change is required. This round applied the existing
RVV plugin-owned route-family, typed-body authority, common EmitC neutrality,
and real RVV evidence rules to one bounded active route; it did not introduce
a new durable architecture rule.
