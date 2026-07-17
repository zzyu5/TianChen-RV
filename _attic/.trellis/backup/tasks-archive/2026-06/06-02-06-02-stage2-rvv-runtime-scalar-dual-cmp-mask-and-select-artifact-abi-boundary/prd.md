# Stage2 RVV runtime-scalar dual-cmp mask-and-select artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV compiler path fail-closed and executable:

```text
selected tcrv.exec RVV variant
  -> typed_runtime_scalar_dual_compare_mask_and_select_pre_realized_body
  -> RVV plugin-local realization into setvl / with_vl / load / splat /
     load / splat / load / load / compare / compare / mask_and / select /
     store
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact validation
  -> generated bundle/header/object
  -> ssh rvv correctness evidence
```

Route authority must remain the typed/realized `tcrv_rvv` body and RVV
plugin/provider facts. Route ids, artifact names, metadata mirrors, exact
intrinsic spellings, descriptors, C strings, scripts, runtime counts, test
names, direct route-entry support, or source-front-door markers are
mirrors/evidence only.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar dual-cmp mask-and-select artifact ABI boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no entries.
* Initial `git log --oneline -8` started at
  `21d00baf rvv: validate signed widening macc artifact facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md` require selected typed RVV
  body -> RVV plugin-local realization/provider facts -> common EmitC ->
  target artifact -> real `ssh rvv` evidence for runtime/correctness claims.
* The archived signed widening MAcc task provides the closest completed
  production pattern: provider-owned facts, target validator consumption,
  generated-bundle dry-run evidence, real `ssh rvv` execution, task archive,
  and one coherent commit.
* Current repository evidence shows `runtime_scalar_dual_cmp_mask_and_select`
  already has a pre-realized fixture, generated-bundle script support, direct
  route-entry fail-closed test, provider/target support, and runtime harness
  skeleton.
* The main production gap is that the dual compare/select route-family facts
  are still duplicated between provider, target validation, script constants,
  and manual tests. In particular the current dual route operand binding
  summary is compacted too far: it omits full runtime ABI role/C-name structure
  and `hdr` participation for exported compare/scalar parameters.
* Target artifact metadata is bounded to 512 bytes per value. The corrected
  binding summary must preserve all logical operands, role/C-name facts,
  `abi`, `hdr`, and required use tokens; if needed, shorten only the provider
  plan label and use-token spellings.

## Requirements

* Keep support rooted in selected typed/pre-realized `tcrv_rvv` body,
  RVV plugin-local realization, computed-mask select route-family plan,
  compare/select statement plan, route operand binding facts, provider-built
  route facts, and target validator consumption.
* Support exactly m1/i32
  `runtime_scalar_dual_cmp_mask_and_select`.
* Validate runtime ABI order:
  `cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n`.
* Validate primary and secondary runtime scalar ABI roles:
  `rhs_scalar_a` as `rhs-scalar-value` and `rhs_scalar_b` as
  `rhs-secondary-scalar-value`.
* Validate two `tcrv_rvv.splat` realizations, two `sle`
  `tcrv_rvv.compare` producers, one `tcrv_rvv.mask_and` with composition
  `and`, one typed compute op `tcrv_rvv.select`, true/false value loads, and
  one result store.
* Validate mask facts:
  `predicate-mask-produced-by-mask-and`,
  `mask-and-of-two-runtime-scalar-compare-produced-masks`, and
  `composed-compare-produced-mask`.
* Validate select layout:
  `select-true-value-when-mask-else-false-value`.
* Provider route description must carry route operand binding plan/summary,
  computed-mask select route-family plan, runtime AVL/VL contract, target
  capability/config/profile facts, required headers, C type mapping, mask/tail
  policy owner, and explicit `provider_supported_mirror`.
* Introduce a plugin-owned dual compare/select fact surface consumed by both
  provider-side validation and target artifact validation. Target validation
  must not keep an independent duplicate source of truth for dual
  runtime-scalar constants.
* Correct the dual route operand binding summary so every generated
  header/prototype ABI parameter is represented as:
  `<logical>=<role>:<c-name>:abi|...|hdr`.
* Keep the corrected binding metadata bounded to 512 bytes by abbreviating the
  dual plan label and short use tokens, not by dropping logical operands,
  role/C-name facts, `abi`, `hdr`, or required compare/mask/select tokens.
* Fail closed for stale or missing provider facts, binding order, second
  scalar/second compare facts, predicate facts, mask composition/source/form,
  select layout, typed compute op, route-family plan, mask/tail policy plan,
  header/type facts, target capability/profile facts, provider mirror,
  direct route-entry residue, descriptor/direct-C/source-front-door residue,
  route-id/intrinsic authority, or common EmitC RVV semantic inference.
* Generated-bundle dry-run evidence must record provider-derived dual
  compare/select facts, complete route operand binding, target validator
  consumed facts, headers/types, `provider_supported_mirror`, runtime AVL/VL,
  two scalar thresholds, and the harness coverage contract.
* Runtime correctness must use real `ssh rvv` generated-bundle evidence over
  counts `0,1,16,23,257`, scalar threshold pairs derived from `-37` and `91`,
  and at least two input patterns proving both input masks, composed-mask true
  lanes, composed-mask false lanes, single-mask-only lanes, true/false value
  selection, source/tail sentinel preservation, and runtime `n`/AVL honoring.

## Acceptance Criteria

* [ ] Focused production diff strengthens the RVV plugin/provider and target
      validator fact surface for m1/i32
      `runtime_scalar_dual_cmp_mask_and_select`; no metadata-only closeout.
* [ ] Provider-owned dual compare/select route facts are exposed through a
      small accessor and consumed by target validation rather than duplicated
      target-local constants.
* [ ] Route operand binding plan/summary includes all eight ABI parameters
      with role, C name, `abi`, required use tokens, and `hdr`, while staying
      under the 512-byte target metadata value bound.
* [ ] Selected-body realization/FileCheck proves pre-realized body
      consumption into setvl/with_vl/load/splat/load/splat/load/load/compare/
      compare/mask_and/select/store.
* [ ] Target artifact tests fail closed for stale or missing provider mirror,
      target leaf/profile, ABI order/roles, binding plan/summary, primary and
      secondary scalar roles, second compare facts, predicate facts, mask
      composition/source/form, select layout, route-family plan, mask/tail
      policy plan, header/type facts, direct route-entry residue, and stale
      non-dual facts on single runtime-scalar routes.
* [ ] Existing or tightened REALIZED/PLAN/HEADER checks for the pre-realized
      dual-cmp mask-and-select fixture pass.
* [ ] Generated-bundle dry-run records provider-derived dual compare/select
      facts, route operand binding, target validator consumption, target
      leaf/profile, headers/types, `provider_supported_mirror`, mask-and
      composition, true/false select layout, and runtime AVL/VL facts.
* [ ] Real `ssh rvv` generated-bundle correctness passes for counts
      `0,1,16,23,257` and scalar threshold pairs from `-37` and `91`.
* [ ] Smallest relevant build/test/script commands, direct FileCheck
      equivalents if lit is unavailable, `git diff --check`, and a bounded
      old-authority scan over touched files pass.
* [ ] Trellis task is finished/archived and one coherent commit is created if
      the task completes.

## Out Of Scope

* Broad compare/select matrix, i64/LMUL m2 clone batch, bundle-only task,
  high-level Linalg/Vector frontend, source-front-door positive route,
  one-intrinsic wrapper dialect, report-only commit, common EmitC RVV
  semantics, widening MAcc redo, computed MAcc redo, runtime-scalar MAcc redo,
  reductions, segment/indexed memory, dashboards, descriptor routes,
  direct-C/source exporters, or direct route-entry positive support.
* Treating exact intrinsic spellings, route ids, artifact names, metadata
  mirrors, generated C strings, scripts, runtime count values, test names,
  fixture names, or harness constants as the source of compare predicates,
  mask composition, select layout, dtype/config, runtime ABI, policy, route
  support, or evidence authority.

## Technical Notes

Specs and context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-signed-widening-macc-add-artifact-abi-boundary/prd.md`

Repository files to inspect while deriving implementation:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-dual-cmp-mask-and-select-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`
