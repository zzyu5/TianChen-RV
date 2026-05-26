# Stage2 RVV Conversion Dtype Policy Route-Family Owner

## Goal

Make the existing RVV plugin-owned typed widening conversion route family
route-entry capable for the representative signed `i16mf2 -> i32m1`
conversion under runtime AVL. The production path must start from the selected
`tcrv.exec` RVV variant with
`tcrv_rvv.typed_widening_conversion_pre_realized_body`, realize it through the
RVV selected-body owner registry, consume typed conversion/dtype/SEW/LMUL/
policy/runtime facts through the existing widening conversion family,
route-control, operand-binding, materialization, and statement-plan boundaries,
then build a provider-owned `TCRVEmitCLowerableRoute` and generated RVV
artifact.

## Direction Source

- Direction title: `Stage2 RVV conversion/dtype policy route-family owner`.
- Module owner: RVV plugin-owned route planning/provider boundary for typed
  vector conversion and dtype/SEW/LMUL policy.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7fc274be rvv: route runtime scalar macc entry owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Current specs require the RVV-first route authority chain:
  selected `tcrv.exec` RVV variant -> typed/realized `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- `TypedWideningConversionPreRealizedBodyOp` and `WideningConvertOp` already
  carry conversion kind, source/result dtype, SEW, LMUL, policy, memory form,
  runtime `n`/AVL, and ABI roles in typed RVV structure.
- `RVVSelectedBodyWideningConversionRouteFamilyPlan`, materialization facts,
  math operand-binding facts, route-control provider plan, migrated
  statement-plan owner, generated-bundle evidence, and existing
  `widen_i16_to_i32` fixtures are already present.
- The exact conversion owner already exists with active route consumers, so
  this round must not be evidence-only. The production gap is route-entry
  wiring: the selected-body realization owner registry currently lists
  `"widening conversion"` with `isRouteEntryConsumer == nullptr`, and
  `--direct-pre-realized-route-entry` does not accept `widen_i16_to_i32`.
- The representative executable route remains bounded to signed
  `sign_extend_widen_vf2` / `signed-i16mf2-to-i32m1`, source
  `!tcrv_rvv.vector<i16, "mf2">`, result `!tcrv_rvv.vector<i32, "m1">`,
  unit-stride conversion memory, runtime `n`, and tail/mask agnostic policy.

## Requirements

1. Make the RVV selected-body realization owner for widening conversion
   route-entry capable only for supported typed widening conversion
   pre-realized bodies.
2. Direct route-entry realization must use the existing owner registry,
   consume the pre-realized body into `setvl`, `with_vl`, source load,
   `tcrv_rvv.widening_convert`, and store before route facts or provider
   route construction.
3. Route-entry conversion must continue to use the existing RVV-owned widening
   conversion route-family plan, materialization facts, math operand-binding
   facts, route-control plan, and migrated statement-plan boundary.
4. Conversion support and executable claims must derive from typed body/config
   facts and provider plans: source element type, result element type, source
   SEW/LMUL, result SEW/LMUL, signedness/conversion kind, conversion relation,
   memory form, runtime `n`/AVL, VL/tail policy, selected capability, operand
   bindings, headers, C vector types, and intrinsic leaves.
5. Unsupported or inconsistent conversion kind, signedness/relation,
   source/result dtype, SEW/LMUL relation, runtime ABI role/order, memory
   shape, or policy must fail closed before route construction or common
   EmitC.
6. Extend generated-bundle route-entry mode so `widen_i16_to_i32` can run
   through the direct pre-realized route-entry bridge and prove the route-entry
   materializer mirrors after provider route construction.
7. Add focused tests and dry-run evidence for direct route-entry
   `widen_i16_to_i32`, including true sign extension, destination width, loop
   VL, runtime count coverage, and tail preservation.
8. Run at least one real `ssh rvv` generated-bundle run for direct
   pre-realized `widen_i16_to_i32` over counts including `0`, a small count,
   exact-VL count, tail count, and a stress count, or record an exact blocker
   without claiming runtime correctness.

## Acceptance Criteria

- [x] `widening conversion` in the selected-body realization owner registry
      has owner-scoped route-entry eligibility for supported
      `TypedWideningConversionPreRealizedBodyOp` bodies.
- [x] Focused C++ tests prove the widening conversion owner is route-entry
      eligible, direct route-entry consumes the pre-realized body, provider
      route construction succeeds, and realized structure contains one
      setvl/with_vl/load/widening_convert/store chain.
- [x] Existing ordinary reduction or other non-route-entry families still fail
      closed, proving route-entry did not become an unbounded pre-realized
      allowlist.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` accepts direct
      pre-realized route-entry `widen_i16_to_i32` and labels the materializer
      as `rvv-route-entry-selected-body-realization`.
- [x] Focused FileCheck dry-run coverage checks direct route-entry
      `widen_i16_to_i32` evidence, provider mirrors, conversion relation,
      source/result SEW/LMUL, source/result C types, emitted sign-widen
      intrinsic, runtime counts, and harness sign-extension/tail checks.
- [x] Real `ssh rvv` generated-bundle evidence for direct route-entry
      `widen_i16_to_i32` covers counts `0,7,16,23,257`, or an exact blocker is
      recorded.
- [x] Non-regression focused checks cover recent runtime-scalar computed-mask
      MAcc, compare/select, MAcc, and contraction route-entry owners.
- [x] Bounded authority scan over touched RVV realization/planning/provider/
      script/test files finds no new descriptor, direct-C, source-export,
      source-front-door, ABI-string-derived, artifact-name-derived,
      script-derived, common-EmitC-derived, route-id-derived,
      metadata-derived, exact-intrinsic-derived, or legacy-i32-derived route
      authority.
- [x] `git diff --check` passes; `check-tianchenrv` passes or an exact blocker
      is reported.
- [x] Trellis task status, journal/archive, and one coherent commit are
      truthful if complete.

## Implementation Result

- Added the widening-conversion route-entry predicate in
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, using the existing typed
  pre-realized body structural checks for operation kind, memory form,
  conversion relation, signature, and policy.
- Registered `"widening conversion"` as an owner-scoped direct route-entry
  consumer and extended route-entry diagnostics to name it explicitly.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` direct
  pre-realized route-entry support and CLI diagnostics for
  `widen_i16_to_i32`.
- Extended `test/Plugin/RVVExtensionPluginTest.cpp` to prove owner registry
  coverage, provider route construction, route-family plan propagation, and
  realized load/convert/store structure.
- Added focused FileCheck dry-run coverage in
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widen-i16-to-i32-dry-run.test`.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-conversion-dtype-policy-route-family-owner`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- Direct pre-realized dry-run artifact:
  `artifacts/tmp/stage2_conversion_dtype_policy_route_entry/direct-pre-realized-widen-i16-to-i32`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Real `ssh rvv` artifact:
  `artifacts/tmp/stage2_conversion_dtype_policy_route_entry_ssh/direct-pre-realized-widen-i16-to-i32-ssh-rvv`
- Real `ssh rvv` result covered counts `0,7,16,23,257` and reported
  `sign_extension_checked tail_preserved` for every count plus
  `PASS op=widen_i16_to_i32 counts=0,7,16,23,257`.
- Non-regression direct route-entry dry-run artifact:
  `artifacts/tmp/stage2_conversion_dtype_policy_route_entry_nonregression/direct-pre-realized-route-entry-nonregression`
- Non-regression op kinds:
  `cmp_select`, `computed_mask_select`, `macc_add`,
  `runtime_scalar_cmp_masked_macc_add`, and `widening_macc_add`.
- Bounded touched-file authority scan found no new production route authority
  from descriptor, direct-C, source-export, source-front-door, ABI string,
  artifact name, script, common EmitC, metadata, route id, exact intrinsic
  spelling, or legacy i32 route forms. The new script test uses intrinsic
  spelling only as post-provider emitted-artifact evidence.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed after repairing
  one FileCheck order issue in the new dry-run test.

## Non-Goals

- No broad dtype/SEW/LMUL matrix.
- No additional conversion clone batch beyond the direct route-entry wiring
  needed for the existing `widen_i16_to_i32` representative.
- No high-level frontend/Linalg route, source-front-door positive route,
  common EmitC RVV semantics, descriptor/direct-C/source-export path,
  dashboard, report-only work, or broad smoke matrix.
- No weakening of runtime-scalar, compare/select, MAcc, contraction, memory,
  reduction, or selected-body owner boundaries.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.

## Technical Approach

1. Start and validate this Trellis task.
2. Add a widening-conversion route-entry consumer predicate in
   `RVVSelectedBodyRealization.cpp`, using existing structural validation
   helpers rather than route ids or artifact names.
3. Register that predicate for the `"widening conversion"` owner and update
   direct route-entry diagnostics.
4. Add focused C++ route-entry owner coverage for `widen_i16_to_i32`, including
   provider route construction and realized op counts.
5. Extend generated-bundle direct pre-realized route-entry allowlist/help text
   for `widen_i16_to_i32`.
6. Add or update a focused script dry-run FileCheck test for direct route-entry
   `widen_i16_to_i32`.
7. Run focused plugin/script checks, generated-bundle dry-run, real `ssh rvv`
   if available, authority scans, `git diff --check`, and `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-conversion-dtype-policy-route-family-owner`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/stage2_conversion_dtype_policy_route_entry --run-id direct-pre-realized-widen-i16-to-i32 --overwrite --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
5. Focused lit/FileCheck script tests for direct route-entry conversion.
6. Real `ssh rvv` generated-bundle run for the same counts if the RVV host is
   reachable.
7. Focused non-regression dry-runs or tests for runtime-scalar computed-mask
   MAcc, compare/select, MAcc, and contraction route-entry owners.
8. Bounded authority scan over touched files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Previous completed tasks:
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-widening-conversion-route-family-ownership/`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-widening-conversion-route-control/`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-conversion-sew-policy-route-closure/`,
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-runtime-scalar-abi-route-entry-owner/`.
- Production files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-*.test`.
