# Stage2 RVV Scalar-Broadcast Elementwise Route-Entry Owner

## Goal

Make standalone scalar-broadcast elementwise add a first-class RVV
route-entry path. The representative executable route is
`scalar_broadcast_add`: selected `tcrv.exec` RVV variant with an explicit
typed `tcrv_rvv.typed_binary_pre_realized_body`, RHS scalar runtime ABI value,
RVV plugin-owned selected-body realization into `setvl` / `with_vl` / load /
splat / binary / store, provider-owned scalar-broadcast elementwise route
facts, `TCRVEmitCLowerableRoute`, generated RVV artifact, and real `ssh rvv`
correctness evidence.

## Direction Source

- Direction title: `Stage2 RVV scalar-broadcast elementwise route-entry owner`.
- Module owner: RVV plugin-owned direct route-entry selected-body realization
  and provider boundary for standalone scalar-broadcast elementwise RVV
  computation.
- Representative route: `scalar_broadcast_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7cd43dd6 rvv: route widening conversion entry owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Specs require the RVV-first route authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- The scalar-broadcast elementwise route family already exists for realized
  `tcrv_rvv` structure. Existing C++ tests validate provider facts for
  `scalar_broadcast_add`, `scalar_broadcast_sub`, and `scalar_broadcast_mul`
  from already-realized `with_vl` bodies.
- `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`
  already proves the materialization pass can consume a pre-realized typed
  binary body into load / `tcrv_rvv.splat` / binary / store.
- The direct route-entry path is not yet explicit for standalone
  `scalar_broadcast_add`: the production route-path C++ test has no
  `rvv_pre_route_scalar_broadcast_add` direct route-entry case, and
  `scripts/rvv_generated_bundle_abi_e2e.py` does not list scalar-broadcast
  elementwise routes as `--direct-pre-realized-route-entry` supported.
- The route-entry owner predicate for the elementwise/compare-select owner is
  broad for typed binary pre-realized bodies. This round should make the
  scalar-broadcast add eligibility explicit enough for direct route-entry
  consumers without moving RVV semantics into provider/common EmitC or scripts.

## Requirements

1. Direct route-entry realization must accept the representative
   `scalar_broadcast_add` typed binary pre-realized body through the existing
   RVV selected-body realization owner registry.
2. Route-entry eligibility must be derived from typed body facts: operation
   kind `add`, memory form `rhs-scalar-broadcast`, SEW32, LMUL m1,
   tail/mask agnostic policy, runtime `rhs_scalar` ABI value, `lhs` input
   buffer, `out` output buffer, and runtime `n` / AVL.
3. Realization must consume the pre-realized body before provider facts are
   collected and must create one realized `setvl`, one `with_vl`, one vector
   load, one scalar splat, one binary add, and one store.
4. Provider construction must continue to consume the existing
   scalar-broadcast elementwise route-family plan, materialization facts,
   elementwise/select operand-binding facts, route-control provider plan, and
   elementwise arithmetic statement plan.
5. Unsupported or inconsistent scalar role/order, scalar dtype/config,
   memory form, op kind, lhs/out roles, runtime `n`, SEW/LMUL, policy, or
   stale route mirrors must fail closed before route construction or common
   EmitC.
6. Generated-bundle direct pre-realized route-entry mode must support
   `scalar_broadcast_add` and label the materializer as the route-entry
   selected-body realization path.
7. Runtime evidence must distinguish true runtime scalar broadcast add from
   lhs passthrough, constant scalar behavior, scalar-only fill,
   vector-vector fallback, wrong scalar sign/value, and tail clobbering.

## Acceptance Criteria

- [x] Focused production change in the RVV selected-body route-entry owner
      path or generated-bundle route-entry wiring makes standalone
      `scalar_broadcast_add` direct route-entry explicit.
- [x] C++ tests prove `rvv_pre_route_scalar_broadcast_add` is owner-registry
      route-entry eligible, is realized before route facts, reaches the
      scalar-broadcast elementwise provider plan, and emits one provider-built
      statement-plan loop.
- [x] C++ fail-closed coverage proves wrong or missing `rhs_scalar` runtime
      role and unsupported scalar-broadcast op kind are rejected before route
      construction.
- [x] Generated-bundle direct pre-realized dry-run for `scalar_broadcast_add`
      checks route-entry materializer, scalar-broadcast family plan,
      runtime ABI order, route-control facts, RHS scalar splat leaf, provider
      mirrors, emitted RVV C, and harness oracle metadata.
- [x] Real `ssh rvv` generated-bundle run for direct route-entry
      `scalar_broadcast_add` covers counts including `0`, a small count,
      exact-VL count, a tail count, and a stress count, with varied negative
      and positive scalar values.
- [x] Focused non-regression covers recent conversion, runtime-scalar MAcc,
      compare/select, MAcc, and contraction route-entry owners.
- [x] Bounded touched-file authority scan finds no new descriptor,
      direct-C/source-export, source-front-door, ABI-string-derived,
      artifact-name-derived, script-derived, common-EmitC-derived,
      route-id-derived, metadata-derived, exact-intrinsic-derived, or
      legacy-i32-derived route authority.
- [x] `git diff --check` passes; `check-tianchenrv` passes or an exact blocker
      is reported.
- [x] Trellis task status, journal/archive, and one coherent commit are
      truthful if complete.

## Implementation Result

- Tightened the elementwise/compare-select route-entry predicate in
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` so typed binary direct
  route-entry eligibility is based on RVV-owned operation kind and memory-form
  facts instead of accepting every typed binary pre-realized body by class.
- Added `rvv_pre_route_scalar_broadcast_add` production route-path coverage in
  `test/Plugin/RVVExtensionPluginTest.cpp`, proving direct route-entry
  realization consumes the typed binary pre-realized body before route facts,
  materializes load / RHS scalar splat / binary add / store structure, reaches
  the scalar-broadcast elementwise provider plan, and builds one provider
  statement-plan loop.
- Added owner-local negative tests for unsupported scalar-broadcast op kind
  and wrong `rhs_scalar` runtime ABI role.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` so direct pre-realized
  route-entry mode supports the representative `scalar_broadcast_add` route.
- Strengthened the scalar-broadcast elementwise generated harness oracle with
  active-lane checks that distinguish RHS runtime scalar influence, LHS vector
  influence, negative scalar sign/value behavior, and tail preservation.
- Added focused FileCheck dry-run coverage in
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-add-dry-run.test`.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-scalar-broadcast-elementwise-route-entry-owner`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- Focused direct route-entry dry-run artifact:
  `artifacts/tmp/stage2_scalar_broadcast_elementwise_route_entry/direct-pre-realized-scalar-broadcast-add`
- Focused lit/FileCheck:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter direct-pre-realized-scalar-broadcast-add`
  from `build/test`, passed 1/1 selected test.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Real `ssh rvv` artifact:
  `artifacts/tmp/stage2_scalar_broadcast_elementwise_route_entry_ssh/direct-pre-realized-scalar-broadcast-add-ssh-rvv`
- Real `ssh rvv` result covered counts `0,7,16,23,257` and RHS scalars
  `-37,19`, reporting `PASS op=scalar_broadcast_add counts=0,7,16,23,257 rhs_scalars=-37,19`.
  For nonzero counts, the harness reported RHS scalar influence, LHS influence,
  negative scalar sign-distinguishing lanes for `-37`, and `tail_preserved`.
- Non-regression direct route-entry dry-run artifact:
  `artifacts/tmp/stage2_scalar_broadcast_elementwise_route_entry_nonregression/direct-pre-realized-route-entry-nonregression`
- Non-regression op kinds:
  `widen_i16_to_i32`, `runtime_scalar_cmp_masked_macc_add`, `cmp_select`,
  `computed_mask_select`, `macc_add`, `scalar_broadcast_macc_add`, and
  `widening_macc_add`.
- Production diff authority scan:
  `git diff -U0 -- lib/Plugin/RVV/RVVSelectedBodyRealization.cpp scripts/rvv_generated_bundle_abi_e2e.py test/Plugin/RVVExtensionPluginTest.cpp | rg ...`
  returned no matches for descriptor, direct-C/source-export,
  source-front-door, legacy i32, ABI-string-derived, artifact-name-derived,
  route-id-derived, metadata-derived, exact-intrinsic-derived, or common-EmitC
  authority additions. The new FileCheck test mentions exact intrinsics only
  as post-provider emitted RVV C evidence and uses implicit negative checks for
  forbidden authority residue.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 386/386.

## Non-Goals

- No broad scalar-broadcast matrix and no new dtype/LMUL clone batch.
- No new MAcc/contraction variant, high-level frontend/Linalg route,
  source-front-door positive route, common EmitC RVV semantics, dashboard,
  report, or broad smoke matrix.
- No weakening of conversion, runtime-scalar MAcc, compare/select, MAcc,
  contraction, memory, reduction, or selected-body owner boundaries.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.

## Technical Approach

1. Add direct route-entry `scalar_broadcast_add` coverage to the production
   route-path C++ test fixture and, if needed, tighten owner route-entry
   eligibility around supported scalar-broadcast typed body facts.
2. Add fail-closed C++ mutations for wrong `rhs_scalar` role and unsupported
   scalar-broadcast op kind.
3. Extend `scripts/rvv_generated_bundle_abi_e2e.py` direct pre-realized
   route-entry support for `scalar_broadcast_add` only.
4. Add focused FileCheck dry-run coverage for direct pre-realized
   `scalar_broadcast_add`.
5. Run focused C++ tests, script self-test/dry-run, real `ssh rvv` evidence,
   non-regression dry-runs, authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-scalar-broadcast-elementwise-route-entry-owner`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
4. `./build/bin/tianchenrv-rvv-extension-plugin-test`
5. Focused dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/stage2_scalar_broadcast_elementwise_route_entry --run-id direct-pre-realized-scalar-broadcast-add --overwrite --op-kind scalar_broadcast_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 19 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
6. Focused lit/FileCheck script dry-run test for direct route-entry
   `scalar_broadcast_add`.
7. Real `ssh rvv` generated-bundle run for the same counts/scalars if the RVV
   host is reachable.
8. Focused non-regression dry-runs or tests for conversion, runtime-scalar
   MAcc, compare/select, MAcc, and contraction route-entry owners.
9. Bounded authority scan over touched files.
10. `git diff --check`
11. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Previous completed task read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-conversion-dtype-policy-route-family-owner/prd.md`.
- Production files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-*.test`.
