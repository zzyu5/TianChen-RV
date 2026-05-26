# Stage2 RVV Masked Elementwise Route-Family Owner

## Goal

Strengthen the RVV plugin-owned route-family boundary for standalone masked
elementwise binary arithmetic, centered on the existing `masked_add`
representative as the current typed `computed_masked_add` shape: a selected
RVV variant with explicit typed `tcrv_rvv` structure, compare-produced mask,
lhs passthrough for inactive lanes, runtime AVL, and tail/mask policy facts.
The production path is selected `tcrv.exec` RVV variant -> typed or
pre-realized `tcrv_rvv` masked binary body -> RVV selected-body realization
when needed -> RVV-owned elementwise arithmetic family plan, route-control
plan, residual operand-binding facts, and statement plan -> provider-built
`TCRVEmitCLowerableRoute` -> common EmitC -> generated RVV artifact and real
`ssh rvv` evidence.

## Direction Source

- Direction title: `Stage2 RVV masked elementwise route-family owner`.
- Module owner: RVV plugin-owned selected-body realization and
  planning/provider boundary for standalone masked elementwise binary
  computation.
- Representative executable route: one i32 `masked_add` route that represents
  computed-mask masked add in the current typed surface:
  `compare(lhs, rhs) -> masked_binary(mask, lhs passthrough, lhs, rhs) ->
  store`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f2349729 rvv: strengthen standalone reduction route owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Specs require RVV route support to derive from typed `tcrv_rvv`
  body/config/runtime facts, with RVV plugin-owned realization and provider
  route construction before common EmitC/export.
- Existing explicit and pre-realized `masked_add` fixtures already carry the
  compare-produced mask and passthrough-preserving inactive-lane dataflow:
  `tcrv_rvv.compare` feeds `tcrv_rvv.masked_binary`, and the passthrough
  operand is the loaded `lhs` vector.
- Existing provider code has an elementwise arithmetic route-family plan,
  residual operand-binding facts, migrated elementwise arithmetic statement
  plan, and generated-bundle support for `masked_add`.
- The useful production gap is not to add a parallel route or an add/sub/mul
  clone batch. This round should strengthen masked elementwise ownership by
  adopting it into the route-control provider-plan boundary and by tightening
  mask/tail generated-bundle evidence for the representative route.

## Requirements

1. Keep the executable representative to one computed-mask masked add shape.
   Use the existing `masked_add` operation spelling unless code evidence shows
   a new spelling is required; do not add a broad masked add/sub/mul, dtype,
   or LMUL clone batch.
2. Masked elementwise route-control must be an RVV plugin-owned provider
   boundary that joins typed config facts, selected target capability facts,
   runtime AVL/VL control, runtime ABI order, tail policy, and mask policy
   before route statement construction.
3. The elementwise arithmetic statement plan must fail closed if masked
   elementwise routes lack the route-control plan, carry stale materialization
   facts from another selected analysis, have stale runtime `n`/AVL facts, or
   have stale SEW/LMUL/tail/mask policy/selected capability facts.
4. Existing masked dataflow checks must remain structural: the mask must be
   produced by `tcrv_rvv.compare` in the same VL scope, `masked_binary` must
   consume that mask, inactive passthrough must come from the loaded lhs
   vector, active lhs/rhs operands must come from typed loads, and output must
   be a typed store.
5. Provider route facts and generated artifact mirrors may expose
   provider-supported status, route operand bindings, mask role/source,
   inactive-lane contract, and policy only after route construction. They must
   not become route authority.
6. Generated-bundle dry-run and RVV runtime evidence must distinguish true
   masked add from unmasked add, inverted mask, passthrough-only behavior,
   lhs/rhs passthrough confusion, stale mask reuse, inactive-lane clobbering,
   tail clobbering, and metadata-derived behavior.
7. Recent reduction, scalar-broadcast, conversion, runtime-scalar,
   compare/select, MAcc, contraction, and memory owners must not regress.
8. Bounded touched-file scans must show no new descriptor, source-front-door,
   direct-C/source-export, ABI-string-derived, route-id-derived,
   artifact-name-derived, script-derived, common-EmitC-derived,
   metadata-derived, exact-intrinsic-derived, or legacy-i32-derived route
   authority.

## Acceptance Criteria

- [x] Production RVV planning/provider code adopts masked elementwise
      arithmetic into route-control provider ownership.
- [x] The route-control owner registry tests prove masked elementwise is
      classified exactly once and exposes a masked elementwise control flag.
- [x] Statement-plan tests prove `masked_add` consumes route-control facts
      before route statement construction and rejects stale runtime ABI,
      policy, capability, and materialization-analysis facts.
- [x] Direct selected-body or pre-realized route tests prove the
      compare-produced mask, same-VL use, lhs passthrough, active lhs/rhs
      operands, runtime `n`, and store shape reach provider-built route facts.
- [x] Generated-bundle dry-run for `masked_add` covers counts including zero,
      small, exact-VL, tail, and stress cases, and exposes a
      `mask_tail_policy_boundary` summary for masked elementwise evidence.
- [x] Real `ssh rvv` generated-bundle run for `masked_add` covers the same
      representative route over counts including zero, small, exact-VL, tail,
      and stress cases.
- [x] Focused non-regression covers recent standalone reduction,
      scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
      contraction, and memory route-entry paths.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
      focused C++ tests, focused lit/FileCheck or dry-run checks, bounded
      authority scan, `git diff --check`, and `check-tianchenrv` pass or exact
      blockers are reported.
- [x] Trellis task state, journal/archive, and one coherent commit are
      truthful if complete.

## Implementation Result

- Added `controlsMaskedElementwiseArithmetic` to the RVV route-control provider
  plan and registered a `masked elementwise arithmetic` route-control owner.
- Made masked elementwise statement planning require the RVV-owned
  route-control provider plan before attaching compare, active add,
  passthrough merge, and store steps.
- Extended C++ coverage so `masked_add` proves route-control joins typed
  config facts, selected target capability facts, runtime AVL, ABI order, and
  tail/mask policy before statement planning.
- Added C++ fail-closed coverage for stale runtime ABI order, stale tail
  policy, stale selected capability facts, and stale materialization facts
  from another selected route analysis.
- Extended generated-bundle evidence for masked elementwise routes with
  `mask_tail_policy_boundary`, including compare-produced mask source,
  mirror-only metadata role, active compute intrinsic, merge intrinsic,
  inactive-lane contract, selected ABI order, and runtime counts.
- Updated explicit and pre-realized masked-add dry-run tests to cover
  `0,7,16,23,257`.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so masked
  elementwise route-control adoption is a durable RVV plugin contract.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-masked-elementwise-route-family-owner`
  - Result: implement/check context valid.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - Result: `RVV extension plugin smoke test passed`.
- Explicit selected-body masked-add generated-bundle dry-run:
  `artifacts/tmp/stage2_masked_elementwise_route_family_owner/masked-add-dry-run`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
  - Counts: `0,7,16,23,257`.
  - Evidence includes `mask_tail_policy_boundary` with typed RVV authority,
    compare-produced mask producer, active compute intrinsic, masked merge
    intrinsic, provider mirrors, ABI order, and runtime-count summary.
- Pre-realized selected-body masked-add generated-bundle dry-run:
  `artifacts/tmp/stage2_masked_elementwise_route_family_owner/pre-realized-masked-add-dry-run`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
  - Counts: `0,7,16,23,257`.
- Focused lit/FileCheck:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'masked-add-dry-run'`
  from `build/test`
  - Result: passed 2/2 selected tests.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Real `ssh rvv` generated-bundle run:
  `artifacts/tmp/stage2_masked_elementwise_route_family_owner_ssh/masked-add-ssh-rvv`
  - Result: `PASS op=masked_add counts=0,7,16,23,257`.
  - Per-case output reported true and false mask lanes, plus
    `passthrough_preserved_lanes`, for all nonzero counts.
- Focused non-regression direct route-entry dry-run:
  `artifacts/tmp/stage2_masked_elementwise_route_family_owner_nonregression/direct-pre-realized-route-entry-nonregression`
  - Op kinds: `scalar_broadcast_add`, `widen_i16_to_i32`,
    `runtime_scalar_cmp_masked_macc_add`, `cmp_select`,
    `computed_mask_select`, `macc_add`, `scalar_broadcast_macc_add`,
    `widening_macc_add`, `widening_dot_reduce_add`,
    `standalone_reduce_add`, and `strided_load_unit_store`.
- Bounded authority scan over touched production/test/script files found no
  new descriptor, source-front-door, source-export/direct-C, legacy-i32,
  route-id-derived, artifact-name-derived, script-derived, common-EmitC-
  derived, metadata-derived, ABI-string-derived, or status-derived authority
  additions. Added exact intrinsic strings are evidence assertions for
  provider-emitted artifacts only, not route authority.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`
  - Result: 386/386 passed.

## Spec Update Judgment

Spec update was required. This task changed the long-term RVV route-control
contract by adopting masked elementwise arithmetic into the route-control
provider-plan boundary. The update was recorded in
`.trellis/spec/extension-plugins/rvv-plugin.md` under the Route-Control
Provider-Plan Boundary and Elementwise Arithmetic Statement-Plan Boundary.

## Non-Goals

- No broad masked-op matrix.
- No masked add/sub/mul expansion, dtype/LMUL clone batch, or new high-level
  frontend/Linalg route.
- No new source-front-door positive route.
- No common EmitC RVV semantic inference.
- No descriptor-driven computation or direct source/C exporter route.
- No dashboard/report/evidence-only task.
- No weakening of recently landed reduction, scalar-broadcast, conversion,
  runtime-scalar, compare/select, MAcc, contraction, or memory owners.

## Technical Approach

1. Add masked elementwise arithmetic to the RVV route-control provider owner
   registry and expose a dedicated plan flag.
2. Make the elementwise arithmetic statement-plan boundary require that
   route-control plan for masked elementwise before building compare/add/merge
   statements.
3. Extend focused C++ coverage around route-control owner classification,
   masked add statement-plan consumption, and stale dependency diagnostics.
4. Extend generated-bundle evidence for masked elementwise routes with a
   `mask_tail_policy_boundary` summary and run focused dry-run/runtime
   representative checks.
5. Run targeted non-regression, scans, formatting checks, and the project
   quality gate.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-masked-elementwise-route-family-owner`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
4. `./build/bin/tianchenrv-rvv-extension-plugin-test`
5. Focused generated-bundle dry-run for `masked_add` with counts
   `0,7,16,23,257`.
6. Focused lit/FileCheck or generated-bundle dry-run tests for the masked add
   artifact/evidence.
7. Real `ssh rvv` generated-bundle run for `masked_add` with the same counts
   if the RVV host is reachable.
8. Focused non-regression dry-runs/tests for standalone reduction,
   scalar-broadcast, widening conversion, runtime-scalar MAcc, compare/select,
   MAcc, contraction, and memory route-entry paths.
9. Bounded authority scan over touched production/test/script files.
10. `git diff --check`
11. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous completed task read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-standalone-reduction-route-family-owner/prd.md`.
- Production files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  masked add generated-bundle tests under `test/Scripts/`,
  and directly related compare/select, masked MAcc, scalar-broadcast,
  reduction, memory, and generated-bundle non-regression tests.
