# Stage2 RVV Standalone Reduction Route-Family Owner

## Goal

Strengthen the existing standalone vector reduction route-family owner around
the bounded `standalone_reduce_add` representative. The production path is:
selected `tcrv.exec` RVV variant with a typed
`tcrv_rvv.typed_standalone_reduce_pre_realized_body` -> RVV selected-body
route-entry realization -> realized `setvl` / `with_vl` / unit load /
`standalone_reduce` / scalar-result store structure -> RVV-owned standalone
reduction family plan, route-control plan, math operand-binding facts, and
statement plan -> provider-built `TCRVEmitCLowerableRoute` -> common EmitC ->
generated RVV artifact and `ssh rvv` evidence.

## Direction Source

- Direction title: `Stage2 RVV standalone reduction route-family owner`.
- Module owner: RVV plugin-owned route planning/provider and selected-body
  route-entry boundary for standalone vector reduction/accumulation.
- Representative executable route: one unit-stride i32 `standalone_reduce_add`
  that stores one scalar output under runtime AVL.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6c0e8b86 rvv: route scalar broadcast add entry owner`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  brief before source edits.

## Current Repository Facts

- Specs require the RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- The repository already has a named standalone reduction realization owner,
  route-family plan, route-control plan participation, math operand-binding
  facts, standalone reduction statement plan, and provider consumption.
- `test/Plugin/RVVExtensionPluginTest.cpp` already contains a direct
  `rvv_pre_route_standalone_reduce_add` route-entry consumer that reaches
  `rvv-standalone-reduction-route-family-plan.v1`.
- `scripts/rvv_generated_bundle_abi_e2e.py` already supports
  `standalone_reduce_add` in pre-realized direct route-entry mode, and
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run.test`
  already checks route-entry materialization, provider mirrors, reduction
  metadata, and harness oracle shape.
- The remaining useful production owner work is therefore not to add a new
  route beside the existing one. This round should improve the existing owner
  wiring/diagnostics/evidence, specifically around provider-owned runtime ABI
  validation and direct route-entry proof for the standalone-reduction realized
  body shape.

## Requirements

1. Keep `standalone_reduce_add` as the executable representative for this
   round; do not add a broad reduction opcode, dtype, or LMUL clone batch.
2. The production RVV provider must validate standalone reduction runtime ABI
   facts as provider-owned route-family facts, not as route id, artifact name,
   script, descriptor, ABI-string-derived semantic authority, or common EmitC
   inference.
3. The standalone reduction route-family plan must fail closed with targeted
   diagnostics when the source input, accumulator seed, scalar output, or
   runtime `n` ABI facts are stale or inconsistent with the typed body and
   route-family contract.
4. Direct route-entry tests must prove the pre-realized standalone reduction
   body is consumed before route facts and creates realized typed structure
   including one unit source load, one `tcrv_rvv.standalone_reduce`, and one
   scalar-result store.
5. Existing scalar-broadcast, widening conversion, runtime-scalar MAcc,
   compare/select, MAcc, contraction, memory, and computed-mask owners must not
   regress.
6. Generated-bundle evidence must cover direct pre-realized
   `standalone_reduce_add` with counts including zero, small, exact-VL, tail,
   and stress cases. Runtime evidence must distinguish true horizontal add
   reduction from first-lane/last-lane copy, elementwise add, constant result,
   missing seed initialization, wrong signed values, or tail clobbering.
7. Bounded scans over touched production/test/script files must show no new
   descriptor, source-front-door, source-export/direct-C, route-id-derived,
   artifact-name-derived, script-derived, common-EmitC-derived,
   metadata-derived, exact-intrinsic-derived, or legacy-i32-derived route
   authority.

## Acceptance Criteria

- [x] Production RVV planning/provider code strengthens standalone reduction
      route-family validation for provider-owned ABI facts.
- [x] C++ tests prove `rvv_pre_route_standalone_reduce_add` direct route-entry
      realizes the expected load / standalone reduction / store typed body
      before provider facts are collected.
- [x] C++ fail-closed coverage proves stale source input ABI and stale runtime
      `n` ABI facts are rejected with targeted diagnostics before route
      statement construction or common EmitC.
- [x] Generated-bundle dry-run for direct pre-realized `standalone_reduce_add`
      covers counts `0,7,16,23,257` and records route-entry realization,
      standalone reduction family plan, runtime ABI order, provider mirrors,
      scalar result boundary, statement plan, and harness oracle.
- [x] Real `ssh rvv` generated-bundle run for the same representative route
      covers counts including zero, small, exact-VL, tail, and stress cases.
- [x] Focused non-regression covers recent scalar-broadcast, widening
      conversion, runtime-scalar MAcc, compare/select, MAcc, contraction, and
      standalone reduction owners.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
      focused C++ tests, focused lit/FileCheck, authority scan,
      `git diff --check`, and `check-tianchenrv` pass or exact blockers are
      reported.
- [x] Trellis task status, journal/archive, and one coherent commit are
      truthful if complete.

## Implementation Result

- Strengthened
  `verifyRVVSelectedBodyStandaloneReductionScalarResultRuntimeABI(...)` in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` so standalone reduction
  route-family validation checks the full provider-owned runtime ABI sequence:
  plain `lhs,acc,out,n`, computed-mask `cmp_lhs,cmp_rhs,src,acc,out,n`, and
  runtime-scalar computed-mask `cmp_lhs,rhs_scalar,src,acc,out,n`.
- Kept validation inside RVV planning/provider ownership. Common EmitC,
  artifact metadata, route ids, scripts, descriptors, and ABI strings still do
  not infer reduction semantics.
- Extended the production route-path C++ test so direct
  `rvv_pre_route_standalone_reduce_add` proves the pre-realized body is
  consumed into one source load, one `tcrv_rvv.standalone_reduce`, and one
  scalar-result store before provider facts are collected.
- Added C++ fail-closed coverage for stale standalone reduction source input
  ABI and stale runtime `n` ABI role diagnostics.
- Updated the direct pre-realized standalone reduce-add dry-run FileCheck to
  include stress count `257`.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-standalone-reduction-route-family-owner`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
  - Result: `RVV extension plugin smoke test passed`
- Focused generated-bundle dry-run:
  `artifacts/tmp/stage2_standalone_reduction_route_family_owner/direct-pre-realized-standalone-reduce-add`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`
  - Counts: `0,7,16,23,257`
  - Evidence includes route-entry realization, pre-realized body consumption,
    standalone reduction family plan, scalar-result runtime boundary,
    provider-supported mirror, statement plan, and harness oracle.
- Focused lit/FileCheck:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-standalone-reduce-add`
  from `build/test`, passed 1/1 selected test.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused non-regression direct route-entry dry-run:
  `artifacts/tmp/stage2_standalone_reduction_route_family_owner_nonregression/direct-pre-realized-route-entry-nonregression`
  - Op kinds: `scalar_broadcast_add`, `widen_i16_to_i32`,
    `runtime_scalar_cmp_masked_macc_add`, `cmp_select`,
    `computed_mask_select`, `macc_add`, `scalar_broadcast_macc_add`,
    `widening_macc_add`, and `widening_dot_reduce_add`.
- Real `ssh rvv` artifact:
  `artifacts/tmp/stage2_standalone_reduction_route_family_owner_ssh/direct-pre-realized-standalone-reduce-add-ssh-rvv`
  - Result:
    `PASS op=standalone_reduce_add counts=0,7,16,23,257 seeds=-11,17`
  - Per-case output reported expected scalar sums and `tail_preserved` for
    both seeds.
- Bounded authority scan over touched production/test/script files returned no
  matches for descriptor, source-front-door, source-export/direct-C,
  legacy-i32, route-id-derived, artifact-name-derived, script-derived,
  common-EmitC-derived, metadata-derived, ABI-string-derived, or
  exact-intrinsic-derived authority additions.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`
  - Result: 386/386 passed.

## Spec Update Judgment

No `.trellis/spec/**` update is needed for this round. The durable contract was
already present in `extension-plugins/rvv-plugin.md`: standalone reduction must
derive route support from typed body/config/runtime facts, validate runtime ABI
and policy/provider facts in the RVV plugin, keep provider mirrors after route
construction, and keep common EmitC neutral. This implementation tightened code
and tests to that existing spec rather than creating a new long-term rule.

## Non-Goals

- No broad standalone reduction matrix.
- No new reduction dtype, LMUL, mask, min/max, contraction, MAcc, Linalg, or
  high-level frontend clone batch.
- No source-front-door positive route.
- No common EmitC RVV semantic inference.
- No descriptor-driven computation or direct source/C exporter route.
- No dashboard/report/evidence-only task.
- No weakening of recently landed scalar-broadcast, conversion,
  runtime-scalar, compare/select, MAcc, contraction, or memory owners.

## Technical Approach

1. Strengthen standalone reduction route-family ABI validation in
   `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, keeping it provider-owned
   and derived from the typed realized route analysis.
2. Extend focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` for
   direct route-entry realized standalone-reduction body shape and stale ABI
   diagnostics.
3. Update direct pre-realized standalone reduction dry-run coverage to include
   the stress count `257`.
4. Run focused build/tests, generated dry-run, `ssh rvv` representative run if
   reachable, non-regression dry-runs, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-standalone-reduction-route-family-owner`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
4. `./build/bin/tianchenrv-rvv-extension-plugin-test`
5. Focused direct dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/stage2_standalone_reduction_route_family_owner --run-id direct-pre-realized-standalone-reduce-add --overwrite --op-kind standalone_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
6. Focused lit/FileCheck for direct pre-realized standalone reduction dry-run.
7. Real `ssh rvv` generated-bundle run for `standalone_reduce_add` with the
   same counts if the RVV host is reachable.
8. Focused non-regression dry-runs/tests for scalar-broadcast, widening
   conversion, runtime-scalar MAcc, compare/select, MAcc, contraction, and
   standalone reduction route-entry paths.
9. Bounded authority scan over touched production/test/script files.
10. `git diff --check`
11. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/variant-pipeline/index.md`.
- Previous completed task read:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-scalar-broadcast-elementwise-route-entry-owner/prd.md`.
- Production files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run.test`,
  directly related standalone reduction generated-bundle tests.
