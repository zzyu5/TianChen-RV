# Evidence Summary: Stage2 RVV standalone reduction route-family ABI/provider ownership

## Production Movement

- Added `verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts` in the RVV route-planning/provider surface.
- Wired the standalone reduction statement plan and provider fact verification into selected-body RVV EmitC route construction before `TCRVEmitCLowerableRoute` construction.
- Expanded generated-bundle ABI metadata checks so standalone reduction artifact fields mirror provider/target facts for source/scalar-result type channels, runtime ABI order, mask/inactive policy, compare/merge/RHS-splat leaves, load/store/reduction leaves, and provider support mirrors.
- No common EmitC semantic branch was added. Common materialization remains a consumer of the provider-built route payload.
- No target-artifact production change was needed; existing target validation continued to consume rebuilt provider routes and route descriptions as mirror checks.

## Focused Checks

- `rtk git diff --check`: passed.
- `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: passed.
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`: passed.
- `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`: passed.
- `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`: passed with `RVV extension plugin smoke test passed`.
- `rtk cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`: passed after the first dry-run exposed a stale local `tcrv-opt` binary that did not know `tcrv_rvv.runtime_abi_value`.
- `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: passed, `Total Discovered Tests: 464`, `Passed: 464 (100.00%)`.

## Generated-Bundle Evidence

- Dry-run passed for `standalone_reduce_add` with counts `0,1,16,17,257`:
  `artifacts/tmp/05-31-stage2-rvv-standalone-reduction-provider/dry-run/pre-realized-standalone-reduce-add`.
- Dry-run passed for `computed_mask_standalone_reduce_add` with counts `0,1,16,17,257` and selected-boundary positive routing:
  `artifacts/tmp/05-31-stage2-rvv-standalone-reduction-provider/dry-run/pre-realized-computed-mask-standalone-reduce-add`.
- Dry-run passed for `runtime_scalar_cmp_masked_standalone_reduce_add` with counts `0,1,16,17,257` and RHS scalars `-37,91`:
  `artifacts/tmp/05-31-stage2-rvv-standalone-reduction-provider/dry-run/pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add`.
- Direct pre-realized route-entry probe failed closed for `standalone_reduce_add`, `computed_mask_standalone_reduce_add`, and `runtime_scalar_cmp_masked_standalone_reduce_add` with the expected retired-shortcut diagnostic.

## ssh rvv Evidence

- `ssh rvv` generated-bundle correctness passed for `standalone_reduce_add` over counts `0,1,16,17,257` and seeds `-11,17`.
- `ssh rvv` generated-bundle correctness passed for `computed_mask_standalone_reduce_add` over counts `0,1,16,17,257` and seeds `-11,17`.
- `ssh rvv` generated-bundle correctness passed for `runtime_scalar_cmp_masked_standalone_reduce_add` over counts `0,1,16,17,257`, seeds `-11,17`, and RHS scalars `-37,91`.

## Authority Scan

Bounded new-additions scan over touched files found no new central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived route authority.

The broad touched-file scan still finds existing literal strings in tests, script metadata constants, direct-route-entry fail-closed diagnostics, and this task record. Those are mirrors or negative probes, not newly added production authority.
