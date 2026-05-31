# Evidence Summary: Stage2 RVV widening conversion route-family ABI/provider ownership

## Production Movement

- Added `verifyRVVSelectedBodyWideningConversionRouteProviderFacts` to the
  RVV route-planning/provider surface.
- Wired widening conversion statement-plan construction and provider fact
  verification into selected-body RVV EmitC route construction before
  `TCRVEmitCLowerableRoute` construction.
- The provider-side closure now verifies:
  - exact widening conversion family-plan ownership for consumer routes;
  - stale widening facts rejected on non-conversion routes;
  - typed result config facts and source/result SEW/LMUL/type relation;
  - conversion relation and conversion intrinsic;
  - materialization facts for headers, VL type, source/result vector types,
    setvl, source load, conversion, and store;
  - math operand-binding facts for `lhs`, `out`, and runtime `n`;
  - route-control provider facts from the same selected analysis;
  - route description/runtime ABI/artifact mirror agreement;
  - no stale mask/scalar/standalone-reduction/accumulation/contraction facts.
- Common EmitC was not changed. It remains a neutral consumer of the
  provider-built route payload.
- Target artifact production code was not changed; existing target/export and
  generated-bundle checks consume rebuilt provider route facts and mirror
  metadata.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the executable
  widening conversion route-provider facts preflight contract.

## Focused Checks

- `rtk proxy cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`: passed.
- `rtk proxy artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`: passed with `RVV extension plugin smoke test passed`.
- `rtk proxy python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: passed.
- `rtk proxy python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`: passed.
- `rtk proxy git diff --check`: passed.
- `rtk proxy cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: passed, `Total Discovered Tests: 464`, `Passed: 464 (100.00%)`.

## Generated-Bundle Evidence

- Dry-run passed for selected-boundary `widen_i32_to_i64` and
  `widen_i16_to_i32` with counts `0,1,16,17,257`:
  `artifacts/tmp/05-31-stage2-rvv-widening-conversion-provider/dry-run/pre-realized-widening-conversion`.
- Direct pre-realized route-entry probe failed closed for `widen_i32_to_i64`
  with the expected retired-shortcut diagnostic:
  `artifacts/tmp/05-31-stage2-rvv-widening-conversion-provider/direct-entry/direct-pre-realized-widen-i32-to-i64`.
- Direct pre-realized route-entry probe failed closed for `widen_i16_to_i32`
  with the expected retired-shortcut diagnostic:
  `artifacts/tmp/05-31-stage2-rvv-widening-conversion-provider/direct-entry/direct-pre-realized-widen-i16-to-i32`.

## ssh rvv Evidence

- `ssh rvv` generated-bundle correctness passed for `widen_i32_to_i64` over
  counts `0,1,16,17,257`; both signed input patterns passed
  sign-extension, wide-magnitude, and tail-preservation checks.
- `ssh rvv` generated-bundle correctness passed for `widen_i16_to_i32` over
  counts `0,1,16,17,257`; both signed input patterns passed sign-extension and
  tail-preservation checks.
- Output markers:
  - `tcrv_rvv_generated_bundle_abi_widen_i32_to_i64_ok counts=0,1,16,17,257`
  - `tcrv_rvv_generated_bundle_abi_widen_i16_to_i32_ok counts=0,1,16,17,257`

## Authority Scan

- Production-only added-line scan over
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` found no new
  source-front-door, descriptor, artifact-name, route-id, common-EmitC,
  status, exact-intrinsic, legacy-i32, `rvv-i32m1`, or `RVVI32M1` authority.
- The touched-file added-line scan found only two exact intrinsic strings in
  `test/Plugin/RVVExtensionPluginTest.cpp`; both are negative/stale mutation
  probes (`__riscv_vle16_v_i16mf2` and `__riscv_vadd_vv_i64m2`), not
  production route authority.

## Non-Regression

- The full RVV extension plugin C++ smoke test passed after the provider
  closure landed, covering adjacent selected-body route-family tests including
  standalone reduction, statement-plan owners, route-control, and provider
  owner registry checks.
- `check-tianchenrv` passed 464/464.
