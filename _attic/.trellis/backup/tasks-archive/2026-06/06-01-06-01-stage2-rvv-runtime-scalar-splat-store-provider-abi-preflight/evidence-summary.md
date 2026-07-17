# Evidence Summary: Stage2 RVV runtime-scalar splat-store provider ABI preflight

## Production Movement

- Added `verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts` to the
  RVV route-planning/provider surface.
- Wired runtime scalar splat-store statement-plan construction and provider
  fact verification into selected-body RVV EmitC route construction before
  `TCRVEmitCLowerableRoute` construction.
- The provider-side closure now verifies:
  - exact runtime scalar splat-store family-plan ownership for consumer routes;
  - stale runtime scalar splat-store facts rejected on non-consumer routes;
  - typed config facts for SEW/LMUL, policy, VL type, result vector type/C
    type, setvl leaf, store leaf, and result name;
  - materialization facts for required headers, VL type, result vector type/C
    type, setvl, runtime scalar splat, and store;
  - residual operand-binding facts for `rhs_scalar`, `out`, and runtime `n`;
  - route-control provider facts from the same selected analysis;
  - route description, runtime ABI, and artifact mirror agreement;
  - no stale source/mask/conversion/reduction/accumulation/contraction/segment
    facts as route authority;
  - provider-ready statement leaves for setvl, scalar splat, and store.
- Common EmitC was not changed. It remains a neutral consumer of the
  provider-built route payload.
- Target artifact production code was not changed.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the executable
  runtime scalar splat-store route-provider facts preflight contract.

## Focused Checks

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`: passed with `RVV extension plugin smoke test passed`.
- `git diff --check`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: passed, `Total Discovered Tests: 464`, `Passed: 464 (100.00%)`.

## Generated-Bundle Evidence

- Dry-run passed for selected-boundary `runtime_i32_splat_store` with counts
  `0,1,16,17,257` and scalar values `11,-5`:
  `artifacts/tmp/06-01-stage2-rvv-runtime-scalar-splat-store-provider/dry-run/selected-runtime-i32-splat-store`.

## ssh rvv Evidence

- `ssh rvv` generated-bundle correctness passed for
  `runtime_i32_splat_store` over counts `0,1,16,17,257` and scalar values
  `11,-5`; every case reported `tail_preserved`.
- Output markers:
  - `tcrv_rvv_generated_bundle_abi_runtime_i32_splat_store_ok counts=0,1,16,17,257 rhs_scalars=11,-5`
  - `PASS op=runtime_i32_splat_store counts=0,1,16,17,257 rhs_scalars=11,-5`

## Authority Scan

- Production-only added-line scan over
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` found no new
  `RVVI32M1`, `rvv-i32m1`, source-front-door, source-artifact, descriptor,
  direct-C/source-export, exact `__riscv`, or legacy-i32 authority.
- Broader production scan hits on `description`, ABI, and artifact wording are
  mirror comparisons and stale-fact rejection diagnostics, not support
  authority.
- Test/spec added lines mention route-id/artifact-name/exact-intrinsic/common
  EmitC/source-front-door/descriptor/legacy-i32 only in negative review
  language or stale-authority rejection expectations.

## Non-Regression

- The focused RVV extension plugin C++ smoke test passed after the provider
  closure landed, including the latest widening conversion provider preflight
  tests and adjacent route-family provider checks.
- `check-tianchenrv` passed 464/464.
