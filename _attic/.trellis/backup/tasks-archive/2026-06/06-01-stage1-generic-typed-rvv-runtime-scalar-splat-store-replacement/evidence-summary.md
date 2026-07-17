# Evidence Summary: Stage1 generic typed RVV runtime-scalar splat-store replacement

## Production Movement

- Replaced the runtime scalar splat-store operation mnemonic and operation-kind
  authority from `RuntimeI32SplatStore` / `runtime_i32_splat_store` to the
  generic typed `RuntimeScalarSplatStore` / `runtime_scalar_splat_store`
  route surface.
- Kept the route bounded to the current typed SEW32/LMUL m1 instance, but made
  the RVV provider derive accepted type, scalar C type, policy, setvl, splat,
  store, target profile, and route-family facts from typed body/config/runtime
  facts rather than from an i32-shaped route name.
- Updated the pre-realized runtime scalar splat-store verifier and realization
  owner to accept `op_kind = "runtime_scalar_splat_store"` and to reject stale
  operation mnemonics before selected-body realization.
- Updated provider planning, residual statement planning, route-control and
  provider preflight checks, construction protocol metadata, target artifact
  validation, generated-bundle expectations, lit fixtures, and C++ tests to use
  the generic typed route surface.
- Common EmitC/export semantic behavior was not moved into common code; common
  paths still consume provider-built route payloads and mirrors only.

## Focused Checks

- `git diff --check`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`: passed with `RVV extension plugin smoke test passed`.
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test && artifacts/tmp/tianchenrv-build/bin/tianchenrv-construction-protocol-common-test`: passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`: passed, `Total Discovered Tests: 464`, `Passed: 464 (100.00%)`.

## Generated-Bundle Evidence

- Dry-run passed for pre-realized selected-boundary
  `runtime_scalar_splat_store` with counts `0,1,16,23,257` and scalar values
  `-37,19`:
  `artifacts/tmp/06-01-stage1-runtime-scalar-splat-store/dry-run/pre-realized-runtime-scalar-splat-store`.

## Authority Scan

- `rg -n "RuntimeI32SplatStore|runtime_i32_splat_store|runtime-i32-splat-store|runtime_i32_splat|runtime-i32-splat" include lib test scripts .trellis/spec` returned no matches.
- Added-line scan over include/lib/test/scripts/spec for
  `RuntimeI32SplatStore`, `runtime_i32_splat_store`, `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, and exact
  `__riscv_.*_i32m1` found no active production authority additions; the only
  hit is an `implicit-check-not="tcrv_rvv.i32_"` assertion in the generated
  bundle dry-run lit test.

## Runtime Evidence

- No new RVV runtime correctness or performance claim was made in this Stage 1
  replacement round, so no `ssh rvv` runtime evidence was collected.

## Continuation

- This bounded owner is complete. Later Stage 1/Stage 2 owners may broaden
  typed runtime scalar splat-store configs only as ordinary typed config
  instances, not by reintroducing dtype-prefixed operation names, route ids, or
  artifact-name authority.
