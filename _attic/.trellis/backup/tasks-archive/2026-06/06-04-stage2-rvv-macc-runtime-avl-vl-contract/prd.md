# Stage2 RVV MAcc runtime AVL/VL contract migration

## Direction

Migrate the RVV MAcc route-family validation contract so target artifact
validation consumes the provider-owned
`RVVRuntimeAVLVLSelectedBoundaryContract` before accepting MAcc-family runtime
ABI, statement-plan, layout, scalar-broadcast, computed-mask, runtime-scalar,
payload mirror, and stale-mirror facts.

## Module Goal

Make the selected typed RVV MAcc body carry runtime `n` / AVL / VL authority
through the RVV plugin-owned runtime AVL/VL selected-boundary contract:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv MAcc body
  -> RVVRuntimeAVLVLSelectedBoundaryContract embedded in
     RVVMAccRouteValidationContract
  -> provider-built TCRVEmitCLowerableRoute setvl/loop/MAcc statements
  -> target artifact validation consumes that contract
  -> common EmitC materialization remains neutral
```

## Boundaries

- Do not add new MAcc operations, widening-dot coverage, conversion/dtype
  routes, LMUL batches, segment2, compare/select, reduction coverage,
  high-level frontend lowering, source-front-door routes, dashboards, or broad
  smoke matrices.
- Do not move RVV semantics into common EmitC/export.
- Do not infer runtime control from route ids, artifact names, test names,
  manifests, C strings, descriptor residue, or mirror metadata.
- Run `ssh rvv` only if emitted C/C++, runtime ABI order, runtime counts,
  statement ordering, accumulator/result behavior, or correctness behavior
  changes.

## Acceptance Criteria

- [x] `RVVMAccRouteValidationContract` embeds
  `RVVRuntimeAVLVLSelectedBoundaryContract`.
- [x] The MAcc provider populates the embedded runtime AVL/VL contract from
  selected typed body/config/runtime facts, including config contract id,
  runtime VL contract id, runtime AVL source, selected `with_vl` boundary and
  scope, setvl intrinsic/type, full-chunk VL, loop VL, loop induction, runtime
  `n` ABI role/order, remaining AVL, pointer advancement, bounded-slice, and
  multi-VL facts.
- [x] Target artifact MAcc validation consumes the embedded runtime AVL/VL
  contract before accepting runtime ABI facts, setvl/pre-loop/loop facts,
  accumulator/result layout, scalar-broadcast facts, computed-mask and
  runtime-scalar variants, headers, type mappings, route payload mirrors, and
  stale mirror rejection.
- [x] MAcc statement-plan validation reads setvl, VL C type, full-chunk VL,
  loop VL, loop induction, remaining AVL, pointer advancement, loop bound,
  loop step, and runtime AVL parameter from the embedded runtime contract
  instead of reconstructing them from description strings.
- [x] Missing, stale, or mismatched runtime AVL source, runtime VL contract id,
  selected `with_vl` scope, setvl intrinsic/type, full-chunk VL, loop VL, loop
  induction, runtime `n` ABI role/order, and statement-plan facts fail closed.
- [x] Focused provider/target build checks, MAcc target/export tests,
  generated-bundle dry-run/lit coverage, old-authority scan, and
  `git diff --check` pass.

## Implementation Results

- Added `configContractID` and
  `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVMAccRouteValidationContract`.
- Populated the MAcc embedded runtime AVL/VL contract from
  `populateRVVMAccCommonValidationContract(...)` after runtime ABI parameters
  are known, deriving the provider-owned MAcc setvl intrinsic from SEW/LMUL and
  using the canonical `size_t` VL C type.
- Rewired target MAcc validation so the shared runtime AVL/VL contract is
  validated immediately after config contract id and before runtime ABI,
  payload, statement-plan, and mirror checks.
- Rewired MAcc statement-plan validation to consume the embedded runtime
  contract for setvl, VL type, full-chunk/loop VL, loop induction, remaining
  AVL, pointer advancement, loop bounds/steps, and runtime `n` ABI checks.
- Extended target artifact tests with positive embedded-contract assertions for
  plain, scalar-broadcast, computed-mask, runtime-scalar computed-mask,
  runtime-scalar computed-mask LMUL m2, and widening MAcc route variants.
- Added fail-closed MAcc tests for stale runtime AVL source, missing runtime VL
  contract, stale selected `with_vl` scope, stale setvl callee, stale VL C
  type, stale full-chunk VL, stale loop VL, stale loop induction, stale
  runtime `n` ABI role, stale remaining AVL metadata, and stale pointer
  advancement metadata.
- Updated the EmitC route spec to make the MAcc runtime AVL/VL embedded
  contract a durable target-consumer rule.

## Evidence Results

- `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
  passed after self-repairing an initial local visibility error by using the
  MAcc provider-local setvl derivation helper instead of an unavailable
  selected-body helper.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'macc-add|scalar-broadcast-macc-add|computed-masked-macc-add|runtime-scalar-cmp-masked-macc-add|widening-macc-add'`
  passed from `build/test` with 28 focused tests.
- `git diff --check` passed.
- Added-line old-authority scan over touched files found no added
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor,
  source-front-door, or exact `__riscv_*_i32m1` authority residue.
- `ssh rvv` was not rerun because this round changed provider/target
  validation contracts, target C++ tests, Trellis records, and spec text only;
  emitted C/C++, runtime ABI order, runtime counts, statement ordering,
  accumulator/result behavior, mask/tail behavior, harness, and
  correctness/performance semantics were unchanged.
