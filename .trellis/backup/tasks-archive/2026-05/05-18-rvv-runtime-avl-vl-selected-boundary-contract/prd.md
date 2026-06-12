# RVV runtime AVL/VL selected-boundary contract

## Goal

Make the bounded RVV `i32m1` add/sub/mul selected boundary a production
contract shared by the RVV source front door, construction protocol,
plugin-owned EmitC route provider, generated header/bundle metadata, and target
artifact validation.

The explicit contract is:

```text
runtime ABI parameter n
  -> runtime AVL SSA operand
  -> tcrv_rvv.setvl result
  -> visible !tcrv_rvv.vl SSA value
  -> tcrv_rvv.with_vl scope
  -> load/load/i32_add|i32_sub|i32_mul/store
  -> plugin-owned materialized EmitC route
  -> object/header/bundle artifact metadata
```

This is a production compiler-path task, not another evidence-only family proof.

## What I Already Know

- The previous archived task closed source-front-door add/sub/mul evidence with
  one-command dry-run coverage and real `ssh rvv` PASS markers.
- Current specs require RVV to remain a plugin family inside unified TCRV, not
  an independent backend or descriptor-driven compute path.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and
  `lib/Dialect/RVV/IR/RVVDialect.cpp` already model `runtime_abi_value`,
  `setvl`, `with_vl`, i32 load/add/sub/mul/store, and dialect-level forbidden
  metadata checks.
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h` and
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp` already define the shared i32m1
  SEW32/LMUL m1/agnostic-policy and runtime AVL/VL artifact metadata contract.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already validates exactly
  one `setvl`, exactly one `with_vl`, visible VL consumption, explicit runtime
  `n`, ordered ABI values `lhs, rhs, out, n`, and the selected add/sub/mul op
  before building a route.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` already rejects stale route ids,
  fallback-only RVV candidates, wrong runtime ABI names, mismatched runtime ABI
  parameters, missing construction metadata, stale AVL/VL metadata, and
  descriptor/direct-C/source-export residue.
- The gap is that target artifact candidate validation should cross-check the
  selected emission-plan candidate against the actual selected variant body by
  rebuilding the operation-specific RVV EmitC route. Otherwise stale
  emission-plan metadata can claim one arithmetic route while the selected body
  has another supported arithmetic op.

## Requirements

1. The source front door must continue to materialize `lhs, rhs, out, n` with
   `n` as the runtime element-count ABI parameter feeding `tcrv_rvv.setvl`.
2. The selected RVV body must continue to expose one visible `setvl` result
   consumed by one `with_vl` scope and all load/compute/store ops.
3. RVV target artifact validation must reject a selected emission-plan
   candidate when its operation-specific route metadata, runtime ABI identity,
   or structured ABI parameters do not agree with the plugin-owned EmitC route
   rebuilt from the selected variant body.
4. RVV target artifact validation must keep rejecting fallback-only candidates,
   descriptor metadata, direct-C/source-export residue, deleted route ids, and
   stale runtime AVL/VL metadata.
5. Generated declaration-only headers and bundle indexes must surface the
   runtime AVL/VL contract, including `n`, `setvl`, `with_vl`, `emitc.for`, and
   multi-VL metadata.
6. Existing positive source-front-door add/sub/mul object/header/bundle
   behavior must keep working.

## Acceptance Criteria

- [x] Production C++ target validation rebuilds the RVV operation-specific
      EmitC route from the selected variant body and compares it with the
      selected emission-plan candidate.
- [x] A stale selected boundary where the candidate says add but the actual
      selected body contains sub fails before object/header/bundle emission.
- [x] Existing positive source-front-door add/sub/mul target artifact and
      generated-bundle dry-run coverage still passes.
- [x] Existing malformed/missing `n`, missing/disconnected `setvl`,
      non-consuming `with_vl`, wrong config/policy, fallback-only, stale
      descriptor/direct-C/source-export, and runtime ABI metadata failures stay
      fail-closed.
- [x] Targeted residue scans over touched RVV plugin/target/tests show no
      descriptor route authority, no direct C semantic exporter, no
      source-export route, and no tests protecting old paths.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` runs if practical; if not, record the reason.

## Non-Goals

- No new RVV dtype, SEW/LMUL mode, tail/mask policy family, arithmetic op, or
  generic RVV lowering.
- No broader MLIR vector lowering or scalar fallback compute rebuild.
- No descriptor tables, descriptor adapters, direct C/source-export compute
  paths, compatibility wrappers, legacy route ids, or Python compiler-core
  logic.
- No core pass branch on RVV semantics; RVV-specific checks stay in RVV plugin
  or RVV target-support code.
- No script/report/smoke-matrix-only result.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-runtime-avl-vl-selected-boundary-contract`
- `cmake --build build --target tcrv-translate tcrv-opt TargetArtifactExportTest -j2`
- Focused lit for touched RVV target/source/bundle tests and existing RVV
  malformed boundary tests.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run ... --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23`
- Targeted `rg` scans over touched RVV plugin/target/tests for descriptor,
  direct-C, source-export, and old route residue.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Definition Of Done

- The production target artifact path rejects stale selected-boundary metadata
  before artifact emission.
- Positive RVV add/sub/mul source-front-door artifact behavior remains intact.
- The Trellis task is updated truthfully, finished/archived if complete, and
  committed as one coherent change.

## Completion Notes

- Implemented a production RVV target artifact validation cross-check in
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`. When a real module candidate is
  available, the validator resolves the selected variant, rebuilds the
  operation-specific RVV EmitC route from that body, verifies the rebuilt route,
  and compares route id, route source-op provenance, and ordered ABI mappings
  against the selected emission-plan candidate.
- The exact explicit contract now enforced at target export is:
  `runtime ABI n -> setvl AVL -> visible VL SSA -> with_vl scope ->
  load/load/i32_add|i32_sub|i32_mul/store -> operation-specific RVV EmitC
  route -> object/header/bundle metadata`.
- Added a stale selected-boundary regression in
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`: the
  selected body is rewritten from add to sub after emission-plan generation,
  and export now fails before object emission.
- Expanded generated RVV header evidence checks so declaration-only headers
  surface the runtime AVL/VL contract metadata: config contract, runtime VL
  contract, `vl_def`, `vl_scope`, runtime ABI order, `emitc.for`, and multi-VL
  support.
- No descriptor, direct-C/source-export, compatibility wrapper, legacy route id,
  scalar fallback compute, Python compiler-core, new dtype, new op, or core
  RVV semantic branch was added.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-runtime-avl-vl-selected-boundary-contract`
- `cmake --build build --target tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/vector-materialized-target-artifact-exporters.mlir ../test/Target/RVV/vector-source-target-artifact-exporters.mlir ../test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-missing-abi.mlir ../test/Conversion/EmitC/rvv-first-slice-vl-contract-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-config-vl-contract-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-runtime-avl-vl-selected-boundary-contract-dry-run --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-runtime-avl-vl-selected-boundary-contract --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- Targeted residue scan over touched RVV target/tests for descriptor,
  direct-C, source-export, and deleted route residue. Hits were only negative
  FileCheck assertions and RVV target fail-closed rejection text.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

Real `ssh rvv` run artifact path:

```text
artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-runtime-avl-vl-selected-boundary-contract
```

Remote PASS markers:

```text
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23
tcrv_rvv_generated_bundle_abi_sub_ok counts=7,16,23
PASS op=sub counts=7,16,23
tcrv_rvv_generated_bundle_abi_mul_ok counts=7,16,23
PASS op=mul counts=7,16,23
```

## Spec Update Review

No `.trellis/spec/` edit was required. The existing RVV plugin spec already
states that a selected route id whose expected add/sub/mul op does not match
the typed arithmetic op in the selected body must fail before target artifact
export, and the lowering-runtime EmitC route spec already requires selected
artifact routes to validate runtime ABI ownership metadata and materialized
EmitC provenance before object/header/bundle output.
