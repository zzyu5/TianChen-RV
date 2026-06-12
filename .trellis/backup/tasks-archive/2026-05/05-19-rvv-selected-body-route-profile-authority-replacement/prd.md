# RVV selected-body route-profile authority replacement

## Goal

Replace the RVV plugin/provider selected-body route authority bottleneck where
route construction is organized around finite per-case specialization mappings.
The provider should construct existing supported routes from typed
`tcrv_rvv` selected-body facts plus the RVV config contract: operation kind,
dtype/SEW, LMUL, policy, memory form, vector/mask dataflow types, and runtime
AVL/ABI facts.

This round is not a new RVV coverage expansion. Its goal is to introduce a
coherent RVV-owned route-profile boundary so exact intrinsic spellings and C
type names are target-owned leaves after legality has been established, not
architecture authority.

## Requirements

* Create an RVV plugin-owned route-profile mapping boundary that consumes typed
  selected-body operation/config/memory-form facts before building
  `TCRVEmitCLowerableRoute`.
* Preserve existing executable/route-supported behavior for:
  * m1 arithmetic selected-body routes.
  * m2 arithmetic selected-body routes.
  * m1 RHS broadcast selected-body routes.
  * m1 compare/select selected-body routes.
* Ensure stale route ids, artifact names, ABI labels, descriptor residue, test
  names, source-front-door metadata, or common export metadata cannot select RVV
  operation, dtype, LMUL, policy, memory form, or intrinsic mapping.
* Keep unsupported operation/config/profile combinations fail-closed with clear
  diagnostics before artifact construction.
* Keep common EmitC/materializer/target export neutral: RVV semantic decisions
  remain in the RVV plugin/provider and target-owned support bundle leaves.
* If the full provider cleanup is too large, complete a coherent submodule that
  removes table authority from arithmetic/config/memory-form route construction
  and leave an exact continuation point for compare/select profile folding.

## Acceptance Criteria

* [x] Existing supported routes are constructed from typed selected-body/config
      facts rather than a finite clone-growth specialization table.
* [x] Exact RVV intrinsic spellings and C type names are selected only as target
      leaves of a legal profile.
* [x] Unsupported selected-body/config/profile combinations fail before target
      artifact construction.
* [x] Negative tests prove stale metadata/route-id/artifact-name attempts do not
      select operation, dtype, LMUL, policy, memory form, or intrinsic mapping.
* [x] Focused tests preserve m1 arithmetic, m2 arithmetic, m1 RHS broadcast, and
      m1 compare/select route artifacts.
* [x] Focused generated-bundle dry runs cover representative preserved routes.
* [x] Real `ssh rvv` evidence is collected for route families whose executable
      provider emission behavior is materially rewritten, if executable status
      is claimed.
* [x] Bounded residue scans over touched provider/config/target/export files show
      no descriptor/direct-C/source-export authority, no source-front-door
      default authority, no old `RVVI32M1` route-table authority, no finite clone
      growth milestone, and no common EmitC/target RVV semantic branching.

## Definition of Done

* PRD and task context match the implemented bounded module.
* Focused build/test/translation checks pass or have an exact documented
  blocker.
* Trellis task status and notes are truthful.
* The task is finished/archived when complete.
* One coherent commit records PRD, implementation, tests, and evidence updates.

## Technical Approach

Introduce or refactor a route-profile surface inside the RVV plugin/provider.
The route profile should be derived from the selected `tcrv_rvv` body and
`RVVConfigContract`, then validated for target-supported lowering. Intrinsic
spelling and exported C type names remain target support leaves attached after
the profile is legal.

The preferred bounded slice is arithmetic/config/memory-form route construction
because the previous completed task exposed LMUL m2 arithmetic support and the
current finite mapping table is the immediate clone-growth pressure point.
Compare/select can remain supported through existing behavior if the arithmetic
slice is completed cleanly and the continuation point is explicit.

## Implementation Result

* Replaced the provider-local finite per-case
  `RVVSelectedBodySpecializationMapping` table with a profile derivation
  surface in `RVVEmitCRouteProvider.cpp`:
  * `RVVSelectedBodyOperationProfile` derives operation mnemonic, result name,
    mask name, and compare/select shape from the typed selected-body compute
    op.
  * `RVVSelectedBodyConfigProfile` derives SEW32, LMUL m1/m2, policy, vector
    type names, C vector types, setvl/load/store target leaves, and config/VL
    contract facts from the selected `setvl` / `with_vl` config.
  * `RVVSelectedBodyTargetLeafProfile` derives exact compute, compare, and RHS
    broadcast intrinsic spellings only after operation/config/memory-form
    legality succeeds.
* Preserved existing m1 add/sub/mul, m2 add/sub/mul, m1 RHS broadcast
  add/sub/mul, and m1 compare/select routes.
* Kept unsupported combinations fail-closed. In particular, LMUL m2 RHS
  broadcast still rejects as an unsupported route profile instead of becoming a
  new coverage expansion.
* Strengthened provider description verification so stale EmitC route IDs,
  target artifact route IDs, runtime ABI labels, config contract IDs, config/VL
  mirrors, vector/mask C types, and intrinsic spellings fail against the
  provider-derived profile.
* Kept target/export behavior neutral. Target/RVV still rebuilds the provider
  route from the selected typed body before consuming metadata; common target
  export still has no RVV semantic branch.

## Validation Result

Focused build passed:

```bash
cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2
```

Focused C++ tests passed:

```bash
./build/bin/tianchenrv-rvv-extension-plugin-test
./build/bin/tianchenrv-target-artifact-export-test
```

Focused lit passed for preserved routes and profile/config negatives:

```bash
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv \
  ../test/Target/RVV/explicit-selected-body-artifact-add.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-m2-add.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-m2-sub.mlir \
  ../test/Target/RVV/explicit-selected-body-artifact-m2-mul.mlir \
  ../test/Conversion/EmitC/rvv-first-slice-materialization-negative.mlir \
  ../test/Conversion/EmitC/rvv-first-slice-config-vl-contract-negative.mlir
```

Generated-bundle local checks passed:

```bash
python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-route-profile-default-dry --overwrite --op-kind add --op-kind sub --op-kind mul --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --rhs-broadcast-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-route-profile-broadcast-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --lmul-m2-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-route-profile-lmul-m2-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Real `ssh rvv` evidence passed:

* `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-route-profile-default-rvv`
  passed add/sub/mul/cmp_select counts `7,16,23`.
* `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-route-profile-broadcast-rvv`
  passed RHS-broadcast add/sub/mul counts `7,16,23`.
* `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-route-profile-lmul-m2-rvv`
  passed LMUL m2 add/sub/mul counts `7,16,23`.

Residue scans passed:

* Provider scan found no `RVVSelectedBodySpecializationMapping`,
  `lookupRVVSelectedBodySpecializationMapping`,
  `kRVVSelectedBodySpecializationMappings`, `computeIntrinsic`, or
  `supportsRHSBroadcastLoad`.
* Descriptor/direct-C/source-export terms in touched target files remain only
  in rejection filters and diagnostics.
* Common `TargetArtifactExport.cpp` contains no RVV intrinsic/header or
  `tcrv_rvv.i32*` semantic branching.
* Remaining route ID/runtime ABI references in target/RVV are mirror
  validations against the rebuilt provider route.
* `git diff --check` passed.
* Trellis context validation passed for this task.

## Out of Scope

* Adding broadcast m2, compare/select m2, reductions, conversions, new dtype/SEW,
  new LMUL, tail/mask policy expansion, or performance tuning.
* Restoring high-level Linalg/Vector/StableHLO frontends or source-front-door
  default authority.
* Adding one-intrinsic wrapper dialects or high-level kernel ops.
* Moving RVV semantics into common EmitC/materializer/target export.
* Treating reports, helper-only changes, dashboards, or broad smoke matrices as
  the main milestone.

## Technical Notes

Initial task source: Hermes Direction Brief appended in the worker prompt on
2026-05-19.

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-19-rvv-lmul-m2-arithmetic-selected-body-executable-route/prd.md`
* `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
* `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `lib/Target/TargetArtifactExport.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-cmp-select.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-broadcast-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-m2-add.mlir`
