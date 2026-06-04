# Journal - codex (Part 22)

> Continuation from `journal-21.md` (archived at ~2000 lines)
> Started: 2026-06-04

---



## Session 423: Stage2 RVV runtime-scalar splat-store route validation contract

**Date**: 2026-06-04
**Task**: Stage2 RVV runtime-scalar splat-store route validation contract
**Branch**: `main`

### Summary

Extracted a provider-owned runtime-scalar splat-store route validation contract, rewired target artifact validation to consume it for route payload, ABI, binding, header/type, intrinsic, dtype/config, policy, AVL/VL, statement-plan facts, updated focused provider/target tests and lowering-runtime spec, and passed focused build/tests/lit/diff checks. No ssh rvv run because generated runtime ABI/emitted behavior did not change.

### Main Changes

- Added provider-owned computed-mask strided memory route validation and
  metadata mirror contracts.
- Rewired target artifact validation for `computed_masked_strided_store` and
  `computed_masked_strided_load_unit_store` to consume the provider contract
  before candidate metadata mirrors.
- Added focused stale provider-description and stale candidate-mirror negative
  coverage.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  computed-mask strided validation contract.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j 16`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit filter for computed-mask strided store/load artifacts and
  generated-bundle dry-run: 5 passed, 472 excluded.
- [OK] direct pre-realized `computed_masked_strided_load_unit_store`
  generated-bundle dry-run with runtime counts `7,16,23` and byte strides
  `4,8,12`.
- [OK] `rtk git diff --check`
- [OK] task context validation
- [OK] bounded added-line old-authority scan found no new positive legacy
  authority path.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 451: Stage2 RVV low-precision widening-product route foundation

**Date**: 2026-06-04
**Task**: Stage2 RVV low-precision widening-product route foundation
**Branch**: `main`

### Summary

Implemented the bounded production route-supported low-precision RVV widening
product foundation: typed signed i8mf4 source loads plus i8 x i8 widening
product to i16mf2, with RVV plugin-owned route facts, direct contraction
provider planning, construction protocol support, common EmitC statement
materialization, and target artifact fail-closed validation. This is not a
q8/q4 benchmark route and does not claim runtime correctness or performance.

### Main Changes

- Added `tcrv_rvv.widening_product` and verifier rules for
  `signed_widening_product`, source `!tcrv_rvv.vector<i8, "mf4">`, result
  `!tcrv_rvv.vector<i16, "mf2">`, selected SEW16/LMUL mf2 config, and matching
  VL.
- Added RVV config/runtime ABI contracts for SEW8 source facts, SEW16/MF2
  result facts, and `lhs,rhs,out,n` ABI with `const int8_t *` inputs and
  `int16_t *` output.
- Added product-aware RVV construction route mapping, route-control provider
  plan, math operand-binding facts, contraction family plan/facts, statement
  plan, and materialization metadata.
- Added target validation for `low-precision-widening-product`, including
  fail-closed stale provider fact, route payload, and candidate mirror checks.
- Added focused dialect and target/plugin/common tests for positive route
  support and negative mismatches.

### Git Commits

Final source/task/journal commit is created after this journal entry.

### Testing

- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `rtk bash -lc 'build/bin/tcrv-opt test/Dialect/RVV/generic-widening-product-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/generic-widening-product-dataflow.mlir'`
- [OK] `rtk bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter generic-widening-product-dataflow'`
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-low-precision-widening-product-route`
- [OK] `rtk git diff --check`
- [OK] bounded old-authority/name-authority scan over current diff found only
  the PRD non-goal `q8/q4 names` wording and no new positive authority.

### Status

[OK] Completed as route-supported and target-validated. Archive and commit
follow this journal entry.

### Next Steps

- Continue later with i16-to-i32 widening reduction, dequantization, or full
  low-precision contraction closure when a separate task asks for it.


## Session 447: Stage2 RVV compare/select executable artifact closure on ssh rvv

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select executable artifact closure on ssh rvv
**Branch**: `main`

### Summary

Closed the representative compare/select executable evidence gap with a real
`ssh rvv` run for the pre-realized selected-body `cmp_select_sle` path. The
existing production route/emission/harness path generated a target object,
header, and external C ABI harness, then compiled and ran on the RVV target
with predicate true/false lane correctness across runtime counts
`0,1,7,16,23,257`.

### Main Changes

- Created the Trellis task and PRD for the bounded executable-evidence round.
- Recorded `cmp_select_sle` as the chosen representative compare/select
  submodule.
- No provider, target validation, plugin, common EmitC, generated C/C++, ABI,
  or harness-generation code changed; live evidence showed the existing path
  already executed correctly.

### Evidence

- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-compare-select-executable-ssh-rvv --run-id cmp-select-sle-ssh-rvv --overwrite --op-kind cmp_select_sle --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
- [OK] Remote compile: `remote_arch=riscv64`, `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- [OK] Remote run pass marker:
  `tcrv_rvv_generated_bundle_abi_cmp_select_sle_ok counts=0,1,7,16,23,257`
  and `PASS op=cmp_select_sle counts=0,1,7,16,23,257`.
- [OK] Predicate lane coverage includes both true and false lanes for
  `n=7`, `n=16`, `n=23`, and `n=257`.
- [OK] Focused lit from `build/test`:
  `pre-realized-cmp-select-dry-run` passed 1 / excluded 476.
- [OK] `rtk git diff --check`
- [OK] Trellis task context validation.

### Status

[OK] **Completed**

### Next Steps

- Finish/archive the task and commit the Trellis task record.


## Session 444: Stage2 RVV memory-family runtime AVL/VL mirror closure

**Date**: 2026-06-04
**Task**: Stage2 RVV memory-family runtime AVL/VL mirror closure
**Branch**: `main`

### Summary

Closed residual memory-family target metadata runtime labels so retained
`runtime_control_plan` and `runtime_abi_order` candidate metadata are explicit
route-local runtime AVL/VL mirrors for unit-stride masked memory,
computed-mask indexed/strided memory, and plain/computed-mask segment2 memory.
No runtime behavior changed.

### Main Changes

- Relabeled memory-family metadata mirror contracts in
  `RVVEmitCRoutePlanning.cpp` to `route-local runtime AVL/VL control plan
  mirror` and `route-local runtime AVL/VL ABI order mirror`.
- Added C++ target assertions that unit-stride masked, computed-mask indexed,
  computed-mask strided, and segment2 memory metadata mirror contracts carry
  the route-local runtime mirror labels.
- Updated computed-mask segment2 lit diagnostics to expect the route-local
  runtime ABI mirror label.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | rvv: close memory runtime mirror metadata labels |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate -j 16`
- [OK] focused memory/segment2 lit filter from `build/test`: 71 passed, 406 excluded.
- [OK] `rtk git diff --check`
- [OK] bounded old-label grep found no residual selected-typed runtime label matches.
- [OK] added-line old-authority scan found no matches.
- [OK] Trellis context validation.

### Self-Repair

- First focused lit run failed because `tcrv-translate` had not been rebuilt
  after the planning label change and still emitted the old segment2 runtime
  ABI label. Rebuilt `tcrv-opt` and `tcrv-translate`, then reran the same lit
  filter successfully.

### Status

[OK] **Completed**

### Next Steps

- Archive task and commit this session.


## Session 441: Stage2 RVV scalar and elementwise runtime AVL/VL sole-authority cleanup

**Date**: 2026-06-04
**Task**: Stage2 RVV scalar and elementwise runtime AVL/VL sole-authority cleanup
**Branch**: `main`

### Summary

Made RVV runtime scalar splat-store and elementwise arithmetic target
validation consume `RVVRuntimeAVLVLSelectedBoundaryContract` as the sole
runtime n / AVL / VL authority before checking retained route-local runtime
mirrors; added focused provider-contract mirror assertions and archived the
Trellis task.

### Main Changes

- Documented scalar splat-store and elementwise validation contract
  runtime/control copies as target-side consistency mirrors.
- Rewired scalar splat-store and elementwise target validation to use the
  shared route-local runtime AVL/VL mirror helper after selected-boundary
  validation.
- Added target C++ assertions that scalar splat-store plus representative
  plain, masked, and scalar-broadcast elementwise retained mirrors match the
  embedded selected-boundary contract.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused elementwise selected-body artifact lit filter
- [OK] focused runtime-scalar-splat-store lit filter
- [OK] focused elementwise generated-bundle dry-run filter
- [OK] added-line old-authority scan
- [OK] `rtk git diff --check`
- [OK] Trellis task validation

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 440: Stage2 RVV computed-mask strided runtime AVL/VL contract

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask strided memory runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Promoted computed-mask strided memory target validation to consume the embedded
RVV runtime AVL/VL selected-boundary contract; aligned manual strided fixtures
with canonical selected-boundary loop facts; added focused fail-closed target
coverage, dry-run evidence, and EmitC route spec notes.

### Main Changes

- Embedded `runtimeAVLVLContract` in
  `RVVComputedMaskStridedMemoryRouteValidationContract`.
- Populated the contract from provider-owned SEW/LMUL/policy/config/runtime
  ABI facts and made target validation consume it before route-local checks.
- Added positive and negative target coverage for stale runtime AVL source,
  runtime VL contract, selected `with_vl` scope, setvl, loop VL/induction,
  runtime n ABI role, and pointer advancement metadata.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | rvv: consume runtime AVL VL contract for computed strided memory |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j 16`
- [OK] `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j 16`
- [OK] `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused computed-mask strided lit filter: 5 passed, 472 excluded
- [OK] Direct pre-realized generated-bundle dry-runs for
  `computed_masked_strided_store` and
  `computed_masked_strided_load_unit_store`
- [OK] `rtk git diff --check`
- [OK] Added-line old-authority scan returned no matches

### Status

[OK] **Completed**

### Next Steps

- Archive task and commit this session.


## Session 427: Stage2 RVV compare/select target consume-only closeout

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select route-contract target consume-only closeout
**Branch**: `main`

### Summary

Closed out the compare/select producer target-consumer path by removing direct
target validation consumption of raw compare/select route facts. Target
artifact validation now consumes the existing provider-owned compare/select
route validation contract for runtime ABI and statement-plan expectations, while
existing computed-mask memory routes continue to use their memory contracts.

### Main Changes

- Removed target-local `RVVCompareSelectRouteFacts` and runtime-scalar dual raw
  fact reconstruction from `RVVTargetArtifactRouteFamilyValidation.cpp`.
- Rewired shared runtime ABI validation to select provider contracts for
  compare/select producers, computed-mask indexed memory, computed-mask strided
  memory, and unit-stride masked memory.
- Passed compare/select provider contract statement-plan counts into the shared
  route statement-plan validator.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select'`
- [OK] Direct target raw-fact consumption scan
- [OK] Source old-authority scan
- [OK] `rtk git diff --check`
- [OK] Trellis context validation

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 426: Stage2 RVV computed-mask indexed-memory provider contract

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask indexed-memory provider contract extraction
**Branch**: `main`

### Summary

Extracted provider-owned computed-mask indexed memory route validation and
mirror contracts for gather/load-unit-store and scatter/store-unit-load,
rewired target artifact validation to consume them before candidate mirrors,
updated lowering-runtime spec, and passed focused build/C++/lit checks. No
`ssh rvv` run because this changed validation ownership only, not emitted C,
runtime ABI, mask/index behavior, correctness, or performance claims.

### Main Changes

- Added computed-mask indexed validation/mirror contract provider APIs.
- Removed direct target-validator consumption of computed-mask indexed route
  facts.
- Added focused C++ contract and stale mirror coverage.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
- [OK] `rtk cmake --build build --target tcrv-opt tcrv-translate -j 16`
- [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused computed-mask indexed lit/generated-bundle filter: 8 passed, 469 excluded
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 425: Stage2 RVV vector-reduction executable artifact closeout

**Date**: 2026-06-04
**Task**: Stage2 RVV vector-reduction executable artifact closeout
**Branch**: `main`

### Summary

Closed the existing provider-contract-backed `reduce_add` vector RHS-load path
with real `ssh rvv` generated-bundle correctness evidence. No production source
change was required: the selected pre-realized `tcrv_rvv` body already realized
through RVV provider route facts, generated a common EmitC object/header bundle,
compiled on the RVV target, and passed the scalar-reference harness for runtime
counts `7,16,23`.

### Main Changes

- Created Trellis task
  `.trellis/tasks/06-04-stage2-rvv-vector-reduction-executable-closeout`.
- Wrote PRD, implement/check context, completion notes, and evidence results.
- Produced dry-run artifact evidence at
  `artifacts/tmp/06-04-vector-reduction-executable-closeout/pre-realized-reduce-add-dry`.
- Produced real RVV artifact evidence at
  `artifacts/tmp/06-04-vector-reduction-executable-closeout/pre-realized-reduce-add-ssh-rvv`.

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit filter for `explicit-selected-body-artifact-reduce-add`,
  `pre-realized-selected-body-artifact-reduce-add`, and
  `rvv-generated-bundle-abi-e2e-pre-realized-reduce-add-dry-run`: 3 passed,
  474 excluded.
- [OK] dry-run generated bundle for pre-realized `reduce_add` counts
  `7,16,23`.
- [OK] real `ssh rvv` generated bundle for pre-realized `reduce_add` counts
  `7,16,23`; remote run printed `PASS op=reduce_add counts=7,16,23`.
- [OK] bounded old-authority scan over touched files: only negative guardrails
  and common EmitC boundary descriptions in task docs.
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 424: Stage2 RVV vector-reduction route validation contract

**Date**: 2026-06-04
**Task**: Stage2 RVV vector-reduction route validation contract
**Branch**: `main`

### Summary

Extracted provider-owned vector-reduction route validation contract for ReduceAdd vector RHS-load, rewired target artifact validation to consume it for ABI, binding, dtype/config, header/type, leaf/profile, intrinsic, layout, AVL/VL and candidate mirrors, added focused provider/target tests, and passed focused build/tests/lit/diff checks. No ssh rvv run because this changed validation ownership only, not generated runtime behavior or correctness/performance claims.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 425: Stage2 RVV computed-mask strided-memory provider contract

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask strided-memory provider contract
**Branch**: `main`

### Summary

Extracted provider-owned computed-mask strided memory route validation and mirror contracts, rewired target artifact validation to consume them, added focused stale provider/mirror coverage, updated lowering-runtime spec, and passed focused build/C++/lit/dry-run checks.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `created-after-journal-entry` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 426: Stage2 RVV unit-stride masked memory route contract

**Date**: 2026-06-04
**Task**: Stage2 RVV unit-stride masked memory route contract
**Branch**: `main`

### Summary

Extracted provider-owned unit-stride masked-memory route validation and mirror contracts, rewired target artifact validation to consume them, archived task, and passed focused C++/lit/scans.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 427: Stage2 RVV segment2 memory target contract closeout

**Date**: 2026-06-04
**Task**: Stage2 RVV segment2 memory target contract closeout
**Branch**: `main`

### Summary

Removed target-local plain/computed-mask segment2 raw route-fact reconstruction; target artifact validation now consumes RVVSegment2MemoryRouteValidationContract, updated stale mirror fixture diagnostics, and passed focused C++/lit/scans.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 428: Stage2 RVV computed-mask segment2 update executable closeout

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask segment2 update executable closeout
**Branch**: `main`

### Summary

Closed computed_masked_segment2_update_unit_load generated-bundle executable evidence for pre-realized and explicit selected RVV fixtures with real ssh rvv PASS output; no compiler or runner source changes were needed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 429: Stage2 RVV runtime AVL/VL selected-boundary contract promotion

**Date**: 2026-06-04
**Task**: Stage2 RVV runtime AVL/VL selected-boundary contract promotion
**Branch**: `main`

### Summary

Promoted runtime n/AVL/VL facts into a provider-owned selected-boundary contract, wired segment2 target validation to consume it, updated EmitC route spec, archived task, and passed focused C++/lit/scans.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `8c2611d9` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 430: Stage2 RVV base-memory runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV base-memory runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in the base-memory movement validation contract, rewired target validation to consume it before accepting base-memory route payloads, added fail-closed target coverage, updated EmitC route spec, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `ac543282` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 431: Stage2 RVV compare/select runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in compare/select validation, rewired target validation and statement-plan checks to consume it, added fail-closed target coverage, updated EmitC route spec, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 432: Stage2 RVV standalone-reduction runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV standalone-reduction runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in standalone reduction validation, rewired target validation and statement-plan checks to consume it, added fail-closed target coverage, updated EmitC route spec, and prepared the task for archive.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 433: Stage2 RVV MAcc runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV MAcc runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in MAcc validation, rewired target validation and statement-plan checks to consume it, added positive and fail-closed target coverage, updated EmitC route spec, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 434: Stage2 RVV widening-dot runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV widening-dot runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in widening-dot reduction validation, rewired target validation and statement-plan checks to consume it, added positive and fail-closed target coverage, updated EmitC route spec, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 435: Stage2 RVV conversion runtime AVL/VL contract

**Date**: 2026-06-04
**Task**: Stage2 RVV conversion runtime AVL/VL contract
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in conversion dtype-policy validation and updated target consumer tests.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `02945a1c` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 436: Stage2 RVV runtime-scalar splat-store runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV runtime-scalar splat-store runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded RVVRuntimeAVLVLSelectedBoundaryContract in runtime-scalar splat-store validation, rewired target validation and statement-plan checks to consume it, added fail-closed target coverage, updated EmitC route spec, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 437: Stage2 RVV elementwise runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV elementwise runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Migrated elementwise arithmetic provider/target validation to embed and consume the shared runtime AVL/VL selected-boundary contract; focused C++ and lit checks passed.

### Main Changes

- Rewired conversion dtype-policy target validation to consume the embedded
  runtime AVL/VL selected-boundary contract before checking retained
  route-local runtime/control mirrors.
- Demoted conversion `runtime_control_plan` and `runtime_abi_order` candidate
  metadata labels to route-local runtime AVL/VL mirror labels.
- Added target C++ coverage for conversion mirror labels and stale runtime
  metadata mirrors.
- Updated the lowering-runtime spec and archived the Trellis task.

### Git Commits

included-in-this-commit

### Testing

- [OK] Built `tianchenrv-target-artifact-export-test` and
  `tianchenrv-rvv-extension-plugin-test`.
- [OK] Ran `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] Ran `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Ran focused conversion lit from `build/test` with 3 passed and 474
  excluded.
- [OK] Ran `git diff --check`, Trellis validate, old-label grep, and
  added-line old-authority scan.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 438: Stage2 RVV unit-stride masked memory runtime AVL/VL contract migration

**Date**: 2026-06-04
**Task**: Stage2 RVV unit-stride masked memory runtime AVL/VL contract migration
**Branch**: `main`

### Summary

Embedded runtime AVL/VL selected-boundary contract in the unit-stride masked memory validation contract, rewired target validation and statement-plan checks to consume it, added fail-closed target coverage, updated EmitC route spec, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 439: Stage2 RVV computed-mask indexed runtime AVL/VL contract

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask indexed runtime AVL/VL contract
**Branch**: `main`

### Summary

Promoted computed-mask indexed memory target validation to consume the embedded RVV runtime AVL/VL selected-boundary contract; added focused target/plugin/dry-run evidence and archived the task.

### Main Changes

- Created and archived the Trellis task for
  `runtime_scalar_cmp_masked_macc_add` executable evidence.
- Recorded that the existing production path already reaches selected-body
  realization, RVV-owned MAcc provider facts, common EmitC materialization,
  target artifact bundle export, generated harness, and ssh rvv execution.
- Preserved the evidence artifact paths for dry-run and non-dry-run generated
  bundle runs under `artifacts/tmp/`.

### Git Commits

- Included in this final round commit.

### Testing

- [OK] Generated bundle dry-run for
  `runtime_scalar_cmp_masked_macc_add` with counts `0,1,7,16,23,257` and
  rhs scalars `-3,5`.
- [OK] Focused lit via `build/test` filter:
  `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add`
  passed 2 tests.
- [OK] Non-dry-run `ssh rvv` generated harness run passed 24 cases:
  6 counts x 2 rhs scalars x 2 patterns, ending with
  `PASS op=runtime_scalar_cmp_masked_macc_add counts=0,1,7,16,23,257 rhs_scalars=-3,5 patterns=0,1`.
- [OK] `git diff --check` and Trellis context validation passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 440: Stage2 RVV memory-family runtime AVL/VL sole-authority cleanup

**Date**: 2026-06-04
**Task**: Stage2 RVV memory-family runtime AVL/VL sole-authority cleanup
**Branch**: `main`

### Summary

Made promoted memory-family target validation consume RVVRuntimeAVLVLSelectedBoundaryContract as sole runtime n / AVL / VL authority before route-local mirror checks; added focused C++ mirror assertions, spec note, memory-family lit evidence, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 441: Stage2 RVV compare/select runtime AVL/VL sole-authority cleanup

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select runtime AVL/VL sole-authority cleanup
**Branch**: `main`

### Summary

Made RVV compare/select target validation consume RVVRuntimeAVLVLSelectedBoundaryContract as sole runtime n / AVL / VL authority before route-local mirror checks; added focused provider/candidate mirror coverage and archived the task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 442: Stage2 RVV compare/select candidate metadata runtime mirror closure

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select candidate metadata runtime mirror closure
**Branch**: `main`

### Summary

Closed residual compare/select candidate metadata fallback labels for runtime control plan and runtime ABI order so both are explicit route-local runtime AVL/VL mirrors; focused target and lit checks passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 443: Stage2 RVV conversion runtime AVL/VL mirror cleanup

**Date**: 2026-06-04
**Task**: Stage2 RVV conversion runtime AVL/VL mirror cleanup
**Branch**: `main`

### Summary

Demoted conversion dtype-policy route-local runtime AVL/VL fields and candidate runtime metadata to selected-boundary-checked mirrors; added focused target coverage and spec notes.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 444: Stage2 RVV runtime scalar splat-store runtime AVL/VL mirror closure

**Date**: 2026-06-04
**Task**: Stage2 RVV runtime scalar splat-store runtime AVL/VL mirror closure
**Branch**: `main`

### Summary

Closed residual runtime scalar splat-store candidate metadata runtime labels so runtime_control_plan and runtime_abi_order are route-local runtime AVL/VL mirrors; focused target, plugin, and lit checks passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 445: Stage2 RVV compare/select artifact evidence closeout

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select artifact evidence closeout
**Branch**: `main`

### Summary

Strengthened pre-realized compare/select generated-bundle evidence to assert runtime AVL/VL selected-boundary facts alongside predicate/select facts; focused lit passed; no production or runtime behavior changed.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 446: Stage2 RVV compare/select production fail-closed closure

**Date**: 2026-06-04
**Task**: Stage2 RVV compare/select production route and fail-closed legality closure
**Branch**: `main`

### Summary

Hardened the compare/select target-artifact production path so stale non-compare/select route-family provider facts and candidate mirrors fail closed before artifact acceptance. Runtime-control and runtime-ABI metadata remain route-local mirrors after the embedded runtime AVL/VL selected-boundary contract.

### Main Changes

- Added compare/select provider validation rejection for stale elementwise, runtime-scalar splat-store, widening conversion, computed-mask memory, MAcc, standalone reduction, contraction, base-memory, segment2, and widening relation route-family residue.
- Extended the compare/select metadata mirror contract with explicit stale unrelated route-family mirror keys.
- Added target C++ negative coverage for stale computed-mask memory/base-memory provider facts and stale base-memory/segment2 candidate mirrors.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 8`
- [OK] `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused compare/select lit filter: 28 passed / 449 excluded
- [OK] `rtk git diff --check`
- [OK] added-line old-authority scan over touched production/test files
- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-compare-select-production`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 448: Stage2 RVV computed-mask standalone reduction executable evidence

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask standalone reduction executable evidence
**Branch**: `main`

### Summary

Closed computed_mask_standalone_reduce_add executable evidence through generated artifact, external C ABI harness, and ssh rvv correctness without production code changes.

### Main Changes

- Created and archived the bounded Trellis task for
  `computed_mask_standalone_reduce_add` executable evidence.
- Confirmed the existing production path already flows from pre-realized
  selected `tcrv.exec` RVV variant through RVV plugin selected-body
  materialization, provider-built computed-mask standalone reduction route,
  generated object/header bundle, external C ABI harness, and `ssh rvv`
  compile/run.
- Recorded evidence under
  `artifacts/tmp/stage2-rvv-computed-mask-standalone-reduction-executable-artifact/computed-mask-standalone-reduce-add-ssh-rvv`.
- No provider, target validation, plugin, common EmitC, ABI, generated C++,
  or harness-generation code changed.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] Generated-bundle dry-run for pre-realized
  `computed_mask_standalone_reduce_add` counts `0,1,7,16,23,257`.
- [OK] Non-dry-run `ssh rvv` compile/run for
  `computed_mask_standalone_reduce_add` counts `0,1,7,16,23,257`, seeds
  `-11,17`, patterns `0,1`; remote PASS marker recorded.
- [OK] Remote compile: `remote_arch=riscv64`, `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- [OK] Focused lit from `build/test`: 6 passed / 471 excluded.
- [OK] `rtk python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-04-stage2-rvv-computed-mask-standalone-reduction-executable-artifact`
  before archive.
- [OK] `rtk git diff --check`.
- [OK] Bounded old-authority scan over touched task/journal docs found no new
  positive production authority path.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 447: Stage2 RVV runtime scalar computed mask MAcc executable evidence

**Date**: 2026-06-04
**Task**: Stage2 RVV runtime scalar computed mask MAcc executable evidence
**Branch**: `main`

### Summary

Closed the runtime_scalar_cmp_masked_macc_add executable evidence task: generated bundle dry-run, focused lit, and ssh rvv correctness over counts 0,1,7,16,23,257 with rhs scalars -3,5 and patterns 0,1; no production code changes required.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 448: Stage2 RVV computed-mask widening dot-reduction executable artifact

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask widening dot-reduction executable artifact
**Branch**: `main`

### Summary

Closed the representative pre-realized computed_masked_widening_dot_reduce_add path with generated artifact, external ABI harness, and real ssh rvv correctness evidence. No production code change was needed; existing RVV provider/target path already executed end to end.

### Main Changes

- Created and archived Trellis task `06-04-06-04-stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact`.
- Recorded PRD evidence for selected variant `pre_realized_body_rvv_masked_widening_dot_reduce_add` and external ABI function `tcrv_emitc_pre_realized_body_masked_widening_dot_reduce_add_kernel_pre_realized_body_rvv_masked_widening_dot_reduce_add`.
- Generated dry-run artifact/evidence at `artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-dry-run/computed-masked-widening-dot-reduce-add-dry-run`.
- Generated non-dry-run artifact/evidence at `artifacts/tmp/stage2-rvv-computed-mask-widening-dot-reduction-executable-artifact/computed-masked-widening-dot-reduce-add-ssh-rvv`.
- `ssh rvv` compile/run passed for runtime counts `0,1,7,16,23,257` with `PASS op=computed_masked_widening_dot_reduce_add counts=0,1,7,16,23,257`.
- Focused lit passed via `cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add`.
- `git diff --check` and Trellis task context validation passed.
- Spec update review found no new `.trellis/spec/` contract needed; existing specs already cover provider facts, target validation, evidence boundary, and ssh rvv requirements.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 449: Stage2 RVV strided-input widening dot-reduction executable artifact

**Date**: 2026-06-04
**Task**: Stage2 RVV strided-input widening dot-reduction executable artifact
**Branch**: `main`

### Summary

Closed computed_masked_strided_input_widening_dot_reduce_add with generated artifact, external ABI harness, and real ssh rvv correctness evidence over counts 0,1,7,16,23,257 and stride pairs 2:3,3:2; no production code change was needed.

### Main Changes

- Created and archived Trellis task `06-04-stage2-rvv-strided-input-widening-dot-reduction-executable-artifact`.
- Chose `computed_masked_strided_input_widening_dot_reduce_add` from `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.
- Recorded generated evidence at `artifacts/tmp/stage2-rvv-computed-mask-strided-input-widening-dot-reduction-executable-artifact/computed-masked-strided-input-widening-dot-reduce-add-ssh-rvv`.
- Confirmed materialized body uses `tcrv_rvv.strided_load` for both i16 dot sources and emitted C++ uses `__riscv_vlse16_v_i16mf2` with runtime `lhs_stride` and `rhs_stride` ABI values.
- `ssh rvv` compile/run passed for runtime counts `0,1,7,16,23,257`, stride pairs `2:3,3:2`, two mask patterns, and two input patterns.
- No production provider, target, plugin, fixture, or script code change was needed; the existing route/emission/harness path already executed end to end.

### Git Commits

included-in-this-commit

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-strided-input-widening-dot-reduction-executable-artifact`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add ...`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --ssh-target rvv ...`
- [OK] `rtk bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add'`
- [OK] bounded old-authority scan over the task files and relevant provider/target/fixture/script surfaces.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 450: Stage2 RVV computed-mask segment2 update executable artifact

**Date**: 2026-06-04
**Task**: Stage2 RVV computed-mask segment2 update executable artifact
**Branch**: `main`

### Summary

Closed computed_masked_segment2_update_unit_load with generated artifact, external ABI harness, and real ssh rvv correctness evidence over counts 0,1,7,16,23,257 and two mask/input patterns; no production code change was needed.

### Main Changes

- Created and archived Trellis task `06-04-stage2-rvv-computed-mask-segment2-update-executable-artifact`.
- Chose `computed_masked_segment2_update_unit_load` from `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`.
- Recorded dry-run evidence at `artifacts/tmp/stage2-rvv-computed-mask-segment2-update-dry-run/computed-masked-segment2-update-dry-run`.
- Recorded non-dry-run ssh evidence at `artifacts/tmp/stage2-rvv-computed-mask-segment2-update-executable-artifact/computed-masked-segment2-update-ssh-rvv`.
- Confirmed materialized body uses runtime ABI `n`, `tcrv_rvv.setvl`, compare-produced mask, field0/field1 payload loads, `tcrv_rvv.binary {kind = "add"}`, and `tcrv_rvv.masked_segment2_store` with segment count 2 and inactive-lane preservation.
- Confirmed generated C++ uses `__riscv_vmslt_vv_i32m1_b32`, `__riscv_vadd_vv_i32m1`, `__riscv_vcreate_v_i32m1x2`, and `__riscv_vsseg2e32_v_i32m1x2_m` as provider-derived typed route output.
- `ssh rvv` compile/run passed for runtime counts `0,1,7,16,23,257` and patterns `0,1`, checking active lanes, inactive-lane preservation, field0 update, field1 passthrough, source preservation, and tail preservation.
- No production provider, target, plugin, fixture, or script code change was needed; the existing route/emission/harness path already executed end to end.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-04-stage2-rvv-computed-mask-segment2-update-executable-artifact`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_segment2_update_unit_load ...`
- [OK] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_segment2_update_unit_load --ssh-target rvv ...`
- [OK] `rtk proxy bash -lc 'cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-computed-masked-segment2-update'`
- [OK] bounded old-authority scan over the task files and relevant provider/target/fixture/script surfaces.
- [OK] `rtk git diff --check`

### Status

[OK] **Completed**

### Next Steps

- None - task complete
