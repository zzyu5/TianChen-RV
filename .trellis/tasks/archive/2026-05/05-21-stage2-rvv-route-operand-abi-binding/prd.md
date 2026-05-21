# Stage2 RVV route operand ABI binding contract

## Goal

Introduce a production RVV-plugin-owned route operand ABI binding contract so existing executable RVV routes derive materialized operands and mirror metadata from one authoritative typed-body/runtime source. This round is bounded to the existing `macc_add` accumulator route plus representative byte-strided load and store routes.

## Why now

The previous macc accumulator executable slice exposed a real drift: route/header metadata described `lhs,rhs,acc,out,n`, but a materialized accumulator load used `out`. That means the next bottleneck is operand role binding between the typed `tcrv_rvv` body/runtime facts, RVV route planning/provider, generated source/header mirrors, and actual emitted code.

## Scope

- Add or repair a `RouteOperandBindingPlan`-equivalent contract in RVV route planning/provider ownership.
- Make `macc_add` consume the contract for `lhs`, `rhs`, `acc`, `out`, and `n`.
- Make at least one byte-strided load route consume the contract for `src`, `stride`, `out`, and `n`.
- Make at least one byte-strided store route consume the contract for `dst`, `src`, `stride`, and `n`.
- Ensure generated source/header mirror fields are checked against the same binding contract and fail closed on mismatch.
- Keep common EmitC/export materialization semantically neutral; it may consume already-derived route operands but must not infer RVV roles.

## Out of Scope

- No new Stage2 operation coverage.
- No dtype/LMUL clone batches.
- No high-level Linalg/Vector/frontend lowering.
- No source-front-door positive routes.
- No broad route matrix/dashboard/report-only work.
- No helper-only cleanup as the main achievement.
- No revival of legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door route authority.

## Requirements

- The binding contract must record which typed body value, runtime role, or mem window supplies each materialized load, store, call operand, loop/control operand, and mirror field for the bounded consumers.
- Provider emission must derive actual operand expressions from the binding contract, not from route ids, helper strings, artifact names, duplicate ad-hoc C snippets, or common/export inference.
- Missing, duplicate, swapped, stale, or mirror-inconsistent roles must fail closed with targeted diagnostics.
- The contract must be production-consumed by the named routes in the same round.
- The implementation must stay in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck for compiler behavior. Python may only support generated-bundle/evidence checks.

## Acceptance Criteria

- [x] Production RVV route planning/provider code contains an explicit operand binding contract or equivalent.
- [x] `macc_add` materialized loads/stores/call/control operands and mirrors derive from the contract.
- [x] Representative byte-strided load materialized source/stride/output/control operands and mirrors derive from the contract.
- [x] Representative byte-strided store materialized destination/source/stride/control operands and mirrors derive from the contract.
- [x] Positive lit or generated-source checks prove operand origins for macc and byte-strided memory routes.
- [x] Negative tests fail closed for accumulator loaded from `out`, store using source as destination, stride/n role swaps, missing or duplicate ABI roles, mirror/header mismatch, stale route id authority, descriptor/direct-C/source-front-door authority, and common/export semantic inference where the current test surface can express them.
- [x] Generated-bundle dry runs pass for macc and representative byte-strided load/store routes.
- [x] Real `ssh rvv` evidence passes for macc and representative byte-strided load/store routes when executable runtime evidence is claimed.
- [x] Active-authority scan shows no new positive legacy i32/RVVI32M1/source-front-door/descriptor/common-export RVV semantic authority.
- [x] `check-tianchenrv`, `git diff --check`, and focused checks pass, or any skipped check is justified by concrete environment failure.

## Completion Evidence

- Production contract: `RVVRouteOperandBindingPlan` records logical operand, runtime ABI role/C parameter, and materialized uses; `macc_add`, `strided_load_unit_store`, and `unit_load_strided_store` derive actual provider operands and route/header mirror metadata from it.
- Self-repair: added explicit logical-operand-to-runtime-role validation after noticing role swaps were not contract-local fail-closed; plugin unit coverage now checks `n`/stride role swaps, missing logical operands, duplicate logical operands, duplicate/wrong roles, missing materialized uses, and mirror-order mismatch.
- Positive/negative lit coverage is in the macc and byte-strided target artifact fixtures plus new operand-binding negative tests under `test/Conversion/EmitC/`.
- Generated-bundle dry-runs passed for explicit and pre-realized selected-body modes with `macc_add`, `strided_load_unit_store`, and `unit_load_strided_store`, counts `7,16,23`, and strides `4,8,12`.
- Real `ssh rvv` runs passed for explicit and pre-realized selected-body modes with the same route/count/stride set.
- Final checks: `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`; script `--self-test`; `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`; `build/bin/tianchenrv-rvv-extension-plugin-test`; `cmake --build build --target check-tianchenrv -j2` (`261/261`); `git diff --check`; active-authority scan on production/test diff found no new positive legacy i32/RVVI32M1/source-front-door/descriptor/common-export RVV semantic authority.

## Technical Notes

- Read first: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant previous task: `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-multiply-add-accumulator-executable-slice/`.
- Relevant code surfaces: `include/TianChenRV/Support/RuntimeABI.h`, `lib/Support/RuntimeABIContract.cpp`, `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`, `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`, `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`, `scripts/rvv_generated_bundle_abi_e2e.py`, and macc/byte-strided memory tests.
