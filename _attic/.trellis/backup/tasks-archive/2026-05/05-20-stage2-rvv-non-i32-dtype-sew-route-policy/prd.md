# Stage2 RVV non-i32 dtype/SEW route-policy executable slice

## Goal

Make one bounded non-i32 RVV typed route instance real: an `i64` element type,
SEW64, LMUL m1, unit-stride binary selected-body path must be route-supported,
materialized through the RVV plugin route plan, emitted through the common EmitC
path, packaged by the generated-bundle script, and executable on `ssh rvv` for
representative counts if the target reports ELEN64 support.

The purpose is to prove that Stage2 RVV dtype/SEW authority is carried by the
selected typed `tcrv_rvv` body/config/capability/runtime facts, not by legacy
i32m1 route ids, helper names, artifact names, ABI strings, descriptors, or
common export branches.

## Direction Source

Hermes Direction Brief:

- Direction title: `Stage2 RVV non-i32 dtype/SEW route-policy executable slice`
- Module owner: RVV plugin-owned dtype/SEW route policy for realized typed
  `tcrv_rvv` bodies.
- Bounded positive proof: one `i64` SEW64 LMUL m1 unit-stride binary path.
- Runtime/correctness evidence: real `ssh rvv` PASS for counts such as
  7, 16, and 23 if executable correctness is claimed.

## What I Already Know

- HEAD is `a4cac384 rvv: add scalar broadcast executable path`.
- The worktree was clean at session start.
- There is no current Trellis task, so this task was created before source
  edits.
- The current RVV-first authority chain is:

  ```text
  tcrv.exec envelope
    -> selected RVV variant
    -> typed low-level tcrv_rvv vector-level body
    -> RVV plugin-owned legality / selected-body realization / route provider
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV intrinsic C/C++ or equivalent backend representation
    -> target artifact
    -> ssh rvv evidence when correctness/runtime/performance is claimed
  ```

- Stage1 policy remains strict: do not revive `RVVI32M1*`, finite positive
  `tcrv_rvv.i32_*` route authority, `!tcrv_rvv.i32m*` lowerable authority,
  `rvv-i32m1` route ids, descriptor/direct-C/source-front-door authority, or
  common/export RVV semantic branches.
- This round is Stage2 coverage on the corrected typed RVV surface, not Stage1
  cleanup and not broad dtype/LMUL expansion.

## Requirements

- Add or repair typed RVV config/body/planning support so an `i64` element type
  with SEW64 and LMUL m1 is an ordinary typed route instance.
- Keep dtype, SEW, LMUL, policy, memory form, operation kind, n/AVL, and ABI
  roles structural in the selected body/config/capability/runtime facts.
- Ensure RVVSelectedBodyRealization, if touched, only materializes typed generic
  RVV structure and does not alter computation semantics, dtype semantics,
  parameter roles, dispatch/fallback, or runtime AVL values.
- Ensure RVVEmitCRoutePlanning derives C type/header/intrinsic leaf choices from
  typed route facts and capability facts.
- Ensure provider/materializer consume the route plan and common EmitC/export
  remains neutral.
- Generated-bundle support may be extended only as tooling around the compiler
  path; it must not become compiler route authority.
- Unsupported ELEN/SEW/LMUL/dtype/op/policy combinations must fail closed with
  targeted diagnostics.

## Acceptance Criteria

- [x] Positive route/materialization evidence exists for one `i64` SEW64 LMUL
  m1 unit-stride binary selected-body path.
- [x] The `i64` path is implemented as a generic typed route instance, not as a
  finite `i64_*` route table or a new dtype-prefixed op namespace.
- [x] Exact intrinsic spelling is a provider-derived leaf only, not a route id
  or upstream authority.
- [x] Negative tests cover at least mismatched element type/SEW, unsupported
  capability, wrong LMUL/policy, missing AVL/runtime roles, and incomplete typed
  body structure as fail-closed cases.
- [x] Generated-bundle dry-run covers the `i64` path.
- [x] If ELEN64 is available on `ssh rvv`, generated-bundle runtime evidence
  passes for representative counts such as 7, 16, and 23.
- [x] If ELEN64 is unavailable, the task reports route-supported/capability
  diagnostic closure truthfully and does not invent runtime evidence.
- [x] Active-authority scan confirms no reintroduction of legacy i32/source/
  descriptor/common-export route authority.
- [x] Focused build/lit/script checks pass for touched RVV config, dialect,
  realization, route planning/provider/materializer, generated bundle, and
  tests.
- [x] `git diff --check` passes and final worktree is clean after commit.

## Out Of Scope

- Broad dtype/LMUL matrix expansion.
- Floating-point coverage.
- Conversion/widening framework.
- High-level frontend lowering or source-front-door positive routes.
- One-intrinsic wrapper dialects.
- Reduction, matmul, compare/select, broadcast, masked, or strided side quests
  beyond what is necessary to avoid regressing existing paths.
- Dashboards, report-only inventory, performance claims, or global autotuning.
- Any Python compiler-core implementation.

## Technical Notes

Read first:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- Archived Stage1 legacy i32 cleanup task.
- Archived selected-body realization extraction task.
- Archived route-provider planning extraction task.
- Archived compare/select executable ABI task.
- Archived scalar broadcast executable path task.

Likely source/test inspection targets:

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing generated-bundle tests/evidence patterns for add,
  scalar_broadcast_add, cmp_select, reduce, macc, masked, and strided paths.

## Definition Of Done

- Trellis task reflects truthful status.
- PRD acceptance criteria are either satisfied or explicitly left open with the
  exact continuation point.
- Focused checks and real RVV evidence are recorded.
- Task is finished/archived only if module behavior is complete.
- One coherent commit is created if the task is complete.

## Completion Evidence

- Added bounded SEW64/LMUL m1 config contract and i64 runtime ABI parameter
  contract for selected RVV bodies.
- Extended generic RVV dataflow verification so integer vector element width
  must match enclosing `with_vl` SEW, with SEW64 accepted only for LMUL m1.
- Extended `typed_binary_pre_realized_body` realization so the i64 add case
  materializes generic `setvl/with_vl/load/binary/store` structure with
  `!tcrv_rvv.vector<i64, "m1">`.
- Extended RVVEmitCRoutePlanning so i64 add derives `int64_t` ABI types,
  `vint64m1_t`, SEW64 metadata, and e64/i64 intrinsic leaves from the typed
  config/profile. Common EmitC stays a route consumer.
- Added positive FileCheck coverage for selected-body realization, route
  metadata/header export, EmitC materialization, generated-bundle dry-run, and
  real `ssh rvv` runtime execution.
- Added fail-closed negative coverage for i64 sub, SEW64/LMUL m2, wrong ABI C
  type, non-agnostic policy, element/SEW mismatch, missing capability
  metadata, missing AVL runtime binding, and incomplete body structure.
- Real RVV evidence:

  ```text
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --artifact-root /tmp/tianchenrv-pre-realized-i64-add-ssh \
    --run-id sshrun --overwrite --op-kind i64_add \
    --runtime-count 7 --runtime-count 16 --runtime-count 23 \
    --tcrv-opt build/bin/tcrv-opt \
    --tcrv-translate build/bin/tcrv-translate \
    --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj \
    --ssh-target rvv
  ```

  Result: `PASS op=i64_add counts=7,16,23`.

- Focused and full checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  focused lit for i64 materialization/negative/target/script plus setvl/with-vl
  tests: 6/6 passed; focused C++ RVV/plugin/construction/target tests passed;
  `cmake --build build --target check-tianchenrv -j2`: 187/187 passed;
  active-authority added-line and new-file scan had no positive hits;
  `git diff --check` passed.
