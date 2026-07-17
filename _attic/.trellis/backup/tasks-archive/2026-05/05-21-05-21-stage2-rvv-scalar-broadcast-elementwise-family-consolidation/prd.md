# Stage2 RVV scalar-broadcast elementwise production family consolidation

## Goal

Consolidate the existing `scalar_broadcast_add` executable path into a named
RVV plugin-owned scalar-broadcast elementwise production-family boundary.

This task does not add another evidence-only closeout. The current production
path already supports the bounded executable behavior:

```text
out_i32[i] = lhs_i32[i] + rhs_scalar_i32
```

for runtime `n`, SEW32, LMUL m1, agnostic policy, and runtime scalar addends.
The missing module behavior is that scalar-broadcast route ownership is still
spread through special-case route planning/provider checks. This round should
make the scalar-broadcast elementwise family an explicit validated plan or
equivalent shared boundary so selected-body realization, route planning,
provider emission, construction metadata, target mirrors, and generated
artifacts all consume the same typed body/config/runtime facts.

## Direction Source

- Direction title: `Stage2 RVV scalar-broadcast elementwise production family consolidation`.
- Current module owner: RVV plugin-owned scalar-broadcast elementwise route
  family for the existing `scalar_broadcast_add` path.
- Starting repository root: `/home/kingdom/phdworks/TianchenRV`.
- Starting branch: `main`.
- Starting HEAD: `2ee5970e rvv: prove scalar broadcast with runtime addends`.
- Starting worktree: clean.
- No `.trellis/.current-task` existed before this task was created from the
  Direction Brief.

## What I Already Know

- Prior task `05-20-stage2-rvv-vector-scalar-broadcast-executable-path`
  created the typed runtime scalar path: `rhs-scalar-value`,
  `tcrv_rvv.splat`, `memory_form = "rhs-scalar-broadcast"`, selected-body
  realization to `setvl/with_vl/load/splat/binary/store`, provider route, and
  `ssh rvv` evidence.
- Prior task `05-21-05-21-stage2-rvv-runtime-scalar-broadcast-add-production`
  strengthened generated-bundle evidence for runtime scalar addends `-37` and
  `91` at counts `7,16,23`, but changed only harness/tests/Trellis metadata.
- Specs require the authority chain to remain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level `tcrv_rvv`
  body -> RVV plugin legality/realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when executable correctness is claimed.
- Current code already has the structural body pieces, but route planning still
  handles scalar-broadcast through scattered checks such as
  `hasScalarBroadcast`, `RHSScalarBroadcast`, `ScalarBroadcastAdd`, direct
  runtime ABI order assignment, and provider branch emission.
- The consolidation target is not a broad broadcast framework. It is one
  bounded plugin-local scalar-broadcast elementwise family plan for the
  existing add instance.

## Requirements

1. Add a named RVV plugin-local scalar-broadcast elementwise route-family plan
   or equivalent shared boundary for the current `scalar_broadcast_add`
   instance.
2. The plan must be derived from typed `tcrv_rvv` body/config/runtime facts:
   lhs vector input role, RHS scalar runtime role, output role, runtime `n`/AVL,
   element type, SEW, LMUL, policy, memory form, operation kind, scalar splat,
   binary compute, store, and provider leaf facts.
3. Selected-body realization must continue consuming the same facts when it
   realizes pre-realized scalar-broadcast bodies into
   `setvl/with_vl/load/splat/binary/store`.
4. Route planning must validate the scalar-broadcast elementwise plan and then
   apply plan facts to `RVVSelectedBodyEmitCRouteDescription`; route support
   must not be recovered from route ids, helper names, artifact strings,
   descriptors, ABI names, exact intrinsic spelling, or common EmitC/export.
5. Provider emission must consume validated plan/description facts. It may
   still materialize provider-owned RVV intrinsic leaves, but the leaves must
   be outputs of RVV route planning.
6. Unsupported scalar role, dtype/config, policy, n/AVL, broadcast form,
   stale route-id authority, or incomplete typed body structure must fail
   closed with targeted diagnostics.
7. Generated-bundle dry-run and real `ssh rvv` evidence from `2ee5970e` must
   remain passing for counts `7,16,23` and scalar addends `-37,91`.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      production-family consolidation.
- [x] Route planning exposes a named scalar-broadcast elementwise family plan
      or equivalent shared boundary.
- [x] The plan is built only from typed body/config/runtime facts and provider
      derivations.
- [x] Existing selected-body realization for `rhs-scalar-broadcast` remains
      structurally aligned with the route plan.
- [x] Provider emission and target mirrors consume plan-derived facts and keep
      route ids/artifacts/ABI strings/status as mirrors only.
- [x] Positive tests cover selected-body realization, route plan/mirror facts,
      provider/header facts, EmitC materialization, and generated artifact
      output for `scalar_broadcast_add`.
- [x] Negative tests fail closed for missing scalar runtime role, scalar/vector
      dtype mismatch, invalid broadcast form, missing n/AVL, invalid
      policy/config, stale route-id authority, and incomplete typed body
      structure where those cases are meaningful for this bounded slice.
- [x] Generated-bundle dry-run passes for `scalar_broadcast_add` at counts
      `7,16,23` with RHS scalars `-37,91`.
- [x] Real `ssh rvv` generated-bundle run passes for the same counts/scalars,
      preserving runtime scalar addend, runtime `n`, and tail sentinels.
- [x] Active-authority scan shows no new positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-seed, descriptor/direct-C/source-export,
      public exact-intrinsic route authority, or common/export RVV semantic
      authority.
- [x] Focused checks, `check-tianchenrv`, `git diff --check`, task validation,
      and final clean worktree pass.

## Non-Goals

- No broad broadcast framework.
- No dtype or LMUL clone batch.
- No masked or strided broadcast expansion.
- No compare/select, contraction, matmul, Linalg, or frontend lowering.
- No source-front-door positive route.
- No one-intrinsic wrapper dialect.
- No dashboard/report-only/helper-only completion.
- No revival of legacy i32 route authority. The e32m1 case is allowed only as
  an ordinary bounded instance of the generic typed RVV surface.
- No movement of scalar role, dtype, broadcast form, policy, ABI order, or
  intrinsic choice into common EmitC/export or target metadata.

## Validation Plan

1. Validate task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-05-21-stage2-rvv-scalar-broadcast-elementwise-family-consolidation`
2. Build focused targets:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
3. Run focused C++ tests for RVV dialect/plugin/construction/target export.
4. Run focused FileCheck tests for scalar-broadcast selected-body realization,
   route plan/emission-plan mirrors, target header, EmitC materialization, and
   negative fail-closed cases.
5. Run script checks:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
6. Run generated-bundle dry-run for `scalar_broadcast_add` counts `7,16,23`
   with RHS scalars `-37,91`.
7. Run real `ssh rvv` generated-bundle correctness for the same counts/scalars.
8. Run active-authority scans over touched active RVV include/lib/script/test
   paths.
9. Run `git diff --check`.
10. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- scalar-broadcast tests under `test/Target/RVV`,
  `test/Conversion/EmitC`, `test/Transforms/LoweringBoundary`,
  `test/Dialect/RVV`, and `test/Scripts`.

## Definition Of Done

- [x] The active production scalar-broadcast add path has a named plugin-local
      family boundary or equivalent route-plan object.
- [x] Focused checks and full `check-tianchenrv` pass.
- [x] Task status, PRD, and journal notes are truthful.
- [x] Task is finished/archived when complete.
- [x] One coherent commit records implementation, validation, Trellis closeout,
      and evidence.

## Implementation Result

- Added `RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan` as the
  plugin-local family boundary for the current `scalar_broadcast_add` route.
- Route planning now derives scalar-broadcast operation, memory form, runtime
  ABI order, target leaf profile, provider support mirror, required headers,
  C type mapping, VL/vector type facts, and provider-owned intrinsic leaves
  from typed body/config/runtime facts.
- Provider emission consumes validated plan-derived leaf facts for setvl,
  vector load, RHS scalar splat, elementwise add, store, and required headers.
- Generated artifact mirrors now expose scalar-broadcast target leaf profile,
  provider-supported mirror, required header declarations, and C type mapping
  without moving semantics into common EmitC/export.
- Focused positive and fail-closed tests cover route-plan mirrors, stale mirror
  wording, stale route-id authority, stale RHS load-as-broadcast authority, and
  adjacent typed-body/runtime negative cases.

## Validation Performed

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused FileCheck for scalar-broadcast selected-body realization, emission
  plan mirrors, target header export, EmitC materialization, and negative
  selected-body/EmitC/RVV verifier cases.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Generated-bundle dry-run:
  `scalar_broadcast_add`, counts `7,16,23`, RHS scalars `-37,91`.
- Real `ssh rvv` generated-bundle run:
  `PASS op=scalar_broadcast_add counts=7,16,23 rhs_scalars=-37,91`.
- Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-rvv-dialect-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `git diff --check`
- Active-authority scan over the diff: new exact `__riscv_*_i32m1` mentions are
  RVV plugin-owned leaf outputs/tests; the only new `rvv-i32m1` mention is a
  fail-closed stale route-id negative test; no new positive `RVVI32M1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export RVV semantic authority.
- `cmake --build build --target check-tianchenrv -j2` passed, 248/248.

## Final Status

Completed and ready to archive.
