# Stage2 RVV conversion dtype-policy runtime AVL/VL sole-authority cleanup

## Goal

Make the RVV conversion dtype-policy target-artifact runtime-boundary consumer
treat `RVVRuntimeAVLVLSelectedBoundaryContract` as the sole acceptance
authority for runtime `n` / AVL / VL facts. Widening conversion may keep
route-local runtime/control fields and candidate metadata only as mirrors
checked after the shared selected-boundary contract.

## What I already know

* The previous completed commit is
  `87032c2d rvv: make reduction runtime mirrors non-authoritative`.
* The live conversion dtype-policy contract already embeds
  `RVVRuntimeAVLVLSelectedBoundaryContract`, based on the archived
  `conversion-dtype-policy-runtime-avl-vl-contract` task.
* Live target validation already calls
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` for conversion before
  most conversion payload checks.
* Residual conversion code still treats route-local runtime fields with
  provider-derived/provider-owned wording, including direct runtime ABI/order
  checks around `validateRVVConversionDtypePolicyRuntimeABIFacts(...)` and
  metadata labels such as selected typed RVV conversion dtype-policy runtime
  AVL/VL control plan / runtime ABI order.
* The reduction sole-authority cleanup established the local pattern: validate
  the shared runtime AVL/VL selected-boundary contract first, then validate any
  retained `runtimeControlPlanID`, `runtimeABIOrder`, `setvl`, VL type, and
  EmitC AVL/VL statement-name copies as `route-local runtime AVL/VL mirror`
  facts only.

## Requirements

* Conversion target validation must consume
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before accepting or
  checking route-local runtime/control mirrors.
* Retained conversion `runtimeControlPlanID`, `runtimeABIOrder`,
  `setVLIntrinsic`, `vlCType`, `emitCFullChunkVLName`, `emitCLoopVLName`, and
  `emitCLoopInductionName` fields must be documented or validated only as
  route-local runtime AVL/VL mirrors against the embedded selected-boundary
  contract.
* Missing or stale conversion route-local runtime mirrors must fail closed with
  diagnostics that say `route-local runtime AVL/VL mirror`.
* Conversion runtime ABI parameter validation may continue to check the
  provider ABI parameter list for `lhs`, `out`, and runtime `n`, but runtime
  ABI order acceptance must flow through the embedded runtime AVL/VL contract
  and the route-local mirror check.
* Conversion candidate metadata labels for `tcrv_rvv.runtime_control_plan` and
  `tcrv_rvv.runtime_abi_order` must be explicit mirror-only labels.
* Existing conversion dtype/config/source/result validation remains
  plugin-owned and unchanged except where needed to route runtime authority
  through the shared selected-boundary contract.
* Common EmitC/export remains neutral. Runtime control must not be inferred
  from route ids, artifact names, tests, manifests, C strings, descriptors,
  or mirror metadata.

## Acceptance Criteria

* [x] Conversion target validation calls the shared runtime AVL/VL
      selected-boundary validator before route-local runtime mirror checks.
* [x] Conversion retained runtime/control fields are checked through the
      route-local runtime AVL/VL mirror helper or equivalent diagnostics.
* [x] Stale conversion route-local runtime mirrors fail closed with diagnostic
      fragments that include `route-local runtime AVL/VL mirror`.
* [x] Conversion candidate metadata labels for runtime control plan and runtime
      ABI order are mirror-only.
* [x] Focused conversion target tests cover stale runtime metadata mirrors not
      overriding the embedded selected-boundary contract.
* [x] Existing conversion source/result dtype policy, conversion relation,
      memory form, headers, type mappings, ABI mappings, statement-plan, and
      stale non-conversion family validation remain intact.
* [x] `tianchenrv-target-artifact-export-test` passes.
* [x] `tianchenrv-rvv-extension-plugin-test` is run if provider/planning code
      changes.
* [x] Focused conversion lit/generated-bundle dry-run checks remain green.
* [x] Before/after bounded grep confirms old selected-typed runtime-control /
      runtime-ABI labels were removed from touched conversion planning/target
      paths.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/source-front-door authority, route-id or
      artifact-name authority, or exact `__riscv_*_i32m1` route authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Evidence Plan

* Build focused tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused conversion lit/generated-bundle dry-run after locating the live
  lit filter.
* Run before/after bounded grep for conversion runtime-control/runtime-ABI
  labels in touched validation/planning paths.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-conversion-runtime-avl-vl-sole-authority-cleanup`

## SSH RVV Rationale

Do not run `ssh rvv` unless implementation changes emitted C/C++, runtime ABI
order/counts, statement ordering, conversion correctness, runtime behavior, or
performance behavior. The intended diff is validation/metadata/test/spec
cleanup only.

## Out Of Scope

* No new conversion operations, dtype/LMUL cases, intrinsic cases, or
  conversion coverage expansion.
* No reduction, contraction, compare/select, elementwise, scalar-splat-store,
  memory, source-front-door, high-level frontend, dashboard, or broad route
  family sweep.
* No movement of RVV semantics into common EmitC/export.
* No inference of runtime control from route ids, artifact names, test names,
  manifests, C strings, descriptors, or mirror metadata.

## Definition Of Done

Conversion dtype-policy target route validation accepts runtime `n` / AVL / VL
facts only through the embedded runtime AVL/VL selected-boundary contract.
Retained route-local runtime fields and candidate metadata are explicit mirrors,
focused checks pass, the task is finished/archived, and a coherent commit
records the production cleanup.

## Implementation Summary

* Documented `RVVConversionDtypePolicyRouteValidationContract`
  `runtimeControlPlanID` and `runtimeABIOrder` as route-local runtime AVL/VL
  mirrors whose acceptance authority is the embedded runtime AVL/VL contract.
* Rewired conversion target validation so the shared
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` check runs before
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)`.
* Removed direct conversion target-side runtime ABI order and runtime control
  plan acceptance checks from conversion-local provider-owned validation.
* Kept conversion ABI parameter validation for the provider `lhs`, `out`, and
  runtime `n` parameter list.
* Kept conversion source/result dtype policy, conversion relation, source and
  destination memory forms, headers, type mappings, ABI mappings, and
  statement-plan validation intact.
* Changed conversion candidate metadata runtime labels to
  `route-local runtime AVL/VL control plan mirror` and
  `route-local runtime AVL/VL ABI order mirror`.
* Added target C++ coverage that conversion route-local runtime mirrors match
  the embedded selected-boundary contract, conversion metadata labels are
  mirror-only, and stale runtime-control/runtime-ABI candidate metadata fails
  closed with route-local runtime AVL/VL mirror diagnostics.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` to record conversion
  route-local runtime mirror and candidate runtime metadata mirror obligations.

## Evidence Results

* Built focused targets:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Ran:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
* Ran:
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
  (`RVV extension plugin smoke test passed`)
* Ran focused conversion lit from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'widen-i(16-to-i32|32-to-i64).*dry-run'`
  with 3 passed and 474 excluded.
* Self-repair: the first lit invocation from the repository root failed because
  `build/test/lit.site.cfg.py` resolves `../../test/lit.cfg.py` relative to
  the current working directory. Rerunning from `build/test` passed.
* Ran bounded grep for old conversion runtime labels in touched source/test/spec
  files: no matches.
* Ran added-line old-authority scan over touched source/test/spec files for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  source-front-door/source-artifact, descriptor/direct-C/source-export,
  exact `__riscv_*_i32m1`, status/readiness, `emission_plan`, route-id, and
  artifact-name authority: no matches.
* Ran `rtk git diff --check`: passed.

## SSH RVV Decision

`ssh rvv` was not run. This round changes provider/target validation,
metadata mirror labels, C++ target tests, Trellis task files, and spec text
only. It does not change emitted C/C++, runtime ABI order/counts, statement
ordering, conversion computation, runtime behavior, correctness behavior, or
performance behavior.

## Spec Update Judgment

Spec update was required because conversion dtype-policy now has explicit
route-local runtime AVL/VL mirror obligations after the embedded selected
boundary. The update is limited to `.trellis/spec/lowering-runtime/emitc-route.md`.
