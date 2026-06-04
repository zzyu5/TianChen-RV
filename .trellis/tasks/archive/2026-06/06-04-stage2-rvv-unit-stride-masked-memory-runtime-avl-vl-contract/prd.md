# Stage2 RVV unit-stride masked memory runtime AVL/VL contract migration

## Goal

Migrate the existing unit-stride masked memory route-family validation
contract so `RVVUnitStrideMaskedMemoryRouteValidationContract` embeds
`RVVRuntimeAVLVLSelectedBoundaryContract`, and require target artifact
validation to consume that embedded runtime contract before accepting masked
unit-load/store route payloads, runtime/control facts, mask/tail policy facts,
statement-plan facts, ABI/type/header facts, and artifact mirrors.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv unit-stride masked memory body
  -> RVV plugin-owned unit-stride masked memory route facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> RVVUnitStrideMaskedMemoryRouteValidationContract
  -> target artifact validation consumes the embedded contract
  -> provider-built TCRVEmitCLowerableRoute setvl/mask/load/store statements
  -> common EmitC materialization
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV unit-stride masked memory runtime AVL/VL contract migration`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `a467a5cb rvv: consume runtime AVL VL contract for elementwise`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no spawned
  subagents or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` keep RVV route authority in
  the selected `tcrv.exec` envelope, typed low-level `tcrv_rvv` body, RVV
  plugin-owned legality/route provider, common EmitC materialization, and
  target artifact validation.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned authority for
  runtime `n`/AVL/VL selected-boundary facts before target artifact acceptance.
* The previous unit-stride masked-memory provider-contract task already added
  `RVVUnitStrideMaskedMemoryRouteValidationContract` and rewired target
  validation away from direct raw unit-stride masked route fact consumption.
* Live inspection shows `RVVUnitStrideMaskedMemoryRouteValidationContract`
  still lacks `runtimeAVLVLContract`, and its provider builder still returns
  without calling `getRVVRuntimeAVLVLSelectedBoundaryContract(...)`.
* Live inspection shows the unit-stride masked-memory target consumer still
  checks runtime control plan, ABI order, setvl/VL, and statement-plan facts
  through family-local fields rather than first validating an embedded
  `RVVRuntimeAVLVLSelectedBoundaryContract`.

## Requirements

* Add `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` to
  `RVVUnitStrideMaskedMemoryRouteValidationContract`.
* Populate the embedded contract in
  `getRVVUnitStrideMaskedMemoryRouteValidationContract(...)` from canonical
  provider-owned facts: SEW, LMUL, tail/mask policy, config contract,
  `setVLIntrinsic`, VL C type, runtime ABI order, runtime ABI parameter list,
  and the unit-stride masked-memory consumer label.
* Target validation must call
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before accepting
  route-local runtime/control, header/type, ABI, statement-plan, mask/load/store
  facts, and before candidate mirror acceptance can succeed.
* Statement-plan validation for the unit-stride masked-memory rebuilt route
  must consume setvl callee, VL type, full-chunk VL, loop VL, loop induction,
  and runtime n/AVL ABI parameter from the embedded runtime contract rather
  than mutable route-description mirrors.
* Fail closed for missing, stale, or mismatched runtime AVL source, runtime VL
  contract id, selected `with_vl` boundary/scope, setvl intrinsic, VL type,
  full-chunk VL, loop VL, loop induction, runtime n ABI role/order, remaining
  AVL metadata, pointer advancement metadata, and masked-memory statement-plan
  facts.
* Keep common EmitC/export neutral. Do not infer runtime control from route
  ids, artifact names, test names, manifests, C strings, descriptor residue,
  source-front-door metadata, or mirror metadata.

## Acceptance Criteria

* [x] `RVVUnitStrideMaskedMemoryRouteValidationContract` embeds
      `RVVRuntimeAVLVLSelectedBoundaryContract`.
* [x] The provider builder fills the embedded runtime contract through
      `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` using provider-owned
      typed/config/runtime facts.
* [x] Target unit-stride masked-memory description validation calls the shared
      runtime AVL/VL selected-boundary validator before route-local checks.
* [x] Rebuilt route statement validation uses the embedded runtime contract for
      setvl/VL/loop/runtime n facts and rejects stale statement plans.
* [x] Focused C++ target coverage proves positive embedded contract population
      and negative stale/missing runtime contract facts for unit-stride masked
      memory.
* [x] Focused unit-stride masked-memory generated-bundle/lit coverage remains
      green.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy i32, route-id, artifact-name, descriptor,
      source-front-door/source-export/direct-C, common-EmitC, exact-intrinsic,
      or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal/archive state, and commit state are
      truthful.

## Non-Goals

* Do not add new memory operations.
* Do not migrate computed-mask indexed/strided memory, reductions,
  elementwise, conversion, MAcc, widening-dot, scalar/offload/IME, source
  front-door routes, or high-level frontend lowering.
* Do not move RVV semantics into common EmitC/export.
* Do not run broad smoke matrices unless focused checks expose a shared
  regression.
* Do not run `ssh rvv` unless emitted C, runtime ABI order, runtime counts,
  statement ordering, masked load/store behavior, correctness behavior, or
  performance behavior changes.

## Evidence Plan

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit/generated-bundle filter for unit-stride masked memory routes.
* Bounded scan over touched files for old route authority and direct
  runtime-control reconstruction residue.
* `rtk git diff --check`
* No `ssh rvv` if this remains validation-contract ownership only and emitted
  C/runtime behavior is unchanged.

## Implementation Results

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` now embeds
  `RVVRuntimeAVLVLSelectedBoundaryContract runtimeAVLVLContract` in
  `RVVUnitStrideMaskedMemoryRouteValidationContract`.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` fills the embedded runtime
  contract through `getRVVRuntimeAVLVLSelectedBoundaryContract(...)` using
  provider-owned unit-stride masked-memory SEW, LMUL, tail/mask policy,
  config contract, setvl callee, VL C type, runtime ABI order, and runtime ABI
  parameters.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` validates the
  embedded runtime AVL/VL selected-boundary contract before accepting
  route-local runtime control facts, runtime ABI order, headers/types,
  masked-memory payload facts, statement-plan facts, or mirrors.
* The unit-stride masked-memory route statement path now passes the embedded
  runtime contract into the existing compare/select mask statement-plan
  checker so setvl/VL/loop/runtime n facts come from the provider contract.
* `test/Target/TargetArtifactExportTest.cpp` checks positive embedded runtime
  contract population for unit-stride masked-memory contract users and adds
  fail-closed mutations for stale runtime AVL source, runtime VL contract,
  selected `with_vl` scope, setvl callee, VL C type, full-chunk VL, loop VL,
  loop induction, runtime n ABI role, and pointer advancement metadata.
* `.trellis/spec/lowering-runtime/emitc-route.md` records the promoted
  unit-stride masked-memory runtime AVL/VL selected-boundary contract field,
  consumer order, validation matrix, and test requirements.

## Evidence Results

* Build:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
  passed after self-repair. Existing switch/unused-function warnings remained.
* Target test:
  `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
* RVV plugin test:
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* Focused lit/generated-bundle:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'masked-unit-(load-store|store)|computed-masked-unit-load-store|runtime-scalar-cmp-masked-(memory|store|load-store)'`
  from `build/test` selected 24 of 477 tests and all 24 passed.
* `rtk git diff --check` passed.
* Direct target raw-fact scan:
  `rtk rg -n "RVVUnitStrideMaskedMemoryRouteFacts|getRVVUnitStrideMaskedMemoryRouteFacts|getRVVUnitStrideMaskedMemoryFactsForDescription" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  returned no matches.
* Added-line old-authority scan over touched source/test files returned no
  matches for legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact/source-export,
  descriptor/direct-C, or exact `__riscv_*_i32m1` authority.
* `ssh rvv` was not run because this task changed validation-contract
  ownership and target acceptance order only; emitted C, runtime ABI order,
  runtime counts, statement ordering, masked load/store behavior, correctness
  behavior, and performance behavior were not changed.
