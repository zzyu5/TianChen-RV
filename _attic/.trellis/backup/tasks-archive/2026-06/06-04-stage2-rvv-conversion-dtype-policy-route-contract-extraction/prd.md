# Stage2 RVV conversion dtype-policy route-family provider contract extraction

## Goal

Extract a provider-owned RVV conversion dtype-policy route validation contract
for the existing widening conversion route family and rewire target artifact
validation to consume that contract as a mirror/check client. The bounded
production scope is the existing selected-body `widen_i32_to_i64` and
`widen_i16_to_i32` routes; this task must not add new conversion coverage or
change emitted conversion runtime behavior.

## What I Already Know

* The repository is clean at task creation, with recent commits extracting
  compare/select, standalone reduction, MAcc, and widening dot provider
  validation contracts.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires widening conversion
  provider preflight to validate typed body/config/runtime facts, family plan,
  materialization facts, math operand binding, route-control facts, and
  statement-plan leaves before `TCRVEmitCLowerableRoute` construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVWideningConversionRouteFacts` as the provider-owned fact surface for
  `widen_i32_to_i64` and `widen_i16_to_i32`.
* Live code already has `RVVWideningConversionRouteFacts`,
  `getRVVWideningConversionRouteFacts(...)`, and provider-side
  `verifyRVVSelectedBodyWideningConversionRouteProviderFacts(...)`.
* `RVVEmitCRouteProvider.h` exposes validation contracts for standalone
  reduction, compare/select, MAcc, and widening dot routes, but not for
  conversion dtype-policy routes.
* `RVVTargetArtifactRouteFamilyValidation.cpp` still validates conversion
  target facts directly from `getRVVWideningConversionRouteFacts(...)` and
  target-local statement expectations instead of consuming a provider-owned
  conversion validation/mirror contract.

## Requirements

* Keep scope to the existing selected-body widening conversion route variants:
  `WidenI32ToI64` and `WidenI16ToI32`.
* Add a provider-owned conversion dtype-policy route validation contract in the
  RVV provider interface, derived from the existing canonical widening
  conversion facts and selected route description.
* The contract must cover provider support mirrors, route payload, header/type
  facts, route operand binding plan/summary, runtime ABI order and parameters,
  source/result dtype and SEW/LMUL relation, conversion kind/relation,
  source/destination memory form, tail/mask policy, intrinsic leaves, result
  naming, and statement-plan expectations.
* Add a provider-owned metadata mirror contract for conversion dtype-policy
  target candidate validation, including stale cross-family mirror rejection.
* Rewire conversion target artifact provider-fact validation and candidate
  mirror validation to consume the new provider-owned contracts.
* Keep target artifact validation as a consume-only client. It may compare the
  rebuilt route and candidate metadata to contract fields; it must not choose
  conversion semantics from route ids, artifact names, metadata, C strings,
  scripts, or test names.
* Preserve existing generated conversion route behavior, runtime ABI order,
  emitted conversion statements, and dry-run/runtime evidence shape.

## Acceptance Criteria

* [x] `RVVEmitCRouteProvider.h` exposes a conversion dtype-policy
      `RouteValidationContract` accessor and metadata mirror contract accessor
      analogous to the existing compare/select, MAcc, standalone reduction, and
      widening dot contracts.
* [x] Provider implementation builds the conversion contract from
      `getRVVWideningConversionRouteFacts(...)` and the selected route
      description; missing or unsupported conversion operations return no
      contract.
* [x] Target conversion provider-fact validation requires the contract before
      accepting rebuilt route payload, header/type mappings, ABI mappings,
      source/result dtype policy, conversion relation, memory forms, and
      statement plan.
* [x] Target conversion candidate mirror validation consumes the provider-owned
      metadata mirror contract and rejects stale non-conversion family mirrors.
* [x] Focused C++ target artifact tests cover positive conversion validation
      and fail-closed stale/missing/mismatched contract fields.
* [x] Focused conversion lit/dry-run tests still pass for existing
      `widen_i32_to_i64` and `widen_i16_to_i32` fixtures.
* [x] Bounded scans over touched files show no descriptor/source-front-door/
      legacy-i32/common-EmitC route authority was introduced.
* [x] No `ssh rvv` run is required unless emitted runtime ABI order, conversion
      computation, source/result dtype layout, SEW/LMUL relation, mask/tail
      behavior, correctness, or performance claims change.

## Out Of Scope

* New conversion variants, narrowing/unsigned/float/saturating conversion, new
  dtype or LMUL clone batches, frontend/source-front-door routes, or one
  intrinsic wrapper expansion.
* Changes to compare/select, reduction, MAcc, widening dot, memory movement, or
  elementwise route-family behavior except for stale cross-family rejection
  lists required by this conversion contract.
* Moving conversion semantics into target validation, common EmitC/export, route
  ids, artifact metadata, C strings, scripts, or test names.
* Runtime correctness/performance reruns unless the generated ABI or emitted
  conversion sequence changes.

## Technical Notes

* Primary files inspected: `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Closest completed pattern: compare/select provider contract extraction from
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-compare-select-route-contract-extraction/`.
* Closest conversion archive:
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-widening-conversion-production-validation-boundary/`.
* Expected implementation shape: keep `RVVWideningConversionRouteFacts` as the
  canonical fact source, add a copied-string validation contract plus metadata
  mirror contract, then route target validation through those accessors.

## Implementation Notes

* Added `RVVConversionDtypePolicyRouteValidationContract`,
  `RVVConversionDtypePolicyRouteMetadataMirrorContractSet`, and public provider
  accessors in `RVVEmitCRouteProvider.h`.
* Built the contracts in `RVVEmitCRoutePlanning.cpp` from existing
  `RVVWideningConversionRouteFacts` plus selected-route dynamic statement names.
  Unsupported/non-conversion operations return `std::nullopt`.
* Rewired conversion target provider-fact validation in
  `RVVTargetArtifactRouteFamilyValidation.cpp` to require the provider contract
  before checking route id, headers, type mappings, runtime ABI mappings,
  source/result dtype policy, conversion relation, memory forms, and statement
  plan.
* Rewired conversion target candidate mirror validation to consume the provider
  metadata mirror contract, including stale non-conversion mirror rejection.
* Extended `TargetArtifactExportTest.cpp` to validate the provider contract and
  metadata mirror contract for both existing widening conversion variants and to
  reject non-conversion operations at the accessor boundary.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  widening conversion validation/mirror contract signatures and target-consumer
  rules.

## Validation Notes

* `cmake --build build --target tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  ../test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir
  ../test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir
  ../test/Target/RVV/pre-realized-selected-body-artifact-widen-i32-to-i64.mlir`
  from `build/test` passed 3/3.
* Bounded added-diff scan found no newly introduced
  descriptor/source-front-door/source-artifact/legacy-i32/common-direct-C route
  authority patterns.
* `git diff --check`
* Spec update performed in `.trellis/spec/lowering-runtime/emitc-route.md` for
  the new conversion dtype-policy route validation and metadata mirror
  contract APIs.
* No `ssh rvv` run: this task changes provider validation contracts and target
  artifact validation consumption only; it does not change emitted conversion
  code, runtime ABI order, source/result dtype layout, SEW/LMUL relation, or
  mask/tail behavior.
