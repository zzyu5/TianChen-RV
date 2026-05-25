# Journal - codex (Part 15)

> Continuation from `journal-14.md` (archived at ~2000 lines)
> Started: 2026-05-25

---



## Session 206: Stage2 RVV elementwise broadcast route closure

**Date**: 2026-05-25
**Task**: Stage2 RVV elementwise broadcast route closure
**Branch**: `main`

### Summary

Closed scalar_broadcast_sub through the RVV-owned elementwise arithmetic statement-plan boundary, added focused C++/lit/generated-bundle evidence, recorded ssh rvv correctness for counts 0/7/16/23 and rhs scalars -37/91, updated the RVV plugin spec, and archived Trellis state.

### Main Changes

- Added ordinary Add/Sub/Mul elementwise arithmetic as a route-control
  provider-plan consumer for the existing vector-RHS-load path.
- Required the elementwise arithmetic statement-plan builder to consume the
  shared route-control provider plan before building setvl/load/binary/store
  steps for ordinary elementwise routes.
- Added focused C++ positive and fail-closed coverage for typed config,
  selected capability, runtime AVL/VL, policy, same-analysis materialization,
  and operand-binding ownership.
- Updated the RVV plugin spec with the ordinary elementwise route-control
  consumer contract.

### Git Commits

| Hash | Message |
|------|---------|
| `same-commit` | (see git log) |

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-elementwise-route-control-plan`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused lit/FileCheck for explicit/pre-realized/generic ordinary
  elementwise artifacts: 7/7 passed.
- [OK] Bounded changed-line authority scan found no new legacy/source-front-door
  or mirror-only authority additions.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 228: Stage2 RVV direct-provider contraction route-provider owner

**Date**: 2026-05-25
**Task**: Stage2 RVV direct-provider contraction route-provider owner
**Branch**: `main`

### Summary

Moved direct-provider contraction statement construction behind an RVV-owned
route-provider owner boundary while keeping route construction, common EmitC
materialization, selected-body realization, emitted statement order, and
runtime ABI behavior unchanged.

### Main Changes

- Added `RVVSelectedBodyDirectContractionRouteProviderOwner` and
  `RVVSelectedBodyDirectContractionRouteStatementPlan` APIs, including exact-one
  owner selection and non-consumer empty-plan behavior.
- Moved widening MAcc, plain/strided dot-reduction, and computed-mask
  dot-reduction statement construction into the direct contraction owner.
- Rewired `RVVEmitCRouteProvider` to attach the owner-returned pre-loop and loop
  statements before the generic provider-local fallback path.
- Removed the active central provider branches for direct contraction operand
  binding, source loads, computed-mask dot loops, products, reductions, and
  stores.
- Updated the RVV plugin spec with the durable direct contraction
  route-provider owner boundary and route-control provider-plan consumption
  contract.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Direct-provider contraction dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_strided_input_widening_dot_reduce_add --run-id direct-contraction-computed-mask-strided ...`
- [OK] Migrated statement-plan dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind add --run-id migrated-add-statement-plan ...`
- [OK] `git diff --check`
- [OK] Bounded added-line authority scan found only spec prohibition language for
  legacy i32/source-front-door/descriptor/direct-C/source-export/common-EmitC/
  metadata/route-id authority, and no new active authority leaks.
- [OK] Provider scan found no active central direct contraction operation-name
  branches in `RVVEmitCRouteProvider.cpp`; only the direct owner call remains.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

### Runtime Evidence

No `ssh rvv` rerun was needed because this round changed route-provider
ownership and fail-closed validation boundaries only. It did not change emitted
RVV computation semantics, runtime ABI, dispatch/fallback behavior, target
artifact semantics, or performance claims.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 228: Stage2 RVV route-provider family owner registry

**Date**: 2026-05-25
**Task**: Stage2 RVV route-provider family owner registry
**Branch**: `main`

### Summary

Moved migrated RVV statement-plan provider consumption behind an explicit owner registry, preserved provider-built route behavior, documented the new executable API in the RVV plugin spec, and verified focused plus full checks.

### Main Changes

- Added `RVVSelectedBodyMigratedRouteStatementPlanOwner`, `getRVVSelectedBodyMigratedRouteStatementPlanOwners()`, and `isRVVSelectedBodyMigratedRouteStatementPlanConsumer(...)`.
- Replaced the central manual sequence in `getRVVSelectedBodyMigratedRouteStatementPlan(...)` with registry-selected exact-one owner dispatch.
- Covered migrated statement-plan owners for elementwise arithmetic, compare/select, widening conversion, standalone reduction, plain MAcc, base memory movement, computed-mask memory, segment2 memory, and computed-mask accumulation.
- Added focused C++ coverage for owner count/order/family tags, hook presence, exact-once classification, non-consumer empty-plan behavior, and missing dependency failure through the migrated owner boundary.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to capture the migrated statement-plan owner registry signatures, contracts, validation matrix, and required tests.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-route-provider-family-owner-registry`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind cmp_select --runtime-count 0 --runtime-count 7 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_route_provider_owner_registry --run-id direct-route-entry-cmp-select --overwrite`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --runtime-count 0 --runtime-count 7 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_route_provider_owner_registry --run-id explicit-add-statement-plan --overwrite`
- [OK] Added-line authority scan over touched C++/test files found no new legacy i32/source-front-door/descriptor/ABI/artifact/script/common-EmitC/metadata/route-id authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

### Runtime Evidence

No `ssh rvv` rerun was needed because emitted target sequence, runtime ABI, dispatch/fallback behavior, statement order, and computation semantics were preserved; this round changed route-provider ownership dispatch only.

### Continuation

- Direct-provider contraction statement construction still lives in `RVVEmitCRouteProvider.cpp`; extract it into a route-provider owner in the next round while preserving its route-control and math operand-binding checks.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 227: Stage2 RVV selected-body realization owner-local hooks

**Date**: 2026-05-25
**Task**: Stage2 RVV selected-body realization owner-local hooks
**Branch**: `main`

### Summary

Migrated the route-entry-capable non-elementwise RVV selected-body realization families from the shared existing-family branch helper to owner-local hooks for standalone reduction, MAcc, and base memory movement. Deferred non-route-entry owners remain explicit shared-helper continuation points.

### Main Changes

- Added owner-local realization hooks for `standalone reduction`, `MAcc`, and `base memory movement` in `RVVSelectedBodyRealization.cpp`.
- Updated the selected-body realization owner registry so migrated owners dispatch directly to their hooks, while the shared helper fails closed if asked to own those migrated families.
- Preserved existing typed validators/materializers and route-control provider evidence; no common EmitC, artifact, runtime ABI, dispatch/fallback, or computation semantic changes were made.
- Added focused C++ tests for distinct migrated hook pointers, deferred shared-helper continuation owners, scalar-broadcast MAcc route-entry realization, explicit selected-body unaffected evidence, and owner-local fail-closed MAcc diagnostics for family mismatch, memory_form/op_kind mismatch, LMUL/config mismatch, and runtime n/AVL role mismatch.
- Recorded that no `.trellis/spec` update was required because the existing RVV plugin spec already defines the owner-local selected-body realization and route-control registry contracts.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `rtk ninja -C build tianchenrv-rvv-extension-plugin-test`
- [OK] `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Representative selected-body artifact lit filter for pre-realized strided-load/unit-store, standalone-reduce-add, scalar-broadcast-macc-add, and explicit scalar-broadcast MAcc.
- [OK] Representative generated-bundle dry-run lit filter for direct pre-realized route entry, scalar-broadcast MAcc, and standalone reduction.
- [OK] `rtk ninja -C build check-tianchenrv`: 379/379 passed.
- [OK] `rtk git diff --check`
- [OK] Changed-line authority scan over touched C++/test files found no new legacy i32/source-front-door/descriptor/ABI/artifact/common-EmitC route-authority additions.

### Status

[OK] **Completed**

### Next Steps

- Migrate the remaining deferred non-route-entry selected-body realization owners away from `realizePreRealizedRVVExistingFamilyOwner(...)`: runtime scalar splat-store, runtime scalar computed-mask store/load-store, reduction, computed-mask MAcc, contraction, widening conversion, computed-mask memory, and segment2 memory.


## Session 223: Stage2 RVV computed-mask accumulation route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV computed-mask accumulation route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated existing computed-mask MAcc accumulation routes with the shared RVV
route-control provider plan before computed-mask accumulation statement
planning, added focused fail-closed C++ coverage, updated the RVV plugin spec,
and verified focused/full checks.

### Main Changes

- Added `controlsComputedMaskAccumulation` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Made `computed_masked_macc_add` and
  `runtime_scalar_cmp_masked_macc_add` route-control consumers for their
  existing computed-mask accumulation memory forms.
- Required route-control construction to validate same-analysis accumulation
  family/materialization facts, runtime AVL/VL, typed config, selected target
  capability, policy, runtime ABI, mask-producer classification,
  accumulator/MAcc contracts, materialization leaves, and mirrors before
  computed-mask accumulation statement construction.
- Required the computed-mask accumulation statement-plan builder to consume the
  route-control provider plan and same-analysis math operand-binding facts
  before building setvl/load/splat/compare/MAcc/merge/store statements.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record computed
  mask accumulation MAcc as a route-control provider-plan consumer.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-computed-mask-accum-route-control`
- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded added-line authority scan; broad scan only reported spec
  mirror-only wording, local `result` variables, and intentional negative
  `metadata_n`; narrow scan found no route-id/descriptor/source-front-door/
  source-artifact/direct-C/source-export/legacy-i32/script/common-EmitC/
  metadata/artifact-derived authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379 tests passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 221: Stage2 RVV widening conversion route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV widening conversion route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated the existing active RVV widening conversion routes with the shared
route-control provider-plan boundary before widening conversion statement
planning. The route-control owner now validates same-analysis conversion
family/materialization facts, runtime AVL/VL, typed config, selected
capability, policy, source/result type policy, conversion-form facts, runtime
ABI mirrors, and math operand-binding ownership before provider route
attachment.

### Main Changes

- Added `controlsWideningConversion` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Made the existing `widen_i32_to_i64` and `widen_i16_to_i32` routes explicit
  route-control consumers without adding conversion coverage.
- Required `getRVVSelectedBodyWideningConversionRouteStatementPlan(...)` to
  consume the route-control provider plan before constructing setvl/load/
  convert/store statements.
- Migrated `widen_i32_to_i64` through the same widening conversion
  statement-plan path already used by `widen_i16_to_i32`, keeping emitted
  semantics and ABI unchanged.
- Added focused C++ positive and fail-closed coverage for conversion
  route-control consumption, stale same-analysis materialization, stale
  source/type markers, policy mismatch, unsupported capability/config, and
  provider statement-plan attachment.
- Updated the RVV plugin spec to list widening conversion as a route-control
  consumer.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-widening-conversion-route-control`
- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused lit/FileCheck and dry-run script filter from `build/test`:
  60 selected tests passed for `widen-i(16|32)|widening-conversion|conversion`.
- [OK] Bounded added-line authority scan found no new source-front-door,
  descriptor, direct-C/source-export, common-EmitC, artifact/script, route-id,
  ABI-string, or legacy-i32 authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

### Runtime Evidence

No `ssh rvv` rerun was needed because emitted target statement order, runtime
ABI, target mirrors, generated-bundle script behavior, and runtime behavior
were unchanged.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 220: Stage2 RVV segment2 memory route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV segment2 memory route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated existing plain and computed-mask segment2 memory statement planning
with the shared RVV route-control provider-plan boundary before segment2
statement construction. The route-control owner now validates same-analysis
family/materialization facts, runtime AVL/VL, typed config, selected
capability, policy, segment direction, memory form, computed-mask segment facts,
runtime ABI mirrors, and operand-binding ownership before provider route
attachment.

### Main Changes

- Added `controlsSegment2Memory` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Made the four existing production-active segment2 memory routes explicit
  route-control consumers:
  `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`,
  `computed_masked_segment2_load_unit_store`, and
  `computed_masked_segment2_store_unit_load`.
- Required `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)` to consume
  the route-control provider plan before constructing segment2 setvl/load/
  compare/tuple/segment/store statements.
- Added focused C++ positive and fail-closed coverage for segment2
  route-control consumption, stale same-analysis materialization, runtime AVL,
  policy, capability, runtime ABI mirror, direction, memory-form, and operand
  binding facts.
- Updated the RVV plugin spec to list segment2 memory as a route-control
  consumer and to document the segment2 statement-plan route-control dependency.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-segment2-route-control`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `git diff --check`
- [OK] Bounded added-line authority scan: only spec mirror-only text, ordinary
  `description` field access, and negative stale `metadata_n` ABI mirror test
  matched; no new source-front-door, descriptor, common-EmitC, artifact/script,
  or legacy-i32 authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

### Runtime Evidence

No `ssh rvv` rerun was needed because emitted target statement order, runtime
ABI, target mirrors, generated artifacts, and runtime behavior were unchanged.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 214: Stage2 RVV runtime AVL/VL and policy provider-plan boundary

**Date**: 2026-05-25
**Task**: Stage2 RVV runtime AVL/VL and policy provider-plan boundary
**Branch**: `main`

### Summary

Introduced a bounded RVV plugin-local route-control provider-plan boundary for
the already migrated base memory movement and standalone reduction families.
The boundary joins typed config facts, selected target capability facts, and the
owning family runtime AVL/VL control plan before route statement/provider
construction, keeping target artifacts and scripts as mirror-only consumers.

### Main Changes

- Added structural selected target capability facts to selected-body route
  analysis, so capability/config/policy validation is not derived from route
  description mirror strings.
- Added `RVVSelectedBodyRouteControlProviderPlan` and
  `getRVVSelectedBodyRouteControlProviderPlan(...)` to validate runtime AVL/VL,
  SEW/LMUL, tail policy, mask policy, config contract, runtime ABI order,
  setvl/with_vl names, loop-control fields, and selected target capability
  legality for supported consumers.
- Rewired base memory movement provider-plan construction to consume the
  route-control provider plan before returning the ordered statement plan.
- Rewired standalone reduction statement-plan construction to consume the
  route-control provider plan before returning the ordered statement plan.
- Added focused C++ positive and fail-closed tests for base memory movement and
  standalone reduction route-control consumption.
- Updated the RVV plugin spec with the durable Route-Control Provider-Plan
  Boundary contract, validation matrix, test requirements, and wrong-vs-correct
  examples.

### Git Commits

Pending final session commit in this turn.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-vl-policy-provider-plan`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused lit/FileCheck from `build/test`:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='pre-realized-selected-body-artifact-(strided-load-unit-store|standalone-reduce-add)'`
- [OK] Bounded authority scan over touched planning/provider/test/spec files:
  no new name-, metadata-, descriptor-, ABI-string-, script-, artifact-,
  common-EmitC-, source-front-door-, or legacy-i32-derived AVL/VL or policy
  authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.

### Status

[OK] **Completed**

### Next Steps

- Continue route-control provider-plan adoption for any remaining migrated RVV
  families that still rely on their pre-existing local checks instead of this
  shared owner boundary.

*** End of File


## Session 212: Stage2 RVV route-family module boundary closure

**Date**: 2026-05-25
**Task**: Stage2 RVV route-family module boundary closure
**Branch**: `main`

### Summary

Closed scalar_broadcast_macc_add through an explicit RVV plugin-local
route-family module boundary. The scalar-broadcast MAcc family plan now owns
legality, typed body/config/runtime facts, route mirrors, materialization
facts, ordered statement planning, and provider verification before common
EmitC and target artifact mirror consumption.

### Main Changes

- Added `RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan` and registered the
  `scalar-broadcast MAcc` owner in the reduction/accumulation/contraction
  route-family registry.
- Rewired central scalar_broadcast_macc_add route analysis to derive/apply the
  family plan instead of populating target leaves, runtime ABI, header/type,
  intrinsic, and layout mirrors ad hoc.
- Extended materialization facts, math operand-binding verification, and
  plain/scalar-broadcast MAcc statement planning so provider route
  construction consumes the family-owned plan.
- Mirrored `tcrv_rvv.scalar_broadcast_macc_route_family_plan` through
  emission-plan metadata, target bundle validation, generated-bundle evidence,
  and header artifact comments as mirror-only evidence.
- Updated the RVV plugin spec with the durable plain/scalar-broadcast MAcc
  statement-plan and family-plan contract.

### Git Commits

Pending final session commit in this turn.

### Testing

- [OK] Focused build:
  `ninja -C build tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate`.
- [OK] Focused C++ test:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Explicit and pre-realized FileCheck pipelines for emission-plan and
  generated-header route-family mirrors.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `scalar_broadcast_macc_add`.
- [OK] Real `ssh rvv` generated-bundle ABI/e2e for counts `7,16,23` and RHS
  scalars `-37,91`.
- [OK] Bounded authority scan found no new descriptor/source-front-door,
  common-EmitC, harness, name-derived, metadata-derived, or legacy i32 route
  authority.
- [OK] `git diff --check`.
- [OK] `ninja -C build check-tianchenrv`: 379/379 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 211: Stage2 RVV selected dispatch fallback envelope closure

**Date**: 2026-05-25
**Task**: Stage2 RVV selected dispatch fallback envelope closure
**Branch**: `main`

### Summary

Closed the bounded scalar_broadcast_macc_add selected dispatch/fallback
envelope by requiring structural tcrv.exec dispatch-case and fallback facts
before RVV route construction, mirroring the verified envelope through
provider/artifact/header evidence, and proving the selected legal generated
bundle on ssh rvv.

### Main Changes

- Added RVV provider-side selected dispatch envelope collection for dispatch
  case routes, including selected case, runtime guard, and fallback variant
  coherence checks.
- Kept fallback as an envelope boundary by refusing RVV route construction for
  dispatch-fallback roles and by requiring fallback targets to be direct
  conservative fallback variants.
- Mirrored `tcrv_rvv.selected_dispatch_case_mirror` and
  `tcrv_rvv.selected_dispatch_fallback_mirror` through route metadata,
  target bundle validation, and generated header evidence.
- Added positive artifact/header FileCheck coverage and negative diagnostics
  for missing required runtime guard, wrong guard role, missing fallback target,
  and ineligible fallback target, alongside existing capability-gating
  negatives.

### Git Commits

Pending final session commit in this turn.

### Testing

- [OK] Focused lit for selected dispatch/fallback envelope, explicit and
  pre-realized scalar_broadcast_macc_add artifact/header mirrors, and existing
  target-capability gating coverage: 5/5 passed.
- [OK] Generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/selected-dispatch-fallback-envelope-dry`.
- [OK] Real `ssh rvv` generated-bundle ABI/e2e for
  `scalar_broadcast_macc_add`, counts `7,16,23`, rhs scalars `-37,91`.
- [OK] Changed-line authority scan found no new descriptor/source-front-door,
  common-EmitC, harness, name-derived, metadata-derived, or legacy i32 route
  authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j 8`: 379/379 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 207: Stage2 RVV multi-family selected-body composition closure

**Date**: 2026-05-25
**Task**: Stage2 RVV multi-family selected-body composition closure
**Branch**: `main`

### Summary

Closed scalar_broadcast_macc_add as a bounded multi-family selected-body composition route through typed RVV body facts, RVV-owned statement planning, common EmitC materialization, generated-bundle dry-run evidence, fail-closed diagnostics, full check-tianchenrv, and real ssh rvv correctness for counts 0/7/16/23 with rhs scalars -37/91.

### Main Changes

- Added `RVVSelectedBodyRealizationOwner` plus public owner registry lookup APIs.
- Routed pre-realized body discovery, route-entry realization selection, and the public selected-body realization entrypoint through owner selection.
- Preserved existing family validators/materializers while documenting the remaining continuation point: split non-elementwise owner hooks out of the shared existing-family branch helper.
- Added owner-registry C++ coverage and representative route-path owner matching for elementwise/compare-select, base memory movement, standalone reduction, and non-route-entry reduction.
- Added RVV plugin spec contracts for selected-body realization owner registry signatures, validation/error behavior, and tests.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Representative `tcrv-opt` / `tcrv-translate` paths for pre-realized compare/select, pre-realized strided-load/unit-store, and explicit selected-body add.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind cmp_select --op-kind strided_load_unit_store --runtime-count 0 --runtime-count 7 --runtime-count 23 --stride-bytes 4 ...`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --runtime-count 0 --runtime-count 7 --runtime-count 23 ...`
- [OK] Negative direct route-entry dry-run for `reduce_add` failed closed before bundle generation.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.
- [OK] `git diff --check`
- [OK] Changed-line authority scan over touched C++/test files found no new legacy i32/source-front-door/descriptor/ABI/artifact/common-EmitC route-authority additions.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 208: Stage2 RVV selected-body realization closure

**Date**: 2026-05-25
**Task**: Stage2 RVV selected-body realization closure
**Branch**: `main`

### Summary

Implemented plugin-local direct pre-realized scalar_broadcast_macc_add realization through RVV route-entry, with fail-closed verifier coverage, generated-bundle evidence, ssh rvv correctness, and archived Trellis task.

### Main Changes

- Implemented bounded TypedMAccPreRealizedBodyOp scalar_broadcast_macc_add support with RHS scalar role/type validation.
- Realized pre-realized scalar_broadcast_macc_add inside RVVSelectedBodyRealization.cpp into setvl/with_vl/load/splat/load/macc/store before provider route construction.
- Extended generated-bundle ABI evidence tooling and lit coverage for direct pre-realized route-entry mode.
- Verified with focused lit, py_compile, git diff --check, full check-tianchenrv, and ssh rvv counts 0/7/16/23 with rhs_scalar -37/91.


### Git Commits

| Hash | Message |
|------|---------|
| `same-commit-as-session-entry` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 209: Stage2 RVV selected exec-envelope ABI closure

**Date**: 2026-05-25
**Task**: Stage2 RVV selected exec-envelope ABI closure
**Branch**: `main`

### Summary

Closed the bounded scalar_broadcast_macc_add selected exec-envelope ABI path by linking tcrv_rvv runtime ABI values to same-kernel tcrv.exec ABI declarations, mirroring those bindings through provider metadata/header evidence, and proving ssh rvv correctness for explicit and direct pre-realized selected-body routes.

### Main Changes

- Added optional `exec_binding` on `tcrv_rvv.runtime_abi_value` and verifier checks that same-kernel `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` declarations match role, C type/name, ownership, purpose, and access.
- Added opt-in `tcrv_rvv.require_exec_abi_bindings` route-provider validation and mirror-only `tcrv_rvv.exec_abi_bindings` metadata through emission plan and RVV target header evidence.
- Updated explicit and pre-realized scalar-broadcast MAcc fixtures plus generated-bundle dry-run FileCheck expectations.
- Added fail-closed tests for wrong exec op kind, wrong ABI role, wrong C name/type, wrong ownership, and missing required exec binding.
- Verified build, focused FileCheck paths, generated-bundle dry-runs, real `ssh rvv` runs for n=7,16,23 with rhs=-37,91 on both explicit and direct pre-realized selected-body routes, changed-line authority scan, `git diff --check`, and `check-tianchenrv` 376/376.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 210: Stage2 RVV target capability selected-route gating

**Date**: 2026-05-25
**Task**: Stage2 RVV target capability selected-route gating
**Branch**: `main`

### Summary

Closed scalar_broadcast_macc_add selected-route gating through explicit tcrv.exec target/capability facts, provider legality mirrors, focused fail-closed tests, check-tianchenrv, and ssh rvv evidence.

### Main Changes

- Added RVV plugin-local selected target-capability facts collection from
  `tcrv.exec.variant requires` and `TargetCapabilitySet`.
- Gated route construction on a single available RVV provider, correct exact
  `rvv` kind, non-ambiguous selected ownership, and optional
  SEW/LMUL/policy compatibility with typed `tcrv_rvv` config facts.
- Mirrored verified target capability provider/legality facts through provider
  route metadata, target artifact validation, generated headers, dry-run
  evidence, and real `ssh rvv` bundle evidence.
- Added fail-closed verifier/provider tests and updated the bounded
  `scalar_broadcast_macc_add` explicit/pre-realized fixtures.

### Git Commits

Pending final session commit in this turn.

### Testing

- [OK] `git diff --check`
- [OK] Focused C++ build and tests:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`
- [OK] Focused 6-test lit set for target-capability gating, artifact/header
  mirrors, and generated-bundle dry-runs.
- [OK] `ssh rvv` generated-bundle ABI/e2e for
  `scalar_broadcast_macc_add`, counts `7,16,23`, rhs scalars `-37,91`.
- [OK] `cmake --build build --target check-tianchenrv -j2`:
  378/378 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 213: Stage2 RVV standalone reduction route-family boundary reuse

**Date**: 2026-05-25
**Task**: Stage2 RVV standalone reduction route-family boundary reuse
**Branch**: `main`

### Summary

Closed direct pre-realized route-entry reuse for the existing executable
`standalone_reduce_add` route family. The production route-entry bridge now
realizes the pre-realized standalone reduction body before route facts are
collected, then the existing standalone reduction family plan, math operand
binding facts, migrated statement plan, provider route, common EmitC, target
mirrors, and generated-bundle evidence carry the route.

### Main Changes

- Added standalone reduction to the RVV route-entry selected-body realization
  allowlist, gated on structural standalone-reduction op kind and memory form.
- Added a C++ production route-path case proving direct route-entry realization
  reaches `rvv-standalone-reduction-route-family-plan.v1`, while ordinary
  pre-realized `reduce` remains fail-closed.
- Enabled direct pre-realized generated-bundle evidence for
  `standalone_reduce_add` and exposed the standalone reduction statement-plan
  family, pre-loop callees, loop callees, seed source, operand order, store
  pointer, and reduction store VL.
- Updated the RVV plugin spec so standalone reduction is a listed route-entry
  realization family only with matching RVV-owned family facts, operand
  binding, migrated statement plan, and tests.

### Git Commits

Pending final session commit in this turn.

### Testing

- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] Direct pre-realized generated-bundle dry-run for
  `standalone_reduce_add`, counts `7,16,23`
- [OK] Focused lit/FileCheck:
  `rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run`
- [OK] `ssh rvv` generated-bundle ABI/e2e for
  `standalone_reduce_add`, counts `7,16,23`, seeds `-11,17`
- [OK] Bounded authority scans over touched source/test paths and generated
  evidence/harnesses
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 211: Stage2 RVV base memory-movement route-family boundary

**Date**: 2026-05-25
**Task**: Stage2 RVV base memory-movement route-family boundary
**Branch**: `main`

### Summary

Closed the base memory movement route-family evidence boundary for the strided executable fixture while preserving RVV plugin-owned provider planning and neutral EmitC/artifact consumption.

### Main Changes

- Created and archived Trellis task `stage2-rvv-base-memory-movement-route-family-boundary` from the Direction Brief.
- Confirmed the production RVV provider/family path already owns base memory family plans and migrated statement plans for the six route-supported base memory forms at C++ route-planning/provider level.
- Added generated-bundle `base_memory_movement_boundary` evidence for base memory operations, including structural authority, mirror-only route metadata, ordered statement callees, pointer roles, artifact paths, and runtime-count non-authority labeling.
- Strengthened explicit and pre-realized `strided_load_unit_store` generated-bundle dry-run checks and the pre-realized selected-body target artifact mirror checks.
- Updated the RVV plugin spec with durable generated-bundle and ssh-rvv evidence requirements for executable base memory movement routes.
- Checks passed: py_compile, explicit/pre-realized dry-runs, focused lit/FileCheck tests, RVV plugin smoke binary, real ssh rvv correctness for counts 7,16,23 and stride bytes 4,8,12, git diff --check, and full check-tianchenrv 379/379.


### Git Commits

Pending final session commit in this turn.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] Explicit and pre-realized `strided_load_unit_store` generated-bundle dry-runs.
- [OK] Focused script and target lit/FileCheck tests.
- [OK] RVV plugin smoke binary and real `ssh rvv` correctness for counts `7,16,23` and stride bytes `4,8,12`.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` with 379/379 tests passing.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 212: Stage2 RVV base memory-movement production family ownership closure

**Date**: 2026-05-25
**Task**: Stage2 RVV base memory-movement production family ownership closure
**Branch**: `main`

### Summary

Closed a production C++ base-memory provider-plan boundary that joins verified family plan, same-analysis materialization/binding facts, validated mirrors, and ordered statement plan before migrated route construction; focused RVV plugin test, git diff --check, authority scan, and check-tianchenrv 379/379 passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 213: Stage2 RVV scalar MAcc route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV scalar MAcc route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated scalar_broadcast_macc_add with the shared RVV route-control provider-plan boundary before scalar MAcc statement planning, added focused C++ positive/fail-closed coverage, updated the RVV plugin spec, and verified focused plus full check-tianchenrv.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 215: Stage2 RVV elementwise arithmetic route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV elementwise arithmetic route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated ordinary Add/Sub/Mul elementwise arithmetic with the shared RVV route-control provider-plan boundary before statement construction, added focused fail-closed C++ coverage, updated the RVV plugin spec, and verified focused plus full check-tianchenrv.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 216: Stage2 RVV scalar-broadcast route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV scalar-broadcast route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated scalar-broadcast elementwise Add/Sub/Mul with the shared RVV route-control provider-plan boundary.

### Main Changes

- Added scalar-broadcast elementwise Add/Sub/Mul as RVV route-control consumers through `controlsScalarBroadcastElementwise`.
- Required scalar-broadcast route-control construction to use same-analysis scalar-broadcast route-family and materialization facts before exposing AVL/VL, policy, typed config, selected target capability, runtime ABI, and mirror facts.
- Required the scalar-broadcast elementwise statement-plan boundary to consume the RVV-owned route-control provider plan before setvl/load/splat/binary/store statement construction.
- Extended focused RVV plugin tests with positive add/sub/mul control-plan evidence and negative fail-closed cases for stale runtime AVL role, policy mismatch, unsupported selected capability, stale same-analysis facts, runtime ABI mirror mismatch, and stale scalar operand binding.
- Updated the RVV plugin spec to list scalar-broadcast elementwise arithmetic as an explicit route-control provider-plan consumer.
- Validation passed: task validate, git diff --check, RVV extension plugin test build/run, tcrv-opt/tcrv-translate build, focused scalar-broadcast lit/FileCheck filter, bounded authority scan, and check-tianchenrv.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 217: Stage2 RVV plain compare-select route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV plain compare-select route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated the existing plain cmp_select route with the shared RVV route-control provider-plan boundary before compare/select statement planning, with focused fail-closed coverage, generated-bundle evidence, ssh rvv proof, spec update, and full check-tianchenrv.

### Main Changes

- Added `controlsPlainCompareSelect` to `RVVSelectedBodyRouteControlProviderPlan` and made `CmpSelect` an explicit route-control provider-plan consumer.
- Required route-control construction for plain compare/select to validate same-analysis plain compare/select family materialization facts, typed config facts, selected target capability facts, runtime AVL/VL control, runtime ABI order, tail/mask policy, and mirror consistency.
- Required `getRVVSelectedBodyCompareSelectRouteStatementPlan(...)` to consume the RVV-owned route-control plan before building the plain compare/select setvl/load/compare/select/store statement plan.
- Kept computed-mask, runtime-scalar, and dual compare/select subfamilies outside this migration.
- Extended `RVVExtensionPluginTest.cpp` with positive plain compare/select control-plan consumption and fail-closed cases for missing family facts, stale runtime AVL role, policy mismatch, unsupported selected capability, stale same-analysis materialization, runtime ABI mirror mismatch, and stale operand binding.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to list plain compare/select as a route-control consumer and document the statement-plan control dependency.
- Validation passed: task context validation, `git diff --check`, RVV extension plugin test build/run, `tcrv-opt`/`tcrv-translate` build, focused lit filter `cmp-select|compare-select` (11 selected tests), pre-realized `cmp_select` generated-bundle dry-run, real `ssh rvv` `cmp_select` counts `7,16,23`, added-line authority scan, and `check-tianchenrv` 379/379.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 218: Stage2 RVV computed-mask select route-control provider plan

**Date**: 2026-05-25
**Task**: Stage2 RVV computed-mask select route-control provider plan
**Branch**: `main`

### Summary

Computed-mask select now consumes the shared RVV route-control provider plan before compare/select statement-plan construction; focused/full checks passed.

### Main Changes

- Added `controlsComputedMaskSelect` and route-control provider-plan consumption for vector, runtime-scalar, and dual runtime-scalar computed-mask select variants.
- Compare/select statement planning now fail-closes before route construction when computed-mask select lacks same-analysis route-control, materialization, runtime AVL/VL, policy, capability, producer-source, or operand-binding facts.
- Updated the RVV plugin spec to list computed-mask select as an adopted route-control consumer.
- Verification: task validate; `tianchenrv-rvv-extension-plugin-test`; focused compare/select lit 16/16; generated-bundle dry-run for pre-realized `computed_mask_select`; `git diff --check`; `check-tianchenrv` 379/379.


### Git Commits

| Hash | Message |
|------|---------|
| `same-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 219: Stage2 RVV computed-mask memory route-control

**Date**: 2026-05-25
**Task**: Stage2 RVV computed-mask memory route-control
**Branch**: `main`

### Summary

Integrated non-segment computed-mask memory route planning with the shared RVV route-control provider plan, added fail-closed provider tests, updated RVV plugin spec, ran focused RVV tests, generated-bundle dry-run, and check-tianchenrv.

### Main Changes

- Added computed-mask memory consumption of `RVVSelectedBodyRouteControlProviderPlan`
  for the existing non-segment runtime-scalar, unit, strided, indexed gather,
  and indexed scatter computed-mask memory routes.
- Required same-analysis computed-mask memory family/materialization facts,
  route-control ownership, memory operand binding, runtime ABI mirrors,
  mask-producer facts, memory-form facts, typed config, selected capability,
  and policy facts before statement-plan construction.
- Added focused provider positive/negative coverage and updated the RVV plugin
  spec for the computed-mask memory route-control boundary.

### Git Commits

- same-commit

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] computed-mask memory generated-bundle dry-run under
  `artifacts/tmp/stage2_rvv_computed_mask_memory_route_control/pre-realized-computed-mask-memory-route-control-dry`
- [OK] focused lit filter over `Target/RVV` and `Scripts`: 53 selected tests
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379 tests

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 222: Stage2 RVV runtime scalar splat-store route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV runtime scalar splat-store route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated runtime_i32_splat_store with the shared RVV route-control provider plan, added focused fail-closed C++ coverage, updated RVV plugin spec, and verified focused/full checks.

### Main Changes

- Added `controlsRuntimeScalarSplatStore` to the shared
  `RVVSelectedBodyRouteControlProviderPlan`.
- Made existing `runtime_i32_splat_store` route-control construction require
  same-analysis runtime scalar splat-store family/materialization facts before
  exposing runtime AVL/VL, typed config, selected capability, scalar splat,
  store, vector result, policy, and mirror facts.
- Required the runtime scalar splat-store provider path to consume the
  route-control plan before fallback splat/store statement construction.
- Added focused C++ positive/fail-closed coverage and updated the RVV plugin
  spec to list runtime scalar splat-store as a route-control consumer.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-runtime-scalar-splat-store-route-control`
- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] focused lit filter `(runtime-i32-splat-store|runtime-scalar-splat-store)`: 5 selected tests
- [OK] generated-bundle dry-run for explicit and pre-realized `runtime_i32_splat_store`
  under `artifacts/tmp/stage2_runtime_scalar_splat_store_route_control`
- [OK] bounded added-line authority scan; only a negative stale intrinsic mirror
  test added `__riscv_vle32_v_i32m1`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379 tests

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 224: Stage2 RVV contraction route-control provider-plan integration

**Date**: 2026-05-25
**Task**: Stage2 RVV contraction route-control provider-plan integration
**Branch**: `main`

### Summary

Integrated existing RVV contraction direct-provider routes with the shared route-control provider plan before provider construction; added focused fail-closed C++ coverage, updated RVV plugin spec, and verified focused plus full check-tianchenrv.

### Main Changes

- Added `controlsContraction` to `RVVSelectedBodyRouteControlProviderPlan`.
- Made the five active contraction routes consume the shared route-control
  provider plan before direct provider statement construction:
  `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Route-control construction now validates same-analysis contraction family
  and materialization facts, typed config, selected capability, runtime
  AVL/VL, policy, runtime ABI, widening MAcc/dot classification,
  accumulator/result layout, optional strided-input facts, optional
  computed-mask facts, and materialization leaves.
- Direct provider construction now requires the route-control plan and
  RVV-owned math operand-binding facts before emitting contraction statements.
- Updated the RVV plugin spec to record contraction as a route-control
  consumer while preserving the direct-provider path and avoiding a
  wrapper-only statement-plan requirement.

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-contraction-route-control-provider-plan`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded authority scan over touched planning/provider/test/spec files
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379 tests

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 225: Stage2 RVV route-control owner registry consolidation

**Date**: 2026-05-25
**Task**: Stage2 RVV route-control owner registry consolidation
**Branch**: `main`

### Summary

Consolidated adopted RVV route-control consumers behind a plugin-local owner registry, preserved route-control same-analysis and capability checks, added focused C++ registry coverage, updated RVV spec, and verified generated-bundle dry-runs plus check-tianchenrv.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 226: Stage2 RVV selected-body realization owner registry

**Date**: 2026-05-25
**Task**: Stage2 RVV selected-body realization owner registry
**Branch**: `main`

### Summary

Added an RVV selected-body realization owner registry, routed pre-realized body discovery/route-entry/public realization through owner selection, preserved existing family validators/materializers, added C++ owner/path coverage, documented the registry contract, and verified focused/full checks plus generated-bundle dry-runs.

### Main Changes

- Added `RVVSelectedBodyRealizationOwner` plus public owner registry lookup APIs.
- Routed pre-realized body discovery, route-entry realization selection, and the public selected-body realization entrypoint through owner selection.
- Preserved existing family validators/materializers while documenting the remaining continuation point: split non-elementwise owner hooks out of the shared existing-family branch helper.
- Added owner-registry C++ coverage and representative route-path owner matching for elementwise/compare-select, base memory movement, standalone reduction, and non-route-entry reduction.
- Added RVV plugin spec contracts for selected-body realization owner registry signatures, validation/error behavior, and tests.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Representative `tcrv-opt` / `tcrv-translate` paths for pre-realized compare/select, pre-realized strided-load/unit-store, and explicit selected-body add.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind cmp_select --op-kind strided_load_unit_store --runtime-count 0 --runtime-count 7 --runtime-count 23 --stride-bytes 4 ...`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --runtime-count 0 --runtime-count 7 --runtime-count 23 ...`
- [OK] Negative direct route-entry dry-run for `reduce_add` failed closed before bundle generation.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 379/379 passed.
- [OK] `git diff --check`
- [OK] Changed-line authority scan over touched C++/test files found no new legacy i32/source-front-door/descriptor/ABI/artifact/common-EmitC route-authority additions.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 227: Stage2 RVV deferred selected-body realization owner-local hooks

**Date**: 2026-05-25
**Task**: Stage2 RVV deferred selected-body realization owner-local hooks
**Branch**: `main`

### Summary

Migrated the remaining deferred RVV selected-body realization owners to registry-selected owner-local hooks, preserved route-control/provider behavior, added focused owner-local coverage, and verified full check-tianchenrv.

### Main Changes

- Migrated the remaining deferred selected-body realization owners to distinct owner-local registry hooks: runtime scalar splat-store, runtime scalar computed-mask store/load-store, reduction, computed-mask MAcc, contraction, widening conversion, computed-mask memory, and segment2 memory.
- Removed the active shared existing-family realization helper from registry authority. The remaining private materialization branch takes the already selected body from an owner-local hook and does not rediscover or classify the owner/body.
- Preserved route-entry ownership for elementwise/compare-select, standalone reduction, MAcc, and base memory movement while keeping non-route-entry families fail-closed at the direct route-entry bridge.
- Strengthened C++ registry coverage so all thirteen selected-body realization owners use distinct hook pointers and route-entry eligibility remains owner-scoped.
- Added runtime scalar splat-store owner-local negative coverage for null body dispatch, non-owned body dispatch, and wrong op_kind, plus successful owner-local realization to setvl/with_vl.
- Self-repaired owner-local wrapper null-body handling so `llvm::isa`-based owner predicates are not called with null operations.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-25-stage2-rvv-deferred-selected-body-owner-local-hooks`
- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `scripts/rvv_generated_bundle_abi_e2e.py --dry-run` for pre-realized computed-mask MAcc, pre-realized runtime scalar splat-store, and explicit runtime scalar splat-store selected body.
- [OK] Bounded added-line authority scan found no new source-front-door, descriptor, common-EmitC, artifact/script, route-id, ABI-string, or legacy-i32 authority.
- [OK] `cmake --build build --target check-tianchenrv`: 379/379 passed.

### Runtime Evidence

No `ssh rvv` rerun was needed because emitted RVV body semantics, runtime ABI, dispatch/fallback behavior, statement order, and target artifact semantics were preserved; this round changed selected-body realization ownership and fail-closed dispatch only.


### Git Commits

| Hash | Message |
|------|---------|
| `pending-final-session-commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete
