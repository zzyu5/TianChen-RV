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
| `pending-final-session-commit` | (see git log) |

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
