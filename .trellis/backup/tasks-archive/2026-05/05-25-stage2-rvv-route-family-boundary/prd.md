# Stage2 RVV route-family module boundary closure

## Goal

Close one explicit RVV plugin-local route-family boundary for the already
executable `scalar_broadcast_macc_add` path. The RVV provider must consume a
validated family plan/owner for legality, mirrors, materialization facts,
operand binding, and ordered statement planning before it builds the
`TCRVEmitCLowerableRoute`, while common EmitC and target artifact code remain
neutral mirror consumers.

## What I Already Know

* Current HEAD is `865edb12 rvv: close selected dispatch fallback envelope`.
* The working tree was clean at task creation.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes Direction Brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-dispatch-fallback-envelope/prd.md`
  proved selected dispatch/fallback envelope linkage, target capability mirrors,
  generated-bundle evidence, and `ssh rvv` correctness for
  `scalar_broadcast_macc_add`.
* The spec requires the current RVV authority chain to remain:
  selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness is claimed.
* `tcrv.exec` owns ABI/runtime/dispatch/fallback envelope facts only. It must
  not own RVV compute, dtype, schedule, intrinsic spelling, or route authority.
* Common EmitC and target artifact export may consume provider-built route
  payloads and mirror explicit provider facts, but must not infer RVV semantics
  from route ids, metadata, artifact names, ABI strings, descriptors, manifests,
  source-front-door markers, or harness constants.
* Current code already has a top-level selected-body route-family provider owner
  registry and several cluster/standalone family verifiers. However
  `scalar_broadcast_macc_add` still lacks its own validated route-family plan
  entry in that registry.
* Current code has `kRVVScalarBroadcastMAcc*` constants and a
  `RVVSelectedBodyPlainMAccRouteStatementPlan` that emits the ordered route, but
  the scalar-broadcast MAcc target leaf/profile/mirror/layout fields are still
  populated ad hoc in central route analysis instead of being applied from an
  explicit scalar-broadcast MAcc family plan.

## Requirements

1. Add one coherent RVV plugin-local route-family plan/owner boundary for
   `scalar_broadcast_macc_add`.
2. The family plan must be derived from structural typed body/config/runtime
   facts: operation kind, memory form, `setvl`/`with_vl`, loads, scalar splat,
   MAcc compute op, store, SEW/LMUL/policy, ABI roles, target leaves, capability
   and selected dispatch facts already gathered upstream.
3. The production top-level provider owner registry must consume this family
   owner and fail closed when the route is a consumer without the validated plan,
   or when a non-consumer carries a stale scalar-broadcast MAcc plan.
4. The family verifier must reject mismatched route description mirrors,
   runtime ABI order/parameters, typed config fields, MAcc layout/result
   contracts, intrinsic/type/header mirrors, and route operand binding closure
   before provider route construction or artifact authority.
5. The provider route construction must consume family-owned materialization
   facts and statement plan data for `scalar_broadcast_macc_add`; it must not
   reconstruct this route from route ids, metadata, ABI strings, artifact names,
   descriptors, common EmitC logic, or harness constants.
6. Target artifact metadata may mirror the scalar-broadcast MAcc route-family
   plan with an explicit `tcrv_rvv.*_route_family_plan` label, and validation
   must remain mirror-only downstream of the provider-built route.
7. Existing explicit and pre-realized `scalar_broadcast_macc_add` executable
   flows must continue to pass.

## Acceptance Criteria

* [x] C++ provider/planning tests prove `scalar_broadcast_macc_add` is a
      top-level route-family provider consumer and that the owner registry
      contains and verifies the scalar-broadcast MAcc owner.
* [x] C++ fail-closed tests cover missing family plan, stale family plan on a
      non-consumer, mismatched family-plan mirror, and at least one missing
      statement-plan dependency before route construction.
* [x] FileCheck evidence for explicit and pre-realized
      `scalar_broadcast_macc_add` still shows the ordered RVV route and explicit
      provider/route-family mirror labels.
* [x] Generated-bundle dry-run evidence still succeeds for the migrated family.
* [x] `ssh rvv` correctness succeeds for the generated migrated family across
      at least three runtime counts if executable behavior is claimed.
* [x] Bounded authority scan over touched RVV planning/provider/target/test/
      fixture/script files finds no new central ad hoc scalar-broadcast MAcc
      authority, metadata/name-derived authority, descriptor/source-front-door
      authority, common-EmitC semantic ownership, or legacy i32 route authority.
* [x] Focused checks for the changed behavior pass.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Definition Of Done

* Task status, PRD, context files, and workspace journal reflect the final
  state truthfully.
* Focused failures caused by this work are repaired and rerun.
* The task is finished and archived when complete.
* One coherent commit is created when the task reaches completion.

## Out Of Scope

* New RVV operation coverage, dtype/LMUL batches, high-level frontend lowering,
  source-front-door positive routes, global autotuning, dashboards, broad smoke
  matrices, scalar fallback compute, IME/Offload/TensorExt work, future plugin
  work, one-op-per-intrinsic wrapper growth, compatibility wrappers for legacy
  i32 authority, or new descriptor-driven computation.
* Moving RVV compute semantics into common EmitC, target artifact plumbing,
  `tcrv.exec`, route ids, metadata fields, manifests, artifact names, ABI
  strings, descriptors, tests, scripts, or harness constants.

## Technical Notes

* Created from the Hermes Direction Brief in the Codex worker prompt on
  2026-05-25.
* Specs read before PRD: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-dispatch-fallback-envelope/prd.md`.
* Initial implementation targets:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, focused RVV plugin tests, and
  scalar-broadcast MAcc generated-bundle FileCheck fixtures.
* Implemented the scalar-broadcast MAcc family plan as
  `RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan`, with a dedicated owner
  in the reduction/accumulation/contraction family registry and top-level
  provider verification coverage.
* The family plan now owns the operation, memory form, runtime AVL/VL control,
  runtime ABI order/parameters, header/type/intrinsic leaves, MAcc layout,
  route mirror payloads, materialization facts, and statement-plan dependency
  for `scalar_broadcast_macc_add`.
* The explicit mirror key is
  `tcrv_rvv.scalar_broadcast_macc_route_family_plan =
  rvv-scalar-broadcast-macc-route-family-plan.v1`. Target support and the
  generated-bundle runner only consume it as mirror evidence after provider
  route construction.
* Focused C++ coverage was added in `test/Plugin/RVVExtensionPluginTest.cpp`
  for owner registry membership/classification, missing plan, stale plan,
  mismatched mirror, missing statement-plan dependency, ordered statement-plan
  construction, and provider route consumption.
* FileCheck fixtures for explicit and pre-realized
  `scalar_broadcast_macc_add` now assert the route-family plan mirror in both
  emission-plan metadata and generated header artifact output.
* Generated-bundle evidence was regenerated with final binaries:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-broadcast-macc-family-dry`,
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-broadcast-macc-family-prerealized-dry`,
  and
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/scalar-broadcast-macc-family-ssh`.
* `ssh rvv` correctness passed for runtime counts `7,16,23` and RHS scalars
  `-37,91`, ending with
  `PASS op=scalar_broadcast_macc_add counts=7,16,23 rhs_scalars=-37,91`.
* Bounded authority scans over touched RVV planning/provider/target/script/test
  files found only existing fail-closed descriptor/direct-C/source-front-door
  guards, negative legacy i32 tests, route-id stale tests, and the new
  family-plan/mirror/test references. No scalar-broadcast MAcc route authority
  remains central ad hoc, metadata-derived, name-derived, descriptor-derived,
  ABI-string-derived, common-EmitC-derived, artifact-derived, or
  harness-derived.
* `.trellis/spec/extension-plugins/rvv-plugin.md` was updated with the durable
  plain/scalar-broadcast MAcc statement-plan and family-plan contract.
* Checks run:
  `ninja -C build tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  explicit and pre-realized FileCheck pipelines for emission-plan/header
  artifact evidence, generated-bundle explicit dry-run, generated-bundle
  pre-realized dry-run, generated-bundle `ssh rvv` run, bounded authority scans,
  `git diff --check`, and `ninja -C build check-tianchenrv`.
* `clang-format` was not available in PATH or under `/usr/lib/llvm-20/bin`;
  the only formatting repair needed was applied manually and verified with
  `git diff --check`.
