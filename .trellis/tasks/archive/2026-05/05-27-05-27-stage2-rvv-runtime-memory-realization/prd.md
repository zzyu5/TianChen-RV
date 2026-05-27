# Stage2 RVV Selected-Body Realization Producer For Runtime Memory/Control Path

## Goal

Expand the RVV plugin-local selected-body realization producer path from the
completed `computed_mask_select` proof to the independent
`runtime_i32_splat_store` runtime memory/control consumer.

The bounded production path for this round is:

```text
selected tcrv.exec RVV variant with runtime_param / mem_window-like ABI bindings
  -> typed pre-realized tcrv_rvv runtime_i32_splat_store body
  -> RVV plugin-local selected-body realization owner registry
  -> realized typed tcrv_rvv setvl / with_vl / splat / store body
  -> existing runtime AVL/VL route-control provider facts
  -> runtime scalar splat-store route-family and migrated statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact / harness
  -> ssh rvv correctness evidence for executable claims
```

This task must prove that runtime `n`/AVL, VL placement, scalar binding,
output binding, dtype/config/policy, provider route facts, and artifact ABI
order survive realization as typed structure and provider-owned facts. They
must not be inferred from route ids, scripts, artifact names, ABI strings,
mirror metadata, exact intrinsic spellings, common EmitC, direct route-entry
shortcuts, or legacy i32 helper authority.

## Direction Source

- Direction title: `Stage2 RVV selected-body realization producer for runtime memory/control path`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `4997baa5 rvv: add selected-body realization producer path`.
- `.trellis/.current-task` was absent, so this Trellis task was created from
  the supplied Hermes Direction Brief.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows for implementation.

## What I Already Know

- The current RVV-first authority chain is `tcrv.exec` envelope -> selected
  RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality,
  selected-body realization, and route provider -> `TCRVEmitCLowerableRoute`
  -> common EmitC -> target artifact -> `ssh rvv` evidence when runtime or
  correctness is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV compute, dtype, SEW/LMUL, policy, schedule,
  runtime AVL, scalar binding, store binding, route support, intrinsic
  spelling, or target artifact authority.
- The current code already has a `runtime scalar splat-store` selected-body
  realization owner. That owner consumes
  `tcrv_rvv.typed_runtime_scalar_splat_store_pre_realized_body` and produces
  realized `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, `tcrv_rvv.splat`, and
  `tcrv_rvv.store`.
- The runtime scalar splat-store owner is intentionally not direct route-entry
  eligible. Its route-entry predicate is null, so direct/pre-realized
  route-entry cannot be the positive authority for this consumer.
- Existing route planning already derives and verifies
  `RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan`, runtime AVL/VL
  route-control provider facts, residual operand bindings for
  `rhs_scalar,out,n`, and migrated statement-plan construction for this route.
- Existing target export validation already rebuilds the provider route and
  validates runtime scalar splat-store headers, type mappings, ABI order,
  setvl/full-chunk VL, runtime loop bounds, loop setvl remaining AVL,
  scalar splat, store operands, selected typed RVV source provenance, and
  stale elementwise/conversion route facts.
- Existing generated-bundle tooling has a `runtime_i32_splat_store` expectation
  and a pre-realized fixture path, but there is no focused script dry-run test
  for this consumer analogous to the completed selected-body producer proof for
  `computed_mask_select`.
- Existing C++ production route-path tests prove selected-boundary producer
  movement for `computed_mask_select`; they do not yet include
  `runtime_i32_splat_store` as the independent runtime/memory/control producer
  case.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake /
   lit / FileCheck. Python changes are allowed only for generated-bundle
   tooling and evidence.
2. Use the RVV plugin-local selected-body realization owner registry as the
   production producer boundary for `runtime_i32_splat_store`.
3. Do not make `runtime_i32_splat_store` direct route-entry eligible. If a
   direct/pre-realized shortcut would bypass the producer, it must remain
   unsupported or be demoted with a targeted failure.
4. The realized body must structurally contain `setvl`, `with_vl`, runtime
   scalar `splat`, and `store`, with runtime `n`/AVL and `rhs_scalar,out,n`
   bindings consumed from typed ABI values.
5. Provider route construction must consume the realized body through the
   existing runtime scalar splat-store route-family plan, residual operand
   binding facts, runtime AVL/VL route-control provider plan, and migrated
   statement-plan owner before building `TCRVEmitCLowerableRoute`.
6. Target artifact validation must continue to consume rebuilt provider route
   facts for `runtime_i32_splat_store` before accepting header/artifact claims.
7. Unsupported or inconsistent input must fail closed before provider/common
   route construction where the current test surface exposes it: missing or
   wrong runtime binding, wrong scalar binding, wrong output binding,
   dtype/config/policy mismatch, wrong runtime AVL/VL relation, wrong setvl
   placement, wrong loop relation, missing capability, stale route id, stale
   mirror metadata, direct-route-entry-only authority, script/artifact-name
   authority, exact-intrinsic-as-authority, and common-EmitC semantic choice.
8. Do not add broad route coverage, dtype/LMUL clone batches, high-level
   Linalg/frontend lowering, one-intrinsic wrapper dialects, new route-family
   owners, descriptor/source-front-door/direct-C routes, dashboards, reports,
   or unrelated follow-on polish.
9. Do not weaken completed `computed_mask_select`, mask/tail policy,
   VL/control runtime-AVL, conversion dtype-policy, compare/select,
   computed-mask memory, standalone reduction/accumulation, MAcc, segment2,
   scalar-broadcast, contraction, or base-memory route paths.

## Acceptance Criteria

- [x] A focused production code/test diff proves `runtime_i32_splat_store`
      artifact generation flows through RVV selected-body realization before
      provider facts and `TCRVEmitCLowerableRoute` construction.
- [x] The C++ route-path test covers `runtime_i32_splat_store` as a
      selected-boundary producer-only case, not a direct route-entry success
      case.
- [x] The realized body check proves the pre-realized runtime scalar
      splat-store body is erased and replaced with exactly the expected
      `setvl` / `with_vl` / `splat` / `store` structure, with no load or binary
      compute.
- [x] Provider route description checks prove the realized body reaches the
      `rvv-runtime-scalar-splat-store-route-family-plan.v1` provider plan and
      builds one migrated statement-plan loop.
- [x] Target artifact ABI validation remains provider-route rebuilt and
      rejects stale or missing runtime scalar splat-store facts before
      accepting artifact/header claims.
- [x] Generated-bundle dry-run covers pre-realized
      `runtime_i32_splat_store` through the selected lowering-boundary path,
      including counts `0`, `1` or small, exact, tail, and stress cases plus a
      runtime scalar value.
- [x] Direct pre-realized route-entry remains unsupported for
      `runtime_i32_splat_store`, with focused evidence if touched.
- [x] Focused non-regression covers completed adjacent selected-body producer
      and runtime-control paths, especially `computed_mask_select`,
      scalar-broadcast, conversion dtype-policy, mask/tail policy, segment2,
      MAcc, standalone reduction/accumulation, contraction, and base memory.
- [x] A bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] Focused builds/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis task status, context, journal, and archive state are truthful,
      and one coherent commit is created if all acceptance criteria are met.

## Technical Approach

Use `runtime_i32_splat_store` as the representative independent runtime
memory/control consumer. The implementation should avoid introducing a new
route family or direct route-entry positive path. The likely production move is
to extend the selected-body producer route-path C++ coverage so this consumer
must pass through `materializeSelectedLoweringBoundary`, then through the
existing runtime scalar splat-store provider and migrated statement-plan owner.

The generated-bundle consumer should be added as a focused pre-realized
selected-body dry-run lit test. It must assert the selected lowering-boundary
materializer path, `selected_body_realization_producer`, runtime ABI order,
runtime-control plan, route operand binding plan, runtime scalar splat-store
route-family plan, typed compute op `tcrv_rvv.splat`, no descriptor/direct-C/
source-export/source-front-door, and no legacy `tcrv_rvv.i32_` authority.

If inspection reveals a live direct/pre-realized route-entry shortcut for
`runtime_i32_splat_store`, demote it instead of wrapping it. Current evidence
shows the owner registry has null route-entry eligibility for this family, so
the expected implementation is to preserve that fail-closed boundary and add
the missing production consumer/evidence coverage.

## Validation Plan

1. Validate task context after PRD/context setup.
2. Inspect and, if needed, modify:
   `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
   `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
   `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
   `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
   `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
   `scripts/rvv_generated_bundle_abi_e2e.py`,
   `test/Plugin/RVVExtensionPluginTest.cpp`, and focused
   `test/Scripts` / `test/Target/RVV` consumers.
3. Add the bounded `runtime_i32_splat_store` selected-boundary producer case
   to C++ production route-path coverage.
4. Add the focused generated-bundle dry-run lit consumer for pre-realized
   `runtime_i32_splat_store`.
5. Run focused RVV plugin build/test.
6. Run focused lit/FileCheck tests for the new generated-bundle consumer and
   directly related runtime splat-store target fixture.
7. Run generated-bundle dry-run manually for
   `runtime_i32_splat_store` with counts including `0`, small, exact, tail,
   and stress plus runtime scalar values.
8. Run real `ssh rvv` generated-bundle execution for
   `runtime_i32_splat_store` over those counts if the environment is available.
9. Run focused non-regressions for `computed_mask_select` and adjacent mature
   selected-body paths.
10. Run bounded authority scan, `git diff --check`, Trellis validation, and
    `check-tianchenrv` or record an exact blocker.

## Out Of Scope

- Broad Stage2 route coverage expansion.
- New dtype/LMUL clone sets.
- High-level Linalg, Vector, StableHLO, source-front-door, Toy,
  TensorExtLite, Template, IME, Offload, or future plugin routes.
- One-intrinsic wrappers or new dtype-prefixed `tcrv_rvv.i32_*` helper ops.
- New route-family owners or unrelated polish for mask/tail, VL/control,
  conversion, reduction, MAcc, segment2, base memory, contraction,
  gather/scatter, or compare/select.
- Descriptor-driven compute, direct-C/source-export routes, or legacy
  `RVVI32M1` / `rvv-i32m1` executable compatibility.
- Dashboard, report-only, prompt-only, helper-only, or evidence-only work as
  the main achievement.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/index.md`.
- Predecessor archived tasks read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-selected-body-realization-producer/`
  and
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-vl-control-runtime-avl-boundary-owner/`.
- Initial code surfaces inspected:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-i32-splat-store.mlir`,
  and related `test/Scripts` generated-bundle consumers.

## Validation Evidence

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed and now covers
  `rvv_pre_route_runtime_i32_splat_store` as a selected-boundary producer-only
  case. The check verifies one `tcrv_rvv.splat`, one `tcrv_rvv.store`, no
  `tcrv_rvv.load`, no `tcrv_rvv.binary`, the
  `rvv-runtime-scalar-splat-store-route-family-plan.v1` route description, and
  one provider-built statement-plan loop.
- Focused lit from `build/test` passed for
  `Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-i32-splat-store-dry-run.test`
  and
  `Target/RVV/pre-realized-selected-body-artifact-runtime-i32-splat-store.mlir`.
- Generated-bundle dry-run passed for pre-realized
  `runtime_i32_splat_store` with runtime counts `0,1,16,23,257` and
  `rhs_scalar` values `-37,19`.
- Direct pre-realized route-entry negative evidence failed closed before bundle
  generation for `runtime_i32_splat_store`:
  `--direct-pre-realized-route-entry is bounded ... got ['runtime_i32_splat_store']`.
- Real `ssh rvv` generated-bundle evidence passed for `runtime_i32_splat_store`
  with counts `0,1,16,23,257` and `rhs_scalar` values `-37,19`; every case
  reported `tail_preserved`, then
  `PASS op=runtime_i32_splat_store counts=0,1,16,23,257 rhs_scalars=-37,19`.
- Bounded pre-realized selected-body non-regression dry-run passed for
  `computed_mask_select`, `masked_add`, `cmp_select`, `widen_i32_to_i64`,
  `computed_masked_unit_load_store`, `standalone_reduce_add`,
  `computed_masked_macc_add`, `segment2_interleave_unit_load`,
  `scalar_broadcast_add`, `widening_dot_reduce_add`,
  `strided_load_unit_store`, and `runtime_i32_splat_store`.
- Added-hunk authority scan found no new positive descriptor, source-front-door,
  direct-C, route-id, artifact-name, script-derived, common-EmitC,
  exact-intrinsic, direct-route-entry-only, or legacy-i32 authority.
- `git diff --check` passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-27-05-27-stage2-rvv-runtime-memory-realization` passed.
- `check-tianchenrv` initially failed because two full lit checks were launched
  concurrently and raced on shared generated `build/*/Output` directories. After
  clearing those generated output directories and rerunning a single instance,
  `cmake --build build --target check-tianchenrv -j2` passed with `391` tests.

## Open Questions

None blocking. Repository evidence already identifies the bounded production
move: make `runtime_i32_splat_store` a selected-boundary producer proof case
and add generated-bundle dry-run evidence without creating a direct
route-entry success path.

## Definition Of Done

`runtime_i32_splat_store` is proven to flow from a selected pre-realized
`tcrv.exec` RVV variant through the RVV selected-body realization producer into
realized typed `tcrv_rvv` structure, through the existing runtime AVL/VL
route-control and provider route facts, into generated artifact/harness
evidence and `ssh rvv` correctness evidence where available. Task state is
truthful, scans/checks are recorded, and one coherent commit records the
completed round.
