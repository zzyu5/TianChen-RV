# Stage2 RVV Selected-Body Realization Producer

## Goal

Make RVV plugin-local selected-body realization an active production producer
for one already-owned route family, rather than keeping generated artifact
evidence tied only to pre-realized/direct route-entry fixtures.

The bounded migrated consumer for this round is `computed_mask_select`. It is
already covered by the compare/select mask route-family owner and the mask/tail
policy route-family owner. This task must move one active artifact/harness
consumer through:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv computed-mask select body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body
  -> existing compare/select and mask/tail route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact/harness
  -> ssh rvv correctness evidence
```

The producer must preserve computation semantics, dtype/config semantics,
parameter roles, variant origin, required capabilities, dispatch/fallback
behavior, runtime `n`/AVL values, VL/control facts, mask provenance,
mask/tail policy, inactive-lane behavior, and provider-owned route facts.

## Direction Source

- Direction title: `Stage2 RVV selected-body realization producer for one owned route family`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `5a1524d6 rvv: add mask tail policy route owner`.
- `.trellis/.current-task` was absent, so this Trellis task was created from
  the supplied Direction Brief.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows for implementation.

## What I Already Know

- The current RVV-first authority chain is `tcrv.exec` envelope -> selected
  RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality,
  selected-body realization, and route provider -> `TCRVEmitCLowerableRoute`
  -> common EmitC -> target artifact -> `ssh rvv` evidence when runtime or
  correctness is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV compute, dtype, schedule, mask policy, tail
  policy, intrinsic spelling, route support, or artifact authority.
- Existing `RVVSelectedBodyRealization` code already contains owner registry
  machinery and elementwise/compare-select realization logic for
  `TypedComputedMaskSelectPreRealizedBodyOp`.
- Existing provider planning already has compare/select and mask/tail owner
  facts for `computed_mask_select`, including route-family plan mirrors,
  mask producer/source/form facts, runtime ABI order, type/header mappings,
  and provider-supported mirrors.
- The previous completed task made mask/tail policy ownership explicit for
  `computed_mask_select` and `computed_masked_unit_load_store`, including
  target artifact ABI validation and generated-bundle/`ssh rvv` evidence.
- The remaining bottleneck is production movement: at least one active
  generated artifact path must be produced from selected-body realization and
  then consumed by the existing route-family owner, not accepted only because a
  direct/pre-realized route-entry fixture or metadata names the route.
- Common EmitC/export must remain neutral. It may consume provider-built route
  payloads and mirrors after route construction, but it must not infer RVV
  realization family, dtype, mask/tail policy, intrinsic spelling, ABI roles,
  or route support from route ids, artifact names, scripts, descriptors,
  source-front-door fixtures, metadata, or exact intrinsic names.

## Requirements

1. Keep the implementation in the MLIR/C++/TableGen/CMake/lit stack. Python is
   allowed only for generated-bundle tooling and evidence.
2. Use the RVV plugin-local selected-body realization owner registry as the
   production producer boundary for the migrated consumer.
3. Migrate one active generated artifact/harness consumer for
   `computed_mask_select` so it is produced from the selected-body realization
   path before route-provider facts are collected.
4. The realized body must be structural typed `tcrv_rvv`: `setvl`/`with_vl`,
   typed loads, compare-produced mask, select, store, runtime ABI value use,
   typed config, policy, and selected capability facts must be visible as body
   or provider facts before `TCRVEmitCLowerableRoute` construction.
5. The existing compare/select and mask/tail route-family owner/provider path
   must consume the realized body exactly as it consumes hand-authored or
   already-realized selected bodies.
6. Target artifact ABI validation must validate the migrated realization-path
   consumer from provider-built route facts. Artifact names, script options,
   emission-plan status, route ids, ABI strings, or mirror metadata must not be
   sufficient for success.
7. Unsupported or inconsistent realization input must fail closed before
   provider/common route construction when the current test surface exposes the
   condition: missing runtime or mem binding, wrong operand mapping, changed
   dtype/config/policy, changed mask provenance, changed VL/AVL relation,
   missing RVV capability, stale route id, stale mirror metadata,
   direct-route-entry-only authority, artifact-name/script-derived authority,
   and common-EmitC semantic invention.
8. Do not add a new route-family owner, dtype/LMUL clone set, high-level
   frontend lowering, one-intrinsic wrapper dialect, new dtype-prefixed
   `tcrv_rvv.i32_*` helper, source-front-door positive route, descriptor
   compute path, dashboard/report task, or broad coverage batch.
9. Do not weaken completed base memory, segment2 update/store/load,
   segment2 deinterleave/interleave, computed-mask MAcc, scalar-broadcast
   MAcc, runtime-scalar MAcc, standalone reduction/accumulation,
   compare/select, computed-mask memory, conversion dtype-policy,
   VL/control runtime-AVL, mask/tail policy, scalar-broadcast, or contraction
   route paths.

## Acceptance Criteria

- [ ] A production code diff makes `computed_mask_select` artifact generation
      flow through RVV selected-body realization before provider facts and
      `TCRVEmitCLowerableRoute` construction.
- [ ] The migrated path consumes and erases the typed pre-realized body, then
      exposes a realized typed `tcrv_rvv.with_vl` body with structural
      `setvl`, compare, mask, select, load/store, runtime ABI, dtype/config,
      policy, and capability facts.
- [ ] Provider route construction for `computed_mask_select` consumes the
      realized body through the existing compare/select and mask/tail owner
      plans; no provider/common code synthesizes realization semantics.
- [ ] Target artifact ABI validation for the migrated consumer rejects stale
      or missing realization/provider facts before accepting generated artifact
      or header claims.
- [ ] Focused C++ or lit tests prove positive selected-body realization
      producer movement and fail-closed diagnostics for missing runtime/mem
      binding, wrong operand mapping, changed dtype/config/policy, changed mask
      provenance, changed VL/AVL relation, missing capability, stale route id
      or stale mirror metadata, direct-route-entry-only authority,
      artifact-name/script-derived authority, and common-EmitC semantic
      invention where those are exposed by existing test hooks.
- [ ] Generated-bundle dry-run covers the migrated realization-path
      `computed_mask_select` consumer.
- [ ] Real `ssh rvv` generated-bundle execution covers `computed_mask_select`
      over counts including `0`, small, exact, tail, and stress cases.
- [ ] Focused non-regression covers completed base memory, segment2
      update/store/load, segment2 deinterleave/interleave, computed-mask MAcc,
      scalar-broadcast MAcc, runtime-scalar MAcc, standalone reduction,
      compare/select, computed-mask memory, conversion dtype-policy,
      VL/control runtime-AVL, mask/tail policy, scalar-broadcast, and
      contraction paths.
- [ ] A bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis task status, context, journal, and archive state are truthful,
      and one coherent commit is created if all acceptance criteria are met.

## Technical Approach

Use `computed_mask_select` as the representative because it sits directly at
the intersection of selected-body realization, compare/select provider facts,
mask/tail policy ownership, target artifact ABI validation, generated bundle
tooling, and real `ssh rvv` correctness evidence.

Implementation should inspect the current direct/pre-realized generated-bundle
and target artifact paths first. If a selected-boundary realization path
already exists, migrate one active consumer or script mode to use it as the
default proof path and demote or constrain the duplicate direct/pre-realized
shortcut for this consumer. If the path is incomplete, add the minimal
production plumbing in RVV selected-body realization and directly consumed
provider/target/script surfaces so the route-family owner consumes realized
typed body facts.

Do not introduce a parallel metadata acceptance path. Provider and target
checks should validate post-realization structural facts and provider-built
mirrors; mirrors remain evidence of provider output, not authority.

## Validation Plan

1. Validate task context after PRD/context setup.
2. Inspect `RVVSelectedBodyRealization`, `RVVEmitCRoutePlanning`,
   `RVVEmitCRouteProvider`, `RVVTargetSupportBundle`, generated-bundle tooling,
   and focused plugin/target tests for `computed_mask_select`.
3. Implement the bounded production migration for the selected-body
   realization producer path.
4. Add or update focused C++/lit tests for positive producer movement and
   fail-closed stale/missing/mismatched authority cases.
5. Run focused RVV plugin/target builds and tests for touched surfaces.
6. Run generated-bundle dry-run for migrated `computed_mask_select`.
7. Run real `ssh rvv` generated-bundle execution for `computed_mask_select`
   over `0`, small, exact, tail, and stress counts.
8. Run focused non-regression dry-runs for completed adjacent route families.
9. Run bounded touched-file authority scan, `git diff --check`, Trellis
   validation, and `check-tianchenrv` or record an exact blocker.

## Out Of Scope

- Broad route coverage batches.
- New dtype/LMUL clone sets.
- High-level Linalg, Vector, StableHLO, source-front-door, Toy,
  TensorExtLite, Template, IME, Offload, or future plugin routes.
- One-intrinsic wrappers or new dtype-prefixed `tcrv_rvv.i32_*` helper ops.
- New route-family owners or follow-on polish for mask/tail, VL/control,
  conversion, reduction, MAcc, segment2, base memory, contraction,
  gather/scatter, or unrelated route families.
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
- Guides read:
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Predecessor archived task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-mask-tail-policy-route-family-owner/`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
