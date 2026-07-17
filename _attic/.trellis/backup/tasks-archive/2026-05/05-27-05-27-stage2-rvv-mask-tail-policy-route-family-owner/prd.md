# Stage2 RVV Mask/Tail Policy Route-Family Owner

## Goal

Introduce one RVV plugin-local mask/tail policy owner boundary for active
selected-body RVV routes that consume masks. The owner must carry mask operand
provenance, mask form/type, mask and tail policy, inactive-lane behavior,
VL/AVL relation, dtype/config, operation family, runtime ABI binding, provider
route facts, target artifact ABI validation, generated RVV artifact facts, and
runtime evidence through the production route path.

The bounded migrated consumers for this round are:

- existing non-segment computed-mask memory, with
  `computed_masked_unit_load_store` as the primary masked route representative;
- existing computed-mask select, with `computed_mask_select` as the adjacent
  compare-produced mask consumer.

The shared path must remain:

```text
selected tcrv.exec RVV variant
  -> explicit typed tcrv_rvv body/config/runtime/mask facts
  -> RVV plugin-owned mask/tail policy owner
  -> provider-built route facts and statement plans
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI consumer
  -> generated RVV C artifact/harness
  -> ssh rvv evidence when correctness is claimed
```

## Direction Source

- Direction title: `Stage2 RVV mask/tail policy route-family owner`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `3f05adad chore: repair stray classroom trellis task`.
- `.trellis/.current-task` was absent, so this Trellis task was created from
  the supplied Hermes Direction Brief.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows for implementation.

## What I Already Know

- Specs require current RVV executable work to start from selected
  `tcrv.exec` RVV variants with explicit typed `tcrv_rvv` bodies, not route
  ids, artifact names, descriptors, source-front-door fixtures, ABI strings,
  exact intrinsic spellings, common EmitC, or mirror metadata.
- The previous completed runtime AVL/VL task added a
  `RVVSelectedBodyRouteControlProviderPlan` boundary. That boundary already
  carries tail and mask policy mirrors with runtime control, but it does not
  by itself own mask provenance, mask form/type, inactive-lane behavior, or
  cross-family mask/tail policy classification.
- Existing family plans for masked/compare routes contain mask facts such as
  `maskProducerSource`, `maskRole`, `maskSource`, `maskMemoryForm`,
  `maskTypeName`, `maskCType`, `inactiveLaneContract`, and
  `maskedPassthroughLayout`. These facts are currently duplicated across
  route-family plans and metadata emission.
- Existing active statement-plan boundaries cover computed-mask memory and
  compare/select. This task should add/extract the shared mask/tail policy
  owner and make those statement plans consume it, not replace them with a new
  central statement-builder.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV dtype, operation kind, mask support, tail/mask
  policy, inactive-lane semantics, intrinsic family, or route support.
- Common EmitC/export must remain neutral. It may materialize provider-built
  route payload and carry explicit mirrors after route construction, but it
  must not infer mask/tail policy from route names, metadata, artifacts,
  scripts, ABI spellings, descriptors, source-front-door data, or exact
  intrinsic names.

## Requirements

1. Add or extract one explicit RVV plugin-local mask/tail policy route-family
   owner boundary.
2. The owner must select active consumers through an RVV-local owner registry
   and fail closed on ambiguous ownership.
3. The owner must support at least computed-mask memory and computed-mask
   select as active production consumers in this round.
4. The owner must derive and validate mask source/provenance, mask role,
   mask form/type, mask memory form, tail policy, mask policy, inactive-lane
   contract, passthrough/select layout, VL/AVL relation, dtype/config,
   runtime ABI order, header/type mappings, intrinsic-family facts,
   provider-supported mirror labels, and selected capability facts from the
   same typed selected-body route analysis and provider-built facts.
5. Computed-mask memory and computed-mask select statement-plan construction
   must consume the shared owner before attaching provider-built statements to
   `TCRVEmitCLowerableRoute`.
6. Target artifact ABI validation must consume provider-built mask/tail policy
   facts for the migrated consumers before accepting generated artifact or
   header claims.
7. Unsupported or inconsistent family, stale route id, stale mirror metadata,
   missing mask binding, wrong mask provenance, wrong mask type/form, wrong
   tail or mask policy, wrong inactive-lane behavior, wrong VL/mask relation,
   dtype/config mismatch, wrong ABI order, missing operand binding, wrong
   type/header mapping, exact-intrinsic-as-authority, artifact-name authority,
   or script-derived authority must fail closed with targeted diagnostics where
   the current test surface exposes the condition.
8. Do not weaken completed base memory, segment2, MAcc, standalone reduction,
   compare/select, computed-mask memory, conversion dtype-policy, VL/control,
   scalar-broadcast, contraction, or runtime-scalar route owners.

## Acceptance Criteria

- [ ] A production `RVVSelectedBodyMaskTailPolicy...` owner boundary exists in
      RVV plugin-local C++ and is selected through an owner registry.
- [ ] `computed_masked_unit_load_store` or an equivalent active
      non-segment computed-mask memory route consumes the owner before
      statement-plan construction and provider route construction.
- [ ] `computed_mask_select` or an equivalent active compare-produced mask
      consumer consumes the same owner before statement-plan construction and
      provider route construction.
- [ ] Provider-built routes for both migrated consumers prove agreement among
      typed config, selected capability, runtime AVL/VL control, operand
      binding, mask/tail policy, materialization facts, statement plan,
      header/type/intrinsic facts, ABI order, and provider-supported mirrors.
- [ ] Target artifact ABI validation rejects missing/stale/mismatched
      mask/tail policy owner claims for the migrated consumers.
- [ ] Focused C++ or lit tests cover positive owner membership/dispatch,
      provider/statement-plan consumption, target ABI consumption, and
      fail-closed diagnostics for at least stale/missing owner plan, missing
      mask binding, wrong mask producer/source/form, policy mismatch,
      inactive-lane mismatch, wrong ABI order, and mirror/artifact authority
      attempts exposed by current code.
- [ ] Generated-bundle dry-run covers both migrated consumers.
- [ ] Real `ssh rvv` generated-bundle execution covers the masked memory
      representative over counts including `0`, small, exact, tail, and stress
      cases.
- [ ] Focused non-regression covers completed segment2 update/store/load,
      segment2 deinterleave/interleave, computed-mask MAcc,
      scalar-broadcast MAcc, runtime-scalar MAcc, standalone reduction,
      compare/select, computed-mask memory, conversion dtype-policy,
      VL/control runtime-AVL, scalar-broadcast, contraction, and base memory
      paths.
- [ ] Bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis task status, context, journal, and archive state are truthful,
      and one coherent commit is created if all acceptance criteria are met.

## Validation Plan

1. Validate this task after PRD/context setup.
2. Inspect computed-mask memory, computed-mask select, route-control provider,
   operand-binding facts, statement-plan builders, route materialization facts,
   target artifact ABI validation, generated-bundle script, and focused tests.
3. Add the mask/tail policy owner boundary in the existing RVV planning header
   and implementation.
4. Make computed-mask memory and computed-mask select statement-plan builders
   consume the owner plan.
5. Add provider route mirrors and target ABI validation only after provider
   route construction.
6. Add focused tests for owner selection, positive consumption, and fail-closed
   stale/missing/mismatched facts.
7. Run focused C++/lit tests for touched RVV planning/provider/target surfaces.
8. Run generated-bundle dry-runs for both migrated consumers.
9. Run real `ssh rvv` generated-bundle evidence for the masked memory
   representative over `0`, small, exact, tail, and stress counts.
10. Run focused non-regression, bounded authority scan, `git diff --check`,
    Trellis validation, and `check-tianchenrv` or record the exact blocker.

## Out Of Scope

- Broad Stage2 route coverage batches.
- New dtype/LMUL clone sets.
- High-level Linalg, Vector, StableHLO, source-front-door, Toy,
  TensorExtLite, Template, IME, Offload, or future plugin routes.
- One-intrinsic wrapper dialects or new dtype-prefixed `tcrv_rvv.i32_*`
  helpers.
- Descriptor-driven compute, direct-C/source-export routes, or legacy
  `RVVI32M1` / `rvv-i32m1` executable compatibility.
- Dashboard, report-only, prompt-only, helper-only, or evidence-only work as
  the main achievement.
- Follow-on polish for VL/control, conversion, segment2, MAcc, reduction,
  compare/select, gather/scatter, or unrelated route families unless required
  to keep this owner truthful.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/index.md`.
- Guides read:
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Predecessor archived tasks read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-vl-control-runtime-avl-boundary-owner/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-27-trellis-control-plane-stray-classroom-task-repair/prd.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
