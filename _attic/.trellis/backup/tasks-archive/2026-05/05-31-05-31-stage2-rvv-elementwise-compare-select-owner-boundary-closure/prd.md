# Stage2 RVV elementwise compare-select owner-boundary closure

## Goal

Turn the completed elementwise/compare-select selected-body owner interface
split into an operational compiler-path closure:

```text
selected tcrv.exec RVV variant
  -> owner-local elementwise/compare-select selected-body realization
  -> realized typed tcrv_rvv compare/select or computed-mask select facts
  -> RVV route-family/provider plans
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC materialization
  -> generated target artifact evidence
```

The goal is production-path movement, not another header-only cleanup. The
owner-local realization API must be the selected-body owner surface for the
elementwise/compare-select cluster. Route-family planning and the route
provider may consume only realized typed facts, verified family/provider plans,
operand-binding facts, and statement plans; they must not consume owner
metadata, mirror metadata, route ids, artifact names, direct route-entry
claims, or common EmitC-derived semantics as authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV elementwise compare-select owner-boundary artifact closure`.
- Module owner: `RVVElementwiseSelectedBodyRealizationOwner` boundary for the
  elementwise/compare-select selected-body cluster.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Session start facts: worktree was clean; HEAD was
  `8375f6c4 rvv: split elementwise selected-body owner interface`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief.

## What I Already Know

- The previous archived task
  `05-31-stage2-rvv-elementwise-compare-select-owner-interface-split` moved
  elementwise/compare-select selected-body owner declarations into
  `RVVElementwiseSelectedBodyRealizationOwner.h` and removed them from the
  EmitC route-family planning header.
- The previous task explicitly preserved behavior and recorded that no route or
  executable semantics changed.
- The relevant RVV spec now requires owner-family public predicates,
  realization hooks, owner-local result types, and diagnostics to live in
  dedicated owner interfaces, while EmitC route-family planning headers expose
  route-plan, mirror-validation, and operand-binding APIs only.
- The relevant RVV spec already defines an
  `Elementwise/Compare-Select Selected-Body Realization Boundary` and a
  `Selected-Body Compare/Select Route Statement Plan` boundary. This task must
  prove those boundaries are connected in the production path for at least
  plain `cmp_select` and `computed_mask_select`.
- Support levels must remain separate:
  parseable/verifier-legal selected-body IR is not route support; route support
  is not executable correctness or performance evidence; mirror metadata is
  not authority.

## Requirements

1. Inspect and, if needed, harden the production elementwise/compare-select
   selected-body path so the central selected-body dispatcher consumes the
   dedicated owner-local interface for the cluster.
2. Keep `RVVEmitCElementwiseRouteFamilyPlanOwners.h` free of selected-body
   owner declarations. It may expose only route operation predicates,
   route-family consumer predicates, route-plan derivation/application/
   validation, route-description mirror validation, operand-binding helpers,
   materialization-fact consumers, and statement-plan/provider-plan APIs.
3. Ensure route-family planning and `RVVEmitCRouteProvider` consume realized
   typed compare/select facts, verified family/provider plans,
   route-control/provider facts, operand-binding facts, and compare/select
   statement plans, not owner symbols or metadata as route authority.
4. Ensure at least plain `cmp_select` and `computed_mask_select` selected
   boundary paths flow from owner-local realization into provider route facts
   and generated target artifact evidence.
5. If the current implementation already has most of the positive path, add
   explicit owner/provider contracts and fail-closed diagnostics at the first
   incorrect boundary instead of adding wrapper-only files.
6. Add or repair focused tests that prove the owner boundary is operationally
   visible in the compiler path and that stale/non-authoritative inputs fail
   closed before route/provider/common EmitC artifact construction.
7. Preserve existing elementwise arithmetic, masked elementwise,
   scalar-broadcast elementwise, runtime-scalar compare/select, and
   dual-runtime-scalar compare-mask-and-select behavior unless a directly
   touched adapter requires a local update.
8. Avoid broad coverage expansion: no new dtype/LMUL clone batch, no Linalg or
   frontend lowering, no TensorExt/IME/Offload work, no source-front-door
   positive route, no direct route-entry support, no one-intrinsic wrapper
   dialect, and no common EmitC semantic branch.

## Acceptance Criteria

- [ ] The selected-body dispatcher reaches the elementwise/compare-select
      cluster through `RVVElementwiseSelectedBodyRealizationOwner.h` and its
      owner-local implementation; no route-family planning header is needed to
      expose selected-body owner APIs.
- [ ] Plain `cmp_select` selected-boundary realization carries operation kind,
      predicate kind, select layout, dtype/config, policy, runtime n/AVL/VL,
      operand binding, ABI roles, and realized load/compare/select/store facts
      into the route-provider path.
- [ ] `computed_mask_select` selected-boundary realization carries mask source,
      producer/source facts, computed-mask layout, predicate/select facts,
      dtype/config, policy, runtime n/AVL/VL, operand binding, ABI roles, and
      realized compare/mask/select/store facts into the route-provider path.
- [ ] `RVVEmitCRouteProvider` consumes compare/select provider plans,
      materialization facts, elementwise/select operand-binding facts,
      route-control provider facts, and compare/select statement plans before
      constructing the `TCRVEmitCLowerableRoute`.
- [ ] Route-family planning and provider code do not use selected-body owner
      symbols, owner-local result objects, route ids, artifact names,
      emission-plan/status fields, ABI strings, exact intrinsic spellings,
      source-front-door markers, descriptors, scripts, common EmitC branches,
      or legacy i32 helper names as authority.
- [ ] Fail-closed diagnostics cover wrong or stale predicate, mask source,
      select layout, operand binding, dtype/config/policy, AVL/VL relation,
      stale route id, mirror metadata-as-authority, direct-route-entry-only
      claim, artifact-name-derived claim, exact-intrinsic-as-authority, and
      common-EmitC semantic invention where those inputs can enter this
      bounded path.
- [ ] Generated-bundle dry-run or target artifact checks cover both
      `cmp_select` and `computed_mask_select` through selected lowering-boundary
      realization. `ssh rvv` evidence is required only if this task claims new
      executable correctness or performance.
- [ ] Direct pre-realized route-entry-only claims remain fail-closed; the
      public selected lowering-boundary producer must consume the selected
      pre-realized body before provider facts are collected.
- [ ] Existing focused RVV plugin tests for elementwise/compare-select owner
      selection, realization, provider consumption, and fail-closed cases pass.
- [ ] Related cmp_select and computed_mask_select target/script tests pass as
      consumers and evidence, without using artifact metadata as authority.
- [ ] Bounded owner/API scan proves the route-family planning header remains
      free of selected-body owner declarations.
- [ ] Bounded authority scan over touched production files finds no new
      central ad hoc, name-derived, metadata-derived, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only,
      or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` passes, or the exact blocker is recorded.
- [ ] Task status/context/journal are truthful; task is finished/archived and
      one coherent commit is created if acceptance passes.

## Out Of Scope

- No Stage2 coverage expansion outside the elementwise/compare-select selected
  body owner/provider/artifact path.
- No unrelated contraction, segment, memory-movement, standalone reduction,
  MAcc, or conversion owner rewrite unless a directly touched boundary requires
  a small local adapter.
- No new direct pre-realized route-entry support.
- No source-front-door/source-artifact authority.
- No route support based on mirrors, route ids, artifact names, exact intrinsic
  names, ABI strings, scripts, descriptors, manifests, or common EmitC.
- No runtime/correctness/performance claim without real `ssh rvv` evidence.

## Technical Notes

- Specs read before PRD: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archived task read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-elementwise-compare-select-owner-interface-split/prd.md`,
  `implement.jsonl`, and `check.jsonl`.
- Relevant workspace journal entries read from `.trellis/workspace/codex/journal-19.md`,
  especially Session 350 and nearby owner-cleanup entries.
- Direction brief named production files for inspection:
  `include/TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related
  cmp_select / computed_mask_select target/script tests.
- No blocking user question remains; the supplied Direction Brief and specs are
  specific enough to proceed to implementation after context validation.
