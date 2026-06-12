# Expand: Stage2 RVV regular computed-mask memory provider-boundary closure

## Goal

Close the RVV plugin-owned provider preflight boundary for the regular
non-segment computed-mask memory route family before
`TCRVEmitCLowerableRoute` construction.

This round extends the provider-boundary invariant that was just closed for
`runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store` to the regular selected-body consumers:
`computed_masked_unit_load_store`, `computed_masked_strided_store`,
`computed_masked_strided_load_unit_store`,
`computed_masked_indexed_gather_load_unit_store`, and
`computed_masked_indexed_scatter_store_unit_load`.

## Direction Source

- Direction title: `Expand: Stage2 RVV regular computed-mask memory
  provider-boundary closure`.
- Module owner: RVV selected-body EmitC route-provider preflight for the
  regular computed-mask memory family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Session start facts: no `.trellis/.current-task`; worktree clean; HEAD
  `c9719a20 rvv: close runtime-scalar computed-mask memory provider boundary`.
- This task was created from the supplied Hermes Direction Brief because no
  current task existed.

## What I Already Know

- The previous archived task added
  `verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(...)`
  and made `RVVEmitCRouteProvider` run it before route construction for the two
  runtime-scalar computed-mask memory routes.
- Specs require the current RVV authority chain:

  ```text
  selected tcrv.exec RVV variant
    -> typed/pre-realized tcrv_rvv body
    -> RVV selected-body realization
    -> route-family provider plans and operand/materialization/statement facts
    -> provider preflight
    -> TCRVEmitCLowerableRoute
    -> neutral common EmitC
  ```

- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV compute, dtype, memory form, mask semantics,
  inactive-lane behavior, stride/index semantics, or route support.
- The regular computed-mask memory family already has selected-body consumers,
  computed-mask memory family plans, memory operand-binding facts, route-control
  facts, and computed-mask memory statement plans. The missing boundary is a
  provider preflight that proves those facts before route construction.
- Common EmitC, target artifact export, scripts, route ids, ABI names,
  artifact names, exact intrinsic strings, descriptors, and mirror/status
  fields are not RVV route authority.

## Requirements

1. Add or repair one coherent RVV-owned provider-boundary preflight module for
   the regular non-segment computed-mask memory forms:
   - `computed_masked_unit_load_store`;
   - `computed_masked_strided_store`;
   - `computed_masked_strided_load_unit_store`;
   - `computed_masked_indexed_gather_load_unit_store`;
   - `computed_masked_indexed_scatter_store_unit_load`.
2. The preflight must run after verified route-family provider plans,
   materialization facts, memory operand-binding facts, route-control facts,
   and computed-mask memory statement-plan construction, but before
   `TCRVEmitCLowerableRoute` construction.
3. The preflight must consume and validate same-analysis:
   - typed config facts for dtype, SEW, LMUL, policy, VL/vector/mask C types,
     setvl, vector load, mask, store, stride, and index leaves where required;
   - computed-mask memory family plan facts, including operation kind, memory
     form, mask producer/source, inactive-lane/passthrough contract, stride
     source, index source, index EEW/unit/uniqueness, provider mirror fields,
     and required intrinsic/header/type mirrors;
   - route materialization facts and provider-ready statement-plan leaves;
   - memory operand-binding facts for compare inputs, source, destination,
     runtime `n`, source/destination stride, and index operands as applicable;
   - shared route-control facts for runtime `n`/AVL/VL, typed config, selected
     capability, mask/tail policy, runtime ABI order, and provider mirrors.
4. The provider must fail closed before route construction when any selected
   regular computed-mask memory route lacks fresh materialization facts, memory
   operand-binding facts, computed-mask statement-plan facts, typed config
   facts, mask/tail policy facts, inactive-lane/passthrough facts, runtime
   `n`/AVL/VL facts, ABI facts, stride/index facts where applicable, or
   provider mirror facts.
5. The implementation must keep runtime-scalar computed-mask memory behavior
   as a non-regression path. Do not reopen its semantics except where shared
   helper reuse is needed.
6. Keep common EmitC neutral and provider-built. Do not move RVV memory or mask
   semantics into common EmitC, target export, scripts, descriptors, metadata
   mirrors, route ids, or exact intrinsic spelling as authority.
7. If the full five-route family is too large for one safe round, complete one
   coherent unit + strided provider-boundary submodule and leave the task state
   explicit about indexed continuation. The preferred target is all five forms.

## Acceptance Criteria

- [x] PRD and task context truthfully represent the Direction Brief, current
      repo state, previous archived task, and relevant Trellis specs.
- [x] A focused production diff lands in RVV EmitC route planning/provider
      ownership code and directly necessary headers. Tests alone are not
      sufficient.
- [x] `RVVEmitCRouteProvider` or the RVV planning layer calls the regular
      computed-mask memory provider preflight after route-family,
      materialization, memory-binding, route-control, and statement-plan facts
      exist, but before creating `TCRVEmitCLowerableRoute`.
- [x] Positive C++ coverage proves provider preflight acceptance for
      `computed_masked_unit_load_store`.
- [x] Positive C++ coverage proves provider preflight acceptance for
      `computed_masked_strided_store`.
- [x] Positive C++ coverage proves provider preflight acceptance for
      `computed_masked_strided_load_unit_store`.
- [x] Positive C++ coverage proves provider preflight acceptance for
      `computed_masked_indexed_gather_load_unit_store`.
- [x] Positive C++ coverage proves provider preflight acceptance for
      `computed_masked_indexed_scatter_store_unit_load`.
- [x] C++ fail-closed coverage proves targeted diagnostics for missing/stale
      materialization facts, memory operand binding, computed-mask statement
      plan, mask/tail policy, inactive-lane/passthrough, typed config,
      runtime-control `n`/AVL/VL, ABI facts, stride/index facts, stale provider
      mirrors, and wrong memory form before route construction.
- [x] Existing runtime-scalar computed-mask memory provider preflight tests
      remain passing.
- [x] Focused generated-bundle dry-runs and target artifact tests pass for the
      scoped regular computed-mask memory forms, or an exact blocker is
      recorded.
- [x] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [x] `git diff --check` passes.
- [x] Focused tests pass, and `check-tianchenrv` passes or the exact blocker is
      recorded.
- [x] Task status, context, and journal are truthful; if acceptance passes, the
      task is finished/archived and one coherent commit is created.

## Out Of Scope

- Do not reopen `runtime_scalar_cmp_masked_store` or
  `runtime_scalar_cmp_masked_load_store` beyond non-regression and shared
  helper reuse.
- No segment2 memory, standalone reduction, MAcc, widening dot, compare/select
  redo, conversion, new dtype/LMUL coverage, source-front-door routes,
  high-level Linalg/frontend lowering, one-intrinsic wrapper dialects, broad
  smoke matrices, dashboards, reports, or evidence-only packaging.
- No edits to `tcrv.exec` that encode compute semantics.
- No RVV semantic choices in common EmitC or target artifact code.
- No Python implementation of compiler core, dialects, route provider,
  capability model, lowering, or emission. Python remains evidence tooling.
- No runtime, correctness, or performance claim without real `ssh rvv`
  evidence.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archived task read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-runtime-scalar-computed-mask-memory-provider-boundary/`.
- Relevant workspace journal entries read:
  `.trellis/workspace/codex/journal-19.md`, Session 351, and
  `.trellis/workspace/codex/journal-14.md`, Session 179.
- Primary production files for inspection and possible edit:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Direct consumer evidence to keep focused:
  `test/Plugin/RVVExtensionPluginTest.cpp`, directly related regular
  computed-mask memory generated-bundle tests, directly related target artifact
  tests, and runtime-scalar computed-mask memory preflight non-regression.
- No blocking user question remains; the supplied Direction Brief and specs are
  specific enough for one bounded implementation round.

## Completion Notes

- Production owner changes:
  `verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(...)` was
  added to the RVV EmitC route planning owner and called from
  `buildRVVSelectedBodyEmitCLowerableRouteFromAnalysis(...)` for the five
  regular computed-mask memory routes before `TCRVEmitCLowerableRoute`
  construction.
- Provider-boundary forms closed:
  `computed_masked_unit_load_store`, `computed_masked_strided_store`,
  `computed_masked_strided_load_unit_store`,
  `computed_masked_indexed_gather_load_unit_store`, and
  `computed_masked_indexed_scatter_store_unit_load`.
- Fail-closed diagnostics covered in C++ plugin tests: missing materialization
  plan, stale memory operand binding, stale statement-plan flags/leaves, stale
  mask/tail policy, stale typed config, runtime-control/provider-plan drift,
  stale provider mirrors, wrong memory form, missing destination stride ABI,
  and missing index ABI.
- Focused evidence:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  regular computed-mask lit filter
  `computed-masked-(unit-load-store|strided-store|strided-load|indexed-gather-load|indexed-scatter-store)`,
  runtime-scalar non-regression lit filter
  `runtime-scalar-cmp-masked-(store|load-store)|runtime-scalar-computed-mask-(store|load-store)`,
  `git diff --check`, and `ninja -C build check-tianchenrv`.
- Full check evidence: `check-tianchenrv` passed all 464 lit tests.
- Authority scan: production diff added no descriptor, source-front-door,
  route-id, artifact-name, common-EmitC, script, or exact-intrinsic authority.
  Existing full-file hits are prior RVV/test legacy strings or PRD prohibition
  text, not this round's new production authority.
- No `ssh rvv` run was performed, so this task makes no runtime,
  correctness-on-hardware, or performance claim.
