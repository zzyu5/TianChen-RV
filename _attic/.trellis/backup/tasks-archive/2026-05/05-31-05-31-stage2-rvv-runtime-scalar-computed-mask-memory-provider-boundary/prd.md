# Stage2 RVV runtime-scalar computed-mask memory provider-boundary closure

## Goal

Close the RVV plugin-owned provider preflight boundary for
`runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store` before
`TCRVEmitCLowerableRoute` construction.

This task must make the provider/planning layer prove runtime scalar compare
binding, compare-produced mask facts, memory form, inactive-lane/passthrough
behavior, ABI order, runtime `n`/AVL/VL, typed config, route-family plan, and
mirror freshness for both runtime-scalar computed-mask memory routes. Target
artifact and generated-bundle paths remain consumers unless a narrow provider
blocker requires repair.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime-scalar computed-mask memory provider-boundary closure`.
- Module owner: RVV plugin-owned provider preflight boundary for
  `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Session start facts: no `.trellis/.current-task`; worktree clean; HEAD
  `e5e87942 rvv: close runtime-scalar compare-select artifact boundary`.
- This task was created from the supplied Hermes Direction Brief because no
  current task existed.

## What I Already Know

- The latest compare/select work closed runtime-scalar compare/select through
  provider, artifact, generated-bundle, and runtime evidence boundaries. This
  task is the next bounded memory-provider boundary, not a compare/select redo.
- Specs require the current RVV-first authority chain:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized tcrv_rvv body
    -> RVV selected-body realization
    -> route-family provider plans and statement/operand-binding facts
    -> provider preflight
    -> TCRVEmitCLowerableRoute
    -> neutral EmitC
  ```

- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV compute, dtype, memory form, mask semantics,
  inactive-lane behavior, or route support.
- Runtime-scalar computed-mask store/load-store already have selected-body
  owner/route-family structure and target/generated-bundle tests, but the
  Direction Brief identifies the missing boundary as an equivalent provider
  preflight before route construction.
- Common EmitC and target artifact code may consume provider-built route facts
  and mirror metadata after route construction. They must not choose RVV
  semantics from route ids, artifact names, ABI strings, scripts, exact
  intrinsic spellings, descriptors, source-front-door residue, or mirror-only
  status fields.

## Requirements

1. Add or repair one coherent RVV-owned provider-boundary module for
   `runtime_scalar_cmp_masked_store` and
   `runtime_scalar_cmp_masked_load_store`.
2. The boundary must run before `TCRVEmitCLowerableRoute` construction and
   consume the selected-body route analysis, materialization facts,
   computed-mask memory route-family plan, computed-mask memory statement plan,
   memory operand-binding facts, route-control/runtime facts, mask/tail facts,
   typed config facts, selected target capability facts, and runtime ABI
   parameters.
3. The positive store case must prove same-analysis runtime scalar threshold
   binding, lhs/source/destination/runtime `n` ABI binding, computed compare
   mask producer facts, store memory form, inactive-lane store contract,
   mask/tail policy, typed SEW/LMUL/dtype relation, runtime `n`/AVL/VL
   relation, fresh family-plan/mirror metadata, and provider-ready statement
   facts.
4. The positive load-store case must prove all shared store facts plus the
   load-store memory form, source/load value role, destination role, and
   explicit old-destination passthrough/inactive-lane preservation behavior.
5. Fail closed before route construction for:
   - missing or stale runtime scalar binding;
   - missing lhs/source/destination/runtime `n` ABI binding;
   - wrong store versus load-store memory form;
   - missing old-destination passthrough for load-store;
   - wrong inactive-lane contract;
   - missing or stale compare-produced mask facts;
   - wrong predicate or mask/tail policy;
   - stale computed-mask memory family-plan id;
   - stale owner/mirror metadata;
   - wrong SEW/LMUL/dtype relation;
   - wrong runtime `n`/AVL/VL relation;
   - direct-route-entry-only claims;
   - artifact-name/script-derived authority;
   - exact-intrinsic-as-authority;
   - common-EmitC semantic choice.
6. Keep target artifact and generated-bundle paths as consumers. Only change
   them if the provider-boundary implementation exposes a narrow blocker.
7. Keep common EmitC neutral and provider-built. Do not move RVV memory or mask
   semantics into common EmitC, target export, scripts, descriptors, or
   metadata mirrors.
8. Do not broaden to runtime-scalar reductions, runtime-scalar MAcc,
   compare/select redo, widening conversion, segment2, base memory,
   non-runtime computed-mask memory expansion, dtype/LMUL clone batches,
   high-level Linalg/frontend work, one-intrinsic wrappers, dashboards,
   reports, or broad smoke matrices.

## Acceptance Criteria

- [ ] PRD and task context truthfully represent the Direction Brief, current
      repo state, and relevant Trellis specs.
- [ ] A focused production diff lands in RVV EmitC route planning/provider code
      and directly necessary headers. Tests alone are not sufficient.
- [ ] `RVVEmitCRouteProvider` or the RVV planning layer calls the new/repaired
      provider preflight after verified route-family plans, materialization
      facts, memory operand-binding facts, route-control facts, and statement
      facts, but before creating `TCRVEmitCLowerableRoute`.
- [ ] Positive C++ coverage proves provider preflight acceptance for
      `runtime_scalar_cmp_masked_store`.
- [ ] Positive C++ coverage proves provider preflight acceptance for
      `runtime_scalar_cmp_masked_load_store`.
- [ ] C++ fail-closed coverage proves targeted diagnostics for stale/missing
      runtime scalar, mask, memory-form, passthrough, ABI, typed-config,
      runtime-control, and mirror facts before route construction.
- [ ] Existing direct pre-realized fail-closed tests remain passing for the two
      runtime-scalar computed-mask memory routes.
- [ ] Focused generated-bundle dry-run or lit non-regression passes for both
      selected-boundary paths with `route_entry_realization: false`.
- [ ] Existing relevant target artifact tests remain passing, or an exact
      blocker is recorded.
- [ ] Bounded owner/API scan shows selected-body owner declarations remain out
      of route-family planning headers unless already allowed by spec.
- [ ] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused tests pass, and `check-tianchenrv` passes or the exact blocker is
      recorded.
- [ ] Task status, context, and journal are truthful; if acceptance passes, the
      task is finished/archived and one coherent commit is created.

## Out Of Scope

- No artifact/runtime closure as the main milestone.
- No runtime-scalar reductions, runtime-scalar MAcc, compare/select redo,
  widening conversion, segment2, base memory, non-runtime computed-mask memory
  expansion, dtype/LMUL clone batches, Linalg/frontend lowering,
  one-intrinsic wrappers, dashboards, reports, or broad smoke matrices.
- No edits to `tcrv.exec` that encode compute semantics.
- No RVV semantic choices in common EmitC or target artifact code.
- No Python implementation of compiler core, dialects, route provider,
  capability model, lowering, or emission. Python remains evidence tooling.
- No runtime, correctness, or performance claim without real `ssh rvv`
  evidence.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous archived tasks read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-runtime-scalar-compare-select-provider-boundary-closure/prd.md` and
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-runtime-scalar-compare-select-artifact-runtime-closure/prd.md`.
- Relevant workspace journal entry read:
  `.trellis/workspace/codex/journal-19.md`, Session 350.
- Primary production files for inspection and possible edit:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  and `lib/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp`
  if present.
- Direct consumers to keep passing unless a provider blocker is found:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  directly related runtime-scalar computed-mask memory generated-bundle tests,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.
- No blocking user question remains; the supplied Direction Brief and specs are
  specific enough for one bounded implementation round.
