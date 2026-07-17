# Stage2 RVV runtime-scalar compare-select artifact runtime closure

## Goal

Close the executable artifact/runtime boundary for
`runtime_scalar_cmp_select` and
`runtime_scalar_dual_cmp_mask_and_select`. The previous task closed the
provider preflight before `TCRVEmitCLowerableRoute` construction; this task
must carry the same route-supported family through target artifact validation,
generated bundle ABI, generated C harness behavior, and real `ssh rvv`
correctness evidence.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime-scalar compare-select artifact/runtime closure`.
- Module owner: generated-artifact and runtime ABI boundary for
  `runtime_scalar_cmp_select` and
  `runtime_scalar_dual_cmp_mask_and_select`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Session start facts: no `.trellis/.current-task`; worktree clean; HEAD
  `1bf95d6a rvv: close runtime-scalar compare-select provider boundary`.
- This task was created from the supplied Hermes Direction Brief because no
  current task existed.

## What I Already Know

- Commit `1bf95d6a` added provider-side runtime-scalar compare/select
  preflight coverage and deliberately made no runtime correctness claim.
- The relevant authority chain is:

  ```text
  selected tcrv.exec RVV runtime-scalar compare/select variant
    -> typed pre-realized tcrv_rvv body
    -> RVV selected-body realization
    -> provider-validated compare/select route facts
    -> TCRVEmitCLowerableRoute
    -> neutral EmitC
    -> target artifact ABI and generated bundle
    -> ssh rvv correctness evidence
  ```

- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. Runtime scalar operand role, buffer roles, runtime `n`, and ABI order
  must be consumed by selected typed RVV body/provider facts before artifact
  export.
- Target artifact export may validate metadata mirrors after rebuilding the
  provider route, but artifact names, route ids, ABI strings, exact intrinsic
  names, scripts, and mirror fields are not route or executable authority.
- Current code already has a compare/select-mask route-family target validator
  and generated-bundle dry-run tests for runtime-scalar compare/select variants;
  this task must verify whether those are sufficient for executable closure and
  add production movement where a blocker exists.

## Requirements

1. Preserve operation kind, runtime scalar operand role, scalar splat/use,
   lhs/rhs/source/output buffer roles, predicate kind, select layout,
   computed-mask ownership, mask/tail and inactive-lane policy, runtime
   `n`/AVL/VL relation, SEW/LMUL, provider-supported mirror facts, artifact
   ABI order, and generated C behavior for both runtime-scalar shapes.
2. Ensure target artifact validation consumes rebuilt provider route facts and
   mirror metadata only after route construction for:
   - `runtime_scalar_cmp_select`
   - `runtime_scalar_dual_cmp_mask_and_select`
3. Ensure generated-bundle evidence proves selected-boundary materialization,
   route-provider facts, runtime scalar ABI order, runtime `n`/AVL/VL, scalar
   splat, predicate/select behavior, mask composition for the dual route, and
   tail sentinel preservation.
4. Ensure generated C/harness behavior checks runtime counts including
   `0`, `1`, exact VL, tail, and stress cases, with at least two signed i32
   data patterns, scalar values, predicates, and select layouts relevant to
   the supported family.
5. Fail closed on unsupported or inconsistent selected bodies, stale route ids,
   wrong ABI names/order, wrong runtime scalar binding, missing runtime `n`,
   missing buffer binding, wrong predicate/select layout, changed mask/tail
   policy, common EmitC semantic invention, script-derived authority,
   artifact-name-derived authority, exact-intrinsic-as-authority,
   direct-route-entry-only claims, and pre-realized-fixture-only executable
   claims.
6. Keep common EmitC neutral. Do not infer RVV semantics from common
   materialization, target artifact names, route ids, ABI strings, manifests,
   descriptors, source-front-door residue, or exact intrinsic spellings.
7. Do not reopen provider preflight except for blockers discovered while
   making artifact/runtime execution work.
8. Do not expand to computed-mask store/load-store, runtime-scalar reductions,
   MAcc/dot-reduce, widening conversion, segment2, new dtype/LMUL clone batches,
   Linalg/frontend lowering, one-intrinsic wrappers, dashboards, reports, or
   broad smoke matrices.

## Acceptance Criteria

- [x] PRD and task context truthfully represent the Hermes brief and current
      repo state.
- [x] A focused production diff lands in target artifact/runtime boundary,
      generated-bundle/harness evidence logic, or directly necessary route
      metadata plumbing. Tests alone are not sufficient.
- [x] Generated-bundle dry-run passes for both `runtime_scalar_cmp_select` and
      `runtime_scalar_dual_cmp_mask_and_select`, showing selected-boundary
      materialization and `provider_supported_mirror` facts.
- [x] Generated C/harness evidence shows correct runtime scalar ABI, buffer
      ABI, `n`/AVL/VL handling, predicate/select behavior, mask/tail behavior,
      and no direct route-entry shortcut.
- [x] Real `ssh rvv` correctness run passes for counts `0`, `1`, exact VL,
      tail, and stress cases, with at least two signed i32 data/scalar patterns
      for both runtime-scalar compare/select shapes.
- [x] Fail-closed coverage exists for stale ABI/runtime scalar/provider/mirror/
      route authority relevant to this artifact/runtime boundary.
- [x] Non-regression checks pass for plain `cmp_select`, `computed_mask_select`,
      and previous runtime-scalar provider-preflight checks.
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

- No new route family.
- No provider-preflight refactor unless required by artifact/runtime blockers.
- No computed-mask memory, reductions, MAcc/dot-reduce, conversion, segment2,
  source-front-door, Linalg/frontend, dtype/LMUL matrix expansion, dashboard,
  or report work.
- No Python implementation of compiler core, dialects, route provider,
  capability model, lowering, or emission. Python remains evidence tooling.
- No runtime/correctness/performance claim without real `ssh rvv` evidence.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous archived task read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-runtime-scalar-compare-select-provider-boundary-closure/{task.json,prd.md,implement.jsonl,check.jsonl}`.
- Relevant production files for this round:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Directly related tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-select-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`,
  typed i64/LMUL variants for the same shapes, and related direct
  pre-realized fail-closed tests.
- Relevant workspace journal entry read:
  `.trellis/workspace/codex/journal-19.md`, Session 350.
- No blocking user question remains; the direction is bounded enough to
  implement.
