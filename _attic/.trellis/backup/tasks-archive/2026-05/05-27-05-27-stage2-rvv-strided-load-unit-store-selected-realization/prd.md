# Stage2 RVV strided_load_unit_store Selected-Body Realization Migration

## Goal

Move `strided_load_unit_store` generated artifact execution behind the RVV
plugin-local selected-body realization producer and remove its active direct
pre-realized route-entry authority.

The production path for this operation must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv strided memory body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving i32 element type, source
     mem_window, output mem_window, runtime byte-stride parameter, unit output
     store, strided input load, runtime n/AVL/VL, setvl/with_vl placement,
     policy, required capabilities, and artifact ABI order
  -> base memory movement route-family facts
  -> route materialization facts
  -> memory operand-binding facts
  -> route-control provider plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

Direct pre-realized route-entry support for `strided_load_unit_store` must be
demoted or deleted. A generated artifact for this operation must not be
accepted from a direct route-entry shortcut, pre-realized fixture authority,
route id, artifact name, script option, ABI string, exact intrinsic spelling,
common EmitC behavior, descriptor residue, source-front-door metadata, or
legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV strided load/unit-store selected-body
  realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f6cd67ac rvv: demote widen i32 route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents or parallel agent
  workflows.

## Current Repository Facts

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` keeps
  `TypedStridedMemoryPreRealizedBodyOp` under the `base memory movement`
  selected-body realization owner.
- The same owner still marks all `TypedStridedMemoryPreRealizedBodyOp` bodies
  as direct route-entry eligible through
  `isPreRealizedRVVBaseMemoryMovementRouteEntryOp`.
- The strided selected-body validator already requires operation kind
  `strided_load_unit_store`, memory form `strided-load-unit-store`,
  `stride_unit = "byte"`, SEW32, LMUL m1, tail/mask agnostic policy,
  source role `source-input-buffer`, output role `output-buffer`, runtime
  count role `runtime-element-count`, source byte-stride role
  `source-byte-stride`, direct-child placement under the selected variant, and
  non-empty selected variant `requires`.
- The selected-body realization owner already produces a realized body with
  `setvl`, `with_vl`, `strided_load`, `move`, and `store` before route facts
  are collected.
- `scripts/rvv_generated_bundle_abi_e2e.py` still includes
  `strided_load_unit_store` in `supports_direct_pre_realized_route_entry`, so
  script-level direct mode accepts it before selected-boundary materialization.
- The pre-realized generated-bundle dry-run for
  `strided_load_unit_store` already uses
  `--tcrv-materialize-selected-lowering-boundaries`, but it needs to explicitly
  assert `route_entry_realization: false` and selected-body producer evidence.
- `test/Plugin/RVVExtensionPluginTest.cpp` currently exercises
  `rvv_pre_route_strided_load_unit_store` as route-entry eligible through the
  base memory owner. It must be changed to selected-boundary producer positive
  coverage plus direct route-entry fail-closed coverage.
- Archived task
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-widen-i32-to-i64-selected-realization/`
  completed the same owner-vs-route-entry narrowing pattern for widening
  conversion: selected-boundary realization stayed positive, direct mode failed
  closed, generated-bundle dry-run recorded `route_entry_realization: false`,
  real `ssh rvv` passed, and `check-tianchenrv` passed 398/398.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, harnesses,
   and generated-bundle guardrails.
2. Keep `strided_load_unit_store` in the RVV `base memory movement`
   selected-body realization owner. The selected-boundary producer must realize
   the typed pre-realized body before provider route facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `strided_load_unit_store`. Direct shortcut requests must fail closed before
   bundle generation or provider route construction.
4. Preserve typed facts for operation kind `strided_load_unit_store`,
   i32 element type, SEW32 LMUL m1, source mem_window/runtime ABI role,
   output mem_window/runtime ABI role, runtime byte-stride parameter, runtime
   `n`/AVL/VL values, byte-strided source load, unit-stride output store,
   `tcrv_rvv.move` transfer, setvl/with_vl placement, tail/mask policy,
   required capabilities, provider route facts, memory operand-binding facts,
   route-control facts, and artifact ABI order `src,out,n,stride_bytes`.
5. Fail closed for unsupported or inconsistent selected-boundary input,
   including missing runtime_param, missing mem_window/runtime ABI value, wrong
   stride binding, wrong memory role, wrong load/store form, wrong element
   type or config, wrong policy, wrong AVL/VL relation, wrong setvl placement,
   missing capability, stale route id or mirror metadata,
   direct-route-entry-only authority, artifact-name/script-derived authority,
   exact-intrinsic-as-authority, and common-EmitC semantic invention.
6. Reuse existing selected-body realization, base memory route-family,
   route materialization, route-control provider, memory operand-binding,
   target artifact, and generated-bundle boundaries where they already express
   the required facts. Do not introduce a new central route table or common
   EmitC semantic branch.
7. Do not start cmp_select, standalone_reduce_add, computed-masked segment2
   load/store/update, segment2 interleave/deinterleave, high-level
   Linalg/frontend lowering, new dtype/LMUL clone batches, one-intrinsic
   wrapper dialects, selected-body realization framework rewrites,
   dashboard/report work, broad smoke matrices, or evidence-only tasks.
8. Do not add proof-only tests for completed widening conversion, widening dot,
   MAcc, or computed-mask widening-dot paths except as bounded non-regression
   needed by touched files.

## Acceptance Criteria

- [x] Production code no longer treats `strided_load_unit_store` as direct
      pre-realized route-entry eligible, while the `base memory movement`
      selected-body realization owner still realizes it through the public
      selected lowering-boundary producer path.
- [x] C++ tests prove a typed pre-realized `strided_load_unit_store` body
      belongs to the base memory movement realization owner, is not a direct
      route-entry consumer, and fails closed when
      `realizePreRealizedRVVRouteEntrySelectedBody` is used as a direct
      route-entry shortcut.
- [x] C++ or lit tests prove the selected-boundary path consumes realized typed
      facts, including `setvl`, `with_vl`, `strided_load`, `move`, `store`,
      source/output runtime ABI roles, runtime byte-stride role, runtime
      `n`/AVL/VL, base memory family plan, memory operand-binding facts,
      route-control facts, ABI order, provider-supported mirrors, and
      byte-strided source/unit-store memory forms.
- [x] The generated-bundle script rejects
      `--direct-pre-realized-route-entry --op-kind strided_load_unit_store`
      before route-entry materialization or bundle generation.
- [x] Generated-bundle dry-run for pre-realized `strided_load_unit_store`
      passes through `--tcrv-materialize-selected-lowering-boundaries`,
      records `route_entry_realization: false`, records selected-body producer
      evidence, records realized strided memory/provider facts, and records no
      direct route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts `0`, `1`,
      exact, tail, and stress cases with multiple runtime byte strides and
      tail preservation.
- [x] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum the RVV extension plugin
      smoke test and focused lit for selected-boundary
      `strided_load_unit_store`.
- [x] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis status, journal/archive, and final commit are truthful.

## Validation Plan

1. Validate Trellis task context.
2. Inspect current selected-body realization, route planning/provider,
   target-support bundle, generated-bundle script, focused plugin tests, and
   target/script tests for the exact active direct route-entry authority.
3. Narrow RVV base memory route-entry eligibility so `strided_load_unit_store`
   remains selected-boundary capable but is not direct route-entry eligible.
4. Update generated-bundle tooling/lit coverage so direct pre-realized
   route-entry mode fails closed for `strided_load_unit_store`.
5. Strengthen selected-boundary C++/lit/script coverage so evidence explicitly
   checks `route_entry_realization: false`, selected-body producer evidence,
   strided load/unit-store facts, provider mirrors, and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-run for
   `strided_load_unit_store`.
8. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and multiple runtime byte strides through the selected lowering-boundary
   producer path.
9. Run focused adjacent non-regression for explicit selected-body
   `strided_load_unit_store` and unrelated remaining direct route-entry cases
   that share the predicate boundary.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Out Of Scope

- Compare/select, standalone reduction, computed-mask segment2
  load/store/update, segment2 interleave/deinterleave, additional dtype or
  LMUL clone batches, high-level Linalg/Vector/StableHLO frontend work,
  source-front-door construction, descriptor-driven compute, direct C/source
  export paths, one-intrinsic wrapper dialects, selected-body framework
  rewrite, dashboards, prompt-only work, and legacy `RVVI32M1` /
  `rvv-i32m1` compatibility.
- Additional proof-only work for completed widening conversion, widening dot,
  widening MAcc, computed-mask widening dot, strided widening dot, or
  computed-mask strided widening dot paths except as bounded non-regression.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant predecessor task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-widen-i32-to-i64-selected-realization/prd.md`.
- Relevant workspace journal read:
  `.trellis/workspace/codex/journal-16.md`, especially Session 267.
- Initial production owner files inspected:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-load-unit-store-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-strided-load-unit-store-dry-run.test`.
