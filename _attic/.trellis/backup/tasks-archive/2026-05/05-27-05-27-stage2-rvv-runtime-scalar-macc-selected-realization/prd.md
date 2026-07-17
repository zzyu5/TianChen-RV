# Expand Stage2 RVV Runtime-Scalar MAcc Selected-Body Realization Migration

## Goal

Move the production `runtime_scalar_cmp_masked_macc_add` generated-artifact
path behind the RVV plugin-local selected-body realization producer, and
demote the remaining direct pre-realized route-entry shortcut for this
runtime-scalar computed-mask MAcc consumer.

The bounded production chain for this round is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv runtime_scalar_cmp_masked_macc_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / compare load / rhs scalar splat
     / payload loads / accumulator load / compare / masked_macc / store body
  -> computed-mask accumulation MAcc route-family provider facts
  -> route materialization facts
  -> math operand-binding facts including rhs_scalar as abi|splat|cmp-rhs
  -> route-control provider plan
  -> computed-mask accumulation statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

`runtime_scalar_cmp_masked_macc_add` support must come from realized typed
`tcrv_rvv.masked_macc` body facts consumed by the existing computed-mask
accumulation MAcc provider path. It must not be claimed from direct
route-entry acceptance, pre-realized fixture names, route ids, artifact names,
script options, ABI strings, exact intrinsic spelling, common EmitC behavior,
descriptor residue, source-front-door metadata, or legacy i32 helper authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV runtime-scalar MAcc selected-body
  realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c8c937f5 rvv: demote computed mask macc route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The archived `05-27-stage2-rvv-macc-selected-realization` task migrated
  vector-compare `computed_masked_macc_add` off direct route-entry authority
  and left `runtime_scalar_cmp_masked_macc_add` as the adjacent direct
  route-entry-capable computed-mask MAcc sub-family.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` currently has
  `isPreRealizedRVVComputedMaskMAccRouteEntryOp(...)` reject
  `TypedComputedMaskMAccPreRealizedBodyOp` but accept
  `TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp` when its `op_kind` and
  `memory_form` match the runtime-scalar MAcc fixture.
- The selected-body realization owner for computed-mask MAcc already validates
  runtime-scalar threshold binding, accumulator/result layout, mask facts,
  SEW/LMUL/policy, selected requires, runtime ABI roles, and exclusivity, then
  materializes the compare load, runtime scalar splat, payload loads,
  accumulator load, compare, `masked_macc`, and store structure.
- `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --op-kind runtime_scalar_cmp_masked_macc_add` already uses
  `--tcrv-materialize-selected-lowering-boundaries` and records
  `route_entry_realization: false`.
- `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --direct-pre-realized-route-entry --op-kind
  runtime_scalar_cmp_masked_macc_add` is still positive, and the corresponding
  lit test asserts `materializer = rvv-route-entry-selected-body-realization`.
- Existing planning/provider code already includes
  `RuntimeScalarComputedMaskedMAccAdd` in the computed-mask accumulation family
  plan, math operand-binding facts, route-control provider plan, statement-plan
  boundary, target mirrors, generated artifact ABI, and harness evidence.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep the selected-boundary `runtime_scalar_cmp_masked_macc_add` producer
   path positive. It must realize the typed pre-realized body into `setvl`,
   `with_vl`, compare load, runtime scalar splat, payload loads, accumulator
   load, compare, `masked_macc`, and store before provider route facts are
   collected.
3. Demote or delete the `runtime_scalar_cmp_masked_macc_add` direct
   pre-realized route-entry shortcut. The direct shortcut must fail closed with
   a targeted diagnostic or script error instead of producing a generated
   artifact.
4. Preserve the already migrated vector-compare `computed_masked_macc_add`
   selected-boundary path and its fail-closed direct route-entry behavior.
5. Preserve unrelated supported direct route-entry families unless their
   behavior is directly affected by the shared computed-mask MAcc predicate.
6. Preserve typed facts for operation kind, runtime scalar compare threshold
   binding, computed-mask provenance, accumulator layout, multiplicand/addend
   binding, inactive-lane passthrough semantics, memory roles, runtime `n`/AVL/
   VL values, dtype/config/policy, setvl placement, loop relation, required
   RVV capability, provider route facts, and artifact ABI order.
7. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where exposed by current hooks: missing
   runtime param, missing mem_window/runtime ABI value, wrong runtime scalar
   threshold binding, wrong accumulator binding, wrong multiplicand/addend
   binding, wrong mask binding, dtype/config/policy mismatch, wrong AVL/VL
   relation, wrong setvl placement, missing capability, stale route id, stale
   mirror metadata, direct-route-entry-only authority, artifact-name/script-
   derived authority, exact-intrinsic-as-authority, and common-EmitC semantic
   invention.
8. Do not add broad route coverage, `widening_macc_add` migration,
   `scalar_broadcast_macc_add` migration, dtype/LMUL clone sets, high-level
   Linalg/frontend lowering, one-intrinsic wrapper dialects, selected-body
   framework rewrites, descriptor/source-front-door positive routes,
   dashboards, or evidence-only helper work as the main achievement.

## Acceptance Criteria

- [x] Production code demotes `runtime_scalar_cmp_masked_macc_add` direct
      pre-realized route-entry eligibility while keeping the selected-boundary
      producer path positive.
- [x] C++ tests prove `runtime_scalar_cmp_masked_macc_add` is not route-entry
      eligible for direct route-entry realization, and that the selected-body
      realization producer still realizes it before provider route construction.
- [x] C++ or lit tests prove provider route facts for selected-boundary
      `runtime_scalar_cmp_masked_macc_add` consume realized typed body facts
      and include computed-mask accumulation route-family plan, runtime control
      plan, math operand-binding facts for `rhs_scalar`, statement plan, ABI
      order, mask provenance, accumulator layout, inactive-lane contract, and
      provider-supported mirror.
- [x] Focused fail-closed coverage rejects wrong runtime scalar threshold
      binding, wrong accumulator binding, wrong result layout, wrong mask
      provenance, dtype/config/policy mismatch, wrong runtime `n` binding,
      stale op kind, and direct-route-entry-only authority.
- [x] The direct pre-realized route-entry script path for
      `runtime_scalar_cmp_masked_macc_add` fails closed or is removed from
      positive lit coverage.
- [x] Generated-bundle dry-run for `runtime_scalar_cmp_masked_macc_add` passes
      through the selected lowering-boundary path and records no direct
      route-entry materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts including `0`,
      `1`, exact-VL, tail, and stress cases with at least two runtime scalar
      threshold/input/accumulator patterns.
- [x] Focused non-regression covers completed selected-body producer paths and
      adjacent MAcc paths changed or risked by this demotion:
      `computed_masked_macc_add`, `scalar_broadcast_add`,
      `runtime_i32_splat_store`, `computed_mask_select`, plain `macc_add`,
      `scalar_broadcast_macc_add`, and the runtime-scalar MAcc selected path.
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

1. Validate Trellis context.
2. Update the RVV selected-body route-entry predicate so
   `runtime_scalar_cmp_masked_macc_add` uses the selected-boundary producer
   path only.
3. Update generated-bundle tooling and script tests so
   `--direct-pre-realized-route-entry --op-kind
   runtime_scalar_cmp_masked_macc_add` no longer succeeds.
4. Keep or strengthen selected-boundary lit coverage for
   `pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
   and the generated-bundle pre-realized dry-run.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-run for
   `runtime_scalar_cmp_masked_macc_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and at least two `rhs_scalar` values.
8. Run focused non-regression for `computed_masked_macc_add`,
   `computed_mask_select`, `runtime_i32_splat_store`, `scalar_broadcast_add`,
   `macc_add`, `scalar_broadcast_macc_add`, and adjacent computed-mask MAcc
   fixtures as needed.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Stage2 coverage expansion beyond the bounded runtime-scalar computed-mask
  MAcc selected-body realization migration.
- New MAcc dtype/LMUL clone batches, new accumulator helper op families, or
  additional intrinsic wrappers.
- `widening_macc_add`, `scalar_broadcast_macc_add`, Linalg/Vector/StableHLO
  frontend work, source-front-door construction, descriptor-driven compute,
  direct C/source-export paths, and legacy `RVVI32M1` / `rvv-i32m1`
  compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-macc-selected-realization/`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Implementation Result

- Deleted the shared computed-mask MAcc direct route-entry predicate in
  `RVVSelectedBodyRealization.cpp`, so both vector-compare
  `computed_masked_macc_add` and runtime-scalar
  `runtime_scalar_cmp_masked_macc_add` are no longer positive direct
  pre-realized route-entry consumers.
- Kept the selected-boundary producer path positive for runtime-scalar MAcc.
  The generated-bundle evidence records
  `materializer: tcrv-materialize-selected-lowering-boundaries`,
  `route_entry_realization: false`,
  `selected_body_realization_producer:
  rvv-plugin-local-selected-body-realization-owner-registry`,
  `typed_compute_op: tcrv_rvv.masked_macc`, and
  `rhs_scalar: rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr`.
- Updated C++ plugin tests so computed-mask MAcc owner registration has no
  route-entry consumer, both computed-mask MAcc selected-boundary producer
  consumers are recognized, and the runtime-scalar direct route-entry call
  fails closed as another realization family.
- Updated generated-bundle tooling so
  `--direct-pre-realized-route-entry --op-kind
  runtime_scalar_cmp_masked_macc_add` is rejected before route-entry
  materialization or generated bundle emission.
- Replaced the old positive direct route-entry lit coverage for
  `runtime_scalar_cmp_masked_macc_add` with fail-closed script coverage.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-27-05-27-stage2-rvv-runtime-scalar-macc-selected-realization`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt
  tcrv-translate -j2` passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed after one
  self-repair: the owner-scoped route-entry eligibility test initially still
  expected the computed-mask MAcc owner to expose a route-entry consumer.
- Direct route-entry script path for `runtime_scalar_cmp_masked_macc_add`
  failed closed with
  `--direct-pre-realized-route-entry is bounded to ... got
  ['runtime_scalar_cmp_masked_macc_add']`.
- Selected-boundary generated-bundle dry-run for
  `runtime_scalar_cmp_masked_macc_add` passed for counts `0,1,16,23,257` and
  RHS scalars `-37,91`.
- Selected-boundary generated-bundle dry-run for `computed_masked_macc_add`
  passed as adjacent non-regression and recorded `route_entry_realization:
  false`.
- Real `ssh rvv` generated-bundle run for
  `runtime_scalar_cmp_masked_macc_add` passed for counts `0,1,16,23,257` and
  RHS scalars `-37,91`; the harness reported
  `tcrv_rvv_generated_bundle_abi_runtime_scalar_cmp_masked_macc_add_ok`.
- Bounded touched-file authority scan found expected existing negative checks,
  diagnostics, and route/intrinsic mirrors, with no new positive runtime-scalar
  MAcc route or executable claim from direct route-entry-only,
  pre-realized-fixture-only, route-id, artifact-name, script-derived, common
  EmitC-derived, exact-intrinsic-derived, descriptor-derived, source-front-door,
  ABI-string-derived, central ad hoc, or legacy-i32 authority.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 391/391.
