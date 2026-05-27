# Stage2 RVV MAcc Selected-Body Realization Production Migration

## Goal

Move the production `computed_masked_macc_add` generated-artifact path behind
the RVV plugin-local selected-body realization producer, and demote the direct
pre-realized route-entry shortcut for this consumer.

The bounded production chain for this round is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv computed_masked_macc_add body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv setvl / with_vl / compare loads / payload loads
     / accumulator load / compare / masked_macc / store body
  -> computed-mask MAcc route-family provider facts
  -> RVV route materialization facts
  -> RVV math operand-binding facts
  -> RVV route-control provider plan
  -> computed-mask accumulation statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

`computed_masked_macc_add` support must come from realized typed
`tcrv_rvv.masked_macc` body facts consumed by the existing computed-mask MAcc
route family. It must not be claimed from direct route-entry acceptance,
pre-realized fixture names, route ids, artifact names, script options, ABI
strings, exact intrinsic spelling, common EmitC behavior, descriptor residue,
source-front-door metadata, or legacy i32 helper authority.

## Direction Source

- Direction title: `Expand: Stage2 RVV MAcc/accumulator selected-body realization family`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d634a09f rvv: demote scalar broadcast route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- `computed_masked_macc_add` already has a selected-boundary fixture at
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-macc-add.mlir`.
  The selected-boundary path materializes lowering boundaries before emission
  plans and target artifact export.
- `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --op-kind computed_masked_macc_add` already uses
  `--tcrv-materialize-selected-lowering-boundaries` and records the computed
  mask MAcc boundary facts in evidence.
- The same operation is still direct route-entry eligible through
  `isPreRealizedRVVComputedMaskMAccRouteEntryOp(...)`, and the generated-bundle
  script/lit coverage has a positive direct route-entry test for
  `computed_masked_macc_add`. This is the stale shortcut to demote.
- The RVV selected-body realization owner registry already has a
  `computed-mask MAcc` owner. Its realization hook validates runtime ABI roles,
  mask provenance, accumulator layout, SEW/LMUL/policy, selected requires, and
  realized-body exclusivity, then materializes `setvl`, `with_vl`, loads,
  compare, `masked_macc`, and store.
- Existing planning/provider code already exposes the computed-mask
  accumulation family plan, route materialization facts, math operand-binding
  facts, route-control provider plan, and computed-mask accumulation
  statement plan. Common EmitC and target artifacts must remain consumers of
  provider-built facts only.
- `runtime_scalar_cmp_masked_macc_add` is an adjacent computed-mask MAcc
  sub-family. This round is centered on the vector-compare
  `computed_masked_macc_add` active consumer and should not broaden into a
  coverage batch unless a small coherence update is required by the shared
  predicate or script guardrail.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling and script
   guardrails.
2. Keep the selected-boundary `computed_masked_macc_add` producer path
   positive. It must realize the typed pre-realized body into `setvl`,
   `with_vl`, compare loads, payload loads, accumulator load, compare,
   `masked_macc`, and store before provider route facts are collected.
3. Demote or delete the `computed_masked_macc_add` direct pre-realized
   route-entry shortcut. The direct shortcut must fail closed with a targeted
   diagnostic or script error instead of producing a generated artifact.
4. Preserve unrelated supported direct route-entry families unless their
   behavior is directly affected by the computed-mask MAcc predicate change.
5. Preserve typed facts for operation kind, accumulator layout, multiplicand
   and addend binding, compare-produced mask provenance, inactive-lane
   passthrough semantics, memory roles, runtime `n`/AVL/VL values,
   dtype/config/policy, setvl placement, loop relation, required RVV
   capability, provider route facts, and artifact ABI order.
6. Unsupported or inconsistent selected-boundary input must fail closed before
   provider/common route construction where exposed by current hooks: missing
   runtime param, missing mem_window/runtime ABI value, wrong accumulator
   binding, wrong multiplicand/addend binding, wrong mask binding,
   dtype/config/policy mismatch, wrong AVL/VL relation, wrong setvl placement,
   missing capability, stale route id, stale mirror metadata, direct-route
   entry-only authority, artifact-name/script-derived authority,
   exact-intrinsic-as-authority, and common-EmitC semantic invention.
7. Do not add broad route coverage, new dtype/LMUL clones, high-level
   Linalg/frontend lowering, one-intrinsic wrapper dialects, selected-body
   framework rewrites, descriptor/source-front-door positive routes,
   dashboards, mask/tail polish, VL/control polish, conversion/reduction/
   segment/base-memory polish, gather/scatter expansion, or proof-only helper
   work as the main achievement.

## Acceptance Criteria

- [x] Production code demotes `computed_masked_macc_add` direct pre-realized
      route-entry eligibility while keeping the selected-boundary producer path
      positive.
- [x] C++ tests prove `computed_masked_macc_add` is not route-entry eligible
      for direct route-entry realization, and that the selected-body
      realization producer still realizes it before provider route
      construction.
- [x] C++ or lit tests prove provider route facts for selected-boundary
      `computed_masked_macc_add` consume realized typed body facts and include
      computed-mask accumulation route-family plan, runtime control plan, math
      operand-binding facts, statement plan, ABI order, mask provenance,
      accumulator layout, and provider-supported mirror.
- [x] Focused fail-closed coverage rejects wrong mask provenance, wrong
      accumulator layout, wrong result layout, dtype/config/policy mismatch,
      wrong runtime `n` binding, stale op kind, and direct-route-entry-only
      authority.
- [x] The direct pre-realized route-entry script path for
      `computed_masked_macc_add` fails closed or is removed from positive lit
      coverage.
- [x] Generated-bundle dry-run for `computed_masked_macc_add` passes through
      the selected lowering-boundary path and records no direct route-entry
      materializer.
- [x] Real `ssh rvv` generated-bundle execution covers counts including `0`,
      `1`, exact-VL, tail, and stress cases with at least two accumulator/input
      value patterns.
- [x] Focused non-regression covers completed selected-body producer paths for
      `computed_mask_select`, `runtime_i32_splat_store`,
      `runtime_scalar_cmp_masked_load_store`, and `scalar_broadcast_add`, plus
      adjacent MAcc route-family paths changed or risked by this demotion.
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
   `computed_masked_macc_add` uses the selected-boundary producer path only.
3. Update generated-bundle tooling and script tests so
   `--direct-pre-realized-route-entry --op-kind computed_masked_macc_add` no
   longer succeeds.
4. Keep or strengthen selected-boundary lit coverage for
   `pre-realized-selected-body-artifact-computed-masked-macc-add.mlir` and the
   generated-bundle pre-realized dry-run.
5. Run focused C++ plugin tests and focused lit/script tests.
6. Run generated-bundle selected-boundary dry-run for
   `computed_masked_macc_add`.
7. Run real `ssh rvv` generated-bundle evidence for counts `0,1,16,23,257`
   and the existing MAcc value-pattern harness.
8. Run focused non-regression for `computed_mask_select`,
   `runtime_i32_splat_store`, `runtime_scalar_cmp_masked_load_store`,
   `scalar_broadcast_add`, `macc_add`, `scalar_broadcast_macc_add`, and
   adjacent computed-mask MAcc/runtime-scalar MAcc fixtures as needed.
9. Run authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Stage2 coverage expansion beyond the bounded MAcc/accumulator selected-body
  migration for `computed_masked_macc_add`.
- New MAcc dtype/LMUL clone batches, new accumulator helper op families, or
  additional intrinsic wrappers.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door, descriptor-driven compute, direct C/source-export paths,
  or legacy `RVVI32M1` / `rvv-i32m1` compatibility.
- Dashboard/report-only/prompt-only work as the main achievement.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant archived task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-scalar-broadcast-selected-realization-migration`.
- Primary production files to inspect or modify:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and focused target/script tests.

## Implementation Result

- `computed_masked_macc_add` vector pre-realized bodies are no longer accepted
  by `isPreRealizedRVVComputedMaskMAccRouteEntryOp(...)`; the adjacent
  runtime-scalar computed-mask MAcc route-entry family remains eligible.
- Generated-bundle tooling no longer allows
  `--direct-pre-realized-route-entry --op-kind computed_masked_macc_add`; that
  request now fails closed before route-entry materialization or bundle
  generation.
- The selected-boundary generated-bundle path for
  `computed_masked_macc_add` remains positive and records
  `route_entry_realization: false`.
- C++ plugin coverage now treats `rvv_pre_route_computed_masked_macc_add` as a
  selected-boundary producer consumer of the computed-mask accumulation route
  family, not as a direct route-entry shortcut.
- Focused script coverage replaces the previous positive direct route-entry
  dry-run with a fail-closed test.
- No `.trellis/spec` update was needed because this round applied existing
  Stage2 selected-body realization and MAcc authority rules without adding a
  new durable rule.

## Validation Evidence

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Selected-boundary dry-run:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257`
  produced `dry_run_success` under
  `artifacts/tmp/stage2_rvv_macc_selected_realization/selected-boundary-computed-masked-macc-add-dry`.
- Direct route-entry fail-closed probe for
  `--direct-pre-realized-route-entry --op-kind computed_masked_macc_add`
  rejected the request with `got ['computed_masked_macc_add']` before route
  materialization.
- Focused lit non-regression from `build/test` with filter
  `computed-mask-select|runtime-i32-splat-store|runtime-scalar-computed-mask-load-store|scalar-broadcast-add|computed-masked-macc-add|direct-pre-realized-runtime-scalar-cmp-masked-macc-add`
  passed 22/22.
- Real `ssh rvv` generated-bundle run for `computed_masked_macc_add` passed
  counts `0,1,16,23,257`, preserving inactive accumulator/tail lanes and
  reporting:
  `tcrv_rvv_generated_bundle_abi_computed_masked_macc_add_ok counts=0,1,16,23,257`.
- Bounded touched-file authority scan over the diff found no new added route or
  executable claim depending on descriptor, source-export/front-door,
  route-id, artifact-name, script-derived, metadata-derived, common-EmitC,
  exact-intrinsic, direct-route-entry-only, pre-realized-fixture-only, or
  legacy-i32 authority.
- `git diff --check`
- Single-instance `check-tianchenrv`: 391/391.
