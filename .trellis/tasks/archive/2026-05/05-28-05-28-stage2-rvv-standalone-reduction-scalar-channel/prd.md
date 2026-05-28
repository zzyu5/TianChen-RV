# PRD: Stage2 RVV standalone reduction scalar accumulator/result channel

## Context

The previous RVV Stage2 task completed typed route-family derivation for
`runtime_scalar_cmp_masked_standalone_reduce_add` on SEW32 LMUL m1 and SEW64
LMUL m1. It intentionally left SEW32 LMUL m2 fail-closed because the production
standalone reduction path only carried one selected vector result/store channel.

RVV SEW32 LMUL m2 standalone reduction requires two typed channels:

- source/work vector channel: SEW32 LMUL m2 for vector input, mask, runtime-scalar
  compare, inactive-neutral merge, and reduction source.
- scalar accumulator/result channel: SEW32 LMUL m1 for the reduction seed,
  reduction accumulator/result vector, scalar lane-0 result layout, and scalar
  result store.

Direct pre-realized route-entry support must remain fail-closed. The proving
path must consume a selected pre-realized typed body, realize it through the RVV
plugin-local selected-body realization, derive provider route facts, lower via
the common neutral EmitC materializer, and produce an RVV C artifact that passes
real `ssh rvv` correctness checks.

## Goal

Introduce the missing typed scalar reduction accumulator/result channel for
standalone reductions, proven by
`runtime_scalar_cmp_masked_standalone_reduce_add` SEW32 LMUL m2.

The module owner is the RVV typed selected-body realization plus route
planning/provider boundary. Production behavior must move in RVV dialect/config
contract, construction/selected-body realization, route planning/provider facts,
and generated-bundle artifact ABI boundary as needed.

## Scope

Implement one coherent production path for:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv runtime-scalar masked standalone reduce-add body
  -> RVV selected-body realization with distinct source/work vector channel and
     scalar accumulator/result channel
  -> RVV provider facts derived from typed body/config/runtime facts
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC materializer
  -> generated RVV C artifact
  -> ssh rvv correctness evidence
```

The selected body must structurally carry:

- source vector operand config, dtype, SEW, LMUL, policy, memory form, and VL.
- scalar accumulator input role and layout.
- scalar output role and layout.
- mask and runtime-scalar comparison facts.
- runtime `n` / AVL / VL relation.
- reduction op kind.
- source SEW/LMUL to scalar accumulator/result SEW/LMUL relation.
- ABI order for inputs, accumulator seed, scalar threshold, scalar output, and
  runtime length.

The provider/materialization path must derive:

- source vector type and C type from typed config/body facts.
- scalar accumulator/result vector type and C type from typed reduction channel
  facts.
- reduction intrinsic plan from typed source/result relation.
- accumulator splat and result store intrinsics from the scalar result channel.
- route facts and mirror metadata from typed facts, not from names or artifacts.

## Acceptance Criteria

1. SEW32 LMUL m2
   `runtime_scalar_cmp_masked_standalone_reduce_add` has a positive selected-body
   realization path with:
   - `route_entry_realization: false`
   - `pre_realized_body_consumed: true`
   - typed source/work channel = SEW32 LMUL m2
   - typed scalar accumulator/result channel = SEW32 LMUL m1

2. Route/provider evidence exposes distinct source and scalar
   accumulator/result channel facts, and those facts are consumed by the route
   statement plan and generated artifact boundary.

3. Generated RVV artifact compiles and runs on real `ssh rvv` for SEW32 LMUL m2,
   with correctness checks covering:
   - count 0
   - count 1
   - exact VL-sized count
   - tail count
   - stress count
   - signed data
   - at least two runtime scalar thresholds

4. Existing SEW32 LMUL m1 and SEW64 LMUL m1 standalone reduction coverage remains
   supported and non-regressed.

5. Direct pre-realized route-entry authority remains fail-closed for standalone
   reductions.

6. Fail-closed diagnostics cover unsupported or inconsistent:
   - missing accumulator role/layout.
   - missing result role/layout.
   - mismatched scalar/vector channel relation.
   - wrong SEW/LMUL relation.
   - wrong mask/runtime-scalar binding.
   - wrong AVL/VL relation.
   - stale metadata/name-derived authority.
   - attempted direct route-entry shortcut.

7. Authority scans show no new production/default RVV route authority derived
   from central ad hoc tables, route ids, artifact names, ABI strings,
   descriptors, source-front-door patterns, common EmitC inference, exact
   intrinsic spelling, direct route-entry-only paths, pre-realized fixture-only
   paths, or legacy i32 helper surfaces.

8. Run:
   - focused lit/FileCheck tests for the changed dialect, realization, route,
     artifact, and negative cases.
   - generated-bundle dry-run proving selected-boundary facts.
   - `ssh rvv` generated-artifact correctness for the proving path.
   - bounded authority-leak scan.
   - `git diff --check`.
   - `check-tianchenrv`, or record the exact blocker if it cannot complete.

## Non-goals

- Do not add new reduction operations beyond the SEW32 LMUL m2 proving path.
- Do not start widening dot/MAcc work, compare/select expansion, masked memory
  expansion, segment2, source-front-door lowering, high-level frontend lowering,
  dashboards, reports, broad smoke matrices, or evidence-only tasks.
- Do not add one-intrinsic wrapper dialects or dtype/LMUL clone batches.
- Do not re-open direct pre-realized route-entry shortcuts.
- Do not use descriptors, artifact names, route ids, ABI strings, exact
  intrinsic spellings, common EmitC, or scripts as executable authority.

## Relevant Inputs

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- archived task
  `05-28-05-28-stage2-rvv-typed-runtime-scalar-cmp-masked-standalone-reduction-route-family-derivation`

## Starting Fact

Start HEAD is `2d2d1b69`, which archived the previous typed runtime-scalar
masked standalone reduction route-family derivation task. The current task was
created because no active Trellis task existed at session start.
