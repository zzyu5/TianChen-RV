# Stage2 RVV direct route-entry contraction executable boundary

## Goal

Close the executable boundary for one representative direct route-entry RVV
contraction workflow:
`computed_masked_strided_input_widening_dot_reduce_add`.

The workflow must start from a selected `tcrv.exec` RVV variant containing the
pre-realized contraction body, use the production RVV route-entry realization
consumer added by the previous task, feed the consolidated direct contraction
provider plan, materialize through the common EmitC route, generate the RVV
artifact and harness, and produce real `ssh rvv` correctness evidence.

This task exists because the previous task proved direct route-entry
realization reachability but did not rerun real RVV execution for that newly
reachable route-entry workflow.

## Direction Source

- Direction title: `Stage2 RVV direct route-entry contraction executable boundary`.
- Module owner: direct route-entry executable workflow for pre-realized RVV
  contraction selected bodies.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d66d96dd rvv: route pre-realized contractions through owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## Current Repository Facts

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-selected-body-contraction-realization-owner`
  made the `contraction` selected-body realization owner route-entry-capable.
- That previous task added direct route-entry C++ coverage for
  `widening_dot_reduce_add`, `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- The previous task's direct route-entry generated-bundle evidence was dry-run
  only. It explicitly did not rerun `ssh rvv` because emitted statements were
  believed unchanged.
- Earlier archived sessions already produced non-route-entry and
  materialize-boundary-first `ssh rvv` evidence for computed-mask strided
  widening dot-reduce. This task must not reuse those artifacts as proof for
  the direct route-entry path.
- The relevant long-term specs require:
  - selected-body realization before provider route facts are collected;
  - route-entry support through the RVV realization owner registry;
  - direct contraction provider-plan consumption before
    `TCRVEmitCLowerableRoute` construction;
  - common EmitC/export neutrality;
  - real `ssh rvv` evidence for executable runtime/correctness claims.

## Requirements

1. Exercise the production direct route-entry workflow for
   `computed_masked_strided_input_widening_dot_reduce_add` from selected
   `tcrv.exec` RVV variant through generated RVV artifact and harness.
2. The route-entry workflow must use the production owner code:
   - RVV selected-body route-entry realization bridge;
   - contraction realization owner;
   - realized typed low-level `tcrv_rvv` body;
   - consolidated direct contraction provider plan;
   - `TCRVEmitCLowerableRoute`;
   - common EmitC materializer.
3. The generated artifact evidence must prove the production path carries:
   - compare predicate and same-VL compare-produced mask source;
   - runtime `lhs_stride` and `rhs_stride` ABI values;
   - signed `i16mf2` by `i16mf2` widening relation into an `i32`
     accumulator/result;
   - accumulator seed and scalar `i32 out[0]` result layout;
   - runtime `n` / AVL / VL flow;
   - tail and mask policy;
   - selected capability and realized `tcrv_rvv` body;
   - provider route facts, provider-supported mirrors, header/type/intrinsic
     mirrors, and artifact ABI mirrors after provider route construction.
4. `ssh rvv` execution must run the direct route-entry generated bundle over
   counts including zero, small, exact/vector-like, and tail/stress cases.
5. The harness/oracle must distinguish true masked signed strided widening
   dot-reduce from:
   - unit-stride fallback;
   - unmasked reduction;
   - wrong predicate;
   - wrong stride;
   - add-only;
   - multiply-only or no accumulator seed;
   - non-widening or wrong sign extension;
   - vector-store instead of scalar `out[0]`;
   - inactive-lane contribution;
   - sentinel clobbering.
6. Preserve focused non-regression for plain and computed-mask direct
   route-entry contraction paths plus recent widening-dot-reduce executable
   boundaries.
7. If the direct route-entry artifact/runtime plumbing already works without
   production edits, keep production code unchanged or minimal and record the
   exact production consumers that satisfy the path.
8. If the direct route-entry artifact/runtime plumbing fails, fix the active
   production blocker. Do not add wrapper tests, fallback fixtures, or
   metadata-only evidence to hide the failure.
9. Do not introduce route-entry support authority from scripts, test names,
   route ids, artifact names, ABI strings, descriptors, comments, mirror
   metadata, common EmitC, source-front-door fixtures, or legacy i32 helpers.

## Acceptance Criteria

- [ ] A direct route-entry dry-run for representative
      `computed_masked_strided_input_widening_dot_reduce_add` succeeds using
      `--direct-pre-realized-route-entry`, not the older explicit or
      materialize-boundary-first fixture path.
- [ ] The dry-run/generated artifact evidence shows the pre-realized body was
      consumed into realized `tcrv_rvv` structure and that the provider path
      reached the consolidated contraction route-provider plan before common
      EmitC materialization.
- [ ] The generated bundle compiles and runs on `ssh rvv` for runtime counts
      including `0`, a small count, an exact/vector-like count, and at least one
      tail/stress count.
- [ ] The correctness oracle distinguishes masked signed strided widening
      dot-reduce from the listed negative behavior classes.
- [ ] Focused non-regression covers plain direct route-entry contraction and
      computed-mask direct route-entry contraction paths.
- [ ] Focused non-regression covers recent widening-dot-reduce executable
      boundaries that this workflow depends on.
- [ ] If runtime/artifact plumbing is modified, add or update a focused
      fail-closed route-entry check for malformed or unsupported route-entry
      input.
- [ ] Bounded touched-file authority scan finds no new positive authority from
      legacy i32, source-front-door, descriptor, ABI-string, script,
      artifact-name, common-EmitC, metadata/status, route-id, or test-name
      sources.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

- No new contraction variant or dtype/LMUL clone batch.
- No high-level frontend, Linalg, Vector, StableHLO, source-front-door positive
  route, dashboard, broad smoke matrix, report-only work, or helper-only task.
- No common EmitC RVV semantics.
- No weakening of the selected-body realization owner registry or consolidated
  direct contraction provider plan.
- No use of older explicit/pre-realized non-route-entry artifacts as proof for
  the direct route-entry executable claim.

## Technical Approach

1. Validate/start the Trellis task and inspect the production files and tests
   named by the brief.
2. Reproduce the direct route-entry generated-bundle dry-run for
   `computed_masked_strided_input_widening_dot_reduce_add`.
3. Inspect the dry-run output and generated files for route-entry evidence:
   realization consumption, realized `tcrv_rvv` facts, provider route facts,
   route-control/direct-contraction provider facts, ABI/header/type/intrinsic
   mirrors, and harness oracle checks.
4. Run the generated bundle on `ssh rvv` for counts covering `0`, small,
   exact/vector-like, and tail/stress cases.
5. If evidence is missing, fix the production owner path or generated-bundle
   artifact plumbing that lost the production facts.
6. Run focused non-regression for plain and computed-mask direct route-entry
   contraction, plus recent widening-dot-reduce executable boundaries.
7. Run focused tests, authority scan, `git diff --check`, and
   `check-tianchenrv`.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-direct-route-entry-contraction-executable-boundary`
2. Focused dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_direct_route_entry_contraction_executable_boundary --run-id direct-route-entry-computed-mask-strided-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
3. Focused `ssh rvv` generated-bundle run for the same direct route-entry path
   and counts.
4. Focused non-regression dry-runs or tests for plain and computed-mask direct
   route-entry contraction paths.
5. Focused non-regression for recent widening-dot-reduce executable boundaries.
6. Focused C++/lit/script tests touched by any code change.
7. Bounded authority scan over touched production/test/script/spec files.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived task read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-selected-body-contraction-realization-owner/task.json`,
  `prd.md`, `implement.jsonl`, and `check.jsonl`.
- Workspace journal entries read:
  session 237 for computed-mask strided widening dot-reduce evidence,
  session 238 for direct contraction provider-plan consolidation, and
  session 239 for route-entry contraction realization ownership.
- Initial production/test surfaces to inspect:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and directly related
  direct-route-entry/contraction tests under `test/Scripts` and
  `test/Target/RVV`.

## Definition Of Done

The representative direct route-entry
`computed_masked_strided_input_widening_dot_reduce_add` workflow is evidenced
end-to-end from pre-realized selected body through production RVV realization,
direct contraction provider route construction, common EmitC materialization,
generated artifact/harness, and real `ssh rvv` correctness. The task remains
bounded, no non-authoritative route source is introduced, focused regressions
and full checks pass or have exact blockers, Trellis state is truthful, and a
single coherent commit records the completed work.

## Implementation Summary

- No production compiler, script, test, or spec code change was needed. Current
  HEAD already carried the production route-entry workflow added by
  `d66d96dd`.
- Created this Trellis task/PRD/context and validated that the active
  production consumers are:
  - RVV selected-body route-entry realization bridge;
  - `contraction` selected-body realization owner;
  - realized typed `tcrv_rvv` body;
  - consolidated direct contraction route provider plan;
  - `TCRVEmitCLowerableRoute`;
  - common EmitC materialization and target artifact export.
- Generated direct route-entry artifact evidence for
  `computed_masked_strided_input_widening_dot_reduce_add` at:
  `artifacts/tmp/stage2_rvv_direct_route_entry_contraction_executable_boundary/direct-route-entry-computed-mask-strided-ssh`.
- The generated harness oracle checks masked signed strided widening
  dot-reduce against scalar expected output, inactive nonzero products,
  runtime `lhs_stride=2` and `rhs_stride=3`, active positive and negative
  products, seed addition, scalar `out[0]` output, and sentinel preservation.

## Verification Results

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-direct-route-entry-contraction-executable-boundary`
  passed before implementation checks.
- Direct route-entry dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_direct_route_entry_contraction_executable_boundary --run-id direct-route-entry-computed-mask-strided-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`.
- Real `ssh rvv` generated-bundle execution passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --artifact-root artifacts/tmp/stage2_rvv_direct_route_entry_contraction_executable_boundary --run-id direct-route-entry-computed-mask-strided-ssh --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`.
- `ssh rvv` result:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,7,16,23,257 lhs_stride=2 rhs_stride=3`.
- The per-op evidence records:
  - `local_bundle_generation.materializer =
    rvv-route-entry-selected-body-realization`;
  - `route_entry_realization = true`;
  - `pre_realized_body_consumed = true`;
  - realized `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, unit compare loads,
    `tcrv_rvv.strided_load` for lhs/rhs, `tcrv_rvv.compare {kind = "slt"}`,
    `tcrv_rvv.masked_widening_dot_reduce`, and `tcrv_rvv.store`;
  - `tcrv_rvv.route_operand_binding_plan =
    rvv-route-operand-binding:masked_strided_wdot.v1`;
  - `tcrv_rvv.contraction_route_family_plan =
    rvv-contraction-route-family-plan.v1`;
  - `tcrv_rvv.provider_supported_mirror =
    provider_supported_mirror:rvv-contraction-family-plan-validated`;
  - runtime ABI order
    `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`;
  - `tcrv_rvv.mask_source =
    compare-produced-mask-same-vl-scope`;
  - `tcrv_rvv.widening_dot_relation =
    signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32`;
  - header prototype with `const int16_t *lhs`, `const int16_t *rhs`,
    `const int32_t *acc`, `int32_t *out`, `size_t n`,
    `size_t lhs_stride`, and `size_t rhs_stride`.
- Remote compile evidence used `ssh rvv`, `remote_arch=riscv64`,
  `/usr/bin/clang`, and `clang -O2 -march=rv64gcv -mabi=lp64d`.
- Direct route-entry plain/computed-mask non-regression dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind widening_dot_reduce_add --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_direct_route_entry_contraction_executable_boundary --run-id direct-route-entry-plain-computed-mask-nonregression --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`.
- Explicit selected-body widening-dot non-regression dry-run passed for
  `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Pre-realized selected-body materialize-boundary-first widening-dot
  non-regression dry-run passed for the same four op kinds.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Final `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-direct-route-entry-contraction-executable-boundary`
  passed.
- `git diff --check` passed.
- Bounded authority scan over non-task diffs found no changed
  production/test/script/spec files and no added legacy-i32,
  source-front-door, descriptor, ABI-string, script, artifact-name,
  common-EmitC, metadata/status, route-id, or test-name authority source.
- `cmake --build build --target check-tianchenrv -j2` passed with
  381/381 tests.
- Spec-update judgment: no `.trellis/spec/**` edit was needed. This round did
  not introduce a new executable contract; it provided the missing runtime
  evidence for a route-entry contract already encoded in the RVV plugin and
  EmitC route specs.
