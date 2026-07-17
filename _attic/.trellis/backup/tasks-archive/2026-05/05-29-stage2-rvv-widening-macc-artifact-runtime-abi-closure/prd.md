# Stage2 RVV widening-MAcc artifact/runtime ABI closure

## Goal

Close the selected typed RVV `widening_macc_add` artifact/runtime ABI path
end to end. The production route must flow from selected `tcrv.exec` RVV
variant, through typed widening-MAcc `tcrv_rvv` body, RVV-owned provider facts,
the target-owned `widening-macc-contraction` route-family validator, generated
object/header bundle export, and real `ssh rvv` correctness evidence.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening-MAcc artifact/runtime ABI
  closure`.
- Module owner: RVV plugin route/provider facts plus target artifact export
  and generated-bundle boundary for the selected typed `widening_macc_add`
  route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1041a27d rvv: close vector reduction runtime abi evidence`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- Commit `1041a27d` closed the sibling `reduce_add` path with validator-backed
  generated-bundle dry-run and real `ssh rvv` evidence for counts `0`, `1`,
  `16`, `17`, and `257`.
- The prior route-family validator coverage task added a target-owned
  `widening-macc-contraction` validator for `widening_macc_add`, but that
  path still needs current artifact/runtime ABI closure evidence.
- Specs require route and artifact authority to come from the selected typed
  `tcrv_rvv` body, RVV provider facts, rebuilt `TCRVEmitCLowerableRoute`, and
  target-owned family validator. Route ids, artifact names, ABI strings,
  exact intrinsic spellings, scripts, direct route entries, and metadata are
  mirrors or diagnostics only.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles.
  They do not define RVV compute, dtype, memory form, widening relation, or
  route support.
- `widening_macc_add` is a direct-provider contraction route. The provider must
  consume contraction family/materialization/math facts, route-control facts,
  and the direct contraction provider plan before constructing
  `TCRVEmitCLowerableRoute`; statement construction must not rediscover facts
  from names, metadata, ABI strings, or intrinsic mirrors.

## Requirements

1. Keep `widening_macc_add` on the selected typed RVV path:
   selected `tcrv.exec` RVV variant -> typed widening-MAcc `tcrv_rvv` body ->
   RVV provider route facts -> target-owned `widening-macc-contraction`
   validator -> generated bundle.
2. Generated-bundle dry-run evidence must explicitly show provider-derived
   widening-MAcc typed body/config/runtime facts and target-owned validator
   consumption.
3. Evidence and validation must preserve operation kind, unit-stride memory
   form, source operand roles, configured source dtype, configured
   accumulator/result dtype, widening relation, accumulator update semantics,
   runtime `n`/AVL/VL relation, setvl placement, tail/mask policy, route
   operand binding, runtime ABI order, artifact ABI order, and provider support
   mirror labels.
4. Missing or stale provider facts, target-validator facts, source/result dtype
   relation, accumulator/result binding, widening relation, ABI order,
   AVL/VL relation, setvl placement, mirror metadata, stale non-widening-MAcc
   facts, or direct/pre-realized-only authority must fail closed before
   artifact acceptance.
5. Preserve common EmitC/export neutrality. Common code may materialize
   provider payloads but must not choose RVV semantics, infer dtype/config, or
   invent widening-MAcc behavior.
6. Run real `ssh rvv` generated-bundle correctness evidence for counts
   including `0`, `1`, exact-VL, tail, and stress cases, with signed source
   patterns, nonzero accumulator/seed inputs, widened accumulator expected
   results, and sentinel preservation where applicable.
7. Keep `reduce_add` validator-backed generated-bundle evidence from the
   previous task non-regressed.
8. Keep direct pre-realized route-entry support false for this family unless a
   later task explicitly changes that architecture.

## Acceptance Criteria

- [x] PRD and Trellis context are truthful and started from the Hermes brief.
- [x] Production or directly adjacent generated-bundle/tooling behavior closes
      `widening_macc_add` authority exposure; this is not only a broad smoke
      test or report.
- [x] `widening_macc_add` generated-bundle dry-run proves provider-built
      widening-MAcc facts and target-owned validator consumption.
- [x] Dry-run output labels provider/candidate metadata as mirrors and exposes
      an explicit widening-MAcc boundary summary.
- [x] Focused C++ or lit coverage proves stale/missing widening-MAcc provider
      facts or candidate mirrors fail closed before artifact acceptance.
- [x] Direct route-entry support remains false for this family.
- [x] Real `ssh rvv` generated-bundle run passes for counts `0`, `1`,
      exact-VL, tail, and stress with signed source patterns, nonzero
      accumulator inputs, widened expected results, and sentinel preservation
      where applicable. Evidence:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-widening-macc-explicit-ssh-rvv`
      and
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-widening-macc-prerealized-ssh-rvv`.
- [x] Sibling `reduce_add` generated-bundle evidence remains non-regressed.
- [x] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      support authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Trellis task status, journal/archive state, and commit state are
      truthful at the end of the round.

## Current Round Status

- Continued the dirty in-progress task rather than creating a replacement
  task.
- Hardened the target-owned `widening-macc-contraction` validator to require
  provider-derived runtime ABI order/roles, route operand binding facts,
  contraction support mirrors, header/type facts, and the exact widening-MAcc
  accumulator/result/relation facts before artifact acceptance.
- Extended generated-bundle evidence with `widening_macc_boundary`, explicit
  provider/target-validator fact summaries, mirror-only metadata labels, and
  direct pre-realized route-entry support remaining false.
- Strengthened the harness so multi-lane cases must include signed positive
  and negative products, nonzero accumulators, tail sentinel preservation, and
  products outside the source-width range. The continuation self-repair fixed
  the pre-realized input pattern after real `ssh rvv` evidence initially failed
  because it did not produce widening products for `n=16`.
- Added C++ target artifact coverage that accepts the valid selected-body
  widening-MAcc candidate and rejects stale provider support, stale runtime ABI
  order/role, stale operand binding, stale source/result dtype relation, stale
  widening relation, stale non-family facts, and stale/missing candidate
  mirrors.
- Added `.trellis/spec/testing/mlir-testing-contract.md` guidance for
  widening-MAcc generated-bundle evidence because the pre-realized harness
  coverage failure exposed a durable evidence-quality requirement.
- Checks passed:
  - `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - explicit and pre-realized `widening_macc_add` dry-runs for counts `0`,
    `1`, `16`, `17`, and `257`
  - focused lit filter
    `rvv-generated-bundle-abi-e2e-(explicit|pre-realized)-widening-macc-add-dry-run`
    passed 2/2
  - `ssh -o BatchMode=yes -o ConnectTimeout=8 rvv 'echo rvv-ok'`
  - explicit and pre-realized `widening_macc_add` `ssh rvv` generated-bundle
    runs for counts `0`, `1`, `16`, `17`, and `257`
  - `reduce_add` dry-run non-regression for counts `0`, `1`, `16`, `17`,
    and `257`
  - evidence JSON assertion for authority, mirror-only metadata,
    target-validator consumption, ABI order, widening relation, and direct
    pre-realized route-entry remaining false
  - bounded touched-file authority scan
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2` passed 456/456

## Out Of Scope

- `widening_dot_reduce_add`, computed-mask widening-dot, strided-input
  widening-dot, and any new contraction family coverage.
- Conversions beyond what `widening_macc_add` already needs.
- Standalone reductions, vector reductions, segment2, compare/select, dtype or
  LMUL clone batches, high-level Linalg/frontend lowering, and one-intrinsic
  wrappers.
- Dashboards, reports, broad smoke matrices, or helper-only evidence that is
  not consumed by the production or directly adjacent generated-bundle path.
- Reopening direct pre-realized route-entry support.

## Technical Approach

1. Inspect the current `widening_macc_add` selected-body route/provider facts,
   direct contraction provider plan, target validator, generated-bundle script,
   and directly relevant tests.
2. Identify the narrow missing boundary: likely generated-bundle evidence
   summaries, dry-run assertions, runtime harness ABI checks, or focused target
   validator negative coverage.
3. Implement only the bounded production or directly adjacent behavior needed
   for the evidence path to prove provider/target-validator-backed authority.
4. Run focused local checks first, including generated-bundle dry-run for
   `widening_macc_add`.
5. Run real `ssh rvv` generated-bundle correctness evidence over required
   runtime counts.
6. Run sibling `reduce_add` non-regression, bounded authority scans,
   `git diff --check`, and `check-tianchenrv` if focused checks pass.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. `build/bin/tianchenrv-rvv-extension-plugin-test`
4. `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind widening_macc_add --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
5. `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
6. `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
7. `git diff --check`
8. Bounded authority scan over touched RVV planning/provider/target/script/test
   files.
9. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/validation/experiment-reference.md`
- `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-vector-reduction-artifact-runtime-abi-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-route-family-validator-coverage-gate/prd.md`

Primary files to inspect or modify:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Directly relevant widening-MAcc tests only as consumers/evidence.
