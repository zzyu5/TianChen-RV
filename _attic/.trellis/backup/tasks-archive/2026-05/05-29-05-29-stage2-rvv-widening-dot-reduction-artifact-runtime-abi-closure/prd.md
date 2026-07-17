# Stage2 RVV widening dot-reduction artifact/runtime ABI closure

## Goal

Close the selected typed RVV non-computed-mask widening dot-reduction
artifact/runtime ABI path end to end for `widening_dot_reduce_add` and
`strided_input_widening_dot_reduce_add`. The production route must flow from
selected `tcrv.exec` RVV variant, through typed or pre-realized
`tcrv_rvv.widening_dot_reduce` body, RVV provider route facts, the target-owned
`widening-dot-reduction` route-family validator, generated object/header bundle
export, and real `ssh rvv` correctness evidence when runtime correctness is
claimed.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening dot-reduction artifact/runtime
  ABI closure`.
- Module owner: target-owned RVV widening dot-reduction artifact validator and
  generated-bundle runtime ABI boundary for `widening_dot_reduce_add` and
  `strided_input_widening_dot_reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f91ecd49 rvv: close widening macc runtime abi evidence`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- Commit `f91ecd49` closed the adjacent `widening_macc_add` generated-bundle
  artifact/runtime ABI path with target-owned validator consumption,
  generated-bundle dry-run evidence, real explicit and pre-realized `ssh rvv`
  evidence, and `check-tianchenrv` 456/456.
- The repo already has a generic `widening-dot-reduction` target route-family
  validator and generated-bundle support for plain, strided-input, and
  computed-mask widening dot-reduction subfamilies.
- The current task is intentionally narrower than the existing generic family:
  close only the non-computed-mask `widening_dot_reduce_add` and
  `strided_input_widening_dot_reduce_add` artifact/runtime ABI boundary. Do not
  start `computed_masked_widening_dot_reduce_add` or
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Current specs require selected RVV artifact export to rebuild provider route
  facts from the selected typed `tcrv_rvv` body and treat candidate metadata as
  mirrors after provider route construction.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles.
  They do not define RVV compute, dtype, memory form, widening dot relation, or
  route support.
- Existing inspection found that the dot-reduction validator already checks
  rebuilt headers, type mappings, ABI mappings, statement plan, typed compute
  op, dot relation/layout facts, and strided facts in a generic way. This
  round must harden the non-computed-mask plain/strided subfamily to the same
  MAcc-style exact provider/target-validator ABI closure: provider support
  label, runtime ABI order and roles, route operand binding, dtype relation,
  dot relation/layouts, source load form, strided presence/absence facts, and
  candidate mirrors.

## Requirements

1. Keep both routes on the selected typed RVV path:
   selected `tcrv.exec` RVV variant -> typed/pre-realized
   `tcrv_rvv.widening_dot_reduce` body -> RVV provider route facts ->
   target-owned `widening-dot-reduction` validator -> generated bundle.
2. The target validator must consume provider-derived runtime ABI order and
   parameter roles for:
   - plain: `lhs,rhs,acc,out,n`;
   - strided-input: `lhs,rhs,acc,out,n,lhs_stride,rhs_stride`.
3. The target validator must consume exact provider route operand binding plan
   and binding summary facts for dot lhs/rhs sources, accumulator seed, scalar
   result output, runtime `n`, and strided lhs/rhs strides where present.
4. The target validator must consume dot-product relation, accumulator/result
   layout, source/result dtype relation, source load form, setvl/VL control,
   reduction/store statement facts, required headers/type mapping, and mirror
   metadata labels.
5. Strided-input routes must fail closed when provider strided facts are
   missing or stale. Unit-stride routes must fail closed when stale strided
   facts are present.
6. Unsupported or stale ABI order, stale ABI roles, stale binding plan, wrong
   dot relation, wrong accumulator/result layout, wrong source/result dtype
   relation, wrong source load form, route-id-only support, metadata-only
   support, exact-intrinsic-only support, common-EmitC semantic choice, or
   stale non-family facts must fail closed with targeted diagnostics.
7. Generated-bundle dry-run evidence for explicit and pre-realized plain and
   strided routes must expose an explicit widening dot-reduction boundary
   summary showing provider-derived authority, target validator consumption,
   mirror-only metadata, runtime ABI facts, strided facts where applicable, and
   direct pre-realized route-entry support remaining false.
8. Real `ssh rvv` generated-bundle correctness evidence must run for plain and
   strided routes over counts `0`, `1`, exact-VL, tail, and stress cases when
   runtime/correctness is claimed.
9. Preserve common EmitC/export neutrality. Common code may materialize
   provider payloads but must not choose RVV semantics, infer dtype/config, or
   invent widening dot-reduction behavior.
10. Keep `widening_macc_add` and sibling `reduce_add` generated-bundle evidence
    non-regressed.

## Acceptance Criteria

- [x] PRD and Trellis context are truthful and started from the Hermes brief.
- [x] Production movement occurs in
      `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`; tests alone
      are not treated as the main result.
- [x] Plain and strided widening dot-reduction provider facts are accepted only
      when runtime ABI order/roles, route operand binding, provider support,
      dtype relation, dot relation/layouts, source load form, and statement
      facts are valid.
- [x] Plain unit-stride routes reject stale strided-input facts; strided-input
      routes reject missing or stale strided facts.
- [x] Candidate mirrors are checked for provider support, binding plan/summary,
      runtime ABI order, headers/type mapping, source/result dtype facts, dot
      relation/layouts, strided facts, and stale non-family facts.
- [x] Focused C++ target artifact tests cover positive and fail-closed
      behavior for both `widening_dot_reduce_add` and
      `strided_input_widening_dot_reduce_add`.
- [x] Explicit and pre-realized generated-bundle dry-run tests cover counts
      `0`, `1`, exact-VL, tail, and stress for both routes and check the new
      widening dot-reduction boundary evidence.
- [x] Real `ssh rvv` generated-bundle runs pass for both routes and both
      explicit/pre-realized inputs over counts `0`, `1`, exact-VL, tail, and
      stress, or the exact blocker is recorded.
- [x] Direct pre-realized route-entry support remains false for this family.
- [x] Bounded touched-file authority scan finds no name-, metadata-,
      descriptor-, ABI-string-, script-, artifact-name-, common-EmitC-,
      source-front-door-, route-id-, exact-intrinsic-, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived support authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Trellis task status, journal/archive state, and commit state are truthful
      at the end of the round.

## Current Round Status

- Continued the dirty in-progress task instead of creating a replacement task.
- Hardened the target-owned `widening-dot-reduction` validator so
  non-computed-mask `widening_dot_reduce_add` and
  `strided_input_widening_dot_reduce_add` require provider-derived runtime ABI
  order/roles, exact route operand binding plan/summary, provider support,
  header/type facts, source/accumulator/result SEW/LMUL, dot relation/layout,
  source load form, scalar store VL, strided fact presence or absence, and stale
  non-family fact rejection before artifact acceptance.
- Extended generated-bundle evidence with `widening_dot_reduction_boundary`,
  explicit provider route facts, target-validator consumption facts,
  mirror-only metadata labels, direct pre-realized route-entry support remaining
  false, and strided-vs-unit-stride distinguishing harness coverage.
- Added focused C++ target artifact coverage for valid plain/strided acceptance
  and stale provider support, stale operand binding, stale ABI role, stale
  source/result dtype relation, stale dot relation, stale/missing strided facts,
  stale non-family facts, and stale/missing candidate mirrors.
- Added `.trellis/spec/testing/mlir-testing-contract.md` guidance for
  widening dot-reduction generated-bundle evidence because the new boundary is
  a durable evidence contract.
- Checks passed:
  - `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - explicit and pre-realized `widening_dot_reduce_add` dry-runs for counts
    `0`, `1`, `16`, `17`, and `257`
  - explicit and pre-realized `strided_input_widening_dot_reduce_add` dry-runs
    for counts `0`, `1`, `16`, `17`, and `257`
  - focused lit filter
    `rvv-generated-bundle-abi-e2e-(explicit|pre-realized)(-strided-input)?-widening-dot-reduce-add-dry-run`
    passed 4/4
  - `ssh -o BatchMode=yes -o ConnectTimeout=8 rvv 'echo rvv-ok'`
  - explicit and pre-realized `widening_dot_reduce_add` `ssh rvv`
    generated-bundle runs for counts `0`, `1`, `16`, `17`, and `257`
  - explicit and pre-realized `strided_input_widening_dot_reduce_add`
    `ssh rvv` generated-bundle runs for counts `0`, `1`, `16`, `17`, and
    `257`
  - `widening_macc_add` and `reduce_add` dry-run non-regression for counts
    `0`, `1`, `16`, `17`, and `257`
  - bounded touched-file authority scan
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2` passed 456/456

## Out Of Scope

- `computed_masked_widening_dot_reduce_add`.
- `computed_masked_strided_input_widening_dot_reduce_add`.
- Reworking `widening_macc_add`, direct route-entry demotions, Stage1 surface
  work, Linalg/frontend lowering, new dtype/LMUL clone batches, dashboards,
  reports, broad smoke matrices, or evidence-only packaging without
  target-validator production movement.
- Reopening direct pre-realized route-entry support.

## Technical Approach

1. Harden the non-computed-mask widening dot-reduction target route-family
   validator with MAcc-style exact runtime ABI, role, binding, dtype, layout,
   dot relation, source load, strided, provider support, and candidate mirror
   checks.
2. Extend generated-bundle evidence with a plain/strided
   `widening_dot_reduction_boundary` summary and self-test guards so dry-runs
   prove provider/target-validator-backed authority instead of route names or
   metadata-only support.
3. Add focused C++ target artifact coverage for valid plain/strided acceptance
   and stale/missing provider/candidate facts failing closed.
4. Update the explicit and pre-realized plain/strided generated-bundle dry-run
   FileCheck consumers.
5. Run focused local checks, then real `ssh rvv` generated-bundle evidence over
   required counts, then non-regression, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. `build/bin/tianchenrv-rvv-extension-plugin-test`
4. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
5. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
6. Explicit and pre-realized dry-runs for
   `widening_dot_reduce_add` and `strided_input_widening_dot_reduce_add` with
   runtime counts `0`, `1`, `16`, `17`, and `257`.
7. Focused lit FileCheck for the four generated-bundle dry-run tests.
8. `ssh -o BatchMode=yes -o ConnectTimeout=8 rvv 'echo rvv-ok'`
9. Explicit and pre-realized `ssh rvv` generated-bundle runs for both routes
   with runtime counts `0`, `1`, `16`, `17`, and `257`.
10. `widening_macc_add` and `reduce_add` dry-run non-regression over the same
    counts.
11. Bounded touched-file authority scan.
12. `git diff --check`
13. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/validation/experiment-reference.md`
- `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-widening-macc-artifact-runtime-abi-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-vector-reduction-artifact-runtime-abi-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-route-family-validator-coverage-gate/prd.md`

Primary files to inspect or modify:

- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/TargetArtifactExportTest.cpp`
- Directly relevant generated-bundle dry-run tests for explicit and
  pre-realized plain/strided widening dot-reduction.
