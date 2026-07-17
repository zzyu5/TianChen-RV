# Stage2 RVV computed-mask widening dot-reduction runtime ABI closure

## Goal

Close the selected typed RVV computed-mask widening dot-reduction artifact and
runtime ABI path end to end for `computed_masked_widening_dot_reduce_add` and
`computed_masked_strided_input_widening_dot_reduce_add`. The production route
must flow from selected `tcrv.exec` RVV variant, through typed or pre-realized
`tcrv_rvv.masked_widening_dot_reduce` body, RVV provider route facts, the
target-owned `widening-dot-reduction` route-family validator, generated
object/header bundle export, and real `ssh rvv` correctness evidence when
runtime correctness is claimed.

## Direction Source

- Direction title: `Expand: Stage2 RVV computed-mask widening dot-reduction
  artifact runtime ABI closure`.
- Module owner: target-owned RVV computed-mask widening dot-reduction runtime
  ABI/artifact validation plus generated-bundle evidence for
  `computed_masked_widening_dot_reduce_add` and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `9c351944 rvv: close widening dot reduction runtime abi
  evidence`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- Commit `9c351944` closed the adjacent non-computed
  `widening_dot_reduce_add` and `strided_input_widening_dot_reduce_add`
  runtime ABI/artifact path with target-owned validator facts,
  generated-bundle ABI evidence, explicit/pre-realized dry-runs, and real
  `ssh rvv` correctness.
- The current specs require RVV artifact export to rebuild provider route
  facts from the selected typed `tcrv_rvv` body and treat candidate metadata
  as mirrors after provider route construction.
- The widening dot-reduction validator already has a generic computed-mask
  presence check, and `scripts/rvv_generated_bundle_abi_e2e.py` already has a
  `computed_masked_widening_dot_reduce_boundary`. The gap for this round is
  stricter target-owned consumption and evidence parity with the just-closed
  non-computed path.
- The existing computed-mask dry-run tests cover the two operation kinds, but
  their runtime counts are currently narrower than this task's required
  `0`, `1`, exact-VL, tail, and stress set.
- Current `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime
  roles only. They must not define RVV compute, dtype, mask semantics, dot
  relation, route support, or artifact authority.

## Requirements

1. Keep both routes on the selected typed RVV path:
   selected `tcrv.exec` RVV variant -> typed/pre-realized
   `tcrv_rvv.masked_widening_dot_reduce` body -> RVV provider route facts ->
   target-owned `widening-dot-reduction` validator -> generated bundle.
2. The target validator must consume provider-derived runtime ABI order and
   parameter roles for:
   - unit-stride computed-mask: `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`;
   - strided-input computed-mask:
     `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`.
3. The target validator must consume exact provider route operand binding plan
   and binding summary facts for compare lhs/rhs, dot lhs/rhs, accumulator seed,
   scalar result output, runtime `n`, and strided lhs/rhs strides where present.
4. The target validator must consume computed-mask facts: mask role, mask
   source, mask memory form, predicate kind, compare intrinsic, compare operand
   binding, masked widening product, masked merge, inactive-lane zeroing
   requirement, mask type, and mask/tail policy mirrors.
5. The target validator must consume widening dot facts: source dtype and LMUL,
   accumulator/result dtype and LMUL, widening i16-to-i32 relation, dot
   relation/layout, scalar store VL, source/result memory form, headers/type
   mapping, route-control provider facts, and provider support mirror.
6. Strided-input computed-mask routes must fail closed when provider strided
   facts are missing or stale. Unit-stride computed-mask routes must fail
   closed when stale strided facts are present.
7. Unsupported or stale ABI order, stale ABI roles, stale binding plan, wrong
   mask facts, wrong dtype/LMUL relation, wrong dot relation, wrong
   accumulator/result layout, wrong source load form, stale non-family facts,
   route-id-only support, metadata-only support, exact-intrinsic-only support,
   script-derived support, or common-EmitC semantic choice must fail closed
   with targeted diagnostics.
8. Generated-bundle dry-run evidence for explicit and pre-realized unit-stride
   and strided computed-mask widening dot routes must expose
   `computed_masked_widening_dot_reduce_boundary` with provider-derived
   authority, target-validator consumption, mirror-only metadata, runtime ABI
   order, route operand binding, mask/dtype/layout facts, strided facts where
   applicable, and direct pre-realized route-entry support remaining false.
9. Real `ssh rvv` generated-bundle correctness evidence must run for both
   operation kinds and both explicit/pre-realized inputs over counts `0`, `1`,
   exact-VL, tail, and stress cases when runtime/correctness is claimed.
10. Preserve common EmitC/export neutrality. Common code may materialize
    provider payloads but must not choose RVV semantics, infer dtype/config, or
    invent computed-mask widening dot behavior.
11. Keep the four non-computed widening dot routes non-regressed.

## Acceptance Criteria

- [ ] PRD and Trellis context are truthful and started from the Hermes brief.
- [ ] Production movement occurs in the target-owned widening dot-reduction
      artifact validator; tests or script evidence alone are not treated as the
      main result.
- [ ] `computed_masked_widening_dot_reduce_add` and
      `computed_masked_strided_input_widening_dot_reduce_add` are accepted only
      when provider-derived runtime ABI order/roles, route operand binding,
      provider support, mask facts, dtype/LMUL relation, dot relation/layouts,
      source load form, and statement facts are valid.
- [ ] Unit-stride computed-mask routes reject stale strided-input facts;
      strided-input computed-mask routes reject missing or stale strided facts.
- [ ] Candidate mirrors are checked for provider support, binding plan/summary,
      runtime ABI order, headers/type mapping, mask facts, source/result dtype
      facts, dot relation/layouts, strided facts, and stale non-family facts.
- [ ] Focused C++ target artifact tests cover positive and fail-closed behavior
      for both computed-mask widening dot routes.
- [ ] Explicit and pre-realized generated-bundle dry-run tests cover counts
      `0`, `1`, exact-VL, tail, and stress for both routes and check the
      computed-mask widening dot boundary evidence.
- [ ] Real `ssh rvv` generated-bundle runs pass for both routes and both
      explicit/pre-realized inputs over counts `0`, `1`, exact-VL, tail, and
      stress, or the exact blocker is recorded.
- [ ] Direct pre-realized route-entry support remains false for this family.
- [ ] Non-computed `widening_dot_reduce_add` and
      `strided_input_widening_dot_reduce_add` explicit/pre-realized dry-runs
      remain passing.
- [ ] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      support authority.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] Trellis task status, journal/archive state, and commit state are truthful
      at the end of the round.

## Out Of Scope

- Reopening already closed non-computed widening dot validator behavior except
  shared factoring directly needed for computed-mask closure.
- Reopening direct pre-realized route-entry support.
- Segment2, compare/select, widening conversion, MAcc, standalone reduction,
  high-level frontend/Linalg work, dashboards, broad smoke matrices, and
  evidence-only bookkeeping without production validator movement.
- Moving RVV semantics into common EmitC/export, route ids, artifact metadata,
  scripts, descriptors, source-front-door paths, or exact intrinsic spellings.

## Technical Approach

1. Harden the computed-mask half of the widening dot-reduction target
   route-family validator with exact runtime ABI, role, binding, mask,
   dtype/layout, dot relation, source load, strided, provider support, and
   candidate mirror checks.
2. Extend generated-bundle evidence so
   `computed_masked_widening_dot_reduce_boundary` records target-validator
   consumed facts, mirror-only metadata labels, direct pre-realized route-entry
   support remaining false, exact runtime ABI order, route operand binding, and
   strided-vs-unit-stride distinction.
3. Add focused C++ target artifact coverage for valid computed-mask unit-stride
   and strided acceptance plus stale/missing provider and candidate facts
   failing closed.
4. Update explicit and pre-realized computed-mask widening dot generated-bundle
   dry-run FileCheck consumers for the required runtime counts and boundary
   facts.
5. Run focused local checks, real `ssh rvv` generated-bundle evidence,
   non-computed widening dot non-regression, authority scan, `git diff --check`,
   and `check-tianchenrv`.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. `build/bin/tianchenrv-rvv-extension-plugin-test`
4. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
5. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
6. Explicit and pre-realized dry-runs for
   `computed_masked_widening_dot_reduce_add` and
   `computed_masked_strided_input_widening_dot_reduce_add` with runtime counts
   `0`, `1`, `16`, `17`, and `257`.
7. Focused lit/FileCheck for the four computed-mask widening dot dry-run tests.
8. `ssh -o BatchMode=yes -o ConnectTimeout=8 rvv 'echo rvv-ok'`
9. Explicit and pre-realized `ssh rvv` generated-bundle runs for both
   computed-mask routes with runtime counts `0`, `1`, `16`, `17`, and `257`.
10. Explicit and pre-realized non-computed `widening_dot_reduce_add` and
    `strided_input_widening_dot_reduce_add` dry-run non-regression over the
    same counts.
11. Bounded touched-file authority scan.
12. `git diff --check`
13. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-widening-dot-reduction-artifact-runtime-abi-closure/prd.md`

Primary files to inspect or modify:

- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/TargetArtifactExportTest.cpp`
- Computed-mask widening dot generated-bundle dry-run tests under
  `test/Scripts/`.
