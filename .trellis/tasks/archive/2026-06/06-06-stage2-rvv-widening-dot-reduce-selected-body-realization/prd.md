# Stage2 RVV widening dot reduce selected-body realization

## Goal

Close the bounded Stage 2 selected-body realization task for plain
`widening_dot_reduce_add`: a pre-realized selected `tcrv.exec` RVV variant must
be consumed by the RVV plugin-local contraction selected-body realization owner,
materialized into an explicit typed `tcrv_rvv.widening_dot_reduce` body, and
then consumed by the existing provider-derived route through target artifact
and generated-bundle evidence. The realization boundary must preserve runtime
ABI roles, dtype/config facts, scalar accumulator seed/result layout, runtime
`n`/AVL use, dispatch/fallback shape, and provider-owned route authority.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV plugin-local selected-body realization for widening-dot reduce`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `5342434f rvv: close explicit widening dot reduce abi`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- Commit `5342434f` closed the explicit selected-body
  `widening_dot_reduce_add` executable ABI route and proved generated-bundle
  execution on real `ssh rvv`.
- The current production realization code already has a contraction
  selected-body owner for
  `tcrv_rvv.typed_widening_dot_reduce_pre_realized_body` in
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`.
- The current positive pre-realized fixture already starts from
  `tcrv_rvv.typed_widening_dot_reduce_pre_realized_body` and checks that
  `--tcrv-materialize-selected-lowering-boundaries` removes that op and
  emits `setvl`, `with_vl`, i16mf2 lhs/rhs loads,
  `tcrv_rvv.widening_dot_reduce`, and store.
- The current provider route path already reports provider-derived facts for
  `widening_dot_reduce_add`, including runtime ABI order
  `lhs,rhs,acc,out,n`, operand binding plan/summary, contraction plan,
  source/result SEW/LMUL, accumulator/result layouts, widening-dot relation,
  intrinsic/header/type facts, and target artifact mirrors.
- Existing pre-realized generated-bundle dry-run evidence already proves the
  pre-realized body is consumed before route planning, but its focused test
  still uses runtime count `23` instead of the current brief's `17` tail case.
- Existing tests include stale-authority checks for several other pre-realized
  families, but the plain pre-realized widening-dot target fixture does not yet
  carry a local stale `route_id` negative check.

## Requirements

- Start from
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`.
- Preserve the explicit selected-body fixture and explicit generated-bundle
  dry-run path as regressions.
- Keep the production route authority in the realized typed `tcrv_rvv` body and
  RVV provider facts, not in route ids, artifact names, script options,
  metadata mirrors, direct pre-realized route-entry shortcuts, descriptors, or
  common EmitC semantics.
- Ensure the pre-realized selected body is consumed by
  `--tcrv-materialize-selected-lowering-boundaries` before emission planning
  and target header export.
- Ensure a stale route-id-only authority attempt on the plain pre-realized
  widening-dot body fails closed before route/provider facts.
- Ensure generated-bundle dry-run evidence covers runtime counts
  `0,1,16,17,257` and patterns `0,1` for the pre-realized path.
- Run real `ssh rvv` generated-bundle execution for the pre-realized path if
  reachable, using counts `0,1,16,17,257` and patterns `0,1`.
- Retain scalar seed behavior from `acc[0]`, scalar `out[0]` result,
  source/accumulator preservation, output tail preservation, signed widening
  products, and add-only/mul-only distinguishing checks.

## Acceptance Criteria

- [x] The pre-realized target fixture proves realization into typed
      `tcrv_rvv.widening_dot_reduce` before route planning.
- [x] The same fixture proves stale `route_id` authority on the pre-realized
      body fails closed.
- [x] Existing explicit selected-body target/header fixture remains passing.
- [x] Pre-realized generated-bundle dry-run succeeds for counts
      `0,1,16,17,257` and patterns `0,1`.
- [x] Existing explicit generated-bundle dry-run remains passing.
- [x] Real `ssh rvv` generated-bundle execution for the pre-realized path
      passes if reachable, or the exact external blocker is recorded.
- [x] Provider/target facts remain provider-derived mirrors after realization:
      runtime ABI order, operand binding, contraction route-family plan,
      typed compute op, source/result SEW/LMUL, accumulator/result layout,
      widening-dot relation, reduction store VL, required headers/type mapping,
      target leaf profile, and provider supported mirror.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python remains limited to generated-bundle support tooling; this task should
  not need script implementation changes unless focused validation exposes a
  real script blocker.
- No dtype/LMUL clone batch, strided or computed-mask dot expansion,
  widening-MAcc follow-up, high-level frontend, performance/autotuning,
  source-front-door positive route, direct pre-realized route-entry shortcut,
  descriptor-driven compute, or common EmitC semantic branch is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or a genuine external execution blocker is
  proven.

## Technical Approach

Use the current production contraction realization owner as the implementation
surface and avoid widening the route family. Add the missing local negative
fixture check for stale `route_id` authority on the plain pre-realized
widening-dot body. Update the pre-realized generated-bundle dry-run test to use
the brief's tail case `17`. Then run the focused materialization, emission
plan, header, generated-bundle dry-run, explicit regression, script self-test
only if the script changes, real `ssh rvv` execution if reachable, and final
diff/context checks.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  and `.trellis/spec/testing/index.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-explicit-widening-dot-reduce-abi-closure/prd.md`.
- Fixtures read:
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`,
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run.test`.
- Implementation files inspected:
  `include/TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h`
  and `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`.

## Completion Evidence

- Local build refresh passed:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Focused pre-realized target fixture passed for:
  `REALIZED`, `PLAN`, `HEADER`, and the new `STALE-AUTH` negative.
- The new `STALE-AUTH` negative injects
  `route_id = "rvv-i32m1"` into the plain
  `typed_widening_dot_reduce_pre_realized_body` and confirms it is rejected as
  authority metadata before route/provider facts.
- Focused explicit target fixture regression passed for `PLAN` and `HEADER`.
- Pre-realized generated-bundle dry-run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-dot-reduce-add-stage2-realization-dry-run`.
- Direct FileCheck of the pre-realized dry-run evidence passed for `ROOT`,
  `WDOT`, and `HARNESS`. The dry-run evidence now uses counts
  `0,1,16,17,257`.
- Explicit generated-bundle dry-run regression passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-dot-reduce-add-stage2-realization-regression-dry-run`.
- Real `ssh rvv` generated-bundle execution passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-dot-reduce-add-stage2-realization-ssh`.
- Generated bundle path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-dot-reduce-add-stage2-realization-ssh/widening_dot_reduce_add/generated_bundle`.
- Remote PASS marker:
  `PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1`.
- Runtime evidence covered counts `0,1,16,17,257` for both patterns `0` and
  `1`, including zero-count seed behavior, single element, VL-boundary `16`,
  tail `17`, and multi-chunk `257`.
- Runtime evidence confirmed signed horizontal widening dot products,
  seed-added behavior, add-only and mul-only distinguishing cases, scalar
  output in `out[0]`, source preservation, accumulator preservation, and output
  tail preservation.
- Provider/target facts preserved in dry-run and ssh evidence include:
  runtime ABI order `lhs,rhs,acc,out,n`,
  route operand binding plan
  `rvv-route-operand-binding:widening_dot_reduce.v1`,
  typed compute op `tcrv_rvv.widening_dot_reduce`,
  memory form `vector-rhs-load`,
  contraction route-family plan `rvv-contraction-route-family-plan.v1`,
  source SEW/LMUL `16/mf2`,
  accumulator/result SEW/LMUL `32/m1`,
  accumulator layout `scalar-i32-seed-lane0-from-accumulator-input`,
  result layout `store-dot-reduction-lane0-to-output-scalar`,
  widening relation `signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32`,
  reduction store VL `1`,
  required headers `stddef.h,stdint.h,riscv_vector.h`,
  target leaf profile `rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1`,
  and provider mirror
  `provider_supported_mirror:rvv-contraction-family-plan-validated`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-target-artifact-export-test` and
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script validation passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` and
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- Bounded old-authority scan over added diff lines only found the intentional
  `STALE-AUTH` negative `route_id = "rvv-i32m1"` and its FileCheck line; no
  positive executable route-authority growth was introduced.
- `git diff --check` passed.
- `git diff --cached --check` passed before staging.
- Trellis context validation passed with five implement entries and five check
  entries.

## Self-Repair Performed

- Initial direct FileCheck commands failed because `FileCheck` was not in
  `PATH`; reran with `/usr/lib/llvm-20/bin/FileCheck` and
  `/usr/lib/llvm-20/bin/not`.
- Initial generated-bundle dry-run commands failed because `llvm-readobj` was
  not in `PATH`; reran with `/usr/lib/llvm-20/bin/llvm-readobj`.

## Spec Update Judgment

- No `.trellis/spec/` update was needed. This round did not introduce a new
  selected-body owner API, route contract, target artifact contract, generated
  bundle harness convention, or runtime evidence rule. It applied the existing
  RVV plugin-local selected-body realization owner registry, provider-owned
  widening dot-reduce route validation contract, and RVV testing evidence
  requirements to the plain `widening_dot_reduce_add` pre-realized fixture.
