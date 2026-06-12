# Stage2 RVV computed-masked strided-input widening dot-reduction artifact ABI boundary

## Goal

Implement or prove one bounded Stage 2 end-to-end artifact/runtime ABI path for
the existing pre-realized
`computed_masked_strided_input_widening_dot_reduce_add` selected RVV body. The
accepted path must run through RVV plugin-owned selected-body realization,
contraction route-family facts, provider-built `TCRVEmitCLowerableRoute`,
common EmitC materialization, RVV target artifact bundle export, and real
`ssh rvv` correctness evidence when executable behavior is claimed.

## What I Already Know

* The Direction Brief is the task source: this round is exactly about the
  combined computed-mask plus non-unit-stride input widening dot-reduction
  boundary, not the previously completed computed-mask-only or strided-only
  variants.
* The previous archived tasks proved `computed_masked_widening_dot_reduce_add`
  and `strided_input_widening_dot_reduce_add` separately. This task must verify
  that compare/mask facts and explicit lhs/rhs stride ABI values survive
  together into the same provider-built route and generated artifact ABI.
* Live inspection shows a focused MLIR fixture and script dry-run test already
  exist for
  `computed_masked_strided_input_widening_dot_reduce_add`; production support
  must be checked before adding only missing guards or evidence.
* Route authority must come from typed `tcrv_rvv` body/config/runtime facts and
  RVV plugin/provider validation. Route ids, artifact names, manifests, test
  names, descriptors, C strings, exact intrinsic spellings, and common EmitC
  code are mirrors or emitted evidence only.

## Requirements

* The RVV plugin/contraction owner validates or realizes the selected
  pre-realized
  `typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body`
  before route construction.
* Provider route facts and generated-bundle evidence preserve compare/mask
  producer facts, mask policy, explicit `lhs_stride`/`rhs_stride` ABI roles,
  strided source-load roles, signed widening dot relation, accumulator seed and
  carry, scalar result ABI, runtime `n`/AVL, dtype/config, route operand
  bindings, and VL/tail policy.
* Common EmitC/export remains semantically neutral; RVV semantics stay in RVV
  plugin/provider/target validators.
* Direct pre-realized route-entry export fails closed for this operation.
* Missing or stale mask/stride/source/accumulator/result binding fails before
  target artifact acceptance.
* Runtime evidence distinguishes non-unit strided loads, masked-off lanes,
  signed widening multiply-accumulate/reduction, seed addition, scalar output,
  and tail preservation.

## Acceptance Criteria

* [x] The selected pre-realized body is consumed into realized `tcrv_rvv`
  compare, strided source loads, masked widening-dot reduction, and scalar
  result store body structure before route construction.
* [x] Provider route facts and generated artifact bundle evidence show compare
  facts, mask role/source/form, mask policy, lhs/rhs stride ABI binding,
  strided source-load facts, source roles, signed widening-dot relation,
  accumulator seed/carry, scalar result ABI, runtime `n`/AVL, dtype/config,
  route operand-binding facts, and tail policy.
* [x] Target artifact validation rejects stale or missing combined
  mask/stride/source/accumulator/result binding before artifact acceptance.
* [x] Direct pre-realized route-entry mode fails closed for
  `computed_masked_strided_input_widening_dot_reduce_add`.
* [x] Real `ssh rvv` correctness evidence covers counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count, with at least two
  stride/data/mask patterns that distinguish non-unit strided loads,
  masked-off lanes, widening multiply/reduction, seed addition, scalar output,
  and tail preservation.
* [x] A bounded old-authority scan over touched plugin/provider/materializer,
  target, script, test, and task/spec files classifies remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] Focused build/test/script checks pass.
* [x] Task status, journal, archive state, commit, and final report are
  truthful.

## Definition Of Done

* Production validators or route-fact checks protect the combined
  computed-mask and strided-input widening-dot ABI boundary, with focused
  regression coverage.
* Focused MLIR/FileCheck, generated-bundle dry-run/script checks, C++ target
  validation, direct route-entry negative evidence, and `ssh rvv` correctness
  evidence pass or an exact external blocker is documented without claiming
  runtime correctness.
* The task is completed and archived per repository convention if all criteria
  are met.
* The work is committed as one coherent commit and the worktree is clean.

## Technical Approach

First compare the existing combined fixture/script/route owner against the two
completed single-boundary tasks. If production already carries most facts, add
the smallest missing fail-closed production guard or route-fact validation that
protects the combined boundary. Evidence should stay focused on the bounded
operation and should not broaden Stage 2 coverage.

## Out Of Scope

* No dtype/LMUL clone batch, plain/computed-mask-only/strided-only widening-dot
  rework, widening conversion closeout, compare/select expansion, high-level
  frontend authority, per-Linalg lowering, source-front-door positive route,
  dashboard, broad smoke matrix, or performance claim.
* No new dtype-prefixed helper op, old `i32m1` route-authority compatibility
  wrapper, descriptor-driven compute, or Common EmitC semantic inference.

## Technical Notes

* Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior tasks read:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/prd.md`
  and
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/prd.md`.
* Initial code/test inspection covered `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `scripts/rvv_remote_probe.py`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`.

## Completion Notes

* Live repository inspection showed the combined production path already had
  the selected-body realization and route-family shape for
  `computed_masked_strided_input_widening_dot_reduce_add`, including compare
  mask production, strided source loads, masked widening dot reduction, scalar
  result store, and generated bundle evidence.
* This round closed the missing combined ABI/header binding boundary:
  `RVVEmitCContractionRouteFamilyPlanOwners.cpp` now marks the combined route's
  compare operands, dot source operands, and lhs/rhs stride operands as
  header/prototype ABI participants in the provider-owned operand-binding
  summary. The target validator and generated-bundle script now require the
  exact same provider summary.
* `test/Target/TargetArtifactExportTest.cpp` now mutates the combined route's
  operand-binding summary, dot-lhs ABI role, accumulator role, output role,
  lhs/rhs stride roles, mask source, and compare predicate, and proves the RVV
  target artifact validator fails closed before artifact acceptance.
* Focused dry-run evidence:
  `artifacts/tmp/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/focused-dry-run/pre-realized-computed-masked-strided-input-widening-dot-reduce-add`.
  It records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, route operand-binding plan
  `rvv-route-operand-binding:masked_strided_wdot.v1`, selected ABI roles
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, compare predicate
  `slt`, computed mask role/source/form, strided input facts, signed widening
  dot relation, scalar seed/carry facts, scalar output ABI, and direct
  route-entry unsupported status.
* Direct route-entry negative evidence:
  `artifacts/tmp/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/focused-direct-fail`.
  The command failed as expected with:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): computed_masked_strided_input_widening_dot_reduce_add`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/final-ssh-rvv/pre-realized-computed-masked-strided-input-widening-dot-reduce-add`.
  Counts `0,1,16,17,257` passed for stride/data/mask cases
  `(lhs_stride=2,rhs_stride=3,mask_pattern=0,input_pattern=0)` and
  `(lhs_stride=3,rhs_stride=2,mask_pattern=1,input_pattern=1)` with
  `compare_masked_strided_signed_horizontal_dot`, `seed_added`,
  `inactive_lanes_skipped`, `skipped_source_elements_ignored`,
  `scalar_output`, and `tail_preserved`.
* Spec update:
  `.trellis/spec/lowering-runtime/emitc-route.md` now records the provider
  operand-binding summary contract: exported runtime ABI parameters in target
  headers/prototypes must have provider-derived binding entries with explicit
  header/prototype participation markers, and target artifact validation must
  compare provider and candidate mirrors exactly.
* Bounded old-authority scan over the relevant owner/provider/materializer,
  target, script, focused tests, and spec/task files found no new added-line
  production hits for legacy RVV authority. Remaining whole-file hits are
  classified as provider-derived emitted-intrinsic evidence in focused RVV
  tests/scripts, negative descriptor/source-front-door guardrails, legacy
  parse/fail-closed fixtures, historical remote-probe examples, task scan-term
  inventory, or same-analysis selected-route diagnostics; none is route, dtype,
  mask, stride, accumulator, result, or ABI authority for this task.

## Checks

* [x] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries | FileCheck ... --check-prefix=REALIZED`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck ... --check-prefix=PLAN`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | FileCheck ... --check-prefix=HEADER`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/focused-dry-run --run-id pre-realized-computed-masked-strided-input-widening-dot-reduce-add --overwrite --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* [x] FileCheck `ROOT`, `MSDOT`, and `HARNESS` prefixes against the focused
  dry-run artifacts.
* [x] Direct pre-realized route-entry negative command with
  `--direct-pre-realized-route-entry --op-kind computed_masked_strided_input_widening_dot_reduce_add`
  failed with the expected retired shortcut diagnostic.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/final-ssh-rvv --run-id pre-realized-computed-masked-strided-input-widening-dot-reduce-add --overwrite --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
* [x] Bounded old-authority scan over touched files and added lines.
* [x] `rtk git diff --check`
