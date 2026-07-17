# Stage2 RVV strided-input widening dot-reduction artifact ABI boundary

## Goal

Implement or prove one bounded Stage 2 end-to-end artifact/runtime ABI path for
the existing pre-realized `strided_input_widening_dot_reduce_add` selected RVV
body. The accepted path must run through RVV plugin-owned selected-body
realization, contraction route-family facts, provider-built
`TCRVEmitCLowerableRoute`, common EmitC materialization, RVV target artifact
bundle export, and real `ssh rvv` correctness evidence when executable
behavior is claimed.

## What I Already Know

* The Direction Brief is the task source: this round is exactly about the
  non-computed-mask `strided_input_widening_dot_reduce_add` boundary, not
  computed-mask-strided variants or wider Stage 2 contraction expansion.
* The previous archived task proved the computed-mask unit-stride
  widening-dot boundary and pinned accumulator/result ABI guards, generated
  bundle evidence, direct route-entry fail-closed evidence, and `ssh rvv`
  correctness.
* Live inspection shows the pre-realized strided-input widening-dot ODS
  surface, focused MLIR fixture, script dry-run test, contraction route-family
  planning, direct contraction provider-plan path, and target artifact
  validation scaffolding already exist.
* The route must derive stride ABI values, strided source-load roles, source
  dtype/config facts, signed widening dot relation, accumulator seed/carry,
  scalar result ABI, runtime `n`/AVL, and VL/tail policy from typed
  `tcrv_rvv` body/config/runtime facts and RVV provider facts. Route ids,
  artifact names, manifests, test names, C strings, descriptors, and exact
  intrinsic spellings are mirrors or emitted evidence only.

## Requirements

* The RVV plugin/contraction owner validates or realizes the selected
  pre-realized `typed_strided_input_widening_dot_reduce_pre_realized_body`
  before route construction.
* Provider route facts and generated-bundle evidence preserve:
  `lhs`/`rhs` source roles,
  explicit `lhs_stride`/`rhs_stride` ABI roles,
  strided source-load facts,
  i16mf2 source dtype/config,
  signed widening dot-product relation,
  accumulator seed ABI and loop carry,
  scalar result/output ABI,
  runtime `n`/AVL,
  route operand-binding facts,
  and tail policy.
* Common EmitC/export must remain semantically neutral; RVV semantics stay in
  RVV plugin/provider/target validators.
* Direct pre-realized route-entry export must fail closed for this operation.
* Missing or stale stride/source/accumulator/result binding must fail before
  target artifact acceptance.
* Runtime evidence must distinguish non-unit strided loads from contiguous
  loads, widening multiply-reduction from add-only/mul-only behavior, seed
  addition, scalar output, and tail preservation.

## Acceptance Criteria

* [x] The selected pre-realized body is consumed into realized
  `tcrv_rvv.strided_load`, `tcrv_rvv.strided_load`,
  `tcrv_rvv.widening_dot_reduce`, and scalar-result store body structure before
  route construction.
* [x] Provider route facts and generated artifact bundle evidence show stride
  ABI binding, strided source-load facts, source roles, signed widening-dot
  relation, accumulator seed/carry, scalar result ABI, runtime `n`/AVL,
  dtype/config, route operand-binding facts, and tail policy.
* [x] Target artifact validation rejects stale or missing strided source,
  stride, accumulator, and result binding before artifact acceptance.
* [x] Direct pre-realized route-entry mode fails closed for
  `strided_input_widening_dot_reduce_add`.
* [x] Real `ssh rvv` correctness evidence covers counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count, with at least two
  stride/data patterns that distinguish non-unit strided loads from contiguous
  loads, signed widening multiply/reduction, seed addition, scalar output, and
  tail preservation.
* [x] A bounded old-authority scan over touched owner/provider/materializer,
  target, script, test, and task/spec files classifies remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] Focused build/test/script checks pass.
* [x] Task status, journal, archive state, commit, and final report are
  truthful.

## Non-Goals

* No computed-masked-strided variant closure in this task.
* No dtype/LMUL clone batch, plain or computed-mask unit-stride widening-dot
  rework, widening conversion closeout, compare/select expansion, high-level
  frontend authority, per-Linalg lowering, source-front-door positive route,
  dashboard, broad smoke matrix, or performance claim.
* No new dtype-prefixed helper op, old `i32m1` route-authority compatibility
  wrapper, descriptor-driven compute, or Common EmitC semantic inference.

## Definition Of Done

* Production validators or route-fact checks protect the strided-input
  widening-dot ABI boundary, with focused regression coverage.
* Focused MLIR/FileCheck, generated-bundle dry-run/script checks, C++ target
  validation, direct route-entry negative evidence, and `ssh rvv` correctness
  evidence pass or an exact external blocker is documented without claiming
  runtime correctness.
* The task is completed and archived per repository convention if all criteria
  are met.
* The work is committed as one coherent commit and the worktree is clean.

## Technical Notes

* Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task read:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/prd.md`.
* Initial code/test inspection covered `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Target/TargetArtifactExportTest.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Notes

* Live repository inspection showed the production path already carries the
  pre-realized strided-input widening-dot body through RVV selected-body
  realization, contraction route-family facts, direct contraction provider
  plan, provider-built `TCRVEmitCLowerableRoute`, common EmitC materialization,
  and RVV target artifact validation.
* This round pinned the missing focused strided boundary proof:
  `test/Target/TargetArtifactExportTest.cpp` now mutates the strided
  widening-dot route operand binding summary plus `acc`, `out`, `lhs_stride`,
  and `rhs_stride` runtime ABI roles and requires fail-closed target artifact
  validation before artifact acceptance.
* `scripts/rvv_generated_bundle_abi_e2e.py` now records two strided runtime
  cases in `widening_dot_reduction_boundary.provider_route_facts` and generates
  harness coverage for stride/data pairs `(lhs_stride=2,rhs_stride=3,pattern=0)`
  and `(lhs_stride=3,rhs_stride=2,pattern=1)`.
* Focused dry-run evidence:
  `artifacts/tmp/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/focused-dry-run/pre-realized-strided-input-widening-dot-reduce-add`.
  It records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, route operand-binding plan
  `rvv-route-operand-binding:strided_widening_dot_reduce.v1`, selected ABI
  roles `lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, strided source memory form,
  stride source mirrors, signed widening-dot relation, scalar seed/carry facts,
  scalar output ABI, and direct route-entry unsupported status.
* Direct route-entry negative evidence:
  `artifacts/tmp/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/focused-direct-fail`.
  The command failed as expected with:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): strided_input_widening_dot_reduce_add`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/final-ssh-rvv/pre-realized-strided-input-widening-dot-reduce-add`.
  Counts `0,1,16,17,257` passed for both stride/data pairs with
  `strided_signed_horizontal_dot`, `seed_added`, `skipped_source_elements_ignored`,
  `scalar_output`, and `tail_preserved`.
* Bounded old-authority scan over touched files found no added-line hits in
  this round. Existing whole-file hits are classified as provider-derived
  emitted-intrinsic evidence in focused RVV tests/scripts, negative
  descriptor/source-front-door guardrails, legacy parse/fail-closed fixtures,
  unrelated existing test coverage, or PRD scan-term inventory; none is route,
  dtype, stride, accumulator, or ABI authority for this task.
* Spec-update review: no `.trellis/spec/` edit is needed. The durable
  selected-body realization, contraction provider, widening-dot evidence, and
  target validator contracts already exist in the RVV plugin, EmitC route, and
  MLIR testing specs; this round only adds focused proof and guard coverage for
  an existing bounded route.

## Checks

* [x] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/focused-dry-run --run-id pre-realized-strided-input-widening-dot-reduce-add --overwrite --op-kind strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* [x] `FileCheck` ROOT, SDOT, and HARNESS prefixes against the focused dry-run
  artifacts from
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-strided-input-widening-dot-reduce-add-dry-run.test`
* [x] Direct pre-realized route-entry negative command with
  `--direct-pre-realized-route-entry --op-kind strided_input_widening_dot_reduce_add`
  failed with the expected retired shortcut diagnostic.
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries | FileCheck ... --check-prefix=REALIZED`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck ... --check-prefix=PLAN`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | FileCheck ... --check-prefix=HEADER`
* [x] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/final-ssh-rvv --run-id pre-realized-strided-input-widening-dot-reduce-add --overwrite --op-kind strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
* [x] Bounded old-authority scan over touched files and added lines.
* [x] `rtk git diff --check`
* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi`
