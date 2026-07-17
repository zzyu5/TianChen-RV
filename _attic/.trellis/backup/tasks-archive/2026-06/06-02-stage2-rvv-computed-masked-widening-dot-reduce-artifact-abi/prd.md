# Stage2 RVV computed-masked widening dot-reduction artifact ABI boundary

## Goal

Implement or prove one bounded Stage 2 end-to-end artifact/runtime ABI path for
the existing pre-realized `computed_masked_widening_dot_reduce_add` selected RVV
body. The accepted path must run through RVV plugin-owned selected-body
realization, contraction route-family facts, provider-built
`TCRVEmitCLowerableRoute`, common EmitC materialization, RVV target artifact
bundle export, and real `ssh rvv` correctness evidence when executable
behavior is claimed.

## What I Already Know

* The Direction Brief is the task source: this round is exactly about
  `computed_masked_widening_dot_reduce_add`, not strided variants, broad
  contraction coverage, plain widening-dot rework, or frontend authority.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/`
  proved the plain `widening_dot_reduce_add` artifact ABI boundary through
  selected-body realization, contraction route facts, target validation, dry-run
  evidence, direct route-entry fail-closed evidence, and `ssh rvv` correctness.
* Live inspection shows the computed-mask widening-dot ODS surface, focused MLIR
  fixture, script dry-run test, contraction route-family planning, route
  validation, and target artifact validation scaffolding already exist.
* The route must derive compare/mask facts, mask policy, source roles, signed
  widening dot relation, accumulator seed/carry, scalar result ABI, runtime
  `n`/AVL, dtype/config, and VL/tail policy from typed `tcrv_rvv` body/config
  and provider facts. Metadata, route IDs, artifact names, test names,
  descriptors, C strings, and exact intrinsic spellings are mirrors or emitted
  evidence only.

## Requirements

* The RVV plugin/contraction owner validates or realizes the selected
  pre-realized `typed_computed_mask_widening_dot_reduce_pre_realized_body`
  before route construction.
* Provider route facts and generated-bundle evidence preserve:
  compare lhs/rhs role and predicate,
  compare-produced mask role/source/form,
  dot lhs/rhs source roles and i16mf2 source dtype/config,
  signed masked widening dot-product relation,
  accumulator seed ABI and loop carry,
  scalar result/output ABI,
  runtime `n`/AVL and VL/tail policy,
  provider route operand-binding plan and exact operand summary.
* Common EmitC/export must remain semantically neutral; RVV semantics stay in
  RVV plugin/provider/target validators.
* Direct pre-realized route-entry export must fail closed.
* Missing or stale mask/source/accumulator/result binding must fail before
  target artifact acceptance.
* Evidence must be focused on this operation and the touched owner path.

## Acceptance Criteria

* [x] The selected pre-realized body is consumed into realized
  `tcrv_rvv.compare` and `tcrv_rvv.masked_widening_dot_reduce` body structure
  before route construction.
* [x] Provider route facts and generated artifact bundle evidence show compare
  facts, mask role/source/form, source roles, signed widening-dot relation,
  accumulator seed/carry, scalar result ABI, runtime `n`/AVL, dtype/config,
  route operand-binding facts, and tail/mask policy.
* [x] Target artifact validation rejects stale direct/mirror-derived
  mask/source/accumulator/result facts for the computed-mask widening-dot route.
* [x] Direct pre-realized route-entry mode fails closed for
  `computed_masked_widening_dot_reduce_add`.
* [x] Real `ssh rvv` correctness evidence covers counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count, with data/mask patterns
  that distinguish masked-off lanes, widening multiply/reduction, seed
  addition, and scalar output.
* [x] A bounded old-authority scan over touched files classifies remaining hits
  for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] Focused build/test/script checks pass.
* [x] Task status, journal, archive state, commit, and final report are
  truthful.

## Non-Goals

* No strided-input or computed-mask-strided variant closure in this task.
* No dtype/LMUL clone batch, plain widening-dot rework, widening conversion
  closeout, compare/select expansion, high-level frontend authority,
  per-Linalg lowering, source-front-door positive route, dashboard, broad smoke
  matrix, or performance claim.
* No new dtype-prefixed helper op, old `i32m1` route-authority compatibility
  wrapper, descriptor-driven compute, or Common EmitC semantic inference.

## Definition Of Done

* Production validators or route-fact checks protect the computed-mask
  widening-dot ABI boundary, with focused regression coverage.
* Focused MLIR/FileCheck, generated-bundle dry-run/script checks, C++ target
  validation, and `ssh rvv` correctness evidence pass or an exact external
  blocker is documented without claiming runtime correctness.
* The task is completed and archived per repository convention if all criteria
  are met.
* The work is committed as one coherent commit and the worktree is clean.

## Technical Notes

* Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task read:
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/prd.md`.
* Initial code/test inspection covered `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the focused MLIR/script tests.

## Completion Notes

* Live repository inspection showed the production path already had the core
  computed-mask widening-dot route chain: pre-realized typed selected body
  validation in the RVV contraction owner, realization into
  `tcrv_rvv.compare` + `tcrv_rvv.masked_widening_dot_reduce`, contraction
  route-family facts, provider-built route operands, Common EmitC
  materialization, and RVV target artifact route-family validation.
* This round pinned the missing focused boundary proof:
  `test/Target/TargetArtifactExportTest.cpp` now mutates the computed-mask
  widening-dot `acc` and `out` runtime ABI roles separately and proves the
  target artifact route-family validator rejects stale accumulator/result
  binding before artifact acceptance.
* `scripts/rvv_generated_bundle_abi_e2e.py --self-test` now asserts that the
  computed-mask widening-dot generated-bundle boundary summary preserves
  `acc` as `accumulator-input-buffer`, `out` as `output-buffer`, seed source
  `acc[0]`, loop carry `out[0]`, scalar store VL `1`, direct route-entry
  unsupported status, and runtime counts `0,1,16,17,257`.
* Focused dry-run evidence:
  `artifacts/tmp/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/focused-dry-run/pre-realized-computed-masked-widening-dot-reduce-add`.
  It records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`, route operand-binding plan
  `rvv-route-operand-binding:masked_widening_dot_reduce.v1`, selected ABI
  roles `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, compare predicate `slt`, mask
  source `compare-produced-mask-same-vl-scope`, masked product intrinsic
  evidence, seed/carry facts, scalar output ABI, and direct route-entry
  unsupported status.
* Direct route-entry negative evidence:
  `artifacts/tmp/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/focused-direct-fail.log`.
  The command failed as expected with:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): computed_masked_widening_dot_reduce_add`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/final-ssh-rvv/pre-realized-computed-masked-widening-dot-reduce-add`.
  Counts `0,1,16,17,257` passed with
  `compare_masked_signed_horizontal_dot`, `seed_added`,
  `inactive_lanes_skipped`, `scalar_output`, and `tail_preserved`.
* Bounded old-authority scan over the relevant owner/provider/materializer,
  target, script, and focused test files found no new added-line hits in this
  round. Existing hits are classified as provider-derived emitted-intrinsic
  evidence in focused computed-mask widening-dot tests/scripts, negative
  descriptor/source-front-door guardrails, legacy parse/fail-closed fixtures,
  historical remote-probe examples, or diagnostic text about same-analysis
  selected-route consistency; none is route, dtype, mask, accumulator, or ABI
  authority for this task.

## Checks

* [x] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [x] `rtk git diff --check`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/focused-dry-run --run-id pre-realized-computed-masked-widening-dot-reduce-add --overwrite --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* [x] Direct pre-realized route-entry negative command with
  `--direct-pre-realized-route-entry --op-kind computed_masked_widening_dot_reduce_add`
  failed with the expected retired shortcut diagnostic.
* [x] FileCheck equivalent for
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-widening-dot-reduce-add-dry-run.test`:
  `ROOT`, `MDOT`, and `HARNESS` prefixes against focused dry-run artifacts.
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries | FileCheck ... --check-prefix=REALIZED`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck ... --check-prefix=PLAN`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | FileCheck ... --check-prefix=HEADER`
* [x] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/final-ssh-rvv --run-id pre-realized-computed-masked-widening-dot-reduce-add --overwrite --op-kind computed_masked_widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi`
