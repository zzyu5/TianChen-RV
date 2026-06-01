# Stage2 RVV widening conversion artifact ABI boundary

## Goal

Carry exactly one existing pre-realized `widen_i16_to_i32` selected body from a
selected `tcrv.exec` RVV variant through RVV plugin-local selected-body
realization, provider-owned source/result conversion facts,
`TCRVEmitCLowerableRoute`, common EmitC materialization, RVV target artifact
bundle generation, and real `ssh rvv` correctness evidence.

This round focuses on the conversion/dtype/SEW/LMUL Stage 2 bottleneck. The
executable path must prove that source i16/mf2 facts, result i32/m1 facts,
`sign_extend_widen_vf2`, runtime n/AVL, source-load/result-store roles, and
tail policy are carried structurally from the typed `tcrv_rvv` body and
provider route facts, not inferred from route ids, artifact names, manifests,
test names, descriptor residue, C strings, exact intrinsic spellings, or
common export code.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief as
  `.trellis/tasks/06-01-stage2-rvv-widening-conversion-artifact-abi`.
* The repository started clean on `main` at
  `27c75172 rvv: prove dual runtime-scalar mask-select artifact abi`.
* The immediately preceding archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/`
  proved the dual runtime-scalar mask/select body through selected-body
  realization, provider facts, generated artifact ABI, real `ssh rvv`
  correctness, fail-closed direct route-entry coverage, and clean archive/commit
  state.
* `.trellis/spec/index.md` requires the RVV authority chain to remain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body ->
  RVV plugin legality/realization/provider -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` assigns RVV dtype/config,
  operation, policy, ABI binding, selected-body realization, intrinsic mapping,
  and fail-closed diagnostics to the RVV plugin.
* `.trellis/spec/lowering-runtime/emitc-route.md` forbids common EmitC/export
  from inferring RVV dtype, SEW, LMUL, conversion relation, operation kind,
  policy, ABI semantics, or route support from mirrors, metadata, names,
  manifests, descriptors, or artifact fields.
* `.trellis/spec/testing/mlir-testing-contract.md` requires positive generated
  RVV artifact evidence to start from explicit typed vector-level `tcrv_rvv`
  bodies and requires real `ssh rvv` evidence for runtime correctness claims.
* Initial code inspection shows a bounded widening-conversion path already
  exists for `widen_i16_to_i32` / `sign_extend_widen_vf2`:
  `TypedWideningConversionPreRealizedBodyOp`,
  `RVVWideningConversionSelectedBodyRealizationOwner`,
  widening-conversion statement/control/provider route planning,
  common EmitC materialization, target support bundle metadata, the
  `rvv_generated_bundle_abi_e2e.py` harness, and focused Target/Script tests.

## Requirements

* Use exactly one selected body for this round:
  `widen_i16_to_i32`, with pre-realized op kind
  `sign_extend_widen_vf2` and conversion relation
  `signed-i16mf2-to-i32m1`.
* The positive path must start from the selected-boundary pre-realized body and
  realize before route construction:

  ```text
  selected tcrv.exec RVV variant
    -> typed pre-realized widen_i16_to_i32 body
    -> RVV widening-conversion selected-body realization owner
    -> realized tcrv_rvv setvl/with_vl/load/widening_convert/store body
    -> widening-conversion route-family plan
    -> operand-binding and statement-plan facts
    -> provider preflight
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV target artifact bundle
    -> external callable ABI consumer
    -> ssh rvv evidence
  ```

* Runtime `n`/AVL, source i16/mf2 type/config, result i32/m1 type/config,
  source-load role, result-store role, conversion relation, conversion kind,
  memory form, and tail policy must come from the typed selected `tcrv_rvv`
  body and provider-validated route facts.
* Provider/export evidence may mirror route metadata after provider route
  construction, but route ids, artifact names, manifests, test names,
  descriptors, emission-plan fields, C strings, exact intrinsic spellings, and
  common EmitC code must not become dtype/config, conversion, policy, ABI, or
  route authority.
* The generated bundle harness must check:
  `out[i] = (int32_t)lhs[i]`, runtime `n = 0` skips writes, output tail storage
  beyond `n` remains sentinel preserved, and the data patterns distinguish sign
  extension from zero extension, missing conversion, and plain copy.
* Real `ssh rvv` correctness evidence must cover representative counts
  `0,1,16,23,257` and at least two source data patterns containing mixed-sign
  and wide-magnitude int16 values.
* Direct pre-realized route-entry or artifact export for widening conversion
  must fail closed before provider route construction/export unless production
  evidence shows a spec-backed direct route-entry API was intentionally
  reintroduced.
* Keep changes bounded to this single production/export/runtime boundary. If
  the existing path already expresses the ABI faithfully, prove it and add only
  the missing focused fail-closed guard, harness repair, or evidence pin.

## Acceptance Criteria

* [x] Trellis PRD/context files validate successfully and the task is started.
* [x] Focused evidence shows `widen_i16_to_i32` is realized by the RVV
  widening-conversion selected-body owner before route construction.
* [x] Focused provider/export evidence shows source i16/mf2, result i32/m1,
  `sign_extend_widen_vf2`, `signed-i16mf2-to-i32m1`, source load, result store,
  runtime `n`/AVL, and tail policy survive into the generated bundle.
* [x] Generated-bundle dry-run for `--pre-realized-selected-body --op-kind
  widen_i16_to_i32` passes for counts `0,1,16,23,257`.
* [x] Direct pre-realized route-entry mode for widening conversion fails closed
  with the selected lowering-boundary diagnostic before route/provider export.
* [x] Real `ssh rvv` compile/run correctness evidence passes for counts
  `0,1,16,23,257`, with sign-extension-distinguishing input data and tail
  sentinel preservation.
* [x] Focused lit/FileCheck, C++ target/plugin tests, or script self-tests
  covering the touched path pass.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec/task files classifies any remaining hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] `git diff --check` passes.
* [x] The task is finished/archived and one coherent commit is created, or the
  exact blocker and continuation point are recorded.

## Definition of Done

* The executable compiler path for the base `widen_i16_to_i32` selected body is
  proven through selected-body realization, generated RVV target artifact bundle
  export, and external ABI execution on `ssh rvv`.
* Runtime correctness claims are backed by real `ssh rvv` output.
* No dtype/LMUL clone batch, `widen_i32_to_i64` closeout, compare/select
  expansion, reduction/contraction expansion, high-level frontend authority,
  source-front-door path, descriptor compute path, common EmitC semantic
  inference, performance claim, or broad smoke matrix is added.
* Task status, workspace journal, archived task record, final commit, and final
  report truthfully describe what was completed and what was not.

## Out of Scope

* `widen_i32_to_i64` closeout, dtype/LMUL clone expansion, compare/select
  expansion, memory/reduction/contraction expansion, high-level frontend or
  Linalg authority, source-front-door positive routes, dashboards, tuning
  databases, readiness state machines, or performance claims.
* New descriptor-driven computation, direct-C/source-export route support,
  source-front-door positive routes, or common EmitC/export inference of RVV
  conversion semantics.
* Treating route ids, artifact names, manifests, exact intrinsic spellings,
  status fields, emission-plan metadata, scripts, or test names as source/result
  dtype, LMUL, conversion relation, policy, ABI, or route authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-dual-runtime-scalar-mask-select-artifact-abi/prd.md`.
* Inspected/search targets:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVWideningConversionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widen-i16-to-i32-dry-run.test`,
  and `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`.

## Completion Notes

* The selected body for this round was exactly the pre-realized
  `widen_i16_to_i32` / `sign_extend_widen_vf2` widening conversion body.
* The existing production C++ path already carried the selected body through
  RVV plugin-local realization, widening-conversion route-family planning,
  operand-binding facts, statement-plan facts, provider preflight,
  `TCRVEmitCLowerableRoute`, common EmitC, and RVV target artifact bundle
  export. No C++ owner/provider/materializer/target code change was needed.
* The evidence gap was script/test-local: `conversion_sew_policy_boundary`
  exposed source/result type policy and emitted C++ checks, but did not
  explicitly pin selected source ABI roles, source-load/result-store statement
  plan, provider route facts, tail/mask policy mirrors, or the retired direct
  pre-realized route-entry status in one boundary summary.
* `scripts/rvv_generated_bundle_abi_e2e.py` now adds those fields to
  `conversion_sew_policy_boundary` and self-tests that fake bundle generation
  preserves widening-conversion provider-backed source/result/statement facts.
* The focused dry-run test now FileChecks those boundary fields for
  `widen_i16_to_i32`.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/focused-dry-run-v2/pre-realized-widen-i16-to-i32`.
  It records `input_mode = pre-realized-selected-body`,
  `pre_realized_body_consumed = true`,
  `selected_body_realization_producer = rvv-plugin-local-selected-body-realization-owner-registry`,
  source i16/mf2, result i32/m1, `sign_extend_widen_vf2`,
  `signed-i16mf2-to-i32m1`, source-load/result-store ABI roles, provider route
  facts, runtime `lhs,out,n`, and tail/mask agnostic policy.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/focused-direct-fail-v2/direct-pre-realized-widen-i16-to-i32`.
  The command exited with the expected retired direct route-entry diagnostic:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): widen_i16_to_i32`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/final-ssh-rvv-v2/pre-realized-widen-i16-to-i32`.
  Counts `0,1,16,23,257` passed for two source data patterns. Each case
  reported `sign_extension_checked`, `two_input_patterns_checked`, and
  `tail_preserved`; final remote output reported
  `tcrv_rvv_generated_bundle_abi_widen_i16_to_i32_ok counts=0,1,16,23,257`.
* Bounded old-authority scan:
  the touched dry-run test has only negative `implicit-check-not` guardrail
  hits for `descriptor`, `direct-C`, `source-export`, and `tcrv_rvv.i32_`,
  plus provider-derived emitted-C evidence checks for
  `__riscv_vwcvt_x_x_v_i32m1` and `__riscv_vse32_v_i32m1`. The task PRD hits
  are guardrail/non-goal text. Existing full-file exact intrinsic spelling hits
  in `scripts/rvv_generated_bundle_abi_e2e.py` remain provider-derived emitted
  C/C++ evidence checks for existing RVV route families, not route authority.
* Spec update decision: no `.trellis/spec/` update was needed. Existing RVV
  plugin, EmitC route, variant pipeline, and MLIR testing contracts already
  cover this boundary; this round made the widening-conversion generated-bundle
  evidence stricter and durable.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-widening-conversion-artifact-abi`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/focused-dry-run-v2 --run-id pre-realized-widen-i16-to-i32 --overwrite --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/focused-direct-fail-v2 --run-id direct-pre-realized-widen-i16-to-i32 --overwrite --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj` exited with the expected unsupported direct route-entry diagnostic.
* [x] FileCheck `ROOT`, `WIDEN`, and `HARNESS` prefixes against
  `focused-dry-run-v2`.
* [x] `tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir --tcrv-materialize-selected-lowering-boundaries | FileCheck ... --check-prefix=REALIZED`
* [x] `tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck ... --check-prefix=PLAN`
* [x] `tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck ... --check-prefix=HEADER`
* [x] `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test`
* [x] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk cmake --build build --target tianchenrv-target-artifact-export-test`
* [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-01-stage2-rvv-widening-conversion-artifact-abi/final-ssh-rvv-v2 --run-id pre-realized-widen-i16-to-i32 --overwrite --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
