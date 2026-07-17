# Stage2 RVV standalone reduce-add executable ABI closure

## Goal

Close the executable ABI evidence for the existing typed RVV
`standalone_reduce_add` route. The bounded path is a selected `tcrv.exec` RVV
variant with an explicit typed `tcrv_rvv.standalone_reduce {kind = "add"}`
body, RVV provider-derived reduction facts, `TCRVEmitCLowerableRoute`, common
EmitC/export materialization, generated target artifact/header/object, and a
focused external ABI harness that runs on `ssh rvv` against scalar reference
semantics.

This is an executable-closure task for one reduction route. It is not a new
reduction route foundation, metadata-only mirror task, benchmark, high-level
frontend, or Gearbox expansion.

## What I already know

- No `.trellis/.current-task` existed when this round began, so this task was
  created from the Hermes Direction Brief.
- The worktree was clean at the start, and the latest commit was
  `f34dcc72 rvv: mirror standalone reduction kind`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-typed-reduction-accumulation-route/`
  completed provider-derived `reduction_kind` mirrors and target validation,
  but explicitly left generated artifact plus `ssh rvv` correctness closure as
  the next continuation point.
- `RVVSelectedBodyEmitCRouteDescription`,
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`,
  `RVVStandaloneReductionRouteFacts`, and
  `RVVStandaloneReductionRouteValidationContract` already carry standalone
  reduction facts including `reductionKind`, accumulator/result layout,
  `reductionStoreVL`, runtime ABI order, runtime ABI parameters, and route
  operand binding summary.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has standalone reduction
  expectations, metadata checks, generated C/C++ shape checks, and a generated
  external ABI harness for `standalone_reduce_add`.
- The script default runtime counts do not include `n=0`; this task must run
  explicit counts so the evidence includes zero, small, full-chunk, tail, and
  multi-chunk cases.

## Requirements

- Preserve the authority chain:
  typed selected `tcrv_rvv.standalone_reduce` body -> RVV provider facts ->
  lowerable route -> common EmitC/export -> generated artifact -> harness
  execution.
- Generated artifact evidence must prove the provider-derived standalone
  reduction contract before runtime:
  `reduction_kind = add`, accumulator layout, result layout,
  `reduction_store_vl = 1`, runtime ABI order `lhs,acc,out,n`, runtime ABI
  parameter roles/types, and route operand binding with exported `abi|hdr`
  markers.
- The ABI harness must initialize `lhs`, `acc`, and `out`; call the generated
  prototype in `lhs,acc,out,n` order; compare `out[0]` against a scalar add
  reduction seeded by `acc[0]`; verify `n=0`; preserve `acc[0]`; preserve the
  source buffer; and preserve scalar-output sentinel lanes beyond lane 0.
- Runtime evidence must run on the real RVV target via `ssh rvv` with explicit
  counts covering `n=0`, small `n`, full-chunk `n`, tail `n`, and multi-chunk
  `n`.
- Missing provider facts, stale `reduction_kind`, stale accumulator/result
  layout, stale route binding/runtime ABI, stale type/header facts, or missing
  selected-body/provider facts must fail before execution.
- If live code inspection shows the route and harness are already sufficient,
  finish by recording generated artifact and `ssh rvv` evidence rather than
  adding redundant compiler code.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      executable-closure task and its spec basis.
- [x] Focused build of `tcrv-opt` and `tcrv-translate` succeeds.
- [x] The selected-body target artifact test for
      `explicit-selected-body-artifact-standalone-reduce-add.mlir` passes for
      positive plan/header checks and stale mirror diagnostics.
- [x] Generated bundle evidence for `standalone_reduce_add` includes provider
      mirrors for `reduction_kind = add`, accumulator layout, result layout,
      `reduction_store_vl`, runtime ABI order `lhs,acc,out,n`, route operand
      binding, required headers, and C type mapping before runtime.
- [x] The generated harness source calls the generated function as
      `lhs, acc, out, n`, initializes `lhs/acc/out`, computes a scalar
      reference, checks `acc` and source preservation, and protects scalar
      output sentinel lanes.
- [x] `ssh rvv` execution passes for explicit counts `0, 1, 16, 17, 257`, with
      seeds `-11,17` and patterns `0,1`.
- [x] Negative/preflight checks prove stale `reduction_kind`, stale
      accumulator layout, stale route operand binding, stale type mapping, and
      direct pre-realized route-entry / missing provider selected-body facts
      fail before executable runtime.
- [x] Bounded old-authority/q-name scan over touched files and added diff
      lines shows no new positive legacy `i32m1`, descriptor, source-front-door,
      route-string/artifact-name/ABI-string/test-name, or common EmitC semantic
      authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are recorded.
- [x] The task is finished/archived and the final commit records one coherent
      closure.

## Out of Scope

- Performance claims or benchmarking.
- Broad add/min/max/masked/widening matrices.
- High-level Linalg/Vector/StableHLO frontend lowering.
- Per-Linalg route authority or one-intrinsic wrapper routes.
- Common EmitC reduction semantics.
- New Gearbox/autotuning work.
- Source-front-door positive routes.
- Route-string, artifact-name, ABI-string, test-name, descriptor, intrinsic
  spelling, or metadata as route authority.

## Technical Notes

- Required specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  and `.trellis/spec/guides/*.md`.
- Primary implementation/evidence surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.
- Minimal runtime command shape:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --op-kind standalone_reduce_add \
    --runtime-count 0 \
    --runtime-count 1 \
    --runtime-count 16 \
    --runtime-count 17 \
    --runtime-count 257 \
    --overwrite
  ```

## Decision

Use the existing `standalone_reduce_add` typed route and generated-bundle script
as the executable closure surface if live checks confirm they already carry the
provider-derived reduction contract. Add only the smallest code or script change
needed to make the required contract/evidence explicit; otherwise, complete the
task by running focused compiler checks, negative checks, dry-run bundle
inspection, and real `ssh rvv` evidence with explicit runtime counts.

## Completion Evidence

- Live inspection showed no new compiler/source edit was required: the existing
  provider, target validator, generated-bundle script, and standalone
  reduction harness already carried the required provider-derived contract.
- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate`.
- Positive and stale-mirror FileCheck runs passed for
  `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`:
  `PLAN`, `HEADER`, `STALE-REDUCTION-KIND`, `STALE-REDUCTION-ACC`,
  `STALE-REDUCTION-BINDING`, and `STALE-REDUCTION-TYPE`.
- Script checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Dry-run generated-bundle evidence passed:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --dry-run \
    --pre-realized-selected-body \
    --op-kind standalone_reduce_add \
    --runtime-count 0 \
    --runtime-count 1 \
    --runtime-count 16 \
    --runtime-count 17 \
    --runtime-count 257 \
    --run-id standalone-reduce-add-dryrun \
    --overwrite
  ```

  Evidence path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/standalone-reduce-add-dryrun/standalone_reduce_add/evidence.json`.
  The reduction boundary records authority as provider-derived typed
  `tcrv_rvv` body/config/runtime facts, `reduction_kind = add`,
  `reduction_store_vl = 1`, runtime ABI order `lhs,acc,out,n`, scalar-result
  boundary
  `scalar-result-out0-seeded-before-loop-and-carried-across-runtime-vl-chunks.v1`,
  source/result vector type `!tcrv_rvv.vector<i32, "m1">`, C type
  `vint32m1_t`, and route operand binding:
  `lhs=...:abi|...|hdr`, `acc=...:abi|...|hdr`,
  `out=...:abi|...|hdr`, `n=...:abi|...|hdr`.
- Generated harness source:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/standalone-reduce-add-dryrun/standalone_reduce_add/rvv_generated_bundle_abi_standalone_reduce_add_harness.c`.
  It calls the generated function as `lhs, acc, out, n`, computes
  `(int32_t)(acc[0] + sum_i(lhs[i]))`, checks `acc[0]` preservation, checks
  source preservation, and checks scalar-output sentinel lanes beyond `out[0]`.
- Generated RVV C++:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/standalone-reduce-add-dryrun/standalone_reduce_add/materialized_rvv_emitc.cpp`.
  It seeds `out[0]` from `acc[0]`, uses runtime `n` with
  `__riscv_vsetvl_e32m1`, loads via `__riscv_vle32_v_i32m1`, reduces with
  `__riscv_vredsum_vs_i32m1_i32m1`, and stores lane 0 with
  `__riscv_vse32_v_i32m1(..., 1)`.
- Real RVV evidence passed:

  ```bash
  python3 scripts/rvv_generated_bundle_abi_e2e.py \
    --pre-realized-selected-body \
    --op-kind standalone_reduce_add \
    --runtime-count 0 \
    --runtime-count 1 \
    --runtime-count 16 \
    --runtime-count 17 \
    --runtime-count 257 \
    --run-id standalone-reduce-add-ssh \
    --overwrite
  ```

  Evidence path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/standalone-reduce-add-ssh/standalone_reduce_add/evidence.json`.
  Remote target was `ssh rvv`; remote compile and run both succeeded. The final
  runtime marker was:
  `PASS op=standalone_reduce_add counts=0,1,16,17,257 seeds=-11,17 patterns=0,1`.
  Runtime output covered all 20 combinations of counts `0,1,16,17,257`, seeds
  `-11,17`, and patterns `0,1`, with `tail_preserved source_preserved`.
- Direct pre-realized route-entry negative passed:
  `--direct-pre-realized-route-entry` with `standalone_reduce_add` failed
  before bundle export with the retired direct route-entry diagnostic requiring
  the public selected lowering-boundary producer.
- Additional checks passed:
  `build/bin/tcrv-opt test/Dialect/RVV/standalone-reduction-dataflow.mlir --split-input-file --verify-diagnostics`;
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- `tianchenrv-target-artifact-export-test` exited 0 with no output.
  `tianchenrv-rvv-extension-plugin-test` printed
  `RVV extension plugin smoke test passed`.
- Bounded scan over changed tracked content found only Trellis task files.
  There are no compiler/source added lines, so no new positive legacy
  `i32m1`, q-name, descriptor, source-front-door, artifact-name, route-string,
  ABI-string, test-name, intrinsic-spelling, or common EmitC semantic authority
  was introduced.
- `git diff --check` and `git diff --cached --check` passed.
- Spec update review: no `.trellis/spec/` update was needed. The existing RVV
  plugin, EmitC route, variant-pipeline, and testing specs already required
  typed-body authority, provider-owned standalone reduction facts, explicit
  runtime evidence, and mirror-only metadata.
