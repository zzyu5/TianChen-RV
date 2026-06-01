# Stage2 RVV widening dot-reduction accumulator artifact ABI boundary

## Goal

Implement one bounded Stage 2 end-to-end artifact/runtime ABI path for the existing pre-realized `widening_dot_reduce_add` selected RVV body. The production path must run through the RVV plugin-owned contraction route family, derive route facts from the typed selected `tcrv_rvv` body, materialize through the common EmitC route, produce an RVV target artifact bundle, and only claim runtime correctness with real `ssh rvv` evidence.

## What I already know

* The current task brief is the direction source: this round is exactly about `widening_dot_reduce_add`, not a broad Stage 2 expansion.
* The previous completed task proved `widen_i16_to_i32` conversion artifact ABI with provider-derived source/result type facts, conversion relation, route facts, tail policy, direct-route fail-closed evidence, two input patterns, and `ssh rvv` correctness.
* The expected production chain is selected `tcrv.exec` RVV variant -> RVV plugin-owned realized `tcrv_rvv` widening dot/reduction body -> contraction route-family provider facts -> `TCRVEmitCLowerableRoute` -> common EmitC materialization -> RVV target artifact bundle -> `ssh rvv` correctness evidence.
* The module owner files named by the brief include RVV contraction plan owners, construction protocol, route planning/provider, common EmitC materializer, RVV target artifact validation/support bundle, and the generated bundle ABI / remote probe scripts.
* This is not allowed to close as a report-only, harness-only, broad smoke, or helper-only task.

## Assumptions

* A pre-realized selected-body fixture already exists at `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`.
* The route skeleton for contraction/reduction likely exists but needs either production validation/route-fact guards or generated-artifact ABI propagation.
* Runtime correctness must be demonstrated on the real RVV machine via the repository's existing `ssh rvv` tooling.

## Requirements

* Validate or realize the selected `widening_dot_reduce_add` body before route construction in the RVV plugin/contraction owner.
* Preserve into provider route facts and generated bundle evidence:
  * source vector roles,
  * signed widening dot relation,
  * accumulator/reduction layout,
  * scalar or store-backed result ABI,
  * runtime `n`/AVL binding,
  * dtype/config facts including SEW/LMUL and source/result widths,
  * VL and tail policy.
* Keep common EmitC/export code semantically neutral. RVV semantics must come from the RVV provider/route facts, not artifact names, route IDs, manifests, test names, descriptors, C strings, or exact intrinsic spelling.
* Add or preserve fail-closed diagnostics for direct pre-realized route/artifact export or missing source/accumulator/result binding.
* Add focused tests and scripts only for this owner path.

## Acceptance Criteria

* [x] The RVV plugin/contraction owner validates the pre-realized `widening_dot_reduce_add` selected body before constructing a route.
* [x] Focused test evidence shows source roles, signed widening dot relation, accumulator/reduction layout, result ABI, runtime `n`/AVL, dtype/config, and tail policy survive into route facts and generated artifact bundle metadata/evidence.
* [x] Generated artifact compile/run correctness is demonstrated via `ssh rvv` for representative counts including `0`, `1`, a VL-boundary count, a tail count, and a larger count.
* [x] Input data distinguishes widening multiply-accumulate/reduction semantics from plain add/copy behavior.
* [x] Direct pre-realized route/artifact export and missing source/accumulator/result bindings fail closed with targeted diagnostics.
* [x] A bounded old-authority scan over touched plugin/provider/materializer/target/script/test/spec files classifies any remaining hits for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, and `selected route`.
* [x] Focused build/test/script checks pass.
* [x] Trellis task status, context, and journal are truthful, and the final result is committed as one coherent commit if complete.

## Non-Goals

* No computed-mask, strided-input, dtype/LMUL clone batch, widening conversion closeout, compare/select expansion, high-level frontend authority, per-Linalg lowering, source-front-door positive route, common EmitC semantic inference, dashboard, broad smoke matrix, or performance claim.
* No new dtype-prefixed helper op family or old `i32m1` route-authority compatibility wrapper.
* No treating route IDs, artifact names, manifests, test names, C strings, descriptors, or exact intrinsic spellings as authority for dtype, accumulator layout, reduction semantics, policy, or ABI.

## Definition of Done

* The production route-owner path, not only harness/report code, enforces the ABI boundary.
* Focused MLIR/FileCheck, script dry-run, and generated bundle ABI checks pass.
* `ssh rvv` correctness evidence is captured when runtime behavior is claimed, or the exact external blocker is documented without claiming runtime correctness.
* Task is finished/archived per repository convention if complete.
* Worktree is clean after one coherent commit, unless an exact blocker prevents completion.

## Technical Notes

* Required reading from brief: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/mlir-testing-contract.md`, archived task `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-widening-conversion-artifact-abi/`, `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, RVV contraction/provider/materializer/target files, `scripts/rvv_generated_bundle_abi_e2e.py`, `scripts/rvv_remote_probe.py`, and the focused tests named in the brief.
* Live repository inspection and exact command evidence will be appended as the round proceeds.

## Completion Notes

* The production C++ path already used the intended owner chain for the selected pre-realized `widening_dot_reduce_add` body: RVV plugin-local validation/realization, contraction route-family owner, math operand-binding facts, direct contraction provider plan, `TCRVEmitCLowerableRoute`, common EmitC materialization, and target-owned widening-dot route-family validation.
* This round pinned the missing focused guard/evidence in two places:
  * `test/Target/TargetArtifactExportTest.cpp` now separately mutates plain widening-dot `lhs` and `out` runtime ABI roles and proves the target artifact route-family validator rejects stale source/result binding before artifact acceptance.
  * `scripts/rvv_generated_bundle_abi_e2e.py` self-test now verifies the widening-dot generated-bundle boundary summary preserves selected ABI roles, provider route facts, statement-plan seed/carry/store facts, direct route-entry unsupported status, and runtime counts.
* The pre-realized dry-run script test now uses counts `0,1,16,23,257` and FileChecks the `selected_source_abi`, `statement_plan`, and runtime-count evidence fields for `widening_dot_reduce_add`.
* Final dry-run evidence:
  `artifacts/tmp/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/focused-dry-run/pre-realized-widening-dot-reduce-add`.
  It records `input_mode = pre-realized-selected-body`, `pre_realized_body_consumed = true`, source ABI roles `lhs,rhs,acc,out,n`, provider route facts `rvv-route-operand-binding:widening_dot_reduce.v1`, statement-plan seed `acc[0]`, loop carry `out[0]`, scalar store VL `1`, and `direct_pre_realized_route_entry_supported = false`.
* Direct pre-realized route-entry negative evidence:
  `artifacts/tmp/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/focused-direct-fail.log`.
  The command failed as expected with:
  `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): widening_dot_reduce_add`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/final-ssh-rvv/pre-realized-widening-dot-reduce-add`.
  Counts `0,1,16,23,257` passed with `signed_horizontal_dot`, `seed_added`, `scalar_output`, `tail_preserved`, widening-products evidence, add-only distinguishing checks, and mul-only/no-seed distinguishing checks.
* `llvm-lit` is not installed in this environment, so the focused script lit file was verified with the same underlying `FileCheck` invocations for the generated `ROOT`, `WDOT`, and `HARNESS` outputs.
* Bounded added-line old-authority scan:
  `git diff -U0 -- ... | rg '^\\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route)'`
  found only three added exact intrinsic strings in the script self-test:
  `__riscv_vwmul_vv_i32m1`, `__riscv_vmv_v_x_i32m1`, and `__riscv_vredsum_vs_i32m1_i32m1`. These are provider-derived emitted-C evidence checks inside `provider_route_facts`; they are not route, dtype, or ABI authority.

## Checks

* [x] `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [x] `rtk git diff --check`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/focused-dry-run --run-id pre-realized-widening-dot-reduce-add --overwrite --op-kind widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
* [x] Direct pre-realized route-entry negative command with `--direct-pre-realized-route-entry --op-kind widening_dot_reduce_add` failed with the expected unsupported shortcut diagnostic.
* [x] FileCheck equivalent for `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run.test`: `ROOT`, `WDOT`, and `HARNESS` prefixes against focused dry-run artifacts.
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries | FileCheck ... --check-prefix=REALIZED`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck ... --check-prefix=PLAN`
* [x] `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | FileCheck ... --check-prefix=HEADER`
* [x] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/final-ssh-rvv --run-id pre-realized-widening-dot-reduce-add --overwrite --op-kind widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-opt --tcrv-translate /home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-widening-dot-reduce-artifact-abi`
