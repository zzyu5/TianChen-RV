# Stage2 RVV strided-input widening dot-reduce-add executable artifact ABI boundary

## Goal

Make the RVV strided-input widening dot-reduce-add selected-body route executable as generated RVV artifacts with truthful ABI/runtime evidence, or fail closed at the exact production boundary if any stride value, strided memory-form fact, lhs/rhs input role, widened product type, reduction accumulator/result role, dtype/SEW/LMUL/config/policy fact, runtime AVL/VL fact, ABI/header/prototype binding, route-family validation contract, EmitC statement plan, or target artifact mirror is missing or stale.

This round is scoped to the base strided-input `strided_input_widening_dot_reduce_add` seam:

```text
selected or pre-realized tcrv_rvv strided widening dot-reduce body
  -> RVV plugin-owned contraction family facts
  -> direct contraction provider/statement plan with lhs/rhs stride bindings
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> generated bundle compile/run on ssh rvv when executable behavior is claimed
```

## What I already know

* Current branch at task creation: `main`.
* Initial worktree state at task creation: clean.
* Baseline commit at task creation: `7773648e rvv: prove widening dot reduce artifact abi`.
* The previous archived task closed the non-strided `widening_dot_reduce_add` executable artifact ABI seam with explicit and pre-realized `ssh rvv` evidence and no production source change.
* Specs require widening dot-reduce target validation to consume provider-owned route validation contracts for plain, strided-input, computed-mask, and computed-mask-strided variants. Target validation must not reconstruct truth from route names, metadata, scripts, C strings, artifact names, or intrinsic spellings.
* Specs require operand-binding summaries to include every generated header/prototype ABI parameter; exported stride parameters must carry provider-derived `abi` and `hdr` participation.
* Current production owner constants include the strided ABI order `lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, memory layout `element-strided-lhs-rhs-dot-source-unit-stride-output-runtime-abi`, stride sources `runtime_abi:lhs_stride` and `runtime_abi:rhs_stride`, source memory form `strided-load`, destination memory form `unit-stride-store`, and strided load intrinsic `__riscv_vlse16_v_i16mf2`.
* Current route operand binding for this family includes `lhs_stride=lhs-input-stride:lhs_stride:abi|str|addr|hdr` and `rhs_stride=rhs-input-stride:rhs_stride:abi|str|addr|hdr`.
* The explicit and pre-realized strided MLIR fixtures expose typed `tcrv_rvv` bodies with `i16/mf2` strided loads, `i32/m1` dot reduction, scalar seed/result layout, runtime `n`, `lhs_stride`, and `rhs_stride`.
* Existing generated-bundle lit tests are dry-run only. They check metadata, header/prototype ABI order, harness reference logic, and fail-closed direct pre-realized route-entry behavior, but their root evidence records `ssh_evidence: false`.

## Requirements

* Scope is one production workflow submodule: `strided_input_widening_dot_reduce_add` from explicit selected body or pre-realized selected body through route planning, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.
* Keep route authority local to typed `tcrv_rvv` body/config/runtime facts and RVV plugin-owned contraction provider facts. Route ids, helper names, artifact names, test names, metadata mirrors, exact intrinsic spellings, scripts, and common EmitC code must not become semantic authority.
* Confirm or harden lhs/rhs source roles, lhs/rhs stride ABI roles, strided memory form, stride source facts, byte/element address calculation as exposed by the route, widened product relation, accumulator seed role, scalar result/output role, source/result SEW/LMUL, tail/mask policy, runtime AVL/VL, header/prototype ABI order, operand-binding plan/summary, statement-plan counts and wiring, and target validation contract.
* If the current path is only dry-run-supported or under-validated, change the production seam in the smallest coherent way that makes the selected-body artifact executable or fail closed with targeted diagnostics.
* If production code already carries the complete seam, record an exact no-source-change justification and provide positive executable evidence rather than adding report-only or helper-only changes.
* Do not expand this round to computed-mask or computed-mask-strided dot-reduce unless the base strided seam is already complete and the same coherent executable boundary fits without diluting this task.

## Acceptance Criteria

* [x] The explicit `strided_input_widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] The pre-realized `strided_input_widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] Provider/target validation consumes RVV-owned strided widening dot-reduce contract facts for lhs/rhs source roles, lhs/rhs stride roles and sources, source/destination memory forms, source/accumulator/result type facts, route operand binding, runtime AVL/VL, statement-plan shape, headers/type mappings, ABI mappings, scalar seed/output, and stale cross-family residues.
* [x] Relevant generated-bundle dry-run and fail-closed tests for strided-input widening dot-reduce pass.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] If runtime correctness is claimed, generated bundle compile/run passes on real `ssh rvv` with representative runtime counts and both stride pairs `2:3` and `3:2`.
* [x] Bounded old-authority scan over touched files and added diff lines shows no new positive legacy `i32m1`, source-front-door, descriptor, direct-C, route-id, artifact-name, helper-name, or common-EmitC semantic authority.
* [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

* PRD, task context, and notes truthfully reflect the strided-input widening dot-reduce executable artifact ABI module.
* Production source changes are made only if repository evidence shows the strided seam is dry-run-only, stale, under-validated, or wrongly wired.
* Tests stay focused on changed strided-input behavior.
* Trellis task is validated, finished, archived, and committed in one coherent commit if complete; otherwise it remains open with an exact continuation point.

## Technical Approach

1. Inspect relevant specs, the prior archived base dot-reduce task, current strided fixtures, provider/statement/target validation code, and the generated bundle ABI runner.
2. Run explicit and pre-realized strided generated-bundle evidence with representative counts, including zero, small counts, vector-boundary counts, and multi-chunk counts.
3. If evidence fails, classify the failure as script/harness, EmitC generation, route/provider validation, target validation, ABI/header binding, stride address binding, or remote RVV runtime.
4. Repair the smallest production seam needed, then rerun focused checks.
5. If evidence succeeds without source changes, record the exact no-source-change justification and finish with evidence plus focused validation.

## Decision (ADR-lite)

**Context**: The previous completed task proved the base unit-stride widening dot-reduce-add artifact ABI seam. The next bottleneck is the same contraction/reduction family with strided lhs/rhs input memory and explicit stride ABI binding.

**Decision**: Treat `strided_input_widening_dot_reduce_add` as the only required owner for this round. Use typed body/config/runtime facts and RVV direct contraction provider facts as authority. Use common EmitC/export only as neutral materialization and packaging.

**Consequences**: The round may be a production seam fix if runtime evidence exposes a stale or missing boundary, or a no-source-change executable evidence closeout if current production code is already correct. It must not become computed-mask expansion, broad dot-reduce matrix work, or report-only Trellis work.

## Out of Scope

* Computed-mask or computed-mask-strided widening dot-reduce expansion unless the base strided seam is already complete and the same boundary can be closed coherently.
* Broad dot-reduce matrix, dtype/LMUL clone batches, performance tuning, dashboards, readiness reports, or source-front-door positive routes.
* Base non-strided widening dot-reduce rework except as reference.
* MAcc, scalar-broadcast, product-reduce, dequant, clamp, standalone reduction, memory, segment2, compare/select, widening conversion, Linalg/Vector/StableHLO frontend, or per-Linalg route authority.
* Common EmitC invention of RVV semantics.

## Technical Notes

* Specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
* Prior task reference read: `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-widening-dot-reduce-add-artifact-abi/prd.md`.
* Initial code/fixture references inspected: `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`, `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, `scripts/rvv_generated_bundle_abi_e2e.py`, and strided-input widening dot-reduce lit/MLIR fixtures.

## Completion Notes

Production source change justification:

* No compiler source change was required for the strided-input widening dot-reduce-add executable artifact ABI seam. Current production code already derives and validates the strided route from RVV-owned typed body/config/runtime facts: `i16/mf2` lhs/rhs strided source loads, `i32/m1` accumulator/result facts, scalar seed from `acc[0]`, scalar result store through `out[0]`, SEW32 LMUL m1 policy, runtime AVL/VL, ABI order `lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, and provider-owned stride operand bindings.
* `RVVEmitCContractionRouteFamilyPlanOwners.cpp` validates the strided widening dot-reduce description against canonical provider facts before route construction, including memory form, runtime control plan, runtime ABI order, target leaf profile, required headers, C type summary, operand-binding plan/summary, contraction route-family plan, typed compute op, source/result SEW/LMUL, strided memory layout, lhs/rhs stride sources, widening dot relation, product/reduction/store/setvl intrinsics, strided load intrinsic, vector C types, and runtime ABI parameters.
* The pre-realized selected-body validator requires the strided-input body to be a direct child of the selected variant, to use `op_kind = signed_widening_dot_reduce_add`, `memory_form = strided-input-widening-dot-reduce`, `stride_unit = element`, tail/mask agnostic policy, correct lhs/rhs/acc/out/n/lhs_stride/rhs_stride runtime ABI roles, and `StridedLoadOp` plus `WideningDotReduceOp` realization.
* `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the provider-owned widening dot-reduce validation contract and rejects rebuilt routes whose route id, headers, type mappings, ABI mappings, statement plan, strided facts, or metadata mirrors disagree with provider facts. Candidate metadata remains mirror-only.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-strided-widening-dot-reduce-add-probe \
  --run-id explicit-strided-input-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind strided_input_widening_dot_reduce_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 17 --runtime-count 23 \
  --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=strided_input_widening_dot_reduce_add counts=0,1,7,16,17,23,257 stride_pairs=2:3,3:2 data_patterns=2
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-strided-widening-dot-reduce-add-probe \
  --run-id pre-realized-strided-input-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind strided_input_widening_dot_reduce_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 17 --runtime-count 23 \
  --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=strided_input_widening_dot_reduce_add counts=0,1,7,16,17,23,257 stride_pairs=2:3,3:2 data_patterns=2
```

Both runs reported `ssh_evidence: true`, `status: success`, runtime ABI order `lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, provider route facts, strided input facts, target validator consumed facts, and `runtime_counts_are_execution_cases_not_widening_dot_authority: true`. The pre-realized run additionally recorded `materializer: tcrv-materialize-selected-lowering-boundaries` and `pre_realized_body_consumed: true`.

Focused checks run:

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-explicit-strided-input-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-strided-input-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-strided-input-widening-dot-reduce-add-fail-closed|explicit-selected-body-artifact-strided-input-widening-dot-reduce-add|pre-realized-selected-body-artifact-strided-input-widening-dot-reduce-add'` passed 5 selected tests.
* `git diff --check` and `git diff --cached --check`.
* Bounded old-authority scan over added task diff lines found no positive legacy `i32m1`, `RVVI32M1`, `tcrv_rvv.i32_`, source-front-door, descriptor, direct-C, source-export, or `rvv-i32m1` authority.

Spec update decision:

* No `.trellis/spec/` update is needed. This round did not introduce a new route contract or convention; it revalidated the existing RVV plugin direct contraction and strided widening dot-reduce target validation contracts with executable artifact evidence.

## Current Phase

Finish.
