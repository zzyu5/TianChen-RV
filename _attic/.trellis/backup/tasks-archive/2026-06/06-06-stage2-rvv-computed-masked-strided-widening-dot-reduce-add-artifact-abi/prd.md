# Stage2 RVV computed-masked strided-input widening dot-reduce-add executable artifact ABI boundary

## Goal

Make the RVV `computed_masked_strided_input_widening_dot_reduce_add` selected-body route executable as generated RVV artifacts with truthful ABI/runtime evidence, or fail closed at the exact production boundary if any compare-mask fact, strided input memory fact, inactive-lane contribution policy, i16 source / i32 accumulator-result typing, scalar seed/result contract, runtime AVL/VL fact, per-operand ABI/header binding, RVV plugin route validation, EmitC materialization, target artifact mirror, or generated bundle ABI fact is stale or missing.

This round is scoped to one production workflow submodule:

```text
selected or pre-realized tcrv_rvv computed-mask strided-input widening dot-reduce body
  -> RVV plugin-owned compare/mask/stride/contraction facts
  -> direct contraction provider/statement plan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> generated bundle compile/run on ssh rvv when executable behavior is claimed
```

## What I already know

* Current branch at task creation: `main`.
* Initial worktree state at task creation: clean.
* Baseline commit at task creation: `be42fe3d rvv: prove computed masked widening dot reduce artifact abi`.
* No `.trellis/.current-task` existed at session start, so this task was created from the Hermes Direction Brief before source changes.
* The previous archived `computed_masked_widening_dot_reduce_add` task proved explicit and pre-realized generated bundles on `ssh rvv` with compare-produced mask facts, inactive-lane zero contribution, scalar seed/result behavior, and no production source changes.
* The previous archived `strided_input_widening_dot_reduce_add` task proved explicit and pre-realized generated bundles on `ssh rvv` with `lhs_stride`/`rhs_stride` ABI/header bindings, strided source load facts, and no production source changes.
* Current specs require RVV executable routes to start from typed `tcrv_rvv` body/config/runtime facts, RVV plugin legality/realization/provider facts, `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact export, and `ssh rvv` evidence for runtime/correctness claims.
* Common EmitC/export may carry provider payloads but must not infer RVV dtype, mask semantics, operation kind, ABI roles, stride semantics, intrinsic names, or schedules.
* Existing dry-run fixtures already cover explicit and pre-realized `computed_masked_strided_input_widening_dot_reduce_add` bundles, including runtime ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, operand-binding plan `rvv-route-operand-binding:masked_strided_wdot.v1`, `abi|cmp|mask|hdr` compare bindings, `abi|sld|mlhs/mrhs|i16|hdr` strided dot-source bindings, scalar seed/output bindings, `abi|str|addr|hdr` stride bindings, `computed-mask-strided-input-widening-dot-reduce` memory form, `__riscv_vlse16_v_i16mf2`, `__riscv_vwmul_vv_i32m1_m`, inactive-lane zeroing, and harness oracle checks for active/inactive lanes, signed widening products, source strides, scalar output, and tail preservation.
* Current production code already has the relevant direct-provider contraction owner surface: `getRVVSelectedBodyDirectContractionRouteProviderPlan` requires same-analysis contraction materialization, route-control, math operand binding, compare/dot/acc/out/n/stride ABI facts, and setvl/load/strided-load/compare/masked-product/merge/reduction/store leaves before route construction.
* Current statement-plan owner consumes the prevalidated direct contraction provider plan to build compare vector loads, strided source loads, compare mask, zero splat, masked widening product, inactive-lane zero merge, scalar seed/reduction, and scalar output store.
* Current target validation consumes `getRVVWideningDotReduceRouteValidationContract(...)` for computed-mask strided-input widening dot-reduce, then validates headers, type mappings, ABI mappings, runtime AVL/VL, statement counts, stride facts, mask facts, intrinsic facts, and candidate mirror metadata before accepting target artifacts.

## Requirements

* Scope is one production workflow submodule: `computed_masked_strided_input_widening_dot_reduce_add` from explicit selected body and pre-realized selected body through route planning, selected-body realization, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.
* Route authority must stay in typed `tcrv_rvv` body/config/runtime facts and RVV plugin-owned direct contraction provider facts. Route ids, helper names, artifact names, test names, metadata mirrors, exact intrinsic spellings, scripts, common EmitC code, and target-local constants must not become semantic authority.
* Confirm or harden compare lhs/rhs roles, dot lhs/rhs roles, lhs/rhs stride roles, accumulator seed role, scalar output role, runtime count role, compare predicate, mask role/source/form, inactive-lane zeroing policy, dot-product relation, source/result SEW/LMUL, tail/mask policy, runtime AVL/VL, header/prototype ABI order, route operand-binding plan/summary, direct statement-plan sequence, and target validation contract.
* If the current path is dry-run-only, stale, under-validated, or conflates stride binding, strided memory semantics, compare-mask construction, inactive-lane contribution, source/result dtype, scalar seed/result, ABI, or runtime AVL/VL facts, change the smallest coherent production seam needed to make the selected-body artifact executable or fail closed with targeted diagnostics.
* If production code already carries the complete seam, record an exact no-source-change justification and provide positive executable evidence rather than adding report-only, helper-only, or broad coverage changes.

## Acceptance Criteria

* [x] The explicit `computed_masked_strided_input_widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] The pre-realized `computed_masked_strided_input_widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] Provider/target validation consumes RVV-owned computed-mask strided-input widening dot-reduce contract facts for compare roles, dot source roles, stride roles, accumulator/output roles, mask provenance, inactive-lane zeroing, source/result type facts, route operand binding, runtime AVL/VL, statement-plan shape, headers/type mappings, ABI mappings, scalar seed/output, and stale cross-family residues.
* [x] Relevant generated-bundle dry-run and fail-closed tests for computed-mask strided-input widening dot-reduce pass.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] If runtime correctness is claimed, generated bundle compile/run passes on real `ssh rvv` with representative runtime counts including zero, small counts, vector-boundary counts, and multi-chunk counts, and with more than one stride pattern.
* [x] Bounded old-authority scan over touched files and added diff lines shows no new positive legacy `i32m1`, source-front-door, descriptor, direct-C, route-id, artifact-name, helper-name, script, test-name, metadata, or common-EmitC semantic authority.
* [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

* PRD, task context, and notes truthfully reflect the computed-masked strided-input widening dot-reduce executable artifact ABI module.
* Production source changes are made only if repository evidence shows the computed-mask strided-input seam is dry-run-only, stale, under-validated, or wrongly wired.
* Tests stay focused on changed computed-mask strided-input widening dot-reduce behavior.
* Trellis task is validated, finished, archived, and committed in one coherent commit if complete; otherwise it remains open with an exact continuation point.

## Technical Approach

1. Inspect relevant specs, the previous archived computed-mask and strided widening dot-reduce tasks, computed-mask strided fixtures, provider/statement/target validation code, and the generated bundle ABI runner.
2. Run explicit and pre-realized computed-mask strided-input generated-bundle evidence with representative counts, including zero, small counts, vector-boundary counts, and multi-chunk counts.
3. Ensure runtime evidence executes more than one stride pair and exercises both active and inactive mask lanes, signed i16 widening products, nonzero seed, scalar output, skipped-source checks, and tail preservation.
4. If evidence fails, classify the failure as script/harness, selected-body realization, EmitC generation, direct contraction provider validation, target validation, ABI/header binding, stride semantics, inactive-lane semantics, or remote RVV runtime.
5. Repair the smallest production seam needed, then rerun focused checks.
6. If evidence succeeds without source changes, record the exact no-source-change justification and finish with evidence plus focused validation.

## Decision (ADR-lite)

**Context**: The previous completed tasks separately proved computed-mask widening dot-reduce and strided-input widening dot-reduce artifact ABI boundaries. The next bottleneck is their combined route family: compare-produced masks plus runtime strided i16 dot sources in the same widening dot-reduce path.

**Decision**: Treat `computed_masked_strided_input_widening_dot_reduce_add` as the only required owner for this round. Use typed body/config/runtime facts and RVV direct contraction provider facts as authority. Use common EmitC/export only as neutral materialization and packaging.

**Consequences**: The round may be a production seam fix if runtime evidence exposes a stale or missing boundary, or a no-source-change executable evidence closeout if current production code is already correct. It must not become a broad dot/reduction/stride/dtype matrix, MAcc rework, computed-mask memory rework, performance tuning, frontend work, or report-only Trellis work.

## Out of Scope

* Broad dot/reduction/MAcc matrix, dtype/LMUL clone batches, product-reduce/dequant/clamp work, unrelated computed-mask memory or MAcc rework, performance tuning, dashboards, readiness reports, or source-front-door positive routes.
* Base non-masked widening dot-reduce, unit-stride computed-mask widening dot-reduce, or plain strided-input widening dot-reduce rework except as bounded references.
* High-level Linalg/Vector/StableHLO frontend work, per-Linalg route authority, common EmitC invention of RVV semantics, or descriptor-driven/source-export paths.
* Mass rewrite of memory, segment2, reduction, compare/select, widening conversion, or unrelated mask routes outside this artifact seam.

## Technical Notes

* Specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/index.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/index.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/guides/index.md`, `.trellis/spec/guides/plugin-locality-review-guide.md`, `.trellis/spec/guides/capability-first-design-guide.md`, and `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Prior task references read: `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-widening-dot-reduce-add-artifact-abi/prd.md` and `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-strided-input-widening-dot-reduce-add-artifact-abi/prd.md`.
* Initial code/fixture references inspected: `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`, `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, `scripts/rvv_generated_bundle_abi_e2e.py`, and computed-mask strided-input widening dot-reduce lit/MLIR fixtures.

## Completion Notes

Production source change justification:

* No compiler source change was required for the computed-masked strided-input widening dot-reduce-add executable artifact ABI seam. Current production code already derives and validates the route from RVV-owned typed body/config/runtime facts: compare `cmp_lhs/cmp_rhs` unit-stride i32 inputs, dot `lhs/rhs` element-strided i16/mf2 inputs, `lhs_stride/rhs_stride` runtime ABI values, i32/m1 accumulator/result boundary, scalar seed from `acc[0]`, scalar output through `out[0]`, SEW32 LMUL m1 policy, runtime AVL/VL, ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, compare predicate `slt`, compare-produced mask provenance, inactive-lane zeroing before reduction, and provider-owned route operand bindings.
* `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` requires same-analysis contraction materialization, route-control, and math operand-binding facts before route construction. For this route it requires compare lhs/rhs, dot lhs/rhs, accumulator, output, runtime count, lhs/rhs stride ABI bindings, plus setvl, store, contraction compute, source load, widening product, scalar seed splat, compare load, compare, masked widening product, masked merge, and strided source load leaves.
* `RVVEmitCStatementPlanOwners.cpp` builds the direct contraction statement sequence from the prevalidated provider plan: full-chunk `setvl`, seed splat from `acc[0]`, seed store to `out[0]`, loop `setvl`, compare vector loads, signed compare, strided dot input loads, zero vector splat, masked widening product, merge inactive lanes to zero, scalar seed from `out[0]`, horizontal reduction, and scalar store.
* `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the provider-owned widening dot-reduce validation contract and rejects candidate artifacts whose route token, headers, type mappings, ABI mappings, runtime AVL/VL contract, statement plan, computed-mask facts, strided-input facts, inactive-lane facts, or metadata mirrors disagree with provider facts. Candidate metadata remains mirror-only.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-computed-masked-strided-input-widening-dot-reduce-add-probe \
  --run-id explicit-computed-masked-strided-input-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind computed_masked_strided_input_widening_dot_reduce_add \
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
PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,7,16,17,23,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-computed-masked-strided-input-widening-dot-reduce-add-probe \
  --run-id pre-realized-computed-masked-strided-input-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind computed_masked_strided_input_widening_dot_reduce_add \
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
PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,7,16,17,23,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2
```

Both runs reported `ssh_evidence: true`, `status: success`, authority `provider-derived typed tcrv_rvv computed-mask widening dot-reduce body/config/runtime facts`, runtime ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, typed compute op `tcrv_rvv.masked_widening_dot_reduce`, memory form `computed-mask-strided-input-widening-dot-reduce`, compare predicate `slt`, mask role `predicate-mask-produced-by-compare`, mask source `compare-produced-mask-same-vl-scope`, mask memory form `compare-produced-mask`, inactive lane contract `compare-false lanes contribute zero before horizontal reduction`, active lane contract `compare-true lanes contribute signed i16*i16 widening products to the i32 scalar seed`, source type `i16/mf2`, result type `i32/m1`, route operand binding plan `rvv-route-operand-binding:masked_strided_wdot.v1`, strided input facts `lhs_stride_source = runtime_abi:lhs_stride`, `rhs_stride_source = runtime_abi:rhs_stride`, `source_memory_form = strided-load`, `destination_memory_form = unit-stride-store`, `strided_load_intrinsic = __riscv_vlse16_v_i16mf2`, and `runtime_counts_are_execution_cases_not_dot_or_mask_authority: true`.

Focused checks run:

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-fail-closed|explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add|pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add'` passed 5 selected tests.
* `git diff --check` and `git diff --cached --check`.
* Bounded old-authority scan over source diff found no production source changes. Bounded scan over added task diff lines found only negative/out-of-scope mentions of `descriptor`, `direct-C`, `source-front-door`, route ids, artifact names, helper names, scripts, tests, metadata, and common EmitC authority, with no positive legacy route authority.

Spec update decision:

* No `.trellis/spec/` update is needed. This round did not introduce a new route contract or convention; it revalidated the existing RVV plugin direct contraction, computed-mask strided-input widening dot-reduce statement plan, operand-binding, selected-body realization, and target validation contracts with executable artifact evidence.

## Current Phase

Finish.
