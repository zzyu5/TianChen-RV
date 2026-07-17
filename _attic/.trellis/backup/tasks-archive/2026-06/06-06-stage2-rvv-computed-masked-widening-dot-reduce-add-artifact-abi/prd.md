# Stage2 RVV computed-masked widening dot-reduce-add executable artifact ABI boundary

## Goal

Make the RVV computed-masked widening dot-reduce-add selected-body route executable as generated RVV artifacts with truthful ABI/runtime evidence, or fail closed at the exact production boundary if any compare-derived mask fact, inactive-lane policy, lhs/rhs/acc/out/n ABI binding, scalar seed/result boundary, dtype/SEW/LMUL/config/policy fact, runtime AVL/VL fact, route-family validation contract, EmitC statement plan, header/prototype binding, target artifact mirror, or generated bundle ABI fact is stale or missing.

This round is scoped to the base computed-mask unit-stride `computed_masked_widening_dot_reduce_add` seam:

```text
selected or pre-realized tcrv_rvv computed-mask widening dot-reduce body
  -> RVV plugin-owned compare/mask/contraction facts
  -> direct contraction provider/statement plan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> generated bundle compile/run on ssh rvv when executable behavior is claimed
```

## What I already know

* Current branch at task creation: `main`.
* Initial worktree state at task creation: clean.
* Baseline commit at task creation: `9f392efa rvv: prove strided widening dot reduce artifact abi`.
* No `.trellis/.current-task` existed at session start, so this task was created from the Hermes Direction Brief before source changes.
* The previous archived strided-input widening dot-reduce task closed with no compiler source changes: explicit and pre-realized generated bundles passed on `ssh rvv` with strided input ABI/header and validation facts already aligned.
* Current specs require RVV executable routes to start from typed `tcrv_rvv` bodies, RVV plugin legality/realization/provider facts, `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact export, and `ssh rvv` evidence for runtime/correctness claims.
* Common EmitC/export may carry provider payloads but must not infer RVV dtype, mask semantics, operation kind, ABI roles, intrinsic names, or schedules.
* Current production code already contains a computed-mask widening dot-reduce route-family surface: runtime ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, operand-binding plan `rvv-route-operand-binding:masked_widening_dot_reduce.v1`, typed compute op `tcrv_rvv.masked_widening_dot_reduce`, compare predicate `slt`, mask source `compare-produced-mask-same-vl-scope`, mask role `predicate-mask-produced-by-compare`, mask memory form `compare-produced-mask`, inactive-lane zeroing requirement `masked-widening-products-zero-inactive-lanes-before-reduction`, source `i16/mf2`, result `i32/m1`, scalar seed from `acc[0]`, scalar result store through `out[0]`, runtime AVL/VL, and target artifact validation hooks.
* Existing generated-bundle tests for explicit and pre-realized computed-mask widening dot-reduce are dry-run tests; runtime correctness needs focused executable evidence if the production seam is complete.

## Requirements

* Scope is one production workflow submodule: `computed_masked_widening_dot_reduce_add` from explicit selected body and pre-realized selected body through route planning, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.
* Route authority must stay in typed `tcrv_rvv` body/config/runtime facts and RVV plugin-owned contraction provider facts. Route ids, helper names, artifact names, test names, metadata mirrors, exact intrinsic spellings, scripts, and common EmitC code must not become semantic authority.
* Confirm or harden compare lhs/rhs roles, dot lhs/rhs roles, accumulator seed role, scalar output role, runtime count role, compare predicate, mask role/source/form, inactive-lane zeroing policy, dot-product relation, source/result SEW/LMUL, tail/mask policy, runtime AVL/VL, header/prototype ABI order, route operand-binding plan/summary, direct statement-plan sequence, and target validation contract.
* If the current path is only dry-run-supported or under-validated, change the smallest coherent production seam needed to make the selected-body artifact executable or fail closed with targeted diagnostics.
* If production code already carries the complete seam, record an exact no-source-change justification and provide positive executable evidence rather than adding report-only, helper-only, or broad coverage changes.

## Acceptance Criteria

* [x] The explicit `computed_masked_widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] The pre-realized `computed_masked_widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] Provider/target validation consumes RVV-owned computed-mask widening dot-reduce contract facts for compare roles, dot source roles, accumulator/output roles, mask provenance, inactive-lane zeroing, source/result type facts, route operand binding, runtime AVL/VL, statement-plan shape, headers/type mappings, ABI mappings, scalar seed/output, and stale cross-family residues.
* [x] Relevant generated-bundle dry-run and fail-closed tests for computed-mask widening dot-reduce pass.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] If runtime correctness is claimed, generated bundle compile/run passes on real `ssh rvv` with representative runtime counts including zero, small counts, vector-boundary counts, and multi-chunk counts.
* [x] Bounded old-authority scan over touched files and added diff lines shows no new positive legacy `i32m1`, source-front-door, descriptor, direct-C, route-id, artifact-name, helper-name, or common-EmitC semantic authority.
* [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

* PRD, task context, and notes truthfully reflect the computed-masked widening dot-reduce executable artifact ABI module.
* Production source changes are made only if repository evidence shows the computed-mask seam is dry-run-only, stale, under-validated, or wrongly wired.
* Tests stay focused on changed computed-mask widening dot-reduce behavior.
* Trellis task is validated, finished, archived, and committed in one coherent commit if complete; otherwise it remains open with an exact continuation point.

## Technical Approach

1. Inspect relevant specs, the previous archived strided dot-reduce task, computed-mask fixtures, provider/statement/target validation code, and the generated bundle ABI runner.
2. Run explicit and pre-realized computed-mask generated-bundle evidence with representative counts including zero, small counts, vector-boundary counts, and multi-chunk counts.
3. If evidence fails, classify the failure as script/harness, selected-body realization, EmitC generation, direct contraction provider validation, target validation, ABI/header binding, inactive-lane semantics, or remote RVV runtime.
4. Repair the smallest production seam needed, then rerun focused checks.
5. If evidence succeeds without source changes, record the exact no-source-change justification and finish with evidence plus focused validation.

## Decision (ADR-lite)

**Context**: The previous completed task proved the strided-input widening dot-reduce-add artifact ABI seam. The next bottleneck is the same widening dot-reduce family with compare-produced masks and inactive-lane behavior before combining computed masks with strided inputs.

**Decision**: Treat `computed_masked_widening_dot_reduce_add` as the only required owner for this round. Use typed body/config/runtime facts and RVV direct contraction provider facts as authority. Use common EmitC/export only as neutral materialization and packaging.

**Consequences**: The round may be a production seam fix if runtime evidence exposes a stale or missing boundary, or a no-source-change executable evidence closeout if current production code is already correct. It must not become computed-mask strided-input expansion, broad dot-reduce matrix work, or report-only Trellis work.

## Out of Scope

* Computed-masked strided-input widening dot-reduce-add in this round.
* Broad dot-reduce matrix, dtype/LMUL clone batches, product-reduce/dequant/MAcc rework, performance tuning, dashboards, readiness reports, or source-front-door positive routes.
* Base non-masked widening dot-reduce or strided-input widening dot-reduce rework except as reference.
* High-level frontend work, per-Linalg route authority, common EmitC invention of RVV semantics, or descriptor-driven/source-export paths.

## Technical Notes

* Specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/index.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/index.md`, `.trellis/spec/variant-pipeline/index.md`, `.trellis/spec/guides/index.md`, `.trellis/spec/guides/plugin-locality-review-guide.md`, and `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Prior task reference read: `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-strided-input-widening-dot-reduce-add-artifact-abi/prd.md`.
* Initial code/fixture references inspected: `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, `scripts/rvv_generated_bundle_abi_e2e.py`, and computed-mask widening dot-reduce lit/MLIR fixtures.

## Completion Notes

Production source change justification:

* No compiler source change was required for the computed-masked widening dot-reduce-add executable artifact ABI seam. Current production code already derives and validates the route from RVV-owned typed body/config/runtime facts: compare `cmp_lhs/cmp_rhs` unit-stride i32 inputs, dot `lhs/rhs` unit-stride i16/mf2 inputs, i32/m1 accumulator/result boundary, scalar seed from `acc[0]`, scalar output through `out[0]`, SEW32 LMUL m1 policy, runtime AVL/VL, ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, compare predicate `slt`, compare-produced mask provenance, inactive-lane zeroing before reduction, and provider-owned route operand bindings.
* `RVVEmitCContractionRouteFamilyPlanOwners.cpp` validates the computed-mask widening dot-reduce description against canonical provider facts before route construction, including runtime ABI order, route operand-binding plan `rvv-route-operand-binding:masked_widening_dot_reduce.v1`, exact binding summary, contraction route-family plan, typed compute op `tcrv_rvv.masked_widening_dot_reduce`, source/result SEW/LMUL, mask role/source/form, inactive-lane zeroing requirement, widening dot relation, masked widening product leaf, zero merge leaf, reduction/store/setvl leaves, headers, type mappings, and stale strided/non-family residues.
* The pre-realized selected-body validator requires the computed-mask body to be a direct child of the selected variant, to use `op_kind = signed_masked_widening_dot_reduce_add`, `predicate_kind = slt`, `memory_form = computed-mask-unit-stride-widening-dot-reduce`, compare-produced mask facts, tail/mask agnostic policy, correct `cmp_lhs/cmp_rhs/lhs/rhs/acc/out/n` runtime ABI roles and C types, and `tcrv_rvv.compare` plus `tcrv_rvv.masked_widening_dot_reduce` realization.
* `RVVEmitCStatementPlanOwners.cpp` builds the direct contraction statement sequence from the provider plan: full-chunk `setvl`, seed splat from `acc[0]`, seed store to `out[0]`, loop `setvl`, compare vector loads, signed compare, dot input loads, zero vector splat, masked widening product, merge inactive lanes to zero, scalar seed from `out[0]`, horizontal reduction, and scalar store.
* `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the provider-owned widening dot-reduce validation contract and rejects candidate artifacts whose route id, headers, type mappings, ABI mappings, statement plan, computed-mask facts, inactive-lane facts, or metadata mirrors disagree with provider facts. Candidate metadata remains mirror-only.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-computed-masked-widening-dot-reduce-add-probe \
  --run-id explicit-computed-masked-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind computed_masked_widening_dot_reduce_add \
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
PASS op=computed_masked_widening_dot_reduce_add counts=0,1,7,16,17,23,257
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-computed-masked-widening-dot-reduce-add-probe \
  --run-id pre-realized-computed-masked-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind computed_masked_widening_dot_reduce_add \
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
PASS op=computed_masked_widening_dot_reduce_add counts=0,1,7,16,17,23,257
```

Both runs reported `ssh_evidence: true`, `status: success`, runtime ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, mask role `predicate-mask-produced-by-compare`, mask source `compare-produced-mask-same-vl-scope`, mask memory form `compare-produced-mask`, inactive lane contract `compare-false lanes contribute zero before horizontal reduction`, active lane contract `compare-true lanes contribute signed i16*i16 widening products to the i32 scalar seed`, and `runtime_counts_are_execution_cases_not_dot_or_mask_authority: true`.

Focused checks run:

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-explicit-computed-masked-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-widening-dot-reduce-add-fail-closed|explicit-selected-body-artifact-computed-masked-widening-dot-reduce-add|pre-realized-selected-body-artifact-computed-masked-widening-dot-reduce-add'` passed 5 selected tests.
* `git diff --check` and `git diff --cached --check`.
* Bounded old-authority scan over source diff found no production source changes. Bounded scan over added task diff lines found only negative/out-of-scope mentions of `descriptor`, `direct-C`, and `source-front-door`, with no positive legacy route authority.

Spec update decision:

* No `.trellis/spec/` update is needed. This round did not introduce a new route contract or convention; it revalidated the existing RVV plugin direct contraction, computed-mask widening dot-reduce statement plan, operand-binding, and target validation contracts with executable artifact evidence.

## Current Phase

Finish.
