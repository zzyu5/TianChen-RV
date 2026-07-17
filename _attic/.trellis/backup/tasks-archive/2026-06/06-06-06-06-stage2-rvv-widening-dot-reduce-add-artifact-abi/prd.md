# Stage2 RVV widening dot-reduce-add executable artifact ABI boundary

## Goal

Make the base RVV widening dot-reduce-add selected-body route executable as generated RVV artifacts with truthful ABI/runtime evidence, or fail closed at the exact production boundary if any typed body fact, paired narrow vector input role, widened product/result type, horizontal reduce-add accumulator/result role, SEW/LMUL/config/policy fact, runtime AVL/VL fact, ABI/header/prototype binding, route-family validation contract, EmitC statement plan, or target artifact mirror is missing or stale.

This round is scoped to the base `widening_dot_reduce_add` seam only. The purpose is to prove or harden the path:

```text
selected or pre-realized tcrv_rvv widening dot-reduce body
  -> RVV plugin-owned contraction family facts
  -> direct contraction provider/statement plan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact bundle
  -> generated bundle compile/run on ssh rvv when executable behavior is claimed
```

## What I already know

* Current branch at task creation: `main`.
* Initial worktree state at task creation: clean.
* Recent baseline commit: `0a87204e rvv: prove runtime scalar masked macc lmul m2 artifact abi`.
* Relevant specs place `widening_dot_reduce_add` under direct RVV contraction provider ownership, not common EmitC semantics.
* The base explicit and pre-realized fixtures already model `lhs,rhs,acc,out,n` ABI order, `i16/mf2` dot source loads, `i32/m1` accumulator/result, `signed_widening_dot_reduce_add`, scalar seed from `acc[0]`, scalar output through `out[0]`, and runtime AVL/VL.
* Existing script/lit coverage for base widening dot-reduce generates bundle harnesses in dry-run mode and records provider/target metadata, but the named dry-run tests do not themselves prove remote runtime execution.
* Prior memory warns that recent RVV rounds sometimes drifted into evidence-only Trellis changes; this PRD requires a source/no-source decision grounded in the production seam inspection.

## Requirements

* Scope is one production workflow submodule: base `widening_dot_reduce_add` from typed `tcrv_rvv` selected body or pre-realized body through route planning, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.
* Keep route authority local to typed `tcrv_rvv` body/config/runtime facts and RVV plugin-owned contraction provider facts. Route ids, helper names, artifact names, test names, metadata mirrors, exact intrinsic spellings, and common EmitC code must not become semantic authority.
* Confirm or harden paired narrow source roles (`lhs`, `rhs`), widened product/reduction relation, accumulator seed role, scalar result/output role, source/result SEW/LMUL, policy, runtime AVL/VL, header/prototype ABI order, operand-binding plan/summary, statement-plan counts and wiring, and target validation contract.
* If the current path is only dry-run-supported or under-validated, change the production seam in the smallest coherent way that makes the base selected-body artifact executable or fail closed with targeted diagnostics.
* If production code already carries the complete seam, record an exact no-source-change justification and provide positive executable evidence rather than adding report-only or helper-only changes.
* Do not expand this round to strided-input, computed-mask, computed-mask-strided, MAcc, product-reduce, dequant, clamp, unrelated reduction, memory, compare/select, or widening conversion routes.

## Acceptance Criteria

* [x] The base explicit `widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] The base pre-realized `widening_dot_reduce_add` generated bundle either compiles/runs on `ssh rvv` with correctness evidence or fails closed at the precise stale/missing executable-boundary fact.
* [x] Provider/target validation consumes RVV-owned widening dot-reduce contract facts for source/accumulator/result type facts, route operand binding, runtime AVL/VL, statement-plan shape, headers/type mappings, ABI mappings, scalar seed/output, and stale cross-family residues.
* [x] Relevant dry-run and fail-closed tests for base widening dot-reduce pass.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] If runtime correctness is claimed, generated bundle compile/run passes on real `ssh rvv` with representative runtime counts including boundaries around vector chunking.
* [x] Bounded old-authority scan over touched files and added diff lines shows no new positive legacy `i32m1`, source-front-door, descriptor, direct-C, route-id, artifact-name, helper-name, or common-EmitC semantic authority.
* [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

* PRD, task context, and notes truthfully reflect the base widening dot-reduce executable artifact ABI module.
* Production source changes are made only if repository evidence shows the base seam is dry-run-only, stale, under-validated, or wrongly wired.
* Tests stay focused on changed base behavior.
* Trellis task is validated, finished, archived, and committed in one coherent commit if complete; otherwise it remains open with an exact continuation point.

## Technical Approach

1. Inspect relevant specs, the prior archived MAcc task, current widening dot-reduce fixtures, provider/statement/target validation code, and the generated bundle ABI runner.
2. Start from executable evidence attempts for explicit and pre-realized base `widening_dot_reduce_add` with representative runtime counts.
3. If evidence fails, classify the failure as script/harness, EmitC generation, route/provider validation, target validation, ABI/header binding, or remote RVV runtime.
4. Repair the smallest production seam needed, then rerun focused checks.
5. If evidence succeeds without source changes, record the exact no-source-change justification and finish with evidence plus focused validation.

## Decision (ADR-lite)

**Context**: The previous completed task closed the runtime-scalar-cmp masked MAcc LMUL m2 artifact ABI boundary. The next bottleneck is contraction plus horizontal reduction: `widening_dot_reduce_add`.

**Decision**: Treat base `widening_dot_reduce_add` as the only owner for this round. Use typed body/config/runtime facts and RVV direct contraction provider facts as authority. Use common EmitC/export only as neutral materialization and packaging.

**Consequences**: The round may be a production seam fix if runtime evidence exposes a stale or missing boundary, or a no-source-change executable evidence closeout if current production code is already correct. It must not become strided/computed-mask expansion or report-only Trellis work.

## Out of Scope

* Strided-input widening dot-reduce expansion.
* Computed-mask or computed-mask-strided widening dot-reduce expansion.
* Broad dot/reduce matrix, dtype/LMUL clone batches, performance tuning, dashboards, readiness reports, or source-front-door positive routes.
* MAcc, product-reduce, dequant, clamp, standalone reduction, memory, segment2, compare/select, widening conversion, Linalg/Vector/StableHLO frontend, or per-Linalg route authority.
* Common EmitC invention of RVV semantics.

## Technical Notes

* Specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
* Prior task reference read: `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-runtime-scalar-cmp-masked-macc-lmul-m2-artifact-abi/prd.md`.
* Initial code/fixture references inspected: `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`, `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, `scripts/rvv_generated_bundle_abi_e2e.py`, and base widening dot-reduce lit/MLIR fixtures.

## Completion Notes

Production source change justification:

* No compiler source change was required for the base widening dot-reduce-add executable artifact ABI seam. Current production code already derives and validates the plain dot-reduction facts from RVV-owned typed body/config/runtime facts: `i16/mf2` lhs/rhs source vector roles, `i32/m1` accumulator/result type facts, scalar seed from `acc[0]`, scalar result store through `out[0]`, SEW32 LMUL m1 policy, runtime AVL/VL, ABI order `lhs,rhs,acc,out,n`, and provider-owned route operand binding summary.
* `RVVEmitCContractionRouteFamilyPlanOwners.cpp` validates the route description against canonical `RVVWideningDotReduceRouteFacts` before route construction, including memory form, runtime control plan, runtime ABI order, target leaf profile, required headers, C type summary, operand-binding plan/summary, contraction route-family plan, typed compute op, source/result SEW/LMUL, widening dot relation, product/reduction/store/setvl intrinsics, vector C types, and runtime ABI parameters.
* The direct contraction provider boundary requires same-analysis route-control provider facts and math operand-binding facts for the exact `WideningDotReduceAdd` sub-family before creating `TCRVEmitCLowerableRoute`.
* `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the provider-owned widening dot-reduce validation contract and rejects rebuilt routes whose route id, headers, type mappings, ABI mappings, or statement plan disagree with provider facts. Candidate metadata remains mirror-only.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-widening-dot-reduce-add-probe \
  --run-id explicit-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind widening_dot_reduce_add \
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
PASS op=widening_dot_reduce_add counts=0,1,7,16,17,23,257 patterns=0,1
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-widening-dot-reduce-add-probe \
  --run-id pre-realized-widening-dot-reduce-add-ssh \
  --overwrite \
  --op-kind widening_dot_reduce_add \
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
PASS op=widening_dot_reduce_add counts=0,1,7,16,17,23,257 patterns=0,1
```

Both runs reported `ssh_evidence: true`, `status: success`, runtime ABI order `lhs,rhs,acc,out,n`, provider route facts, target validator consumed facts, and `runtime_counts_are_execution_cases_not_widening_dot_authority: true`. The pre-realized run additionally recorded `materializer: tcrv-materialize-selected-lowering-boundaries` and `pre_realized_body_consumed: true`.

Focused checks run:

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-explicit-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-widening-dot-reduce-add-fail-closed|explicit-selected-body-artifact-widening-dot-reduce-add|pre-realized-selected-body-artifact-widening-dot-reduce-add'` passed 5 selected tests.

Spec update decision:

* No `.trellis/spec/` update is needed. This round did not introduce a new route contract or convention; it revalidated the existing RVV plugin direct contraction and widening dot-reduce target validation contracts with executable artifact evidence.

## Current Phase

Finish.
