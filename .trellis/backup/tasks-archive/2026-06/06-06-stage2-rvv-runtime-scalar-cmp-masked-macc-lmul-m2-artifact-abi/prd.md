# Stage2 RVV runtime-scalar-cmp masked MAcc LMUL m2 artifact ABI boundary

## Goal

Make the existing pre-realized RVV runtime-scalar-cmp masked MAcc add LMUL m2 selected-body route executable as generated RVV artifacts with truthful ABI/runtime evidence, or fail closed at the exact production boundary if any LMUL/config, mask, runtime scalar compare, operand role, inactive-lane policy, accumulator/result, header/prototype, ABI order, AVL/VL, or statement fact is missing or stale.

The purpose is to continue the prior runtime-scalar-cmp masked MAcc add closeout at the LMUL m2 boundary, proving that LMUL/config, mask type, vector C type, intrinsic/header spelling, and ABI validation come from provider-owned typed body/config/runtime facts rather than the base m1 route shape, route ids, metadata, helper names, test names, or common EmitC semantics.

## Requirements

* Scope is one production workflow submodule: the pre-realized runtime-scalar-cmp masked MAcc add LMUL m2 selected-body route family from typed `tcrv_rvv` body facts through RVV route planning, common EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.
* Keep route authority local to the RVV plugin and typed `tcrv_rvv` body/config/runtime facts. Common EmitC/export may materialize neutral mechanics only.
* Validate LMUL m2 vector/config facts, mask type, runtime scalar comparison operands, predicate/mask construction, active/inactive lane policy, vector multiplicand/addend/accumulator/result roles, SEW/LMUL/policy, runtime AVL/VL, ABI/header/prototype binding, and route-family validation contracts.
* If the current LMUL m2 path is dry-run-only, stale, under-validated, or conflates boundary facts, rewire or harden the production path instead of adding report-only evidence.
* Add an explicit LMUL m2 counterpart only when repository inspection shows the production boundary needs it and the change fits this coherent round.
* Produce positive executable evidence only when the selected-body route reaches generated bundle compile/run correctness on real `ssh rvv`; otherwise produce focused fail-closed evidence for the precise stale or missing boundary fact.
* Preserve the existing Stage 2 selected-body realization contract: pre-realized selected body is realized plugin-locally without changing computation semantics, dtype semantics, parameter roles, selected variant origin, required capabilities, dispatch/fallback behavior, or runtime AVL values.

## Acceptance Criteria

* [x] The active LMUL m2 runtime-scalar-cmp masked MAcc route path either materializes executable generated RVV artifacts through the selected-body boundary or fails closed with targeted diagnostics for missing/stale executable-boundary facts.
* [x] Route/provider validation derives or checks LMUL m2 config, mask/vector C types, compare predicate construction, inactive-lane policy, operand roles, header/prototype binding, ABI mapping, and runtime AVL/VL from typed body/config/runtime facts.
* [x] Relevant dry-run/generated-bundle tests cover the LMUL m2 pre-realized runtime-scalar-cmp masked MAcc seam, including one positive executable or one focused fail-closed case.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests pass.
* [x] If runtime correctness is claimed, generated bundle compile/run passes on `ssh rvv`.
* [x] A bounded old-authority scan over touched files and added diff lines shows no new dependence on route id, artifact name, helper name, test name, metadata, exact intrinsic spelling, or common EmitC RVV branches as semantic authority.
* [x] `git diff --check` and `git diff --cached --check` pass.

## Definition of Done

* PRD and task context are truthful for the bounded LMUL m2 executable artifact ABI module.
* Production code is changed only if repository evidence shows the LMUL m2 path is dry-run-only or under-validated.
* Tests are focused on changed behavior and the requested LMUL m2 seam.
* Trellis task status and workspace journal are updated.
* The completed round is archived and committed in one coherent commit, or left open with an exact continuation point if incomplete.

## Technical Approach

1. Inspect the prior archived runtime-scalar-cmp masked MAcc add task and the current LMUL m2 tests to identify whether the LMUL m2 seam is executable, dry-run-only, or intentionally fail-closed.
2. Inspect RVV MAcc selected-body realization, route-family planning, compare/select statement planning, statement planning, route provider construction, and target artifact validation around runtime-scalar-cmp masked MAcc and LMUL/config derivation.
3. If the seam lacks production executable support, add the smallest provider/validation/materialization change that lets LMUL m2 facts flow from typed body/config/runtime facts into `TCRVEmitCLowerableRoute` and generated artifacts.
4. If the seam is already production-supported, add missing positive executable evidence and focused fail-closed validation without source churn, and document the no-source-change justification.
5. Run focused tests, generated bundle checks, `ssh rvv` runtime evidence if claiming executable behavior, old-authority scan, diff checks, task finish/archive, and commit.

## Decision (ADR-lite)

**Context**: The previous task completed base runtime-scalar-cmp masked MAcc add evidence and left LMUL m2 runtime-scalar-cmp files as the next bounded reference.

**Decision**: Continue the same MAcc/mask direction at the LMUL m2 executable artifact ABI boundary. Treat typed `tcrv_rvv` body/config/runtime facts and RVV provider validation as authority; treat route ids, metadata, helper names, test names, artifact names, exact intrinsic spellings, and common EmitC as mirrors or mechanics only.

**Consequences**: This keeps the task narrow and evidence-driven. It may result in no production source changes if inspection proves the route is already wired and only missing executable evidence. If production facts are missing or stale, the fix must be in the actual route/provider/validation/export seam, not a report, helper, or compatibility wrapper.

## Out of Scope

* Broad MAcc matrix expansion.
* LMUL m4/m8 or dtype clone batches.
* Unrelated widening, scalar-broadcast, computed-mask-only, product-reduce, dequant, clamp, reduction, compare/select, widening conversion, memory, or segment2 rework except as reference.
* High-level Linalg/Vector/StableHLO frontend work.
* Per-Linalg route authority or high-level kernel ops.
* Performance tuning databases, dashboards, reports, or readiness state machines.
* Source-front-door positive routes.
* Common EmitC invention of RVV semantics.

## Technical Notes

* Read first from brief: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, archived task `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-runtime-scalar-cmp-masked-macc-add-artifact-abi/`, RVV route planning/provider/materialization/validation files, generated-bundle ABI script, LMUL m2 dry-run/fail-closed tests, and the pre-realized LMUL m2 MLIR fixture.
* Current branch at task creation: `main`.
* Initial worktree state at task creation: clean.
* Last relevant commit before this task: `3306b004 rvv: prove runtime scalar masked macc artifact abi`.

## Completion Notes

Production source change justification:

* No compiler source change was required for this LMUL m2 runtime-scalar-cmp masked MAcc add executable artifact ABI seam. Current production code already validates the pre-realized body as `op_kind = runtime_scalar_cmp_masked_macc_add`, `predicate_kind = sle`, memory form `runtime-scalar-computed-mask-unit-stride-macc`, mask role/source/form, accumulator/result layout, SEW32 LMUL m1/m2, TA/MA policy, and ABI roles for `cmp_lhs`, `rhs_scalar`, `lhs`, `rhs`, `acc`, `out`, and `n`.
* The RVV MAcc owner path obtains runtime-scalar computed-mask MAcc facts with `analysis.typedConfigFacts.sew` and `analysis.typedConfigFacts.lmul`; the target artifact validation contract obtains the same route facts with `description.sew` and `description.lmul`. LMUL m2 therefore does not rely on the no-argument LMUL m1 convenience path.
* The provider/validator seam carries the LMUL m2 facts into the generated artifact boundary: `rvv-selected-body-sew32-lmul-m2-tail-agnostic-mask-agnostic.v1`, `vint32m2_t`, `vbool16_t`, `__riscv_vsetvl_e32m2`, `__riscv_vmv_v_x_i32m2`, `__riscv_vmsle_vv_i32m2_b16`, `__riscv_vmacc_vv_i32m2`, `__riscv_vmerge_vvm_i32m2`, and `__riscv_vse32_v_i32m2`.
* Runtime ABI order and operand binding remain provider-owned: `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n` and `rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1;cmp_lhs=lhs-input-buffer:cmp_lhs:abi|cmp-lhs|cmp-call|hdr;rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr;lhs=dot-lhs-input-buffer:lhs:abi|lhs-load|macc-lhs|hdr;rhs=dot-rhs-input-buffer:rhs:abi|rhs-load|macc-rhs|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|macc-acc|macc-pass|hdr;out=output-buffer:out:abi|store|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr`.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-runtime-scalar-cmp-masked-macc-lmul-m2-probe \
  --run-id pre-realized-runtime-scalar-cmp-masked-macc-lmul-m2-ssh \
  --overwrite \
  --op-kind runtime_scalar_cmp_masked_macc_add_lmul_m2 \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 17 --runtime-count 23 \
  --runtime-count 257 \
  --rhs-scalar -37 --rhs-scalar 91 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=runtime_scalar_cmp_masked_macc_add_lmul_m2 counts=0,1,7,16,17,23,257 rhs_scalars=-37,91 patterns=0,1
```

The remote harness ran counts `0,1,7,16,17,23,257`, runtime scalar thresholds `-37,91`, and patterns `0,1`. It reported active/inactive lane coverage, inactive accumulator preservation, add-only and mul-only distinguishing lanes, and tail preservation. Evidence JSON recorded `ssh_evidence: true`, `status: success`, the pre-realized LMUL m2 fixture path, materialized selected-body consumption, and typed config artifact closure with LMUL m2 intrinsics.

Focused checks run:

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-macc-add-lmul-m2-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-lmul-m2-fail-closed|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add-lmul-m2'` passed 3 selected tests.
* Bounded old-authority scan over added task diff lines found no positive legacy route authority.
* `git diff --check`
* `git diff --cached --check`

## Spec Update Decision

No `.trellis/spec/` update is needed. This round revalidated existing RVV plugin, EmitC-route, MAcc owner, target artifact validation, and generated-bundle evidence contracts for the LMUL m2 runtime-scalar computed-mask MAcc seam. The relevant durable rule already exists: provider/target consumers that have selected SEW/LMUL facts must call parameterized MAcc facts accessors with the actual selected `sew` and `lmul`.

## Current Phase

Finish.
