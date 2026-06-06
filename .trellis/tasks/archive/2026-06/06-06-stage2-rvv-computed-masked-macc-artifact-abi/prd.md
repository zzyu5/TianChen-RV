# Stage2 RVV computed-masked MAcc executable artifact ABI boundary

## Goal

Make the existing pre-realized computed-masked multiply-accumulate selected-body RVV route executable as a generated target artifact with truthful ABI/runtime evidence. The owned boundary is the computed-masked MAcc executable artifact path: typed `tcrv_rvv` body facts, computed-mask provenance, accumulator/input/output roles, multiply/add semantics, dtype/config/policy, runtime AVL/VL, per-operand ABI/header bindings, RVV plugin route validation, common EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.

## Why Now

Commit `98989069` completed scalar-broadcast MAcc as an executable artifact evidence closeout. Computed-masked MAcc is the adjacent Stage2 route-family seam: it keeps the contraction-supporting accumulator path, but adds compare-produced mask provenance and inactive-lane policy. This round must prove that the existing computed-mask MAcc path is executable from selected/pre-realized typed body to generated RVV artifact, or fail closed at the precise missing artifact/ABI boundary.

## What I Already Know

- This is Stage2 RVV work on the corrected typed low-level `tcrv_rvv` route surface, not Stage1 legacy `i32m1` route growth.
- Production authority must remain `tcrv.exec` selected RVV variant -> typed `tcrv_rvv` body -> RVV plugin-owned selected-body realization and route facts -> `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact -> generated bundle -> `ssh rvv` evidence when executable behavior is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime roles only; they do not own MAcc compute semantics, mask provenance, inactive-lane policy, dtype/config, intrinsic spelling, or route support.
- Common EmitC/export may carry provider-built payloads but must not infer accumulator, multiplicand, computed-mask, output, operation, or runtime facts from names, route ids, artifact metadata, or diagnostic mirrors.
- The previous scalar-broadcast MAcc task is the reference for generated-bundle evidence shape and final evidence recording, but computed-masked MAcc must be audited independently.
- Initial audit shows a production selected-body owner exists for `TypedComputedMaskMAccPreRealizedBodyOp`, materializing compare lhs/rhs, payload lhs/rhs, accumulator load, compare-produced mask, `tcrv_rvv.masked_macc`, and store before route/provider construction.
- Initial audit shows MAcc route-family owner facts already include computed-mask ABI order, `abi|hdr` operand binding summary, mask role/source/form, inactive-lane contract, passthrough layout, MAcc arithmetic/layout facts, header/type mappings, and provider-supported mirrors.
- Initial audit shows the target artifact fixture already has focused stale mirror rejection for route id, provider mirror, binding plan, ABI order, headers, C type mapping, accumulation plan, accumulator layout, runtime-scalar binding swap, predicate, mask role/source/form, source memory form, passthrough layout, arithmetic kind, and result layout.
- Existing generated-bundle tests cover explicit selected-body and pre-realized selected-body computed-masked MAcc in dry-run mode. The missing executable blocker is non-dry-run generated bundle execution on `ssh rvv` for the pre-realized computed-masked MAcc path.

## Requirements

- Preserve plugin ownership: RVV selected-body owner/provider code must validate computed-mask provenance, accumulator/vector multiplicand roles, output role, op kind, dtype/config/policy, runtime AVL/VL, inactive-lane behavior, and per-operand ABI/header bindings before constructing an executable computed-masked MAcc route.
- Preserve common EmitC neutrality: common route materialization must consume provider payload only and must not invent RVV MAcc, mask, inactive-lane, dtype, VL policy, or ABI semantics.
- Prove the pre-realized computed-masked MAcc selected-body route through materialized selected boundary, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness before claiming executable behavior.
- Keep fail-closed evidence focused on executable-boundary facts such as missing/stale computed-mask provenance, inactive-lane policy, accumulator/input/output role, multiply-add op fact, header/prototype binding, ABI order, generated C type, or route-family validation contract.
- If production code is already valid, avoid source churn and record a precise no-source-change justification backed by focused positive and fail-closed evidence.
- Keep support levels separate: parseable/verifier-legal `tcrv_rvv` is not route-supported; route-supported is not executable without complete ABI/runtime/export support and real evidence for runtime correctness claims.

## Acceptance Criteria

- [x] PRD and Trellis context identify the computed-masked MAcc executable artifact ABI boundary, non-goals, and relevant specs.
- [x] Repository audit records whether the current computed-masked MAcc path is dry-run-only, stale, under-validated, or already production-valid.
- [x] Production code is changed when the executable boundary is incomplete or under-validated; otherwise the PRD records a precise no-source-change justification backed by focused evidence.
- [x] Positive evidence covers pre-realized computed-masked MAcc through selected boundary materialization, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness if runtime behavior is claimed.
- [x] Focused fail-closed evidence rejects at least one stale/missing executable-boundary computed-mask MAcc fact before artifact acceptance or executable-route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run tests pass, including explicit and pre-realized computed-masked MAcc tests and direct pre-realized route-entry fail-closed coverage.
- [x] Bounded old-authority scan over touched files and added diff lines shows no new `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*` route authority, source-front-door route, descriptor compute path, or exact intrinsic spelling as route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are clean after commit.

## Out of Scope

- No broad MAcc matrix, dtype/LMUL clone batch, runtime-scalar compare-masked MAcc expansion except as bounded reference, scalar-broadcast MAcc rework, high-level Linalg/Vector/StableHLO frontend, source-front-door positive route, or performance tuning database.
- No mass rewrite of memory, segment2, product/dequant, standalone reduction, compare/select, conversion, contraction, or unrelated Stage2 ownership.
- No per-Linalg route authority, high-level kernel op, one-op-per-intrinsic wrapper, dashboard/index/report-only closeout, or compatibility wrapper preserving old route authority.
- No Python implementation of compiler core, dialects, passes, plugin registry, capability model, lowering, emission, or route semantics. Python may only support generated-bundle tooling and evidence collection.
- No common EmitC invention of RVV mask, inactive-lane, accumulator, MAcc operation, dtype, VL policy, or ABI semantics.

## Technical Notes

- Read specs: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
- Read previous task reference: `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-scalar-broadcast-macc-artifact-abi/prd.md`.
- Read bounded source/test files from the task brief:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/RVVComputedMaskMAccSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - computed-masked MAcc script and target fixtures named in the brief.

## Audit Conclusion

The production computed-masked MAcc artifact/ABI seam is already production-valid under the bounded task criteria. No production provider, target validator, or generated-bundle script source change is justified in this round because the audited path already:

- realizes pre-realized `tcrv_rvv.typed_computed_mask_macc_pre_realized_body` into `setvl -> with_vl -> compare lhs/rhs loads -> payload lhs/rhs loads -> accumulator load -> compare-produced mask -> tcrv_rvv.masked_macc -> store`;
- derives computed-mask MAcc provider facts from typed body/config/runtime structure, including runtime ABI order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, `abi|hdr` operand binding summary, compare predicate, mask role/source/form, inactive-lane contract, passthrough layout, accumulator/result layout, required headers, C type mapping, and runtime AVL/VL facts;
- validates provider facts through the MAcc route-family owner and computed-mask accumulation statement-plan owner before constructing `TCRVEmitCLowerableRoute`;
- validates target artifact mirrors for route, provider support, binding plan, runtime ABI order, header requirements, type mapping, accumulation plan, accumulator/result layout, predicate, mask role/source/form, source memory form, passthrough layout, and arithmetic kind;
- rejects stale pre-realized direct route-entry shortcut mode for computed-masked MAcc with `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): computed_masked_macc_add`.

The precise missing executable blocker from the initial audit was lack of non-dry-run generated-bundle `ssh rvv` evidence for the pre-realized computed-masked MAcc path in this task context. This round closes that blocker with focused generated-bundle execution.

## Implementation Summary

- No production source file changed.
- Created this Trellis task and PRD to record the bounded audit, no-source-change justification, evidence commands, and acceptance state.
- Produced non-dry-run generated-bundle evidence for pre-realized computed-masked MAcc.

## Evidence

- Non-dry-run generated-bundle evidence passed on `ssh rvv`:
  - command owner: `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_masked_macc_add --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id pre-realized-computed-masked-macc-add --overwrite`
  - artifact directory: `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-masked-macc-add`
  - root evidence: `status: success`, `dry_run: false`, `ssh_evidence: true`, `input_mode: pre-realized-selected-body`, `pre_realized_selected_body: true`
  - selected variant: `pre_realized_body_rvv_computed_masked_macc_add`
  - ABI name: `rvv-generic-computed-masked-macc-add-callable-c-abi.v1`
  - prototype: `void tcrv_emitc_pre_realized_body_computed_masked_macc_add_kernel_pre_realized_body_rvv_computed_masked_macc_add(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *lhs, const int32_t *rhs, const int32_t *acc, int32_t *out, size_t n);`
  - runtime counts: `1,7,16,17,257`
  - remote target: `rvv`, `remote_arch=riscv64`, remote clang `/usr/bin/clang`, `Ubuntu clang version 18.1.3`
  - pass marker: `tcrv_rvv_generated_bundle_abi_computed_masked_macc_add_ok`
  - coverage: active-lane `acc + lhs * rhs`, inactive-lane accumulator preservation, add-only and mul-only distinguishing cases, signed products, source preservation, and tail sentinel preservation.
- Focused checks passed:
  - `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - from `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-macc-add` passed 6 selected tests.
- Bounded old-authority scan over the added task files found no positive old route authority. Matches were limited to PRD acceptance/non-goal guardrail text for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `source-front-door`, and `descriptor`.

## Spec Update Decision

No `.trellis/spec/` update is needed. This round verifies existing RVV plugin and EmitC-route contracts: computed-mask MAcc selected-body realization is plugin-owned; provider-derived typed body/config/runtime facts own route support; common EmitC remains neutral; target artifact validation rejects stale mirrors; and runtime correctness claims require `ssh rvv` evidence.

## Current Phase

Finish.
