# Stage2 RVV scalar-broadcast MAcc executable artifact ABI boundary

## Goal

Make the existing pre-realized scalar-broadcast multiply-accumulate selected-body RVV route executable as a generated target artifact with truthful ABI/runtime evidence. The owned boundary is the scalar-broadcast MAcc executable artifact path: typed `tcrv_rvv` body facts, accumulator/input/output roles, scalar broadcast provenance, multiply/add semantics, dtype/config/policy, runtime AVL/VL, operand ABI/header bindings, RVV plugin route validation, common EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.

## Why Now

Commit `d4b3fcb0` hardened plain scalar-broadcast add/sub/mul executable ABI/header facts and proved that route family through generated bundles on `ssh rvv`. Scalar-broadcast MAcc reuses the scalar-broadcast binding shape but adds accumulator and contraction-supporting multiply-add roles, so it is the adjacent Stage2 route-family seam most likely to be stale or under-validated.

## What I Already Know

- The task is Stage2 RVV work on the corrected typed low-level `tcrv_rvv` route surface, not Stage1 legacy `i32m1` route growth.
- Production authority must remain `tcrv.exec` selected RVV variant -> typed `tcrv_rvv` body -> RVV plugin-owned realization/legality/route facts -> `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact -> generated bundle -> `ssh rvv` evidence when executable behavior is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime roles only; they do not own MAcc compute semantics, dtype/config, scalar broadcast semantics, intrinsic spelling, or route support.
- Common EmitC/export may carry provider-built payloads but must not infer accumulator, multiplicand, scalar broadcast, output, operation, or runtime facts from names, route IDs, artifact metadata, or diagnostic mirrors.
- The previous scalar-broadcast elementwise task is the reference for executable ABI/header binding tokens (`abi`/`hdr`), generated-bundle workflow, and stale mirror rejection.
- Existing MAcc files and fixtures named in the brief must be audited before source edits to determine whether the MAcc executable path is dry-run-only, stale, under-validated, or already production-valid.

## Requirements

- Preserve plugin ownership: RVV provider code must validate accumulator, vector multiplicand, scalar broadcast operand, output, op kind, dtype/config/policy, runtime AVL/VL, and per-operand ABI/header bindings before constructing an executable MAcc route.
- Preserve common EmitC neutrality: common route materialization must consume provider payload only and must not invent RVV MAcc semantics.
- Make the pre-realized scalar-broadcast MAcc selected-body route executable through materialized selected boundary, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness when executable behavior is claimed.
- Harden fail-closed behavior for at least one stale or missing executable-boundary fact, such as missing scalar broadcast provenance, swapped accumulator/input/output role, stale multiply-add op fact, stale header/prototype binding, wrong generated C type, wrong ABI value mapping, or stale route-family validation contract.
- Keep support levels separate: parseable/verifier-legal `tcrv_rvv` is not route-supported; route-supported is not executable without complete ABI/runtime/export support and real evidence for runtime correctness claims.
- Keep the task bounded to scalar-broadcast MAcc. Any add/sub/mul changes may be used only as reference or shared-boundary repair when required by the MAcc seam.

## Acceptance Criteria

- [x] PRD and Trellis context identify scalar-broadcast MAcc executable artifact ABI boundary, non-goals, and relevant specs.
- [x] Repository audit records whether the current MAcc path is dry-run-only, stale, under-validated, or already production-valid.
- [x] Production code is changed when the executable boundary is incomplete or under-validated; otherwise the PRD records a precise no-source-change justification backed by focused evidence.
- [x] Positive evidence covers pre-realized scalar-broadcast MAcc through selected boundary materialization, emission plan, target artifact export, generated bundle compile, and `ssh rvv` correctness if runtime behavior is claimed.
- [x] Focused fail-closed evidence rejects at least one stale/missing executable-boundary MAcc fact before artifact acceptance or executable-route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run tests pass, including existing scalar-broadcast MAcc tests and any added focused coverage.
- [x] Bounded old-authority scan over touched files and added diff lines shows no new `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*` route authority, source-front-door route, descriptor compute path, or exact intrinsic spelling as route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are clean after commit.

## Out of Scope

- No broad MAcc matrix, dtype/LMUL clone batch, computed-masked MAcc expansion, standalone reduction, compare/select, conversion, product/dequant, segment2, memory-family, source-front-door positive route, or high-level Linalg/Vector/StableHLO frontend work.
- No per-Linalg route authority, high-level kernel op, one-op-per-intrinsic wrapper, performance tuning database, dashboard/index/report-only closeout, or compatibility wrapper preserving old route authority.
- No Python implementation of compiler core, dialects, passes, plugin registry, capability model, lowering, emission, or route semantics. Python may only support generated-bundle tooling and evidence collection.
- No common EmitC invention of RVV accumulator, scalar broadcast, MAcc operation, dtype, VL policy, or ABI semantics.

## Technical Notes

- Read specs: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
- Read previous task reference: `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-scalar-broadcast-elementwise-artifact-abi/prd.md`.
- Read MAcc route files named in the brief before editing:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - scalar-broadcast MAcc tests and fixtures listed in the brief.

## Audit Conclusion

The production scalar-broadcast MAcc executable artifact seam is already production-valid under the bounded task criteria. No production provider, target validator, or generated-bundle script source change is justified in this round because the audited path already:

- realizes pre-realized `tcrv_rvv.typed_macc_pre_realized_body` into `setvl -> load lhs -> splat rhs_scalar -> load accumulator -> tcrv_rvv.macc -> store out`;
- derives scalar-broadcast MAcc provider facts from typed body/config/runtime structure, including runtime ABI order `lhs,rhs_scalar,acc,out,n`, `abi|hdr` operand binding summary, scalar splat provenance, accumulator/result layout, MAcc arithmetic kind, required headers, C type mapping, and runtime AVL/VL facts;
- validates provider facts and statement payloads in the MAcc target artifact family validator, including ABI mappings, type mappings, route headers, scalar splat step, accumulator load, MAcc operand order, output store, and stale mirror rejection;
- rejects stale pre-realized direct route-entry shortcut mode for scalar-broadcast MAcc with `--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): scalar_broadcast_macc_add`;
- already carries target fixture fail-closed checks for stale route id, route-family plan, runtime ABI order, missing `hdr` binding token, and stale C type mapping in `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-macc-add.mlir`.

The only gap found was coverage shape: there was an explicit selected-body generated-bundle dry-run test and a pre-realized direct-route-entry fail-closed test, but no positive pre-realized generated-bundle dry-run test for scalar-broadcast MAcc. This round adds that focused test.

## Implementation Summary

- Added `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-scalar-broadcast-macc-add-dry-run.test`.
- The new test fixes the positive generated-bundle coverage for pre-realized scalar-broadcast MAcc and checks:
  - pre-realized selected-body input mode and body consumption;
  - generated function prototype and selected variant;
  - route id, typed compute op, memory form, runtime ABI order, exec ABI bindings, route operand binding plan, and `abi|hdr` operand facts;
  - scalar-broadcast MAcc family plan, target leaf profile, provider-supported mirror, required headers, C type mapping, arithmetic kind, accumulator layout, and result layout;
  - emitted C++ scalar splat, MAcc, load, store, and setvl intrinsics as output evidence only;
  - generated harness oracle `acc[index] + lhs[index] * rhs_scalar`, signed products, explicit accumulator use, and tail preservation.

## Evidence

- Dry-run audit command passed:
  - `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind scalar_broadcast_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --rhs-scalar -37 --rhs-scalar 91`
  - artifact directory: `artifacts/tmp/rvv_generated_bundle_abi_e2e/audit-pre-realized-scalar-broadcast-macc-add`
  - status: `dry_run_success`
- Non-dry-run generated-bundle evidence passed on `ssh rvv`:
  - command owner: `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind scalar_broadcast_macc_add`
  - artifact directory: `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-scalar-broadcast-macc-add`
  - root evidence: `status: success`, `dry_run: false`, `ssh_evidence: true`, `input_mode: pre-realized-selected-body`, `pre_realized_selected_body: true`
  - selected variant: `pre_realized_body_rvv_scalar_broadcast_macc_add`
  - ABI name: `rvv-generic-scalar-broadcast-macc-add-callable-c-abi.v1`
  - prototype: `void tcrv_emitc_pre_realized_body_scalar_broadcast_macc_add_kernel_pre_realized_body_rvv_scalar_broadcast_macc_add(const int32_t *lhs, int32_t rhs_scalar, const int32_t *acc, int32_t *out, size_t n);`
  - runtime counts: `0,1,16,17,257`
  - RHS scalar values: `-37,91`
  - pass marker: `tcrv_rvv_generated_bundle_abi_scalar_broadcast_macc_add_ok`
  - coverage: explicit accumulator, signed products, scalar-broadcast RHS, and tail sentinel preservation.
- Focused checks passed:
  - `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - from `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "pre-realized-scalar-broadcast-macc-add"` passed 2 selected tests
  - from `build/test`: `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "scalar-broadcast-macc-add"` passed 5 selected tests
- Bounded old-authority scan over touched files found no new route authority. Matches were limited to PRD guardrail text, negative `implicit-check-not` checks, and emitted-code intrinsic evidence in the new test; exact intrinsic spellings are checked as generated output only, not route authority.

## Spec Update Decision

No `.trellis/spec/` update is needed. This round implements and verifies existing RVV plugin and EmitC-route contracts: provider-owned typed body/config/runtime facts own route support; common EmitC remains neutral; target artifact validation rejects stale mirrors; and runtime correctness claims require `ssh rvv` evidence.

## Current Phase

Finish.
