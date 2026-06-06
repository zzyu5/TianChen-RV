# Stage2 RVV scalar-broadcast elementwise arithmetic executable artifact ABI boundary

## Goal

Make the existing pre-realized scalar-broadcast elementwise add/sub/mul selected-body RVV route family executable as generated target artifacts with truthful ABI/runtime evidence. The owner boundary is the scalar-broadcast executable artifact ABI path: typed `tcrv_rvv` vector body facts, scalar-to-vector broadcast provenance, op kind, dtype/config/policy, runtime AVL/VL, operand ABI/header bindings, RVV plugin route validation, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.

## Why Now

The prior computed-masked segment2 artifact ABI task closed the memory-movement boundary with non-dry-run generated bundle evidence. The next Stage2 bottleneck is the elementwise/broadcast class, where scalar operands exercise distinct ABI binding and selected-body realization behavior from memory movement.

## What I Already Know

- The task is Stage2 RVV work on the corrected typed low-level `tcrv_rvv` route surface, not Stage1 legacy `i32m1` route growth.
- The production chain must remain `tcrv.exec` selected RVV variant -> typed `tcrv_rvv` body -> RVV plugin-owned validation/route facts -> `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact -> generated bundle -> `ssh rvv` evidence when executable behavior is claimed.
- Scalar-broadcast add/sub/mul already have relevant pre-realized artifact fixtures and generated bundle support.
- Initial audit found scalar-broadcast add already used executable ABI/header binding tokens (`abi`/`hdr`) in provider binding summaries, but scalar-broadcast sub/mul still used mirror-only tokens (`runtime-abi-mirror`/`header-mirror`) in the production operand-binding plan and provider-facts verifier. That made the sub/mul target artifact boundary under-validated against the documented provider operand binding contract.

## Requirements

- Preserve plugin ownership: RVV plugin code must validate scalar broadcast provenance, vector/scalar operand roles, op kind, dtype/config/policy, and runtime AVL/VL before declaring an executable artifact route.
- Preserve common EmitC neutrality: common materialization may lower route facts but must not invent RVV scalar-broadcast semantics.
- Make positive scalar-broadcast artifact execution real for add/sub/mul or a coherent completed subset, including materialized selected boundary, emission plan, target artifact export, generated bundle compilation, and `ssh rvv` correctness when runtime correctness is claimed.
- Harden fail-closed behavior for stale or missing executable-boundary facts, with at least one focused negative case for a stale/missing scalar-broadcast artifact ABI fact.
- Keep support levels separate: parseable/verifier-legal `tcrv_rvv` is not route-supported, and route-supported is not executable without complete ABI/runtime/export support.

## Acceptance Criteria

- [x] PRD and Trellis context identify the scalar-broadcast elementwise artifact ABI boundary and non-goals.
- [x] Repository evidence proves whether the existing scalar-broadcast path is dry-run-only, stale, under-validated, or already production-valid.
- [x] Production code is changed when the executable boundary is incomplete or under-validated; otherwise the PRD records a source-no-change justification backed by focused evidence.
- [x] Positive evidence covers scalar-broadcast add/sub/mul or the explicit coherent subset through generated artifact export and generated bundle ABI checks.
- [x] Runtime correctness is only claimed with non-dry-run `ssh rvv` evidence.
- [x] Focused fail-closed evidence covers at least one missing/stale executable-boundary fact, such as scalar broadcast provenance, operand role mapping, op kind, header/prototype binding, generated C type, ABI value mapping, or route-family validation contract.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run tests pass; non-dry-run `ssh rvv` evidence is collected for executable claims.
- [x] Bounded old-authority scan over touched files and diff lines shows no new legacy `RVVI32M1`/`rvv-i32m1`/finite `tcrv_rvv.i32_*` route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status --short` are clean after commit.

## Out of Scope

- No broad elementwise matrix, dtype/LMUL clone batch, scalar-broadcast macc, segment2 rework, high-level Linalg/Vector/StableHLO frontend, per-Linalg route authority, tuning database, dashboard/report-only closeout, source-front-door positive route, or unrelated memory/product/dequant/contraction/reduction/compare/select/conversion ownership rewrite.
- No Python implementation of compiler core, dialects, passes, plugin registry, capability model, lowering, emission, or route semantics.
- No compatibility wrapper that preserves old i32 route authority as a positive executable path.

## Technical Notes

- Read first list from brief: `.trellis/spec/index.md`, RVV plugin spec, EmitC route spec, archived segment2 ABI task, scalar-broadcast route planning/provider/statement owners, target artifact route-family validation, generated bundle ABI runner, and scalar-broadcast tests/fixtures.
- The task should finish/archive and create one coherent commit only if the module behavior is complete and verified.

## Implementation Summary

- `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` now derives and validates scalar-broadcast add/sub/mul operand-binding summaries with executable ABI/header participation tokens (`abi`/`hdr`) for `lhs`, `rhs_scalar`, `out`, and `n`.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` now requires the same `hdr` token for scalar-broadcast add/sub/mul provider operand-binding facts before route construction.
- `scripts/rvv_generated_bundle_abi_e2e.py` now expects scalar-broadcast sub/mul generated bundle metadata to match the executable ABI/header binding summary.
- Scalar-broadcast sub/mul FileCheck fixtures were updated to the corrected provider summary, and `pre-realized-selected-body-artifact-scalar-broadcast-sub.mlir` now proves stale mirror-only binding metadata fails before target artifact export.
- Added `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-scalar-broadcast-mul-dry-run.test` so pre-realized scalar-broadcast add/sub/mul all have generated-bundle dry-run coverage.

## Current Phase

Finish.

## Evidence

- `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate` passed.
- From `build/test`, `../bin/tianchenrv-rvv-extension-plugin-test` passed.
- From `build/test`, `../bin/tianchenrv-target-artifact-export-test` passed.
- From `build/test`, `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "scalar-broadcast"` passed 23 selected tests.
- Non-dry-run generated-bundle evidence passed on `ssh rvv`:
  - command owner: `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind scalar_broadcast_add --op-kind scalar_broadcast_sub --op-kind scalar_broadcast_mul`
  - artifact directory: `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-scalar-broadcast-add-sub-mul`
  - root evidence: `status: success`, `dry_run: false`, `ssh_evidence: true`, `input_mode: pre-realized-selected-body`, `pre_realized_selected_body: true`
  - selected variants: `pre_realized_body_rvv_scalar_broadcast_add`, `pre_realized_body_rvv_scalar_broadcast_sub`, `pre_realized_body_rvv_scalar_broadcast_mul`
  - ABI names: `rvv-generic-scalar-broadcast-add-callable-c-abi.v1`, `rvv-generic-scalar-broadcast-sub-callable-c-abi.v1`, `rvv-generic-scalar-broadcast-mul-callable-c-abi.v1`
  - runtime counts: `0,1,7,16,23,257`
  - RHS scalar values: `-37,91`
  - pass markers: `tcrv_rvv_generated_bundle_abi_scalar_broadcast_add_ok`, `tcrv_rvv_generated_bundle_abi_scalar_broadcast_sub_ok`, `tcrv_rvv_generated_bundle_abi_scalar_broadcast_mul_ok`
  - coverage included RHS scalar influence, LHS influence, negative-scalar distinguishing cases where applicable, and tail sentinel preservation.

## Spec Update Decision

No `.trellis/spec/` update is needed. This round implements an already documented contract: provider operand-binding summaries for exported runtime ABI/header parameters must carry `abi` and `hdr`, and target artifact validation must reject stale mirrors. The task corrected scalar-broadcast sub/mul production code and evidence to match that existing spec; it did not introduce a new architectural rule.
