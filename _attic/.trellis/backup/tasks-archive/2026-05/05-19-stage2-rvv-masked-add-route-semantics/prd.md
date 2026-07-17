# Stage2 generic RVV masked add route semantics

## Goal

Add one bounded Stage 2 generic typed `tcrv_rvv` masked elementwise add selected-body route. The route must carry mask, passthrough/result vector, tail/mask policy, vector type, runtime VL, and runtime ABI facts as structural RVV body facts, then let the RVV provider derive the masked intrinsic/header/artifact payload only after validation.

This task closes the next structured-kernel gap after executable arithmetic, broadcast, compare/select, and reduce-add evidence: explicit mask/tail-policy execution semantics for one masked add submodule.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Current HEAD before this task is `555ca83 rvv: close generic reduction executable path`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes Direction Brief as `.trellis/tasks/05-19-stage2-rvv-masked-add-route-semantics` and started.
- Relevant specs require the RVV route authority chain:
  `tcrv.exec` selected RVV variant -> explicit typed `tcrv_rvv` body -> RVV plugin legality/provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- Stage 1 guardrails remain active: no positive legacy `rvv-i32m1`, `RVVI32M1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor, or artifact-name authority.
- Current generic typed RVV surface already has `load`, `broadcast_load`, `binary`, `compare`, `select`, `reduce`, and `store`.
- Current provider accepts add/sub/mul, cmp_select, and reduce_add. It derives config/type/header/intrinsic leaves from typed body/config facts and currently only supports tail agnostic / mask agnostic policy.
- Existing compare/select supplies a structural mask value, but there is no single compute op that models masked arithmetic with explicit passthrough/maskedoff vector semantics.

## Requirements

1. Add one bounded generic typed RVV masked add body surface, expected as a `tcrv_rvv.masked_binary` compute op or an equivalent equally explicit structure.
2. The masked add body must structurally consume:
   - one `!tcrv_rvv.mask<elem, lmul>` mask;
   - one passthrough/maskedoff `!tcrv_rvv.vector<elem, lmul>`;
   - lhs/rhs `!tcrv_rvv.vector<elem, lmul>` operands;
   - the active `!tcrv_rvv.vl` token;
   - explicit `kind = "add"`.
3. The verifier must reject mask/vector/VL/policy mismatches:
   - mask and vector element type/LMUL mismatch;
   - passthrough/lhs/rhs/result vector mismatch;
   - wrong VL token relative to the surrounding `tcrv_rvv.with_vl`;
   - unsupported masked operation kind;
   - missing or mismatched policy metadata on `setvl` / `with_vl`.
4. The RVV provider must validate the typed body before route construction and fail closed for unsupported masked forms. It may derive one masked add intrinsic leaf from validated typed facts; common EmitC/export must remain neutral.
5. Positive materialization/artifact coverage must use an explicit selected `tcrv.exec` RVV variant with typed `tcrv_rvv` body authority.
6. Update generated-bundle evidence tooling only as a runner/checker for the generated artifacts. Python must not implement compiler IR, provider logic, lowering, or emission semantics.
7. Preserve existing add/sub/mul, broadcast, compare/select, and reduce-add behavior.

## Acceptance Criteria

- [x] `tcrv_rvv` verifier accepts a valid masked add typed body and rejects mask/vector/VL/policy/kind mismatches with targeted diagnostics.
- [x] RVV construction protocol recognizes the masked add selected-body route as one provider-owned specialization without broad mask framework expansion.
- [x] RVV EmitC route provider derives masked add route metadata, intrinsic/header facts, mask type mapping, and materialized route payload only from typed body/config/runtime facts.
- [x] Positive target artifact/lit coverage proves masked add reaches provider-derived route metadata and generated header/object bundle path through generic typed `tcrv_rvv` body structure.
- [x] Negative coverage proves unsupported masked forms and stale legacy/source-front-door authority remain fail-closed before target artifact construction.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py --op-kind masked_add` supports dry-run verification and real `ssh rvv` correctness evidence.
- [x] Focused build/tests for touched RVV dialect/provider/construction/script/target paths pass.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, or descriptor authority is reintroduced.
- [x] Task context is validated, task is finished/archived when complete, and one coherent commit is created.

## Non-Goals

- No broad mask framework, mask operation family, mask loads/stores, mask reductions, contraction/matmul, conversions, dtype/LMUL clone batch, global tuning, dashboards, or readiness state machines.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV routes.
- No descriptor-driven computation or Python implementation of compiler core/dialect/pass/plugin/lowering/emission logic.
- No performance claim.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, `tianchenrv-target-artifact-export-test`, and `tianchenrv-construction-protocol-common-test`.
3. Run focused RVV dialect lit tests for masked add verifier positives/negatives.
4. Run focused target artifact lit/FileCheck tests for masked add materialization and fail-closed unsupported cases.
5. Run generated-bundle dry-run for `--op-kind masked_add`.
6. Run `ssh rvv` correctness evidence for `--op-kind masked_add` only if the generated bundle compiles/runs through the same evidence harness.
7. Run `git diff --check`.
8. Run active-authority scan over `include/TianChenRV`, `lib/Plugin/RVV`, `lib/Dialect/RVV`, `test/Target/RVV`, and `scripts`.
9. Run broader `check-tianchenrv` if shared route/export behavior changed enough to justify it.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Prior task context:
  `.trellis/tasks/archive/2026-05/05-19-stage1-gate-a-rvv-route-identity-cleanup/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-generic-typed-rvv-elementwise-predicate-route-coverage/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`.
- Initial implementation surface:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  focused tests under `test/Dialect/RVV` and `test/Target/RVV`.

## Definition Of Done

- One coherent masked add selected-body route is supported through typed `tcrv_rvv` body validation, RVV provider route construction, neutral common EmitC materialization, target artifact generation, and focused evidence.
- Existing Stage 2 selected-body routes remain intact.
- The task report distinguishes route-supported, dry-run artifact evidence, and executable `ssh rvv` evidence.

## Implementation Summary

- Added `tcrv_rvv.masked_binary` as the bounded generic masked elementwise add compute op. The verifier accepts only `kind = "add"` for this task and checks compare-produced mask flow, same `with_vl` scope, VL token consistency, vector/mask type compatibility, passthrough/lhs/rhs/result type agreement, and existing policy facts.
- Extended RVV construction protocol selected-body routing with the `masked_add` route, typed op identity, ABI name, role sequence, and artifact summary metadata.
- Extended the RVV EmitC route provider so masked add is route-supported only after typed validation. The materialized payload is compare mask -> active unmasked add -> provider-derived merge with passthrough -> store, leaving common EmitC/export neutral.
- Extended target artifact export and generated-bundle evidence tooling with `--op-kind masked_add`, dry-run checks, metadata mirror checks, and executable harness expectations.
- Added positive dialect/materialization/artifact/script tests and negative fail-closed coverage for unsupported masked forms. Updated old diagnostic fixtures whose accepted generic op list now includes `tcrv_rvv.masked_binary`.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage2-rvv-masked-add-route-semantics`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused dialect/materialization/negative FileCheck commands for masked add.
- [OK] positive target artifact plan/header FileCheck commands for masked add.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] masked-add generated-bundle dry-run at `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-masked-add-dry-run`.
- [OK] real `ssh rvv` masked-add correctness evidence at `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-masked-add-evidence`: counts `1,7,16,17,257` passed.
- [OK] `git diff --check`
- [OK] diff active-authority scan introduced no `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, or descriptor authority. Matches were limited to provider-derived exact intrinsic leaf checks.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 157/157 passed.

## Self-Repair Notes

- Initial masked intrinsic materialization used a five-argument masked add intrinsic shape. The generated object packaging on the current RVV clang rejected that signature, so the route now materializes the bounded semantics as an active unmasked add followed by provider-derived `vmerge` with the passthrough vector and compare mask.
- Broad `check-tianchenrv` initially failed only because old diagnostic fixtures did not list `tcrv_rvv.masked_binary` in the accepted generic op surface. Those expectations were updated and the broad check then passed.
