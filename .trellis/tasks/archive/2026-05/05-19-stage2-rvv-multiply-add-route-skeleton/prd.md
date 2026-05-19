# Stage2 RVV contraction multiply-add route skeleton

## Goal

Add one bounded Stage 2 contraction-supporting multiply-add/accumulate route on
the corrected generic typed `tcrv_rvv` body surface. The positive route starts
from an explicit selected `tcrv.exec` RVV variant, carries lhs/rhs/accumulator
and result roles through a generic typed `tcrv_rvv` compute op, then lets the
RVV provider derive route/header/intrinsic/artifact facts from typed
body/config/runtime structure.

This task is a route-supported and, if evidence succeeds, executable closure for
one integer vector multiply-accumulate case. It is not high-level matmul,
Linalg lowering, broad contraction coverage, or dtype/LMUL clone expansion.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`; initial `pwd`,
  `git status --short`, and `git log --oneline -8` showed a clean worktree at
  `3aa02e8 rvv: validate typed config route facts`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-19-stage2-rvv-multiply-add-route-skeleton` and started.
- Relevant specs require the current RVV authority chain:
  `tcrv.exec` selected RVV variant -> explicit typed low-level `tcrv_rvv` body
  -> RVV plugin legality/provider -> provider-built
  `TCRVEmitCLowerableRoute` -> neutral common EmitC/export -> target artifact.
- Stage 1 guardrails remain active: no positive `rvv-i32m1`, `RVVI32M1`,
  finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door,
  descriptor, artifact-name, or exact intrinsic-spelling route authority may be
  reintroduced.
- Current Stage 2 coverage already has generic typed arithmetic, RHS
  broadcast, compare/select, reduce-add executable closure, masked add
  executable closure, and typed config validation for m1/m2 arithmetic.
- Current generic typed RVV surface includes `tcrv_rvv.load`,
  `tcrv_rvv.broadcast_load`, `tcrv_rvv.binary`, `tcrv_rvv.compare`,
  `tcrv_rvv.select`, `tcrv_rvv.masked_binary`, `tcrv_rvv.reduce`, and
  `tcrv_rvv.store`.
- There is not yet a generic typed body operation that structurally represents
  `acc + lhs * rhs` / multiply-accumulate roles as one contraction-supporting
  route.

## Requirements

1. Add one generic typed RVV multiply-add/accumulate body surface, expected as a
   non-dtype-prefixed op such as `tcrv_rvv.macc`.
2. The body must structurally consume:
   - lhs vector `!tcrv_rvv.vector<elem, lmul>`;
   - rhs vector `!tcrv_rvv.vector<elem, lmul>`;
   - accumulator vector `!tcrv_rvv.vector<elem, lmul>`;
   - active `!tcrv_rvv.vl` token;
   - explicit operation kind, initially one bounded add-accumulate case.
3. The dialect verifier must accept a valid typed multiply-add body and reject:
   - unsupported kind;
   - lhs/rhs/accumulator/result type mismatch;
   - non-generic vector operands;
   - mismatched VL token relative to enclosing `tcrv_rvv.with_vl`;
   - missing/mismatched SEW/LMUL/policy config.
4. The construction protocol must recognize the multiply-add selected-body
   route from structural typed roles without treating manifests, route ids,
   artifact names, status fields, or fixture names as executable authority.
5. The RVV provider must derive route/profile/header/intrinsic/materialization
   facts only after typed body/config/runtime validation. Unsupported
   multiply-add forms must fail closed before target artifact construction.
6. Positive materialization/artifact coverage must use an explicit selected
   RVV variant with typed `tcrv_rvv` body authority.
7. Update generated-bundle evidence tooling only as a runner/checker for
   generated artifacts. Python must not implement compiler IR, provider logic,
   lowering, or emission semantics.
8. Preserve existing arithmetic, broadcast, compare/select, reduce-add, masked
   add, and typed-config behavior.

## Acceptance Criteria

- [x] `tcrv_rvv` verifier accepts a valid generic typed multiply-add body and
      rejects invalid kind/type/VL/config combinations with targeted
      diagnostics.
- [x] RVV construction protocol recognizes the multiply-add selected-body route
      as a provider-owned specialization, without broad contraction framework
      expansion.
- [x] RVV EmitC route provider derives multiply-add route metadata,
      header/intrinsic facts, accumulator layout, and materialized payload only
      from typed body/config/runtime facts.
- [x] Positive materialization and target artifact fixtures prove the route
      reaches provider-derived route metadata and generated header/object bundle
      through generic typed `tcrv_rvv` body structure.
- [x] Negative fail-closed tests cover unsupported multiply-add forms and stale
      legacy/source-front-door authority before target artifact construction.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py --op-kind macc_add` supports
      dry-run verification; real `ssh rvv` correctness evidence is recorded if
      executable status is claimed.
- [x] Focused build/tests for touched RVV dialect/provider/construction/script
      and target paths pass.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`,
      `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-front-door, descriptor, or common/export RVV
      semantic authority is reintroduced.

## Non-Goals

- No high-level Linalg/matmul/convolution lowering.
- No broad contraction family, new dtype family, LMUL clone batch, conversion
  expansion, global tuning, dashboards, or readiness state machines.
- No source-front-door positive RVV routes.
- No descriptor-driven computation.
- No one-intrinsic wrapper surface detached from typed body roles.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.
- No performance claim.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-target-artifact-export-test`, and
   `tianchenrv-construction-protocol-common-test`.
3. Run focused RVV dialect lit tests for multiply-add verifier
   positives/negatives.
4. Run focused EmitC/target artifact lit/FileCheck tests for multiply-add
   materialization and fail-closed unsupported cases.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   script self-test if the script changes.
6. Run generated-bundle dry-run for `--op-kind macc_add`.
7. Run real `ssh rvv` correctness evidence for `--op-kind macc_add` only if
   the generated bundle compiles/runs through the evidence harness.
8. Run `git diff --check`.
9. Run active-authority scan over active RVV include/lib/test/script paths.
10. Run broader `check-tianchenrv` if shared route/export behavior changes
    enough to justify it.

## Implementation Results

- Added non-dtype-prefixed `tcrv_rvv.macc` with bounded `kind = "add"` semantics
  for `accumulator + lhs * rhs` under a selected `tcrv_rvv.with_vl` body.
- Extended the RVV dialect verifier to reject unsupported macc kind, vector
  type disagreement across lhs/rhs/accumulator/result, non-matching VL use, and
  typed SEW/LMUL/policy mismatches through the existing generic vector config
  checks.
- Allowed `tcrv_rvv.load` from an output-buffer runtime ABI value only as a
  typed body value; the RVV provider recognizes that output-buffer load as the
  explicit macc accumulator input and rejects unsupported load/role shapes.
- Extended RVV construction protocol route recognition with `macc_add`,
  provider-owned route/runtime ABI naming, selected role sequence, artifact
  role mirrors, and target artifact export coverage.
- Extended the RVV EmitC route provider to validate three vector loads
  (lhs/rhs/output accumulator), derive `rvv-generic-macc-add-emitc-route`,
  derive the `vmacc` intrinsic leaf after typed config validation, materialize
  accumulator load plus macc compute plus store, and emit explicit macc
  accumulator/result layout metadata mirrors.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with `--op-kind macc_add`
  dry-run and `ssh rvv` correctness harness support without moving compiler
  semantics into Python.

## Validation Results

- [OK] Trellis task context validation.
- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-target-artifact-export-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit set covering dialect macc verifier, macc materialization,
  negative provider rejection, target artifact fixture, macc script dry-run,
  and existing first-slice regression files: 10/10 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] local dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-macc-add-dry-run`.
- [OK] real `ssh rvv` macc correctness evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-macc-add-evidence`, with
  `PASS op=macc_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 161/161 passed.
- [OK] `git diff --check`
- [OK] diff-only active-authority scan: the only added legacy-shaped token is
  provider-derived `__riscv_vmacc_vv_i32m1`; no positive `rvv-i32m1`,
  `RVVI32M1`, `i32_binary_pre_realized_body`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door, descriptor-driven, or common/export
  semantic authority was introduced.

## Self-Repair

- Repaired the macc accumulator-type negative verifier test so the intended
  `tcrv_rvv.macc` type diagnostic is reached instead of being preempted by a
  `tcrv_rvv.load` config mismatch.
- Repaired the macc generated-bundle script FileCheck expectations to match the
  actual evidence schema and harness execution order.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-19-stage1-gate-a-rvv-route-identity-cleanup/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-05-19-stage1-generic-typed-rvv-body-surface/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-typed-config-arithmetic/prd.md`.
- Initial implementation surface:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  focused tests under `test/Dialect/RVV`, `test/Conversion/EmitC`,
  `test/Target/RVV`, and `test/Scripts`.

## Definition Of Done

- One coherent Stage 2 multiply-add/accumulate submodule is represented,
  verified, route-supported, and materialized through the production RVV
  provider path.
- Route-supported, dry-run artifact evidence, and executable `ssh rvv`
  evidence are reported separately and truthfully.
- No legacy i32 route authority, source-front-door positive route,
  descriptor-driven computation, or common EmitC/export semantic inference is
  introduced.
- The task context, PRD, implementation notes, validation, and final report
  match the actual completed module.
