# Stage2 RVV vector-reduction artifact/runtime ABI closure

## Goal

Make the selected typed RVV `reduce_add` vector-reduction artifact path
validator-backed and executable end to end. The production route must flow from
selected `tcrv.exec` RVV variant, through typed `tcrv_rvv.reduce` body,
RVV-owned provider facts, target-owned vector-reduction route-family
validation, generated object/header bundle export, and real `ssh rvv`
correctness evidence.

## Direction Source

- Direction title: `Switch: Stage2 RVV vector-reduction artifact/runtime ABI
  closure`.
- Module owner: RVV plugin route/provider facts plus target artifact export
  boundary for the `reduce_add` route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `319e1ad6 rvv: close artifact route-family validator coverage`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The immediate prior task closed route-family validator coverage and added a
  target-owned `vector-reduction` validator for selected typed `reduce_add`
  artifacts, while making no runtime correctness claim and running no `ssh rvv`.
- Current specs require selected RVV artifact export to rebuild provider route
  facts from the selected typed `tcrv_rvv` body and treat candidate metadata as
  mirrors after provider route construction.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already validates the
  `vector-reduction` route-family provider facts: selected `tcrv_rvv.reduce`,
  vector RHS-load memory form, route operand binding, runtime ABI order,
  setvl/VL loop shape, reduction layouts, vector load/reduction/store
  statements, and stale non-vector-reduction facts.
- `RVVEmitCRouteProvider.cpp` already requires math operand-binding facts for
  `ReduceAdd` before statement construction and builds the route through
  provider-owned setvl, load, reduction, and store statements.
- `RVVEmitCRoutePlanning.cpp` already serializes provider mirrors such as
  runtime ABI order, route operand binding, provider support mirror, reduction
  accumulator/result layout, and reduction store VL.
- `scripts/rvv_generated_bundle_abi_e2e.py` is the local and remote evidence
  driver for generated RVV bundles. It currently has `reduce_add` support, but
  this round must prove that evidence consumes provider/target-validator facts
  rather than route id, artifact name, script convention, or mirror-only
  metadata as support authority.
- The current vector-reduction validator intentionally relies on provider
  route setvl/runtime ABI/VL and reduction facts, not an over-broad
  `runtimeControlPlanID` mirror requirement.

## Requirements

1. Keep `reduce_add` support on the selected typed RVV path:
   selected `tcrv.exec` RVV variant -> typed `tcrv_rvv.reduce` body -> RVV
   provider route facts -> target-owned vector-reduction validator -> generated
   bundle.
2. Ensure generated-bundle dry-run evidence for `reduce_add` explicitly records
   the authority chain as provider-derived typed-body/config/runtime facts and
   target-owned vector-reduction validation.
3. Ensure the generated evidence exposes the facts the validator consumes:
   operation kind, vector RHS-load input memory role, accumulator/seed role,
   scalar lane-0 output store behavior, element dtype, SEW/LMUL, runtime
   `n`/AVL/VL relation, setvl placement, tail/mask policy, provider support
   mirror, route operand binding, runtime ABI order, and artifact ABI order.
4. Ensure mirror fields are labeled as mirrors after provider route
   construction and are never treated as support authority by the script,
   tests, or target exporter.
5. Add focused coverage proving stale or missing vector-reduction provider or
   target-validator facts fail closed before generated-bundle evidence can be
   accepted.
6. Run real `ssh rvv` evidence for `reduce_add` over counts including `0`, `1`,
   exact-VL, tail, and stress cases, with signed input patterns, nonzero
   seed/accumulator inputs, scalar output checks, and tail/sentinel
   preservation where applicable.
7. Preserve the current direct pre-realized route-entry stance:
   `reduce_add` uses the selected-boundary/provider route and does not reopen
   direct pre-realized route-entry support.
8. Keep common EmitC and target support neutral. Do not move RVV semantic
   authority into common EmitC/export, artifact names, route ids, ABI strings,
   scripts, exact intrinsic spellings, descriptors, source-front-door markers,
   or metadata support fields.

## Acceptance Criteria

- [x] PRD and Trellis context are truthful and started from the Hermes brief.
- [x] Production or directly adjacent evidence/tooling behavior changes in the
      vector-reduction route/provider/target/export/generated-bundle boundary,
      not only broad smoke tests.
- [x] `reduce_add` generated-bundle dry-run proves provider-built
      vector-reduction route facts and target-owned validator consumption.
- [x] Dry-run output labels provider/candidate metadata as mirrors and exposes
      an explicit vector-reduction boundary summary.
- [x] C++ or lit coverage proves stale/missing vector-reduction provider facts
      or stale candidate mirrors fail closed before artifact acceptance.
- [x] Direct route-entry support remains false for this family unless a later
      task changes that architecture.
- [x] `ssh rvv` non-dry-run evidence passes for counts `0`, `1`, exact-VL,
      tail, and stress, including signed data and nonzero accumulator/seed
      cases. Evidence:
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-reduce-add-ssh-rvv`.
- [x] Focused target/generated-bundle tests pass.
- [x] Route-family validator fail-closed regression remains covered.
- [x] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      support authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Trellis task status, journal/archive state, and commit state are
      truthful at the end of the round.

## Current Round Status

- Completed local implementation and validation for target-owned
  vector-reduction `reduce_add` provider facts, candidate mirrors, generated
  bundle dry-run evidence, and fail-closed C++ coverage.
- Generated-bundle dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-reduce-add-dryrun-boundary`.
- Completed `ssh rvv` evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-reduce-add-ssh-rvv`.
  The rerun reached the real `rvv` target, compiled with remote `riscv64`
  clang, and passed counts `0`, `1`, `16`, `17`, and `257`.
- The generated harness now uses signed `int32_t` input patterns for both the
  reduction source and RHS seed input, checks the scalar lane-0 result, reports
  `rhs_seed` in mismatch diagnostics, and preserves non-result-lane sentinels.
- Final evidence command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add
  --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17
  --runtime-count 257 --run-id codex-reduce-add-ssh-rvv --overwrite`.
- Spec-update judgment: no `.trellis/spec/**` edit is needed. This round
  follows existing specs for provider-derived typed `tcrv_rvv` authority,
  target-owned route-family validation, mirror-only artifact metadata, and real
  `ssh rvv` evidence.

## Out Of Scope

- Standalone min/max reductions.
- Computed-mask reductions.
- Runtime-scalar masked reductions.
- Widening contractions or widening dot reductions.
- Compare/select, segment2, conversion, dtype/LMUL clone batches, or new
  feature coverage.
- High-level Linalg/frontend lowering.
- One-intrinsic wrapper dialects.
- Validator coverage-only work without generated artifact/runtime closure.
- Dashboards, reports, or broad smoke matrices.
- Reopening direct pre-realized route-entry support for this family.

## Technical Approach

1. Inspect the current `reduce_add` selected-body route, provider description,
   target validator, generated bundle script, and existing direct tests.
2. Identify the narrow missing boundary: likely generated-bundle evidence
   summaries, dry-run assertions, runtime harness checks, or focused target
   validator negative coverage.
3. Implement only the bounded production or directly adjacent behavior needed
   for the evidence path to prove provider/target-validator-backed authority.
4. Run focused local checks first, including generated-bundle dry-run for
   `reduce_add`.
5. Run `ssh rvv` generated-bundle correctness evidence over the required
   runtime counts.
6. Run bounded authority scans, `git diff --check`, and `check-tianchenrv` if
   focused checks pass.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. `build/bin/tianchenrv-rvv-extension-plugin-test`
4. `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
5. `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
6. `git diff --check`
7. Bounded authority scan over touched RVV planning/provider/target/script/test
   files.
8. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-route-family-validator-coverage-gate/prd.md`

Primary files to inspect or modify:

- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/TargetArtifactExportTest.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
