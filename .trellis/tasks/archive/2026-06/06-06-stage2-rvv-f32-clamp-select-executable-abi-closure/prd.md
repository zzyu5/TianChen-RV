# Stage2 RVV f32 clamp/select executable ABI closure

## Goal

Close the bounded executable ABI evidence gap for the already route-supported
and generated-bundle-capable RVV f32 clamp/select path. The owner is exactly:
selected/pre-realized typed `tcrv_rvv` f32 clamp/select body -> RVV provider
route facts -> `TCRVEmitCLowerableRoute` -> target header/artifact/generated
bundle -> real `ssh rvv` correctness evidence with scalar oracle and tail
preservation.

## What I already know

- The repository is on `main`, worktree was clean at task creation, and HEAD is
  `4c3afc5d rvv: close f32 clamp select artifact mirrors`.
- The previous archived f32 clamp/select artifact closure task completed
  provider/header mirror validation and generated-bundle dry-run evidence, but
  explicitly did not claim `ssh rvv` executable correctness.
- `.trellis/spec/testing/mlir-testing-contract.md` has a dedicated f32
  clamp/select generated-bundle evidence contract. The command shape is
  `python3 scripts/rvv_generated_bundle_abi_e2e.py
  --pre-realized-selected-body --op-kind f32_clamp_select`.
- The external ABI must remain:
  `void tcrv_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(const float *input, float lower_bound, float upper_bound, float *out, size_t n);`
- The script already supports remote execution through `--ssh-target rvv` and
  local dry-run bundle generation. A focused dry-run with runtime counts
  `0,1,16,17,257` passed in this round and produced
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T180807Z`.

## Requirements

- Use the existing generated-bundle dry-run path as the starting point.
- Produce the f32 clamp/select generated bundle from the pre-realized selected
  body path without introducing new route-family expansion or common EmitC
  clamp/select semantics.
- Execute the generated bundle on real `ssh rvv` hardware.
- Validate scalar-oracle clamp/select behavior for runtime counts covering
  empty count, single lane, VL-boundary, tail, and multi-chunk cases.
- Validate at least two ordered nontrivial lower/upper bound pairs, both data
  patterns, below-bound, in-bound, and above-bound lanes.
- Preserve source buffer contents and output tail sentinels.
- If the bundle or harness cannot execute, repair only the minimal production
  or script blocker needed to execute this bounded route while preserving
  provider-owned route facts.

## Acceptance Criteria

- [x] `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --op-kind f32_clamp_select --runtime-count 0 --runtime-count 1
  --runtime-count 16 --runtime-count 17 --runtime-count 257 --dry-run`
  passes and records the generated bundle path.
- [x] The same generated-bundle path executes on `ssh rvv` with runtime counts
  `0,1,16,17,257`.
- [x] Remote stdout includes per-case success lines and final PASS markers for
  `patterns=0,1`, at least two bound pairs, scalar tolerance, source
  preservation, and tail preservation.
- [x] Evidence JSON records `ssh_evidence: true`, remote compile success,
  remote run success, the runtime count contract, f32 clamp/select route facts,
  lower/upper ABI roles, ordered bound relation, emitted C++ boundary, and
  runtime AVL/VL boundary.
- [x] Prior target artifact/header mirror checks for f32 clamp/select still
  pass.
- [x] Relevant script self-test and focused `tcrv-opt`/`tcrv-translate` checks
  pass.
- [x] If source changes are needed, touched-file and added-diff scans introduce
  no legacy `i32m1`, source-front-door, descriptor, route-id, artifact-name, or
  common-EmitC semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status`
  are clean after commit.

## Out of Scope

- New clamp/select route-family expansion.
- Dequant/f32 route matrix work.
- High-level frontend work or per-Linalg route authority.
- Broad generated-bundle harness rewrite.
- Performance, autotuning, dashboards, or readiness state machines.
- Common EmitC invention of clamp/select semantics.
- Route-string, artifact-name, ABI-string, test-name, descriptor, manifest, or
  helper-only authority.
- Claiming explicit selected-body mode executable correctness unless it is
  already supported without broadening this task.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-f32-clamp-select-artifact-closure/prd.md`.
- Primary script: `scripts/rvv_generated_bundle_abi_e2e.py`.
- Primary fixtures:
  `test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir`
  and `test/Dialect/RVV/pre-realized-f32-clamp-select-negative.mlir`.
- Relevant production code if repair is needed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Plugin/RVV/EmitC/`, and `lib/Plugin/RVV/Construction/`.

## Completion Evidence

Completed as a bounded f32 clamp/select executable ABI closure. No compiler or
script source repair was needed: the existing generated-bundle path already
supported explicit runtime counts, so this round turned the prior dry-run-only
artifact into real `ssh rvv` correctness evidence.

Generated bundle and evidence:

- Dry-run path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T180807Z`.
- Executable evidence path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T180934Z`.
- Per-op evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T180934Z/f32_clamp_select/evidence.json`.
- Generated object/header/harness:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T180934Z/f32_clamp_select/generated_bundle/`
  and
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T180934Z/f32_clamp_select/rvv_generated_bundle_abi_f32_clamp_select_harness.c`.
- External ABI:
  `void tcrv_emitc_pre_realized_f32_clamp_select_kernel_pre_realized_rvv_f32_clamp_select(const float *input, float lower_bound, float upper_bound, float *out, size_t n);`

Remote `ssh rvv` evidence:

- Command:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind f32_clamp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`.
- Result: `rvv_generated_bundle_abi_e2e: success`.
- Evidence JSON records `ssh_evidence: true`, `remote_compile_succeeded:
  true`, and `remote_run_succeeded: true`.
- Remote compile environment: `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`, `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote stdout includes all cases for counts `0,1,16,17,257`, patterns
  `0,1`, and bound pairs `-1.5:2.25,-8:-0.75`.
- Final PASS marker:
  `PASS op=f32_clamp_select counts=0,1,16,17,257 patterns=0,1 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05`.
- Case coverage includes empty count, single lane, VL-boundary, tail, and
  multi-chunk sizes; lower-clamped, in-bound, and upper-clamped lanes; scalar
  oracle tolerance; source preservation; and output tail sentinel preservation.

Evidence JSON route facts:

- `compare_select_predicate_boundary.authority` is
  `provider-derived typed tcrv_rvv f32 clamp/select body/config/runtime-bound facts`.
- `materialized_body.pre_realized_body_consumed` is `true`.
- Runtime ABI order is `input,lower_bound,upper_bound,out,n`.
- Lower/upper roles are `lower-bound-scalar-value` and
  `upper-bound-scalar-value`.
- Bound order is `lower-bound-before-upper-bound`.
- Clamp relation is
  `input-lower-select-then-upper-select-f32-runtime-bounds`.
- `runtime_avl_vl_boundary` records `runtime_abi:n`,
  `__riscv_vsetvl_e32m1`, loop VL use, and runtime counts
  `0,1,16,17,257`.

Checks run:

- `cmake --build build --target tcrv-opt tcrv-translate`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact`
- `build/bin/tcrv-opt test/Dialect/RVV/pre-realized-f32-clamp-select-negative.mlir --split-input-file --verify-diagnostics`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind f32_clamp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --dry-run`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind f32_clamp_select --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 -m lit -sv test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir test/Dialect/RVV/pre-realized-f32-clamp-select-negative.mlir` was attempted but unavailable locally: `/usr/bin/python3: No module named lit`.

Source edits:

- No compiler, script, fixture, or spec source files were changed.
- Because there were no source changes, no touched-source added-diff legacy
  authority scan was needed for production code. The task context itself did
  not introduce a supported legacy i32m1 route.
