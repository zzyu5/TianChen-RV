# Stage2 executable closure for generic RVV reduction

## Goal

Close the existing generic `tcrv_rvv.reduce {kind = "add"}` Stage 2 skeleton as
one executable selected-body path. The selected `tcrv.exec` RVV variant should
carry explicit typed reduction input, accumulator, result layout, runtime
`n`/AVL binding, provider-derived route facts, generated header/object bundle,
and real `ssh rvv` correctness evidence for representative counts.

This task is executable closure for one already-supported generic reduce-add
instance. It is not new Stage 2 coverage expansion.

## What I Already Know

- The repo root is `/home/kingdom/phdworks/TianchenRV`; initial `git status
  --short` was clean.
- Current HEAD is `8ccc2aa rvv: clean stage1 route identity residues`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief as `.trellis/tasks/05-19-stage2-generic-rvv-reduction-executable-closure`.
- The archived Stage 1 Gate A task completed the active route-identity cleanup:
  positive RVV route/header/ABI/bundle/config naming is now generic typed-body
  based rather than `rvv-i32m1`/`RVVI32M1` based.
- The archived Stage 2 reduction skeleton task added generic `tcrv_rvv.reduce`
  with route-supported EmitC/header/artifact dry-run evidence, but explicitly
  did not collect `ssh rvv` correctness evidence.
- The current positive fixture
  `test/Target/RVV/explicit-selected-body-artifact-reduce-add.mlir` uses
  `load lhs`, `load rhs`, `reduce(add)`, and `store out` with runtime ABI
  parameters `lhs,rhs,out,n`.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently supports executable
  bundle ABI evidence for add/sub/mul/cmp_select and selected-body variants,
  but not `reduce_add`.
- Specs require RVV dtype/config/operation/runtime facts to come from typed
  `tcrv_rvv` body/config/runtime ABI structure and be validated by the RVV
  plugin. Common EmitC/export must remain neutral and must not infer reduction
  semantics from names, route ids, metadata, or descriptors.

## Requirements

1. Make the bounded generic reduce-add selected body executable through the
   existing production path:
   `tcrv.exec` selected RVV variant -> explicit typed `tcrv_rvv` body ->
   RVV provider validation -> `TCRVEmitCLowerableRoute` -> common EmitC ->
   target header/object bundle -> external C ABI consumer.
2. Define and enforce the reduction runtime layout for this bounded slice:
   - `lhs` is the vector input buffer;
   - `rhs` is the vector accumulator/seed buffer;
   - `out` is the output buffer;
   - `n` is the runtime element count / AVL source;
   - result behavior is testable and documented by the selected-body fixture,
     generated C, and evidence harness.
3. Keep reduction facts structural:
   - operation kind comes from `tcrv_rvv.reduce {kind = "add"}`;
   - element type, SEW, LMUL, VL, policy, and memory form come from typed
     `tcrv_rvv` body/config/runtime facts;
   - provider owns intrinsic/header/artifact facts after validation.
4. Add `reduce_add` support to the generated-bundle ABI evidence script without
   making Python a compiler implementation surface. The script may only drive,
   check, and run generated artifacts.
5. Keep common EmitC/export neutral. Do not add common RVV semantic branches,
   descriptor-driven compute, or metadata-derived route construction.
6. Preserve Stage 1 guardrails: do not reintroduce positive `rvv-i32m1`,
   `RVVI32M1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door positive
   routes, or compatibility wrappers.

## Acceptance Criteria

- [ ] The positive reduce-add selected-body fixture generates provider-derived
      route metadata, header, object, and bundle artifacts.
- [ ] The emitted reduction implementation has explicit accumulator/result
      layout matching the PRD and fixture, and it compiles for the RVV target.
- [ ] `scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add` supports
      dry-run bundle verification and real `ssh rvv` compile/link/run evidence.
- [ ] Real `ssh rvv` correctness evidence passes for representative counts,
      including at least one non-one-vector count.
- [ ] Focused local build/tests for touched RVV provider/dialect/construction/
      target/script paths pass.
- [ ] Active-authority scan over `include/TianChenRV`, `lib/Plugin/RVV`,
      `lib/Dialect/RVV`, `test/Target/RVV`, and `scripts` confirms no active
      `rvv-i32m1`, `RVVI32M1`, or `i32_binary_pre_realized_body` route authority
      is reintroduced. Any exact `__riscv_*_i32m1` spelling must be a
      provider-derived intrinsic leaf after typed-body validation.
- [ ] Task context is validated, task status is updated truthfully, the task is
      finished/archived when complete, and one coherent commit is created.

## Non-Goals

- No broad reduction family, additional reduction kinds, dtype/LMUL clone batch,
  conversion coverage, contraction coverage, high-level Linalg/Vector/StableHLO
  frontend lowering, source-front-door positive route, dashboard, tuning, or
  performance claim.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.
- No descriptor-driven computation and no Python implementation of compiler
  core, dialects, passes, plugin registry, capability model, lowering, or
  emission.

## Validation Plan

1. Validate the Trellis task context and start the task.
2. Build focused targets for touched paths, expected to include `tcrv-opt`,
   `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-target-artifact-export-test`, and script self-test when changed.
3. Run focused lit/FileCheck tests for the reduce-add selected-body artifact
   fixture and any changed negative/provider coverage.
4. Run generated-bundle dry-run for `--op-kind reduce_add`.
5. Run generated-bundle real `ssh rvv` evidence for `--op-kind reduce_add` with
   representative counts.
6. Run `git diff --check`.
7. Run an active-authority scan over the active RVV include/lib/test/script
   paths.
8. Run `check-tianchenrv` only if shared source or fixtures changed enough to
   justify the broader suite.

## Technical Notes

- Read specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Read prior PRDs:
  `.trellis/tasks/archive/2026-05/05-19-stage1-gate-a-rvv-route-identity-cleanup/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route/prd.md`.
- Initial code surface:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/explicit-selected-body-artifact-reduce-add.mlir`.

## Implementation Results

- Extended `RVVSelectedBodyEmitCRouteDescription` with provider-derived
  reduction layout mirrors for the bounded reduce-add route:
  `rhs-vector-seed-lane0-per-vl-chunk`,
  `store-reduction-lane0-to-output-chunk-base`, and store VL `1`.
- Updated the RVV EmitC route provider so `tcrv_rvv.reduce {kind = "add"}`
  still derives the reduction intrinsic leaf from typed body/config/runtime
  facts, but the final generated store writes only lane 0 of each dynamic
  reduction chunk to `out[offset]`.
- Added reduction layout metadata to the provider-derived target artifact
  metadata for reduce-add only. Non-reduction routes must continue to carry
  empty reduction layout fields and verify fail-closed if stale values appear.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with explicit
  `--op-kind reduce_add` support. The harness compiles as an external C ABI
  consumer, recomputes chunk boundaries with `__riscv_vsetvl_e32m1`, verifies
  `out[chunk_start] = rhs[chunk_start] + sum(lhs[chunk_start:chunk_start+vl])`,
  and verifies non-result lanes remain untouched.
- Kept common EmitC/export neutral: the only semantic change is in the RVV
  provider-built route payload and its evidence script checks.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage2-generic-rvv-reduction-executable-closure`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind reduce_add --run-id local-reduce-add-dry-run --overwrite --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e`
- [OK] Existing selected-body dry-run regression:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind add --op-kind sub --op-kind mul --op-kind cmp_select --run-id selected-body-existing-ops-dry-run --overwrite --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e`
- [OK] Real RVV correctness evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add --run-id ssh-reduce-add-evidence --overwrite --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --timeout 180 --connect-timeout 10`
- [OK] `ssh rvv` output passed counts `1,7,16,17,257`:
  `tcrv_rvv_generated_bundle_abi_reduce_add_ok counts=1,7,16,17,257` and
  `PASS op=reduce_add counts=1,7,16,17,257`.
- [OK] Evidence root:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-reduce-add-evidence`.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 154/154 lit
  tests.
- [OK] `git diff --check`
- [OK] Diff-only active-authority scan over `include/TianChenRV`,
  `lib/Plugin/RVV`, `lib/Dialect/RVV`, `test/Target/RVV`, and `scripts` found
  no newly introduced `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`,
  finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, or exact
  `__riscv_*_i32m1` authority. Full-path matches remain pre-existing
  negative fail-closed fixtures, supervisor prompt guardrails, legacy source
  seed names, dialect deletion debt, remote probe leaf intrinsics, or RVV
  provider-derived intrinsic leaves.

## Spec Update Decision

No `.trellis/spec/` update was needed. The existing specs already require RVV
plugin-owned reduction layout, provider-derived route facts, common EmitC
neutrality, and real `ssh rvv` evidence for correctness claims. This task
instantiates that contract for one bounded reduce-add route rather than adding
a new durable architecture rule.

## Definition Of Done

- [x] One bounded generic reduce-add selected-body path is executable through
      generated header/object bundle and an external C ABI consumer.
- [x] Accumulator/result layout is provider-derived, explicit, verified by
      artifact metadata, and exercised by the hardware harness.
- [x] Real `ssh rvv` correctness evidence exists for representative counts.
- [x] Existing add/sub/mul/cmp_select selected-body bundle dry-runs still pass.
- [x] No common EmitC/export semantic inference, descriptor-driven path, or
      positive legacy i32 route authority was introduced.
