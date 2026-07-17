# Stage2 RVV Compare/Select Executable Artifacts

## Goal

Close executable generated-bundle evidence for bounded RVV `cmp_select` and
`computed_mask_select` through the selected lowering-boundary path. The round
must demonstrate that already selected typed/pre-realized `tcrv_rvv`
compare/select bodies are consumed by RVV selected-body materialization, feed
provider-derived route facts, lower through `TCRVEmitCLowerableRoute` and
neutral EmitC, export target ABI/header/object bundles, generate RVV C
artifacts, and pass real `ssh rvv` correctness runs.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current branch is `main`; the working tree was clean before task creation.
- Latest commit is `5e2467fa rvv: close widening conversion executable artifacts`.
- No `.trellis/.current-task` existed when this task was created.
- The Direction Brief identifies `cmp_select` and `computed_mask_select` as the
  next adjacent selected-boundary generated-bundle evidence closure.
- Direct pre-realized route-entry support must remain fail-closed for these
  compare/select families in this round.
- Specs require compare/select generated-bundle evidence to expose
  `compare_select_predicate_boundary` facts sourced from typed body/config/runtime
  facts, not from route ids, artifact names, ABI strings, test names, exact
  intrinsic spelling, descriptors, source-front-door paths, or metadata mirrors.

## Requirements

- Use the selected lowering-boundary materializer path for both `cmp_select` and
  `computed_mask_select`.
- Preserve `route_entry_realization=false` and `pre_realized_body_consumed=true`
  for these generated-bundle paths.
- Preserve predicate kind, predicate source/role, mask source, select layout,
  true/false value roles, memory roles, runtime `n`/AVL/VL values, policy, ABI
  order, provider route facts, and generated target artifact facts.
- Ensure direct pre-realized route-entry attempts for compare/select remain
  op-specific fail-closed checks.
- Make production code changes only if the existing path cannot honestly produce
  executable generated-bundle evidence.
- Keep the task tied to the compiler route: selected `tcrv.exec` RVV variant ->
  typed/pre-realized `tcrv_rvv` body -> RVV selected-body materialization ->
  provider-built route -> common EmitC -> target artifact -> RVV C harness.

## Acceptance Criteria

- [x] Focused generated-bundle dry-run passes for `cmp_select`.
- [x] Focused generated-bundle dry-run passes for `computed_mask_select`.
- [x] Dry-run evidence checks selected-boundary materialization, provider facts,
      ABI/header/object bundle facts, `compare_select_predicate_boundary`, and
      absence of descriptor/direct-C/source-export/source-front-door authority.
- [x] `ssh rvv` compile/run/correctness passes for runtime counts including
      `0`, `1`, exact-vector, tail, and stress cases.
- [x] Runtime evidence covers predicate-true, predicate-false, mixed-mask, tail
      preservation, and signed compare behavior where applicable.
- [x] Direct pre-realized route-entry compare/select probes remain fail-closed
      with targeted diagnostics.
- [x] Non-regression coverage for the just-closed widening conversion executable
      artifacts passes.
- [x] Bounded changed-line authority scan finds no direct-route-entry-only,
      metadata-derived, descriptor-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, route-id-derived, exact-intrinsic-derived,
      source-front-door-derived, pre-realized-fixture-only, or legacy-i32 route
      authority.
- [x] `python3 -m py_compile` and focused script self-test pass if
      `scripts/rvv_generated_bundle_abi_e2e.py` is touched.
- [x] Focused lit / C++ checks relevant to compare/select pass.
- [x] `git diff --check` passes and final `git status --short` is clean after
      commit.

## Definition Of Done

- Bounded implementation or evidence closure completed.
- Relevant tests and RVV runtime evidence run and recorded in task notes or
  journal.
- Trellis task status is truthful, finished/archived only if acceptance is met.
- One coherent commit records the round.

## Out Of Scope

- Runtime-scalar select, segment2, strided load/store, standalone reduction,
  MAcc, widening dot, conversion expansion, dtype/LMUL clone batches, high-level
  Linalg/frontend lowering, one-intrinsic wrappers, selected-body framework
  rewrites, dashboards, broad smoke matrices, route-table resurrection, and
  direct pre-realized route-entry support for compare/select.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/guides/capability-first-design-guide.md`
  - `.trellis/spec/guides/plugin-locality-review-guide.md`
  - `.trellis/spec/guides/compute-boundary-review-guide.md`
- Relevant starting files from the Direction Brief:
  - archived task `05-28-stage2-rvv-widening-conversion-executable-artifacts`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-mask-select-dry-run.test`
  - direct pre-realized compare/select fail-closed probes
  - `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select*.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select*.mlir`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`

## Completion Evidence

- Production code changes: none required. The existing selected-boundary
  compiler path honestly produced executable generated-bundle artifacts.
- Focused selected-boundary dry-run evidence:
  - `cmp_select`, `cmp_select_sle` at
    `artifacts/tmp/stage2_compare_select_executable_closure/selected-boundary-dry-run/cmp-select`.
  - `computed_mask_select`, `computed_mask_select_sle` at
    `artifacts/tmp/stage2_compare_select_executable_closure/selected-boundary-dry-run/computed-mask-select`.
- Direct pre-realized route-entry fail-closed evidence:
  - `cmp_select` and `computed_mask_select` failed with the targeted
    selected-boundary-only diagnostic under
    `artifacts/tmp/stage2_compare_select_executable_closure/direct-route-entry-negative`.
- Executable `ssh rvv` evidence:
  - `cmp_select`, `cmp_select_sle`, `computed_mask_select`, and
    `computed_mask_select_sle` compiled and ran successfully for counts
    `0`, `1`, `16`, `23`, and `257` under
    `artifacts/tmp/stage2_compare_select_executable_closure/ssh-rvv-executable/compare-select-executable`.
  - Runtime output reported true/false select coverage, signed compare behavior,
    and tail preservation where applicable.
- Non-regression evidence:
  - `widen_i16_to_i32` and `widen_i32_to_i64` executable generated-bundle checks
    passed on `ssh rvv` for counts `0`, `1`, `16`, `23`, and `257` under
    `artifacts/tmp/stage2_compare_select_executable_closure/widening-non-regression/widening-conversion-non-regression`.
- Focused checks:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - focused lit filter for compare/select generated-bundle and direct-route
    probes: `4/4` passed.
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`: `405/405` passed.
- Authority scan:
  - Changed-line scan found only negative guard strings, explicit
    `route_entry_realization=false`, and emitted-intrinsic evidence checks.
    No new positive route authority came from direct route entries, descriptors,
    script/artifact names, common EmitC, source-front-door paths, or legacy i32
    route surfaces.
