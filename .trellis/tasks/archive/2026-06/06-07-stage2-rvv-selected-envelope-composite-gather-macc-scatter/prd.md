# Stage2 RVV selected-dispatch envelope for composite gather-MAcc-scatter

## Goal

Make the existing runtime-scalar computed-mask indexed gather + MAcc +
indexed scatter RVV composite route consume and preserve the selected
`tcrv.exec` dispatch/fallback envelope ABI boundary through artifact export, or
fail closed when the selected envelope binding/mirror is stale. This continues
after commit `61af3f6a`, which closed executable generated-bundle evidence but
did not advance the top-level selected `tcrv.exec` envelope seam.

## Requirements

- Preserve the authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed/realized `tcrv_rvv`
  body -> RVV provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target
  artifact.
- Keep `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` as ABI/runtime role
  declarations only. The selected `tcrv_rvv` body must import runtime ABI
  values and bind them to those declarations through `exec_binding`.
- Use the existing RVV provider-owned `tcrv_rvv.require_exec_abi_bindings` /
  `tcrv_rvv.exec_abi_bindings` mechanism for the composite route rather than
  adding route-name, artifact-name, descriptor, or common EmitC inference.
- Target artifact validation must compare any provider-derived
  `tcrv_rvv.exec_abi_bindings` mirror against the rebuilt provider description,
  and reject stale or missing candidate metadata.
- Positive composite artifact fixtures must prove both explicit and
  pre-realized selected RVV bodies carry selected dispatch/fallback mirrors and
  exec ABI binding mirrors into plan/header export.
- Negative evidence must cover at least one missing selected-body exec binding
  and one stale exported exec ABI binding mirror.

## Acceptance Criteria

- [x] Explicit composite selected-body artifact test carries
  `mem_window`/`runtime_param` declarations, `exec_binding` references, selected
  dispatch/fallback mirrors, and `tcrv_rvv.exec_abi_bindings` in plan/header
  output.
- [x] Pre-realized composite selected-body artifact test carries the same
  envelope binding and mirror evidence after RVV plugin-local realization.
- [x] Target artifact validation fails closed when candidate
  `tcrv_rvv.exec_abi_bindings` metadata is stale or missing relative to the
  provider description.
- [x] Provider route construction fails closed when a required composite
  runtime ABI value lacks `exec_binding`.
- [x] Existing composite runtime ABI order, provider mirror, route operand
  binding, mask/gather/MAcc/scatter facts, header/prototype, and selected
  dispatch/fallback behavior remain unchanged.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Focused lit/FileCheck tests for the changed composite selected-envelope
  fixtures pass.
- [x] Bounded scan over touched code/tests shows no new positive legacy
  `i32m1`, source-front-door, descriptor, route-id, artifact-name, or common
  EmitC semantic authority.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Final worktree is clean after one coherent commit if the task completes.

## Technical Approach

Use the existing provider-side selected envelope mechanisms instead of
inventing a new route. The composite fixtures will opt into
`tcrv_rvv.require_exec_abi_bindings = true`, add direct kernel
`tcrv.exec.mem_window` / `tcrv.exec.runtime_param` declarations, and attach
`exec_binding = @...` to the composite `tcrv_rvv.runtime_abi_value` ops. The RVV
provider already summarizes these bindings from the same runtime ABI parameter
list that constructs the route.

The production code change is in RVV target artifact validation:
`validateRVVRouteMetadataMirrorsSelectedBody` will require
`tcrv_rvv.exec_abi_bindings` to exactly mirror
`RVVSelectedBodyEmitCRouteDescription::execABIBindingSummary` when the provider
emits one, and reject stale mirror metadata when the provider does not.

## Decision (ADR-lite)

Context: the composite route already has provider route facts, target mirrors,
generated-bundle dry-run, and `ssh rvv` runtime evidence, but the selected
`tcrv.exec` envelope boundary was not proved through mem_window/runtime_param
binding.

Decision: reuse the existing generic RVV exec ABI binding mechanism and add
target-side stale mirror validation. Do not add a composite-specific core op,
common EmitC semantic branch, descriptor path, or new runtime artifact source.

Consequences: the first round is a bounded selected-envelope validation and
artifact rejection closure. It does not add new composite math coverage or
claim new runtime correctness beyond the previously archived `ssh rvv`
evidence.

## Out Of Scope

- No source-front-door or source-artifact positive route.
- No high-level Linalg/Vector/StableHLO frontend.
- No new gather/scatter/MAcc compute semantics in common EmitC or export code.
- No broad composite matrix, dtype/LMUL clone batch, or unrelated route family
  rewrite.
- No performance tuning database, dashboard, report-only closure, or new
  `ssh rvv` runtime correctness claim unless code changes require rerunning it.
- No compatibility wrapper around old i32 route authority.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/guides/capability-first-design-guide.md`
  - `.trellis/spec/guides/plugin-locality-review-guide.md`
  - `.trellis/spec/guides/compute-boundary-review-guide.md`
- Previous archived task read:
  - `.trellis/tasks/archive/2026-06/06-07-06-07-stage2-rvv-composite-gather-macc-scatter-ssh-rvv-runtime-boundary/prd.md`
- Key implementation references:
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  - `test/Target/RVV/selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative.mlir`

## Completion Notes

- Production seam changed: RVV target artifact selected-body validation now
  requires `tcrv_rvv.exec_abi_bindings` metadata to mirror the provider-derived
  selected `tcrv.exec` ABI binding summary.
- Explicit and pre-realized composite artifact fixtures now declare selected
  envelope `mem_window`/`runtime_param` ABI roles, bind final composite
  `tcrv_rvv.runtime_abi_value` parameters through `exec_binding`, and check
  dispatch/fallback plus exec ABI mirrors in plan/header output.
- Fail-closed evidence covers stale exported exec ABI binding metadata, missing
  exported exec ABI binding metadata, and missing selected-body `exec_binding`
  for a required runtime ABI value.
- Checks run:
  - `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - focused `tcrv-opt`/`tcrv-translate` positive and negative pipelines for
    explicit and pre-realized composite envelope fixtures
  - focused lit from `build/test` with LLVM 20 tools for the two changed target
    RVV tests
  - bounded old-authority scan over touched files and added diff lines
  - `git diff --check`
- No new `ssh rvv` runtime correctness claim was made; this round preserves the
  previously archived runtime evidence and closes the selected-envelope
  artifact/export boundary.
