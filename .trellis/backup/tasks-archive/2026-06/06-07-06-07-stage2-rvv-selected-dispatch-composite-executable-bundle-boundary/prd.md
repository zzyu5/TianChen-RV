# Stage2 RVV selected-dispatch composite executable bundle boundary

## Goal

Carry the runtime-scalar-cmp masked indexed gather-MAcc-scatter selected
dispatch/fallback RVV envelope through the generated bundle ABI boundary and
real `ssh rvv` execution evidence, or harden the exact production seam that
prevents that claim.

This task extends commit `b37b2ac7` from structural selected
dispatch/fallback artifact validation into executable generated-bundle
evidence. It must not treat selected dispatch/fallback mirrors, route ids,
artifact names, helper names, or common EmitC metadata as executable authority.

## What I already know

* The previous archived task
  `06-07-stage2-rvv-selected-dispatch-fallback-composite-artifact-boundary`
  added focused fail-closed target artifact coverage for missing or stale
  selected dispatch case, fallback, runtime guard, selected-envelope ABI, and
  selected dispatch/fallback mirror facts.
* That task intentionally made no new runtime correctness claim and ran no new
  `ssh rvv` evidence.
* The existing explicit and pre-realized composite fixtures contain actual
  `tcrv.exec.dispatch` case/fallback structure and selected RVV bodies for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
* The generated-bundle ABI script already has dry-run coverage for the
  composite explicit and pre-realized input modes.
* The current e2e evidence checks provider-backed composite route facts,
  runtime ABI order, indexed gather/scatter forms, mask producer facts, and
  generated harness correctness expectations.
* The remaining bottleneck is whether generated-bundle evidence explicitly
  proves that the selected dispatch/fallback envelope facts survive through
  target artifact export, bundle generation, compile, and remote execution.

## Requirements

* Inspect the real repository state before editing and keep the task grounded
  in live code, specs, and the archived predecessor task.
* Use the existing RVV production authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV provider route -> common EmitC -> target artifact ->
  generated bundle -> `ssh rvv` evidence.
* If the production compiler path is already complete, make the proof concrete
  with focused non-dry-run generated-bundle evidence and an exact no-source-
  change justification.
* If the generated-bundle evidence tooling is under-validating the selected
  dispatch/fallback executable boundary, update that tooling so the evidence
  records and verifies selected dispatch case/fallback mirrors from the target
  artifact bundle as mirror-only facts after provider route construction.
* If compiler production code blocks the executable claim, fix the production
  owner that blocks the path instead of adding standalone negative fixtures.
* Preserve RVV plugin ownership of route, intrinsic, type, ABI, and body facts.
  Common EmitC/export remains semantic-neutral.
* Keep scalar fallback to the selected dispatch/fallback envelope contract
  only. Do not implement a scalar fallback backend.

## Acceptance Criteria

* [x] The selected dispatch/fallback composite route reaches emission plan,
  target artifact export, generated bundle ABI, generated harness compile, and
  non-dry-run `ssh rvv` execution for explicit selected-body input.
* [x] The same boundary is proven for pre-realized selected-body input, after
  selected-body realization consumes the pre-realized body.
* [x] Evidence JSON or focused checks expose the selected dispatch case and
  fallback mirrors for the generated bundle, and mark them as mirrors after
  provider route/artifact validation rather than route authority.
* [x] Existing fail-closed coverage for missing/stale selected dispatch case,
  fallback, runtime guard, selected-envelope ABI bindings, runtime ABI order,
  provider support mirror, and unsupported executable claims remains intact.
* [x] Focused dry-run lit coverage checks any changed generated-bundle evidence
  fields or script behavior.
* [x] Real `ssh rvv` output is captured before claiming runtime correctness.
* [x] Build `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
* [x] Run focused lit/script checks for the changed dispatch/composite/
  generated-bundle behavior.
* [x] Run a bounded old-authority scan over touched files and added diff lines.
* [x] Run `git diff --check`, `git diff --cached --check`, and finish with a
  clean git status after commit.

## Non-goals

* No broad route-family matrix.
* No dtype/LMUL clone batch.
* No new MAcc, gather, scatter, reduction, conversion, or frontend coverage
  family.
* No source-front-door positive route.
* No IME, offload, TensorExt, Template/Toy, or future plugin work.
* No scalar fallback backend implementation beyond the dispatch/fallback
  contract here.
* No dashboard, report-only, artifact-index-only, broad smoke-test-only, or
  Trellis-only closeout.
* No common EmitC invention of RVV semantics.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/variant-pipeline/index.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/validation/index.md`
* `.trellis/spec/validation/experiment-reference.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-selected-dispatch-fallback-composite-artifact-boundary/prd.md`

Primary implementation and evidence files:

* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

## Definition of Done

* The generated-bundle selected-dispatch/fallback composite executable
  boundary is either implemented and evidenced, or the exact blocking seam is
  documented with a continuation point.
* Trellis task state, PRD completion notes, and workspace journal are truthful.
* All focused checks needed for the changed behavior pass.
* A single coherent commit is created if the task is complete.

## Completion Notes

* Production compiler code did not need a semantic change. The previous
  structural task already made target artifact validation fail closed on stale
  or missing selected dispatch/fallback/runtime-guard/ABI/provider facts.
* The under-validated seam was the generated-bundle evidence tooling: it
  verified provider route, runtime ABI, indexed memory, mask producer, and
  harness correctness facts, but did not explicitly preserve the selected
  dispatch case/fallback boundary as a generated-bundle evidence field.
* Updated `scripts/rvv_generated_bundle_abi_e2e.py` so the composite explicit
  and pre-realized expectations carry selected dispatch case/fallback mirrors.
  The script now verifies object/header bundle metadata for selected dispatch
  case, fallback, exec ABI bindings, runtime ABI order, route operand binding
  plan/summary, and provider support mirror before writing
  `selected_dispatch_bundle_boundary`.
* The new evidence field is explicitly labeled
  `mirror-only-after-provider-route-and-selected-dispatch-validation`; runtime
  counts are recorded as execution cases, not dispatch authority.
* Updated the focused generated-bundle dry-run lit test to require the new
  explicit and pre-realized selected-dispatch bundle boundary fields.
* Updated `.trellis/spec/testing/mlir-testing-contract.md` with the reusable
  selected-dispatch generated-bundle evidence JSON contract.
* Non-dry-run `ssh rvv` evidence succeeded for explicit selected-body input:
  `artifacts/tmp/stage2-selected-dispatch-composite-gms-ssh-rvv/explicit-selected-dispatch-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
* Non-dry-run `ssh rvv` evidence succeeded for pre-realized selected-body input:
  `artifacts/tmp/stage2-selected-dispatch-composite-gms-ssh-rvv/pre-selected-dispatch-composite-gms-ssh-rvv/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
* Both remote runs reported
  `PASS op=runtime_scalar_cmp_masked_indexed_gather_macc_scatter counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
* Bounded old-authority scan over touched files and added diff lines found no
  new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, descriptor,
  direct-C, or source-front-door authority.
* Checks run:
  * `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  * `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  * explicit selected-dispatch composite dry-run generated-bundle evidence
  * pre-realized selected-dispatch composite dry-run generated-bundle evidence
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run`
  * `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
  * `build/bin/tianchenrv-rvv-extension-plugin-test`
  * `build/bin/tianchenrv-target-artifact-export-test`
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter|selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative|materialize-dispatch-runtime-guards|variant-dispatch-synthesis|rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run'`
  * explicit selected-dispatch composite non-dry-run `ssh rvv` generated-bundle
    evidence
  * pre-realized selected-dispatch composite non-dry-run `ssh rvv`
    generated-bundle evidence
  * `git diff --check`
