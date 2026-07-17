# RVV source-frontdoor generated artifact invocation closure

## Goal

Close the bounded generated-artifact invocation handoff for the existing MLIR
dynamic vector i32 add/sub source-frontdoor route. The route must start from
source vector/SCF fixtures, lower through the production source frontdoor and
planning pipeline, export the production RVV+scalar dispatch source/header/object
bundle, and invoke that generated dispatch bundle through the runtime ABI caller
used by `RVVScalarDispatch`.

This round is not a dispatch-validation-only or metadata-only milestone. The
expected coherent submodule is the RVV+scalar dispatch e2e harness and focused
tests needed to consume production plan-and-export bundle artifacts generated
from vector source-frontdoor inputs.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- At task creation, `git status --short` was clean and HEAD was
  `2a9b666 feat(rvv): validate dispatch source identity`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief.
- The immediately prior completed task changed `lib/Target/RVV/RVVScalarDispatch.cpp`
  so RVV+scalar dispatch preflight consumes selected RVV source identity and
  rejects stale selected source kinds.
- Older archived tasks already made dynamic vector i32 vadd/vsub default bundle
  export carry selected source identity and source-tail/runtime-length metadata.
- `scripts/rvv_microkernel_e2e.py` already supports
  `--lower-vector-i32-vadd-frontend` and `--lower-vector-i32-vsub-frontend`.
- `scripts/rvv_scalar_dispatch_e2e.py` currently supports linalg frontend and
  existing/preplanned fixtures, but it does not expose equivalent vector/SCF
  source-frontdoor flags for the RVV+scalar dispatch generated bundle invocation
  path.
- Focused TargetArtifactBundleExport tests already show vector dynamic vadd/vsub
  bundle records exist, so the missing closure is e2e consumption/invocation of
  those generated dispatch bundle artifacts rather than a new frontend family.

## Requirements

- Add bounded vector/SCF source-frontdoor entry points to
  `rvv_scalar_dispatch_e2e.py` for i32-vadd and i32-vsub.
- For those entry points, select the checked-in dynamic vector source fixtures
  and route them through the same production path as users would:
  `--tcrv-lower-source-rvv-binary-to-exec` plus
  `--tcrv-execution-planning-pipeline`, or the in-process
  `--tcrv-plan-and-export-target-artifact-bundle` front door.
- The e2e harness must consume generated target bundle records and generated
  source/header/object artifacts, not hand-written surrogate kernels,
  descriptor shape, bundle metadata alone, or explicit-only assumptions.
- The harness must verify that source-frontdoor generated dispatch artifacts
  preserve selected RVV source identity, selected vector config, runtime element
  count, dispatch availability guard, and source-tail/runtime-AVL authority when
  dynamic vector source metadata is present.
- Both vadd and vsub dry-run evidence must exercise the generated external C
  caller branch coverage for `rvv_available=0` and `rvv_available=1`.
- At least one source-frontdoor generated dispatch bundle invocation must be
  compiled/linked/run on `ssh rvv`, or the exact blocker must be recorded after
  local generation and compile.
- Generic core orchestration must remain target-neutral. Do not add RVV/scalar
  semantic branches in shared core passes and do not move compute semantics into
  `tcrv.exec`.

## Acceptance Criteria

- [x] `rvv_scalar_dispatch_e2e.py` accepts exactly one of linalg frontend,
      vector i32-vadd frontend, or vector i32-vsub frontend flags, and fails
      closed for conflicting frontend flags.
- [x] Vector i32-vadd and i32-vsub flags select the production dynamic vector
      fixtures when no explicit input is passed.
- [x] Non-bundle execution planning uses
      `--tcrv-lower-source-rvv-binary-to-exec` for vector flags before the normal
      execution planning pipeline.
- [x] Plan-and-export bundle mode passes the vector source fixture directly to
      `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle`.
- [x] Generated dispatch source and bundle metadata validation covers selected
      source identity, selected config, runtime element count, dispatch guard,
      branch coverage, and dynamic source-tail/runtime-AVL authority for vector
      source-frontdoor artifacts.
- [x] Focused lit/script coverage proves vector dynamic i32-vadd and i32-vsub
      dispatch bundle dry-runs use source-frontdoor inputs and reach generated
      dispatch source/header/object artifacts.
- [x] Local focused checks pass for the script self-test, vadd/vsub vector
      source-frontdoor dry-runs, relevant TargetArtifactBundleExport vector
      tests, and existing RVVScalarDispatch bundle e2e regression.
- [x] Focused `ssh rvv` compile/link/run evidence covers at least one vector
      source-frontdoor generated RVV+scalar dispatch bundle invocation, or an
      exact blocker is recorded.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      archive, and one coherent commit complete if the task is finished.

## Out Of Scope

- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 expansion, LMUL matrix expansion, third dynamic vector operation, broad
  family matrix, broad smoke suite, or performance tuning.
- No new explicit-only frontend flag as the main path; vector-specific flags in
  the e2e harness are bounded runner switches that drive the existing production
  source frontdoor.
- No descriptor-to-C production exporter.
- No descriptor element count or descriptor vector shape as compute, selected
  config, runtime AVL/VL, ABI, dispatch, or artifact authority.
- No moving computation semantics into `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No GCC/vendor compiler default route.
- No Template/Toy/TensorExtLite/IME/Offload work except narrow regressions
  caused by shared validation.
- No runtime, correctness, or performance claims beyond focused evidence
  actually run.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-production-artifact-runtime-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-artifact-runtime-closure/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Primary implementation surfaces:
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`,
  focused dynamic vector bundle tests under
  `test/Target/TargetArtifactBundleExport`.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Run C++ smoke tests:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run script self-test:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run local vector source-frontdoor dispatch bundle dry-runs for i32-vadd and
  i32-vsub through `--use-target-artifact-bundle`
  `--use-plan-and-export-bundle-front-door`.
- Run focused lit:
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  and focused RVVScalarDispatch regressions.
- Run focused `ssh rvv` dispatch bundle invocation for at least one dynamic
  vector source-frontdoor op.
- Run bounded ref-scans for descriptor-only authority, explicit-only misuse, and
  generic core semantic branches.
- Run `git diff --check`, `git diff --cached --check`, and Trellis validation
  before finish/archive.

## Implementation Summary

- Production invocation owner changed: `scripts/rvv_scalar_dispatch_e2e.py`.
- Added bounded vector/SCF source-frontdoor flags:
  `--lower-vector-i32-vadd-frontend` and
  `--lower-vector-i32-vsub-frontend`.
- Those flags default to the production dynamic vector source-frontdoor bundle
  fixtures:
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
  and
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`.
- In two-step mode, the runner now inserts
  `--tcrv-lower-source-rvv-binary-to-exec` before
  `--tcrv-execution-planning-pipeline`; in plan-and-export mode it passes the
  source fixture directly to
  `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle`.
- The runner validates generated dispatch source/header and bundle records for
  dynamic source-frontdoor authority:
  `source_frontend_runtime_avl_authority`,
  `dispatch_runtime_element_count_source`, selected bundle
  `tcrv_frontend.*` metadata, selected RVV source kind
  `frontend-lowering`, runtime element count `n`, and dispatch guard
  `rvv_available`.
- The bundle manifest authority check now requires selected RVV source identity
  metadata to be internally coherent with the active arithmetic family:
  selected source kind, microkernel op, and EmitC source op are checked before
  accepting generated dispatch artifacts. Existing non-vector typed-body bundle
  fixtures remain valid through the existing allowed source-kind contract.
- Focused lit coverage in
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test` now proves dynamic vector
  i32-vadd and i32-vsub source-frontdoor inputs reach generated RVV+scalar
  dispatch source/header/object bundle artifacts and generated external callers.
- No core pass, `tcrv.exec`, descriptor-to-C exporter, target-neutral
  orchestration, or C++ compute/lowering semantics were changed.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- C++ tests passed:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script syntax/self-test passed:
  `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local vector source-frontdoor dispatch bundle dry-runs passed:
  `i32-vadd` and `i32-vsub` with
  `--use-target-artifact-bundle --use-plan-and-export-bundle-front-door`,
  expected kernels `frontend_vector_dynamic_bundle_i32_vadd` and
  `frontend_vector_dynamic_bundle_i32_vsub`.
- Two-step vector source-frontdoor dispatch bundle dry-run passed for `i32-vsub`
  using `--tcrv-lower-source-rvv-binary-to-exec` before planning.
- Focused lit passed from `build/test`:
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`.
- Focused TargetArtifactBundleExport lit passed from `build/test`:
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
  and
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`.
- Focused RVVScalarDispatch lit passed from `build/test`:
  `Target/RVVScalarDispatch` (`14/14`).
- `ssh rvv` evidence passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --lower-vector-i32-vsub-frontend --arithmetic-family=i32-vsub --expect-selected-kernel=frontend_vector_dynamic_bundle_i32_vsub --run-id codex-vector-vsub-ssh --overwrite --timeout 120 --connect-timeout 10`.
  Artifact directory:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-vector-vsub-ssh`.
  Evidence summary: `ssh_evidence_verified = true`, host architecture
  `riscv64`, clang `/usr/bin/clang`, clang version
  `Ubuntu clang version 18.1.3 (1ubuntu1)`. Both source-built and
  bundle-object-built external caller runs printed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- `git diff --check` passed.

## Self-Repair Notes

- The first script validation change incorrectly required every historical
  bundle fixture to use selected source kind `frontend-lowering`. The existing
  generic route fixtures correctly use default typed-body materialization, so
  the check was narrowed: all dispatch bundle records must carry a supported
  selected source kind, while vector source-frontdoor evidence specifically
  requires `frontend-lowering` plus the dynamic `tcrv_frontend.*` authority.
- The first lit update checked `rvv_selected_plan_metadata` after
  `dispatch_runtime_callable_abi`, but the generated dispatch source prints
  selected-plan metadata before the ABI comment. The FileCheck order was
  corrected with DAG checks for the selected-plan metadata.
- Source-tree lit invocation failed because `test/lit.cfg.py` expects build
  config injection. The same focused tests were rerun from `build/test`, which
  is the repository's configured lit entry point.

## Spec Update Judgment

No `.trellis/spec/` update is needed. This round did not introduce a new
dialect op, schema field, plugin protocol, command contract, or architecture
rule. It extended an existing runner/evidence owner so the already-specified
source-frontdoor, selected-source-identity, selected-config, runtime-length,
dispatch-guard, and target-artifact-bundle contracts are consumed during
generated dispatch invocation.

## Finish Status

- Code and evidence criteria above are satisfied.
- Trellis finish/archive completed, and this commit records the coherent task
  closure.
