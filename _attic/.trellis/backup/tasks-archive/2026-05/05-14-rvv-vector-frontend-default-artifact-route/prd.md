# RVV vector frontend default artifact route

## Goal

Promote the bounded MLIR vector i32 add/sub frontend from explicit vector
adapter surfaces into the production/default selected RVV op-family route. The
normal source frontdoor and plan-and-export bundle route must carry dynamic
vector `i32-vadd` and `i32-vsub` source input through source-derived
op-family identity, selected `tcrv_rvv` materialization, selected config,
runtime AVL/VL metadata, EmitC body mapping, generated source/header/object or
bundle artifacts, and existing RuntimeABICallablePlan/RVVScalarDispatch
consumers.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean at task creation; HEAD was
  `d5dc6d7 feat(vector): expose explicit rvv vsub frontend`.
- `.trellis/.current-task` did not exist, so this task was created from the
  Hermes direction brief and started as the current task.
- Current code exposes the production source owner
  `--tcrv-lower-source-rvv-binary-to-exec`, plus explicit vector aliases
  `--tcrv-lower-vector-rvv-i32-vadd-to-exec` and
  `--tcrv-lower-vector-rvv-i32-vsub-to-exec`.
- `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` already runs
  `createLowerSourceRVVBinaryToExecPass()` before the execution-planning
  pipeline, so it is the correct default artifact-route owner for source
  fixtures.
- `LowerSourceRVVBinaryToExec.cpp` currently infers dynamic vector i32 family
  identity from `arith.addi` / `arith.subi`, cross-checks
  `tcrv_frontend_lowering`, creates the shared exec ABI boundary, and records
  dynamic source `%n` / transfer-tail authority.
- `RVVBinaryFamilyRegistry.h` already carries shared finite family records for
  `i32-vadd` and `i32-vsub`, including frontend marker, source arithmetic op,
  dynamic vector source kind, typed microkernel op, EmitC source op, route IDs,
  runtime ABI names, and external ABI group.
- Downstream RVV materialization/planning/export code already validates typed
  selected source identity, selected vector config, runtime ABI mirrors,
  runtime AVL/VL metadata, EmitC body mapping, descriptor quarantine, and
  bundle/source/header/object artifacts.
- The previous completed slice proved explicit vsub vector frontend evidence,
  but tests and e2e commands still make explicit vector aliases the primary
  visible route for some vector artifact evidence.

## Requirements

- The default or normal tool route must lower bounded dynamic MLIR vector
  i32-vadd and i32-vsub source input into selected RVV op-family state without
  requiring a vsub-only explicit pass.
- Vadd and vsub vector source inputs must share the production
  source-frontdoor family construction/mapping surface rather than separate
  ad hoc branches for artifact evidence.
- The route must preserve source identity, op-family identity, selected vector
  config, runtime element-count/AVL/VL authority, EmitC body mapping, and
  RuntimeABI metadata into generated source/header/object or bundle artifacts.
- `RuntimeABICallablePlan`, direct RVV microkernel export, target artifact
  bundle export, and RVVScalarDispatch must continue to consume the existing
  selected route rather than descriptor-only or test-only metadata.
- Explicit vector vadd/vsub aliases may remain compatibility/focused adapter
  entry points, but they must not be the only evidence that source vector input
  reaches generated artifacts.
- Unsupported vector patterns, stale op identity, stale EmitC mapping, missing
  selected config, missing runtime length role data, missing source-tail
  metadata, and descriptor-only production attempts must fail closed before
  artifact output.
- Generic core orchestration must remain neutral: no RVV/IME/offload/scalar
  semantic branches in shared core passes, and no compute semantics in
  `tcrv.exec`.

## Acceptance Criteria

- [x] Default `--tcrv-lower-source-rvv-binary-to-exec` pipeline coverage proves
      dynamic vector i32-vadd and i32-vsub lower to selected `tcrv_rvv`
      materialization with typed family identity and runtime AVL/VL metadata.
- [x] Plan-and-export target artifact bundle coverage proves vector vadd and
      vector vsub source fixtures reach generated source/header/object bundle
      records through the default front door, not only through explicit vector
      alias passes.
- [x] RVV microkernel e2e bundle evidence can use a default vector frontend
      lowering mode that invokes the production source frontdoor and records
      that command/label in evidence for both vadd and vsub.
- [x] RVVScalarDispatch bundle evidence for vector vsub remains green through
      the plan-and-export front door and reports the existing source/header/
      object external ABI bundle with selected config and runtime length data.
- [x] Focused fail-closed tests cover stale family/source identity, missing
      runtime length/source-tail metadata, stale EmitC body mapping, and
      descriptor-only production attempts on the default vector route.
- [x] Existing explicit vadd/vsub vector adapter tests remain as compatibility
      coverage without becoming the sole artifact evidence route.
- [x] Descriptor-only compute/config/runtime authority remains quarantined; any
      descriptor metadata is accepted only as a bounded mirror after typed
      selected authority exists.
- [x] Focused build, C++ tests, lit/FileCheck tests, e2e self-tests/dry-runs,
      `ssh rvv` evidence or exact blocker, ref-scans, `git diff --check`,
      `git diff --cached --check`, Trellis validation, archive, and one coherent
      commit complete the round.

## Out Of Scope

- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 vector frontend expansion, LMUL matrix expansion, third operation,
  broad family matrix, broad smoke suite, or performance tuning.
- No additional explicit-only frontend flag as the main milestone.
- No vsub-only helper as the main route.
- No descriptor-to-C production exporter.
- No descriptor element count or descriptor vector shape as compute, selected
  config, runtime AVL/VL, ABI, or dispatch authority.
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
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-14-bounded-mlir-vector-rvv-selected-route/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-op-family-runtime-abi-dispatch-parity/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Primary implementation surfaces:
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `include/TianChenRV/Transforms/Passes.td`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `scripts/rvv_microkernel_e2e.py`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  and targeted lit fixtures under `test/Transforms/VectorToExec`,
  `test/Target/TargetArtifactBundleExport`, `test/Target/RVVMicrokernel`,
  `test/Target/RVVScalarDispatch`, and `test/Scripts`.
- The preferred production path for this round is:
  dynamic vector source fixture ->
  `--tcrv-lower-source-rvv-binary-to-exec` or
  `--tcrv-plan-and-export-target-artifact-bundle` ->
  execution planning ->
  selected `tcrv_rvv` family op/body ->
  EmitC route ->
  generated source/header/object or dispatch bundle.

## Evidence Plan

- Build touched transform/tool/RVV/export/support targets:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`, and any touched
  RuntimeABI/export tests.
- Run focused lit/FileCheck tests for vector vadd/vsub default source lowering,
  explicit adapter compatibility, plan-and-export bundle output, direct RVV
  microkernel bundle output, RVVScalarDispatch bundle output, and negative
  fail-closed mutations.
- Run `python3 scripts/rvv_microkernel_e2e.py --self-test` and focused dry-run
  default vector vadd/vsub bundle evidence.
- Run `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` if dispatch
  bridge behavior is touched.
- Run focused `ssh rvv` generated artifact invocation for the changed default
  vector route unless blocked by a precise remote/toolchain issue after local
  generation succeeds.
- Run bounded ref-scans for explicit-only vector routes, descriptor-only
  authority, stale vadd-only evidence, and generic core RVV/IME/Sophgo semantic
  branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation
  before finish and after archive.

## Implementation Summary

- Rewired the dynamic vector path in
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp` so both dynamic
  `i32-vadd` and `i32-vsub` use
  `makeDynamicVectorI32SourceFrontendLoweringContract(*source.contract)` from
  the source-derived finite family contract instead of keeping a vadd-specific
  construction branch.
- Updated `include/TianChenRV/Transforms/Passes.td` descriptions to make
  `--tcrv-lower-source-rvv-binary-to-exec` the documented default source
  frontdoor and to describe the explicit vector vadd/vsub passes as
  compatibility adapters.
- Changed `scripts/rvv_microkernel_e2e.py` so vector vadd/vsub frontend modes
  run `--tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline`
  and record that default production pipeline label in evidence.
- Changed `scripts/rvv_scalar_dispatch_e2e.py` plan-and-export evidence naming
  from a linalg-specific fixture-free path to
  `source-rvv-binary-frontend-to-plan-and-export-bundle`.
- Updated focused bundle/script tests so dynamic vector vadd artifact coverage
  and e2e vector vsub bundle coverage prove the default source frontdoor route,
  while the explicit vsub test remains as compatibility coverage.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- C++/script self-tests passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`,
  `python3 scripts/rvv_microkernel_e2e.py --self-test`, and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Focused lit/FileCheck passed for:
  `Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir`,
  `Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir`,
  `Transforms/VectorToExec/vector-dynamic-i32-vsub-explicit-invalid.mlir`,
  `Transforms/VectorToExec/vector-dynamic-i32-binary-invalid.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  `Scripts/rvv-microkernel-bundle-e2e.test`, and
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`.
- Dry-run evidence passed for default vector vadd and vsub direct microkernel
  bundles, plan-and-export bundles, and vsub RVVScalarDispatch bundle evidence.
- Real `ssh rvv` evidence passed for:
  `codex-default-vector-vsub-ssh`,
  `codex-default-vector-vadd-ssh`, and
  `codex-default-vector-vsub-dispatch-ssh`.
  Each run reported `status: success` and `ssh_evidence: true` for the focused
  generated artifact invocation. The claim scope is limited to bounded RVV
  i32-vadd/i32-vsub target-artifact bundle or dispatch correctness for the
  generated route actually run.
- Ref-scans found no new descriptor production authority and no new generic
  core family branch. The only remaining explicit vector pass reference in the
  scanned artifact set is the intentional vsub compatibility RUN line.
- `git diff --check` passed.

## Spec Update Judgment

No `.trellis/spec` update is needed for this round. The implementation did not
add a new command-line signature, payload field, dialect contract, or cross-layer
schema. It rewired existing vector frontend evidence and script modes to the
already-specified default source RVV binary frontdoor, selected-boundary,
EmitC-route, descriptor-quarantine, and runtime-ABI contracts covered by the
spec files listed above.

## Definition Of Done

The normal/default source vector route, not only explicit vector aliases,
proves bounded dynamic i32-vadd and i32-vsub source input can reach selected
RVV family materialization, generated source/header/object or bundle artifacts,
and existing runtime ABI/dispatch consumers with selected config and runtime
length authority preserved. The task is finished, archived, validated, and
committed with the final worktree clean.
