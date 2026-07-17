# RVV source-frontdoor op-family artifact parity

## Goal

Make the bounded MLIR vector dynamic i32 add/sub source-frontdoor route use one
production op-family construction surface from source recognition through
selected RVV artifact/runtime evidence. Current HEAD already routes vadd/vsub
artifact evidence through `--tcrv-lower-source-rvv-binary-to-exec`; this round
closes the remaining transform-owner gap by making the dynamic vector family
admission and explicit compatibility aliases share a table-driven op-family
policy instead of vadd/vsub-specific source branches.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean at task creation; HEAD was
  `e23c4cf feat(vector): route rvv artifacts through source frontdoor`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
- The archived `05-14-rvv-vector-frontend-default-artifact-route` task already
  promoted dynamic vector vadd/vsub artifact and script evidence to
  `--tcrv-lower-source-rvv-binary-to-exec`.
- Current `LowerSourceRVVBinaryToExec.cpp` still exposes transform-local
  `VectorFrontendAdapterMode::{VAddOnly,VSubOnly,ArithmeticFamily}` branches,
  vadd/vsub-specific marker checks, and a dynamic vector arithmetic check that
  names vadd in the policy path.
- `RVVBinaryFamilyRegistry.h` already carries the finite RVV binary family
  table, including frontend marker, source arithmetic op, dynamic vector source
  kind, selected microkernel op, EmitC op, route IDs, runtime ABI names, and
  external ABI group.
- `FiniteBinaryFrontendLowering.h` already carries source-frontdoor runtime
  extent metadata and validates dynamic vector runtime AVL/source-tail
  authority for both `mlir-vector-scf-runtime-i32-vadd.v1` and
  `mlir-vector-scf-runtime-i32-vsub.v1`.
- Downstream RVV materialization/planning/export paths validate selected
  source identity, selected config, runtime length metadata, EmitC route
  metadata, descriptor quarantine, and generated source/header/object or
  bundle artifacts.

## Requirements

- Dynamic MLIR vector i32-vadd and i32-vsub source input must use the same
  source-frontdoor construction policy in the production
  `--tcrv-lower-source-rvv-binary-to-exec` path.
- The transform must derive accepted dynamic-vector families from the finite
  RVV binary family registry or an equivalent common policy surface, not from
  ad hoc vadd/vsub source branches.
- Explicit `--tcrv-lower-vector-rvv-i32-vadd-to-exec` and
  `--tcrv-lower-vector-rvv-i32-vsub-to-exec` aliases may remain compatibility
  entry points, but their restrictions must be expressed through the same
  op-family policy surface used by the default source frontdoor.
- Materialized `tcrv.exec.kernel` output must preserve source identity,
  op-family identity, selected target/profile reference, source runtime `%n`
  roles, and finite runtime ABI mem_window/runtime_param roles for both ops.
- Unsupported vector patterns, stale marker/body identity, stale dynamic
  source-kind policy, missing runtime length/source-tail metadata, explicit
  alias misuse, and descriptor-only frontend attempts must fail closed before
  selected artifact output.
- Generic core orchestration must remain neutral: no RVV/IME/offload/scalar
  semantic branches in shared core passes, and no compute semantics in
  `tcrv.exec`.

## Acceptance Criteria

- [x] `LowerSourceRVVBinaryToExec.cpp` has a shared dynamic vector
      op-family policy that derives accepted families from
      `RVVBinaryFamilyRegistry` records with dynamic vector source-kind
      support.
- [x] Default `--tcrv-lower-source-rvv-binary-to-exec` lit coverage proves
      dynamic vector i32-vadd and i32-vsub lower through the same policy to
      source-derived family metadata and runtime AVL/source-tail authority.
- [x] Explicit vadd/vsub compatibility aliases continue to work, but misuse
      of a marker/body/family outside the alias policy fails before exec
      materialization.
- [x] TargetArtifactBundleExport and script evidence continue to show both
      dynamic vector source-frontdoor ops reaching generated source/header/
      object or bundle artifacts with selected config and runtime length data
      preserved.
- [x] Descriptor-only compute/config/runtime authority remains quarantined.
- [x] Focused build, C++ tests, lit/FileCheck tests, e2e script self-tests or
      focused dry-runs, bounded ref-scans, `git diff --check`,
      `git diff --cached --check`, Trellis validation, archive, and one
      coherent commit complete the round.

## Out Of Scope

- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 vector frontend expansion, LMUL matrix expansion, third operation in the
  dynamic vector source-frontdoor path, broad family matrix, broad smoke suite,
  or performance tuning.
- No additional explicit-only frontend flag as the main milestone.
- No vsub-only helper or test-only wrapper as the main route.
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
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-vector-frontend-default-artifact-route/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-bounded-mlir-vector-rvv-selected-route/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Primary implementation surfaces:
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `include/TianChenRV/Transforms/Passes.td`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  focused tests under `test/Transforms/VectorToExec`,
  `test/Target/TargetArtifactBundleExport`, and `test/Scripts`.

## Evidence Plan

- Build touched transform/plugin/target/tool/test targets:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck for vector dynamic vadd/vsub default source
  lowering, explicit alias compatibility/misuse, invalid vector source cases,
  target artifact bundle export, RVV microkernel bundle e2e, and RVV scalar
  dispatch bundle e2e.
- Run `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` if script behavior
  or artifact evidence assumptions change.
- Run focused local dry-runs for default vector vadd/vsub bundle evidence; run
  focused `ssh rvv` generated artifact invocation if generated source/object
  semantics change or if local generation needs fresh hardware evidence.
- Run bounded ref-scans for explicit-only vector routing, descriptor-only
  authority, stale vadd-only policy, and generic core RVV/IME/Sophgo semantic
  branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation
  before finish and after archive.

## Definition Of Done

The production source-frontdoor dynamic vector route for i32-vadd and
i32-vsub is governed by one op-family policy surface, the explicit vadd/vsub
aliases are compatibility filters on that same policy, downstream generated
artifact/runtime consumers still receive selected family/config/runtime-length
metadata for both ops, descriptor-only authority remains quarantined, and the
task is finished, archived, and committed.

## Implementation Summary

- Replaced the transform-local
  `VectorFrontendAdapterMode::{VAddOnly,VSubOnly,ArithmeticFamily}` split with
  a `VectorFrontendFamilyPolicy` surface in
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`.
- The default vector source-frontdoor policy now derives accepted dynamic
  vector markers/operators from `RVVBinaryFamilyRegistry` records whose
  `dynamicVectorSourceKind` is present, currently `i32-vadd` and `i32-vsub`.
- The explicit vadd/vsub pass names remain compatibility aliases, but their
  marker restrictions are now filters over the same policy surface rather than
  separate vadd/vsub marker-check functions.
- The dynamic vector body recognizer now names the accepted source arithmetic
  operations from the registry-backed dynamic family set rather than from a
  hard-coded vadd branch.
- Added fail-closed `i32-vmul` dynamic vector coverage to prove the default
  source-frontdoor add/sub route does not silently become a third-operation
  vector backend.
- Updated `Passes.td` to document that dynamic vector vadd/vsub construction is
  registry-policy-backed while explicit vector pass names are compatibility
  entry points.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVTransforms tcrv-opt -j2`.
- Focused build passed:
  `cmake --build build --target TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- C++ tests passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`.
- Focused VectorToExec lit passed:
  `vector-dynamic-i32-vadd-to-exec.mlir`,
  `vector-dynamic-i32-vsub-to-exec.mlir`,
  `vector-dynamic-i32-vsub-explicit-invalid.mlir`,
  `vector-dynamic-i32-binary-invalid.mlir`,
  `vector-i32-vadd-invalid.mlir`.
- Focused artifact/script lit passed:
  `plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  `rvv-microkernel-bundle-e2e.test`,
  `rvv-scalar-dispatch-bundle-e2e.test`.
- Script self-tests passed:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`,
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local default source-frontdoor dry-runs passed:
  `codex-family-policy-vector-vadd-dry`,
  `codex-family-policy-vector-vsub-dry`,
  `codex-family-policy-vector-vsub-dispatch-dry`.
- Focused `ssh rvv` evidence passed:
  `codex-family-policy-vector-vadd-ssh` and
  `codex-family-policy-vector-vsub-ssh`, both status `success`,
  `ssh_evidence=true`, runtime counts `7,16,23`.
  Artifact roots:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-family-policy-vector-vadd-ssh`
  and
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-family-policy-vector-vsub-ssh`.
  The generated callers printed
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23` for vadd and
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23` for vsub on
  the bounded source-linked and bundle-object-linked invocation paths.
- Bounded ref-scans found no remaining `VectorFrontendAdapterMode` or
  `VAddOnly`/`VSubOnly` policy branches, no new `hasRVV`/`hasIME`/`hasSophgo`
  generic transform branch, and only expected descriptor-quarantine text in
  focused vector tests.
- `git diff --check` passed.

## Self-Repair

The first focused lit attempt was started before `tcrv-translate` finished
relinking, so two tests failed with a temporary permission error for
`build/bin/tcrv-translate`. After the focused build completed, the identical
lit command passed. No source-code repair was needed for that failure.

## Spec Update Judgment

No `.trellis/spec` update is needed for this round. The task changed the
implementation shape of an existing bounded vector source-frontdoor policy and
added focused fail-closed coverage; it did not add a new dialect contract,
payload schema, public command, artifact field, runtime ABI, or cross-layer
architecture rule beyond the specs already cited above.
