# Bounded MLIR vector to RVV selected route

## Goal

Make the bounded MLIR vector i32 add/sub front door a production-visible route
into the existing selected RVV binary pipeline. Current HEAD already carries
the dynamic vector vadd/vsub implementation through the generic
`--tcrv-lower-source-rvv-binary-to-exec` owner; this round closes the remaining
explicit VectorToExec vsub front-door gap, keeps the production route unified,
and verifies that real MLIR vector/SCF input reaches selected `tcrv.rvv` ops,
EmitC artifacts, runtime ABI plans, and RVV dispatch/runtime evidence.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean at task creation; HEAD was
  `d3285e3 feat(rvv): close op-family runtime abi dispatch parity`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
- The downstream selected RVV route already has selected source identity,
  selected EmitC mapping, runtime length/config validation, generated
  source/header/object/bundle export, and vadd/vsub runtime ABI dispatch
  parity from the archived 05-14 tasks.
- Current source frontend code already has a generic source owner
  `--tcrv-lower-source-rvv-binary-to-exec`, a fixed vector vadd adapter, and a
  dynamic vector/SCF i32 vadd/vsub adapter.
- The current explicit vector pass surface is still named only
  `--tcrv-lower-vector-rvv-i32-vadd-to-exec`; e2e vsub vector mode currently
  routes through the generic source pass rather than an explicit bounded
  VectorToExec vsub adapter.

## Requirements

- Add an explicit bounded vector i32-vsub pass/front-door that delegates to the
  same production source frontend owner and does not create a separate backend
  or descriptor-driven path.
- The vsub vector adapter must accept only the existing dynamic vector/SCF
  `memref<?xi32>, memref<?xi32>, memref<?xi32>, %n:index` shape with
  `arith.subi`, transfer-tail authority, selected target profile, and the
  shared runtime ABI boundary.
- The vsub vector adapter must reject stale add markers/bodies, legacy
  descriptor metadata, unsupported vector shapes, missing runtime length data,
  and transfer `in_bounds = [true]` tail claims before selected artifact output.
- Materialized selected RVV ops must continue to carry source identity,
  op-family identity, selected config, runtime AVL/VL roles, EmitC mapping,
  and descriptor-quarantine metadata through the existing selected route.
- Existing generic source pass behavior must remain the default production
  owner for bounded source front doors; the new vsub pass is an explicit
  bounded adapter entry point, not a forked implementation.
- Generic core orchestration must remain target-neutral. RVV-specific family,
  ABI, EmitC, dispatch, and artifact behavior stays in RVV/scalar
  plugin/target owners.

## Acceptance Criteria

- [x] `tcrv-opt` exposes `--tcrv-lower-vector-rvv-i32-vsub-to-exec` and it is
      implemented through the same C++ VectorToExec source lowering owner.
- [x] The existing vector dynamic i32-vsub fixture lowers through the explicit
      vsub pass, then reaches selected RVV `i32_vsub` materialization and
      generated source artifact output.
- [x] The vector vsub e2e harness uses the explicit vsub VectorToExec pass in
      its planning pipeline label and command, while preserving the generic
      source pass as the default multi-source owner.
- [x] Fail-closed vector tests cover vsub pass rejection of add-body or
      marker/body mismatch and stale transfer-tail/descriptor authority via
      existing negative cases.
- [x] Focused bundle/export evidence proves vector vsub reaches
      `RVVScalarDispatch` source/header/object bundle output with runtime
      length and selected config preserved.
- [x] Focused `ssh rvv` evidence is produced for generated vector vsub
      artifact invocation, or an exact blocker is recorded after local
      generation/compile succeeds.
- [x] Descriptor-only compute/config/runtime authority remains quarantined, and
      a bounded ref-scan shows no new RVV semantic branches in generic core
      orchestration.
- [x] Focused build, lit/FileCheck, script self-test/e2e, `git diff --check`,
      `git diff --cached --check`, Trellis validation before finish and after
      archive, final clean worktree, and one coherent commit complete the
      round.

## Out Of Scope

- No broad vector dialect lowering, linalg/tensor expansion, new dtype, i64
  vector frontend expansion, LMUL matrix expansion, third operation,
  performance tuning, or broad smoke matrix.
- No descriptor-to-C production exporter.
- No descriptor element count or descriptor vector shape as compute, selected
  config, runtime AVL/VL, ABI, or dispatch authority.
- No moving computation semantics into `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No GCC/vendor compiler as the default route.
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-op-family-runtime-abi-dispatch-parity/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-runtime-abi-production-invocation/prd.md`.
- Primary implementation surfaces:
  `include/TianChenRV/Transforms/Passes.td`,
  `include/TianChenRV/Transforms/Passes.h`,
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `scripts/rvv_microkernel_e2e.py`.
- Focused test surfaces:
  `test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir`,
  `test/Transforms/VectorToExec/vector-dynamic-i32-binary-invalid.mlir`,
  `test/Scripts/rvv-microkernel-bundle-e2e.test`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`.

## Evidence Plan

- Build touched transform/tool/test targets:
  `cmake --build build --target TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck for vector dynamic vadd/vsub lowering, invalid
  vector source cases, vector dynamic vsub bundle export, selected route
  quarantine, RVV microkernel bundle e2e, and RVV scalar dispatch/bundle
  regressions.
- Run `python3 scripts/rvv_microkernel_e2e.py --self-test` and focused dry-run
  vector vsub bundle evidence.
- Run focused `ssh rvv` vector vsub generated artifact invocation unless
  blocked by a precise remote/toolchain issue after local generation/compile.
- Run bounded ref-scans for descriptor-only authority, explicit vsub pass
  wiring, and generic core RVV/IME/Sophgo semantic branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation
  before finish and after archive.

## Definition Of Done

The bounded MLIR vector i32-vsub source path has a first-class explicit
VectorToExec front door that reuses the same production source owner as the
existing generic source pass. Vector vsub evidence reaches selected RVV
materialization, EmitC/source/header/object or bundle output, runtime ABI
metadata, and focused RVV execution or an exact blocker. The descriptor-only
path remains quarantined, the task is finished/archived, and one coherent
commit records the round.

## Implementation Summary

- Added explicit public pass/front-door
  `--tcrv-lower-vector-rvv-i32-vsub-to-exec` in `Passes.td`,
  `Passes.h`, `LowerSourceRVVBinaryToExec.cpp`, and `tcrv-opt`.
- Kept the implementation inside the existing source-frontdoor lowering owner:
  the new pass calls `lowerMarkedFrontendVectorInModule(..., VSubOnly)` and
  does not add a target-specific core branch, descriptor exporter, or separate
  backend path.
- Added vsub-only marker validation and dynamic marker/body cross-checking so
  `i32-vsub` markers with `arith.addi` bodies fail before an exec kernel is
  materialized.
- Preserved fixed vector vadd as a vadd-only adapter; the explicit vsub pass
  accepts only the dynamic vector/SCF runtime `%n` route.
- Updated `rvv_microkernel_e2e.py` so `--lower-vector-i32-vsub-frontend` uses
  the explicit vsub pass and reports the explicit pass in the planning label.
- Updated focused lit tests for explicit vector vsub lowering, selected RVV
  materialization, target source export, fail-closed marker/body mismatches,
  and bundle evidence.
- Updated the lowering-runtime spec so the dynamic vector source-tail
  authority scenario names both i32-vadd and i32-vsub explicit vector adapter
  front doors.

## Validation Summary

- Build:
  `cmake --build build --target TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- Rebuild after self-repair:
  `cmake --build build --target TianChenRVTransforms tcrv-opt tcrv-translate -j2`
- C++ tests:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`.
- Script self-test:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`.
- Focused lit from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-explicit-invalid.mlir Transforms/VectorToExec/vector-dynamic-i32-binary-invalid.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Scripts/rvv-microkernel-bundle-e2e.test`;
  result: 5 passed.
- Local explicit vector vsub dry-run:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --arithmetic-family=i32-vsub --lower-vector-i32-vsub-frontend --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count=7 --runtime-count=16 --runtime-count=23 --run-id codex-vector-vsub-explicit-dry --overwrite --timeout 120`;
  result: `status=success`, planned pipeline
  `tcrv-lower-vector-rvv-i32-vsub-to-exec + tcrv-execution-planning-pipeline`.
- RVV direct bundle ssh evidence:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --arithmetic-family=i32-vsub --lower-vector-i32-vsub-frontend --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count=7 --runtime-count=16 --runtime-count=23 --run-id codex-vector-vsub-explicit-ssh --overwrite --timeout 120 --connect-timeout 10`;
  result: `status=success`, `ssh_evidence=true`; both source-linked and
  bundle-object-linked runs printed
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.
  Artifact root:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-vector-vsub-explicit-ssh`.
- RVVScalarDispatch vector input bundle ssh evidence:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --input test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir --expect-selected-kernel=frontend_vector_dynamic_bundle_i32_vsub --run-id codex-vector-vsub-dispatch-ssh --overwrite --timeout 120 --connect-timeout 10`;
  result: `status=success`, `ssh_evidence=true`,
  `ssh_evidence_verified=true`; both source-linked and bundle-object-linked
  runs printed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
  Artifact root:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-vector-vsub-dispatch-ssh`.
- Ref-scan:
  `rg -n "descriptor.*authority|descriptor-to-C|descriptor_compute_authority|hasRVV|hasIME|hasSophgo|lower-vector-rvv-i32-vsub|VSubOnly" include/TianChenRV/Transforms lib/Transforms tools/tcrv-opt scripts/rvv_microkernel_e2e.py test/Transforms/VectorToExec test/Target/TargetArtifactBundleExport test/Scripts/rvv-microkernel-bundle-e2e.test .trellis/spec/lowering-runtime/emission-runtime-contract.md`;
  remaining descriptor hits are existing quarantine comments/checks; no
  `hasRVV`, `hasIME`, or `hasSophgo` semantic branch was added in generic
  transforms.
- Whitespace:
  `git diff --check` passed.
- Trellis validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-bounded-mlir-vector-rvv-selected-route`
  passed.

## Self-Repair

The first focused lit run failed because the new vsub-only pass checked the
frontend marker but did not run marker/body cross-checking in the dynamic
branch. A source with `tcrv_frontend_lowering = "i32-vsub"` and an `arith.addi`
body incorrectly lowered as vadd. The implementation now runs
`crossCheckVectorFrontendMarker` for both `ArithmeticFamily` and `VSubOnly`
dynamic vector modes, and the focused invalid test now covers this regression.
