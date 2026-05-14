# RVV vmul op-family artifact/runtime closure

## Goal

Finish one coherent i32-vmul artifact/runtime closure slice that proves the
shared RVV binary op-family contract is consumed beyond source-frontdoor
admission. This round must carry the selected family identity through
plugin/target materialization, generated artifact export, runtime ABI
invocation contracts, `RVVScalarDispatch`, local generated-artifact execution,
and focused `ssh rvv` evidence when the generated route is complete.

## Current Repository Facts

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd` confirmed the repository root.
- Initial `git status --short` was clean.
- Initial HEAD was `e3c3877 feat(rvv): admit vmul through binary op-family contract`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief as
  `.trellis/tasks/05-14-rvv-vmul-op-family-artifact-runtime-closure`.
- The archived binary op-family task records that e3c3877 admitted i32-vmul
  through the finite RVV binary source-frontdoor contract and added focused
  lowering and bundle/e2e coverage.
- Current code has a shared `RVVBinaryFamilyDescriptor` registry in
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`, including
  i32-vadd, i32-vsub, i32-vmul, i64-vadd, i64-vsub, and i64-vmul.
- Current plugin materialization/emission planning already resolves selected
  family/config data through `RVVBinarySelectedPlan` and
  `RVVBinarySelectedConfigContract`.
- Current RVV target/export/dispatch owners already derive vmul route IDs,
  intrinsic names, runtime ABI names, callable glue roles, artifact stems, and
  dispatch registrations from the shared family descriptors.
- The main bounded gap found before source edits is runtime invocation evidence:
  `RuntimeABIInvocationContract` receives a family ID but does not store or
  print that family field. Generated RVV microkernel and scalar dispatch
  evidence therefore proves callable/ABI names, but not an explicit selected
  op-family contract field.

## Requirements

- Preserve the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck compiler
  stack. Python may only remain a runner, parser, probe, or evidence consumer.
- Keep computation semantics out of `tcrv.exec`.
- Do not add descriptor-driven computation, descriptor-to-C production export,
  descriptor element-count authority, vector-shape runtime authority, or Python
  compiler semantics.
- Do not add RVV semantic special cases in generic core orchestration.
- Keep vmul routed by the shared RVV binary family contract, not a one-off
  vmul branch outside plugin/target family ownership.
- vadd and vsub must remain green through the same generated artifact and
  dispatch evidence surfaces touched by vmul.
- Unsupported, stale, or partially declared family state must fail before
  artifact/runtime claims.

## Acceptance Criteria

- [x] `RuntimeABIInvocationContract` stores the selected finite binary family
      ID as a required bounded field and prints it into generated invocation
      contract comments.
- [x] RVVMicrokernel and RVVScalarDispatch generated artifacts expose the same
      explicit family field for i32-vmul that vadd/vsub use.
- [x] Runtime ABI and dispatch script validators require the printed family
      field to match the selected arithmetic family for generated-artifact
      evidence.
- [x] Focused C++ and lit/FileCheck coverage proves i32-vmul reaches selected
      boundary, materialization/emission mapping, artifact export, runtime ABI
      invocation, and dispatch validation under the shared contract.
- [x] vadd/vsub generated-artifact and dispatch regression checks remain green.
- [x] Local generated-artifact dry-run evidence passes for i32-vmul and focused
      vadd/vsub regressions; runtime execution evidence is supplied by focused
      `ssh rvv` because the generated artifacts target RVV.
- [x] Focused `ssh rvv` compile/run evidence is run for generated i32-vmul if
      the local generated artifact path is complete; otherwise the exact
      remaining production owner blocker is recorded.
- [x] Fail-closed checks cover missing family ID, unknown/stale family identity,
      missing selected config or runtime length role data, stale ABI signature,
      descriptor-only production attempts, explicit-only route misuse, and
      source-identity mismatch where those cases intersect the touched surface.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      finish/archive, one coherent commit, and final worktree cleanliness are
      completed if the task finishes.

## Out Of Scope

- No broad dtype, LMUL, mask/tail, linalg/tensor frontend, performance tuning,
  or arbitrary additional op-family expansion.
- No helper-only, registry-only, manifest-only, documentation-only,
  metadata-only, PRD-only, journal-only, test-only, script-only, smoke-only,
  report-only, artifact-text-only, or evidence-packaging-only milestone.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.
- No descriptor-to-C production exporter.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core passes.
- No replacing clang/LLVM as the default native compiler route.
- No runtime, correctness, or performance claim without focused local or
  `ssh rvv` evidence.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run script self-tests for generated RVV microkernel and scalar dispatch
  evidence parsers.
- Run focused lit/FileCheck for RuntimeABI callable plan, RVVMicrokernel,
  RVVScalarDispatch, TargetArtifactExport, TargetArtifactBundleExport, and the
  scalar-dispatch bundle script e2e tests touched by the family field.
- Run `scripts/rvv_scalar_dispatch_e2e.py` generated-artifact evidence for
  i32-vadd, i32-vsub, and i32-vmul, with local vmul evidence before ssh.
- Run focused `ssh rvv` i32-vmul generated bundle evidence if local generation
  and invocation are complete.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale source identity, and generic core RVV semantic branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation, and
  final worktree cleanliness checks.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-binary-op-family-contract/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-14-rvv-ssh-rvv-generated-artifact-execution-closure/prd.md`.
- Production/code surfaces inspected before PRD:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`,
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h`,
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`, and focused tests/scripts found by
  `rg`.

## Implementation Summary

- Added `familyID` to `support::RuntimeABIInvocationContract`.
- Made `buildRuntimeABIInvocationContract` require non-empty bounded family
  text and made `formatRuntimeABIInvocationContractCommentBody` print
  `family=<selected-family>` next to the callable symbol.
- Kept the production change in the shared Support runtime ABI contract; no
  vmul-only production branch was added in core orchestration.
- Updated `scripts/rvv_microkernel_e2e.py` so generated RVV source evidence
  requires the runtime invocation family field to match the selected arithmetic
  family and returns that field in evidence JSON.
- Updated `scripts/rvv_scalar_dispatch_e2e.py` so embedded RVV artifacts and
  dispatcher wrappers both require the family field in runtime ABI invocation
  contracts.
- Added self-test fail-closed coverage for stale runtime invocation family
  fields.
- Added C++ coverage for direct i32-vsub, direct i32-vmul, dispatch i32-vsub,
  and missing-family invocation contracts in
  `test/Support/RuntimeABICallablePlanTest.cpp`.
- Updated RVV microkernel, scalar-dispatch bundle, and target artifact bundle
  FileCheck coverage so vadd/vsub/vmul generated evidence checks explicit
  family fields in runtime ABI invocation contracts.

## Validation Summary

- `clang-format`, `clang-format-20`, `clang-format-19`, and
  `clang-format-18` were not available in PATH; changed C++ was manually kept
  consistent with neighboring code.
- Focused build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script self-tests passed:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Focused lit passed from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-e2e|rvv-scalar-dispatch-bundle-e2e|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vmul-and-export-target-artifact-bundle|rvv-microkernel-runtime-abi-role-binding|rvv-microkernel-family-sub|rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding|rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding'`;
  result: 10 selected tests passed.
- Local generated-artifact dry-run evidence passed for vadd/vsub/vmul:
  `codex-vmulclosure-vadd-local`,
  `codex-vmulclosure-vsub-local`, and
  `codex-vmulclosure-vmul-local`. The vmul dry-run selected
  `frontend_vector_dynamic_bundle_i32_vmul`, emitted source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`,
  and the generated evidence JSON contains `family: i32-vmul` in both
  dispatch and embedded RVV runtime ABI invocation contracts.
- Focused `ssh rvv` compile/run evidence passed for generated i32-vmul:
  `codex-vmulclosure-vmul-ssh-rvv`, mode `ssh`,
  `ssh_evidence_verified = true`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vmul`, source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- Bounded ref-scan for `descriptor-to-C`, `descriptor-only`,
  `descriptor_element_count.*runtime`, and `explicit-only` across changed
  files found only existing fail-closed test strings and source-runtime
  authority comments; changed Support production code contains only family ID
  propagation and no descriptor or explicit-route authority.
- `git diff --check` passed.
- Trellis validation passed before finish and after archive.
- `git diff --cached --check` passed.
