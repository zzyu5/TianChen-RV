# RVV plugin-owned binary op-family construction contract

## Goal

Turn the already proven source-frontdoor i32 vadd/vsub RVV path into a
reusable plugin/target-owned binary op-family construction contract. This round
must make the selected source-frontdoor lowering, RVV materialization/emission
planning, artifact export, `RuntimeABICallablePlan`, and `RVVScalarDispatch`
consume one shared family contract rather than scattered finite-family
branches. The bounded reuse proof is i32-vmul if current repository support is
complete enough; otherwise the round must centralize vadd/vsub and record the
exact missing owner for vmul.

## Current Repository Facts

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd` confirmed the repository root.
- Initial `git status --short` was clean.
- Initial HEAD was `8f879f3 chore(rvv): archive ssh generated artifact evidence`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief as `.trellis/tasks/05-14-rvv-binary-op-family-contract`.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-14-rvv-ssh-rvv-generated-artifact-execution-closure/prd.md`
  closed generated-artifact local evidence for vadd/vsub and ssh-rvv evidence
  for generated i32-vsub.
- Current code already has a target-side `RVVBinaryFamilyDescriptor` registry
  in `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`.
- Current code already registers i32-vadd, i32-vsub, i32-vmul, i64-vadd,
  i64-vsub, and i64-vmul family descriptors for target/plugin/export/dispatch
  use.
- Current RVV plugin materialization consumes the selected family/config
  contract through `RVVBinarySelectedPlan` and
  `RVVBinarySelectedConfigContract`.
- Current target/export/dispatch support already contains i32-vmul routes and
  ABI metadata.
- The main bounded gap found before implementation is at the vector
  source-frontdoor family-admission contract: i32-vmul has no dynamic vector
  source-kind adapter, while vadd/vsub do.

## Requirements

- Preserve the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck compiler
  stack. Python may only remain a runner, parser, probe, or evidence consumer.
- Keep computation semantics out of `tcrv.exec`.
- Do not add descriptor-driven computation, descriptor-to-C production export,
  descriptor element-count authority, vector-shape runtime authority, or Python
  compiler semantics.
- Do not add RVV semantic special cases in generic core orchestration.
- The op-family authority for bounded RVV binary routes must live in
  plugin/target-owned family descriptors and selected contracts, not in one-off
  source/export/dispatch branches.
- vadd and vsub must continue to route through the same shared family contract.
- i32-vmul should enter through the same selected source identity, op-family
  identity, selected config, ABI signature, runtime length, EmitC/intrinsic
  mapping, artifact naming, and dispatch validation path if repository support
  is complete enough.
- Unsupported or partially declared families must fail closed before artifact,
  runtime, or hardware claims.
- Existing explicit direct typed microkernel routes must remain bounded and
  must not become descriptor authority.
- This task may update tests that encode the old vadd/vsub-only source-frontdoor
  contract. Obsolete negative expectations should be replaced with contract
  reuse coverage rather than preserved as legacy behavior.

## Acceptance Criteria

- [x] The vector source-frontdoor dynamic binary route derives admitted
      families and source-kind metadata from the shared RVV binary family
      registry/contract.
- [x] i32-vadd and i32-vsub still lower from the selected source-frontdoor into
      the normal exec ABI boundary, selected RVV/scalar plans, artifact bundle,
      and runtime dispatch route.
- [x] i32-vmul either routes through that same source-frontdoor artifact/runtime
      path, or the exact missing production owner is recorded and vadd/vsub are
      still centralized.
- [x] i32-vmul does not require descriptor element-count authority,
      descriptor-to-C production export, explicit-only route spoofing, or a
      detached Python semantic model.
- [x] Fail-closed checks cover unknown family markers, stale marker/body
      family mismatch, missing dynamic source-kind adapter, missing selected
      config, stale ABI/runtime role metadata, descriptor-only production
      attempts, explicit-only misuse, and source-identity mismatch where those
      checks are in the touched behavior surface.
- [x] Focused lit/FileCheck or C++ tests prove vadd/vsub continue through the
      contract and i32-vmul either reaches the same selected path or fails at a
      declared missing boundary.
- [x] Focused local generated-artifact/runtime evidence passes for migrated
      vadd/vsub and, if implemented, i32-vmul.
- [x] Focused `ssh rvv` compile/run evidence is run for i32-vmul if its
      generated intrinsic artifact path is complete in this round; otherwise no
      RVV hardware claim is made for i32-vmul.
- [x] Focused build/test commands for touched transform/plugin/target/export
      support pass, plus `git diff --check` and `git diff --cached --check`.
- [x] Trellis task status, implementation notes, validation notes, archive, and
      one coherent commit are completed if the task is finished.

## Out Of Scope

- No new broad vector dialect, linalg/tensor frontend, dtype matrix, LMUL
  matrix, mask/tail matrix, performance tuning, or arbitrary op-family
  expansion.
- No helper-only, manifest-only, documentation-only, metadata-only, PRD-only,
  journal-only, test-only, negative-test-only, script-only, smoke-only,
  report-only, or evidence-packaging-only milestone.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.
- No descriptor-to-C production exporter.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core passes.
- No replacing clang/LLVM as the default native compiler route.
- No runtime, correctness, or performance claim without focused evidence.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run focused lit for source-frontdoor vector/linalg binary lowering,
  RVVMicrokernel, RVVScalarDispatch, TargetArtifactExport, and target artifact
  bundle export tests touched by i32-vmul source-frontdoor admission.
- Run `scripts/rvv_scalar_dispatch_e2e.py` local generated-artifact dry-run or
  compile/run mode for vadd/vsub and i32-vmul if the script supports those
  source-frontdoor fixtures.
- Run focused `ssh rvv` i32-vmul generated bundle evidence only after local
  generated source/header/object/bundle evidence is complete.
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
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-ssh-rvv-generated-artifact-execution-closure/prd.md`.
- Production/code surfaces inspected before PRD:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Target/TargetArtifactExport.cpp`, and focused tests/scripts found by
  `rg`.

## Implementation Summary

- Added bounded i32-vmul dynamic vector source-frontdoor admission by wiring
  `support::kFrontendDynamicVectorI32VMulSourceKind` into the RVV binary family
  registry's i32-vmul descriptor.
- Kept the source-frontdoor construction path generic over registry-admitted
  RVV binary family records; no new descriptor-to-C path, Python compiler
  semantics, or generic core RVV semantic branch was added.
- Updated dynamic vector source diagnostics and pass documentation to describe
  registry-admitted i32 binary arithmetic instead of a vadd/vsub-only proof
  slice.
- Extended `scripts/rvv_scalar_dispatch_e2e.py` runner support so the
  production source-frontdoor bundle evidence can select i32-vmul through the
  same plan-and-export target artifact bundle path.
- Added positive lit coverage for vector/SCF i32-vmul lowering and
  plan-and-export dispatch bundle export, including selected source identity,
  source-tail runtime AVL authority, EmitC intrinsic mapping, dispatch ABI
  contract, and descriptor-quarantine route claims.
- Replaced the obsolete "i32-vmul is not admitted" negative vector test with an
  unknown marker fail-closed case, while retaining marker/body mismatch coverage.
- Updated the lowering-runtime spec so the dynamic vector source-tail authority
  contract explicitly includes registry-admitted i32-vmul.

## Validation Summary

- Focused build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vmul-to-exec|vector-dynamic-i32-binary-invalid|vector-dynamic-i32-vadd-to-exec|vector-dynamic-i32-vsub-to-exec|plan-vector-dynamic-i32-vmul-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle'`
  from `build/test`; result: 7 selected tests passed.
- Script bundle lit passed:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-dispatch-bundle-e2e'`
  from `build/test`; result: 1 selected test passed.
- Script self-test passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local source-frontdoor bundle dry-run evidence passed for:
  `codex-opfamily-vadd-vector-local`,
  `codex-opfamily-vsub-vector-local`, and
  `codex-opfamily-vmul-vector-local`.
- Focused ssh-rvv evidence passed for i32-vmul source-frontdoor bundle:
  `codex-opfamily-vmul-vector-ssh-rvv`, mode `ssh`,
  `ssh_evidence_verified = true`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vmul`, source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- `clang-format` was not available in PATH as `clang-format`,
  `clang-format-18`, `clang-format-19`, or `clang-format-20`; formatting was
  manually kept consistent with surrounding code.
