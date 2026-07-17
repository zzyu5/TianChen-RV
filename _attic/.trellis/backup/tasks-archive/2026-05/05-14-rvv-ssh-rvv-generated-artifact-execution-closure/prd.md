# RVV ssh-rvv generated artifact execution closure

## Goal

Close the current-head hardware execution evidence for the existing generated
source-frontdoor i32 vadd/vsub RVV dispatch artifact path. The bounded route is:
MLIR vector/SCF or linalg source-frontdoor input -> selected `tcrv.exec` ABI
boundary -> materialized `tcrv_rvv` and scalar callable artifacts with source
identity, op-family identity, selected config, ABI signature, and runtime AVL/VL
roles -> `RVVMicrokernel` / `TargetArtifactExport` generated bundle ->
centralized `RuntimeABICallablePlan` / `RVVScalarDispatch` callable invocation
contract -> local generated-artifact invocation and at least one focused
`ssh rvv` clang compile/link/run leg.

This task must not invent another script-only proof layer. If current HEAD
already has the production RuntimeABI/RVVScalarDispatch contract, this round
should re-prove it with fresh generated artifacts and hardware evidence. If a
real blocker appears, fix only the narrow production owner that blocks the
existing route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd` confirmed the repository root.
- Initial `git status --short` was clean.
- Initial HEAD was `31f9955 feat(rvv): prove generated artifact dispatch invocation`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief as
  `.trellis/tasks/05-14-rvv-ssh-rvv-generated-artifact-execution-closure`.
- The adjacent archived RuntimeABI tasks record that current production C++
  already has `RuntimeABIInvocationContract` support in
  `RuntimeABICallablePlan`, and `RVVMicrokernel.cpp` /
  `RVVScalarDispatch.cpp` emit/consume it.
- Current code inspection confirms `RuntimeABIInvocationContract`,
  `buildRuntimeABIInvocationContract`, and
  `formatRuntimeABIInvocationContractCommentBody` exist in support code and are
  used from both `RVVMicrokernel.cpp` and `RVVScalarDispatch.cpp`.
- Current `scripts/rvv_scalar_dispatch_e2e.py` parses
  `embedded_rvv_artifact_contract`,
  `dispatch_runtime_abi_invocation_contract`, bundle index ABI signatures,
  generated header prototypes, source-frontdoor runtime metadata, and can run
  bundle external-ABI evidence locally or over `ssh rvv`.
- Prior journal entries mention local vadd/vsub bundle dry-runs and one `ssh
  rvv` vsub run, but this task must use fresh current-head evidence rather than
  relying on journal text.

## Requirements

- Preserve the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck stack.
  Python may remain only a runner/evidence consumer.
- Keep computation out of `tcrv.exec`; the source-frontdoor body remains the
  source compute authority, and extension-family ops own RVV/scalar behavior.
- Do not add descriptor-driven computation, descriptor-to-C production export,
  descriptor element-count authority, or new Python compiler semantics.
- Do not add family-specific semantic branches in generic core orchestration.
  RVV behavior must remain plugin/target/export local or behind existing
  shared interfaces.
- Use current generated artifacts from the normal source-frontdoor and target
  artifact bundle route. Do not substitute hand-written surrogate kernels,
  artifact-text-only proof, lit metadata alone, explicit-only assumptions, or a
  new wrapper-only path.
- Prove the generated artifact carries and the dispatch/e2e consumer checks:
  selected source identity, op-family identity, selected config, ABI signature,
  runtime element-count authority, embedded RVV invocation contract, and
  dispatcher invocation contract.
- Local generated-artifact evidence must cover both i32-vadd and i32-vsub, or
  one full operation plus a parity regression for the other.
- `ssh rvv` evidence must compile/link/run at least one source-frontdoor
  generated bundle artifact with clang/LLVM RVV intrinsics. If `ssh rvv` blocks,
  finish the local generated-artifact slice and record the exact access,
  toolchain, or runtime blocker without claiming hardware correctness.
- Any code change must be narrowly tied to a real blocker in the existing
  production invocation path or to making the e2e route consume generated
  artifacts more strictly. No broad vector/linalg/dtype/family expansion.

## Acceptance Criteria

- [x] A current-head generated i32-vadd source-frontdoor route reaches dispatch
      source/header/object bundle artifacts through the normal
      plan-and-export target artifact bundle front door.
- [x] A current-head generated i32-vsub source-frontdoor route reaches the same
      normal route with no stale vadd identity or arithmetic.
- [x] The generated source/header/object or bundle evidence records selected
      source identity, op-family identity, selected config, ABI signature,
      runtime element-count/AVL authority, embedded RVV invocation contract,
      and dispatch invocation contract.
- [x] `RuntimeABICallablePlan` / `RVVScalarDispatch` consumption of the generated
      artifact contract is proven by current generated artifact output and
      route-consumer validation, not by a detached Python ABI model.
- [x] Local generated-artifact invocation evidence passes for i32-vadd and
      i32-vsub, or one full op plus explicit local parity regression for the
      other.
- [x] Focused `ssh rvv` clang compile/link/run evidence passes for at least one
      generated source-frontdoor bundle artifact, with command/output summary
      recorded; otherwise the exact blocker is recorded and no hardware claim
      is made.
- [x] Focused C++/lit/script checks covering touched or revalidated behavior
      pass.
- [x] Fail-closed coverage remains in place for missing/stale source identity,
      wrong op-family, missing selected config, missing runtime length role
      data, stale ABI signature, descriptor-only production attempts,
      explicit-only route misuse, and dispatch source identity mismatch.
- [x] Bounded ref-scans confirm this round did not add descriptor authority,
      generic core RVV semantic branches, or unrelated wrapper-only proof.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      finish/archive, and one coherent commit are completed if the task is
      finished.

## Out Of Scope

- No script-only, e2e-harness-only, metadata-only, task-only, journal-only,
  test-only, negative-test-only, wrapper-only, report-only, smoke-only,
  artifact-text-only, or bundle-expectation-only milestone.
- No new compiler semantics in Python.
- No descriptor-to-C production exporter or descriptor/vector-shape runtime
  authority.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branch in generic core passes.
- No broad vector dialect implementation, broad linalg/tensor frontend work,
  new dtype, i64 expansion, LMUL matrix expansion, third op, broad family
  matrix, or performance tuning.
- No GCC/vendor compiler default route.
- No correctness, runtime, or performance claim without actual focused local
  generated-artifact or `ssh rvv` evidence.

## Technical Notes

- Specs read before PRD:
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-generated-artifact-hardware-invocation-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-runtimeabi-generated-artifact-invocation/prd.md`,
  `.trellis/workspace/codex/journal-6.md`.
- Current production/code surfaces inspected before PRD:
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`, and focused RuntimeABI /
  RVVScalarDispatch / RVVMicrokernel / TargetArtifactBundleExport tests.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run script self-tests:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`,
  `python3 scripts/rvv_microkernel_e2e.py --self-test`.
- Run focused lit for script bundle e2e, RuntimeABI callable plan,
  RVVScalarDispatch, RVVMicrokernel, and vector dynamic i32-vadd/i32-vsub target
  artifact bundle export coverage.
- Run local source-frontdoor generated bundle dry-runs for i32-vadd and
  i32-vsub using plan-and-export bundle front door.
- Run one focused non-dry-run `ssh rvv` source-frontdoor generated bundle
  invocation, preferably i32-vsub unless current evidence shows vadd is the
  smaller bounded check.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale source identity, and generic core RVV semantic branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation,
  finish/archive, and commit.

## Implementation Summary

- No compiler/source implementation change was required. Current HEAD
  `31f9955` already contains the production C++ owner path for this task:
  `RuntimeABICallablePlan` owns the runtime invocation contract, and
  `RVVMicrokernel.cpp` plus `RVVScalarDispatch.cpp` emit/consume that contract
  in generated source/header/object bundle artifacts.
- The Hermes brief's concern that the current route had no hardware evidence
  was stale against current repository evidence. This round corrected the task
  truth by rerunning the evidence from current HEAD rather than changing an
  already-correct production path.
- The default source-frontdoor behavior before and after this round is the
  same: vector/SCF source-frontdoor vadd/vsub input uses the normal
  `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` front door,
  not a detached script surrogate.
- The generated vadd/vsub bundle artifacts were emitted under
  `artifacts/tmp/rvv-ssh-rvv-generated-artifact-execution-closure/` with source,
  header, object, bundle index, command summary, and evidence JSON for each
  focused run.
- Trellis task files were created/updated to capture the PRD, current evidence,
  and context for archive/commit. No `.trellis/spec/` update is needed because
  this round introduced no new API, dialect operation, route contract, or
  architecture rule.

## Validation Summary

- Focused build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script self-tests passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` and
  `python3 scripts/rvv_microkernel_e2e.py --self-test`. The scalar dispatch
  self-test reported fail-closed stale selected-family and missing
  selected-plan-metadata checks before passing.
- Focused lit passed using:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-dispatch-bundle-e2e|runtime-abi-callable-plan|RVVScalarDispatch|RVVMicrokernel|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle'`
  from `build/test`; result: 49 selected tests passed.
- Local current-head vadd bundle dry-run passed:
  `codex-sshclosure-vadd-local`, mode `dry-run`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vadd`, pipeline
  `vector-scf-source-frontdoor-to-plan-and-export-bundle`, source SHA256
  `aaf65ba12b0529b3cfe7d95d0802f53c53519b6bc46bf787faadf31ee8f6b60b`.
- Local current-head vsub bundle dry-run passed:
  `codex-sshclosure-vsub-local`, mode `dry-run`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vsub`, pipeline
  `vector-scf-source-frontdoor-to-plan-and-export-bundle`, source SHA256
  `519a959ed18302d3a4c8a6f2d3c309ee72607fd23f86ce72f2e61586b0c2bb10`.
- Both dry-run evidence files validate the dispatch invocation contract:
  ordered roles
  `lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard`,
  runtime `n`, guard `rvv_available`, production owner
  `rvv-scalar-dispatch-target`, and matching vadd/vsub runtime ABI names.
- Focused `ssh rvv` vsub bundle run passed:
  `codex-sshclosure-vsub-ssh`, mode `ssh`, `ssh_evidence=true`,
  `ssh_evidence_verified=true`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vsub`, source SHA256
  `519a959ed18302d3a4c8a6f2d3c309ee72607fd23f86ce72f2e61586b0c2bb10`.
- `ssh rvv` host/toolchain evidence:
  `riscv64`, `/usr/bin/clang`, `Ubuntu clang version 18.1.3 (1ubuntu1)`,
  compile flags `-O2 -march=rv64gcv -mabi=lp64d`, link flags
  `-O2 -march=rv64gcv -mabi=lp64d -no-pie`.
- Remote source-built command chain passed:
  compile generated external caller, compile generated dispatch source
  `artifact-0-runtime-callable-c-source-tcrv-export-rvv-scalar-i32-vsub-dispatch-c.c`,
  link `rvv_bundle_dispatch_external_caller_from_source`, and run it.
  Output:
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- Remote bundle-object command chain passed:
  link generated bundle object
  `artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o`
  with the external caller, then run
  `rvv_bundle_dispatch_external_caller_from_bundle_object`. Output:
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- Generated vsub source/header/index evidence records:
  `dispatch_selected_source_identity` with `family=i32-vsub`,
  `operator=subtract`, `microkernel_op=tcrv_rvv.i32_vsub_microkernel`,
  `emitc_source_op=tcrv_rvv.i32_sub`;
  `selected_binary_config` with `shape=i32m1`, `sew=32`, `lmul=m1`, runtime
  `n`, guard `rvv_available`; `source_frontend_runtime_avl_authority` with
  `source_kind=mlir-vector-scf-runtime-i32-vsub.v1`; and RVV intrinsic
  `__riscv_vsub_vv_i32m1`.
- Embedded RVV invocation contract in current generated artifacts:
  callable
  `tcrv_rvv_i32_vsub_microkernel_frontend_vector_dynamic_bundle_i32_vsub_rvv_first_slice`,
  runtime ABI name `rvv-i32-vsub-runtime-callable-c-function.v1`, ordered roles
  `lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count`,
  runtime count `n`, production owner `rvv-target-export`.
- Dispatch invocation contract in current generated artifacts:
  callable `tcrv_dispatch_i32_vsub_frontend_vector_dynamic_bundle_i32_vsub`,
  runtime ABI name
  `rvv-scalar-i32-vsub-dispatch-runtime-callable-c-function.v1`, ordered roles
  `lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard`,
  runtime count `n`, guard `rvv_available`, production owner
  `rvv-scalar-dispatch-target`.
- Bounded ref-scan confirmed there is no production/source diff under
  `include`, `lib`, `scripts`, `test`, `tools`, `cmake`, or `CMakeLists.txt`.
  Therefore this round added no descriptor authority path, no generic core RVV
  semantic branch, and no wrapper-only proof layer.
- `git diff --check` passed.
- Trellis context validation passed:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-ssh-rvv-generated-artifact-execution-closure`.

## Spec Update Judgment

No `.trellis/spec/` update is required. The work produced fresh current-head
local and hardware evidence for an existing specified RuntimeABI/dispatch
artifact route; it did not introduce a new long-lived rule or compiler API.
