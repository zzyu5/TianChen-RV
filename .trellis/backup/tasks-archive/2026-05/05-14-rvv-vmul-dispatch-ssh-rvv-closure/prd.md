# RVV vmul dispatch and ssh-rvv execution closure

## Goal

Close the current Hermes direction against real repository evidence: `i32-vmul`
must be a reusable RVV binary-family member that travels through the normal
source-frontdoor, selected RVV/scalar planning, generated artifact export,
`RuntimeABICallablePlan`, `RVVScalarDispatch`, local generated-artifact
invocation, and focused `ssh rvv` compile/run evidence. If current HEAD already
contains the production implementation, this round must not invent a redundant
replacement path; it must refresh the focused validation and repair only real
gaps found by inspection or checks.

## Current Repository Facts

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd` confirmed the repository root.
- Initial `git status --short` was clean.
- Initial HEAD was `ba07cea feat(rvv): expose vmul vector artifact closure`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief as
  `.trellis/tasks/05-14-rvv-vmul-dispatch-ssh-rvv-closure`.
- The Hermes brief says `ba07cea` is aligned but incomplete because the live
  summary did not show production changes in `RuntimeABICallablePlan`,
  `RVVScalarDispatch`, `RVVMicrokernel`, `TargetArtifactExport`, or focused
  `ssh rvv` vmul execution evidence.
- Current code inspection shows the brief is likely stale relative to HEAD:
  `RuntimeABIInvocationContract` has a required `familyID` field and prints
  `family=...`; `RVVScalarDispatch` validates embedded RVV invocation contracts
  and prints dispatch invocation contracts with the selected RVV family; the
  RVV registry contains `i32-vmul` source identity, intrinsic prefix, route ids,
  ABI names, glue role, and external ABI group; the explicit vector vmul pass
  is exposed in `Passes.h`, `Passes.td`, and `tcrv-opt`; the vmul scripts have
  direct generated-artifact and dispatch bundle paths.
- Archived PRDs and the workspace journal record prior local and `ssh rvv`
  evidence for generated vmul artifacts, but this task treats those records as
  context only. Current HEAD code and freshly rerun focused checks are the
  authority for this round.

## Requirements

- Preserve the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck compiler
  stack. Python may only remain a runner, artifact parser, probe, or support
  script.
- Keep computation semantics out of `tcrv.exec`.
- Do not add descriptor-driven computation, descriptor-to-C production export,
  descriptor element-count authority, vector-shape runtime authority, or Python
  compiler semantics.
- Do not add RVV semantic branches in generic core orchestration.
- Keep vmul routed by the shared RVV binary family contract, not a vmul-only
  branch outside RVV plugin/target family ownership.
- Reuse the existing source-frontdoor selected boundary, RVV binary registry,
  plugin materialization/emission, `RVVMicrokernel`, `TargetArtifactExport`,
  `RuntimeABICallablePlan`, and `RVVScalarDispatch` production owners.
- Preserve vadd/vsub generated-artifact and dispatch regressions on the same
  touched surfaces.
- Unsupported, stale, descriptor-only, explicit-only, or partially declared
  family states must fail before artifact/runtime claims.

## Acceptance Criteria

- [x] Current HEAD inspection identifies whether any production owner is still
      missing for vmul dispatch/runtime/ssh closure.
- [x] If a real missing production owner is found, it is fixed in
      C++/MLIR/TableGen/CMake/lit code using existing family-owned contracts,
      not a helper-only or descriptor-only workaround.
- [x] If no production gap is found, the task records that the Hermes brief was
      stale and finishes through refreshed evidence rather than redundant code.
- [x] Generated vmul artifacts use the same shared family contract as vadd/vsub
      for source identity, family identity, selected config, runtime ABI
      signature, runtime length role data, EmitC/intrinsic mapping, artifact
      naming, and dispatch validation.
- [x] vmul is locally invoked through the normal generated artifact dispatch
      route, not through a surrogate or artifact-text-only check.
- [x] Focused `ssh rvv` compile/run evidence covers generated vmul if the
      environment is available. If remote evidence cannot complete, the exact
      production or environment blocker is recorded after local generation and
      invocation checks.
- [x] vadd/vsub generated-artifact and dispatch regressions remain green.
- [x] Fail-closed checks cover the touched surfaces for unknown/stale family
      identity, missing selected config, missing runtime length role data,
      missing EmitC body mapping, stale ABI signature, descriptor-only
      production attempts, explicit-only route misuse, and source-identity
      mismatch where practical.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      task finish/archive, and final worktree cleanliness are completed if the
      task finishes.

## Out Of Scope

- No broad dtype, mask/tail, LMUL matrix, linalg/tensor frontend, performance
  tuning, or additional op-family expansion.
- No helper-only, pass-registration-only, tool-exposure-only, microkernel-only,
  registry-only, ABI-contract-only, documentation-only, metadata-only,
  PRD-only, journal-only, test-only, negative-test-only, script-only,
  smoke-only, report-only, artifact-text-only, or evidence-packaging-only
  milestone when a real production gap exists.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.
- No descriptor-to-C production exporter.
- No RVV semantic branches in generic core passes.
- No replacing clang/LLVM as the default native compiler route.
- No runtime, correctness, or performance claim without focused local
  generated-artifact evidence or real `ssh rvv` evidence.

## Evidence Plan

- Inspect the named production owners from the brief:
  `Passes.h`, `Passes.td`, `LowerSourceRVVBinaryToExec.cpp`,
  `RVVBinaryFamilyRegistry.h`, RVV plugin selected-boundary/materialization/
  emission planning, `RVVMicrokernel.cpp`, `TargetArtifactExport.cpp`,
  `RuntimeABICallablePlan`, `RVVScalarDispatch`, and focused scripts/tests.
- Build focused support, target, RVV target, scalar target, RVV plugin,
  transforms, `tcrv-opt`, `tcrv-translate`, and focused C++ test targets.
- Run focused C++ tests:
  `tianchenrv-runtime-abi-callable-plan-test`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-rvv-extension-plugin-test`.
- Run script self-tests:
  `scripts/rvv_microkernel_e2e.py --self-test` and
  `scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run focused lit/FileCheck for vector vadd/vsub/vmul, RuntimeABI,
  RVVMicrokernel, TargetArtifactExport, TargetArtifactBundleExport,
  RVVScalarDispatch, and script e2e tests.
- Run local generated-artifact/runtime evidence for vector/SCF vmul plus
  vadd/vsub dispatch regressions.
- Run focused `ssh rvv` generated vmul evidence if local generation and
  invocation are complete and the environment responds.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale source identity, stale vadd defaults, and generic core RVV
  semantic branches.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-vmul-generated-artifact-ssh-rvv-closure/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-14-rvv-vmul-op-family-artifact-runtime-closure/prd.md`.
- Initial code inspection found implementation evidence in
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `include/TianChenRV/Transforms/Passes.h`,
  `include/TianChenRV/Transforms/Passes.td`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `scripts/rvv_microkernel_e2e.py`, and
  `scripts/rvv_scalar_dispatch_e2e.py`.

## Implementation Summary

- No production source gap was found at current HEAD. The Hermes direction
  brief was stale relative to `ba07cea` and the commits already reachable from
  it.
- Current HEAD already contains the production vmul closure owners:
  `RuntimeABIInvocationContract.familyID` in Support, RVVScalarDispatch
  embedded/direct dispatch invocation-contract validation, registry-derived
  `i32-vmul` routes and ABI names, RVV plugin selected-boundary/materialization/
  emission consumption, generated RVVMicrokernel artifacts, target artifact
  bundle export, explicit vmul vector source adapter, `tcrv-opt` registration,
  and local/ssh-capable e2e script paths.
- This round therefore made no C++/MLIR/TableGen/CMake/script source change.
  The only tracked changes are Trellis task/context records for this refreshed
  current-HEAD closure round.

## Validation Summary

- Trellis context validation passed:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-vmul-dispatch-ssh-rvv-closure`.
- Focused build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget TianChenRVRVVPlugin TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script self-tests passed:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Focused lit passed from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vadd-to-exec|vector-dynamic-i32-vsub-to-exec|vector-dynamic-i32-vmul-to-exec|vector-dynamic-i32-vsub-explicit-invalid|vector-dynamic-i32-vmul-explicit-invalid|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vmul-and-export-target-artifact-bundle|rvv-microkernel-bundle-e2e|rvv-scalar-dispatch-bundle-e2e|rvv-microkernel-runtime-abi-role-binding|rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding|rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding|rvv-scalar-i32-vmul-dispatch'`;
  result: 14 selected tests passed.
- Local direct RVVMicrokernel generated bundle dry-runs passed for:
  `codex-vmul-dispatch-closure-micro-vadd-local`,
  `codex-vmul-dispatch-closure-micro-vsub-local`, and
  `codex-vmul-dispatch-closure-micro-vmul-local`. The vmul source SHA256 was
  `3120d67ddd9850fbee47d0aafae34c8e30ef5dab0537040bba36176687b4411d`.
- Local RVVScalarDispatch generated bundle dry-runs passed for:
  `codex-vmul-dispatch-closure-dispatch-vadd-local`,
  `codex-vmul-dispatch-closure-dispatch-vsub-local`, and
  `codex-vmul-dispatch-closure-dispatch-vmul-local`. The vmul source SHA256
  was `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- Generated local vmul artifacts explicitly contain `family=i32-vmul`,
  `source_kind=mlir-vector-scf-runtime-i32-vmul.v1`,
  `__riscv_vmul_vv_i32m1`, `runtime_element_count_c_name=n`,
  RVVMicrokernel invocation contract ownership, RVVScalarDispatch invocation
  contract ownership, and external caller multiplication checks.
- Focused `ssh rvv` direct RVVMicrokernel evidence passed:
  `codex-vmul-dispatch-closure-micro-vmul-ssh-rvv`, selected kernel
  `frontend_vector_dynamic_i32_vmul`, source SHA256
  `3120d67ddd9850fbee47d0aafae34c8e30ef5dab0537040bba36176687b4411d`.
  Both source-built and bundle-object caller runs printed
  `tcrv_rvv_i32_vmul_microkernel_external_abi_ok counts=7,16,23`.
- Focused `ssh rvv` RVVScalarDispatch evidence passed:
  `codex-vmul-dispatch-closure-dispatch-vmul-ssh-rvv`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vmul`, source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`,
  `ssh_evidence_verified=true`. Both source-built and bundle-object caller
  runs printed
  `tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- Bounded ref-scan found no new tracked source changes and no new production
  descriptor-only, descriptor-to-C, explicit-only, stale vadd default, or
  generic core RVV semantic-branch authority in this round. Existing hits are
  registry/script/test/spec text, descriptor-quarantine checks, or explicit
  bounded source adapters.
- `git diff --check` passed before archive.
