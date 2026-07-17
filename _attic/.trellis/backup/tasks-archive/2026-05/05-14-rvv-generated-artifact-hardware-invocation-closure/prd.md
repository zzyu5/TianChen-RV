# RVV generated artifact hardware invocation closure

## Goal

Prove at current HEAD that generated source-frontdoor i32 vadd/vsub RVV
artifacts travel through the normal generated bundle/export route into the
centralized `RuntimeABICallablePlan` / `RVVScalarDispatch` callable invocation
contract, then through focused local generated-artifact invocation and at least
one `ssh rvv` compile/link/run evidence leg.

This round starts from `8780c6b`, where the support-layer
`RuntimeABIInvocationContract` builder already exists and
`RVVMicrokernel.cpp` / `RVVScalarDispatch.cpp` already consume it. The remaining
work is therefore not to invent another ABI path. The round should add only the
production or route-consumer changes needed to make the current-head generated
artifact invocation proof explicit, fail-closed, and reproducible.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd` confirmed the repository root.
- Initial `git status --short` was clean.
- Initial HEAD was `8780c6b feat(rvv): centralize runtime abi invocation contracts`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief as
  `.trellis/tasks/05-14-rvv-generated-artifact-hardware-invocation-closure`.
- The archived `05-14-rvv-exported-artifact-runtime-consumption` task made
  `RVVScalarDispatch.cpp` consume the embedded RVVMicrokernel selected-source
  identity and runtime ABI invocation contract before dispatch export succeeds.
- The archived `05-14-rvv-runtimeabi-generated-artifact-invocation` task added
  `RuntimeABIInvocationContract` support APIs and wired RVVMicrokernel and
  RVVScalarDispatch to the shared builder. Its archived summary records local
  vadd/vsub dry-runs and one `ssh rvv` vsub run, but this task must re-check
  the live current-head route rather than relying only on the archived summary.
- Current `scripts/rvv_scalar_dispatch_e2e.py` already parses
  `embedded_rvv_artifact_contract` from generated dispatch artifacts and can
  run both bundle external-ABI and direct self-check `ssh rvv` evidence.
- Current e2e evidence records embedded RVV callable contracts, dispatch
  runtime params, generated symbols, selected config, branch coverage, and ssh
  command summaries. The likely narrow gap is making the generated dispatcher's
  own `dispatch_runtime_abi_invocation_contract` first-class e2e evidence and
  validating it against the generated header signature, runtime `n`, and
  `rvv_available` guard.

## Requirements

- Preserve the existing C++/MLIR/LLVM/TableGen/CMake/lit stack. Python may only
  remain runner/evidence tooling.
- Do not add descriptor-driven computation, descriptor-to-C production export,
  new Python compiler semantics, or RVV semantic branches in generic core
  orchestration.
- Keep the active route as bounded vector source-frontdoor i32 vadd/vsub
  source input -> generated `tcrv.exec` ABI boundary -> selected RVV/scalar
  dispatch bundle -> generated source/header/object artifacts -> normal
  RuntimeABI/RVVScalarDispatch callable invocation.
- Validate generated artifact evidence from `tcrv-opt` /
  `tcrv-translate` / target artifact export outputs. Do not substitute
  hand-written surrogate kernels, descriptor shape authority, lit metadata
  alone, or explicit-only assumptions.
- If production C++ already carries the centralized contract correctly, keep C++
  unchanged and add only route-consumer e2e validation that consumes the
  generated production artifacts. If current-head inspection shows a production
  gap, fix the production owner instead of papering over it in Python.
- The e2e route consumer must fail closed when the dispatch invocation
  contract is missing or stale against callable symbol, runtime ABI kind/name,
  ABI signature, runtime element-count role, dispatch guard role, production
  owner, selected family, or generated header surface.
- Local generated-artifact invocation must cover i32-vadd and i32-vsub parity,
  or one full op plus an explicit local parity regression for the other.
- `ssh rvv` evidence must cover at least one source-frontdoor generated
  artifact invocation. If hardware or toolchain access blocks the run, the task
  must finish the local generated-artifact slice, record the exact blocker, and
  avoid runtime correctness/performance claims.

## Acceptance Criteria

- [x] Generated i32-vadd source-frontdoor artifact route reaches dispatch
      source/header/object artifacts through the normal bundle/export front
      door and records selected source identity, op family, selected config,
      ABI signature, runtime length authority, embedded RVV invocation
      contract, and dispatch invocation contract.
- [x] Generated i32-vsub source-frontdoor artifact route reaches the same
      normal route with matching parity evidence and without stale vadd
      identity.
- [x] `rvv_scalar_dispatch_e2e.py` validates and records the generated
      dispatcher's own `dispatch_runtime_abi_invocation_contract` as a
      route-consumer of the current production source, not as an invented
      Python ABI model.
- [x] The e2e self-test includes fail-closed coverage for stale dispatch
      invocation ABI fields and stale dispatch guard/runtime length contract
      data.
- [x] Focused lit/FileCheck evidence for vector vadd/vsub bundle artifacts
      requires the dispatch invocation contract in generated source/header
      output.
- [x] Focused local generated-artifact dry-run evidence passes for i32-vadd and
      i32-vsub source-frontdoor bundle routes.
- [x] Focused `ssh rvv` compile/link/run evidence passes for at least one
      generated source-frontdoor bundle invocation, or an exact blocker is
      recorded after local generated-artifact generation.
- [x] `git diff --check`, `git diff --cached --check`, relevant Trellis
      validation, finish/archive, and one coherent commit complete if the task
      is finished.

## Out Of Scope

- No RuntimeABI-free dispatch-only milestone.
- No callable-plan unit-test-only milestone.
- No script-only, e2e-harness-only, metadata-only, task-only, journal-only,
  test-only, negative-test-only, wrapper-only, report-only, smoke-only,
  artifact-text-only, or bundle-expectation-only milestone unless the route
  consumer is directly validating generated production artifacts and current
  production C++ is already correct.
- No new Python compiler semantics.
- No descriptor-to-C production exporter.
- No descriptor element count or vector shape as compute/config/runtime
  authority.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No broad vector dialect implementation, linalg/tensor frontend expansion,
  new dtype, i64 expansion, LMUL matrix, third op, broad family matrix, or
  performance tuning.
- No GCC/vendor compiler default route.
- No runtime correctness or performance claim without focused evidence
  actually run.

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
  `.trellis/tasks/archive/2026-05/05-14-rvv-runtimeabi-generated-artifact-invocation/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-exported-artifact-runtime-consumption/prd.md`,
  `.trellis/workspace/codex/journal-6.md`.
- Current production files inspected before PRD:
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/TargetArtifactExport.cpp`.
- Current route-consumer and tests inspected before PRD:
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `scripts/rvv_microkernel_e2e.py`,
  `test/Support/RuntimeABICallablePlanTest.cpp`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`.

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
- Run focused lit from `build/test` for
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`,
  `Target/RVVScalarDispatch`,
  `Target/RVVMicrokernel`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  and `Support/runtime-abi-callable-plan.test`.
- Run local source-frontdoor bundle dry-runs for i32-vadd and i32-vsub.
- Run one focused non-dry-run `ssh rvv` source-frontdoor bundle invocation,
  preferably i32-vsub unless current evidence suggests vadd is the smaller
  delta.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale source identity, and generic core RVV semantic branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation,
  finish/archive, and commit.

## Implementation Summary

- No production C++ rewrite was needed. Current HEAD already had
  `RuntimeABIInvocationContract` in `RuntimeABICallablePlan`, RVVMicrokernel
  direct callable contract emission, and RVVScalarDispatch dispatcher contract
  emission through the centralized support builder.
- Updated `scripts/rvv_scalar_dispatch_e2e.py` so the generated dispatch
  route-consumer parses `dispatch_runtime_abi_invocation_contract` from the
  produced dispatch source and validates it against:
  dispatcher function symbol, dispatch runtime ABI kind/name, ordered ABI
  signature roles, runtime element-count C name, dispatch guard C name,
  runtime_param metadata, and production owner
  `rvv-scalar-dispatch-target`.
- The script now writes the validated
  `dispatch_runtime_abi_invocation_contract` into both bundle-mode and direct
  e2e evidence JSON.
- Added e2e self-test fail-closed coverage for stale dispatch runtime ABI
  name, stale runtime element-count C name, and stale dispatch guard C name.
- Updated focused source-frontdoor vadd/vsub bundle FileCheck tests to require
  the dispatch invocation contract in generated source and header output.
- Updated script bundle e2e FileCheck tests to require the dispatch invocation
  contract in generated vadd/vsub source output and evidence JSON.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- C++ smoke tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script self-tests passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`,
  `python3 scripts/rvv_microkernel_e2e.py --self-test`.
- Focused lit passed:
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`
  (`3/3`).
- Focused lit passed:
  `Support/runtime-abi-callable-plan.test`,
  `Target/RVVScalarDispatch`,
  `Target/RVVMicrokernel`
  (`46/46`).
- Local source-frontdoor bundle dry-runs passed:
  `codex-runtimeabi-current-vadd-local` and
  `codex-runtimeabi-current-vsub-local` under
  `artifacts/tmp/runtimeabi-invocation-current/`.
- Local dry-run evidence JSON for vadd and vsub records
  `dispatch_runtime_abi_invocation_contract.contract` with callable symbol,
  runtime ABI name, ordered roles
  `lhs-input-buffer->rhs-input-buffer->output-buffer->runtime-element-count->dispatch-availability-guard`,
  runtime `n`, guard `rvv_available`, and production owner
  `rvv-scalar-dispatch-target`.
- Focused `ssh rvv` evidence passed for generated source-frontdoor i32-vsub
  bundle invocation:
  `codex-runtimeabi-current-vsub-ssh`, `status=success`,
  `ssh_evidence=true`, `ssh_evidence_verified=true`.
  Remote compile, link, run, and output validation all succeeded; both source
  built object and bundle-index object caller runs printed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- Bounded diff ref-scan confirmed this round changed only the route-consumer
  script, script/FileCheck tests, and Trellis task files. No generic core pass,
  descriptor-to-C exporter, descriptor authority path, or RVV semantic branch
  was added.
- `git diff --check` passed.
- Trellis context validation passed for the task.

## Spec Update Judgment

No `.trellis/spec/` update is required. This round did not introduce a new
dialect op, plugin protocol, target artifact route, runtime ABI schema, or
architecture rule. It tightened route-consumer validation and focused evidence
for the already-specified generated RuntimeABI invocation contract.
