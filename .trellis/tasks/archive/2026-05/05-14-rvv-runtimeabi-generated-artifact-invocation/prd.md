# RVV RuntimeABI generated artifact invocation closure

## Goal

Close the callable runtime invocation boundary for generated source-frontdoor
i32 vadd/vsub RVV artifacts after dispatch-side exported artifact contract
consumption landed in `8616cb5`. This round must move the invocation contract
from local artifact/dispatch string construction into the shared
`RuntimeABICallablePlan` support owner and then reuse that production contract
from RVVMicrokernel and RVVScalarDispatch generated artifact invocation paths.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD before this task is
  `8616cb5 feat(rvv): consume exported artifact contracts in dispatch`.
- `git status --short` was clean before task creation.
- `.trellis/.current-task` did not exist, so this task was created from the
  Hermes direction brief as
  `.trellis/tasks/05-14-rvv-runtimeabi-generated-artifact-invocation`.
- The previous archived task
  `05-14-rvv-exported-artifact-runtime-consumption` changed
  `lib/Target/RVV/RVVScalarDispatch.cpp` so dispatch consumes the embedded
  RVVMicrokernel source identity and runtime ABI invocation contract before
  dispatch artifact export succeeds.
- Current `RuntimeABICallablePlan.cpp` builds the IR-backed callable ABI plan
  from `tcrv.exec.mem_window` and `tcrv.exec.runtime_param`, and validates
  metadata mirrors, but it does not yet own a reusable generated artifact
  `runtime_abi_invocation_contract` object or formatting/validation helper.
- Current `RVVMicrokernel.cpp` and `RVVScalarDispatch.cpp` each construct
  invocation-contract comments locally. That leaves the generated artifact
  invocation boundary duplicated across target owners instead of being carried
  by the RuntimeABI callable plan owner.
- The existing generated dispatch route already calls generated RVV/scalar
  callable functions through the normal `RuntimeABI`/`RVVScalarDispatch` route;
  this round should expose and harden that route through production C++ owner
  changes rather than inventing script-only evidence.

## Requirements

- Add an active production C++ support-layer owner for generated callable
  runtime invocation contracts in `RuntimeABICallablePlan`/`RuntimeABIContract`
  territory.
- The support-layer contract must preserve callable symbol, source owner,
  runtime ABI kind/name, optional runtime glue role, ordered ABI parameter
  roles, runtime element-count C name, optional dispatch guard C name, and
  production owner.
- The support-layer builder must fail closed when the runtime element-count
  role is missing, duplicated, stale against the selected runtime length C
  name, or when dispatch guard data is missing/stale for dispatch contracts.
- `RVVMicrokernel.cpp` must emit the generated RVV
  `runtime_abi_invocation_contract` through this support-layer contract rather
  than local ad-hoc formatting.
- `RVVScalarDispatch.cpp` must validate and emit both the embedded RVV
  artifact invocation contract expectation and the generated dispatcher
  invocation contract through the same support-layer contract, while preserving
  selected source identity, op-family identity, selected config, runtime
  length roles, and generated callable calls.
- vadd/vsub source-frontdoor generated artifact routes must remain covered
  through the normal C++ target artifact and dispatch bundle route. It is
  acceptable for focused ssh evidence to cover one generated source-frontdoor
  invocation, provided local vadd/vsub parity and regression evidence remains
  green.
- Python scripts may parse and run generated artifacts only as evidence
  consumers. No Python compiler semantics or descriptor-to-C production route
  may be added.
- Generic core orchestration must remain target-neutral; no RVV semantic
  branch may be added to common core passes or `tcrv.exec`.

## Acceptance Criteria

- [x] `RuntimeABICallablePlan` or adjacent RuntimeABI support code owns a
      structured runtime invocation contract builder/formatter used by
      generated artifact invocation paths.
- [x] The support-layer contract has focused C++ coverage for direct callable
      and dispatch callable invocation contracts, including runtime
      element-count and dispatch-guard fail-closed cases.
- [x] `RVVMicrokernel.cpp` emits RVV source/header invocation contract comments
      from the support-layer contract while preserving existing generated
      vadd/vsub ABI signatures and selected-source/config comments.
- [x] `RVVScalarDispatch.cpp` consumes the same support-layer contract for
      embedded RVV artifact validation and dispatcher source/header invocation
      contract emission.
- [x] Generated source-frontdoor i32-vadd and i32-vsub bundle routes still
      reach generated dispatch source/header/object artifacts through the
      normal TargetArtifactExport/BundleExport route and preserve source
      identity, op-family identity, selected config, ABI signature, runtime
      length roles, and callable branch calls.
- [x] Missing/stale runtime element-count role data, stale ABI signature,
      missing/stale dispatch guard data, op-family mismatch, source identity
      mismatch, and descriptor-only production attempts fail before successful
      generated artifact or dispatch export in focused C++/lit/script coverage
      touched by this round.
- [x] Focused local build/test evidence covers touched support, RVV target,
      target artifact export, plugin/materialization tools, and affected
      vadd/vsub runtime-dispatch routes.
- [x] Focused `ssh rvv` compile/run evidence covers at least one generated
      source-frontdoor artifact invocation, or an exact blocker is recorded
      after local generation and compile.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      finish/archive, and one coherent commit complete if the task is
      finished.

## Out Of Scope

- No RuntimeABI-free dispatch-only milestone.
- No script-only, e2e-harness-only, metadata-only, task-only, journal-only,
  test-only, negative-test-only, wrapper-only, report-only, smoke-only,
  artifact-text-only, or bundle-expectation-only milestone.
- No new Python compiler semantics.
- No descriptor-to-C production exporter, descriptor element-count authority,
  or descriptor/vector shape as compute/config/runtime authority.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No broad vector dialect implementation, linalg/tensor frontend expansion,
  new dtype, i64 expansion, LMUL matrix, third op, broad family matrix, or
  performance tuning.
- No GCC/vendor compiler default route.
- No runtime, correctness, or performance claim without focused evidence
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-exported-artifact-runtime-consumption/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-artifact-export-source-frontdoor-contract/prd.md`,
  `.trellis/workspace/codex/journal-6.md`.
- Production files inspected before PRD:
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Support/RuntimeABIContract.cpp`,
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `include/TianChenRV/Support/RuntimeABIContract.h`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `test/Support/RuntimeABICallablePlanTest.cpp`,
  focused RVVMicrokernel/RVVScalarDispatch/FileCheck tests, and e2e scripts.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-support-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-support-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run focused lit from `build/test` for:
  `Support/runtime-abi-callable-plan.test`,
  `Target/RVVMicrokernel`,
  `Target/RVVScalarDispatch`,
  dynamic vector vadd/vsub target artifact bundle export tests, and
  script e2e tests affected by this round.
- Run script self-tests:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`,
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run local generated source-frontdoor dispatch bundle dry-runs for i32-vadd
  and i32-vsub.
- Run focused `ssh rvv` generated artifact invocation for at least one changed
  source-frontdoor route if reachable.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale source identity, and generic core RVV semantic branches.
- Run `git diff --check`, `git diff --cached --check`, and Trellis validation
  before finish/archive.

## Implementation Summary

- Added `RuntimeABIInvocationContract`,
  `buildRuntimeABIInvocationContract`, `formatRuntimeABIOrderedRoles`, and
  `formatRuntimeABIInvocationContractCommentBody` in
  `RuntimeABICallablePlan`.
- The support-layer builder now validates required bounded text fields, unique
  runtime element-count ABI role, stale runtime count C names, direct-callable
  dispatch-guard misuse, missing/duplicate dispatch guards, and stale dispatch
  guard C names.
- `RVVMicrokernel.cpp` now emits direct RVV source/header
  `runtime_abi_invocation_contract` comments through the support-layer
  contract instead of local ad-hoc formatting.
- `RVVScalarDispatch.cpp` now builds the expected embedded RVV artifact
  invocation contract through the same support API before requiring that
  snippet in the embedded RVVMicrokernel source, and emits dispatcher
  `dispatch_runtime_abi_invocation_contract` comments from the support
  contract.
- `RuntimeABICallablePlanTest.cpp` now covers direct callable and dispatch
  invocation contract formatting plus fail-closed stale runtime count,
  direct-with-guard, missing guard, and stale guard cases.
- Added a code-spec scenario in
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` for the
  generated RuntimeABI invocation contract API and validation matrix.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVRVVTarget TianChenRVTarget tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`.
- C++ tests passed:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed from `build/test` with filter
  `runtime-abi-callable-plan|RVVMicrokernel|RVVScalarDispatch|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|rvv-microkernel-e2e|rvv-scalar-dispatch-bundle-e2e`
  (`50/50` selected tests passed).
- Script self-tests passed:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`,
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local generated source-frontdoor dispatch bundle dry-runs passed for both
  `codex-runtimeabi-vadd-local` and `codex-runtimeabi-vsub-local` under
  `artifacts/tmp/runtimeabi-invocation/`.
- Focused `ssh rvv` generated source-frontdoor dispatch bundle invocation
  passed for `codex-runtimeabi-vsub-ssh` with `status=success`,
  `ssh_evidence=true`, and `ssh_evidence_verified=true`.
- Evidence JSON for vadd/vsub dry-runs and the ssh run records
  `embedded_rvv_artifact_contract.runtime_abi_invocation_contract` with
  generated callable symbol, selected ABI name, ordered roles, runtime `n`,
  and production owner.
- `git diff --check` passed.
- Broad `cmake --build build --target check-tianchenrv -j2` currently reports
  `229/232` lit tests passed and 3 pre-existing broad route/FileCheck failures:
  `Target/ArtifactExport/target-source-artifact-routes.test`,
  `Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`, and
  `Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`. The focused
  tests covering this RuntimeABI invocation-contract task passed; the broad
  failures are stale source/artifact route expectations not touched by this
  round and were not folded into this task to avoid unrelated cleanup.

## Spec Update Judgment

Spec update was required because this round added a support-layer API and
cross-layer generated artifact contract. The new scenario was added to
`.trellis/spec/lowering-runtime/emission-runtime-contract.md` with signature,
contract, validation matrix, cases, tests, and wrong/correct examples.
