# Journal - codex (Part 6)

> Continuation from `journal-5.md` (archived at ~2000 lines)
> Started: 2026-05-14

---



## Session 62: RVV artifact export source-frontdoor contract

**Date**: 2026-05-14
**Task**: RVV artifact export source-frontdoor contract
**Branch**: `main`

### Summary

Moved selected RVV source identity into RVVMicrokernel artifact/export validation and generated source/header comments; verified vadd/vsub bundle route locally and vsub on ssh rvv.

### Main Changes

- Added RVVMicrokernel source/header artifact comments exposing the selected source identity in the same compact contract form consumed by dispatch validation.
- Required finite typed RVV target artifact routes to carry `tcrv_rvv.selected_binary_source_kind` and the expected selected microkernel op in selected-plan metadata before source/header/object export succeeds.
- Extended RVVMicrokernel selected-plan source identity validation beyond `frontend-lowering`, so direct typed vsub/vadd source identity is checked through the same C++ contract.
- Updated focused C++ and lit coverage for vadd/vsub source-frontdoor bundle output, vsub source/header bundle output, and target artifact export fixture metadata.

### Git Commits

Final session commit recorded in git history for this round.

### Testing

- [OK] Built `tcrv-opt`, `tcrv-translate`, RVV target/plugin libraries, `tianchenrv-rvv-extension-plugin-test`, and `tianchenrv-target-artifact-export-test`.
- [OK] Ran `tianchenrv-rvv-extension-plugin-test` and `tianchenrv-target-artifact-export-test`.
- [OK] Ran focused lit for `RVVMicrokernel`, `TargetArtifactBundleExport`, and `RVVScalarDispatch` surfaces.
- [OK] Ran `rvv_microkernel_e2e.py --self-test` and `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Ran local vadd/vsub source-frontdoor bundle dry-runs and one `ssh rvv` vsub generated artifact invocation.
- [OK] Ran `git diff --check` and Trellis task validation.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 67: Direct C semantic exporter erasure completion

**Date**: 2026-05-15
**Task**: `direct-c-semantic-exporter-erasure`
**Branch**: `main`

### Summary

Completed the Wrong Logic deletion owner that remained after `ea6bd22`: stale
direct C positive fixtures and dead private raw C body printer entry points are
gone, while the surviving focused tests assert unsupported/deleted-route
behavior and route absence.

### Main Changes

- Removed legacy lit/e2e fixtures that still expected runtime-callable RVV,
  scalar, or RVV+scalar dispatch direct source/header/object/self-check/bundle
  success.
- Rewrote RVV plugin and i32 binary registry C++ tests to expect unsupported
  deleted-route emission plans and absent target artifact route registration.
- Kept `rvv_probe_to_mlir` coverage for MLIR/profile replay but changed source
  export expectations to generic front-door fail-closed behavior.
- Removed private `printMicrokernelSource` and `printDispatchSource` raw body
  printer entry points from RVV/scalar/dispatch target files.
- Removed stale direct-C e2e runner scripts and rewrote README current-route
  guidance so it describes the deleted routes instead of executable C evidence
  bridges.
- Added Trellis spec scenarios documenting deleted route contracts and test
  ownership.

### Testing

- [OK] Focused C++ build for `tcrv-opt`, `tcrv-translate`, target/plugin
  libraries, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-i32-binary-family-registry-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] `build/bin/tianchenrv-i32-binary-family-registry-test`.
- [OK] Focused lit fail-closed subset: 6/6 passed.
- [OK] `git diff --check`.
- [OK] `git diff --cached --check`.
- [OK] Trellis task validation before archive.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 137/137 passed.

### Status

[OK] **Ready to finish/archive**

### Next Steps

- Run Trellis validation, archive the task with `--no-commit`, and create one
  coherent commit.


## Session 67: Direct C semantic exporter erasure

**Date**: 2026-05-15
**Task**: direct-c-semantic-exporter-erasure
**Branch**: `main`

### Summary

Fail-closed the production/default runtime-callable direct C semantic exporter
surface for RVV microkernel, scalar microkernel, and RVV+scalar dispatch source
routes. The round did not add a replacement generator or Common EmitC route.

### Main Changes

- RVV and scalar selected emission readiness/plans now report unsupported
  deleted-route diagnostics for explicit direct C source routes.
- RVV/scalar/dispatch target exporter and translate registration paths no
  longer publish direct source/object/self-check route authority.
- Target artifact registration and source frontdoor validation now tolerate
  unsupported diagnostics without requiring supported-route capability fields.
- Focused target/scalar/plugin tests were rewritten from generated C/intrinsic
  positive output to absence/fail-closed behavior.
- Spec wording in the touched runtime/RVV/testing layers now records the
  deletion gap instead of treating direct source printers as EmitC authority.

### Testing

- [OK] Focused build: `tcrv-translate`, `tcrv-opt`, `TianChenRVTarget`,
  `TianChenRVRVVPlugin`, `TianChenRVScalarPlugin`, `TianChenRVPlugin`.
- [OK] C++: `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter:
  `target-source-artifact-routes|rvv-microkernel-pipeline|rvv-scalar-i32-vadd-dispatch-generic-route|scalar-target-source-artifact-routes`
  passed 4/4.
- [FAIL] `check-tianchenrv`: 133 passed, 93 failed. Failures are stale direct
  C positive lit/e2e expectations and missing future Common EmitC source
  architecture, not a compile failure.

### Status

[OPEN] **In progress**. Do not archive yet.

### Next Steps

- Continue deletion by removing or rewriting the remaining positive
  RVVMicrokernel/RVVScalarDispatch/source/object/bundle/e2e tests that still
  expect `runtime-callable-c-source`, direct translate options, RVV intrinsic
  C bodies, or direct source bundles.


## Session 67: Descriptor erasure: Offload descriptor-only route deletion

**Date**: 2026-05-15
**Task**: Descriptor Erasure Owner: Offload descriptor-only route deletion
**Branch**: `main`

### Summary

Deleted the Offload runtime descriptor-only target artifact route as a
deletion-only Wrong Logic Deletion Campaign round. Offload still has
capability-gated proposal, legality, and selected lowering-boundary metadata,
but emission planning now fails closed as unsupported until a future Offload
runtime C ABI route is rebuilt.

### Main Changes

- Created and started Trellis task
  `05-15-offload-descriptor-only-route-deletion` with deletion-only PRD and
  spec context.
- Removed the Offload runtime descriptor exporter header, implementation, CMake
  target, built-in target exporter bundle hook, and built-in route metadata
  requirement.
- Removed active Offload descriptor route/artifact getters and selected-plan
  metadata from the Offload plugin.
- Rewrote Offload emission-plan tests to expect unsupported/fail-closed
  diagnostics with no route id, artifact kind, runtime ABI parameter mirror, or
  selected-plan export metadata.
- Deleted descriptor artifact and descriptor bundle lit fixtures/tests that
  only protected the old route.
- Rewrote specs so Offload descriptor output is no longer described as active
  architecture; current Offload artifact/export behavior is unsupported and
  fail-closed.

### Testing

- [OK] CMake reconfigure/build for `TianChenRVBuiltinTargetArtifactExporters`,
  `tianchenrv-offload-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, `tcrv-opt`, and `tcrv-translate`.
- [OK] `build/bin/tianchenrv-offload-extension-plugin-test`.
- [OK] `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] Focused lit subset: 7 Offload/planning/export/coherence tests passed.
- [OK] Directory lit subset: 48 tests across `Target/ArtifactExport`,
  `Target/TargetArtifactBundleExport`, `Target/EmissionManifest`,
  `Transforms/ExecutionPlanning`, and `Transforms/ExecutionPlanCoherence`
  passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`: 226 tests
  passed.
- [OK] `git diff --check` and `git diff --cached --check`.
- [OK] Trellis context validation before finish.
- [OK] Bounded ref-scan: no remaining `OffloadRuntimeDescriptor`,
  `registerOffloadRuntimeDescriptor`, `Target/Offload`,
  `TianChenRVOffloadTarget`, `getOffloadDescriptor`,
  `runtime_offload_descriptor_scope`, `descriptor-only`, or
  `runtime-offload-handoff-descriptor` hits under active code/spec/tests.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 66: RVV binary descriptor authority deletion

**Date**: 2026-05-14
**Task**: RVV binary descriptor authority deletion
**Branch**: `main`

### Summary

Executed a Wrong Logic Deletion Campaign round to erase RVV binary descriptor-era compute and route authority before any selected-config rebuild work continues.

### Main Changes

- Repaired the superseded capability-gated selected-config task into deletion-only task `05-14-rvv-binary-descriptor-authority-deletion`.
- Reverted stale uncommitted selected-config feature work before implementation.
- Replaced `RVVBinaryFamilyRegistry` / `RVVBinaryDescriptor` surfaces with non-authoritative family/route records and deleted `RVVI32BinaryDescriptor.h`.
- Rewrote descriptor-shaped scalar/RVV dispatch records and I32 registry records away from `*Descriptor` type authority names.
- Renamed RVV descriptor element-count metadata to artifact-local component capacity across runtime length contracts, target exports, scripts, fixtures, and tests.
- Rewrote descriptor-as-legal-input tests into fail-closed stale legacy mirror coverage; no compatibility layer or replacement feature was added.

### Testing

- [OK] Focused build for RVV target/plugin, transforms, `tcrv-opt`, `tcrv-translate`, and changed C++ tests.
- [OK] `tianchenrv-i32-binary-family-registry-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] `tianchenrv-rvv-binary-planning-test`.
- [OK] `tianchenrv-rvv-selected-lowering-boundary-test`.
- [OK] `tianchenrv-rvv-binary-variant-legality-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-scalar-extension-plugin-test`.
- [OK] Focused lit: `Target/RVVMicrokernel`, `Target/RVVScalarDispatch`, `Target/TargetArtifactBundleExport`, `Transforms/VectorToExec`, `Transforms/LinalgToExec`, and RVV script tests, 93/93 passed after self-repair.
- [OK] Bounded ref-scan found no RVV descriptor authority identifiers or `descriptor_element_count`; retained `tcrv_rvv.lowering_descriptor` only as legacy mirror/fail-closed metadata.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | chore(rvv): delete binary descriptor authority |

### Status

[OK] **Completed**

### Next Steps

- Continue deletion campaign on remaining legacy lowering mirror metadata when assigned; do not rebuild selected-config features in this deletion-only task.


## Session 66: RVV LMUL m2 selected-config artifact closure

**Date**: 2026-05-14
**Task**: RVV LMUL m2 selected-config artifact closure
**Branch**: `main`

### Summary

Carried exactly one non-default LMUL selected-config slice, i32m2 vadd,
through the plugin-owned selected config profile, RVV materialization/export
fixtures, RuntimeABI/RVVScalarDispatch e2e mappings, local artifact generation,
and focused `ssh rvv` compile/link/run evidence.

### Main Changes

- Added finite i32m2 SEW/LMUL/tail/mask profile facts to
  `RVVProbeCapabilityFacts`, C++ probe-fact validation, and deterministic
  `TargetCapabilitySet` construction.
- Extended `rvv_probe_to_mlir.py` replay so sanitized successful RVV probe
  artifacts emit i32m2 finite config facts without making them runtime
  AVL/VL authority.
- Added i32m2 vadd direct RVVMicrokernel, source-frontdoor artifact,
  RVVScalarDispatch, and target-artifact-bundle fixtures.
- Added e2e default fixture mappings for i32-vadd + i32m2 in the direct
  microkernel and RVVScalarDispatch runners.
- Kept default i32 vadd/vsub/vmul m1 behavior selected when no explicit
  i32m2 shape selector is present.

### Testing

- [OK] Trellis context validation before finish.
- [OK] Focused build for support, target, RVV/scalar target libraries,
  transforms, `tcrv-opt`, `tcrv-translate`, RuntimeABI callable-plan test,
  target artifact export test, and RVV plugin test.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `rvv_probe_to_mlir.py --self-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Focused lit filter: 17 selected tests passed, including default
  i32 vadd/vsub/vmul regressions and new i32m2 vadd fixtures.
- [OK] Local dry-run generated i32m2 vadd direct microkernel artifacts:
  `artifacts/tmp/rvv_microkernel_e2e/codex-rvv-i32m2-vadd-micro-dry`.
- [OK] Local dry-run generated i32m2 vadd dispatch bundle artifacts:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-rvv-i32m2-vadd-dispatch-bundle-dry`.
- [OK] `ssh rvv` direct RVVMicrokernel i32m2 vadd run:
  `codex-rvv-i32m2-vadd-micro-ssh`, source/object external ABI runs
  succeeded with `tcrv_rvv_microkernel_external_abi_ok`.
- [OK] `ssh rvv` RVVScalarDispatch i32m2 vadd bundle run:
  `codex-rvv-i32m2-vadd-dispatch-bundle-ssh`, source/object bundle callers
  succeeded with `tcrv_rvv_scalar_i32_vadd_bundle_external_abi_ok`.
- [OK] Bounded ref-scan found no new production descriptor-to-C path,
  descriptor-only runtime authority, explicit-only authority, or generic core
  RVV semantic branch; descriptor hits are negative tests or pre-existing
  artifact-capacity comments.
- [OK] `git diff --check`.
- [OK] `git diff --cached --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## Session 64: RVV generated artifact hardware invocation closure

**Date**: 2026-05-14
**Task**: RVV generated artifact hardware invocation closure
**Branch**: `main`

### Summary

Closed the current-head route-consumer proof gap for generated source-frontdoor
i32 vadd/vsub dispatch artifacts. The production C++ RuntimeABI contract path
was already present at `8780c6b`; this round made the generated dispatch
`dispatch_runtime_abi_invocation_contract` first-class e2e evidence and
revalidated local plus `ssh rvv` source-frontdoor invocation.

### Main Changes

- Updated `rvv_scalar_dispatch_e2e.py` to parse and validate the generated
  dispatcher's `dispatch_runtime_abi_invocation_contract` against the bundle
  ABI signature, runtime `n`, `rvv_available` guard, and production owner.
- Recorded the validated dispatch invocation contract in bundle-mode and
  direct e2e evidence JSON.
- Added self-test fail-closed coverage for stale dispatch runtime ABI name,
  stale runtime element-count C name, and stale dispatch guard C name.
- Tightened vadd/vsub source-frontdoor bundle FileCheck coverage for generated
  source/header dispatch invocation contract comments and script evidence.

### Testing

- [OK] Focused CMake build for support, target, RVV/scalar target, `tcrv-opt`,
  `tcrv-translate`, and focused test binaries.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] Focused lit for vadd/vsub source-frontdoor bundle artifacts and script
  e2e (`3/3`).
- [OK] Focused lit for RuntimeABI callable plan, RVVScalarDispatch, and
  RVVMicrokernel (`46/46`).
- [OK] Local source-frontdoor bundle dry-runs:
  `codex-runtimeabi-current-vadd-local` and
  `codex-runtimeabi-current-vsub-local`.
- [OK] `ssh rvv` source-frontdoor i32-vsub bundle invocation:
  `codex-runtimeabi-current-vsub-ssh`, `ssh_evidence_verified=true`.
- [OK] `git diff --check` and Trellis validation before finish.

### Status

[OK] **Completed**

### Next Steps

- Archive task and commit the coherent route-consumer evidence closure.

---

## Session 63: RVV exported artifact runtime consumption

**Date**: 2026-05-14
**Task**: RVV exported artifact runtime consumption
**Branch**: `main`

### Summary

Closed the downstream consumption gap for source-frontdoor RVV generated
artifacts. `RVVScalarDispatch` now validates the embedded RVVMicrokernel
artifact's exported selected-source identity and runtime ABI invocation
contract before dispatch source/header/object export can succeed.

### Main Changes

- Added production C++ validation in `RVVScalarDispatch.cpp` for embedded RVV
  callable source contracts:
  `rvv_microkernel_selected_source_identity`,
  `runtime_abi_invocation_contract`, and runtime length authority.
- Added `dispatch_embedded_rvv_artifact_contract_consumed` comments to
  generated dispatch source/header artifacts.
- Updated `rvv_scalar_dispatch_e2e.py` to parse and record
  `embedded_rvv_artifact_contract` from generated dispatch artifacts as a
  runner-only consumer.
- Tightened vector dynamic vadd/vsub bundle FileCheck coverage around embedded
  RVVMicrokernel selected-source and runtime ABI contracts.

### Testing

- [OK] Focused C++/tool build for RVV target/export/plugin and `tcrv-opt` /
  `tcrv-translate`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit for dynamic vector vadd/vsub bundle artifacts.
- [OK] Focused lit for `Target/RVVScalarDispatch`, `Target/RVVMicrokernel`, and
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test` (`46/46`).
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] Local generated bundle dry-runs for vector i32-vadd and i32-vsub.
- [OK] `ssh rvv` generated bundle invocation for vector i32-vsub:
  `codex-exported-contract-vsub-ssh`, `ssh_evidence_verified=true`.
- [OK] `git diff --check`, `git diff --cached --check`, Trellis validation.

### Status

[OK] **Completed**

### Next Steps

- None - task complete

## Session 63: RVV RuntimeABI generated artifact invocation closure

**Date**: 2026-05-14
**Task**: RVV RuntimeABI generated artifact invocation closure
**Branch**: `main`

### Summary

Added support-layer generated RuntimeABI invocation contracts, wired RVVMicrokernel and RVVScalarDispatch to consume them, validated focused local and ssh rvv source-frontdoor evidence.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete

---

## Session 65: RVV ssh-rvv generated artifact execution closure

**Date**: 2026-05-14
**Task**: RVV ssh-rvv generated artifact execution closure
**Branch**: `main`

### Summary

Revalidated current HEAD `31f9955` for the generated source-frontdoor i32
vadd/vsub dispatch artifact path. No compiler/source change was required:
`RuntimeABICallablePlan`, `RVVMicrokernel`, and `RVVScalarDispatch` already
own/consume the generated invocation contract. This round refreshed local
generated bundle evidence for vadd/vsub and closed a fresh `ssh rvv` clang
compile/link/run leg for generated vsub bundle artifacts.

### Main Changes

- Created Trellis task
  `.trellis/tasks/05-14-rvv-ssh-rvv-generated-artifact-execution-closure`.
- Wrote PRD/evidence notes for the current-head route and corrected the stale
  brief assumption that no ssh evidence existed at this head.
- No production C++, Python runner, lit test, spec, or compiler source files
  were changed.

### Testing

- [OK] Focused build for support, target, RVV/scalar target, `tcrv-opt`,
  `tcrv-translate`, and focused test binaries.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] Focused lit filter for script bundle e2e, RuntimeABI callable plan,
  RVVScalarDispatch, RVVMicrokernel, and vector dynamic vadd/vsub bundle
  export: `49/49` selected tests passed.
- [OK] Local generated source-frontdoor bundle dry-runs:
  `codex-sshclosure-vadd-local` and `codex-sshclosure-vsub-local`.
- [OK] `ssh rvv` generated source-frontdoor i32-vsub bundle invocation:
  `codex-sshclosure-vsub-ssh`, `ssh_evidence_verified=true`.
  Remote source-built and bundle-object caller runs both printed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- [OK] Bounded source diff/ref-scan confirmed no production/source diff under
  `include`, `lib`, `scripts`, `test`, `tools`, `cmake`, or `CMakeLists.txt`.
- [OK] `git diff --check` and Trellis validation.

### Status

[OK] **Completed**

### Next Steps

- Archive task and commit the evidence closure.

---

## Session 66: RVV plugin-owned binary op-family construction contract

**Date**: 2026-05-14
**Task**: RVV plugin-owned binary op-family construction contract
**Branch**: `main`

### Summary

Converted the dynamic vector/SCF source-frontdoor from the previous vadd/vsub
bounded slice into a reusable RVV binary op-family admission path for
registry-declared i32-vmul. The new i32-vmul route now reaches the same
selected source identity, selected config, EmitC intrinsic mapping, artifact
bundle, RuntimeABI callable contract, RVVScalarDispatch validation, local
bundle evidence, and ssh-rvv execution evidence as vadd/vsub.

### Main Changes

- Added `mlir-vector-scf-runtime-i32-vmul.v1` as the bounded dynamic vector
  source kind and wired it into the i32-vmul RVV binary family descriptor.
- Kept source-frontdoor family admission registry-backed; no descriptor-to-C
  production path, Python compiler semantics, or core RVV semantic branch was
  added.
- Added vector/SCF i32-vmul transform and target artifact bundle tests.
- Extended the RVV scalar dispatch evidence runner and script lit coverage for
  the i32-vmul vector source-frontdoor bundle path.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` for the
  dynamic vector source-tail authority contract.

### Testing

- [OK] Focused build for support, target, RVV/scalar target, transforms,
  `tcrv-opt`, `tcrv-translate`, and focused test binaries.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused vector/source-frontdoor and bundle lit filter: `7/7` selected
  tests passed.
- [OK] Script bundle lit filter: `1/1` selected tests passed.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Local source-frontdoor bundle dry-runs:
  `codex-opfamily-vadd-vector-local`,
  `codex-opfamily-vsub-vector-local`,
  `codex-opfamily-vmul-vector-local`.
- [OK] `ssh rvv` source-frontdoor i32-vmul bundle invocation:
  `codex-opfamily-vmul-vector-ssh-rvv`, `ssh_evidence_verified=true`,
  selected kernel `frontend_vector_dynamic_bundle_i32_vmul`.

### Status

[OK] **Completed**

### Next Steps

- Archive task and commit the op-family contract change.

---

## Session 67: RVV vmul generated artifact ssh-rvv closure

**Date**: 2026-05-14
**Task**: RVV vmul generated artifact ssh-rvv closure
**Branch**: `main`

### Summary

Closed the remaining public vector/SCF vmul production-entry asymmetry and
refreshed current-head generated-artifact evidence. Current code already routed
vmul through the shared RVV binary family registry, selected boundary,
materialization, selected emission, RVVMicrokernel, RuntimeABI invocation
contracts, target artifact bundle export, and RVVScalarDispatch. This round
added the missing explicit vector/SCF vmul `tcrv-opt` adapter and direct
RVVMicrokernel runner entry, then validated vmul locally and on `ssh rvv`.

### Main Changes

- Added `--tcrv-lower-vector-rvv-i32-vmul-to-exec` as a bounded explicit
  vector/SCF adapter that reuses the registry-backed source-frontdoor path and
  only admits the existing `i32-vmul` family.
- Registered the new pass in `tcrv-opt`.
- Extended `rvv_microkernel_e2e.py` with
  `--lower-vector-i32-vmul-frontend`.
- Added explicit-vmul transform lit and direct RVVMicrokernel bundle script
  lit coverage.
- Updated the lowering-runtime spec so the dynamic vector source-tail authority
  contract names the vmul explicit adapter alias.

### Testing

- [OK] Focused build for support, target, RVV/scalar target, RVV plugin,
  transforms, `tcrv-opt`, `tcrv-translate`, and focused test binaries.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Focused lit filter covering vector dynamic vadd/vsub/vmul, explicit
  vmul fail-closed coverage, target artifact bundle export, RVVMicrokernel,
  RVVScalarDispatch, and script e2e: 14 selected tests passed.
- [OK] Local direct RVVMicrokernel bundle dry-runs for vector/SCF vadd, vsub,
  and vmul.
- [OK] Local RVVScalarDispatch bundle dry-runs for vector/SCF vadd, vsub, and
  vmul.
- [OK] `ssh rvv` direct RVVMicrokernel bundle execution for vector/SCF i32-vmul:
  `codex-vmul-explicit-vector-micro-ssh-rvv`; source-built and bundle-object
  caller runs both printed
  `tcrv_rvv_i32_vmul_microkernel_external_abi_ok counts=7,16,23`.
- [OK] `ssh rvv` RVVScalarDispatch bundle execution for vector/SCF i32-vmul:
  `codex-vmul-explicit-vector-dispatch-ssh-rvv`,
  `ssh_evidence_verified=true`; source-built and bundle-object caller runs both
  printed
  `tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- [OK] Bounded ref-scan found no new descriptor-to-C, descriptor-only,
  explicit-only, or generic core RVV semantic branch authority.
- [OK] `git diff --check`.

### Status

[OK] **Completed and archived; commit pending**

### Next Steps

- Create one coherent commit.


## Session 64: RVV vmul dispatch ssh-rvv closure refresh

**Date**: 2026-05-14
**Task**: RVV vmul dispatch ssh-rvv closure refresh
**Branch**: `main`

### Summary

Refreshed current-HEAD vmul generated artifact dispatch closure: no production source gap found, focused build/C++/lit/local bundle evidence passed, and ssh rvv direct microkernel plus RVVScalarDispatch vmul runs succeeded.

### Main Changes

- Created and archived Trellis task
  `05-14-rvv-vmul-dispatch-ssh-rvv-closure` from the Hermes brief.
- Reconciled the brief against current HEAD and found no missing production
  source owner: `RuntimeABICallablePlan`, `RVVScalarDispatch`,
  `RVVMicrokernel`, `TargetArtifactExport`, the RVV binary family registry,
  source-frontdoor pass/tool exposure, and the e2e runners already carry
  `i32-vmul` through the shared vadd/vsub family contracts.
- Refreshed local generated-artifact bundle evidence for direct
  RVVMicrokernel and RVVScalarDispatch vmul, plus vadd/vsub regressions.
- Refreshed focused `ssh rvv` evidence for generated vector/SCF i32-vmul:
  both direct RVVMicrokernel and RVVScalarDispatch source-built and
  bundle-object caller runs succeeded.
- No C++/MLIR/TableGen/CMake/script production source changes were needed in
  this round; tracked changes are Trellis task/archive and workspace journal
  records for the current-HEAD validation closure.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | (see git log) |

### Testing

- [OK] Trellis context validation before and after archive.
- [OK] Focused build for support, target, RVV/scalar target libraries, RVV
  plugin, transforms, `tcrv-opt`, `tcrv-translate`, RuntimeABI callable-plan
  test, target artifact export test, and RVV plugin test.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Focused lit filter: 14 selected tests passed.
- [OK] Local RVVMicrokernel bundle dry-runs for vector/SCF vadd, vsub, and
  vmul; vmul source SHA256
  `3120d67ddd9850fbee47d0aafae34c8e30ef5dab0537040bba36176687b4411d`.
- [OK] Local RVVScalarDispatch bundle dry-runs for vector/SCF vadd, vsub, and
  vmul; vmul source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- [OK] `ssh rvv` direct RVVMicrokernel generated vmul run:
  `tcrv_rvv_i32_vmul_microkernel_external_abi_ok counts=7,16,23`.
- [OK] `ssh rvv` RVVScalarDispatch generated vmul run:
  `tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- [OK] Bounded ref-scan found no new tracked production descriptor-only,
  descriptor-to-C, explicit-only, stale-vadd-default, or generic core RVV
  semantic-branch authority.
- [OK] `git diff --check`.
- [OK] `git diff --cached --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 65: RVV selected config profile contract

**Date**: 2026-05-14
**Task**: RVV selected config profile contract
**Branch**: `main`

### Summary

Promoted RVV selected config into a plugin-owned profile consumed by selected emission planning, RVVMicrokernel artifact export, TargetArtifactExport validation, and RVVScalarDispatch for existing i32 vadd/vsub/vmul.

### Main Changes

- Created, completed, and archived Trellis task `05-14-rvv-selected-config-profile-contract`.
- Added `RVVBinarySelectedConfigProfile` metadata for hardware facts, compile-time variant config, and runtime AVL/VL/source-frontdoor dynamic extent roles.
- Wired the profile into RVV selected emission planning, RVVMicrokernel source/object evidence, TargetArtifactExport route validation, and RVVScalarDispatch bundle metadata validation.
- Kept descriptor-local `element_count` quarantined as bounded artifact capacity metadata, not selected config or runtime trip-count authority.
- Added fail-closed coverage for stale descriptor-only profile variant metadata and missing runtime-role profile metadata before artifact/runtime claims.
- Extended vadd/vsub/vmul lit checks so selected profile metadata reaches the selected boundary, bundle/export route, generated source comments, and dispatch validation.

### Testing

- [OK] Trellis context validation before finish and after archive.
- [OK] Focused C++/MLIR build for support, target, RVV/scalar target libraries, RVV plugin, transforms, `tcrv-opt`, `tcrv-translate`, RuntimeABI callable-plan test, target artifact export test, and RVV plugin test.
- [OK] `tianchenrv-runtime-abi-callable-plan-test`.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit filter: 22 selected tests passed.
- [OK] `rvv_microkernel_e2e.py --self-test`.
- [OK] `rvv_scalar_dispatch_e2e.py --self-test`.
- [OK] Local RVVMicrokernel generated bundle dry-runs for vector/SCF vadd, vsub, and vmul; vmul source SHA256 `710df3874a679c4307e55f558dda38f0d8c9cafe0b29aa6a6a373220ec7cd035`.
- [OK] Local RVVScalarDispatch generated bundle dry-runs for vector/SCF vadd, vsub, and vmul; vmul source SHA256 `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- [OK] `ssh rvv` direct RVVMicrokernel generated vmul run: `selected-config-profile-micro-vmul-ssh-rvv`.
- [OK] `ssh rvv` RVVScalarDispatch generated vmul run: `selected-config-profile-dispatch-vmul-ssh-rvv`, `ssh_evidence_verified=true`.
- [OK] Artifact inspection confirmed selected profile metadata in generated bundle source/index and dispatch bundle metadata.
- [OK] Bounded ref-scan found no new production descriptor-to-C, descriptor-only runtime authority, explicit-only authority, or generic core RVV semantic branch.
- [OK] `git diff --check`.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 66: Descriptor erasure: lowering descriptor metadata chain deletion

**Date**: 2026-05-15
**Task**: Descriptor erasure: lowering descriptor metadata chain deletion
**Branch**: `main`

### Summary

Deleted RVV/scalar lowering_descriptor and selected_lowering_descriptor metadata chain, rewrote typed-source/EmitC route tests, removed obsolete descriptor fixtures, validated with focused C++ tests and check-tianchenrv.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 67: Core Semantic Branch Erasure: RVV source-to-exec pass family deletion

**Date**: 2026-05-15
**Task**: Core Semantic Branch Erasure: delete RVV source-to-exec pass family
**Branch**: `main`

### Summary

Deleted the core RVV source-to-exec semantic pass family and removed the public
tool/frontdoor paths that let core transforms recognize finite RVV
linalg/vector source semantics and materialize `tcrv.exec`.

### Main Changes

- Created Trellis task `05-15-core-rvv-source-to-exec-erasure`.
- Deleted `lib/Transforms/LowerSourceRVVBinaryToExec.cpp` and removed it from
  `TianChenRVTransforms`.
- Removed `Passes.h` declarations, `Passes.td` definitions, and `tcrv-opt`
  registrations for `tcrv-lower-source-rvv-binary-to-exec`, the RVV/linalg
  pass, the old linalg i32 compatibility aliases, and the vector i32 add/sub/mul
  source adapter aliases.
- Removed `tcrv-translate` plan-and-export bundle pre-pass source lowering.
- Deleted old LinalgToExec/VectorToExec lit tests and the stale
  plan-linalg marker-mismatch bundle test; added deleted-route coverage proving
  removed pass names are unregistered and do not create `tcrv.exec.kernel`.
- Rewrote stale spec/README/support comments so high-level frontend rebuild is
  future plugin/interface-owned work, not a core RVV source branch.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVTransforms tcrv-opt tcrv-translate -j2`.
- [OK] Focused lit:
  `source-rvv-pass-family-deleted`.
- [OK] Focused lit:
  `plan-and-export-target-artifact-bundle-no-viable`.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`: 127/127 passed.
- [OK] Bounded ref-scan found no remaining RVV family registry imports/lookups
  or deleted pass-name registrations in core transform/tool code.
- [OK] `git diff --check`.
- [OK] Trellis context validation.

### Status

[OK] **Completed; archive and commit pending**

### Next Steps

- Archive the Trellis task with `--no-commit`, run cached diff/Trellis
  validation, then create one coherent commit.


## Session 68: Descriptor Erasure: RVV selected microkernel descriptor materialization deletion

**Date**: 2026-05-15
**Task**: Descriptor Erasure: RVV selected microkernel descriptor materialization deletion
**Branch**: `main`

### Summary

Deleted the RVV plugin selected microkernel descriptor materializer and
fail-closed selected emission planning/body verification so selected RVV
planning no longer creates RVV microkernel/dataflow ops or callable ABI metadata
from `selectedPlan.descriptor`.

### Main Changes

- Created and archived Trellis task
  `05-15-rvv-selected-microkernel-descriptor-materialization-deletion`.
- Deleted `RVVBinaryMicrokernelMaterialization` header/implementation and
  removed the RVV plugin CMake source entry.
- Rewired `RVVBinarySelectedLoweringBoundary` to emit only unsupported
  metadata-only `tcrv_rvv.lowering_boundary` records, with no auto-generated
  `tcrv_rvv.*_microkernel`, `setvl`, load/arithmetic/store body, or callable
  ABI mem-window/param materialization.
- Replaced selected emission planning with a fail-closed public surface; no
  selected emission plan/readiness/variant emission plan is built from
  `selectedPlan.descriptor`.
- Removed the selected-plan `RVVBinaryIntrinsicRoute descriptor` mirror and
  made selected-plan route/runtime ABI/intrinsic accessors fail closed.
- Replaced the RVV binary microkernel body verifier implementation with a
  fail-closed error instead of descriptor/config-driven body validation.
- Deleted stale descriptor/body-verifier negative lit tests and rewrote
  plugin, variant, probe, manifest, dispatch, and artifact-export checks to
  assert unsupported RVV metadata-only behavior.
- Updated RVV plugin, emission-runtime, and MLIR testing specs to record the
  deletion boundary and future extension-family IR + EmitC rebuild requirement.

### Testing

- [OK] `cmake --build build --target TianChenRVRVVPlugin TianChenRVRVVTarget -j2`
- [OK] `cmake --build build --target tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-binary-planning-test`
- [OK] `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 125/125 passed.
- [OK] Bounded ref-scan: no code/test hits for `selectedPlan.descriptor`,
  `plan.descriptor`, `buildRVVBinaryVLDataflowMaterialization`,
  `materializeRVVBinaryMicrokernelOp`, or
  `buildRVVBinaryCallableRuntimeABIParameters`.
- [OK] Residual scan classified active finite family/route registry and target
  `RVVMicrokernel.cpp` direct artifact/export code as next deletion owners.
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis validation before finish and after archive.

### Status

[OK] **Archived; commit pending**

### Next Steps

- Continue deletion campaign on remaining target-side finite
  `RVVBinaryFamilyRecord` / `RVVBinaryIntrinsicRoute` registry and direct
  `RVVMicrokernel.cpp` artifact/export authority.
