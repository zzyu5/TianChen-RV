# RVV Runtime ABI Production Invocation

## Goal

Promote the generated RVV `i32-vadd` / `i32-vsub` runtime ABI closure from
script-level invocation evidence into the production C++ compiler/runtime
handoff. The default selected RVV artifact/export path must own the callable
ABI signature, selected runtime length roles, selected vector config, and
dispatch or generated-harness invocation contract before generated source,
header, object, or bundle artifacts are considered complete.

## Why Now

`c3cc4e6` closed useful generated-artifact runtime ABI evidence, but its
production impact was limited to task metadata, journal state, the
`rvv_microkernel_e2e.py` harness, and script lit coverage. That proves the
script can call compiler-produced artifacts, but it does not make
`TargetArtifactExport`, `RVVMicrokernel`, `RVVScalarDispatch`, or the selected
runtime-length contract the ABI authority. The next bottleneck is therefore the
production C++ boundary: generated artifact signatures and dispatch/harness
invocation must be derived from the same compiler-owned callable ABI contract.

## Scope

- Existing generated RVV `i32-vadd` and `i32-vsub` selected routes only.
- `TargetArtifactExport` candidate and bundle preflight for runtime ABI
  signature ownership.
- `RVVMicrokernel` source/header/object emission for generated callable
  signatures, selected runtime AVL/VL authority, selected config, and op-family
  identity.
- `RVVScalarDispatch` or generated harness invocation of the same
  compiler-owned ABI, including the runtime element-count role and dispatch
  availability guard where applicable.
- Existing e2e scripts and script tests only as consumers of the production
  ABI, not as the source of ABI truth.
- Focused local and `ssh rvv` evidence for at least one generated artifact
  invocation, or an exact blocker after local generation/compile succeeds.

## Non-Goals

- No new dtype, i64 expansion, LMUL matrix, third operation, broad family
  matrix, broad smoke suite, or performance tuning.
- No script-only, test-only, artifact-format-only, metadata-only,
  FileCheck-only, negative-test-only, report-only, workspace-journal, or
  helper-only milestone.
- No hand-written surrogate kernels as proof of compiler output.
- No descriptor-to-C production exporter.
- No descriptor element count or descriptor vector shape as compute, config,
  runtime AVL/VL, ABI, or dispatch authority.
- No moving computation semantics into `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No GCC or vendor compiler as the default route.
- No Template/Toy/TensorExtLite/IME/Offload work except narrow regressions
  caused by shared validation.
- No runtime, correctness, or performance claims beyond focused evidence
  actually run.

## Requirements

- Generated RVV source, header, object, and bundle artifacts expose a stable
  callable ABI carrying lhs/rhs/out buffers plus the runtime element-count
  role derived from `tcrv.exec.mem_window` / `tcrv.exec.runtime_param`.
- `TargetArtifactExport` and bundle export validate that candidate
  `runtime_abi_parameters` exactly mirror the IR-backed ABI and route-local
  required roles before any source/header/object/bundle output.
- `RVVMicrokernel` emits vadd/vsub prototypes, source bodies, object evidence,
  and bundle metadata from the same selected config and runtime length
  contracts.
- `RVVScalarDispatch` or a generated harness invocation consumes the same
  production ABI contract, including runtime element-count and dispatch guard
  roles when dispatch is selected.
- The e2e script must parse and invoke compiler-produced artifacts as a harness
  consumer. It must not invent ABI parameter names, function signatures, family
  identity, or runtime length semantics that production C++ has not emitted.
- Selected config and runtime length data must be preserved from IR through
  artifact ABI, generated C, object evidence, bundle index, and dispatch or
  harness invocation.
- Missing, stale, spoofed, or mismatched ABI/runtime-role data must fail closed
  before generated artifact or invocation success.
- Descriptor-only compute/config/runtime authority remains quarantined.
- Generated C remains clang/LLVM-compatible and uses RVV C intrinsics through
  `riscv_vector.h`.
- Generic core orchestration remains target-neutral; RVV-specific ABI and
  dispatch semantics stay in RVV target/export owners.

## Acceptance Criteria

- [x] vadd and vsub generated source/header prototypes are derived from a
      production C++ runtime ABI contract, not from script-local assumptions.
- [x] Direct RVV source/header/object routes require the same ordered ABI
      roles for lhs, rhs, out, and runtime element count.
- [x] RVV+scalar dispatch source/header/object routes require the same ordered
      callable ABI plus the explicit dispatch-availability guard role.
- [x] Bundle records for generated source/header/object components carry
      matching runtime ABI kind/name, external ABI identity, ordered
      `runtime_abi_parameter` records, selected config, and runtime AVL/VL
      metadata.
- [x] `TargetArtifactExport` fails closed on missing runtime length role data,
      stale ABI signatures, vadd/vsub ABI mismatch, descriptor-only production
      attempts, missing EmitC body mapping, stale op identity, and
      missing/conflicting selected config.
- [x] `RVVScalarDispatch` or the generated harness invocation calls the
      compiler-produced artifacts through the emitted production ABI.
- [x] The e2e script remains a consumer of emitted header/source/object
      artifacts and no longer needs independent ABI truth beyond parsing
      compiler-emitted data.
- [x] Focused lit/FileCheck and C++ tests cover vadd and vsub generated
      artifact signatures, dispatch or generated harness invocation, selected
      config/runtime length preservation, and fail-closed ABI mismatch cases.
- [x] Focused `ssh rvv` compile/run evidence is produced for at least one
      generated artifact invocation, or an exact blocker is recorded after
      local generation and compile.
- [x] Bounded ref-scan confirms descriptor-only authority remains quarantined
      and generic core passes gained no RVV semantic branches.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean worktree, and one coherent
      commit complete the round if the task is finished.

## Implementation Summary

- `RuntimeABICallablePlan` now rejects runtime ABI mirror metadata that has the
  right roles but not the IR-backed callable ABI order.
- `TargetArtifactExport` now rejects candidate runtime ABI signatures whose
  route-local required parameters are present but reordered.
- `RVVMicrokernel` emits `runtime_abi_invocation_contract` comments in source
  and header artifacts, and object artifacts now carry
  `runtime_abi_invocation_contract`,
  `runtime_abi_callable_symbol`, `runtime_abi_ordered_roles`, and
  `runtime_abi_production_owner` evidence.
- `RVVScalarDispatch` emits ordered dispatch ABI parameters, callable ABI, and
  `dispatch_runtime_abi_invocation_contract` in source and header artifacts,
  including the explicit dispatch-availability guard role.
- `scripts/rvv_microkernel_e2e.py` now parses the compiler-emitted production
  invocation contract, checks callable symbol/runtime ABI/runtime glue/ordered
  roles/runtime element-count name, records it in evidence, and fails closed
  when the contract is missing.
- Focused lit/C++ tests were updated for production-owned vadd/vsub source,
  header, object, dispatch, script consumer, ordered ABI, and fail-closed
  cases.

## Validation Summary

- Build:
  `cmake --build build --target tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- C++ smoke:
  `build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `build/bin/tianchenrv-target-artifact-export-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`
- Script self-test:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='rvv-microkernel-runtime-abi-role-binding|rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding|rvv-microkernel-e2e|rvv-microkernel-auto-materialization|rvv-microkernel-family-sub|rvv-microkernel-object|rvv-scalar-i32-vadd-dispatch-c|rvv-microkernel-bundle-e2e' .`
  from `build/test`; result: 9 passed.
- RVV runtime evidence:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --run-id codex-prod-abi-vsub-ssh --overwrite --timeout 120`;
  result: `ssh_evidence=true`, source-linked and object-linked external
  caller runs both printed
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16`.
  Artifact root:
  `artifacts/tmp/rvv_microkernel_e2e/codex-prod-abi-vsub-ssh`.
- Ref-scan:
  `rg -n "descriptor.*author|descriptor-to-C|descriptor_compute_authority|runtime_abi_invocation_contract|dispatch_runtime_abi_invocation_contract|hasRVV|hasIME|hasSophgo" lib/Support lib/Target scripts/rvv_microkernel_e2e.py test/Target/RVVMicrokernel test/Target/RVVScalarDispatch test/Scripts/rvv-microkernel-e2e.test`;
  changed files remain in support/target RVV/export/script/test surfaces, with
  no generic core pass edits.
- Whitespace:
  `git diff --check` passed.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/architecture/design-boundaries.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-14-rvv-generated-artifact-runtime-abi-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-14-rvv-emitc-artifact-consumer-closure/prd.md`

Primary production surfaces:

- `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
- `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
- `lib/Target/RVV/RVVMicrokernel.cpp`
- `lib/Target/RVV/RVVScalarDispatch.cpp`
- `lib/Target/TargetArtifactExport.cpp`
- `include/TianChenRV/Target/TargetArtifactExport.h`
- `tools/tcrv-translate/tcrv-translate.cpp`

Consumer harness and regression surfaces:

- `scripts/rvv_microkernel_e2e.py`
- `test/Scripts/rvv-microkernel-e2e.test`
- `test/Scripts/rvv-microkernel-bundle-e2e.test`
- Focused TargetArtifactExport, TargetArtifactBundleExport, RVVMicrokernel,
  and RVVScalarDispatch vadd/vsub tests.

## Evidence Plan

- Build touched RVV target/export/plugin/tool libraries and executables:
  generated headers, RVV plugin/target libraries,
  `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, `tcrv-translate`, and
  affected target artifact tests.
- Run focused C++ tests for target artifact export and RVV extension/plugin
  behavior when applicable.
- Run focused lit/FileCheck filters for:
  `TargetArtifactExport`, `TargetArtifactBundleExport`, `RVVMicrokernel`,
  `RVVScalarDispatch`, selected-route quarantine, selected EmitC body/artifact
  consumer tests, runtime-length/config consumption, object artifacts, and both
  RVV e2e script tests.
- Run the e2e script in local dry-run mode for vadd/vsub when script consumer
  behavior changes.
- Run focused `ssh rvv` compile/run evidence for at least one generated
  artifact invocation unless blocked by a precise remote/toolchain failure.
- Run bounded ref-scans for descriptor-only authority and core-pass neutrality.
- Run Trellis validation before finish and after archive.

## Definition Of Done

The finite generated RVV `i32-vadd` / `i32-vsub` artifact path has a production
runtime ABI closure: C++ target/export owners derive and validate generated
artifact signatures and dispatch or harness invocation from IR-backed ABI
roles, scripts only consume emitted artifacts, descriptor-only authority
remains quarantined, focused local and RVV evidence are recorded, the task is
archived, and one coherent commit records the round.
