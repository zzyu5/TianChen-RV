# RVV op-family runtime ABI dispatch parity

## Goal

Make the migrated RVV `i32-vadd` and `i32-vsub` runtime ABI dispatch path
op-family complete. Both selected routes must use the same production
`RuntimeABICallablePlan`, generated artifact ABI, selected config/runtime-length
authority, and `RVVScalarDispatch` invocation path without accidental
vadd-only or descriptor-owned semantics.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- `main` is clean at `31cb584 feat(rvv): own runtime abi invocation in production`.
- There was no active `.trellis/.current-task` when this round started, so this
  task was created from the Hermes direction brief.
- Prior PRD `05-14-rvv-runtime-abi-production-invocation` moved generated RVV
  artifact runtime ABI invocation into production owners:
  `RuntimeABICallablePlan`, `RVVMicrokernel`, `RVVScalarDispatch`,
  `TargetArtifactExport`, and the RVV e2e harness.
- Prior PRD `05-14-rvv-generated-artifact-runtime-abi-closure` closed direct
  generated artifact invocation for `i32-vadd` and `i32-vsub`, including
  focused `ssh rvv` evidence.
- Current code already has finite family registration tables for `i32-vadd`,
  `i32-vsub`, and `i32-vmul`; `RVVScalarDispatch` mostly resolves dispatch
  identity from selected component metadata, then checks route/ABI registration
  mirrors.
- Current `RuntimeABIContract` still has compatibility helper APIs that derive
  i32 binary ABI parameters, role requirements, dispatch guard parameters, and
  runtime-param specs from `getI32BinaryRuntimeABIContract("i32-vadd")`.
- Current vsub scalar-dispatch positive route coverage exists, but scalar
  dispatch runtime ABI role-binding negative coverage is centered on the vadd
  helper fixture.

## Requirements

- The production finite i32 runtime ABI helper surface must admit explicit
  selected op-family identity for `i32-vadd` and `i32-vsub`.
- Vadd-only helper defaults may remain only as explicit compatibility wrappers;
  new production-facing helper paths must take a selected family contract or
  selected family id.
- `RuntimeABICallablePlan` and route candidate validation must continue to
  validate ordered ABI roles, C names, C types, and ownership against the
  IR-backed callable ABI plan.
- `RVVScalarDispatch` must keep deriving dispatch identity from selected RVV
  and scalar component metadata, then fail closed when RVV/scalar selected
  families, route ids, runtime ABI names, runtime length roles, or selected
  config metadata disagree.
- Generated source/header/object/bundle artifacts for `i32-vadd` and
  `i32-vsub` must expose matching runtime ABI signatures and selected runtime
  length/config evidence.
- Vsub scalar-dispatch coverage must exercise the same runtime ABI role-binding
  and dispatch invocation failure modes as vadd where this round touches that
  path.
- Descriptor-only compute/config/runtime authority remains quarantined; legacy
  descriptors may only be checked as bounded mirror metadata after typed
  selected-plan authority is established.
- Generic core orchestration must remain target-neutral. RVV-specific family,
  ABI, and dispatch behavior stays in RVV/scalar target/plugin/export owners.

## Acceptance Criteria

- [x] `RuntimeABIContract` exposes selected-family-aware finite i32 helper APIs
      for callable parameters, role requirements, runtime length specs,
      dispatch runtime specs, dispatch runtime parameters, and callable
      parameter role binding.
- [x] Existing vadd compatibility helpers are explicitly documented as
      compatibility defaults and remain behavior-preserving for existing call
      sites.
- [x] Vsub candidate/test construction no longer relies on vadd-only helper
      names where a selected family contract is available.
- [x] Focused scalar-dispatch role-binding coverage proves `i32-vsub` uses the
      selected runtime ABI data and fails closed on missing runtime length role,
      stale selected runtime-count metadata, duplicate callable role, wrong C
      type, wrong ownership, unknown runtime role, bad dispatch guard type, and
      detached dispatch ABI metadata.
- [x] Positive vadd and vsub dispatch artifact checks still show ordered roles
      `lhs-input-buffer -> rhs-input-buffer -> output-buffer ->
      runtime-element-count -> dispatch-availability-guard`.
- [x] Generated vsub dispatch source/header/object or bundle evidence still
      calls subtraction semantics and not vadd semantics.
- [x] Bounded ref-scan identifies only explicit compatibility wrappers or
      registration records as remaining vadd-specific code in the touched ABI
      and dispatch surfaces.
- [x] Focused build, C++ tests, lit/FileCheck tests, `git diff --check`,
      Trellis validation before finish and after archive, and final clean
      worktree all pass.
- [x] Focused `ssh rvv` compile/run evidence is produced for the generated
      vsub dispatch artifact invocation, or an exact non-code blocker is
      recorded after local generation/compile succeeds.

## Out of Scope

- No new dtype, i64 expansion, LMUL matrix expansion, third operation, broad
  family matrix, broad smoke suite, or performance tuning.
- No script-only, test-only, artifact-format-only, metadata-only,
  FileCheck-only, negative-test-only, workspace-journal, report-only, or
  helper-only milestone.
- No hand-written surrogate kernels as proof of compiler output.
- No descriptor-to-C production exporter.
- No descriptor element count or descriptor vector shape as compute, config,
  runtime AVL/VL, ABI, or dispatch authority.
- No computation semantics in `tcrv.exec`.
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-runtime-abi-production-invocation/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-generated-artifact-runtime-abi-closure/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Primary implementation surfaces:
  `include/TianChenRV/Support/RuntimeABI.h`,
  `include/TianChenRV/Support/RuntimeABIParam.h`,
  `include/TianChenRV/Support/RuntimeABIMemWindow.h`,
  `include/TianChenRV/Support/RuntimeABIContract.h`,
  `lib/Support/RuntimeABIContract.cpp`,
  `lib/Support/RuntimeABICallablePlan.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/TargetArtifactExport.cpp`.
- Focused test surfaces:
  `test/Support/RuntimeABICallablePlanTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVVScalarDispatch/`,
  `test/Target/TargetArtifactBundleExport/`,
  `test/Scripts/rvv-scalar-dispatch-e2e.test`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`.

## Evidence Plan

- Build touched support/RVV/scalar/export tools and tests:
  `cmake --build build --target tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`.
- Run focused C++ tests:
  `build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `build/bin/tianchenrv-target-artifact-export-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run focused lit/FileCheck tests for scalar dispatch vadd/vsub role binding,
  vsub generic route, target artifact bundle vadd/vsub, and RVV scalar dispatch
  script/bundle e2e.
- Run `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` if the dispatch
  harness or generated caller evidence is touched.
- Run focused `ssh rvv` vsub dispatch artifact invocation unless blocked by a
  precise remote/toolchain issue after local generation/compile.
- Run bounded ref-scan for remaining vadd-only ABI helpers, descriptor-only
  authority, and generic core RVV branches.
- Run `git diff --check` and `git diff --cached --check`.
- Run Trellis validation before finish and after archive.

## Definition of Done

The finite migrated `i32-vadd`/`i32-vsub` runtime ABI dispatch route has one
production selected-family-aware ABI planning surface. Vsub scalar dispatch
uses the same runtime ABI role binding, generated artifact ABI, selected config
and runtime length data, and fail-closed checks as vadd. Descriptor-only
authority remains quarantined, focused local and RVV evidence or an exact
blocker is recorded, the Trellis task is finished/archived, and one coherent
commit records the round.

## Implementation Summary

- Added explicit selected-family overloads for finite i32 runtime ABI helper
  APIs in `RuntimeABI.h`, `RuntimeABIParam.h`, `RuntimeABIMemWindow.h`, and
  `RuntimeABIContract.cpp`.
- Kept the historical no-argument i32 helper APIs as documented `i32-vadd`
  compatibility defaults, while new selected routes can pass `familyID`.
- Extended `RuntimeABICallablePlanTest` so `i32-vadd`, `i32-vsub`, and
  `i32-vmul` all prove selected-family helper parity for callable parameters,
  role requirements, mem-window specs, runtime length specs, dispatch runtime
  specs, dispatch ABI parameters, and role binding.
- Updated vsub/vmul target-artifact test candidate construction to use the
  selected family id when building i32 runtime ABI parameter mirrors.
- Added focused `rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding.mlir`
  coverage for positive vsub dispatch runtime ABI invocation and vsub
  fail-closed mutations: missing runtime length role, stale selected
  runtime-count metadata, duplicate role, wrong type, wrong ownership, unknown
  role, bad guard type, and detached dispatch ABI metadata.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` to
  record the selected-family i32 runtime ABI helper signatures, validation
  matrix, required tests, and the compatibility-only status of no-argument
  i32-vadd helper defaults.

## Validation Summary

- Build:
  `cmake --build build --target tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- Additional RVV plugin build/smoke:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  and `build/bin/tianchenrv-rvv-extension-plugin-test`.
- C++ tests:
  `build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Scripts/rvv-scalar-dispatch-e2e.test Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  from `build/test`; result: 8 passed.
- Script self-test:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local generated dynamic vsub bundle dry-run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --input test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir --expect-selected-kernel frontend_vector_dynamic_bundle_i32_vsub --run-id codex-parity-vsub-dynamic-dry --overwrite --timeout 120`;
  result: `status=success`, `ssh_evidence=false`.
- RVV compile/run evidence:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --input test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir --expect-selected-kernel frontend_vector_dynamic_bundle_i32_vsub --run-id codex-parity-vsub-dynamic-ssh --overwrite --timeout 120 --connect-timeout 10`;
  result: `status=success`, `ssh_evidence=true`,
  `ssh_evidence_verified=true`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vsub`.
  Artifact root:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-parity-vsub-dynamic-ssh`.
- Ref-scan:
  `rg -n "getI32BinaryRuntimeABIContract\\(\\s*\\\"i32-vadd\\\"|appendI32BinaryRuntimeABIParameters\\(|getI32BinaryRuntimeABIParameters\\(|appendI32BinaryRuntimeABIRoleRequirements\\(|getI32BinaryRuntimeABIRoleRequirements\\(|makeI32BinaryDispatchAvailabilityGuard\\(|getI32BinaryDispatchRuntimeABIParameters\\(|bindI32BinaryCallableRuntimeABIParametersByRole\\(|getI32BinaryBufferMemWindowSpecs\\(|getI32BinaryRuntimeElementCountParamSpec\\(|getI32BinaryDispatchRuntimeParamSpecs\\(|descriptor.*authority|descriptor-to-C|hasRVV|hasIME|hasSophgo" include/TianChenRV/Support lib/Support include/TianChenRV/Target/RVV lib/Target/RVV lib/Target/TargetArtifactExport.cpp test/Support/RuntimeABICallablePlanTest.cpp test/Target/TargetArtifactExportTest.cpp test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding.mlir`;
  remaining touched support vadd references are documented compatibility
  defaults, and target-side concrete RVV references remain in RVV target code,
  not generic core orchestration.
- Whitespace:
  `git diff --check` passed.
- Spec update:
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` now records
  the executable selected-family helper contract introduced in this round.
