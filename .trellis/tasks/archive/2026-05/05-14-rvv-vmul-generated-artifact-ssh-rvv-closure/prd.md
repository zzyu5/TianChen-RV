# RVV vmul generated artifact ssh-rvv closure

## Goal

Close `i32-vmul` as a real reusable RVV binary family member by carrying it
through the production generated-artifact and runtime-dispatch route already
used for `i32-vadd` and `i32-vsub`. This round must prove, from current HEAD,
that vmul is selected by the shared RVV binary family contract, materialized by
the RVV plugin, emitted/exported by RVV target artifact owners, invoked through
`RuntimeABICallablePlan` and `RVVScalarDispatch`, and validated by focused
local generated-artifact plus `ssh rvv` evidence when the intrinsic route is
complete.

## Current Repository Facts

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd` confirmed the repository root.
- Initial `git status --short` was clean.
- Initial HEAD was
  `530b498 feat(rvv): bind runtime invocation family contracts`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief as
  `.trellis/tasks/05-14-rvv-vmul-generated-artifact-ssh-rvv-closure`.
- The current direction brief states that the prior round bound
  `RuntimeABICallablePlan` family contracts for vadd/vsub/vmul, but the live
  summary still does not prove production RVV materialization/emission,
  `RVVMicrokernel`, `TargetArtifactExport`, `RVVScalarDispatch`, or focused
  `ssh rvv` vmul generated-artifact execution.
- Archived PRDs for the previous binary-family and runtime-invocation rounds
  record local and ssh vmul evidence, but this task treats those records as
  context only. Current HEAD code and freshly collected evidence are the
  authority for this round.
- The relevant long-term specs require extension-family op authority,
  plugin-owned materialization/emission, EmitC/intrinsic route metadata,
  IR-backed runtime ABI plans, selected-family runtime ABI contracts, and real
  `ssh rvv` evidence before any RVV runtime/correctness claim.
- Current HEAD already carries vmul through the shared RVV binary family
  registry, selected boundary, plugin materialization, selected emission
  planning, direct RVV microkernel export, target artifact bundle export,
  RuntimeABI invocation contracts, and RVVScalarDispatch. The concrete
  asymmetry found in this round was the public explicit vector/SCF adapter:
  vadd and vsub had dedicated `tcrv-opt` entry points, while vmul only entered
  through the generic source front door.
- `scripts/rvv_microkernel_e2e.py` also lacked a direct vector/SCF vmul
  frontend flag, so direct RVVMicrokernel generated-artifact evidence could not
  select the existing vector/SCF vmul fixture without spelling a path manually.

## Requirements

- Preserve the C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck compiler
  stack. Python may only be used for runners, probes, artifact parsing, and
  evidence support.
- Keep computation semantics out of `tcrv.exec`.
- Do not add descriptor-driven computation, descriptor-to-C production export,
  descriptor element-count authority, vector-shape runtime authority, or Python
  compiler semantics.
- Do not add RVV semantic branches in generic core orchestration.
- Route vmul through the finite RVV binary family registry and selected
  contracts, not vmul-only production branches outside plugin/target family
  ownership.
- Rewire production/default paths when a current owner still admits only
  vadd/vsub or still uses stale vadd defaults for vmul.
- Preserve vadd/vsub generated-artifact, dispatch, and prior ssh evidence
  regressions on the touched surfaces.
- Unsupported, stale, descriptor-only, explicit-only, or partially declared
  family states must fail before artifact/runtime claims.

## Acceptance Criteria

- [x] Current code inspection identifies the active vmul production path across
      source-frontdoor selection, RVV selected boundary, plugin materialization,
      selected emission planning, `RVVMicrokernel`, `TargetArtifactExport`,
      `RuntimeABICallablePlan`, and `RVVScalarDispatch`.
- [x] Any missing production owner is fixed in C++/MLIR/TableGen/CMake/lit code
      and vmul uses the same shared selected-family contract as vadd/vsub for
      source identity, family identity, selected config, runtime ABI signature,
      runtime length role data, EmitC/intrinsic mapping, artifact naming, and
      dispatch validation.
- [x] No ad hoc vmul-only branch is added in generic core orchestration; any
      family-specific facts remain in RVV plugin/target family descriptors,
      route manifests, or selected-family contracts.
- [x] Generated vmul artifacts are locally produced and validated through the
      normal dispatch/runtime route, not by a detached script-only or
      descriptor-only path.
- [x] Focused `ssh rvv` compile/run evidence covers generated `i32-vmul` if the
      intrinsic route is complete. If remote evidence cannot complete, the task
      records the exact remaining production or environment blocker after local
      generation/invocation evidence.
- [x] vadd/vsub generated-artifact and dispatch regressions remain green on the
      touched surfaces.
- [x] Fail-closed checks cover unknown op family, stale op-family identity,
      missing selected config, missing runtime length role data, missing EmitC
      body mapping, stale ABI signature, descriptor-only production attempts,
      explicit-only route misuse, and source-identity mismatch where those
      cases intersect the changed module behavior.
- [x] Trellis validation, `git diff --check`, `git diff --cached --check`, task
      finish/archive, one coherent commit, and final worktree cleanliness are
      completed if the task finishes.

## Out Of Scope

- No broad dtype, mask/tail, LMUL matrix, linalg/tensor frontend, performance
  tuning, or arbitrary additional op-family expansion.
- No helper-only, registry-only, ABI-contract-only, manifest-only,
  documentation-only, metadata-only, PRD-only, journal-only, test-only,
  negative-test-only, script-only, smoke-only, report-only, artifact-text-only,
  or evidence-packaging-only milestone.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.
- No descriptor-to-C production exporter or descriptor element-count runtime
  authority.
- No RVV semantic branches in generic core passes.
- No replacing clang/LLVM as the default native compiler route.
- No runtime, correctness, or performance claim without focused local generated
  artifact or real `ssh rvv` evidence.

## Evidence Plan

- Inspect and, if needed, update the production owners named in the task brief:
  RVV binary family registry, source-frontdoor lowering, selected boundary,
  materialization, emission planning, RVV microkernel export,
  target-artifact export, runtime ABI callable planning, RVV scalar dispatch,
  and focused e2e scripts.
- Build focused targets for touched transform/plugin/target/export/support
  owners, including generated headers, RVV plugin/target libraries,
  `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, `tcrv-translate`,
  RuntimeABI callable-plan test, and target artifact export test.
- Run focused C++ tests:
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run focused lit/FileCheck for vadd/vsub/vmul VectorToExec, RuntimeABI,
  RVVMicrokernel, TargetArtifactExport, TargetArtifactBundleExport,
  RVVScalarDispatch, and script e2e surfaces affected by this round.
- Run local generated-artifact/runtime evidence for `i32-vmul` plus focused
  vadd/vsub regressions.
- Run focused `ssh rvv` clang compile/run evidence for generated `i32-vmul`
  when the intrinsic route is complete.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, source-identity mismatch, stale vadd defaults, and generic core RVV
  semantic branches.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-vmul-op-family-artifact-runtime-closure/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-14-rvv-binary-op-family-contract/prd.md`.
- Source inspection and implementation notes will be appended after current
  production owners are inspected.

## Implementation Summary

- Added `createLowerVectorRVVI32VMulToExecPass()` and the TableGen pass
  `--tcrv-lower-vector-rvv-i32-vmul-to-exec`.
- Reused the existing `VectorFrontendPolicyKind` / registry-backed dynamic
  vector source-frontdoor path. The new explicit vmul adapter only restricts
  the accepted source family to the existing `i32-vmul` family descriptor; it
  does not introduce a generic core RVV semantic branch.
- Registered the new pass in `tcrv-opt`, closing the public tool entry point
  gap found by lit.
- Updated the Dynamic Vector Source Tail Authority spec to include the vmul
  explicit adapter alias.
- Added positive and fail-closed lit coverage for the explicit vmul adapter:
  marker mismatch, body/marker mismatch, and fixed-vector misuse fail before
  exec/artifact claims.
- Extended `scripts/rvv_microkernel_e2e.py` with
  `--lower-vector-i32-vmul-frontend`, selecting the existing
  `test/Transforms/VectorToExec/vector-dynamic-i32-vmul-to-exec.mlir` fixture
  for direct RVVMicrokernel evidence.
- Extended `test/Scripts/rvv-microkernel-bundle-e2e.test` so direct
  RVVMicrokernel bundle dry-run coverage includes vector/SCF vmul source,
  source/header/object bundle records, source comments, and a generated
  external caller that checks `lhs * rhs`.

## Validation Summary

- `clang-format`, `clang-format-20`, `clang-format-19`, and
  `clang-format-18` were not available in PATH; changed C++ was kept
  consistent with neighboring code.
- Initial focused lit run failed because `tcrv-opt` did not register the new
  vmul pass. The tool driver registration was added, `tcrv-opt` was rebuilt,
  and the same lit filter passed.
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
- `tcrv-opt --help` now lists
  `--tcrv-lower-vector-rvv-i32-vmul-to-exec` alongside the vadd/vsub explicit
  adapters.
- Local direct RVVMicrokernel generated-artifact bundle dry-runs passed for:
  `codex-vmul-explicit-vector-micro-vadd-local`,
  `codex-vmul-explicit-vector-micro-vsub-local`, and
  `codex-vmul-explicit-vector-micro-local`. The vmul run selected
  `frontend_vector_dynamic_i32_vmul`, route
  `tcrv-export-rvv-i32-vmul-microkernel-c`, runtime counts `7,16,23`, and
  source SHA256 `3120d67ddd9850fbee47d0aafae34c8e30ef5dab0537040bba36176687b4411d`.
- Local RVVScalarDispatch generated-artifact bundle dry-runs passed for:
  `codex-vmul-explicit-vector-dispatch-vadd-local`,
  `codex-vmul-explicit-vector-dispatch-vsub-local`, and
  `codex-vmul-explicit-vector-dispatch-local`. The vmul run selected
  `frontend_vector_dynamic_bundle_i32_vmul` and source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
- Focused `ssh rvv` direct RVVMicrokernel bundle evidence passed:
  `codex-vmul-explicit-vector-micro-ssh-rvv`, mode `ssh`,
  claim scope `bounded RVV i32-vmul target-artifact bundle external caller
  correctness only`, selected kernel `frontend_vector_dynamic_i32_vmul`,
  source SHA256 `3120d67ddd9850fbee47d0aafae34c8e30ef5dab0537040bba36176687b4411d`.
  Both source-built and bundle-object external caller runs printed
  `tcrv_rvv_i32_vmul_microkernel_external_abi_ok counts=7,16,23`.
- Focused `ssh rvv` RVVScalarDispatch bundle evidence passed:
  `codex-vmul-explicit-vector-dispatch-ssh-rvv`, mode `ssh`,
  `ssh_evidence_verified = true`, selected kernel
  `frontend_vector_dynamic_bundle_i32_vmul`, source SHA256
  `4422aadeda307d8d0251b81ce74e090021cdf6024181c315568407c0a4af5928`.
  Both source-built and bundle-object caller runs printed
  `tcrv_rvv_scalar_i32_vmul_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
- Bounded ref-scan over changed files for `descriptor-to-C`, `descriptor-only`,
  `descriptor_element_count.*runtime`, `explicit-only`, selected descriptor
  metadata, and source/runtime authority found no new production descriptor
  authority. Hits were existing spec text, existing descriptor-quarantine test
  checks, and unchanged legacy-descriptor rejection constants in the source
  frontend.
- `git diff --check` passed.
- Trellis validation passed before finish and after archive.
- `git diff --cached --check` passed after staging as the final pre-commit
  check.
