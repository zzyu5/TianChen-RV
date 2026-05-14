# RVV source-frontdoor artifact/runtime closure

## Goal

Close the bounded RVV source-frontdoor artifact/runtime slice by making the
unified dynamic vector i32-vadd/i32-vsub source policy observable in the
production generated artifact and dispatch bundle contract. Current HEAD has
already unified the transform-side family policy; this round focuses on the
artifact/runtime consumer surface so generated bundle metadata preserves the
selected source-frontdoor identity rather than relying only on transform IR or
negative tests.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Worktree was clean at task creation; HEAD was
  `7777a16 feat(vector): unify rvv source-frontdoor family policy`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
- The archived `05-14-rvv-vector-frontend-default-artifact-route` task already
  promoted bounded dynamic vector i32-vadd/i32-vsub artifact evidence to the
  default `--tcrv-lower-source-rvv-binary-to-exec` route.
- The archived `05-14-rvv-source-frontdoor-op-family-artifact-parity` task
  changed `LowerSourceRVVBinaryToExec.cpp` so the dynamic vector add/sub
  transform policy is registry-backed rather than vadd/vsub branch-backed.
- Current target/export code already validates typed selected RVV source
  identity, selected config, runtime AVL/VL metadata, RuntimeABI parameter
  mirrors, EmitC route metadata, descriptor quarantine, and bundle source/
  header/object output.
- The remaining bounded closure point is the generated RVV+scalar dispatch
  bundle metadata: it should carry a complete dispatch-level selected source
  identity for source-frontdoor vadd/vsub, including source kind, family,
  operator, microkernel op, and EmitC source op.

## Requirements

- Dynamic MLIR vector i32-vadd and i32-vsub source input must continue through
  the normal source frontdoor into selected `tcrv_rvv` and scalar fallback
  dispatch artifacts.
- The RVV+scalar dispatch bundle metadata must expose the selected RVV
  source-frontdoor identity as production artifact metadata, not only as
  transform-level metadata or embedded callable-source comments.
- The composite selected source identity must preserve source kind, finite
  op-family identity, selected arithmetic operator, selected RVV microkernel
  op, and selected EmitC source op.
- TargetArtifactBundleExport must continue to consume selected config,
  runtime length, RuntimeABI parameter, and dispatch guard data from the
  selected route, not from descriptor-only metadata.
- Existing direct RVV source/header/object and RVVScalarDispatch consumers
  must remain green for both vadd and vsub.
- Unsupported or stale selected source identity, missing selected config,
  stale runtime length, stale EmitC mapping, descriptor-only production
  attempts, and explicit-only misuse must continue to fail closed before
  artifact or bundle completion.
- Generic core orchestration must remain neutral: no RVV/IME/offload/scalar
  semantic branches in shared core passes and no compute semantics in
  `tcrv.exec`.

## Acceptance Criteria

- [x] `TargetArtifactBundleExport` output for bounded dynamic vector i32-vadd
      and i32-vsub records a complete
      `tcrv_rvv.dispatch_contract_selected_source_identity` value with source
      kind, family, operator, microkernel op, and EmitC source op.
- [x] The RVV+scalar dispatch bundle production path derives that metadata
      from the validated selected RVV component contract/metadata, not from
      descriptor route guesses or explicit frontend-only assumptions.
- [x] Focused lit/FileCheck coverage proves both dynamic vector vadd and vsub
      source-frontdoor bundle indexes carry the complete source identity and
      still preserve selected config/runtime length metadata.
- [x] Existing fail-closed coverage for stale source identity, missing source
      identity, stale selected config, stale runtime length, stale EmitC body
      mapping, descriptor-only authority, and explicit alias misuse remains
      green.
- [x] Focused build, C++ tests, lit/FileCheck tests, e2e self-tests/dry-runs,
      `ssh rvv` evidence or exact blocker, ref-scans, `git diff --check`,
      `git diff --cached --check`, Trellis validation, archive, and one
      coherent commit complete the round.

## Out Of Scope

- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 vector source-frontdoor expansion, LMUL matrix expansion, third dynamic
  vector operation, broad family matrix, broad smoke suite, or performance
  tuning.
- No additional explicit-only frontend flag or vsub-only helper as the main
  milestone.
- No descriptor-to-C production exporter.
- No descriptor element count or descriptor vector shape as compute, selected
  config, runtime AVL/VL, ABI, dispatch, or artifact authority.
- No moving computation semantics into `tcrv.exec`.
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-op-family-artifact-parity/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-vector-frontend-default-artifact-route/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Primary implementation surfaces:
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  focused bundle tests under `test/Target/TargetArtifactBundleExport`,
  and focused script/e2e evidence under `test/Scripts`.

## Evidence Plan

- Build touched target/export/tool/test targets:
  `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck for dynamic vector vadd/vsub default source
  lowering, TargetArtifactBundleExport dynamic vector vadd/vsub, RVV
  microkernel bundle e2e, and RVVScalarDispatch bundle e2e.
- Run `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run focused local dry-runs for source-frontdoor dynamic vector vadd/vsub
  bundle evidence and vsub RVVScalarDispatch plan-and-export evidence.
- Run focused `ssh rvv` generated artifact invocation for at least one
  representative source-frontdoor op unless blocked after local generation and
  compile.
- Run bounded ref-scans for explicit-only vector routing, descriptor-only
  authority, stale vadd-only evidence, and generic core RVV/IME/Sophgo
  semantic branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation
  before finish and after archive.

## Implementation Summary

- Extended
  `RVVSelectedConfig::formatDispatchContractSelectedSourceIdentityMetadataValue`
  so the RVV+scalar dispatch bundle route claim now records:
  `source_kind`, `dtype`, finite `family`, arithmetic `operator`, selected RVV
  `microkernel_op`, selected `emitc_source_op`, and
  `TCRVEmitCLowerableOpInterface`.
- Updated the focused dynamic vector vadd/vsub bundle FileCheck expectations so
  the generated bundle index must expose the complete dispatch-level selected
  source identity for both source-frontdoor ops.
- Left production route ownership unchanged: the value is still derived from
  the selected RVV config/contract consumed by RVVScalarDispatch and
  TargetArtifactBundleExport. No descriptor-to-C route, core `tcrv.exec`
  semantics, explicit-only frontend flag, or family branch in generic core
  orchestration was added.

## Validation Summary

- Build:
  `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`
  from `build/test` passed `2/2`.
- C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Script self-tests:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Local dynamic vector source-frontdoor dry-runs:
  `rvv_microkernel_e2e.py` plan-and-export bundle dry-run for
  `i32-vadd` and `i32-vsub`, both with runtime counts `7,16,23`, both
  `status: success`.
- Local RVV+scalar dispatch dry-runs:
  `rvv_scalar_dispatch_e2e.py` plan-and-export bundle dry-run for
  `i32-vadd` and `i32-vsub`; generated bundle indexes contain
  `tcrv_rvv.dispatch_contract_selected_source_identity` with the new typed
  value for all source/header/object dispatch artifacts.
- `ssh rvv` generated artifact invocation:
  `rvv_microkernel_e2e.py` plan-and-export source-frontdoor bundle runs for
  `codex-source-identity-vsub-ssh` and `codex-source-identity-vadd-ssh` both
  returned `status: success` and `ssh_evidence: true`.
- Ref-scans:
  no new explicit-only adapter mode or stale vadd/vsub-only symbol surfaced;
  `tcrv.exec` hits remain generic runtime/extension-boundary documentation or
  runtime ABI fields; descriptor hits remain the existing quarantine and
  authority checks.
- Hygiene:
  `git diff --check` passed; task context validation passed before finish.

## Self-Repair

- First RVVScalarDispatch dry-run used linalg bundle fixtures but asserted the
  dynamic-vector selected kernel names. The runner correctly failed closed with
  `selected kernel frontend_bundle_i32_vadd did not match expected
  frontend_vector_dynamic_i32_vadd` and the analogous vsub message. Re-ran the
  same commands with `frontend_bundle_i32_vadd` and
  `frontend_bundle_i32_vsub`; both passed and their bundle indexes contain the
  new typed dispatch source identity.

## Spec Update Judgment

No `.trellis/spec/` update is needed. This round did not introduce a new
dialect op, command signature, schema field, plugin protocol, or architecture
rule. It made an already-specified selected-source identity more complete at
the production artifact/dispatch bundle surface and verified the existing
source-frontdoor, selected config, runtime length, EmitC route, RuntimeABI, and
descriptor-quarantine contracts.

## Definition Of Done

The production source-frontdoor dynamic vector route for i32-vadd and i32-vsub
is visible through generated artifact/bundle metadata and runtime-dispatch
consumers, with selected source identity/config/runtime length carried by the
validated route rather than descriptor-only authority. The task is finished,
archived, validated, and committed with the final worktree clean.
