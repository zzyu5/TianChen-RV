# RVV source-frontdoor production artifact/runtime closure

## Goal

Repair and finish the bounded production/default source-frontdoor route for
MLIR vector i32 add/sub inputs. This task must prove that the route from
`LowerSourceRVVBinaryToExec` through selected RVV materialization reaches
generated artifacts, RuntimeABI/RVVScalarDispatch consumption, and focused
`ssh rvv` invocation evidence. The milestone is production route closure, not
metadata-only bundle coverage.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current task was created from the Hermes direction brief because no
  `.trellis/.current-task` existed.
- At task creation the worktree was clean and HEAD was
  `aab384e feat(rvv): preserve source identity in dispatch bundles`.
- The previous completed task made selected RVV source identity more explicit
  in dispatch bundle metadata and touched
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`, but it did not
  by itself prove active closure in the production source-frontdoor,
  artifact/export, RuntimeABI/RVVScalarDispatch, or e2e invocation owners.
- Archived PRDs show recent progress:
  `05-14-rvv-source-frontdoor-op-family-artifact-parity` unified the dynamic
  vector add/sub transform policy through the RVV binary family registry, and
  `05-14-rvv-source-frontdoor-artifact-runtime-closure` extended dispatch
  bundle selected source identity metadata.
- The next bottleneck is to verify and, where necessary, repair the production
  handoff so source-frontdoor vadd/vsub selected state is consumed by artifact
  and runtime invocation surfaces rather than only preserved as standalone
  metadata.

## Requirements

- Dynamic MLIR vector i32-vadd and i32-vsub source inputs must lower through
  the normal default `--tcrv-lower-source-rvv-binary-to-exec` route into
  selected `tcrv_rvv` op-family state.
- The selected route must carry source identity, op-family identity, selected
  config, selected EmitC source/body mapping, and runtime AVL/VL role data into
  generated source/header/object or bundle artifacts.
- `TargetArtifactExport`/`TargetArtifactBundleExport` and either
  `RuntimeABICallablePlan`/`RVVScalarDispatch` or the generated e2e harness
  must consume production selected state, not descriptor shape, bundle metadata
  alone, or explicit-only assumptions.
- The explicit vector frontend compatibility surface may remain only if it
  routes through the same construction contract or fails closed.
- Unsupported vector patterns, stale op identity, missing selected config,
  missing runtime length role data, stale ABI signatures, family mapping
  mismatches, descriptor-only production attempts, missing EmitC body mapping,
  explicit-only route misuse, and missing or conflicting selected config must
  fail closed before artifact/runtime success.
- Generic core orchestration must remain target-neutral: no RVV/IME/offload/
  scalar semantic branches in shared core passes and no computation semantics
  in `tcrv.exec`.

## Acceptance Criteria

- [x] Both vadd and vsub source-frontdoor inputs lower through the default
      source route into selected RVV op-family state with complete source
      identity and selected config.
- [x] Generated source/header/object or bundle artifacts for both ops preserve
      source identity, op-family identity, selected config, and runtime length
      roles.
- [x] At least one production owner beyond tests/metadata is actively repaired
      or proven through source changes where current code is incomplete:
      source-frontdoor selected handoff, RVV materialization, TargetArtifact
      export/bundle export, RuntimeABI/RVVScalarDispatch consumption,
      RVVMicrokernel emission, or the minimal e2e harness needed to invoke
      generated production artifacts.
- [x] RuntimeABI/RVVScalarDispatch or generated e2e invocation consumes the
      selected production state rather than descriptor-only authority or
      explicit-route assumptions.
- [x] Focused fail-closed coverage exists or remains green for unsupported
      vector patterns, stale op identity, missing selected config, missing
      runtime length role data, stale ABI signatures, family mapping mismatch,
      descriptor-only production attempts, missing EmitC body mapping,
      explicit-only misuse, and missing/conflicting selected config.
- [x] Focused local evidence covers both vadd and vsub artifact/runtime paths.
- [x] Focused `ssh rvv` compile/run evidence covers at least one
      source-frontdoor generated artifact invocation, or the exact blocker is
      recorded after local generation and compile.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      archive, and one coherent commit complete the round if the task is
      finished.

## Out Of Scope

- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 expansion, LMUL matrix, third dynamic vector operation, broad family
  matrix, broad smoke suite, or performance tuning.
- No additional explicit-only frontend flag, vsub-only helper, test-only,
  script-only, metadata-only, spec-only, FileCheck-only, wrapper-only, or
  workspace-journal-only milestone.
- No descriptor-to-C production exporter and no descriptor element count or
  vector shape as compute/config/runtime/ABI authority.
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
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-artifact-runtime-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-op-family-artifact-parity/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Primary implementation surfaces to inspect:
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/RVV/RVVScalarDispatch.cpp`,
  `lib/Support/RuntimeABIContract.cpp`,
  `scripts/rvv_microkernel_e2e.py`,
  `scripts/rvv_scalar_dispatch_e2e.py`.

## Evidence Plan

- Build focused touched targets:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit/FileCheck for VectorToExec, TargetArtifactExport,
  TargetArtifactBundleExport, RVVMicrokernel, RVVScalarDispatch, and script e2e
  routes touched by this task.
- Run `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run focused local vadd/vsub source-frontdoor artifact/bundle and dispatch
  dry-runs with runtime counts `7,16,23`.
- Run focused `ssh rvv` generated artifact invocation for at least one
  source-frontdoor op unless blocked after local generation and compile.
- Run bounded ref-scans for explicit-only routing, descriptor-only authority,
  stale op-family identity, and generic core semantic branches.
- Run `git diff --check`, `git diff --cached --check`, Trellis validation
  before finish and after archive.

## Implementation Summary

- Production owner changed: `lib/Target/RVV/RVVScalarDispatch.cpp`.
- `RVVScalarDispatch` now resolves the selected RVV dispatch case variant's
  selected binary source kind before composite dispatch contract validation.
- When a dispatch case carries source identity, the composite dispatch
  selected-config metadata validation now requires selected-plan
  `tcrv_rvv.selected_binary_source_kind` and
  `tcrv_rvv.selected_binary_microkernel_op` to match the
  `RVVBinarySelectedConfigContract`.
- This moves source-frontdoor selected source identity consumption into the
  RVV+scalar dispatch runtime ABI preflight, instead of relying only on later
  component microkernel export or bundle metadata side effects.
- Focused test changed:
  `test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir`
  now mutates the variant source-kind authority to
  `direct-typed-microkernel-body` and verifies dispatch preflight rejects the
  stale selected-plan source identity before emitting the dispatch C artifact.
- No generic core orchestration was changed; no descriptor-to-C production path
  was added; descriptor element count remains component capacity metadata.

## Validation Summary

- Build:
  `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- Focused lit:
  `Target/RVVScalarDispatch` passed `14/14`.
- Focused lit:
  `Target/TargetArtifactBundleExport` passed `18/18`.
- Focused lit:
  `Target/RVVMicrokernel` passed `31/31`.
- Focused lit:
  `vector-dynamic-i32-vadd-to-exec`,
  `vector-dynamic-i32-vsub-to-exec`,
  `plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle`,
  and `plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle`
  passed `4/4`.
- Focused script lit:
  `Scripts/rvv-microkernel-e2e.test`,
  `Scripts/rvv-microkernel-bundle-e2e.test`,
  `Scripts/rvv-scalar-dispatch-e2e.test`, and
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test` passed `4/4`.
- C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test` passed and
  `./build/bin/tianchenrv-rvv-extension-plugin-test` printed
  `RVV extension plugin smoke test passed`.
- Script self-tests:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` passed.
- Local source-frontdoor dry-runs:
  vector i32-vadd and i32-vsub microkernel bundle front-door dry-runs passed
  with runtime counts `7,16,23`.
- Local dispatch dry-runs:
  i32-vadd and i32-vsub dispatch bundle front-door dry-runs passed with
  selected kernels `frontend_bundle_i32_vadd` and `frontend_bundle_i32_vsub`.
- `ssh rvv` evidence:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vsub --lower-vector-i32-vsub-frontend --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --expect-selected-kernel frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --run-id codex-source-frontdoor-vsub-ssh --overwrite --timeout 120 --connect-timeout 10`
  passed. Artifact directory:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-source-frontdoor-vsub-ssh`.
  Remote host facts: `riscv64`, `/usr/bin/clang`,
  `Ubuntu clang version 18.1.3 (1ubuntu1)`. Both source-built and
  bundle-object-built external caller runs printed
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.
- Boundary scan:
  `rg` over `lib/Transforms`, `include/TianChenRV/Dialect/Exec`,
  `lib/Dialect/Exec`, `lib/Target/RVV/RVVScalarDispatch.cpp`, and
  `lib/Target/RVV/RVVMicrokernel.cpp` found only existing RVV target-local
  helper/claim strings plus this task's RVV target-local selected source-kind
  validation; no new core-pass RVV/IME/offload semantic branch was introduced.
- Hygiene:
  `git diff --check` passed before staging.

## Self-Repair Notes

- The first stale-source negative test mutated
  `selected_binary_microkernel_op`, but that was correctly caught by existing
  RVV microkernel component validation before the new dispatch-owner check.
  The test was corrected to mutate the dispatch variant source-kind authority
  to a legal but conflicting value so it exercises the new
  `RVVScalarDispatch` selected-source contract validation.
- The first dispatch dry-run used stale expected selected-kernel names
  `frontend_linalg_bundle_i32_{vadd,vsub}`; rerun with the actual generated
  `frontend_bundle_i32_{vadd,vsub}` names passed.
- `clang-format` was not installed in this environment. The C++ edit was kept
  manually aligned and verified by the focused build plus `git diff --check`.

## Finish Status

- Code and evidence criteria above are satisfied.
- Trellis task was finished and archived under
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-production-artifact-runtime-closure`.
- Archived-task validation required one self-repair: `check.jsonl` was updated
  from the live task PRD path to the archived PRD path before final validation.
- Final staged hygiene and commit are performed by this round's closing
  changeset.
