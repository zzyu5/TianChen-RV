# RVV artifact export source-frontdoor contract

## Goal

Move the selected RVV source-frontdoor identity/config/runtime contract upstream
from dispatch-side surfacing into the C++ artifact export and emission owner
that produces source-frontdoor vadd/vsub artifacts. The bounded route is:
MLIR vector/scf i32 add/sub source input -> selected RVV materialization ->
`tcrv_rvv` typed microkernel/source identity -> TargetArtifactExport and bundle
export -> RVVMicrokernel source/header/object emission -> RuntimeABI and
RVVScalarDispatch consumption.

This round must make an active production C++ change in the artifact/export,
bundle export, RVVMicrokernel emission, selected materialization/emission
planning, or RuntimeABI handoff layer. Dispatch and Python evidence runners may
only consume and verify that production contract.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- At task creation, `git status --short` was clean and HEAD was
  `6d6da4f feat(rvv): surface dispatch selected source identity`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  direction brief and started as the current Trellis task.
- Recent commits and archived PRDs show several same-day rounds have already
  strengthened source-frontdoor identity in `RVVScalarDispatch.cpp`, bundle
  expectations, and script e2e consumption.
- The latest archived production-invocation PRD explicitly changed
  `RVVScalarDispatch.cpp`; it did not close the upstream owner in
  `TargetArtifactExport.cpp`, bundle export implementation,
  `RVVMicrokernel.cpp`, selected emission/materialization, or RuntimeABI export
  handoff.
- Specs require selected RVV binary artifacts to consume typed source identity,
  selected vector config, runtime ABI roles, and selected-boundary source
  identity before source/header/object/bundle output.
- Descriptor-only production authority remains invalid. Descriptors may only
  be legacy mirrors after typed family/body/source authority is established.

## Requirements

- Inspect the current production route before implementation:
  `TargetArtifactExport.cpp`, bundle export implementation,
  `RVVMicrokernel.cpp`, `RVVScalarDispatch.cpp`,
  `RuntimeABIContract.cpp`, selected RVV materialization/planning, source
  lowering, and focused vadd/vsub tests.
- Make an active production C++ owner change outside a dispatch-only/script-only
  surface. Acceptable owners include:
  `lib/Target/TargetArtifactExport.cpp`, target artifact bundle export,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Support/RuntimeABIContract.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`, or
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`.
- Both vadd and vsub source-frontdoor generated artifact paths must carry the
  same selected-source contract through the normal C++ target artifact route:
  source identity, op-family identity, selected config, selected EmitC source
  op/interface, and runtime element-count/AVL role authority.
- Generated source/header/object or bundle output must make the contract
  visible in the exact form consumed by downstream RuntimeABI/RVVScalarDispatch
  validation.
- Dispatch must not infer source-frontdoor identity from descriptor shape, test
  metadata, route file names, or explicit-only assumptions.
- Missing or stale source identity, unsupported source vector pattern, stale op
  identity, missing selected config, missing runtime length role data, stale ABI
  signature, descriptor-only production attempts, missing EmitC body mapping,
  explicit-only route misuse, and dispatch source identity mismatch must fail
  closed before artifact or bundle success.
- Keep generic core orchestration target-neutral: no RVV semantic branches in
  shared core passes and no computation semantics in `tcrv.exec`.
- Do not add Python compiler semantics, descriptor-to-C production export,
  broad vector dialect coverage, new dtype, i64 expansion, LMUL matrix,
  additional op family, performance tuning, or GCC/vendor compiler default
  behavior.

## Acceptance Criteria

- [ ] The implemented patch changes an active C++ production artifact/export,
      RVVMicrokernel, selected materialization/emission planning, RuntimeABI, or
      source-frontdoor selected lowering owner, not only tests, task files,
      scripts, dispatch-only validation, or metadata expectations.
- [ ] Vector source-frontdoor i32-vadd reaches generated source/header/object or
      bundle artifacts through the normal route with selected source identity,
      op-family identity, selected vector config, EmitC source provenance, and
      runtime length role data preserved.
- [ ] Vector source-frontdoor i32-vsub reaches the same generated artifact route
      and contract as vadd, without stale vadd family/operator/intrinsic/ABI
      metadata.
- [ ] Generated artifacts expose the selected-source contract in the form
      RuntimeABI/RVVScalarDispatch consumes; scripts only verify generated
      artifacts or bundle records.
- [ ] Descriptor-only authority, missing/stale selected source identity,
      missing/conflicting selected config, missing runtime element-count role,
      stale ABI signature, missing EmitC body mapping, unsupported vector
      pattern, and explicit-only route misuse fail before successful artifact
      or bundle output.
- [ ] Focused lit/FileCheck coverage proves the positive vadd/vsub source
      frontdoor route and at least the most relevant fail-closed cases touched
      by this round.
- [ ] Focused C++/TableGen build covers generated headers and touched
      transform/plugin/target/export/support libraries and tools, including
      `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
      affected target artifact tests.
- [ ] Local script/tool evidence remains consumer-only and passes for the
      changed generated artifact route.
- [ ] A focused `ssh rvv` generated artifact invocation passes for at least one
      source-frontdoor generated artifact route, or an exact blocker is recorded
      after local generation/compile.
- [ ] `git diff --check`, `git diff --cached --check`, Trellis validation,
      finish/archive, and one coherent commit complete if the task is finished.

## Out Of Scope

- No RVVScalarDispatch-only milestone.
- No bundle expectation/source-identity-only milestone.
- No script-only, e2e-harness-only, metadata-only, task-only, journal-only,
  test-only, negative-test-only, wrapper-only, report-only, or broad smoke-only
  milestone.
- No new Python compiler semantics or Python representation of compiler core.
- No descriptor-to-C production exporter and no descriptor element count or
  vector shape as compute/config/runtime authority.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64, LMUL matrix, third op, broad family matrix, or performance tuning.
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-cpp-artifact-runtime-production-invocation/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-source-frontdoor-generated-artifact-invocation-closure/prd.md`,
  `.trellis/workspace/codex/journal-5.md`.
- Current likely bottleneck from existing evidence: recent work made
  dispatch-side selected source identity visible, but the selected-source
  contract still needs to be owned or enforced at the production
  artifact/export or emission boundary that produces the data dispatch
  validates.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-rvv-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Run focused lit from `build/test` for vadd/vsub VectorToExec,
  TargetArtifactExport, TargetArtifactBundleExport, RVVMicrokernel,
  RVVScalarDispatch, and script e2e surfaces affected by this round.
- Run script self-tests:
  `python3 scripts/rvv_microkernel_e2e.py --self-test` and
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`.
- Run local generated artifact dry-runs for dynamic vector i32-vadd and
  i32-vsub on the changed route.
- Run focused `ssh rvv` generated artifact invocation for at least one changed
  source-frontdoor route if reachable.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale op-family identity, and generic core semantic branches.
- Run `git diff --check`, `git diff --cached --check`, and Trellis validation
  before finish/archive.
