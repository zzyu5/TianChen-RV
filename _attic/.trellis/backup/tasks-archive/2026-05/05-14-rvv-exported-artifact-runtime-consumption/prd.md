# RVV Exported Artifact Runtime Consumption

## Goal

Make the downstream RuntimeABI/RVVScalarDispatch and e2e invocation path consume
the source-frontdoor artifact contract now emitted by RVVMicrokernel-generated
RVV artifacts. For bounded dynamic vector i32 vadd/vsub source-frontdoor
routes, a generated RVV callable artifact must be accepted for dispatch/runtime
invocation only when its exported selected source identity, op-family identity,
selected config, runtime ABI signature, and runtime AVL/VL role data match the
selected route.

This round must change active production code in the runtime/dispatch/artifact
consumption path. Script changes may only validate and invoke generated
artifacts from that production route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD before this task is `0d5f1d7 feat(rvv): export selected source identity in artifacts`.
- `git status --short` was clean before task creation.
- No `.trellis/.current-task` existed, so this Trellis task was created from
  the Hermes direction brief.
- The previous completed task moved selected RVV source identity into
  `RVVMicrokernel.cpp` artifact/export validation and generated source/header
  comments for vadd/vsub.
- Earlier same-day work already made `RuntimeABICallablePlan`,
  `TargetArtifactExport`, `RVVMicrokernel`, `RVVScalarDispatch`, and the e2e
  scripts consume IR-backed callable ABI signatures and runtime element-count
  roles.
- Current `RVVScalarDispatch.cpp` validates embedded RVV source shape,
  selected config, intrinsics, dispatch runtime ABI roles, and bundle metadata.
  The remaining gap is that dispatch/runtime embedding does not yet treat the
  RVVMicrokernel-emitted artifact source-identity and runtime invocation
  contract comments as required downstream input.
- Current `rvv_scalar_dispatch_e2e.py` validates bundle selected-plan metadata,
  source/header runtime AVL authority, dispatch runtime guard/fallback linkage,
  and generated external caller invocation. It does not yet parse and record the
  embedded RVV artifact source identity contract as the invocation authority.

## Requirements

- Add an active production C++ consumer in the RuntimeABI/RVVScalarDispatch
  path, preferably the RVV dispatch embedding preflight that consumes the
  RVVMicrokernel-generated callable source before dispatch source/header/object
  export.
- The consumer must require the embedded RVV callable source artifact to expose
  `rvv_microkernel_selected_source_identity` matching the selected dispatch
  source kind, dtype, family, arithmetic operator, selected RVV microkernel op,
  EmitC source op, and `TCRVEmitCLowerableOpInterface`.
- The consumer must require the embedded RVV callable source artifact to expose
  a `runtime_abi_invocation_contract` matching the generated RVV callable
  symbol, runtime ABI kind/name, runtime glue role, ordered callable ABI roles,
  runtime element-count C name, and production owner.
- The consumer must continue to validate selected vector config, RVV intrinsic
  mapping, runtime AVL/VL authority, and descriptor quarantine before dispatch
  source/header/object or bundle output succeeds.
- The e2e path must parse and record the generated artifact contract from the
  embedded RVV source, and it must fail closed if source identity, callable ABI
  contract, selected config, runtime length authority, or selected family data
  are missing or stale.
- Both vadd and vsub vector source-frontdoor routes must remain covered. If the
  whole route is too large, the coherent fallback is vsub generated artifact
  runtime invocation through the same dispatch contract as vadd, with vadd
  regressions green and the exact parity gap recorded.
- Keep all compiler semantics in C++/MLIR/TableGen/CMake/lit. Python remains
  runner/evidence tooling only.
- Keep generic core orchestration target-neutral. RVV-specific validation stays
  in RVV target/runtime owners, not common passes or `tcrv.exec` compute
  semantics.

## Acceptance Criteria

- [x] `RVVScalarDispatch.cpp` or an equivalent production runtime/dispatch
      owner requires embedded RVVMicrokernel-generated RVV source artifacts to
      carry matching exported source identity before dispatch artifact export.
- [x] The same production owner requires the embedded RVV callable source
      `runtime_abi_invocation_contract` to match selected runtime ABI roles and
      the dispatch-selected runtime element-count name before invocation/export.
- [x] Dynamic vector i32-vadd and i32-vsub source-frontdoor bundle routes still
      generate dispatch source/header/object artifacts through the normal
      TargetArtifactExport/BundleExport route.
- [x] Generated dispatch source/header or bundle evidence makes clear that the
      downstream runtime/dispatch route consumed the exported RVV artifact
      contract rather than descriptor shape, lit metadata, explicit-only
      assumptions, or a hand-written surrogate kernel.
- [x] Missing/stale source identity, wrong op family, missing selected config,
      missing runtime length role data, stale ABI signature, descriptor-only
      production attempts, and dispatch source identity mismatch fail closed in
      focused lit/C++ or script self-test coverage touched by this round.
- [x] `rvv_scalar_dispatch_e2e.py` remains a consumer: it parses generated
      bundle/source/header artifacts, records artifact contract authority, and
      invokes generated artifacts without adding Python compiler semantics.
- [x] Focused local checks pass for touched C++/target/export/runtime tests and
      scripts.
- [x] Focused `ssh rvv` compile/run evidence covers at least one generated
      source-frontdoor artifact invocation, or an exact blocker is recorded
      after local generation and compile.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      finish/archive, and one coherent commit complete the round if finished.

## Out Of Scope

- No artifact-text-only, bundle-index-only, RVVMicrokernel-only,
  dispatch-metadata-only, script-only, e2e-harness-only, metadata-only,
  task-only, journal-only, test-only, negative-test-only, wrapper-only,
  report-only, or smoke-only milestone.
- No new Python compiler semantics.
- No descriptor-to-C production exporter and no descriptor element count or
  vector shape as compute/config/runtime authority.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core orchestration.
- No broad vector dialect implementation, linalg/tensor expansion, new dtype,
  i64 expansion, LMUL matrix, third operation, broad family matrix, or
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
  `.trellis/tasks/archive/2026-05/05-14-rvv-artifact-export-source-frontdoor-contract/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-cpp-artifact-runtime-production-invocation/prd.md`,
  `.trellis/tasks/archive/2026-05/05-14-rvv-runtime-abi-production-invocation/prd.md`,
  `.trellis/workspace/codex/journal-6.md`.
- Current likely production edit:
  `lib/Target/RVV/RVVScalarDispatch.cpp`, especially embedded RVV callable
  source validation in `buildEmbeddedCallableSources` /
  `validateEmbeddedRVVSourceSelectedShape`.
- Current likely consumer edit:
  `scripts/rvv_scalar_dispatch_e2e.py`, adding parser/validation/evidence for
  the RVVMicrokernel-emitted source identity/runtime invocation contract already
  present in the embedded generated RVV source.

## Evidence Plan

- Build focused targets:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Run focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Run focused lit from `build/test` for:
  `Target/RVVScalarDispatch`, `Target/RVVMicrokernel`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`,
  and `Scripts/rvv-scalar-dispatch-bundle-e2e.test`.
- Run script self-tests:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test` and, if affected,
  `python3 scripts/rvv_microkernel_e2e.py --self-test`.
- Run local generated dispatch bundle dry-runs for dynamic vector i32-vadd and
  i32-vsub.
- Run focused `ssh rvv` generated dispatch bundle invocation for at least one
  dynamic vector source-frontdoor route unless blocked by a precise remote or
  toolchain failure.
- Run bounded ref-scans for descriptor-only authority, explicit-only route
  misuse, stale op-family identity, and generic core RVV semantic branches.
- Run `git diff --check`, `git diff --cached --check`, and Trellis validation
  before finish/archive.

## Implementation Summary

- Production owner changed: `lib/Target/RVV/RVVScalarDispatch.cpp`.
- Added embedded RVV artifact contract validation during dispatch callable
  source embedding. `RVVScalarDispatch` now requires the RVVMicrokernel-emitted
  `rvv_microkernel_selected_source_identity` to match the dispatch-selected
  source identity summary before dispatch source/header/object export can
  succeed.
- Added validation that the embedded RVV callable source exports a matching
  `runtime_abi_invocation_contract`: callable symbol, runtime ABI kind/name,
  runtime glue role, ordered callable ABI roles, runtime element-count C name,
  and production owner must match the selected dispatch candidate and selected
  config contract.
- Added embedded runtime-length validation for fixed vector source extent and
  dynamic vector runtime AVL authority, keyed by the selected config contract.
- Dispatch source/header artifacts now print
  `dispatch_embedded_rvv_artifact_contract_consumed`, making the downstream
  contract consumption visible in generated artifacts.
- `scripts/rvv_scalar_dispatch_e2e.py` now parses the generated dispatch source
  as a consumer, checks embedded RVV selected-source identity against dispatch
  identity, checks embedded RVV runtime ABI invocation contract against
  compiler-emitted candidate metadata, records `embedded_rvv_artifact_contract`
  in evidence, and keeps fail-closed self-test coverage for stale runtime ABI
  contract data.
- Focused vector dynamic vadd/vsub bundle tests now require both the dispatch
  consumption marker and the embedded RVVMicrokernel runtime invocation
  contract in generated source artifacts.

## Validation Summary

- Build passed:
  `cmake --build build --target TianChenRVSupport TianChenRVTarget TianChenRVRVVTarget TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Rebuild after self-repair passed:
  `cmake --build build --target TianChenRVRVVTarget tcrv-translate tcrv-opt tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- C++ smoke passed:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed from `build/test`:
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`.
- Focused lit passed from `build/test`:
  `Target/RVVScalarDispatch`, `Target/RVVMicrokernel`,
  `Scripts/rvv-scalar-dispatch-bundle-e2e.test` (`46/46`).
- Script self-tests passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`,
  `python3 scripts/rvv_microkernel_e2e.py --self-test`.
- Local vector source-frontdoor generated bundle dry-runs passed:
  `codex-exported-contract-vadd-local` and
  `codex-exported-contract-vsub-local`.
- Evidence JSON for both local dry-runs records
  `embedded_rvv_artifact_contract`, including
  `rvv_microkernel_selected_source_identity` and
  `runtime_abi_invocation_contract`.
- Focused `ssh rvv` evidence passed for dynamic vector i32-vsub generated
  dispatch bundle invocation:
  `codex-exported-contract-vsub-ssh`, `status: success`,
  `ssh_evidence: true`, `ssh_evidence_verified: true`.
- Bounded ref-scan found changed contract consumption only in RVV target/script
  and focused tests. `hasRVV` hits remain existing RVV target-local helpers;
  no generic core pass or `tcrv.exec` RVV semantic branch was added.
- `git diff --check`, `git diff --cached --check`, and Trellis context
  validation passed before finish.

## Self-Repair Notes

- First C++ build caught accidental pointer-style binding of
  `std::optional` fixed/dynamic runtime extent contracts. The code now binds
  the optional value directly.
- First bundle FileCheck run exposed that the new
  `runtime_abi_invocation_contract` check was placed before the actual emitted
  RVVMicrokernel runtime ABI section. The checks were reordered to match
  generated artifact order.
- First script bundle lit run exposed that the new dispatch consumption marker
  included the literal `source_frontend_runtime_avl_authority`, causing
  non-source-frontdoor dry-runs to attempt parsing a source-frontdoor-only
  comment. The marker now uses the route-neutral
  `rvv_microkernel_runtime_length_contract` value while source-frontdoor routes
  still validate their actual `source_frontend_runtime_avl_authority` comment.

## Spec Update Judgment

No `.trellis/spec/` update is needed. This round did not introduce a new
dialect op, CLI surface, plugin protocol, schema family, or architecture rule.
It tightened an already-specified RVV target/runtime consumption path using the
existing selected source identity, selected config, runtime ABI role, runtime
AVL/VL, EmitC route, and descriptor-quarantine contracts.

## Definition Of Done

The source-frontdoor RVV vadd/vsub generated artifact path has a downstream
runtime/dispatch consumption closure: `RVVScalarDispatch` validates the
exported RVVMicrokernel artifact source identity and runtime ABI invocation
contract before dispatch artifacts are emitted, the e2e runner records and
invokes generated artifacts as a consumer, focused local and `ssh rvv` evidence
are recorded, the task is archived, and one coherent commit records the work.
