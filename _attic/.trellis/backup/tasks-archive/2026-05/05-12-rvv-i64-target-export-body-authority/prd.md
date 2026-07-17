# rvv-i64-target-export-body-authority

## Goal

Rewire the RVV i64 target artifact export path so direct source/header/object
export derives family authority from the selected typed
`tcrv_rvv.i64_vadd_microkernel`, `tcrv_rvv.i64_vsub_microkernel`, or
`tcrv_rvv.i64_vmul_microkernel` body attached to the selected path. Legacy
`tcrv_rvv.lowering_descriptor` may remain only as optional mirror metadata
validated after the typed body is known; it must not select the family,
intrinsic, callable ABI, route id, artifact kind, component group, or emitted C
body.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `17ad405 feat(rvv): quarantine descriptor-only i64 materialization`.
- Worktree was clean at task creation, and `.trellis/.current-task` was absent.
- The previous task
  `.trellis/tasks/archive/2026-05/05-12-rvv-i64-binary-descriptor-exit-closure/`
  is finished and archived and must not be reopened.
- Current specs keep the default route as extension family ops -> common EmitC
  route -> generated intrinsic/runtime C/C++ artifacts. Descriptor-driven
  computation and descriptor-to-C authority remain invalid long-term
  architecture.
- Repository evidence shows `lib/Target/RVV/RVVMicrokernel.cpp` still resolves
  an i64 export family by consulting `tcrv_rvv.lowering_descriptor` before
  scanning selected typed i64 microkernel bodies.
- The module owner for this task is RVV i64 target artifact export body
  authority for direct source/header/object and bundle component validation.

## Requirements

- Discover the selected typed i64 RVV microkernel body first for each selected
  RVV path and use it as the source of finite family, dtype, arithmetic op,
  intrinsic config, callable ABI, route metadata, selected-plan consistency,
  component group, function stem, and generated source behavior.
- If a selected variant also carries `tcrv_rvv.lowering_descriptor`, validate it
  only after the typed i64 body family is known. Matching descriptor metadata is
  allowed only as a non-authoritative legacy mirror.
- A stale descriptor beside a typed i64 body must fail closed before source,
  header, object, or bundle output, with diagnostics stating that the typed body
  is authoritative.
- Descriptor-only i64 target export must fail before output and must not select
  `i64-vadd`, `i64-vsub`, or `i64-vmul` from descriptor text.
- Remove, rewrite, or quarantine any helper/test behavior that encodes
  descriptor-first i64 family selection.
- Keep i32 behavior working. Do not expand this task into a broad i32 rewrite
  unless a shared helper cleanup requires it.

## Non-Goals

- No new arithmetic family, dtype, LMUL, tuning, runtime scheduler, or broad
  smoke matrix.
- No Python compiler implementation; Python remains only runner/artifact
  tooling.
- No descriptor-to-C exporter and no descriptor-owned compute semantics.
- No changes to `tcrv.exec` compute semantics.
- No RVV runtime, correctness, or performance claim unless a fresh real
  `ssh rvv` run is performed.
- No standalone ssh evidence package, metadata-only closeout, helper-only
  cleanup, or report-only finish.

## Acceptance Criteria

- [x] RVV i64 target export selects the finite export family from the selected
      typed i64 microkernel body before any descriptor mirror validation.
- [x] Matching `tcrv_rvv.lowering_descriptor` mirror metadata beside a typed
      i64 body is accepted but cannot change route id, artifact kind,
      intrinsic spelling, function stem, C types, callable ABI, selected-plan
      metadata, or component group.
- [x] Stale `tcrv_rvv.lowering_descriptor` beside a typed i64 body fails closed
      before source/header/object/bundle output and names typed body authority
      in the diagnostic.
- [x] Descriptor-only i64 export fails closed before output and does not select
      an i64 family from descriptor text.
- [x] Focused C++ coverage in `test/Target/TargetArtifactExportTest.cpp` proves
      body-first selection, descriptor-only rejection, stale descriptor
      rejection, and matching mirror acceptance for the bounded i64 families.
- [x] Focused lit/FileCheck coverage for at least i64 sub/mul source paths
      continues to prove externally visible typed-body export behavior through
      `tcrv-opt` / `tcrv-translate`.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Trellis task validation passes for the active task, and for the archived
      path if the task is finished.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      runs if the existing build tree is usable; otherwise the exact blocker is
      recorded and focused affected tests pass.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/supervision-loop.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Prior PRD read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-i64-binary-descriptor-exit-closure/prd.md`
- Source entry points inspected:
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`
  - `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`

## Definition Of Done

The task is done only when the RVV i64 target export family-resolution path is
body-first in production code, descriptor-only and stale-descriptor paths fail
closed before artifact output, focused checks pass, Trellis state is truthful,
the task is finished/archived only if complete, and one coherent commit records
the completed module. If unfinished, leave the task open with the exact
remaining descriptor-authority function/route and the smallest continuation
point.

## Task Status Notes

- Created this task because no current Trellis task existed and the user brief
  explicitly requested `rvv-i64-target-export-body-authority`.
- Initial audit found the descriptor-first authority point in
  `resolveSelectedI64FamilyForPath` inside `lib/Target/RVV/RVVMicrokernel.cpp`.
- Rewired `resolveSelectedI64FamilyForPath` so it scans selected typed i64
  microkernel bodies first, returns the typed body family, and treats
  `tcrv_rvv.lowering_descriptor` only as post-body mirror metadata.
- Added fail-closed i64 diagnostics for stale descriptor mirrors and
  descriptor-only i64 export, both explicitly naming typed body authority.
- Added C++ target artifact export coverage for matching descriptor mirror
  acceptance, stale descriptor rejection, and descriptor-only i64 rejection.
- Added lit coverage to i64 vsub/vmul RVV microkernel source tests for matching
  descriptor mirror acceptance and stale descriptor mirror rejection through
  `tcrv-opt | tcrv-translate`; updated the existing i64 vadd linalg/export
  stale-descriptor diagnostic expectation to the new body-authoritative text.
- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` to
  record the RVV i64 source/header/object body-authority and descriptor-mirror
  fail-closed contract.
- Focused checks passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test -j2`;
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`;
  lit for `rvv-microkernel-i64-vadd.mlir`, `rvv-microkernel-i64-vsub.mlir`,
  `rvv-microkernel-i64-vmul.mlir`,
  `rvv-scalar-i64-vsub-dispatch-generic-route.mlir`,
  `rvv-scalar-i64-vmul-dispatch-generic-route.mlir`,
  `plan-linalg-i64-vsub-and-export-target-artifact-bundle.mlir`, and
  `plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`; and focused
  lit for `linalg-i64-vadd-to-rvv-artifact.mlir`.
- `git diff --check`, `git diff --cached --check`, active Trellis task
  validation, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed; full lit reported 209/209 tests passing.
- `clang-format` / `clang-format-20` were not available on PATH; compilation,
  focused tests, full check, and diff whitespace checks passed.
- No real `ssh rvv` run was performed, so this task makes no RVV runtime,
  correctness, or performance claim.
