# RVV plugin proposal planning descriptor exit

## Goal

Make direct RVV binary proposal planning derive selected source authority from typed RVV family/body evidence only. Legacy `tcrv_rvv.lowering_descriptor` text may remain only as optional mirror or cross-check metadata after typed authority is known; it must not produce a supported direct RVV binary proposal or selected plan by itself.

## Requirements

* Remove, bypass, or quarantine the direct legacy descriptor planning route in RVV binary planning.
* Ensure direct-family resolution derives proposal identity from typed RVV microkernel family/body evidence, not from `tcrv_rvv.lowering_descriptor`.
* Accept a matching legacy descriptor only as a mirror after typed authority has established family, dtype, shape, route, ABI, capabilities, vector config, and emitted body.
* Reject a stale descriptor mirror fail-closed with diagnostics that name the typed authority boundary.
* Ensure descriptor-only direct RVV input cannot select family, dtype, shape, route id, artifact kind, callable ABI, required capabilities, selected vector config, or emitted body.
* Keep descriptorless i32 and i64 RVV proposal planning, selected lowering boundary materialization, target artifact export, and dispatch/bundle behavior coherent.
* Rewrite or remove tests that encode descriptor-driven direct proposal planning as production behavior.
* Update durable Trellis spec text only where it records descriptor text as mirror metadata after typed authority, not plugin planning authority.

## Acceptance Criteria

* [x] Focused C++ test proves descriptor-only direct RVV planning no longer yields a supported selected plan.
* [x] Focused C++ test proves typed-body planning still yields the expected family/shape/route without reattaching descriptor compute authority.
* [x] Focused C++ test proves a matching legacy descriptor mirror is accepted only after typed authority.
* [x] Focused C++ test proves a stale legacy descriptor mirror is rejected.
* [x] Existing focused RVV extension plugin and selected lowering boundary tests remain coherent.
* [x] Relevant linalg-to-RVV artifact lit tests still reflect typed body authority.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes if the existing build tree is usable; otherwise the exact blocker is recorded and narrower affected tests are run.
* [x] Trellis task state validates and is archived only if the descriptor-exit behavior is complete.

## Definition of Done

* Production RVV plugin planning is descriptor-exited for direct RVV binary proposals.
* Tests cover descriptor-only failure, typed-body success, mirror acceptance, and mirror rejection.
* Relevant specs and task context reflect the durable boundary.
* One coherent commit is created if the task is complete.

## Out of Scope

* New arithmetic families, dtypes, LMULs, tuning, cost model, scheduler, or runtime ABI expansion.
* Python compiler implementation or descriptor-to-C export.
* New generic compute semantics in `tcrv.exec`.
* Standalone RVV runtime/correctness/performance evidence; no `ssh rvv` unless this round makes such a claim.
* Broad smoke matrices unrelated to the changed planning boundary.

## Technical Notes

* Starting evidence: no active Trellis task, clean worktree, HEAD `85e791d feat(rvv): make i64 target export body authoritative`.
* Read first:
  * `.trellis/spec/index.md`
  * `.trellis/spec/architecture/unified-riscv-mlir.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  * `.trellis/tasks/archive/2026-05/05-12-rvv-i64-target-export-body-authority/prd.md`
* Initial code focus:
  * `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`
  * `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  * `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  * `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`
  * `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* Initial test focus:
  * `test/Plugin/RVVBinaryPlanningTest.cpp`
  * `test/Plugin/RVVExtensionPluginTest.cpp`
  * `test/Plugin/RVVBinarySelectedLoweringBoundaryTest.cpp`
  * `test/Plugin/RVVBinaryVariantLegalityTest.cpp`
  * relevant `test/Transforms/LinalgToExec/*rvv*artifact*.mlir`
  * relevant `test/Target/RVVMicrokernel/*`

## Task Status Notes

* Created `.trellis/tasks/05-12-rvv-plugin-proposal-planning-descriptor-exit` because no current Trellis task existed and the user brief explicitly requested this task title.
* Replaced the direct legacy descriptor candidate path in `resolveRVVBinaryFamilyForProposal` with typed-body-first resolution plus post-typed-authority legacy descriptor mirror validation.
* Descriptor-only direct RVV planning metadata now fails closed with a recoverable proposal decline and cannot produce a supported direct RVV binary proposal plan.
* Matching legacy `tcrv_rvv.lowering_descriptor` metadata is accepted only after typed RVV microkernel body authority is known; stale descriptor mirrors and stale selected-shape mirror metadata fail closed with diagnostics naming typed authority.
* Focused C++ checks passed:
  * `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test -j2`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
  * `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
  * `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-variant-legality-test -j2`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
  * `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
* Focused lit checks passed:
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'linalg-i64-vadd-to-rvv-artifact|linalg-i64-vsub-to-rvv-artifact|linalg-i64-vmul-to-rvv-artifact|linalg-i32-vsub-to-rvv-artifact|linalg-i32-vmul-to-rvv-artifact|rvv-microkernel-auto-materialization|rvv-microkernel-i64-vadd|rvv-microkernel-i64-vsub|rvv-microkernel-i64-vmul|rvv-microkernel-family-sub|rvv-microkernel-family-mul'` from `artifacts/tmp/tianchenrv-build/test`, 15/15 passed.
  * `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv.*artifact|linalg-i32-vadd-to-exec'` from `artifacts/tmp/tianchenrv-build/test`, 8/8 passed.
* Full check passed: `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`, 209/209 tests passed.
* `git diff --check` and `git diff --cached --check` passed.
* No `ssh rvv` run was performed, so this task makes no RVV runtime, correctness, or performance claim.
