# RVV microkernel body selected-config verifier

## Goal

Add one coherent RVV plugin/target-owned verifier/preflight module that checks
the structured `tcrv_rvv` binary microkernel body against the selected finite
RVV descriptor/config and runtime ABI role contract before RVV source, header,
object, or bundle artifact export can succeed.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Session start worktree was clean at `a9fa442 feat(rvv): export selected vector capability metadata`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes brief.
- The previous task
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-vector-capability-metadata-contract/`
  is complete and must remain archived.
- The selected vector-shape metadata path already carries selected SEW, LMUL,
  tail-policy, mask-policy, vector spelling, capability ids, capacity metadata,
  and runtime AVL/VL boundary metadata through selected emission plans.
- The current RVV dialect, plugin emission planning, and target exporter contain
  overlapping body checks, but they are not a single reusable preflight owner.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only remain tooling.
- Keep RVV semantic validation local to RVV plugin/target code. Do not add RVV
  branches to generic artifact routing or `tcrv.exec`.
- Validate the selected finite descriptor/config against the actual structured
  `tcrv_rvv` microkernel body:
  - `tcrv_rvv.setvl` SEW, LMUL, and policy agree with the selected shape.
  - `tcrv_rvv.with_vl` consumes the exact setvl VL token and repeats matching
    SEW, LMUL, and policy metadata.
  - finite load/arithmetic/store ops match the selected family and dtype/shape.
  - i32 routes use only i32m1/i32m2 tokens consistent with selected LMUL.
  - i64 routes use only i64m1 tokens for selected SEW64/LMUL m1.
  - unsupported family/shape combinations fail closed.
- Validate runtime ABI role binding in the body:
  lhs load, rhs load, and output store roles must be present, unique where
  required, role-correct, and consistent with the IR-backed callable ABI plan.
- Preserve the layering between hardware/capability metadata, compile-time
  selected vector config, runtime AVL/VL values, and descriptor-local
  `element_count`; diagnostics should name the stale or missing layer.
- Wire the verifier into the existing RVV selected emission planning and target
  artifact export preflight path so malformed bodies cannot export even when
  selected-plan metadata is present.
- Cover existing bounded finite binary RVV routes, including current i32/i32m2
  and i64m1 routes where supported by descriptors.

## Acceptance Criteria

- [x] Trellis task exists, is current, and validates before finish/archive.
- [x] A C++ RVV-owned verifier/preflight module exists and is consumed by the
      RVV selected emission/export path.
- [x] Valid existing i32 and i64 bounded RVV binary microkernel export still
      works.
- [x] Negative coverage proves stale or mismatched body structure fails before
      artifact output, including wrong setvl SEW, wrong LMUL, stale tail/mask
      policy, with_vl not consuming the setvl VL token, wrong i32/i64 vector
      token family, missing or duplicate load/store buffer role, and mismatched
      runtime ABI role binding.
- [x] At least one generic artifact/export route exercises the verifier through
      the same candidate preflight path used by bundle/export tests, not only a
      detached helper unit test.
- [x] No RVV runtime, correctness, throughput, latency, or performance claim is
      made without `ssh rvv` evidence.
- [x] `git diff --check` passes.
- [x] Focused RVV/plugin/target tests pass.
- [x] `check-tianchenrv` passes if practical.

## Minimal Validation Plan

- `git diff --check`
- Focused build targets as needed:
  `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- Focused C++ tests:
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  and `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused lit tests under `test/Target/RVVMicrokernel/` and generic artifact or
  bundle routes touched by the verifier.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-microkernel-body-selected-config-verifier`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  when practical.

## Out of Scope

- No generic RVV backend.
- No new family matrix beyond existing finite binary routes unless required for
  a focused verifier test.
- No emitted compute semantic change unless required to expose active verifier
  behavior.
- No RVV-specific branch in generic target artifact routing or generic core
  passes.
- No compute semantics in `tcrv.exec`.
- No Python-owned compiler internals.
- No runtime correctness/performance claim.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-vector-capability-metadata-contract/prd.md`.
- Initial source inspection found overlapping checks in
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`, and
  `lib/Target/RVV/RVVMicrokernel.cpp`. This task should consolidate the
  selected-body/export contract into an RVV-owned verifier rather than moving
  target semantics into generic routes.

## Completion Notes

- Added `RVVBinaryMicrokernelBodyVerifier` under `include/` and `lib/Target/RVV/`.
- Wired `RVVMicrokernel.cpp` export validation through that verifier for finite
  i32/i32m2 and i64m1 binary routes.
- Added generic target source artifact negative coverage for stale
  descriptor-local `element_count` versus selected variant
  `tcrv_rvv.element_count`.
- Updated existing policy and i32m2 body mismatch diagnostics to assert the new
  verifier path.
- Validation run locally only; no `ssh rvv` runtime, correctness, throughput,
  latency, or performance claim was made.

## Continuation Rule If Unfinished

Keep the task open and record the exact missing verifier case, failing test,
and next source function/file to continue from. Do not archive or commit a
misleading completed task.
