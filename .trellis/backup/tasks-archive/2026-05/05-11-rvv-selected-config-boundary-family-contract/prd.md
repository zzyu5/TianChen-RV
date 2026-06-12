# RVV selected config boundary family contract

## Goal

Create one explicit C++ compiler-owned selected RVV binary config contract for
the bounded finite RVV binary families. The contract must carry descriptor-local
dtype/operator metadata, compile-time selected RVV vector config, selected
variant identity, and runtime ABI/control names without conflating those layers,
then be consumed by active planning, lowering-boundary, emission-plan, and
target/export code where practical in this round.

## Background

The archived i64m1 add/sub/mul front-door dispatch evidence family has already
converged through `f2095ab`. Those rounds proved bounded runtime behavior on
`ssh rvv` for the existing front-door wrappers without making performance
claims. Another operator-only evidence task would repeat the same surface.

The current bottleneck is that several stages still reconstruct overlapping
selected RVV facts from separate strings: family descriptor, selected vector
shape metadata, selected-plan metadata, runtime ABI parameters, and dispatch
candidate metadata. This task should make the selected-config boundary an
explicit compiler contract consumed across finite families such as i32m2 and
i64m1.

## Requirements

- Add or refactor a C++ contract in the existing RVV Support/Target/Plugin
  area for bounded finite binary selected config.
- The contract must carry, at minimum:
  - descriptor-local dtype and arithmetic family/operator identity;
  - selected RVV compile-time SEW, LMUL, tail policy, mask policy;
  - selected vector shape, C vector type, intrinsic suffix, and setvl suffix;
  - selected variant identity when a selected path is known;
  - runtime ABI/control names for element count and dispatch availability guard;
  - descriptor-local element-count metadata as a separate finite descriptor
    value, not a runtime AVL/VL value.
- Reuse the same contract for at least one i32m2 path and one i64m1 path.
- Make at least two active compiler/export consumers use the contract instead
  of independently rebuilding the same selected config strings. Preferred
  consumers are RVV binary planning, selected lowering-boundary materialization
  or validation, selected emission-plan metadata, RVV microkernel export, and
  RVV+scalar dispatch export.
- Add fail-closed validation for descriptor dtype/operator versus selected
  config mismatch and stale shape metadata such as i32m2/i64m1 crossing a
  selected boundary.
- Preserve the distinction among:
  - hardware/capability facts, such as VLEN, march, and finite config
    capability ids;
  - compile-time selected RVV config, such as shape, SEW, LMUL, policies,
    vector type, and suffixes;
  - runtime SSA/control values, such as runtime element count, AVL, VL, and
    dispatch availability guard;
  - descriptor-local arithmetic-family metadata, such as dtype/operator,
    route ids, runtime ABI identity, and bounded descriptor element count.
- Keep RVV/scalar behavior plugin- and target-owned. Generic core passes must
  continue routing through plugin/target interfaces and must not branch on RVV,
  scalar, dtype, shape, family, vendor, or target-specific strings.
- Keep `tcrv.exec` compute-free. New or changed attributes may only be
  execution/lowering/emission metadata.

## Acceptance Criteria

- Positive C++ and/or lit/FileCheck coverage proves i32m2 and i64m1 both
  consume the same selected-config contract.
- Planning still selects the correct finite family and shape for i32m2 and
  i64m1, including descriptor-local dtype/operator identity and selected
  compile-time vector config.
- Lowering-boundary, emission-plan, and/or target/export consumers preserve the
  selected-config contract without re-deriving stale shape/type/suffix/runtime
  facts independently where this round touches them.
- Negative coverage fails closed for:
  - descriptor dtype/operator mismatching selected RVV config;
  - stale selected-shape metadata crossing the selected boundary, especially
    i32m2 versus i64m1;
  - runtime-control values being confused with compile-time selected config or
    descriptor-local element count.
- Diagnostics mention selected config mismatch generically from RVV
  plugin/target-owned code and do not add RVV family special cases to generic
  core passes.
- Existing i64-vadd, i64-vsub, and i64-vmul front-door evidence assumptions
  remain valid. This task does not need new `ssh rvv` evidence unless emitted
  runtime semantics change.
- Focused tests for changed plugin/target/transform behavior pass.
- `git diff --check` passes; `check-tianchenrv` is run if practical after
  focused checks pass.
- Trellis task validates, archives only when the active producer/consumer
  contract is complete, and one coherent commit records the module.

## Non-Goals

- No new runtime evidence task as the main deliverable.
- No performance, throughput, benchmark, or broad matrix claim.
- No generic RVV backend and no MLIR vector/scalable-vector lowering.
- No new arbitrary dtype, tensor, reduction, mask dataflow, predicated
  operation, or high-level compute family.
- No Python implementation of compiler internals. Python may only support
  runners or artifact parsing if needed.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout.
- No RVV, scalar, offload, IME, AME, vendor, or target-family branches in
  generic core passes.

## Validation Plan

- Build focused changed C++ targets, especially `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`, and target/export tests touched by
  the implementation.
- Run focused C++ tests for RVV binary planning, selected lowering-boundary,
  selected emission, and target artifact export behavior.
- Run focused lit/FileCheck tests for selected lowering-boundary,
  emission-readiness/materialized emission plans, LinalgToExec i32m2/i64m1
  front-door paths, and TargetArtifactBundleExport selected metadata.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if practical after focused tests pass.
- Do not claim new RVV runtime correctness unless real `ssh rvv` evidence is
  produced in this round.

## Completion Notes

- Added `RVVBinarySelectedConfigContract` under `Target/RVV` as the explicit
  compiler-owned boundary for finite RVV binary selected config.
- The contract now carries descriptor-local dtype/family/operator/lowering
  descriptor, compile-time selected shape/SEW/LMUL/tail/mask/vector
  type/suffix/setvl suffix, selected variant identity, descriptor-local
  element_count, and runtime control C names for element count and dispatch
  availability.
- RVV binary planning now produces the contract for i32m2 and i64m1 selected
  plans instead of storing only a raw shape pointer.
- RVV selected emission planning consumes the contract to materialize
  selected-plan metadata, including descriptor-local binary metadata and
  runtime-control-name metadata.
- RVV+scalar dispatch export consumes the same contract to validate selected
  metadata fail-closed and to print the selected binary config summary from
  the contract instead of rebuilding it from independent strings.
- Export-side descriptor element_count resolution now consumes selected-plan
  metadata first and validates it against variant-local metadata when both are
  present, preserving compatibility with existing materialized i32-vadd
  dispatch fixtures.
- Generic core passes were not given RVV/dtype/family/vendor branches, and no
  compute semantics were added to `tcrv.exec`.
- No new `ssh rvv` evidence was collected, and no new runtime correctness or
  performance claim was made.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit from `artifacts/tmp/tianchenrv-build/test` with filter
  `rvv-scalar-i32-vsub-dispatch-i32m2-generic-route|rvv-scalar-i64-vmul-dispatch-generic-route|linalg-i32-vsub-to-rvv-artifact|linalg-i64-vsub-to-rvv-artifact|linalg-i64-vmul-to-rvv-artifact|plan-linalg-i32m2-vsub|plan-linalg-i64-vmul|materialize-emission-plans-rvv-microkernel-family|rvv-binary-planning|rvv-extension-plugin`:
  10/10 selected tests passed.
- Focused lit for previous i32-vadd dispatch/export regressions:
  `rvv-scalar-i32-vadd-dispatch-object|emission-manifest-rvv-scalar-dispatch-artifacts|rvv-scalar-i32-vadd-dispatch-c|rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding|target-artifact-bundle-positive`:
  5/5 selected tests passed.
- Focused lit for the added runtime-control mismatch negative case:
  `rvv-scalar-i32-vsub-dispatch-i32m2-generic-route`: 1/1 selected test
  passed.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-11-rvv-selected-config-boundary-family-contract`: passed.

## Technical Notes

- Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-11-finite-binary-frontend-lowering-contract/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-frontdoor-dispatch-runtime-abi/prd.md`,
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-vsub-frontdoor-dispatch-ssh-evidence/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-i64m1-vmul-frontdoor-dispatch-ssh-evidence/prd.md`.
- The current selected vector-shape facts already live in
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`.
- The current finite family and descriptor-local ABI facts already live in
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h` and
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`.
- Current planning surfaces include
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h` and
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`.
- Current lowering/emission/export consumers include
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`, and
  `lib/Target/Builtin/RVVScalarDispatch.cpp`.

## Definition Of Done

The task is done when a compiler-owned selected-config contract is active in at
least one producer and at least two active consumers across i32m2 and i64m1,
focused positive and negative tests pass, generic core passes remain
target-neutral, Trellis validates and archives the task, and one coherent
commit records the completed module. If unfinished, leave the task open and
record the exact missing boundary: contract definition, planning producer,
lowering-boundary consumer, emission-readiness/emission-plan consumer, exporter
consumer, i32m2 coverage, i64m1 coverage, or mismatch diagnostics.
