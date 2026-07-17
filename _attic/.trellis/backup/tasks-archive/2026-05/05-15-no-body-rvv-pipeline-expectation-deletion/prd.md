# No-body RVV pipeline expectation deletion

## Goal

Delete the remaining public pipeline, lit, probe, and artifact expectations that
metadata-only RVV capability inputs can synthesize or materialize
`rvv_first_slice` after the no-body default finite-family proposal path was
removed. The public TianChen-RV pipeline must now reflect explicit typed RVV
extension-family body authority: no-body RVV inputs decline, report no viable
RVV proposal, or fall through to scalar fallback semantics as appropriate.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `975ebcc`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The immediately previous task
  `.trellis/tasks/archive/2026-05/05-15-rvv-no-body-default-finite-family-proposal-deletion/prd.md`
  removed the production RVV no-body default finite-family fallback in
  `RVVBinaryPlanning.cpp`.
- That previous task's full `check-tianchenrv` evidence exposed ten remaining
  public test failures that still expect metadata-only RVV inputs to synthesize
  or materialize `rvv_first_slice`.
- `.trellis/spec/index.md` keeps the project in the C++/MLIR/LLVM/TableGen
  stack and rejects descriptor-driven computation as long-term architecture.
- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks and independent backend dialect framing.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says the RVV plugin must not
  create finite RVV binary proposals from no-body `tcrv.exec.kernel` inputs,
  deleted frontend-lowering metadata, or finite-family registry metadata alone.

## Requirements

- Rewrite or delete the ten listed broad lit/probe/artifact expectations that
  protect no-body RVV first-slice proposal or materialization.
- Keep capability-only/no-body RVV inputs fail-closed, declined, no-viable, or
  scalar-fallback according to the surrounding pass semantics.
- Convert positive RVV public pipeline tests to explicit typed RVV
  extension-family body fixtures where a positive `rvv_first_slice` assertion
  is still required.
- Adjust directly coupled production/default pass behavior only if a focused
  failing test proves it still treats RVV capability metadata without typed body
  as selected or materialized RVV authority.
- Delete or rewrite obsolete expected diagnostics, artifact assertions, and
  comments that describe metadata-only RVV first-slice synthesis as current
  behavior.
- Record any new deletion-exposed gap without restoring no-body RVV selection.

## Acceptance Criteria

- [x] The ten previously failing focused lit tests no longer expect
      metadata-only RVV inputs to synthesize or materialize `rvv_first_slice`.
- [x] No-body RVV inputs in the public execution-planning, materialization,
      selection, probe, and artifact surfaces fail closed, decline, select no
      viable RVV proposal, or fall through to scalar fallback as appropriate.
- [x] Positive RVV cases that still require `rvv_first_slice` use explicit
      typed RVV body fixtures instead of capability metadata alone.
- [x] No default finite-family selector, descriptor-driven compute path,
      direct-C semantic exporter, compatibility shim, helper wrapper, or new
      architecture implementation is introduced.
- [x] Focused ref-scan over active code, tests, and specs reports remaining
      `default proposal`, `default_i32_vadd`, `no-body`,
      `metadata-only first slice`, and `rvv_first_slice` uses truthfully,
      distinguishing explicit typed-body positive cases from old authority.
- [x] The ten listed lit tests pass as a focused set, or any remaining failure
      is reported as a deletion-exposed rebuild gap without restoring old
      behavior.
- [x] `check-tianchenrv` is run if practical; any remaining failures are
      classified without reintroducing no-body RVV first-slice synthesis.
- [x] Trellis task status, context, journal, archive state, and one coherent
      commit are produced if the round completes.

## Directly Failing Surfaces To Inspect

- `test/Scripts/rvv-probe-to-mlir.test`
- `test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir`
- `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
- `test/Transforms/ExecutionPlanning/execution-planning-pipeline-no-viable.mlir`
- `test/Transforms/ExecutionPlanning/execution-planning-pipeline-rerun-mismatch.mlir`
- `test/Transforms/VariantMaterialization/plugin-variant-materialization-builtin.mlir`
- `test/Transforms/VariantMaterialization/plugin-variant-materialization-rvv-missing-properties.mlir`
- `test/Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape-invalid.mlir`
- `test/Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape.mlir`
- `test/Transforms/VariantSelection/variant-selection.test`

## Non-goals

- No Common EmitC lowering implementation.
- No new RVV lowering.
- No new RVV ops.
- No new artifact/export route.
- No runtime ABI modeling.
- No `ssh rvv` runtime/correctness/performance evidence.
- No helper wrappers, compatibility selectors, or replacement default family.
- No unrelated scalar or direct-C cleanup unless directly required by one of the
  listed failing tests.

## Minimal Evidence

- Run the ten previously failing lit tests as a focused set after changes.
- Run focused RVV planning/plugin tests from the previous task if a directly
  coupled path is touched.
- Run a focused ref-scan over active code/tests/specs for
  `default proposal`, `default_i32_vadd`, `no-body`,
  `metadata-only first slice`, and `rvv_first_slice` uses that still imply
  capability-only authority.
- Run `check-tianchenrv` if practical.
- Run `git diff --check`.
- Run `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- This task is public-surface cleanup after production no-body RVV family
  selection was deleted. It should not rebuild the replacement RVV lowering
  stack.
- Explicit typed RVV microkernel bodies are valid authority for positive
  `rvv_first_slice` cases. Capability metadata alone is not.
- Scalar fallback may be the correct expectation for mixed public pipelines
  where RVV declines and another plugin/path remains viable.

## Completion Evidence

- Rewrote `test/Scripts/rvv-probe-to-mlir.test` so probe replay with scalar
  fallback expects scalar-only planning after RVV no-body decline, and i64
  target-profile replay fails closed during execution planning instead of
  flowing to source artifact export.
- Updated no-viable public pipeline diagnostics in
  `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
  and
  `test/Transforms/ExecutionPlanning/execution-planning-pipeline-no-viable.mlir`
  to assert the explicit typed-body requirement and no-body rebuild gap.
- Added explicit typed `tcrv_rvv.i32_vadd_microkernel` body authority to
  property/shape tests that still need to exercise RVV property or selected
  shape behavior:
  `plugin-variant-materialization-rvv-missing-properties.mlir`,
  `plugin-variant-materialization-rvv-selected-shape-invalid.mlir`, and
  `plugin-variant-materialization-rvv-selected-shape.mlir`.
- Reworked positive public RVV selection/materialization fixtures in
  `plugin-variant-materialization-builtin.mlir`,
  `rvv-scalar-i32-vadd-dispatch-generic-route.mlir`, and
  `VariantSelectionTest.cpp` so `rvv_first_slice` is backed by explicit typed
  RVV body authority rather than metadata-only capability synthesis.
- No production/default pass behavior needed to be changed in this round; the
  current production no-body RVV behavior already fails closed after commit
  `975ebcc`.
- Focused ten-test lit run passed:
  `Scripts/rvv-probe-to-mlir.test`,
  `Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir`,
  `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  `Transforms/ExecutionPlanning/execution-planning-pipeline-no-viable.mlir`,
  `Transforms/ExecutionPlanning/execution-planning-pipeline-rerun-mismatch.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-builtin.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-rvv-missing-properties.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape-invalid.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape.mlir`,
  and `Transforms/VariantSelection/variant-selection.test`.
- Focused RVV guardrails passed:
  `cmake --build build --target tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-binary-planning-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Full `cmake --build build --target check-tianchenrv -j2` passed with
  114/114 tests.
- `git diff --check` passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-no-body-rvv-pipeline-expectation-deletion`
  passed.
- Focused ref-scan over active code/tests/specs found no active
  `default_i32_vadd` RVV path and no RVV `default proposal` authority. Remaining
  `no-body` matches are the explicit fail-closed diagnostic/tests or unrelated
  scalar fallback/spec text. Remaining `rvv_first_slice` matches are explicit
  typed-body fixtures, lower-level RVV dialect/legality/emission tests, or
  negative checks that assert no RVV first slice.
- Spec update judgment: no `.trellis/spec/` update was needed because
  `.trellis/spec/extension-plugins/rvv-plugin.md` already states that finite
  RVV binary proposals require explicit typed body authority; this round aligned
  public tests and artifact expectations to that existing contract.
