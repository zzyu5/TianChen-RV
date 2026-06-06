# Stage2 RVV computed-masked strided dot resource ABI boundary

## Goal

Make `computed_masked_strided_input_widening_dot_reduce_add` use the same
provider-consumed low-precision resource discipline as the completed base
strided-input widening dot-reduce-add path:

```text
selected/pre-realized typed tcrv_rvv body
  -> computed-mask + strided low-precision contraction resource selection
  -> RVV provider-owned direct-contraction plan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact mirror validation
  -> generated-bundle ABI evidence
  -> ssh rvv correctness evidence when executable behavior is claimed
```

If any computed-mask fact, inactive-lane policy, strided memory fact,
dtype/SEW/LMUL/EMUL fact, runtime AVL/VL fact, ABI/header binding, or selected
resource candidate is missing or stale, the path must fail closed before route
or artifact acceptance.

## What I Already Know

- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief before source edits.
- Initial repo state was `/home/kingdom/phdworks/TianchenRV`, branch `main`,
  clean `git status --short`, with HEAD
  `9fc8a4cc rvv: resource strided dot route`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-06-rvv-strided-input-widening-dot-resource-path/`
  completed the unmasked base `strided_input_widening_dot_reduce_add` resource
  path, target mirror validation, generated-bundle metadata checks, and `ssh rvv`
  correctness evidence.
- The current computed-mask strided route already has explicit and pre-realized
  selected-body fixtures, route/artifact metadata for computed-mask and strided
  facts, dry-run generated-bundle tests, and a direct pre-realized fail-closed
  test.
- Production code currently requires low-precision resource selection for
  product dequant/dequant-clamp and the unmasked
  `StridedInputWideningDotReduceAdd` operation. It does not yet require that
  selection for `ComputedMaskStridedInputWideningDotReduceAdd`.
- The generated-bundle script self-test validates low-precision resource facts
  for unmasked strided-dot, but the computed-mask strided-dot self-test branch
  validates mask/stride/ABI facts without low-precision resource selection.
- Memory-derived context reinforces that this round must avoid evidence-only
  drift and advance the real Stage 2 low-precision contraction resource seam.

## Requirements

- Computed-mask strided-input widening dot-reduce-add must require a selected
  legal `RVVLowPrecisionContractionResourceSelection` before direct contraction
  provider route construction.
- Resource selection must be derived from the same typed selected-body/config,
  computed-mask, inactive-lane, strided memory, runtime ABI, runtime AVL/VL, and
  target capability facts used by the RVV provider. It must not come from route
  ids, artifact names, helper names, test names, descriptors, common EmitC, or
  emitted intrinsic spelling.
- Resource selection validation must check narrow source dtype/SEW/LMUL,
  widening product dtype/SEW/LMUL/EMUL, accumulator/result dtype/SEW/LMUL,
  tail/mask policy, computed-mask memory form, strided input memory form,
  reduction layout, runtime AVL source, runtime ABI order, target capability
  mirrors, legality, and vector register budget.
- Target artifact validation must consume provider-selected resource mirrors
  for the computed-mask strided route and fail closed for stale candidate
  metadata.
- Generated-bundle ABI evidence must expose the resource selection in boundary
  summaries and FileCheck assertions for both explicit and pre-realized selected
  bodies.
- The existing computed-mask predicate, inactive-lane zeroing, masked product,
  masked merge, stride, ABI, header, and runtime count evidence must remain
  intact.
- Executable correctness claims must be backed by `ssh rvv` evidence for
  runtime counts `0,1,16,17,257`, stride pairs `2:3` and `3:2`, and multiple
  mask/input patterns.

## Acceptance Criteria

- [x] PRD and task context identify the bounded module owner, read-first files,
      non-goals, and acceptance evidence.
- [x] Production route planning requires/derives a provider-consumed resource
      selection for `ComputedMaskStridedInputWideningDotReduceAdd` before
      `TCRVEmitCLowerableRoute` construction.
- [x] Plan and description validation reject missing/stale resource candidate,
      resource memory form, dtype/SEW/LMUL/EMUL, runtime AVL source, runtime ABI
      order, mask policy, and target capability mirrors.
- [x] Target artifact validation checks computed-mask strided low-precision
      resource mirrors and includes a focused stale-resource negative.
- [x] Explicit and pre-realized generated-bundle dry-runs expose resource
      metadata, provider boundary facts, mask/stride ABI facts, and harness
      coverage.
- [x] `ssh rvv` executable evidence passes for explicit and pre-realized
      selected bodies when runtime correctness is claimed.
- [x] Focused checks pass: build relevant tools/tests, RVV plugin test, target
      artifact export test, script self-test, focused lit/FileCheck or direct
      RUN-line reproductions, `git diff --check`, `git diff --cached --check`,
      and bounded old-authority scan over touched files/added lines.
- [x] Task status, journal/context notes, archive state, and final commit are
      truthful.

## Outcome

- Production route planning now treats `ComputedMaskStridedInputWideningDotReduceAdd`
  as a strided-input widening dot low-precision resource route, so direct
  provider planning consumes a selected `RVVLowPrecisionContractionResourceSelection`
  before constructing `TCRVEmitCLowerableRoute`.
- The RVV contraction plan owner now derives a distinct computed-mask strided
  resource candidate set, selected candidate, legality scope, selection reason,
  memory form, source/product/accumulator/result dtype and SEW/LMUL/EMUL,
  runtime AVL source, ABI order, mask/tail policy, register budget, and target
  capability mirror contract.
- Target artifact validation and header export now preserve and validate the
  computed-mask strided resource mirrors; explicit and pre-realized target tests
  include stale selected-candidate negatives that fail closed before artifact
  acceptance.
- The generated-bundle ABI script now includes computed-mask strided
  low-precision resource metadata in expected bundle artifacts, provider
  boundary summaries, dry-run FileCheck coverage, and self-test assertions.
- Real `ssh rvv` evidence passed for explicit and pre-realized selected bodies
  with runtime counts `0,1,16,17,257`, stride pairs `2:3` and `3:2`, two mask
  patterns, two input patterns, inactive lanes skipped, scalar seed/result
  behavior, and tail preservation.

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-strided-input-widening-dot-reduce-add'` from `build/test`
- Explicit generated-bundle dry-run for `computed_masked_strided_input_widening_dot_reduce_add`
- Pre-realized generated-bundle dry-run for `computed_masked_strided_input_widening_dot_reduce_add`
- Explicit generated-bundle `ssh rvv` run for runtime counts `0,1,16,17,257`
- Pre-realized generated-bundle `ssh rvv` run for runtime counts `0,1,16,17,257`
- `git diff --check`
- Bounded old-authority scan over touched files and added diff lines; no added
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door, or descriptor-driven route authority was found.

Self-repair performed:

- First generated-bundle dry-run used `llvm-readobj` from PATH and failed
  because the tool was not on PATH; reran with `/usr/lib/llvm-20/bin/llvm-readobj`.
- Focused lit initially exposed overly order-sensitive FileCheck checks for
  emission-plan/header metadata ordering; reordered the checks while preserving
  all resource/artifact fields.

## Out Of Scope

- No broad dot/reduction matrix.
- No dtype/LMUL clone batch.
- No runtime-scalar-cmp masked strided-dot expansion.
- No MAcc, product-dequant, product-reduce dequant-clamp, memory, segment2,
  compare/select, conversion, or unrelated mask route rework except as
  reference.
- No high-level Linalg/Vector/StableHLO frontend.
- No per-Linalg route authority.
- No source-front-door positive route.
- No common EmitC invention of RVV resource, mask, stride, dtype, or ABI
  semantics.
- No dashboard/report/index-only closeout.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task read:
  `.trellis/tasks/archive/2026-06/06-06-rvv-strided-input-widening-dot-resource-path/prd.md`.
- Main implementation files expected:
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused fixtures/tests from the brief:
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-fail-closed.test`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.

## Continuation Point If Larger Than One Round

If this cannot be completed safely in one round, leave the task open at the
exact missing owner: resource candidate derivation, provider-plan consumption,
target mirror validation, generated-bundle ABI evidence, fail-closed negative,
or `ssh rvv` correctness evidence.
