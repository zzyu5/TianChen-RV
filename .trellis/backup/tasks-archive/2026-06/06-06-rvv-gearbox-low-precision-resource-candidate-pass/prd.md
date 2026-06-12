# Stage2 RVV Gearbox low-precision resource-candidate pass pipeline

## Goal

Introduce a bounded RVV plugin-local Gearbox/resource-aware compiler seam for
one low-precision direct-contraction family:
selected typed `widening_product_reduce_dequantize_f32` body -> static
resource candidate construction/selection facts -> provider-consumed route
facts -> RVV-owned target validation mirrors.

This round must make the calibrated Gearbox direction real in compiler source.
The first slice is a static candidate builder/selector, not a full autotuner,
runtime profile database, performance claim, or llama.cpp parity result.

## What I already know

- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes Direction Brief before source edits.
- HEAD before the task was `ab42a018 docs: calibrate rvv gearbox wording`; the
  worktree was clean.
- The currently registered Gearbox pass is
  `--tcrv-rvv-materialize-gearbox-schedules`.
- Current `RVVGearboxSchedules.cpp` only recognizes the bounded
  `dequantize_i32_to_f32` realized `with_vl` body and materializes
  `tcrv_rvv.gearbox.*` facts.
- Low-precision direct-contraction resource selection already has provider and
  target-validation structures for `widening_product_reduce_dequantize_f32`,
  including dtype/SEW/LMUL/EMUL, unroll, accumulator count, vsetvl region count,
  peak live vector groups, vector register budget, runtime AVL/ABI order, and
  legality/rejection fields.
- That low-precision resource selection is currently derived inside the
  contraction route-family provider; there is no separate MLIR pass that
  materializes or validates `tcrv_rvv.low_precision_resource.*` facts on the
  selected body.
- Pre-realized product-dequant fixtures are realized by
  `--tcrv-materialize-selected-lowering-boundaries` into explicit
  `setvl/with_vl/load/load/widening_product/standalone_reduce/dequantize/store`
  structure before route planning.

## Requirements

- Extend RVV plugin-local Gearbox pass ownership so it can materialize bounded
  static low-precision resource candidate and selected-candidate facts for the
  `widening_product_reduce_dequantize_f32` family.
- Derive pass facts from typed selected body/config/runtime facts, not from
  route ids, artifact names, test names, ABI strings, q8/q4 labels, or common
  EmitC semantics.
- Keep the slice bounded to the direct-contraction dequantize representative;
  do not expand to clamp, MAcc, mask, memory, high-level frontend, full
  autotuning, runtime profile caches, or performance evidence in this round.
- Make provider planning consume or validate the pass-produced resource facts
  before accepting the low-precision direct-contraction route. Provider may
  still fill selected target capability mirrors from selected target facts.
- Preserve existing target artifact validation so artifact metadata remains a
  mirror of provider-selected facts and stale metadata-only candidates fail.
- Make the pass idempotent: a second run must preserve matching facts, while
  stale or unsupported facts fail closed with targeted diagnostics.

## Acceptance Criteria

- [ ] `--tcrv-rvv-materialize-gearbox-schedules` (or a bounded extension of the
      RVV Gearbox pass source) materializes
      `tcrv_rvv.low_precision_resource.*` facts for one valid
      `widening_product_reduce_dequantize_f32` selected body.
- [ ] The pass rejects stale selected-candidate/resource facts instead of
      overwriting them silently.
- [ ] The pass rejects unsupported resource facts such as an impossible vector
      register budget / peak-live-vector-group relation.
- [ ] Contraction route-family provider planning consumes or validates the
      pass-produced resource facts before route acceptance; missing or stale
      facts fail closed for this representative.
- [ ] Target artifact validation continues to reject stale
      `tcrv_rvv.low_precision_resource.*` artifact metadata mirrors.
- [ ] lit coverage includes a positive pass fixture and at least one
      fail-closed stale/missing/unsupported resource fact fixture.
- [ ] Focused build/tests pass:
      `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`,
      `build/bin/tianchenrv-rvv-extension-plugin-test`,
      `build/bin/tianchenrv-target-artifact-export-test`, and targeted `lit`.
- [ ] `git diff --check` and `git diff --cached --check` pass.
- [ ] A bounded old-authority scan over touched files/added lines shows no new
      positive legacy `i32m1`, descriptor, source-front-door, direct-C,
      route-id, artifact-name, status, or common-EmitC RVV semantic authority.

## Definition of Done

- The task is finished/archived only if the production compiler seam, tests,
  Trellis notes, and one coherent commit are complete.
- If the full pass/provider/validation chain is too large, this round must
  finish the static build/select plus provider-consumption seam and leave an
  exact continuation point.

## Out of Scope

- No docs-only closeout.
- No broad Gearbox architecture essay.
- No full autotuner, runtime/JIT/profile cache, assembly feedback cache, global
  tuning dashboard, or performance/parity claim.
- No generated-bundle or `ssh rvv` runtime evidence unless executable artifact
  behavior changes.
- No dtype/LMUL clone batch, source-front-door positive route, high-level
  Linalg/Vector/StableHLO frontend, q8/q4-named route authority, or common
  EmitC invention of RVV semantics.
- No unrelated clamp, MAcc, mask, reduction, memory, or future-plugin expansion.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/guides/index.md` plus capability/plugin/compute guides.
- Current calibration artifact read:
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`.
- Related archived tasks read:
  `.trellis/tasks/archive/2026-06/06-06-rvv-gearbox-worktree-coherence/` and
  `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-gearbox-pass-pipeline-wording/`.
- Initial code evidence:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Transforms/Passes.td`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Transforms/RVV/rvv-gearbox-dequantize-i32-to-f32.mlir`, and
  `test/Target/RVV/*widening-product-reduce-dequantize-f32*.mlir`.
