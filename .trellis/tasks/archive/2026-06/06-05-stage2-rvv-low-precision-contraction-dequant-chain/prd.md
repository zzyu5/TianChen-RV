# Stage2 RVV low-precision contraction-to-dequant f32 chain

## Goal

Implement one production route-supported selected-body chain that composes the
existing typed RVV low-precision product-reduction path with the existing
i32-to-f32 runtime-scale dequantization path:

```text
signed i8 unit-load operands
  -> widening product
  -> i16 intermediate
  -> signed i16-to-i32 widening reduction
  -> i32 accumulator/result boundary
  -> runtime-scale i32-to-f32 dequantize
  -> f32 output store/runtime boundary
```

The route must be derived from the selected typed `tcrv_rvv` body/config and
runtime ABI usage, not from route ids, artifact names, ABI strings, test names,
metadata mirrors, q8/q4 labels, descriptor residue, or common EmitC inference.

## What I Already Know

- Commit `4fa75433` closed executable ABI evidence for standalone
  i32-to-f32 dequantization.
- Earlier commits closed route/executable support for low-precision
  product-reduction primitives.
- The next bottleneck is the selected-body composition boundary: one typed RVV
  body must carry contraction facts and dequant facts together, and the RVV
  provider must build a coherent lowerable route or fail closed.
- The current authority chain remains:
  `tcrv.exec` envelope -> selected typed low-level `tcrv_rvv` body -> RVV
  plugin legality/route provider -> `TCRVEmitCLowerableRoute` -> common EmitC
  materializer -> target artifact -> optional `ssh rvv` evidence.

## Requirements

- Production code changes are required before task closure.
- The selected typed body must structurally carry:
  - i8 source signedness and unit-load operands;
  - i16 product/intermediate type;
  - signed i16-to-i32 widening reduction semantics;
  - explicit i32 accumulator/result boundary;
  - runtime scale binding consumed by dequantization;
  - f32 dequantized output value and output store/runtime boundary;
  - SEW/LMUL/policy, AVL/VL, operand/result stores, and chain ordering.
- RVV plugin route planning must compose product-reduction and dequant facts
  from typed structure.
- Target artifact validation must reject missing scale, dtype-chain mismatch,
  missing i32 accumulator/result boundary, stale mirrors, unsupported config,
  and route-string authority.
- Focused positive and negative tests must prove the composed route behavior.
- If executable closure is feasible after route-supported production support,
  add focused generated-bundle/`ssh rvv` correctness evidence; otherwise stop
  at route-supported + generated artifact validation and leave an exact
  continuation point.

## Non-goals

- No zero-point/clamp expansion.
- No q8/q4/llama-specific route.
- No high-level Linalg/frontend scope.
- No handwritten C demo as the main deliverable.
- No compatibility wrapper preserving legacy i32m1 authority.
- No dtype/LMUL clone batch.
- No broad smoke matrix or dashboard/report-only task.
- No repeated standalone product-reduction or standalone dequant evidence as
  the main outcome.

## Acceptance Criteria

- [ ] A selected typed RVV body fixture covers the full contraction-to-dequant
      f32 chain.
- [ ] RVV route planning/provider production code recognizes and validates the
      composed chain from typed body facts.
- [ ] Common EmitC remains neutral; RVV-specific semantics stay in the RVV
      plugin/provider/route-family owner.
- [ ] Target artifact validation accepts the composed route and rejects missing
      scale, dtype-chain mismatch, missing accumulator/result boundary, stale
      provider mirrors, unsupported config, and route-string-only authority.
- [ ] Focused dialect/target/plugin tests cover positive and negative cases.
- [ ] Relevant test binaries/lit tests pass:
      `tianchenrv-rvv-extension-plugin-test`,
      `tianchenrv-target-artifact-export-test`, and focused MLIR/FileCheck
      tests for the changed route.
- [ ] Run `git diff --check`, `git diff --cached --check`, and bounded scans
      over touched files for old i32m1 authority and q-name authority.
- [ ] Trellis task status and journal are truthful; task is finished/archived
      only if acceptance criteria are met.

## Technical Notes

- Specs to consult:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task context to consult:
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-product-reduction-executable-abi-closure/`,
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-i32-to-f32-dequant-route-foundation/`,
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-i32-to-f32-dequant-executable-abi-closure/`.
- Initial code/test files named by the brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the existing focused MLIR
  fixtures under `test/Dialect/RVV/` and `test/Target/RVV/`.
