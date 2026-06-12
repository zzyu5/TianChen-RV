# Stage2 RVV contraction-dequant-clamp selected-body composition foundation

## Goal

Implement one bounded production route-supported Stage 2 RVV selected-body
composition that joins the already-proven low-precision contraction-dequant
chain and the dequant-clamp f32 epilogue chain:

```text
typed i8 source loads
  -> widening product / reduction into i32 accumulator boundary
  -> runtime-scale f32 dequantize
  -> runtime lower/upper f32 clamp/select
  -> f32 output store
```

This is a compiler route-support task. The deliverable is a typed
`tcrv_rvv` selected or pre-realized selected body, RVV plugin-local
realization/provider planning, and target artifact validation that derive route
facts from body/config/runtime structure. It is not a new executable-evidence
task, not q8/q4/llama work, and not a high-level frontend task.

## What I Already Know

- The repository starts clean on `main` at `ff02464e rvv: close dequant clamp
  epilogue ABI evidence`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- `.trellis/spec/index.md` requires the RVV authority chain:
  `tcrv.exec` envelope -> selected typed `tcrv_rvv` body -> RVV plugin
  realization/route provider -> common EmitC -> target artifact.
- `rvv-plugin.md` requires selected pre-realized RVV bodies to be consumed by
  plugin-local selected-body realization before provider route facts are
  collected.
- `emitc-route.md` defines provider-owned route fact surfaces, runtime
  AVL/VL contracts, operand-binding summaries, and target artifact mirrors as
  non-authoritative mirrors.
- Archived contraction-dequant tasks added and proved
  `widening_product_reduce_dequantize_f32`: i8 inputs, i16 product, i32
  reduction/accumulator boundary, runtime scale, f32 dequant result, and f32
  store.
- Archived dequant-clamp tasks added and proved
  `dequant_clamp_f32_epilogue`: i32 source, runtime f32 scale, runtime f32
  lower/upper bounds, f32 clamp/select chain, and f32 store.
- The new route should compose those surfaces in one selected body rather than
  infer support from route ids, artifact names, q-names, ABI strings, status
  fields, metadata mirrors, or existing generated-bundle evidence.

## Requirements

- Add production code, not only fixtures, helper reports, or harness changes.
- The selected/pre-realized typed RVV body must structurally carry:
  - source/accumulator/reduction dtype facts;
  - widening product and reduction relation;
  - i32 accumulator/reduction value boundary;
  - runtime f32 scale role;
  - f32 dequant result type;
  - runtime f32 lower/upper bound roles and unambiguous lower-before-upper
    clamp order;
  - SEW/LMUL/policy and runtime AVL/VL facts;
  - memory roles for source inputs, accumulator input, and f32 output store.
- RVV plugin-local realization and route planning must compose the chain
  without changing computation semantics, dtype semantics, runtime parameter
  roles, selected variant origin, dispatch/fallback behavior, or runtime
  `n`/AVL values.
- Provider route planning must fail closed for missing runtime scale, missing
  lower/upper bounds, swapped or ambiguous bound roles, missing reduction
  facts, dtype-chain mismatch, unsupported config/policy, stale mirrors, and
  route-string/artifact-name/q-name authority.
- Target artifact validation must mirror only provider-derived facts and reject
  stale/mismatched route metadata before accepting generated artifacts.
- Common EmitC/export must remain neutral: it may carry provider-built payloads
  but must not choose RVV dtype, intrinsic, route semantics, bound semantics, or
  contraction/dequant/clamp shape.

## Acceptance Criteria

- [ ] A positive selected/pre-realized RVV fixture reaches route support for the
      full contraction -> dequant -> clamp/select -> f32 store chain.
- [ ] Production RVV dialect/verifier, selected-body realization, provider
      route planning, and/or target validation code is updated as needed for
      the composed route.
- [ ] The route facts and target mirrors include provider-derived contraction,
      dequant, clamp/bounds, runtime ABI, header/type, operand binding,
      runtime AVL/VL, and selected-body provenance facts.
- [ ] Negative tests cover missing runtime scale, missing bounds, swapped or
      ambiguous bound roles, missing or mismatched reduction facts,
      dtype-chain mismatch, unsupported config/policy, stale metadata mirrors,
      and route-string/artifact-name/q-name authority.
- [ ] Focused plugin/target checks pass after implementation, including
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test` if provider or target code
      changes.
- [ ] A focused old-authority scan over touched files shows no new
      `RVVI32M1`, `rvv-i32m1`, dtype-prefixed helper op, source-front-door,
      descriptor, q-name, or artifact-name authority.
- [ ] `git diff --check` and `git diff --cached --check` pass.
- [ ] No `ssh rvv` correctness claim is made unless the generated artifact is
      compiled and run on real `ssh rvv` in this round.

## Out Of Scope

- q8/q4/llama benchmark route or model-name route authority.
- High-level Linalg/Vector/StableHLO frontend lowering.
- Standalone contraction-only, dequant-only, or clamp-only evidence as the
  primary deliverable.
- Zero-point expansion unless it is strictly fail-closed.
- dtype/LMUL clone batch, one-intrinsic wrapper dialect, compatibility wrapper
  preserving old i32m1 authority, dashboard/report-only work, or broad smoke
  matrices.
- New executable correctness evidence unless route-supported compiler support
  is complete and a bounded generated artifact path is ready.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/plugin-protocol/index.md`.
- Predecessor task directories:
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-low-precision-contraction-dequant-chain/`,
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-contraction-dequant-realization/`,
  `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-pre-realized-contraction-dequant-executable-closure/`,
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-dequant-clamp-epilogue-composition-foundation/`,
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-dequant-clamp-epilogue-executable-abi/`.
- Likely impacted files from the brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`.
