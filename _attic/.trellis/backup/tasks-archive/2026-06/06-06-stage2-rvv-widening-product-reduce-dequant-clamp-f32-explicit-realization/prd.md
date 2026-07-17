# Stage2 RVV widening product reduce dequant-clamp f32 explicit selected-body realization

## Goal

Prove the non-pre-realized selected-body boundary for
`widening_product_reduce_dequant_clamp_f32`: a selected `tcrv.exec` RVV variant
contains an explicit typed compound `tcrv_rvv` body, the RVV plugin-local
contraction realization owner expands it into the legal vector-level
`setvl/load/widening_product/standalone_reduce/dequantize/clamp/store`
structure, and the existing provider-derived `TCRVEmitCLowerableRoute` plus
target artifact path consumes those realized facts.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening product reduce dequant-clamp f32 explicit selected-body realization`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `014e2fa0 chore(task): archive 06-06-stage2-rvv-widening-product-reduce-dequant-clamp-f32-artifact-foundation`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- The archived artifact-foundation task closed the pre-realized
  `widening_product_reduce_dequant_clamp_f32` provider/target/generated-bundle
  and real `ssh rvv` ABI evidence path. This task must not recast that evidence
  as the new achievement.
- Existing code already realizes the pre-realized fused body through the RVV
  contraction selected-body owner and the existing provider/target path.
- The current gap is the input surface: an explicit non-pre-realized compound
  selected body must be accepted, validated, realized, and routed without common
  EmitC or target export inventing fused semantics.

## Requirements

- Add a bounded explicit typed compound RVV body surface for
  `widening_product_reduce_dequant_clamp_f32` that carries the same operation
  semantics, widening product operand roles, accumulator/reduction layout,
  dequant scale facts, clamp bounds, dtype/config, runtime ABI roles, VL policy,
  and result layout facts needed by the existing pre-realized path.
- The RVV contraction selected-body realization owner must consume the explicit
  body and materialize the same legal vector-level realized `tcrv_rvv` structure
  before route/provider facts are collected.
- Unsupported or underspecified operation kind, memory form, dtype/config,
  accumulator/reduction layout, dequant scale role, clamp bound roles/order,
  policy, runtime ABI roles, or stale route-authority metadata must fail closed
  before common EmitC or target artifact export.
- Provider/target route evidence must prove the realized explicit body reaches
  the existing provider-derived fused route facts and target header/artifact
  mirrors.
- Preserve the existing pre-realized artifact and generated-bundle evidence
  paths when shared code is touched.

## Acceptance Criteria

- [x] A positive explicit selected-body fixture shows the new non-pre-realized
      compound body is consumed by `--tcrv-materialize-selected-lowering-boundaries`
      and rewritten into `setvl`, `with_vl`, `load`, `widening_product`,
      `standalone_reduce`, `dequantize`, lower/upper splats, compares, selects,
      and store.
- [x] The same fixture reaches `--tcrv-materialize-emission-plans` and target
      header/artifact export with provider-derived fused route facts matching
      the existing `widening_product_reduce_dequant_clamp_f32` route contract.
- [x] Negative checks fail closed for missing or stale operation, dtype/config,
      accumulator/reduction, dequant scale, clamp bound, runtime ABI, policy,
      and stale route-authority facts.
- [x] Existing pre-realized fused artifact and generated-bundle dry-run
      regressions still pass if shared realization/provider/target/script code
      is touched.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      a bounded old-authority scan over touched files and added diff lines pass.

## Definition Of Done

- Implemented in C++/MLIR/TableGen/CMake/lit/FileCheck only for compiler
  behavior; Python remains unchanged unless generated-bundle evidence requires
  a bounded harness update.
- Task context stays truthful and the task is finished/archived when complete.
- One coherent commit is created if the task completes.

## Technical Approach

Add a minimal explicit compound op alongside the existing pre-realized fused op,
sharing the same bounded attributes and operand contract. Reuse the RVV
contraction selected-body realization owner to classify, validate, and realize
the explicit op into the already-supported vector-level body. Keep provider,
Common EmitC, and target export semantics unchanged except for accepting the
realized output from the new body surface.

## Completion Evidence

- Added `tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_body` as the
  explicit non-pre-realized compound selected-body surface for the bounded fused
  route.
- Refactored the RVV dialect verifier so the explicit body and the existing
  pre-realized body share the same field, ABI, dtype/config, policy, and stale
  authority checks.
- Extended the RVV contraction selected-body realization owner to classify the
  explicit body, validate it through a dedicated explicit validation entrypoint,
  and reuse the existing realization plan to materialize the same vector-level
  `setvl/load/widening_product/standalone_reduce/dequantize/clamp/store`
  structure.
- Added
  `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`
  with positive realization, emission-plan, target-header checks, and negatives
  for stale operation, reduction, dtype/config, scale, lower bound, policy,
  route authority, provider mirror, ABI order, and operand binding facts.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the explicit
  fused selected-body realization contract.
- Checks passed:
  `cmake --build build --target tcrv-opt tcrv-translate`;
  focused lit filter for explicit fixture, pre-realized fused artifact fixture,
  and pre-realized fused generated-bundle dry-run;
  `build/bin/tianchenrv-rvv-dialect-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `git diff --check`;
  Trellis context validation.
- No new `ssh rvv` correctness claim is made in this task. The existing
  pre-realized `ssh rvv` evidence from the archived task remains prior
  evidence, not this task's new result.

## Out Of Scope

- No new route-family expansion beyond `widening_product_reduce_dequant_clamp_f32`.
- No broad reduction, matmul, dtype, LMUL, policy, or performance matrix.
- No high-level Linalg/Vector/StableHLO/frontend/source-front-door positive path.
- No common EmitC invention of fused semantics.
- No compatibility wrapper preserving legacy i32 authority.
- No new `ssh rvv` correctness claim unless the existing harness can run
  unchanged and the task owner needs that evidence.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-widening-product-reduce-dequant-clamp-f32-artifact-foundation/prd.md`.
- Primary implementation files inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`,
  and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
