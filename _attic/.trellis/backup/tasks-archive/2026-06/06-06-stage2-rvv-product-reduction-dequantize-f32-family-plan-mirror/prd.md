# Stage2 RVV product-reduction dequantize f32 family-plan type/config mirror repair

## Goal

Repair the direct RVV contraction route-construction boundary for selected or
pre-realized `widening-product-reduce-dequantize-f32` bodies. Family-plan
type/config facts must be derived from, and mirror, the selected typed
`tcrv_rvv` body before provider-owned route facts reach
`TCRVEmitCLowerableRoute`, target artifact validation, generated bundle ABI, or
runtime evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV product-reduction dequantize f32 family-plan type/config mirror repair`

## Entry-Gate State

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` showed pre-existing uncommitted spec/task
  residue:
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-stage2-resource-aware-llama-parity-steering/`.
- That residue belonged to a completed steering/spec task. It was validated
  with `python3 ./.trellis/scripts/task.py validate ...`, passed
  `git diff --check`, and was committed separately as
  `2f483410 docs: record rvv stage2 resource-aware steering`.
- After that entry-gate commit, the worktree contained only this new task
  directory before production code investigation.

## What I Already Know

- Current branch is `main`.
- Recent history starts with
  `0df6b83a rvv: integrate widening product artifact contract core`, which
  integrated standalone low-precision `widening_product` artifact validation
  with `RVVContractionArtifactContractCore`.
- The previous archived PRD reports two adjacent lit failures in:
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
- The reported failure is:
  `direct contraction route construction requires family-plan type/config facts to mirror the selected typed RVV body`.
- Relevant specs require route support to flow from typed selected `tcrv_rvv`
  body/config/runtime facts into RVV-owned family plans, materialization facts,
  math operand bindings, route-control provider plans, and only then
  `TCRVEmitCLowerableRoute`.
- Common EmitC and target artifact export must remain neutral consumers of
  provider-built route facts. They must not infer product/reduction/dequantize
  semantics from artifact names, route ids, ABI strings, metadata, descriptors,
  or test names.

## Requirements

- Locate why direct contraction route construction rejects
  product-reduction dequantize f32 selected bodies for stale or missing
  family-plan type/config mirror facts.
- Repair the RVV-owned contraction family-plan/provider boundary so the
  product-reduction dequantize f32 plan mirrors the selected typed body for the
  relevant source/product/accumulator/result type and config facts.
- Keep product/reduction/dequantize facts owner-local to the direct contraction
  route family; do not move them into Common EmitC, target artifact metadata, or
  descriptor/source-front-door code.
- Preserve existing ABI order, header/type mapping, generated bundle shape, and
  target artifact validation behavior unless the focused failure proves one of
  those mirrors is stale and must be corrected from provider facts.
- If either focused lit case should remain unsupported, fail closed with an
  exact diagnostic explaining the missing typed body/config/family-plan fact.

## Acceptance Criteria

- [ ] The two previously failing target MLIR cases pass, or fail closed with a
      precise spec-consistent diagnostic:
      `explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
      and
      `pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
- [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [ ] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [ ] A relevant lit filter for widening-product reduce/dequantize/dequant-clamp
      target artifacts passes or reports only an intentionally accepted
      fail-closed boundary.
- [ ] `git diff --check` and `git diff --cached --check` pass.
- [ ] Bounded scan over touched files and added diff lines shows no positive
      legacy i32/source-front-door/descriptor/direct-C/source-export or
      metadata-as-authority drift.
- [ ] No `ssh rvv` runtime/correctness/performance claim is made unless this
      round actually repairs and runs the generated executable artifact path.

## Out Of Scope

- Broad contract-core migration.
- Resource-aware LLaMA parity steering or tuning implementation.
- Dashboard/report/status-only cleanup.
- dtype/LMUL clone batches or new MAcc/mask/broadcast expansion.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door positive routes.
- Common EmitC invention of RVV semantics.
- Artifact metadata, route names, descriptors, or helper strings as route
  authority.
- Runtime or performance claims without actual `ssh rvv` evidence.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-low-precision-widening-product-contract-core/prd.md`.
- Primary source files to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Focused failing lit cases:
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.

## Completion Notes

Production behavior repaired:

- `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` now treats
  product-reduction dequantization and dequant-clamp direct contraction routes
  as a two-stage typed/config mirror:
  family-plan route result element must be f32, while the same-analysis typed
  config element may be either the i32 accumulator stage or the f32 result
  stage.
- The verifier still requires exact same-analysis mirrors for typed config ID,
  SEW, LMUL, tail/mask policy, config contract, VL C type, setvl leaf, family
  plan, route-control plan, materialization facts, and math operand-binding
  facts before `TCRVEmitCLowerableRoute` construction.
- `RVVExtensionPluginTest.cpp` now covers the product-reduction dequantize
  direct provider plan, direct statement plan, selected owner selection,
  positive provider-fact verification, and stale family-plan element mirror
  fail-closed behavior.
- `rvv_generated_bundle_abi_e2e.py` now expects the
  `widening_product_reduce_dequantize_f32` artifact-level `tcrv_rvv.element_type`
  mirror to be f32, matching the provider route result element. The source,
  product, accumulator, dequant scale, ABI order, and typed-body structure
  remain route-family-owned facts.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now records the direct
  contraction product-reduction dequantize/dequant-clamp type/config mirror
  contract so future owners do not regress it.

Checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt`
- `cmake --build build --target tcrv-translate`
- `lit -sv . --filter 'artifact-widening-product-reduce-dequantize-f32'`
  from `build/test`: 2 selected tests passed.
- `cmake --build build --target tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `lit -sv . --filter 'artifact-widening-product-reduce-(add|dequantize|dequant-clamp)|realization-widening-product-reduce-dequant-clamp'`
  from `build/test`: 5 selected tests passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- explicit selected-body generated-bundle dry-run for
  `widening_product_reduce_dequantize_f32`: passed.
- pre-realized selected-body generated-bundle dry-run for
  `widening_product_reduce_dequantize_f32`: passed.
- `git diff --check`
- Bounded added-diff old-authority scan: no added positive legacy
  i32/source-front-door/descriptor/direct-C/source-export/helper-name or
  metadata-as-authority matches. Full-file scan matches are existing
  fail-closed script/test inventory.

No `ssh rvv` evidence was run or claimed because this round repaired route
construction, target artifact/header export, and generated-bundle dry-run
evidence only; it did not claim runtime correctness or performance.

## Current Phase

Finish.
