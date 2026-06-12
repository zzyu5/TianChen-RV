# Stage2 RVV low-precision direct-contraction resource candidate seed

## Goal

Introduce the first bounded RVV plugin-local resource candidate and selection
contract for one low-precision direct-contraction representative. The
representative is the existing typed
`widening_product_reduce_dequantize_f32` / dequant-clamp family, because it
already carries signed i8 source facts, i16 product facts, i32 accumulator/reduce
facts, f32 result/dequant facts, runtime AVL, ABI operands, and selected-body
realization coverage without making q8/q4 names authoritative.

The new contract must derive candidate facts from selected typed
`tcrv_rvv` body/config/runtime/capability facts and make the selected candidate
consumed by RVV-owned provider planning and target artifact validation before
route acceptance. It must not claim performance parity or introduce a global
autotuning cache.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV low-precision direct-contraction resource candidate seed`

## Entry-Gate State

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Recent commits were:
  `2cabac98 chore: record journal`,
  `b9bb2569 rvv: repair product dequant family-plan mirror`,
  `2f483410 docs: record rvv stage2 resource-aware steering`,
  `0df6b83a rvv: integrate widening product artifact contract core`,
  `b836fce0 chore: record journal`.
- No `.trellis/.current-task` file existed, so this task was created from the
  Hermes brief before source edits.

## What I Already Know

- Specs require low-precision direct-contraction resource work to keep q8/q4 and
  llama.cpp as pressure tests, not route-name or benchmark-name authority.
- Current `RVVGearboxSchedule` only has a fixed dequant i32->f32 candidate
  mirror; it does not structurally model low-precision direct-contraction
  source/product/accumulator/result resource candidates.
- Current `widening_product_reduce_dequantize_f32` selected-body fixtures carry
  the typed i8 -> i16 product -> i32 reduction -> f32 dequant chain needed for a
  q8_0_q8_0-equivalent pressure representative.
- Current provider route planning already derives direct-contraction facts from
  the selected body, route-family plan, route-control plan, math operand-binding
  facts, and materialization leaves before constructing the route.
- Current target artifact validation rebuilds provider contracts and validates
  candidate mirrors for product/dequant facts; this is the right consumer seam
  for stale resource candidate rejection.
- Existing selected target capability facts expose provider, supported SEW/LMUL,
  and required tail/mask policy. They do not yet expose VLEN or vector register
  budget, so this round should seed an explicit bounded static resource model
  while making that limitation visible.

## Requirements

- Add a structural RVV-local resource candidate/selection contract for the
  bounded low-precision direct-contraction representative.
- Candidate facts must include at minimum:
  source/product/accumulator/result dtype and SEW/LMUL facts, derived EMUL facts,
  memory form, tail/mask policy, unroll factor, accumulator count, reduction
  layout, vsetvl region count, peak live vector-group estimate, runtime
  AVL/VL/ABI mapping, target capability mirrors, legality decision, and
  rejection reason.
- The first positive candidate should derive from the existing
  `widening_product_reduce_dequantize_f32` selected typed body/config/runtime
  path. Dequant-clamp may share the same contract if it is cheap and bounded,
  but the first positive representative is enough.
- The selected candidate must be stored in the direct-contraction provider plan
  or an equivalent owner-local plan and consumed before route construction or by
  target artifact validation. Artifact metadata may mirror the candidate only
  after provider facts exist.
- Fail closed for at least two stale or illegal cases, preferably missing
  low-precision dtype facts, unsupported EMUL/config, peak live vector pressure,
  stale metadata-only selected candidate, or invalid ABI/runtime AVL mapping.
- Keep Common EmitC and target export as neutral consumers of RVV provider-built
  facts. Do not infer RVV semantics from artifact names, q8/q4 names, route ids,
  ABI strings, or emitted intrinsic spelling.

## Acceptance Criteria

- [x] Production code has an RVV plugin-local resource candidate/selection
      structure for the low-precision direct-contraction representative.
- [x] `widening_product_reduce_dequantize_f32` provider planning derives and
      consumes the selected resource candidate before route acceptance.
- [x] Target artifact validation rejects stale metadata-only resource candidate
      mirrors and validates provider-derived selected candidate mirrors.
- [x] C++ coverage validates the legal candidate construction and at least two
      fail-closed cases such as stale candidate mirror, missing low-precision
      dtype/config facts, EMUL/resource pressure overflow, or bad ABI/runtime
      AVL facts.
- [x] Focused lit/FileCheck coverage shows candidate facts are present on the
      selected representative and stale candidate metadata fails closed.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Relevant lit filters for product-reduction dequantize/dequant-clamp and
      resource candidate mirrors pass.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no positive legacy i32m1/source-front-door/descriptor/direct-C or
      metadata-as-authority drift.

## Out Of Scope

- Full autotuning cache, global tuning database, or broad benchmark matrix.
- q8/q4/llama.cpp names as route authority.
- Handwritten intrinsic sequence copied from llama.cpp.
- Performance or llama.cpp parity claims without same-target correctness and
  timing evidence.
- A new dtype-prefixed op family or dtype/LMUL clone batch.
- High-level Linalg/Vector/StableHLO frontend work.
- Source-front-door positive routes.
- Common EmitC invention of RVV semantics.
- Treating prompt edits, reports, helper-only code, or metadata-only mirrors as
  the main achievement.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived tasks read:
  `.trellis/tasks/archive/2026-06/06-06-06-06-rvv-stage2-resource-aware-llama-parity-steering/prd.md`
  and
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-product-reduction-dequantize-f32-family-plan-mirror/prd.md`.
- Primary source files inspected:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
- Runtime/correctness/performance is not claimed unless a later check actually
  runs non-dry-run `ssh rvv` evidence. This PRD only requires route/provider/
  target validation and focused generated-bundle or lit seed evidence.

## Current Phase

finish.

## Implementation Completed

- Added `RVVLowPrecisionContractionResourceSelection` as the RVV-owned
  resource candidate/selection structure and threaded it through the
  contraction family plan, direct-contraction provider plan, selected route
  description, and widening dot/reduce target validation contract.
- Derived the first bounded candidate from the typed product-reduction
  dequantize/dequant-clamp family using selected body/config/runtime/capability
  facts: i8 source, i16 product, i32 accumulator, f32 result, SEW/LMUL/EMUL,
  memory form, tail/mask policy, unroll, accumulator count, reduction layout,
  vsetvl region count, peak live vector-group estimate, register budget,
  runtime AVL source, ABI order, target capability mirrors, legality, and
  rejection reason.
- Made the selected resource candidate consumed before route acceptance by
  direct provider planning and target artifact candidate mirror validation.
- Emitted the selected resource facts as mirror metadata only after provider
  selection exists, surfaced them in target header evidence, and kept target
  export/Common EmitC as neutral consumers.
- Repaired residue scanners in target export and the generated-bundle evidence
  script so `direct-contraction` is not mistaken for forbidden `direct-C`
  source-export authority while true `direct-C` residue remains rejected.

## Verification Completed

- `ninja -C build tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  explicit-selected-body-artifact-widening-product-reduce-dequantize-f32`
  from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  widening-product-reduce-dequant-clamp-f32` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  widening-product-reduce-dequantize-f32` from `build/test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `git diff --check`
- Bounded touched-file scan for legacy `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `source-front-door`, and descriptor
  residues. Matches were existing fail-closed tests/guards or residue scanners,
  not positive route authority.

No runtime correctness or performance claim was made, so no `ssh rvv` timing or
correctness run was required in this round.
