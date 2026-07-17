# Stage2 RVV low-precision widening-product contraction contract-core integration

## Goal

Integrate the standalone low-precision `widening_product` target artifact
validation contract with the shared provider-owned
`RVVContractionArtifactContractCore` for facts that are genuinely common across
RVV contraction artifact validators. Keep low-precision owner facts such as
source/result SEW/LMUL, widening-product relation/intrinsic, source/destination
memory form, and statement-shape details in the widening-product owner
contract, and keep product-reduction/dequant/clamp facts in the existing
widening dot/direct-contraction owner contract.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV low-precision widening-product contraction contract-core integration`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` showed pre-existing uncommitted work:
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/tasks/06-06-06-06-rvv-stage2-resource-aware-llama-parity-steering/`.
  These are outside this task and must be preserved.
- Initial `git log --oneline -8` started at
  `b836fce0 chore: record journal`, followed by
  `52bb0204 rvv: consolidate contraction artifact contract core`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-direct-contraction-route-family-contract/`.
- The archived direct-contraction task introduced
  `RVVContractionArtifactContractCore` and wired it through MAcc and widening
  dot/direct-contraction validators.
- Current code evidence shows `RVVWideningDotReduceRouteValidationContract`
  already owns `RVVContractionArtifactContractCore`, including
  `ProductReductionChain`, `ProductReductionDequantization`, and
  `ProductReductionDequantClampF32` validation kinds.
- Current code evidence shows standalone
  `RVVWideningProductRouteValidationContract` still carries common artifact
  facts as flat fields instead of retaining the shared core.
- The target validator for standalone widening-product currently compares
  provider facts and candidate mirrors through those flat fields.

## Requirements

- Add the shared provider-owned contraction artifact core to
  `RVVWideningProductRouteValidationContract`.
- Populate the standalone widening-product validation contract from
  `getRVVContractionArtifactContractCore(...)` for common facts only:
  route token, memory form, config/runtime control, runtime ABI order, target
  leaf profile, provider mirror, header declarations, C type mapping summary,
  route operand binding plan/summary, typed compute op, VL C type,
  source/result vector type/C type, and runtime ABI parameters.
- Keep standalone widening-product owner facts outside the core:
  source/result SEW and LMUL, tail/mask policy, contraction family plan,
  source/destination memory form, widening-product relation/intrinsic,
  load/store/setvl intrinsics, statement names, loop counts, and type-mapping
  entries.
- Keep product-reduction/dequant/clamp facts in
  `RVVWideningDotReduceRouteValidationContract`; do not migrate
  product/reduction/dequant/clamp semantics into the shared core.
- Update target artifact validation for standalone widening-product to consume
  `contract.core` for the shared facts and keep owner-local checks for
  low-precision facts.
- Preserve existing emitted statements, ABI order, generated headers, generated
  bundle harnesses, and runtime behavior.
- Do not introduce Common EmitC RVV semantic inference, descriptor/source-front
  door authority, route-id acceptance authority, performance tuning, or new
  runtime correctness/performance claims.

## Acceptance Criteria

- [ ] Production source changes make standalone `widening_product` artifact
      validation retain and consume `RVVContractionArtifactContractCore`.
- [ ] MAcc and widening dot/direct-contraction behavior remains unchanged.
- [ ] Product-reduction/dequant/clamp facts stay owner-local in the existing
      widening dot/direct-contraction contract.
- [ ] Target validation still fails closed for stale standalone
      widening-product provider facts, including stale provider mirror,
      source dtype/config, runtime ABI parameter, route operand binding,
      widening-product relation/intrinsic, header/type mapping, route
      statement facts, and stale non-family mirrors.
- [ ] Focused C++ checks pass:
      `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test`.
- [ ] Focused lit dry-runs covering low-precision widening-product generated
      artifact validation pass, including the explicit/pre-realized
      `widening_product_reduce_dequant_clamp_f32` dry-runs from the brief and
      relevant widening-product/product-reduction fixtures.
- [ ] No `ssh rvv` evidence is required unless this round changes emitted
      statements, ABI order, generated headers/harnesses, runtime correctness,
      or performance behavior, or claims new executable behavior.
- [ ] Bounded old-authority scan over touched source/spec/task files and added
      diff lines shows no positive legacy i32/source-front-door/descriptor/
      direct-C/source-export/helper-name/metadata authority drift.
- [ ] `git diff --check`, `git diff --cached --check`, Trellis task validation,
      and final git status are reported truthfully.

## Technical Approach

Use the already-integrated MAcc and widening dot/direct-contraction pattern as
the reference:

```text
RVVWideningProductRouteFacts
  -> RVVContractionArtifactContractCore
  -> RVVWideningProductRouteValidationContract
  -> target artifact provider-facts and candidate-mirror validation
```

The standalone widening-product route is a low-precision product route
(`i8mf4 x i8mf4 -> i16mf2`) and not a product-reduction/dequant/clamp route.
The shared core may carry only the provider-owned artifact facts that are
already common to MAcc and widening dot/direct-contraction validation. The
standalone owner contract remains responsible for widening-product-specific
typed relation, intrinsic, memory form, and statement validation.

## Out Of Scope

- New MAcc, widening dot-reduce, product-reduction, dequant, dequant-clamp,
  dtype, LMUL, or source-front-door coverage.
- Broad contraction matrix expansion.
- Runtime/performance claims or llama/q8/q4 parity claims.
- Common EmitC invention of RVV semantics.
- Descriptor-driven computation, direct-C/source-export positive routes, or
  route-id/artifact-name/test-name authority.
- Global autotuning, dashboards, readiness state machines, or report-only work.

## Technical Notes

- Primary source files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Reference source files:
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`.
- Focused test files:
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-product-reduce-dequant-clamp-f32-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run.test`,
  and relevant `test/Target/RVV/*widening-product*` fixtures.

## Completion Notes

Production source change:

- `RVVWideningProductRouteValidationContract` now retains
  `RVVContractionArtifactContractCore`.
- `populateRVVWideningProductValidationContract(...)` now builds the shared
  core from `RVVWideningProductRouteFacts` for common provider-owned artifact
  facts, then fills standalone owner-local widening-product fields separately.
- `RVVTargetArtifactRouteFamilyValidation.cpp` now consumes
  `contract.core` for standalone widening-product shared artifact facts:
  route token, memory form, config/runtime mirrors, runtime ABI order,
  runtime ABI parameters, required headers, C type mapping summary,
  route operand binding plan/summary, target leaf profile, provider support
  mirror, typed compute op, VL C type, and source/result vector type/C type.
- Standalone widening-product owner-local facts remain outside the core:
  source/result SEW and LMUL, tail/mask policy, contraction route-family plan,
  source/destination memory form, widening-product relation/intrinsic,
  load/store/setvl intrinsics, statement names, and statement counts.
- Product-reduction/dequant/clamp facts remain in
  `RVVWideningDotReduceRouteValidationContract`; this round did not move
  product/reduction/dequant/clamp semantics into the shared core.
- `TargetArtifactExportTest.cpp` now includes standalone widening-product
  fail-closed coverage for stale runtime ABI order, route operand binding
  summary, header declarations, C type mapping summary, and stale candidate
  C type mapping mirrors.

Behavioral boundary:

- No emitted statements, ABI order, generated headers, generated bundle
  harnesses, runtime correctness, or performance behavior were intentionally
  changed.
- No `ssh rvv` evidence was required because this was a provider/target
  validation contract integration only.
- A focused target-MLIR filter over widening-product reduce/add/dequantize/
  dequant-clamp fixtures exposed two adjacent failures in
  `widening_product_reduce_dequantize_f32` route construction:
  explicit and pre-realized dequantize fixtures fail with
  `direct contraction route construction requires family-plan type/config facts
  to mirror the selected typed RVV body`. This failure is outside the standalone
  widening-product artifact-core seam changed here and should be a separate
  continuation point if that product-reduction dequantization route is selected.

Checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(explicit-|pre-realized-)?widening-product-reduce-dequant-clamp-f32-dry-run|rvv-generated-bundle-abi-e2e-(explicit-|pre-realized-)?widening-product.*dry-run'`
  passed 2 selected tests.
- From `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'artifact-widening-product-reduce-(add|dequantize|dequant-clamp)|realization-widening-product-reduce-dequant-clamp'`
  passed 3 selected tests and failed the 2
  `widening-product-reduce-dequantize-f32` tests noted above.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-low-precision-widening-product-contract-core`
- Bounded added-diff old-authority scan over touched task/source/test files:
  no positive legacy i32/source-front-door/descriptor/direct-C/source-export
  authority matches. The only matches were an error message checking rebuilt
  provider route id and a negative `metadata-derived-buffer` test fixture.

## Current Phase

Finish.
