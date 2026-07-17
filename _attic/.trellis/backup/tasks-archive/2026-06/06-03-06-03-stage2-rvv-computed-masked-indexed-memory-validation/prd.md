# Stage2 RVV computed masked indexed memory production validation boundary

## Goal

Close the production provider-to-target validation boundary for the two
computed masked indexed RVV memory routes:
`computed_masked_indexed_gather_load_unit_store` and
`computed_masked_indexed_scatter_store_unit_load`.

The target artifact validator must consume provider-owned `tcrv_rvv`
body/config/runtime facts for compare-produced masks, indexed memory, inactive
lane behavior, typed compute op, route-family plan, headers, type mapping,
operand binding, and explicit provider support mirrors. It must not reconstruct
these facts from route ids, artifact names, test fixture strings, candidate
metadata mirrors, descriptors, or common EmitC/export code.

## Why Now

The previous archived base-memory validation closeout made unmasked indexed
gather/scatter validation consume provider-owned base-memory facts. That same
closeout explicitly left masked base-memory typed compute semantics outside
scope. This task moves the computed masked indexed routes onto the same
provider-owned validation discipline so Stage 2 does not leave mask plus index
semantics as a target-local metadata mirror path.

## Requirements

- Keep the scope to
  `ComputedMaskIndexedGatherLoadUnitStore` and
  `ComputedMaskIndexedScatterStoreUnitLoad`.
- Provide or reuse a production provider-owned computed-mask indexed fact
  surface that records, for each route:
  - operation and memory form;
  - ABI order `cmp_lhs,cmp_rhs,src,index,dst,n`;
  - runtime ABI parameters in that order;
  - SEW/LMUL/tail/mask policy `32/m1/agnostic/agnostic`;
  - typed compute op:
    `tcrv_rvv.masked_indexed_load` for gather and
    `tcrv_rvv.masked_indexed_store` for scatter;
  - memory form:
    `computed-mask-indexed-gather-load-unit-store` for gather and
    `computed-mask-unit-load-indexed-scatter-store` for scatter;
  - compare-produced mask facts: predicate kind, mask role, mask source,
    mask memory form, mask producer source, mask type and C type mapping;
  - inactive lane contract and passthrough/no-write layout;
  - masked indexed memory layout;
  - index facts: `index_source = runtime_abi:index`, `index_eew = 32`,
    `offset_unit = element`, and scatter `index_uniqueness = unique`;
  - source/destination memory forms and indexed data/destination forms;
  - route operand binding plan and full summary with ABI and header
    participation for every exported parameter;
  - computed-mask memory route-family plan and mask/tail policy plan;
  - required headers and C type mapping summary;
  - target leaf profile and explicit `provider_supported_mirror` label.
- Make target artifact validation compare the route description and candidate
  metadata mirrors against that provider-owned fact surface.
- Reject stale gather facts on scatter, stale scatter facts on gather, missing
  mask facts, stale index facts, stale typed-compute facts, stale
  route-family plan, stale header/type mapping, stale target profile, stale
  provider mirror, and accidental unmasked indexed/unit/strided fallback.
- Keep common EmitC/export neutral: it may carry provider-built payloads and
  mirrors, but must not choose RVV semantics or infer indexed/masked facts.
- Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

- [x] Production RVV provider/planning and/or target validation has a focused
      diff that makes computed masked indexed gather/scatter validation consume
      provider-owned facts instead of target-local switch constants.
- [x] Target validation checks provider-derived ABI order, runtime ABI
      parameters, typed compute op, memory form, mask facts, indexed facts,
      inactive lane and passthrough/no-write contracts, source/destination
      memory forms, route-family plan, binding plan/summary, header/type
      mapping, target profile, and provider mirror for both routes.
- [x] C++ target validation tests prove fail-closed behavior for stale or
      missing computed masked indexed facts, including gather/scatter
      cross-contamination and stale candidate metadata mirrors.
- [x] Existing generated-bundle dry-run tests for explicit and pre-realized
      computed masked indexed gather/scatter still pass.
- [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, or legacy `i32m1` route authority is introduced.
- [x] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
- [x] Trellis finish/archive and one coherent commit complete this round.

## Technical Approach

1. Add a provider-owned computed-mask indexed memory route fact surface or
   extend the existing provider fact accessors so the target validator can
   query canonical facts for the two route operations.
2. Replace target-local computed masked indexed expectation helpers with
   comparisons against provider facts. Keep target code responsible for
   verifying the rebuilt route and candidate mirrors, not for inventing route
   semantics.
3. Tighten C++ target validation tests around provider description mutations
   and candidate metadata mutations. Prefer small mutations that prove the
   production validator rejects stale gather/scatter, stale mask/index,
   stale typed compute, stale binding/header/type, stale target profile, and
   stale provider mirror facts.
4. Run the smallest lit/C++ checks that exercise the changed surface plus
   generated-bundle dry-run scripts for explicit and pre-realized gather and
   scatter fixtures.

## Out Of Scope

- No broad masked route matrix.
- No segment path, reductions, macc/dot, standalone compare/select expansion,
  dtype/LMUL/index-width clone batches, or high-level frontend lowering.
- No source-front-door positive routes.
- No standalone evidence-packaging task as the main deliverable.
- No rewrite of unmasked indexed gather/scatter validation except where shared
  helpers are touched incidentally.
- No RVV semantics moved into common EmitC/export.

## Evidence Plan

- Run the focused C++ target artifact test that covers
  `RVVTargetArtifactRouteFamilyValidation`.
- Run lit filters/scripts for computed masked indexed gather/scatter target
  artifact export and generated-bundle dry-runs for explicit and pre-realized
  forms.
- Run direct fail-closed checks for stale/missing masked indexed facts if an
  existing script covers that path; otherwise keep C++ mutation tests as the
  production validation evidence.
- Run a bounded old-authority scan over touched files for legacy route-authority
  markers and `git diff --check`.
- Do not rerun `ssh rvv` unless route emission, runtime ABI behavior, or
  generated runtime semantics change. If this round only tightens production
  validation, reuse archived RVV correctness evidence from the immediately
  preceding computed masked indexed gather/scatter artifact ABI tasks and state
  that no new runtime/correctness claim changed.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/validation/index.md`,
  `.trellis/spec/variant-pipeline/index.md`.
- Prior closeout read:
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-base-memory-route-family-production-validation-closeout/prd.md`.
- Relevant source files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.
- Initial finding: provider-side computed masked indexed realization already
  validates mask/index dataflow strongly, but target validation still has
  computed masked indexed expectations encoded locally in the target file.

## Completion Evidence

- Added provider-owned `RVVComputedMaskIndexedMemoryRouteFacts` for
  computed-mask indexed gather/scatter and made target artifact validation
  consume that fact surface for provider descriptions and candidate mirrors.
- Added C++ target artifact fail-closed coverage for stale typed compute facts,
  stale binding summaries/plans, stale provider mirrors, target profiles,
  header/type facts, index facts, and candidate metadata mirrors.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  computed-mask indexed memory fact-surface contract.
- Passed focused build:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j 16`.
- Passed focused C++ target artifact validation:
  `rtk build/bin/tianchenrv-target-artifact-export-test`.
- Passed generated-bundle dry-run lit checks from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-indexed-gather-load-dry-run'`
  selected 2 tests and passed 2 tests.
- Passed generated-bundle dry-run lit checks from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-indexed-scatter-store-dry-run'`
  selected 2 tests and passed 2 tests.
- Passed target artifact MLIR fixture lit checks from `build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-computed-masked-indexed-(gather-load|scatter-store)'`
  selected 4 tests and passed 4 tests.
- Passed `rtk git diff --check`.
- Code-only bounded old-authority scan over touched implementation/test files
  had no positive-route-authority hits for legacy `i32m1`, source-front-door,
  source-export, direct-C, descriptor, or related stale authority markers.
- No new `ssh rvv` runtime claim was made in this validation-tightening round;
  generated runtime semantics and runtime ABI behavior were not changed.
- Trellis task was archived under
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-indexed-memory-validation/`,
  workspace journal Session 401 was recorded, and the coherent task commit was
  prepared in this round.
