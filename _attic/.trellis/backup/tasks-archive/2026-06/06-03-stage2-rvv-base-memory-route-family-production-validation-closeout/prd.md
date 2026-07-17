# Stage2 RVV base-memory route-family production validation closeout

## Goal

Close out the production validation surface for the indexed base-memory RVV
route family by making `indexed_gather_unit_store` and
`indexed_scatter_unit_load` share one provider-owned fact authority across RVV
provider planning, `TCRVEmitCLowerableRoute` construction, target artifact
validation, and generated-bundle mirrors.

## Why Now

The previous gather and scatter tasks proved executable artifact behavior and
real `ssh rvv` correctness for each indexed route. The scatter round mostly
touched tests, fixtures, specs, and journals while relying on production facts
added earlier. This round must make the active production provider/target
boundary itself the owner, so Stage 2 does not continue as per-route artifact
closeouts.

## Requirements

- Consume `RVVBaseMemoryMovementRouteFacts` or an equivalent RVV
  provider-owned fact surface for both indexed gather/unit-store and indexed
  scatter/unit-load.
- Validate ABI order from provider facts:
  `data,index,out,n` for gather and `src,index,dst,n` for scatter.
- Validate source/destination role polarity, memory form, indexed layout,
  source/destination memory-form fields, and typed compute op from provider
  facts, not route ids, artifact names, C strings, test names, descriptors, or
  mirror metadata.
- Validate indexed facts from the provider surface: `index_eew = 32`,
  `offset_unit = element`, `index_source = runtime_abi:index`, and scatter
  `index_uniqueness = unique`.
- Validate route-family plan, route operand binding plan and summary with
  header participation, required headers, C type mapping, target leaf profile,
  and explicit `provider_supported_mirror` labeling from the provider surface.
- Reject gather/scatter cross-contamination in production validation:
  stale gather facts on scatter, stale scatter facts on gather, source/dest
  polarity swaps, stale route-family plans, stale header/type/binding mirrors,
  provider mirror mismatch, and accidental unit/strided fallback residue.
- Keep common EmitC/export neutral: it may carry provider payloads and mirrors,
  but must not choose RVV base-memory semantics.
- Preserve existing generated-bundle support for explicit and pre-realized
  selected-body indexed gather/scatter routes.

## Acceptance Criteria

- [x] Production provider/planning and/or target validation has a focused diff
      that tightens the provider-owned base-memory fact API boundary for both
      indexed gather and indexed scatter.
- [x] Target artifact validation consumes the provider fact surface for ABI
      order, memory form, indexed layout, index EEW, offset unit, index source,
      scatter index uniqueness, source/destination memory-form polarity,
      route-family plan, binding/header facts, C type mapping, target profile,
      and provider mirror.
- [x] C++ target validation tests prove fail-closed behavior for gather/scatter
      polarity mismatches, stale gather facts on scatter, stale scatter facts on
      gather, ABI order mismatch, index EEW mismatch, offset unit mismatch,
      scatter index uniqueness mismatch, source/destination memory-form
      mismatch, route-family plan mismatch, binding/header/type mismatch,
      target profile mismatch, provider mirror mismatch, and accidental
      unit/strided fallback residue.
- [x] Generated-bundle dry-run evidence remains bounded and confirms the
      production surface still supports explicit and pre-realized selected-body
      indexed gather/scatter paths.
- [x] No new source-front-door, descriptor-driven, route-id, artifact-name,
      common-EmitC, mirror-only, or legacy `i32m1` route authority is introduced.
- [x] Focused build/tests, `git diff --check`, bounded old-authority scan,
      Trellis finish/archive, clean worktree, and one coherent commit complete
      this round.

## Technical Approach

Inspect the current `RVVBaseMemoryMovementRouteFacts` provider surface and the
base-memory branch of `RVVTargetArtifactRouteFamilyValidation.cpp`. If the
production path already shares most constants, consolidate the target-side
validation API so the indexed gather/scatter checks operate through one
validated provider-fact comparison helper instead of scattered field checks or
target-local reconstruction. Add focused C++ mutations that exercise the
production validator through provider descriptions and candidate mirrors, not
only standalone artifact strings.

## Decision (ADR-lite)

**Context**: Indexed gather and scatter are symmetric base-memory movement
routes, but the last two closeout tasks were per-route evidence tasks. A durable
Stage 2 route family needs one production provider/target validation boundary.

**Decision**: Treat the RVV provider-owned base-memory facts accessor as the
canonical source and make target validation compare all shared indexed fields
against that accessor for both gather and scatter. Tests should mutate provider
descriptions and mirrors to prove fail-closed production behavior.

**Consequences**: This keeps the production compiler path authoritative and
reduces drift between gather and scatter. It does not expand Stage 2 coverage,
rerun a broad RVV runtime matrix, or move semantics into common EmitC/export.

## Out Of Scope

- No new RVV kernel family.
- No masked indexed path, segment path, dtype/LMUL/index-width clone batch, or
  broad indexed matrix.
- No high-level Linalg/Vector/frontend/source-front-door positive lowering.
- No standalone artifact evidence task as the main deliverable.
- No common EmitC/export logic that chooses RVV semantics.
- No descriptor-driven computation or metadata/mirror route authority.
- No new dtype-prefixed `tcrv_rvv.i32_*` helper family.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior tasks read:
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-indexed-gather-unit-store-artifact-abi-boundary/`,
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-indexed-scatter-unit-load-artifact-abi-boundary/`.
- Journal context read: `.trellis/workspace/codex/journal-21.md` sessions
  398 and 399.
- Likely production files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.

## Definition Of Done

- Production code owns the indexed base-memory provider/target validation
  boundary for both gather and scatter.
- Focused tests pass and prove the production validator fails closed on stale
  or cross-contaminated facts.
- Existing generated-bundle indexed gather/scatter dry-runs still pass.
- Trellis task state, journal, archive, and commit are truthful.

## Completion Notes

- Removed the target validator's duplicated base-memory expected-field accessor
  layer and made layout/payload validation consume the single
  `RVVBaseMemoryMovementRouteFacts` object already returned by the RVV provider
  surface.
- Added indexed gather/scatter target validation for provider-derived typed
  compute op facts and for candidate `rvv_selected_body_typed_compute_op`
  mirrors, scoped to indexed base-memory routes so masked base-memory routes are
  not pulled into this closeout.
- Added C++ fail-closed coverage for stale indexed gather ABI order, stale
  scatter uniqueness/destination residue on gather, stale indexed gather and
  scatter typed compute facts, and stale typed-compute candidate mirrors.
- Existing scatter/gather stale fact tests still cover ABI order, index source,
  indexed layout, index EEW, offset unit, scatter uniqueness, source/destination
  memory forms, indexed data/destination forms, route-family plan, target
  profile, provider mirror, header/type facts, binding summary, and accidental
  unit/strided/gather residue.
- Bounded generated-bundle dry-runs passed for explicit and pre-realized
  indexed gather/scatter with runtime counts `0,1,16,17,257`.
- Real `ssh rvv` was not rerun as the main deliverable because this closeout
  changed production validation/fact-boundary enforcement only; the immediately
  preceding archived gather and scatter tasks already recorded real RVV
  correctness for both explicit and pre-realized indexed routes over the same
  counts and index-pattern families.
