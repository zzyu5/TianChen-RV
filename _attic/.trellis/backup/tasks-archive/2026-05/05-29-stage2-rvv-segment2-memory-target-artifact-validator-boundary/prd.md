# Stage2 RVV segment2 memory target artifact validator boundary

## Goal

Factor the completed segment2-memory target artifact checks into a coherent
target-owned route-family validator boundary. The production target artifact
dispatcher may remain neutral routing glue, but segment2-memory-specific
artifact acceptance for `computed_masked_segment2_load_unit_store`,
`computed_masked_segment2_store_unit_load`,
`computed_masked_segment2_update_unit_load`,
`segment2_deinterleave_unit_store`, and `segment2_interleave_unit_load` must
live behind the segment2-memory family boundary and consume provider-derived
facts from the rebuilt RVV route.

This round is not an evidence-only follow-up. It must move production code in
the RVV target artifact validation owner while preserving selected-body
semantics, RVV provider route authority, neutral EmitC materialization, target
artifact export, generated-bundle ABI behavior, and existing `ssh rvv`
evidence for the recently closed computed-mask segment2 update path.

## What I already know

- The repo had no current Trellis task when this round began; this task was
  created from the Hermes Direction Brief.
- Initial repository state was `/home/kingdom/phdworks/TianchenRV`, `main`,
  clean worktree, HEAD `4036408c rvv: close computed-mask segment2 update runtime abi`.
- The previous archived task closed `computed_masked_segment2_update_unit_load`
  runtime ABI evidence and left a large segment2-specific block in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Specs require target artifact export to rebuild the provider route and treat
  candidate metadata as mirrors only.
- Segment2 target export must validate provider support mirrors, runtime ABI
  order, operand binding summaries, headers, C type mappings, segment layout,
  memory forms, segment count, mask facts for computed-mask segment2, tuple
  field roles, update/pass-through facts, and the runtime `n`/AVL/VL relation.
- Direct pre-realized segment2 route-entry support must remain fail-closed for
  both plain and computed-mask segment2 bodies.

## Scope

- Production owner:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- Direct consumers and evidence, only as needed:
  - `test/Target/TargetArtifactExportTest.cpp`
  - directly related segment2 target/lit tests
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - directly related generated-bundle script tests
- Provider/planning headers or implementation may be read for authority checks
  and touched only if a minimal signature adjustment is required by the target
  validator boundary. They are not the primary owner for this task.

## Requirements

- Introduce or sharpen a target-owned segment2-memory route-family artifact
  validator boundary, with the central target dispatcher limited to neutral
  route-family dispatch.
- Segment2-memory-specific target validation must live behind that boundary for
  all five active family entries:
  - `computed_masked_segment2_load_unit_store`
  - `computed_masked_segment2_store_unit_load`
  - `computed_masked_segment2_update_unit_load`
  - `segment2_deinterleave_unit_store`
  - `segment2_interleave_unit_load`
- The validator must consume the rebuilt `TCRVEmitCLowerableRoute` and provider
  description as authority. It must not authorize export from route ids,
  artifact names, ABI strings, script expectations, metadata, exact intrinsic
  spellings, source-front-door state, descriptor residue, common EmitC
  branches, direct route-entry support, pre-realized fixtures alone, or legacy
  i32-derived facts.
- The segment2-memory validator must fail closed with targeted diagnostics for
  stale or missing:
  - operation kind and route-family classification;
  - computed mask provenance, mask role/source/memory-form, and mask/tail facts;
  - tuple-field roles and segment count;
  - update arithmetic and pass-through facts for update routes;
  - source/destination memory roles and memory form;
  - runtime ABI order;
  - `tcrv_rvv.provider_supported_mirror`;
  - route operand binding plan and summary;
  - required headers and C/RVV type mappings;
  - AVL/VL/runtime `n` facts;
  - stale non-family facts or mirrors from another segment2 subfamily.
- Existing positive explicit and pre-realized selected-boundary segment2
  generated artifact paths must continue to pass.
- Direct pre-realized segment2 route-entry probes must remain fail-closed.
- Common EmitC/export mechanics must remain neutral and must not gain RVV
  segment2 semantic branches.

## Acceptance Criteria

- [ ] Production diff creates a segment2-memory target artifact validator
  boundary in `RVVTargetArtifactRouteFamilyValidation.cpp`; the central
  dispatcher is neutral routing glue.
- [ ] Target tests prove all five segment2-memory route families still accept
  provider-derived route facts after the boundary split.
- [ ] Target tests reject stale/missing operation kind, mask facts, tuple field
  roles, update/pass-through facts, memory roles, runtime ABI order,
  `provider_supported_mirror`, route operand binding summary, type/header
  mappings, AVL/VL/runtime `n` facts, and non-family residue.
- [ ] Generated-bundle dry-runs pass for computed-mask segment2 update explicit
  and pre-realized selected-boundary paths with direct `route_entry_realization`
  remaining fail-closed.
- [ ] Dry-run or lit non-regression passes for computed-mask segment2 load/store
  and base segment2 deinterleave/interleave.
- [ ] `ssh rvv` non-regression passes for the recently closed computed-mask
  segment2 update explicit and pre-realized selected-boundary paths, or an exact
  hardware blocker is reported.
- [ ] Bounded touched-file authority scan finds no central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, exact-intrinsic-derived,
  direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
  executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused tests pass; `check-tianchenrv` passes or an exact blocker is
  recorded.
- [ ] Task status, context, and journal are truthful; task is finished/archived
  and one coherent commit is created if all criteria pass.

## Non-Goals

- Do not add a new RVV operation family or widen dtype/LMUL coverage.
- Do not resurrect direct pre-realized route-entry support for segment2.
- Do not change selected-body realization semantics or provider route authority.
- Do not move RVV semantics into common EmitC/export.
- Do not create source-front-door positive routes, high-level Linalg/Vector
  frontend lowering, dashboards, reports, or broad smoke matrices.
- Do not rewrite every route-family validator in this round.
- Do not make a cosmetic split that preserves central segment2 semantic
  accumulation without a real target-owned family boundary.

## Technical Notes

- Required specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Previous archived task read:
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-computed-mask-segment2-update-runtime-abi-closure/`
- Code files to inspect before implementation:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/TargetArtifactExportTest.cpp`

## Definition of Done

- Segment2-memory target artifact validation is family-local behind a clear
  target-owned boundary.
- Provider-derived facts, generated artifact paths, direct route-entry
  fail-closed behavior, and RVV runtime evidence are preserved.
- Focused checks plus either `check-tianchenrv` or an exact blocker are recorded.
- The task is finished/archived and committed when complete.
