# Stage2 RVV plain segment2 runtime ABI closure

## Goal

Close the Hermes-requested plain segment2 tuple-memory runtime ABI round for
`segment2_deinterleave_unit_store` and `segment2_interleave_unit_load` against
current HEAD truth. The selected RVV paths must be proven from `tcrv.exec`
ABI/runtime roles through typed or pre-realized `tcrv_rvv` segment2 tuple body
facts, RVV selected-body realization/provider facts, `TCRVEmitCLowerableRoute`,
neutral EmitC, target artifact validation, generated RVV C harnesses, and
`ssh rvv` correctness evidence.

Current inspection shows the production selected-body/provider path already
carries the plain segment2 route closure: plain segment2 bodies are
selected-boundary-only, direct route-entry family registry is empty, and the RVV
provider builds the route from segment2 family/provider plans. The target
artifact consumer, however, only had hardcoded provider-fact validation for the
computed-mask segment2 family; plain deinterleave/interleave were still mostly
checked by comparing candidate mirrors against the supplied provider
description. This round must close that production target-consumer gap and add
focused stale-fact negatives for tuple field roles, source/destination memory
forms, provider mirrors, route operand binding mirrors, and runtime ABI order,
plus dry-run and hardware evidence over boundary counts.

## What I already know

- The repo began this round at `/home/kingdom/phdworks/TianchenRV`, branch
  `main`, clean worktree, HEAD `4f4af653 rvv: close computed-mask segment2
  update-unit-load abi evidence`.
- `.trellis/.current-task` did not exist; this task was created from the Hermes
  Direction Brief.
- Archived task `05-29-stage2-rvv-computed-mask-segment2-update-unit-load-runtime-abi-closure`
  passed target artifact validator, focused lit, explicit/pre-realized
  generated-bundle dry-runs, direct route-entry fail-closed, `ssh rvv`, and
  `check-tianchenrv` evidence for the computed-mask update path.
- `.trellis/workspace/codex/journal-18.md` records that computed-mask
  load/store and update closures intentionally left plain segment2
  deinterleave/interleave as the next adjacent closure.
- `RVVSelectedBodyRealization.cpp` classifies
  `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` and
  `TypedSegment2InterleaveMemoryPreRealizedBodyOp` under the segment2 memory
  realization owner, but that owner has no direct route-entry predicate; direct
  pre-realized segment2 route-entry remains selected-boundary-only/fail-closed.
- `RVVEmitCRoutePlanning.cpp` contains route operand binding plan IDs,
  runtime ABI orders (`src,out0,out1,n` and `src0,src1,dst,n`), provider
  mirrors, source/destination memory forms, field roles, segment count, and
  statement-plan construction for the two plain segment2 paths.
- `RVVEmitCRouteProvider.cpp` consumes the RVV-owned segment2 provider plan to
  add headers, type mappings, ABI mappings, and statement plans before creating
  the lowerable route.
- `RVVTargetArtifactRouteFamilyValidation.cpp` validated computed-mask
  segment2 provider facts against hardcoded expectations but did not yet do the
  same hardcoded production check for plain deinterleave/interleave tuple field
  roles, memory forms, provider mirrors, route operand binding summaries, and
  runtime ABI order.
- Existing lit/script coverage already includes explicit and pre-realized
  selected-body artifact files for both plain paths and direct pre-realized
  fail-closed script tests, but current C++ target artifact stale-fact tests are
  stronger for computed-mask update than for the two plain paths.

## Scope

- Production code may change only if inspection or focused checks prove the
  plain segment2 selected-body/provider/target path is stale or incomplete:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Expected implementation target if production remains correct:
  - `test/Target/TargetArtifactExportTest.cpp`, adding focused plain
    deinterleave/interleave stale-fact negatives for runtime ABI order, tuple
    field roles, source/destination memory forms, provider mirrors, and route
    operand binding mirrors.
- Direct verification consumers:
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/RVV/explicit-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-segment2-interleave-unit-load.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`
  - directly related `test/Scripts/rvv-generated-bundle-abi-e2e-*segment2-*`
    tests for the two plain paths.

## Requirements

- Plain segment2 deinterleave/interleave must remain selected-boundary-only;
  direct pre-realized route-entry must fail closed.
- `tcrv.exec` must remain the execution/ABI envelope. RVV tuple memory,
  field roles, memory forms, route construction, header/type/intrinsic mapping,
  runtime ABI order, and statement sequence remain RVV-provider-owned.
- `segment2_deinterleave_unit_store` must preserve and validate:
  - source role `src` / `lhs-input-buffer`;
  - field outputs `out0`, `out1` with distinct output roles;
  - source memory form `segment2-interleaved-unit-stride-load`;
  - destination memory form `unit-stride-store`;
  - runtime ABI order `src,out0,out1,n`;
  - provider mirror `provider_supported_mirror:rvv-segment2-deinterleave-plan-validated`;
  - route operand binding plan and operands for segment load, field stores, and
    runtime AVL/VL;
  - segment count 2, tuple C type, segment load/extract/store statement facts,
    field order, and tail preservation.
- `segment2_interleave_unit_load` must preserve and validate:
  - field inputs `src0`, `src1` with distinct input roles;
  - destination role `dst` / `segment-interleaved-output-buffer`;
  - source memory form `unit-stride-load`;
  - destination memory form `segment2-interleaved-unit-stride-store`;
  - runtime ABI order `src0,src1,dst,n`;
  - provider mirror `provider_supported_mirror:rvv-segment2-interleave-plan-validated`;
  - route operand binding plan and operands for field loads, tuple creation,
    segment store, and runtime AVL/VL;
  - segment count 2, tuple C type, segment store/tuple statement facts, field
    order, and tail preservation.
- Any new tests must fail closed on stale provider or artifact mirrors rather
  than accepting names, route ids, artifact names, ABI strings, script
  expectations, exact intrinsic spelling, descriptors, source-front-door state,
  common EmitC inference, direct-route-entry-only paths, pre-realized fixtures
  alone, or legacy i32-derived authority.

## Acceptance Criteria

- [x] Current production selected-body/provider/target path for both plain
  segment2 operations is inspected and either changed if stale, or explicitly
  recorded as already correct.
- [x] Target artifact validator C++ tests include plain
  `segment2_deinterleave_unit_store` stale negatives for tuple field role,
  source/destination memory form, provider mirror, route operand binding
  mirror, and runtime ABI order.
- [x] Target artifact validator C++ tests include plain
  `segment2_interleave_unit_load` stale negatives for tuple field role,
  source/destination memory form, provider mirror, route operand binding
  mirror, and runtime ABI order.
- [x] Existing explicit selected-body artifact lit coverage still validates
  provider-supported mirrors, route operand binding, ABI order, headers, type
  mapping, segment family plan, field roles, memory forms, and segment count.
- [x] Existing pre-realized selected-boundary coverage still validates
  realization into typed segment load/move/store or load/load/segment-store
  structure and rejects stale mirrors where covered.
- [x] Generated-bundle dry-runs pass for explicit and pre-realized
  selected-boundary deinterleave/interleave over counts `0`, `1`, exact-VL,
  tail, and stress cases.
- [x] Direct pre-realized route-entry dry-runs remain fail-closed for both
  plain segment2 operations.
- [x] `ssh rvv` explicit and pre-realized runs pass for both paths over counts
  `0`, `1`, exact-VL, tail, and stress cases, proving tuple field correctness,
  field order, tail preservation, and runtime ABI order.
- [x] Computed-mask segment2 load/store/update remain covered as
  non-regression.
- [x] Bounded touched-file authority scan finds no central ad hoc,
  name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
  script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, exact-intrinsic-derived,
  direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
  executable authority.
- [x] `git diff --check` passes.
- [x] Focused checks pass; `check-tianchenrv` passes or an exact blocker is
  recorded.
- [x] Trellis task status, check notes, archive state, journal, and commit are
  truthful.

## Non-Goals

- Do not reopen computed-mask segment2 load/store/update, direct route-entry
  demotion, widening conversion, widening dot, compare/select, reduction,
  dtype/LMUL clone batches, high-level Linalg/frontend lowering, dashboards,
  reports, or broad smoke matrices.
- Do not restore direct pre-realized segment2 route-entry support.
- Do not move RVV segment2 semantics into common EmitC/export.
- Do not treat prompt edits, helper-only changes, broad smoke tests, or report
  text as the main achievement.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Archived/current context read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-computed-mask-segment2-update-unit-load-runtime-abi-closure/`
  - `.trellis/workspace/codex/journal-18.md`
- Current inspection points:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - explicit/pre-realized plain segment2 selected-body artifact fixtures and
    generated-bundle script tests.

## Definition of Done

- The plain segment2 selected-boundary/runtime ABI closure is either
  production-fixed or verified current, with missing focused validator evidence
  added.
- Focused dry-runs, target tests, direct route-entry fail-closed checks,
  `ssh rvv` evidence, authority scan, and final repository checks are recorded.
- The task is finished/archived and one coherent commit is created if all
  acceptance criteria pass.
