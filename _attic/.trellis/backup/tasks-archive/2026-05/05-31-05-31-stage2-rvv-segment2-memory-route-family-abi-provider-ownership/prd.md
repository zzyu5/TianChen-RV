# PRD: Stage2 RVV segment2 memory route-family ABI/provider ownership

## Direction Source

Hermes/user direction switches the next bounded owner to the RVV target artifact
ABI/provider boundary for the segment2 memory route family:

- `segment2_deinterleave_unit_store`
- `segment2_interleave_unit_load`
- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`
- `computed_masked_segment2_update_unit_load`

The previous segment2 generated-artifact task proved selected-boundary dry-run,
real `ssh rvv`, direct-route-entry fail-closed, and `check-tianchenrv`
evidence. It did not move production code. This round must therefore close a
production target-artifact consumer boundary, not repeat an evidence-only
archive.

## Initial Repository State

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `2a81ac89 rvv: close indexed memory artifact ABI boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Module Goal

Harden the production boundary where RVV segment2 provider-derived route facts
are consumed by target artifact validation and candidate metadata is accepted
only as an exact mirror after route construction.

The bounded production submodule for this round is plain segment2 candidate
mirror validation in `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
Computed-mask segment2 mirrors already validate tuple C type, segment
load/store/tuple/field intrinsics, field roles, field names, field memory forms,
mask facts, passthrough facts, update arithmetic facts, runtime ABI order,
route operand binding, headers, type summaries, and provider-supported mirrors.
Plain segment2 provider facts are validated, but the candidate mirror consumer
still does not validate all plain segment2 tuple/field/intrinsic mirrors even
though the generated artifacts emit those mirrors.

## What I Already Know

- The current mainline is RVV-first selected body:
  `tcrv.exec` selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  selected-body realization/provider -> `TCRVEmitCLowerableRoute` -> neutral
  EmitC -> target artifact -> `ssh rvv` evidence when runtime/correctness is
  claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define segment2 compute, tuple type, field roles, memory
  forms, mask/pass-through behavior, route support, or artifact acceptance.
- The segment2 route provider and generated-artifact evidence are already
  present for all five forms.
- Target artifact export must dispatch route-family validation from the rebuilt
  provider description, not from metadata, route ids, artifact names, scripts,
  ABI strings, direct route entries, or exact intrinsic spelling.
- Existing target validation already has a segment2 route-family validator
  registry and positive coverage for all five forms.
- Existing target validation already rejects many provider-side stale segment2
  facts, route statement mutations, stale route id, stale route family facts,
  stale provider mirror, stale binding mirror, stale runtime ABI order, stale
  mask/pass-through facts, and direct route-entry-only authority.
- Remaining bounded gap: plain `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load` candidate mirrors should consume the same
  provider-derived tuple/field/intrinsic facts that are already emitted as
  artifact metadata and already validated for computed-mask segment2 paths.

## Requirements

1. Move production code in the RVV target artifact route-family validation path.
   Tests alone are not sufficient.
2. Keep common EmitC neutral. Do not add RVV segment2 semantic choices to
   common EmitC, scripts, or generic target support.
3. Accept segment2 artifacts only after the rebuilt provider route and provider
   description establish operation kind, tuple/vector C type summary, field0
   and field1 roles, field names, source/destination memory forms, segment
   count, runtime ABI order, AVL/VL loop facts, required headers, route operand
   binding, and provider-supported mirror semantics.
4. For computed-mask segment2 paths, preserve existing validation of computed
   mask producer/source/mask role/mask type, inactive/pass-through policy, and
   update arithmetic facts.
5. For plain segment2 paths, add or repair candidate mirror validation for:
   - `tcrv_rvv.segment_tuple_c_type`;
   - `tcrv_rvv.segment_load_intrinsic` for deinterleave and absence for
     interleave;
   - `tcrv_rvv.segment_store_intrinsic` for interleave and absence for
     deinterleave;
   - `tcrv_rvv.segment_field_extract_intrinsic` for deinterleave and absence
     for interleave;
   - `tcrv_rvv.segment_tuple_create_intrinsic` for interleave and absence for
     deinterleave;
   - `tcrv_rvv.field0_role`, `tcrv_rvv.field1_role`;
   - `tcrv_rvv.field0_name`, `tcrv_rvv.field1_name`;
   - deinterleave field destination memory-form mirrors;
   - interleave field source memory-form mirrors.
6. Stale computed-mask facts on plain segment2 candidates and stale plain
   segment2 facts on computed-mask segment2 candidates must continue to fail
   closed with targeted diagnostics.
7. Preserve indexed memory artifact ABI closure from commit `2a81ac89`.
8. Preserve generated-bundle segment2 behavior and direct pre-realized
   route-entry retirement.

## Non-Goals

- Do not reopen indexed gather/scatter, base strided/masked memory, widening
  dot, MAcc, standalone reduction, compare/select, conversion, or non-segment2
  route-family owners.
- Do not add new segment2 operations, segment counts, dtype/LMUL clones, source
  front-door routes, high-level frontend lowering, one-intrinsic wrapper
  dialects, dashboards, reports, broad smoke matrices, or compatibility paths
  that restore direct pre-realized route-entry authority.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission in Python. Python remains evidence
  tooling only.

## Acceptance Criteria

- [x] Production diff lands in the segment2 target artifact validation owner.
- [x] Positive target artifact validation still accepts all five segment2 forms.
- [x] C++ target artifact tests prove plain segment2 deinterleave candidate
      mirrors fail closed for stale tuple C type, stale field role/name, stale
      field destination memory form, and stale/missing deinterleave-specific
      segment load / field extract mirrors.
- [x] C++ target artifact tests prove plain segment2 interleave candidate
      mirrors fail closed for stale tuple C type, stale field role/name, stale
      field source memory form, and stale/missing interleave-specific tuple
      create / segment store mirrors.
- [x] Existing computed-mask segment2 candidate mirror tests continue to pass,
      including stale mask, field, update arithmetic, runtime ABI, provider
      mirror, binding, source/destination memory, header, type, runtime control,
      and stale plain route-family mirrors.
- [x] Focused generated-bundle dry-run covers all five segment2 forms through
      the selected lowering-boundary path with route entry realization disabled.
- [x] Real `ssh rvv` generated-bundle correctness runs for representative
      counts `0`, `1`, exact-VL, tail, and stress cases, or an exact external
      blocker is recorded.
- [x] Non-regression covers indexed memory artifact ABI closure from
      `2a81ac89`.
- [x] Bounded authority scan over touched production/test/task files finds no
      new central ad hoc, name-derived, metadata-derived, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [x] `git diff --check` passes.
- [x] Focused checks pass, and `check-tianchenrv` passes or the exact blocker
      is recorded.
- [x] Task status, context, journal, evidence summary, and archive state are
      truthful; if acceptance passes, create one coherent commit.

## Validation Plan

1. Build focused target artifact test:
   `cmake --build build --target tianchenrv-target-artifact-export-test -j2`.
2. Run focused C++ target artifact coverage:
   `./build/bin/tianchenrv-target-artifact-export-test`.
3. Run selected-boundary generated-bundle dry-run for the five segment2 forms
   with counts `0,1,16,23,257` and no direct route-entry shortcut.
4. Run the same five-form generated-bundle execution on real `ssh rvv`, unless
   an exact external blocker is recorded.
5. Run direct route-entry fail-closed probe for the segment2 forms.
6. Run indexed memory non-regression from the previous task, at least the
   focused target artifact/export test and generated-bundle dry-run covering the
   four indexed base/computed-mask forms.
7. Run bounded authority scans over touched files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior tasks read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-indexed-memory-route-family-abi-provider-ownership/`,
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-segment2-generated-artifact-executable-boundary/`,
  and
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-segment2-route-family-provider-boundary-closure/`.
- Primary production file for this bounded module:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Directly related consumer test:
  `test/Target/TargetArtifactExportTest.cpp`.
- Evidence tooling:
  `scripts/rvv_generated_bundle_abi_e2e.py`.

No blocking user question remains. The supplied Direction Brief, current specs,
archived task boundaries, and code inspection are specific enough for one
bounded implementation round.

## Completion Notes

Implemented plain segment2 target artifact candidate mirror validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`. Plain
deinterleave/interleave candidates now consume provider-derived
`segment_tuple_c_type`, segment load/store/tuple/field intrinsic mirrors,
field0/field1 roles, field names, and field source/destination memory-form
mirrors before artifact acceptance. Stale computed-mask update mirrors on plain
segment2 candidates also fail closed.

Extended `test/Target/TargetArtifactExportTest.cpp` with focused stale/missing
candidate mirror coverage for plain segment2 deinterleave and interleave while
preserving existing computed-mask segment2 and indexed memory behavior.

Evidence passed:

- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Five-form segment2 selected-boundary generated-bundle dry-run:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/dry-run/pre-realized-segment2`
- Segment2 direct pre-realized route-entry failed closed as expected:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/direct-route-entry-fail-closed/direct-pre-realized-segment2`
- Five-form `ssh rvv` generated-bundle run passed for counts `0,1,16,23,257`:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/ssh-rvv/pre-realized-segment2`
- Indexed memory non-regression dry-run passed:
  `artifacts/tmp/05-31-stage2-rvv-segment2-memory-route-family-abi-provider-ownership/non-regression/pre-realized-indexed-memory`
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`: 464/464 passed.

Spec update judgment: no `.trellis/spec/**` update was needed because this
round implements the already-stated segment2 target export consumer contract:
provider route facts are authority and candidate metadata is mirror-only after
route construction.
