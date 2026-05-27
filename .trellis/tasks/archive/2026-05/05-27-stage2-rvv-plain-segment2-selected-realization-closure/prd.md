# Stage2 RVV plain segment2 selected realization closure

## Goal

Migrate the remaining plain segment2 generated-bundle production paths,
`segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`, away
from active direct pre-realized route-entry shortcut authority. Both paths must
run through the selected RVV lowering-boundary realization flow, produce
realized typed `tcrv_rvv` segment2 memory facts, feed RVV provider route facts
and `TCRVEmitCLowerableRoute`, materialize through neutral EmitC, validate the
target artifact ABI, and produce real `ssh rvv` correctness evidence.

## Direction Source

Hermes Direction Brief:

- Direction title: `Switch: Stage2 RVV plain segment2 selected-body realization closure`.
- Module owner: RVV plugin-local selected-body realization producer and
  generated-bundle boundary for `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load`.
- Previous completed task: computed-masked segment2 load/store/update direct
  route-entry shortcuts were demoted in commit `f02955e8`.
- Current bottleneck: bounded inspection shows only the two plain segment2 ops
  remain as positive `supports_direct_pre_realized_route_entry` support.

## What I Already Know

- The repository is on `main`, clean before this task, with `f02955e8 rvv:
  demote computed mask segment2 route entries` at HEAD.
- There was no active `.trellis/.current-task`, so this task was created from
  the current Direction Brief.
- `.trellis/spec/index.md` defines the current RVV-first authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- `tcrv.exec` owns ABI/runtime envelope roles such as `mem_window` and
  `runtime_param`; it does not own RVV compute, dtype, memory direction,
  schedule, intrinsic spelling, or route support.
- Common EmitC owns neutral materialization of provider-built routes only; it
  must not infer segment2 semantics from route ids, ABI strings, artifact names,
  metadata, exact intrinsics, or scripts.
- The current RVV plugin spec still describes plain segment2 deinterleave and
  interleave as direct route-entry capable. This task intentionally narrows the
  production generated-bundle path so they become selected-boundary realization
  paths, while retaining typed segment2 provider and artifact evidence.
- The previous journal records that computed-mask segment2 selected-boundary
  migration passed focused C++/lit, `ssh rvv`, authority scan, and
  `check-tianchenrv` 403/403.

## Requirements

1. Demote `segment2_deinterleave_unit_store` and
   `segment2_interleave_unit_load` from positive direct pre-realized route-entry
   generated-bundle authority.
2. Preserve or repair the public selected lowering-boundary path so both ops
   realize through the RVV plugin-local selected-body realization producer.
3. Ensure realized bodies preserve segment2 grouping, lane pairing,
   deinterleave/interleave direction, memory roles, element dtype, SEW/LMUL,
   tail/mask policy, runtime `n`/AVL/VL, `setvl` placement, operand binding,
   required capabilities, provider route facts, and artifact ABI order.
4. Ensure route planning/provider consumes typed selected-boundary facts and
   owner-built segment2 planning/statement-plan facts before building
   `TCRVEmitCLowerableRoute`.
5. Ensure target/generated-bundle artifact validation consumes rebuilt provider
   route and ABI facts, not script flags, artifact names, route ids, stale
   mirrors, exact intrinsic spelling, or metadata.
6. Unsupported or inconsistent selected-body input must fail closed with
   targeted diagnostics before provider/common EmitC route construction.
7. After this task, no positive generated-bundle production path should remain
   in `supports_direct_pre_realized_route_entry`; any retained mention must be
   explicitly negative/fail-closed inventory or historical/spec text.

## Acceptance Criteria

- [x] Both plain segment2 ops use the selected lowering-boundary generated-bundle
      path with `route_entry_realization: false`.
- [x] Direct pre-realized route-entry requests for both plain segment2 ops fail
      closed or are explicitly reported unsupported, with diagnostics that point
      to selected-boundary-only ownership.
- [x] Focused C++ and/or lit coverage proves realized typed segment2 facts are
      consumed by the provider path for both plain segment2 ops.
- [x] Focused negative coverage rejects missing or stale dependencies for the
      migrated path, including at least representative cases for direct-only
      route authority, stale route id/mirror metadata, wrong memory role or
      segment2 direction, and missing runtime/mem_window facts where the current
      test surfaces support them.
- [x] Generated-bundle dry-runs for both ops pass through the selected
      lowering-boundary path and expose provider/artifact mirrors as mirrors
      only.
- [x] Real `ssh rvv` generated-bundle runs pass for counts covering
      `0, 1, exact, tail, stress` with representative signed lane-pair data, or
      the exact blocker is recorded.
- [x] Existing computed-mask segment2 selected-boundary paths do not regress.
- [x] A bounded authority scan over touched RVV/plugin/target/script/test files
      finds no new central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived, artifact-name-
      derived, common-EmitC-derived, source-front-door-derived, route-id-
      derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.

## Completion Notes

- The segment2 selected-body realization owner remains the owner for plain
  deinterleave/interleave and computed-mask segment2 bodies, but it no longer
  declares direct route-entry eligibility.
- The generated-bundle script now reports `--direct-pre-realized-route-entry`
  as unsupported for current pre-realized selected-body fixtures, including
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
- Verified selected-boundary dry-runs and real `ssh rvv` runs for both plain
  segment2 ops with runtime counts `0,1,16,17,257`.
- Verified computed-mask segment2 selected-boundary dry-run non-regression,
  focused C++ tests, `git diff --check`, and `check-tianchenrv` 405/405.

## Out Of Scope

- Computed-masked segment2 migration beyond non-regression checks.
- Computed-mask widening dot, widening dot, MAcc, compare/select, standalone
  reduction, strided load/store, widening conversion, dtype/LMUL clone batches,
  high-level Linalg/frontend lowering, one-intrinsic wrapper dialects, selected
  realization framework rewrites, dashboard/report work, broad smoke matrices,
  or evidence-only changes.
- Adding a new positive direct route-entry owner for the plain segment2 pair.
- Treating generated artifact names, script allowlists, route ids, exact
  intrinsic spellings, metadata, or common EmitC logic as support authority.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Relevant previous-session facts read:
  - `.trellis/workspace/codex/journal-17.md`, Session 271.
- Initial production owner files to inspect before implementation:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
- Focused generated-bundle and target tests are consumers/evidence, not route
  authority.

## Validation Plan

1. Inspect the current direct route-entry predicate/script support surface for
   `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
2. Demote the direct route-entry/generated-bundle support while preserving the
   selected lowering-boundary realization path.
3. Update focused C++/lit/script tests so positive evidence uses
   `route_entry_realization: false` and direct shortcut requests fail closed.
4. Run focused build/test targets first, then generated-bundle dry-runs and
   real `ssh rvv` executions for both ops.
5. Run computed-mask segment2 non-regression checks, authority scans,
   `git diff --check`, and `check-tianchenrv` if no earlier exact blocker
   appears.
