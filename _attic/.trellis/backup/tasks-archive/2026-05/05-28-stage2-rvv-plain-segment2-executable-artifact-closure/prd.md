# Stage2 RVV plain segment2 executable artifact closure

## Goal

Prove, and repair if current evidence fails, the executable generated-bundle
artifact path for the selected-boundary-only RVV plain segment2 bodies
`segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.

This task starts from the current RVV-first path:

```text
selected tcrv.exec RVV variant
  -> public selected lowering-boundary producer
  -> RVV plugin-local selected-body realization
  -> realized typed tcrv_rvv segment2 facts
  -> RVV route-family/provider/statement-plan facts
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC
  -> target artifact bundle
  -> ssh rvv compile/run/correctness evidence
```

Direct pre-realized route-entry shortcuts for these bodies must remain
unsupported.

## Direction Source

Hermes Direction Brief:

- Direction title: `Expand: Stage2 RVV plain segment2 selected-body executable artifact closure`.
- Module owner: selected-boundary-to-target-artifact path for
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
- Requested improvement: turn the selected-boundary-only plain segment2 path
  into executable generated-bundle evidence, repairing the narrow production
  owner if current artifact compile/run/correctness fails.

Repository evidence read before implementation:

- Current HEAD is `5c2d39a3 rvv: demote plain segment2 route entries`, with a
  clean worktree before this task was created.
- There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief and set active.
- The archived `05-27-stage2-rvv-plain-segment2-selected-realization-closure`
  task and `.trellis/workspace/codex/journal-17.md` record prior real `ssh rvv`
  runs for both plain segment2 ops with counts `0,1,16,17,257`.
- Because the Hermes brief says the remaining gap may be dry-run-only evidence,
  this task does not rely on those archived claims as current truth. It reruns
  the current HEAD generated-bundle artifact path and treats any compile/run or
  correctness failure as the production blocker to fix.

## Requirements

1. Preserve selected-boundary-only authority for both plain segment2 bodies.
   `--direct-pre-realized-route-entry` must remain fail-closed for
   `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
2. The public selected lowering-boundary path must consume the pre-realized
   body and report `route_entry_realization: false` and
   `pre_realized_body_consumed: true`.
3. Realized typed `tcrv_rvv` segment2 structure must preserve segment count,
   direction, source/destination memory roles, element dtype/config/policy,
   runtime `n`/AVL/VL, setvl placement, ABI argument order, pointer roles, lane
   order, tail behavior, and provider route facts.
4. Route planning/provider/statement-plan construction must consume verified
   typed body/config/runtime/capability facts and RVV-owned segment2 plans. It
   must not infer segment2 semantics from route ids, artifact names, ABI
   strings, tests, descriptors, exact intrinsic spellings, common EmitC, or
   script metadata.
5. Target artifact export must consume the rebuilt provider route and compare
   segment2 mirrors as mirrors only before accepting executable artifact/header
   claims.
6. Non-dry-run generated-bundle evidence must compile and run on `ssh rvv` for
   both plain segment2 bodies. Runtime counts must include `0`, `1`, an exact
   vector-sized case, a tail case, and a stress case.
7. Correctness evidence must check signed element patterns, deinterleave and
   interleave lane order, output preservation outside active `n`, tail
   preservation, and ABI argument order as supported by the generated harness.
8. If current HEAD already satisfies the executable path, this task may be a
   validation-closure round with no production code changes. If any check
   fails, repair the narrow blocker in selected-body realization, route
   planning/provider, target artifact validation, generated harness binding, or
   neutral EmitC materialization.

## Acceptance Criteria

- [x] Dry-run generated-bundle evidence for both ops shows
      `route_entry_realization: false` and `pre_realized_body_consumed: true`.
- [x] Direct pre-realized route-entry probes for both ops fail closed before
      bundle generation.
- [x] Non-dry-run generated-bundle execution on `ssh rvv` passes for both ops
      with counts covering `0`, `1`, exact, tail, and stress cases.
- [x] Evidence JSON / generated harness output confirms provider route facts,
      ABI order, pointer roles, segment2 route-family plan mirrors, lane-order
      contracts, and preservation contracts for both ops.
- [x] Existing computed-mask segment2 selected-boundary dry-run evidence does
      not regress.
- [x] Focused C++/lit/script checks covering the changed or revalidated module
      behavior pass, including direct route-entry fail-closed non-regression.
- [x] Bounded authority scan over touched production files and directly related
      tests/scripts finds no new central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.

## Out Of Scope

- Re-enabling direct pre-realized route-entry support for plain or computed-mask
  segment2.
- Computed-mask segment2 migration beyond non-regression checks.
- Compare/select, standalone reduction, strided load/store, widen conversion,
  MAcc, widening dot, high-level frontend/Linalg lowering, source-front-door
  positive routes, one-intrinsic wrapper dialects, broad selected-body framework
  rewrites, dashboards, report-only work, broad smoke matrices, or test
  renaming as the main milestone.
- Treating generated artifact names, script allowlists, route ids, exact
  intrinsic spellings, metadata, or common EmitC logic as support authority.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`

Relevant previous-task facts read:

- `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-plain-segment2-selected-realization-closure/prd.md`
- `.trellis/workspace/codex/journal-17.md`, Session 272

Initial files to inspect before deciding whether implementation is needed:

- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-segment2-*-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-segment2-*-fail-closed.test`

## Validation Plan

1. Inspect current selected-boundary, direct-route-entry, route/provider,
   target artifact, and generated-bundle surfaces for the two plain segment2
   bodies.
2. Run focused dry-run and direct fail-closed probes.
3. Run non-dry-run generated-bundle executions on `ssh rvv` for both ops with
   runtime counts `0,1,16,17,257`.
4. If a blocker appears, repair the narrow owner and rerun the failed focused
   checks.
5. Run computed-mask segment2 selected-boundary non-regression, focused C++/lit
   checks, authority scan, `git diff --check`, and `check-tianchenrv` when
   feasible.

## Completion Notes

- Current HEAD did not require production code repair. The selected-boundary
  executable artifact path already produced valid non-dry-run generated-bundle
  evidence when rerun from a clean single worker.
- Dry-run evidence for both plain segment2 ops was generated under
  `artifacts/tmp/stage2_plain_segment2_executable_closure/selected-boundary-dry-run`.
  Both ops reported `route_entry_realization: false`,
  `pre_realized_body_consumed: true`, selected-boundary materialization, and
  runtime counts `0,1,16,17,257`.
- Direct route-entry probes for both ops failed closed before bundle generation
  with the selected-boundary-only diagnostic.
- Non-dry-run generated-bundle evidence was generated under
  `artifacts/tmp/stage2_plain_segment2_executable_closure/ssh-rvv-executable`.
  Both ops compiled and ran on `ssh rvv`, with
  `remote_compile_succeeded: true`, `remote_run_succeeded: true`,
  `ssh_evidence: true`, and `PASS` output for counts `0,1,16,17,257`.
- Harness output covered field-order-distinguishing lanes and tail preservation
  for `segment2_deinterleave_unit_store` and
  `segment2_interleave_unit_load`.
- Computed-mask segment2 selected-boundary dry-run non-regression passed under
  `artifacts/tmp/stage2_plain_segment2_executable_closure/computed-mask-segment2-dry-run`.
- Focused checks passed:
  - `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
  - focused lit filter for the two pre-realized plain segment2 dry-run tests,
    the two direct fail-closed tests, and the two pre-realized selected-body
    artifact tests: 6/6 passed.
- A duplicated concurrent full-check attempt caused output-directory races
  (`File exists`, missing `materialized_selected_body.mlir`, and stale bundle
  overwrite diagnostics). After both duplicate jobs exited, generated lit
  `Output` directories were cleaned and a single `check-tianchenrv` run passed
  405/405.
- Bounded authority scan found no new production code changes and no new active
  authority leak. Hits were existing negative checks, rejection paths, or
  direct fail-closed tests.
- No `.trellis/spec/` update was needed: the relevant selected-boundary-only
  segment2, target artifact ABI, and mirror-authority contracts were already
  present and no durable implementation contract changed in this validation
  closure round.
