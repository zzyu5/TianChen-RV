# Stage2 RVV segment2 generated-artifact executable boundary

## Goal

Upgrade the five provider-validated RVV segment2 route forms from
route-supported compiler/provider evidence to executable generated-artifact
evidence, or repair the generated-bundle / target artifact / runtime ABI path
where current evidence proves it is incomplete.

The covered forms are:

- `segment2_deinterleave_unit_store`
- `segment2_interleave_unit_load`
- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`
- `computed_masked_segment2_update_unit_load`

## Direction Source

- Direction title: `Switch: Stage2 RVV segment2 generated-artifact executable boundary`.
- Module owner: RVV target/generated-bundle artifact and runtime ABI path for
  the five provider-validated segment2 forms.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `7ac67a18 rvv: close segment2 route provider boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- Commit `7ac67a18` completed the RVV-owned segment2 route-construction
  provider preflight before `TCRVEmitCLowerableRoute` creation for all five
  segment2 forms. That task explicitly made only a compiler/provider-boundary
  claim and did not run `ssh rvv` evidence.
- The relevant long-term specs require the current RVV path to remain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized `tcrv_rvv` body ->
  RVV selected-body realization/provider facts -> `TCRVEmitCLowerableRoute` ->
  neutral common EmitC -> target artifact -> real `ssh rvv` evidence for
  runtime/correctness claims.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. They do not define segment2 compute, dtype, segment count,
  field binding, mask facts, tail policy, route support, or artifact support.
- Target artifact export for segment2 must rebuild the provider route from the
  selected typed RVV body, consume the rebuilt `TCRVEmitCLowerableRoute` and
  provider description as authority, and treat metadata as mirrors only.
- Common EmitC/export must remain neutral. It may materialize provider-built
  route payloads but must not infer RVV semantics from route ids, artifact
  names, ABI strings, manifests, descriptors, exact intrinsic spellings,
  script options, source-front-door metadata, or status fields.

## Requirements

1. Validate the current generated-bundle and target artifact runtime path for
   all five segment2 forms through the selected lowering-boundary/provider
   path with direct pre-realized route-entry support disabled/fail-closed.
2. If a generated-bundle, target artifact, runtime ABI, harness, or directly
   blocked RVV route/artifact production boundary is incomplete, repair that
   production boundary. Do not add unrelated evidence-only churn or broaden
   route coverage.
3. Preserve field0/field1 source and destination binding, segment count `2`,
   unit-stride memory roles, computed-mask producer and mask type mapping,
   inactive/pass-through policy, runtime `n`/AVL/VL control, ABI order,
   provider/header/type/intrinsic mirrors, and direct-route-entry retirement.
4. Ensure generated artifacts and harnesses expose truthful ABI order matching
   the selected `tcrv.exec` runtime boundary and provider-built route facts.
5. Fail closed with targeted diagnostics for unsupported or inconsistent
   artifact inputs, missing provider-preflight facts, stale mirrors, wrong
   field binding, wrong segment count, wrong computed-mask binding, missing
   runtime control, ABI mismatch, descriptor-derived semantics, script-derived
   compute, artifact-name authority, route-id authority, exact-intrinsic
   authority, direct-route-entry-only authority, or common-EmitC semantic
   choice.
6. Do not reintroduce `supports_direct_pre_realized_route_entry` as executable
   support. Direct pre-realized route-entry requests for the segment2 forms
   remain retired/fail-closed.

## Acceptance Criteria

- [x] PRD and task context truthfully represent the Hermes Direction Brief,
      current repo state, previous archived segment2 provider-boundary task,
      and relevant Trellis specs.
- [x] Focused dry-run generated-bundle evidence passes for all five segment2
      forms through the selected lowering-boundary/provider path with direct
      route-entry disabled.
- [x] Generated RVV C artifacts and harnesses show ABI order matching the
      selected runtime boundary and provider-built route facts for all five
      forms.
- [x] Real `ssh rvv` compile/run correctness passes for representative counts
      including `0`, `1`, an exact VL-sized case, a tail case, and a stress
      case.
- [x] Correctness checks cover deinterleave field outputs, interleave layout,
      computed-mask load/store/update, inactive/pass-through preservation, and
      tail preservation.
- [x] Direct pre-realized route-entry support remains fail-closed for segment2
      selected bodies.
- [x] Non-regression for the route-provider preflight tests added in
      `7ac67a18` passes.
- [x] If production code changes are required, focused C++/lit tests cover the
      changed artifact/runtime boundary and `git diff --check` passes. No
      production code change was required because the current selected-boundary
      generated-bundle and `ssh rvv` evidence passed.
- [x] Bounded authority scan over touched production/test files finds no new
      central ad hoc, name-derived, metadata-derived, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [x] `check-tianchenrv` passes, or the exact blocker is recorded.
- [x] Task status, context, and journal are truthful; if acceptance passes, the
      task is finished/archived and one coherent commit is created.

## Evidence Summary

- Current HEAD required no production code repair: the five form
  selected-boundary path already rebuilt provider routes, generated artifacts,
  produced ABI-matching harnesses, and passed hardware correctness.
- Dry-run artifact root:
  `artifacts/tmp/stage2_segment2_generated_artifact_executable_boundary/selected-boundary-dry-run`.
- `ssh rvv` artifact root:
  `artifacts/tmp/stage2_segment2_generated_artifact_executable_boundary/selected-boundary-ssh-rvv`.
- Covered runtime counts: `0`, `1`, `16`, `23`, and `257`.
- Direct route-entry fail-closed artifact root:
  `artifacts/tmp/stage2_segment2_generated_artifact_executable_boundary/direct-route-entry-fail-closed`.
- No `.trellis/spec/` update was needed because the segment2 target export,
  provider-route rebuild, direct-route-entry retirement, and generated-bundle
  evidence contracts were already represented in the existing specs.

## Validation Plan

1. Validate task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-31-stage2-rvv-segment2-generated-artifact-executable-boundary`.
2. Run script health checks if `scripts/rvv_generated_bundle_abi_e2e.py`
   changes or evidence is claimed:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
3. Run one selected-boundary dry-run covering all five segment2 forms with
   runtime counts `0`, `1`, `16`, `23`, and `257`.
4. Run the same five-form generated-bundle execution on real `ssh rvv` with
   the same counts.
5. Run direct route-entry fail-closed tests for segment2 selected bodies.
6. Run focused provider-boundary non-regression:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
7. Run directly related segment2 generated-bundle lit tests.
8. Run bounded authority scans over changed files and, if no production files
   changed, over the relevant segment2 artifact/runtime owner files.
9. Run `git diff --check`.
10. Run `cmake --build build --target check-tianchenrv -j2`, or record the
    exact blocker.

## Out of Scope

- New segment counts, segment3/segmentN, indexed segment memory, non-segment
  computed-mask cleanup, broad memory-family expansion, high-level
  Linalg/Vector frontend work, one-intrinsic wrapper dialects, route-provider
  redesign beyond blockers found in this artifact path, dashboards/report
  work, broad smoke matrices, or proof-only tests for already completed
  non-segment owners.
- Descriptor-driven computation, source-front-door positive RVV routes,
  route-id/artifact-name/script-derived route support, or common EmitC semantic
  selection.
- Any runtime/correctness claim without real `ssh rvv` evidence.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/guides/index.md`.
- Previous archived task read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-segment2-route-family-provider-boundary-closure/`.
- Directly relevant implementation/evidence surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV`, and `test/Scripts`.
