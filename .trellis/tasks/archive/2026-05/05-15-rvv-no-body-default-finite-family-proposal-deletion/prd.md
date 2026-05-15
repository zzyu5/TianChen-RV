# RVV no-body default finite-family proposal deletion

## Goal

Delete RVV no-body default finite-family proposal authority. RVV binary
planning must require explicit typed RVV extension-family body authority before
selecting a finite RVV binary family. A `tcrv.exec.kernel` with no typed RVV
body must fail closed during proposal planning instead of silently becoming
`i32-vadd` metadata/proposal authority.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `4c63d32`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-15-rvv-smoke-probe-descriptor-residue-deletion/prd.md`
  deleted `tcrv_rvv.smoke_probe_descriptor` and left this RVV no-body default
  finite-family authority as a bounded follow-up.
- `.trellis/spec/index.md` states descriptor-driven computation is invalid as
  long-term architecture and the compiler stack must remain
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks and independent backend dialect framing.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says direct variants without
  typed family/body authority must fail closed or remain unsupported metadata,
  but the same section still contains stale wording that the default automatic
  first-slice proposal requests `i32-vadd` body materialization.
- Active code scan found the production fallback in
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`:
  `defaultResolution.family = &target::rvv::getI32VAddFamilyRegistrationRecord();`.
- `test/Plugin/RVVBinaryPlanningTest.cpp` still protects
  `@default_i32_vadd_no_body` and the "default proposal" path.

## Requirements

- Remove the fallback that assigns `i32-vadd` or any other finite RVV binary
  family when no explicit typed RVV extension-family body is present.
- Make no-body RVV binary proposal planning fail closed with a bounded
  missing-typed-body diagnostic.
- Keep explicit typed RVV body authority as the only finite-family source for
  kernel-based RVV binary proposals.
- Keep direct typed RVV body planning behavior for existing explicit
  `tcrv_rvv.i32_*_microkernel` and `tcrv_rvv.i64_*_microkernel` tests.
- Delete or rewrite tests that protect no-body `default proposal`,
  `default_i32_vadd_no_body`, or descriptorless default RVV proposal behavior.
- Rewrite active spec comments/text that present no-body RVV default
  `i32-vadd` proposal materialization as current behavior.
- Record any missing executable route exposed by deletion as a rebuild gap
  instead of restoring the default finite-family selection.

## Acceptance Criteria

- [x] Active code/tests no longer contain a
      `defaultResolution.family = ...getI32VAddFamilyRegistrationRecord()`
      style no-body fallback path.
- [x] No no-body `tcrv.exec.kernel` can materialize an RVV proposal from finite
      family registry metadata.
- [x] `resolveRVVBinaryFamilyForProposal` reports a missing typed RVV body when
      no explicit typed body is present.
- [x] `buildRVVBinaryProposalPlan(capabilities, kernel, ...)` fails closed for
      no-body RVV kernels and still succeeds for explicit typed RVV body
      kernels.
- [x] Active tests/spec text no longer protect "default proposal" or
      `default_i32_vadd_no_body` finite-family behavior.
- [x] Explicit family-record unit paths that directly pass a family record to
      proposal planning remain intact; they are not no-body kernel authority.
- [x] No common EmitC lowering, new RVV ops, new artifact/export route,
      `ssh rvv` evidence, runtime ABI rebuild, helper wrapper, compatibility
      selector, negative-only quarantine, or new default family is added.
- [x] Focused ref-scans for `defaultResolution`, `default proposal`,
      `default_i32_vadd`, `default_i32_vadd_no_body`, no-body RVV family
      selection, and fallback-authority uses of
      `getI32VAddFamilyRegistrationRecord` are reported truthfully.
- [x] Focused RVV binary planning and RVV extension plugin tests pass, or
      deletion-exposed failures are recorded as rebuild gaps without restoring
      no-body default finite-family selection.
- [x] `git diff --check`, Trellis validation, finish/archive, final clean
      `git status --short`, and one coherent commit are produced if complete.

## Non-goals

- No common EmitC lowering implementation.
- No new RVV dialect ops.
- No new RVV artifact/export routes.
- No `ssh rvv` runtime, correctness, or performance evidence.
- No runtime ABI rebuild.
- No helper wrappers or compatibility selectors.
- No new default family.
- No preservation of the old no-body default path as negative-only quarantine.
- No scalar cleanup or direct-C deleted-route cleanup unless a file is directly
  coupled to this default RVV planning deletion.

## Minimal Evidence

- Focused ref-scan in active RVV plugin code, directly affected tests, and
  active specs for:
  `defaultResolution`, `default proposal`, `default_i32_vadd`,
  `default_i32_vadd_no_body`, `no-body`, `no body`, and fallback-authority uses
  of `getI32VAddFamilyRegistrationRecord`.
- Build/run focused C++ tests:
  `tianchenrv-rvv-binary-planning-test` and
  `tianchenrv-rvv-extension-plugin-test`.
- Run directly affected lit tests if code/test scans or focused failures
  identify any.
- Run `check-tianchenrv` if practical after focused validation.
- Run `git diff --check` and `git diff --cached --check`.
- Run `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- This task deletes kernel no-body fallback authority. It does not delete the
  finite RVV binary family registry itself, because explicit typed RVV bodies
  and direct family-record planning helpers still need those records.
- Direct calls to `buildRVVBinaryProposalPlan(capabilities, family, ...)` are
  outside the no-body kernel fallback path and may remain for focused unit
  coverage.
- `tcrv_frontend_lowering` is already deleted as finite-family authority and
  must remain fail-closed.
- Future executable RVV output remains a rebuild gap until explicit
  extension-family IR and a materialized MLIR EmitC route provide the authority.

## Completion Evidence

- Removed the production no-body fallback in
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`; `resolveRVVBinaryFamilyForProposal`
  now requires explicit typed RVV extension-family body authority before a
  finite RVV binary family can be selected.
- Rewrote `test/Plugin/RVVBinaryPlanningTest.cpp` so a no-body
  `tcrv.exec.kernel` fails closed in both family resolution and
  kernel-based proposal planning.
- Reworked `test/Plugin/RVVExtensionPluginTest.cpp` so no-body RVV profile
  capabilities produce a recoverable decline, while positive proposal tests
  use explicit typed `tcrv_rvv.i32_*_microkernel` body authority.
- Rewrote active selected-source negative fixtures in
  `test/Plugin/RVVBinarySelectedLoweringBoundaryTest.cpp`,
  `test/Plugin/RVVBinaryVariantLegalityTest.cpp`, and
  `.trellis/spec/extension-plugins/rvv-plugin.md` so they no longer preserve
  a default i32-vadd/no-body source-kind path as negative-only quarantine.
- Focused ref-scan over active RVV plugin code, active plugin tests, and the
  RVV plugin spec found no matches for `defaultResolution`,
  `default proposal`, `default_i32_vadd`, `default i32-vadd`,
  `default-i32-vadd`, `default no-body`, `build default i32-vadd`,
  `descriptorless proposal`, or `no-body.*i32-vadd`.
- Remaining `getI32VAddFamilyRegistrationRecord()` references in focused RVV
  tests are direct family-record unit coverage or explicit typed-body fixtures;
  none is a no-body fallback authority path.
- Focused checks passed:
  `cmake --build build --target tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-binary-planning-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-selected-lowering-boundary-test -j2`;
  `./build/bin/tianchenrv-rvv-binary-variant-legality-test`;
  `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`;
  `git diff --check`.
- `clang-format` was not available in this environment (`clang-format` and
  `/usr/bin/clang-format*` were absent), so no mechanical format pass was run.
- Full `cmake --build build --target check-tianchenrv -j2` was practical and
  was run. It failed with 104/114 passing and 10 failures that still expect
  no-body RVV first-slice proposal/materialization or old no-body RVV planning
  error text:
  `Scripts/rvv-probe-to-mlir.test`,
  `Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir`,
  `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  `Transforms/ExecutionPlanning/execution-planning-pipeline-no-viable.mlir`,
  `Transforms/ExecutionPlanning/execution-planning-pipeline-rerun-mismatch.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-builtin.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-rvv-missing-properties.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape-invalid.mlir`,
  `Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape.mlir`,
  and `Transforms/VariantSelection/variant-selection.test`.
- Rebuild gap exposed: broader execution-planning/materialization lit paths and
  selection tests still assume RVV capability metadata alone can synthesize
  `rvv_first_slice`. Future rebuild must provide typed RVV body authority or
  rewrite those broad tests to expect scalar fallback/no viable RVV proposal.
  This round did not restore the deleted default finite-family selection.
- No compatibility layer, helper wrapper, new default family, new EmitC route,
  new RVV op, runtime ABI rebuild, artifact export route, or `ssh rvv`
  evidence path was added.
