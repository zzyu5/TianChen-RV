# RVV selected-boundary descriptor mirror quarantine

## Goal

Refactor the RVV plugin selected proposal, selected lowering-boundary, and
selected emission-planning boundary so the default production path is
authoritative from typed RVV family/body metadata, selected vector config, and
the common EmitC/source artifact route. Legacy `tcrv_rvv.lowering_descriptor`
metadata may remain only as explicit non-authoritative mirror validation after
typed authority is already known.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; task-start worktree was clean.
- Task-start HEAD is `7df47e6 feat(support): retire descriptor callable planning surface`.
- `.trellis/.current-task` was absent, so this task was created as
  `.trellis/tasks/05-12-rvv-selected-boundary-descriptor-mirror-quarantine/`.
- The archived task
  `.trellis/tasks/archive/2026-05/05-12-descriptor-mirror-retirement-from-production-surfaces/`
  is complete and must not be reopened.
- The prior completed module removed descriptor-shaped Support callable
  planning surfaces. This task continues at the adjacent RVV plugin
  selected-boundary surface.
- RVV descriptor strings may remain as finite target registration data or
  fail-closed legacy mirrors, but must not select compute, ABI, source route,
  intrinsic operation, component identity, or artifact source authority.

## Requirements

- Refactor default RVV proposal and selected-plan API names so production
  planning no longer exposes an `attachLoweringDescriptorAttr`-style switch or
  default `getLoweringDescriptor` metadata path.
- Ensure typed/default RVV proposals never attach
  `tcrv_rvv.lowering_descriptor` as default proposal metadata.
- Keep finite RVV family/body selection authoritative through typed family id,
  dtype, arithmetic operation, selected vector-shape metadata, selected
  runtime ABI contract, and microkernel body materialization.
- Ensure selected lowering-boundary materialization rejects descriptor-only RVV
  variants and materializes callable microkernel bodies only from typed
  selected family/body authority or existing selected EmitC attachments.
- If descriptor strings remain, expose them only through explicit legacy mirror
  or target-local registration helper names and validate them fail-closed after
  typed authority is known.
- Update positive/default tests to assert typed family/body/source-kind/
  selected-shape/runtime-ABI/EmitC authority and absence of
  `tcrv_rvv.lowering_descriptor`.
- Rename, update, or delete descriptor-positive tests that encode the old
  production authority. Remaining descriptor tests must be quarantine or
  mirror-validation tests.
- Update durable specs only if code behavior changes need a concise contract
  correction.

## Acceptance Criteria

- [x] Default RVV typed proposals do not attach or require
      `tcrv_rvv.lowering_descriptor`.
- [x] Production selected-boundary planning has no default descriptor getter or
      descriptor-attachment switch as part of the typed plan shape.
- [x] Descriptor-only RVV selected variants fail before supported callable
      boundary/emission/source artifact materialization.
- [x] Descriptor mirror validation remains explicit, legacy-named, and
      non-authoritative when kept.
- [x] Focused C++ and lit/FileCheck coverage proves typed family/body plus
      EmitC authority and descriptor absence for default positive paths.
- [x] Read-only ref-scan shows remaining `tcrv_rvv.lowering_descriptor`
      mentions are target registration data, explicit legacy mirror
      validation, stale-descriptor negative tests, or quarantine tests.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      focused touched-target checks, and `check-tianchenrv -j2` pass when the
      build tree is usable.
- [x] The task is finished/archived and one coherent commit records the module,
      or the task remains open with the exact residual descriptor surface.

## Non-Goals

- No new RVV dtype, arithmetic family, LMUL shape, benchmark, performance
  claim, or broad coverage matrix.
- No fresh `ssh rvv` run unless this task makes a new runtime, correctness, or
  performance claim.
- No Python compiler implementation and no descriptor-to-C exporter.
- No compute semantics in `tcrv.exec`.
- No offload, IME, AME, TensorExt, Template, Toy, scalar, or dispatch redesign
  beyond tiny coordinated updates required by touched RVV boundaries.
- No helper-only, report-only, grep-only, or smoke-only closeout.

## Technical Approach

Start from the selected RVV binary planning surfaces:
`RVVBinaryPlanning`, `RVVExtensionPlugin`,
`RVVBinarySelectedLoweringBoundary`, `RVVBinarySelectedEmissionPlanning`, and
the RVV microkernel materialization/export validation path. Identify every
production-facing descriptor-named hook and classify it as one of:

- finite target registration data that remains target-local;
- stale default production authority that must be removed or renamed;
- explicit legacy mirror validation after typed family/body authority;
- negative/quarantine test fixture.

Then rewire the default proposal and materialization flow so typed selected
family/body metadata drives the selected boundary and supported emission-plan
handoff. Descriptor mirrors should be checked only after the typed plan is
already available and should not be emitted by default proposals.

## Minimal Validation

- `git diff --check`
- `git diff --cached --check`
- Focused C++ build/tests as touched:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-target-artifact-export-test`, and selected-boundary/emission
  targets affected by CMake target names.
- Focused lit/FileCheck filters for affected files under `test/Plugin`,
  `test/Transforms/LoweringBoundary`, `test/Transforms/LinalgToExec`,
  `test/Target/RVVMicrokernel`, `test/Target/RVVScalarDispatch`, and
  `test/Target/TargetArtifactBundleExport`.
- Read-only ref-scan of remaining default RVV proposal/materialization
  descriptor mentions.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and the build tree is usable.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-rvv-selected-boundary-descriptor-mirror-quarantine`

## Definition Of Done

This task is complete only when the default RVV selected-boundary path is
typed-family/body plus selected-config plus EmitC/source-authority driven,
legacy descriptor references are explicit mirror/quarantine only, focused
checks pass, Trellis state is truthful, and one coherent commit records the
module. If unfinished, leave the task open and name the exact remaining
descriptor surface and next file/function.

## Completion Notes

- Removed `attachLoweringDescriptorAttr`, `shouldAttachLoweringDescriptorAttr`,
  and the plugin proposal/selected-plan `getLoweringDescriptor()` surfaces from
  `RVVBinaryPlanning`.
- Removed the RVV extension plugin default proposal branch that could attach
  `tcrv_rvv.lowering_descriptor` from proposal planning metadata.
- Renamed RVV target/selected-config descriptor access consumed by RVV plugin
  code to `getLegacyLoweringDescriptorMirror()` and updated mirror metadata
  wording to say legacy mirror explicitly.
- Updated RVV planning and target registry tests to assert typed family,
  selected source-kind, selected shape, and capability authority instead of a
  descriptor attach switch or default descriptor getter.
- Updated the RVV plugin spec to state that default typed RVV binary proposals
  must not attach `tcrv_rvv.lowering_descriptor`.
- No fresh `ssh rvv` execution was performed; this task makes no runtime,
  correctness, or performance claim.

## Ref-Scan Conclusion

Read-only scan after implementation found no remaining
`attachLoweringDescriptor*`, `shouldAttachLoweringDescriptor*`, or RVV plugin
planning `getLoweringDescriptor()` API. Remaining RVV descriptor references are
target registration fields/helpers, explicit legacy descriptor mirror
validation, stale-mirror negative tests, descriptor-only quarantine tests, or
offload/scalar surfaces outside this RVV module.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-binary-planning-test
  tianchenrv-target-artifact-export-test
  tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-binary-variant-legality-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv -sv .
  --filter
  'Plugin/RVV|Transforms/LoweringBoundary|Transforms/LinalgToExec|Target/RVVMicrokernel|Target/RVVScalarDispatch|Target/TargetArtifactBundleExport'`
  from `artifacts/tmp/tianchenrv-build/test`, 87/87 selected tests passed.
- Read-only ref-scan:
  `rg -l
  'tcrv_rvv\.lowering_descriptor|tcrv_rvv\.selected_lowering_descriptor|loweringDescriptor|getLegacyLoweringDescriptorMirror|lookupRVVBinaryFamilyRegistrationByLegacyLoweringDescriptor'
  ...`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.
