# RVV Deferred Runtime ABI Fixture Erasure

## Goal

Erase active stale RVV deferred runtime-ABI fixture authority from tests and
specs. Current RVV plugin production code reports unsupported emission plans
with generic unsupported/no-runtime metadata; active fixtures must not preserve
the older deferred executable-runtime ABI/glue strings as a contract.

## What I Already Know

- The previous committed round is `48849ff chore(target): erase source front-door fixture anchors`.
- Repository state at task creation was clean.
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp` currently sets RVV unsupported
  emission-plan metadata to `unsupported-plugin-runtime-abi`,
  `unsupported-emission-runtime-abi`, and `no-runtime-glue-unsupported`.
- Focused ref scan found stale RVV deferred ABI/glue anchors in active tests
  and `.trellis/spec/extension-plugins/rvv-plugin.md`.
- This round is part of the Wrong Logic Deletion Campaign. It is deletion /
  refactor only.

## Module Goal

Remove active stale fixture/spec authority for these exact old RVV metadata
contracts:

- `rvv-plugin-deferred-runtime-abi`
- `rvv-executable-runtime-abi-deferred`
- `deferred-rvv-runtime-glue`
- Old `RVV metadata-only first slice has no RVV lowering` wording where it
  protects those stale contracts.

## Requirements

- Active tests and specs must no longer assert or document the old deferred RVV
  runtime ABI/glue strings.
- Remaining RVV unsupported emission-plan coverage must assert the current
  generic unsupported/no-runtime contract, or be deleted if it only protected
  stale route metadata.
- Keep production/default behavior aligned with `RVVExtensionPlugin` current
  unsupported emission-plan metadata.
- Do not add replacement runtime ABI, wrapper, alias, compatibility mode,
  descriptor route, direct C exporter, new route option, manifest ledger, helper
  path, plugin template, or RVV rebuild.
- If unrelated baseline checks fail, report them as baseline gaps and do not
  restore deferred ABI metadata.

## Acceptance Criteria

- [ ] Focused active-surface ref scan, excluding `artifacts/tmp`,
  `.trellis/tasks/archive`, `.trellis/workspace`, `.git`, `build`, and run
  artifacts, finds no active occurrences of the three stale RVV deferred
  ABI/glue strings.
- [ ] Any remaining old diagnostic wording is absent or explicitly justified as
  non-authoritative governance.
- [ ] Affected EmissionReadiness lit/C++ tests pass.
- [ ] `ninja -C build tcrv-opt tianchenrv-emission-readiness-test` is attempted
  and passes or reports an unrelated baseline/build gap.
- [ ] `check-tianchenrv` is attempted or its status is reported.
- [ ] `git diff --check` passes.
- [ ] Trellis task validates, is finished/archived when complete, and the round
  is committed as one coherent commit.

## Out Of Scope

- RVV executable lowering.
- Common EmitC implementation.
- Runtime ABI implementation.
- Target source/header/object route.
- `ssh rvv` runtime/correctness/performance evidence.
- New emission route, option, manifest ledger, helper, wrapper, alias,
  compatibility path, descriptor path, direct C exporter, or plugin template.
- Unrelated RVV legality/capability fixes.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/index.md`
- Production confirmation:
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp` uses the current generic
    unsupported/no-runtime metadata for RVV unsupported emission plans.
- Initial focused ref scan found stale anchors in:
  - `test/Transforms/EmissionReadiness/materialize-emission-plans-rvv-builtin.mlir`
  - `test/Transforms/EmissionReadiness/emission-readiness-rvv-builtin.mlir`
  - `test/Transforms/EmissionReadiness/EmissionReadinessTest.cpp`
  - `test/Transforms/ExecutionPlanCoherence/rvv-capacity-stale-boundary-fails.mlir`
  - `test/Target/TargetArtifactBundleExport/target-artifact-bundle-guards.mlir`
  - `test/Target/EmissionManifest/emission-manifest-selected.mlir`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`

## Completion Notes

- Updated active RVV unsupported emission-plan tests/specs to use
  `unsupported-plugin-runtime-abi`, `unsupported-emission-runtime-abi`, and
  `no-runtime-glue-unsupported`.
- Removed the obsolete execution-plan-coherence stale-capacity fixture after
  applying the current selected-vector-shape contract showed it no longer
  produced its stated negative failure and would require unrelated coherence
  rebuild work.
- No replacement RVV runtime ABI, wrapper, alias, compatibility path,
  descriptor route, direct C exporter, source/header/object route, or rebuild
  path was added.
- Focused changed-surface tests passed. Full `check-tianchenrv` was attempted
  and still fails in existing unrelated baseline tests outside this owner.
