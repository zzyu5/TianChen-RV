# Delete plugin emission-plan source artifact authority

## Goal

Remove runtime-callable C source artifact authority from plugin-owned
`VariantEmissionPlan` construction and validation. This round is deletion-only:
the selected variant to plugin emission-plan boundary must fail closed when a
plugin returns `artifact_kind = "runtime-callable-c-source"` for a supported or
metadata-only plan. Future executable source output remains out of scope until
rebuilt through extension-family ops, a materialized MLIR EmitC module, and a
target-owned C/C++ route.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Worktree was clean before the task was created; HEAD was `58823c0
  chore(target): delete source bundle residue`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief.
* The previous archived work deleted target source bundle residue and left
  target artifact export fail-closed for source artifact kinds.
* Live inspection shows `lib/Plugin/ExtensionPlugin.cpp` still treats
  supported or metadata-only `runtime-callable-c-source` plans as valid when
  structured runtime ABI parameters exist.
* `lib/Target/TargetArtifactExport.cpp` already rejects deleted source artifact
  kinds in target exporter and bundle validation; this round must preserve that
  behavior.

## Requirements

* Supported plugin emission plans must not validate with
  `artifact_kind = "runtime-callable-c-source"`.
* Metadata-only plugin emission plans must not validate with
  `artifact_kind = "runtime-callable-c-source"`.
* Structured runtime ABI parameters must not make a runtime-callable C source
  plugin plan legal.
* Construction-protocol route metadata must not accept deleted direct C source
  artifact kinds as a valid plugin construction route.
* Any remaining source artifact string mention must be a deleted-route
  rejection diagnostic, spec wording for fail-closed behavior, or a test proving
  absence.
* Keep production/default behavior rewired to fail closed; do not add a
  replacement source route next to the old one.

## Acceptance Criteria

* [x] `VariantEmissionPlan` validation rejects supported
      `runtime-callable-c-source` plans with a deleted-route diagnostic.
* [x] `VariantEmissionPlan` validation rejects metadata-only
      `runtime-callable-c-source` plans with a deleted-route diagnostic.
* [x] A plugin-owned plan carrying runtime ABI parameters still fails closed
      when its artifact kind is `runtime-callable-c-source`.
* [x] Construction protocol manifest validation rejects
      `runtime-callable-c-source` and `standalone-c-source` route artifact
      kinds.
* [x] Focused plugin tests cover the deleted source artifact boundary.
* [x] Focused target artifact checks still pass, preserving the prior
      fail-closed target export behavior.
* [x] Focused ref-scans report no source artifact authority outside
      fail-closed diagnostics/spec/test absence checks.

## Out of Scope

* No replacement EmitC route.
* No new source exporter.
* No compatibility mode, wrapper, quarantine, or fallback route.
* No new RVV lowering, source artifact kind, descriptor fallback, or broad
  cleanup.
* No restoration of target source routes to make checks pass.

## Technical Notes

* Primary contract: `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
* Plugin validation surface: `include/TianChenRV/Plugin/ExtensionPlugin.h` and
  `lib/Plugin/ExtensionPlugin.cpp`.
* Construction-protocol route metadata surface:
  `include/TianChenRV/Plugin/ConstructionProtocol.h` and
  `lib/Plugin/Construction/ConstructionProtocol.cpp`.
* Focused tests live under `test/Plugin/`; target-export behavior is guarded by
  `test/Target/TargetArtifactExportTest.cpp` and target artifact lit tests.

## Definition of Done

* PRD and Trellis context files describe this deletion-only scope.
* Code and tests implement the fail-closed plugin emission-plan boundary.
* Focused build/tests and ref-scans are run and reported.
* Task is finished or archived per Trellis convention.
* One coherent commit is created if the task is complete.
