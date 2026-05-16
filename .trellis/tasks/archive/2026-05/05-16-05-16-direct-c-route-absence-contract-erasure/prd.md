# Direct-C route absence contract erasure

## Goal

Erase active spec and target-test contracts that preserve deleted direct-C
target artifact routes by name. Target artifact export coverage should describe
the current generic registry/export behavior and the future materialized EmitC
gap, not make old direct-C route absence a durable API, diagnostic, or test
identity.

## Background

The previous deletion rounds removed generated C skeleton authority, metadata
routes, source artifact kinds, and related direct-C target route surfaces. A
focused follow-up scan still found active spec/test language that keeps the
deleted route history alive as named behavior:

- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` still requires
  tests to assert absence of deleted direct-C route families.
- `.trellis/spec/testing/mlir-testing-contract.md` still names deleted direct-C
  exporter test ownership as active coverage.
- `test/Target/TargetArtifactExportTest.cpp` still reports failures through
  strings such as runtime-callable direct C route metadata, deleted direct C
  exporter bundles, direct C route manifests, and composite route absence after
  direct C deletion.

## Requirements

- Remove or rewrite active spec text that treats deleted direct-C route
  absence as a named target artifact/export contract.
- Rewrite target artifact export C++ test diagnostics and comments so they
  assert current generic behavior: empty built-in target artifact registries,
  no plugin-owned exporter bundle registration for families without current
  materialized artifact routes, generic target translate route registry
  validation, source-artifact fail-closed behavior, and missing materialized
  EmitC routes.
- Preserve generic registry validation that is not tied to deleted direct-C
  route history, including duplicate/invalid registration, malformed metadata,
  runtime ABI role validation, source-artifact rejection, and missing-route
  diagnostics.
- Do not add any replacement exporter, Common EmitC rebuild, source artifact
  enablement, runtime ABI implementation, direct-C compatibility path, route
  quarantine, broad registry redesign, or new extension behavior.
- If checks expose missing new-architecture gaps, report them as gaps rather
  than restoring deleted-route absence contracts.

## Acceptance Criteria

- No active spec or test treats deleted direct-C route absence as a named
  API, diagnostic, or route-family contract.
- No active target test fails specifically because a deleted direct-C route
  name appears or does not appear.
- Generic target route registry, duplicate/invalid registration,
  source-artifact rejection, and missing materialized EmitC route tests remain
  only when phrased around current generic behavior rather than old
  route-family history.
- Focused ref-scan for the Hermes brief terms passes outside archived tasks,
  workspace journals, temporary artifacts, build outputs, and `.git`.
- Focused target artifact export build/test passes, or any failure is reported
  as an expected missing-architecture gap without restoring old contracts.

## Out of Scope

- New target exporter or target translate route.
- Common EmitC rebuild or materialized source artifact enablement.
- Runtime ABI implementation or executable RVV evidence.
- Compatibility wrappers for deleted direct-C routes.
- Broad registry redesign.
- Deleting generic validation unrelated to deleted direct-C route history.

## Verification Plan

- Focused active-surface ref-scan for:
  `deleted direct C`, `runtime-callable direct C`, `direct C route metadata`,
  `direct C exporter bundles`, `direct C route manifests`,
  `after direct C deletion`, `deleted RVV/scalar/dispatch direct C`, and
  `direct descriptor-to-C`, excluding `.trellis/tasks/archive`,
  `.trellis/workspace`, `artifacts/tmp`, `build`, and `.git`.
- Build at least `tcrv-opt`, `tcrv-translate`, and
  `tianchenrv-target-artifact-export-test` if available.
- Run the target artifact export test binary or affected lit tests.
- Attempt `ninja -C build check-tianchenrv`.
- Run `git diff --check`.
- Run Trellis task validation, finish/archive the task if complete, and commit
  one coherent deletion-only change.
