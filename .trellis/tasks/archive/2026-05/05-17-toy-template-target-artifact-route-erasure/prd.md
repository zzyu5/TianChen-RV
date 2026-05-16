# Toy template target artifact route erasure

## Goal

Delete the remaining Toy template target artifact/export route as current
compiler artifact authority. This is a Wrong Logic Deletion Campaign round:
remove the metadata/source target artifact route before any Toy/template
rebuild, replacement route, or new extension work.

## Why Now

The previous task erased Toy source-seed metadata authority, but the target
artifact layer still exposes a Toy metadata/source route. Current repository
evidence names `tcrv-toy-template-artifact` as a Toy template exporter route
that validates a metadata-diagnostic candidate, calls
`emitSelectedEmitCArtifactCppSource`, and prints
`materialized_emitc_cpp_source` as a Toy target artifact payload. Tests under
`test/Target/Toy` and `test/Target/TargetArtifactExportTest.cpp` still protect
that artifact shape.

Under the deletion campaign, this old metadata/source artifact authority must
be deleted rather than hidden, wrapped, or rebuilt in the same round.

## Current Repository Facts To Verify

- `include/TianChenRV/Target/Toy/ToyTargetSupportBundle.h` and
  `lib/Target/Toy/ToyTargetSupportBundle.cpp` are the expected Toy target
  support surfaces.
- `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` is the expected
  built-in exporter registration surface.
- `lib/Target/TargetArtifactExport.cpp` contains the generic target artifact
  export selection and validation behavior.
- `test/Target/Toy/` contains Toy target artifact route fixtures.
- `test/Target/TargetArtifactExportTest.cpp` contains C++ coverage for generic
  and built-in artifact exporter behavior.

## Requirements

- Remove Toy target artifact route registration from production/default
  built-in target exporter registration.
- Remove public translate route exposure for `tcrv-toy-template-artifact`.
- Remove Toy exporter helpers and route-shape code that produce or validate
  metadata/source target artifacts as supported production output.
- Remove or rewrite tests and fixtures that assert:
  - `tcrv-toy-template-artifact`;
  - Toy `metadata-diagnostic` artifact export;
  - Toy `materialized_emitc_cpp_source` positive output;
  - Toy metadata/source target artifact as a supported production route.
- Preserve generic registry behavior and fail-closed diagnostics that do not
  make Toy a built-in supported target artifact route.
- If a generic C++ registry/export test needs a dummy exporter, keep it
  strictly test-local and do not publish Toy as a supported built-in route.
- Keep RVV materialized EmitC object/header/bundle behavior intact except for
  mechanical compile fixes caused by deleting Toy route code.

## Acceptance Criteria

- [ ] Built-in target artifact exporter registration no longer publishes a Toy
  metadata/source target artifact route.
- [ ] `tcrv-translate --help` no longer advertises
  `--tcrv-toy-template-artifact`.
- [ ] Generic target artifact export does not select Toy metadata-diagnostic
  output as a production artifact.
- [ ] Toy target tests disappear or assert deleted/unknown/fail-closed
  behavior only.
- [ ] C++ target artifact tests no longer require Toy target exporter shape or
  count Toy among built-in production artifact routes.
- [ ] Focused scans over Toy target, built-in exporter registration, generic
  artifact export, translate route exposure, and tests show no active
  `tcrv-toy-template-artifact`, Toy metadata/source route, Toy
  `materialized_emitc_cpp_source` positive fixture, or Toy metadata-diagnostic
  positive artifact export.
- [ ] Focused build/tests pass, or remaining failures are reported as missing
  new-architecture rebuild gaps without restoring the deleted route.

## Out Of Scope

- No Toy rebuild through a new route.
- No replacement Toy source/front-door materializer.
- No compatibility wrapper, hidden route, quarantine mode, or legacy mode.
- No new plugin template features.
- No RVV expansion or unrelated RVV behavior changes.
- No descriptor-driven computation or metadata-as-route authority restoration.
- No compiler internals implemented in Python.
- No broad cleanup unrelated to Toy target artifact route deletion.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task context:
  `.trellis/tasks/archive/2026-05/05-17-toy-source-seed-authority-erasure/prd.md`.
- Deletion gaps in future Toy/template executable support should be reported as
  missing rebuild architecture, not patched by restoring Toy metadata/source
  artifact output.
