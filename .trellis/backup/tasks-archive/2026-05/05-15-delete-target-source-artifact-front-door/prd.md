# Delete target source artifact front door

## Goal

Remove the public direct target source-artifact front door so target artifact
export fails closed against direct C semantic source routes. This is a
deletion-only Wrong Logic Deletion Campaign round: remove the old source-only
command/API selection authority and rewrite tests that currently protect it as
legal behavior.

## What I Already Know

* `tools/tcrv-translate/tcrv-translate.cpp` still registers
  `tcrv-export-target-source-artifact`.
* `include/TianChenRV/Target/TargetArtifactExport.h` still exposes
  `exportTargetSourceArtifact`.
* `lib/Target/TargetArtifactExport.cpp` still contains
  `ArtifactSelectionMode::SourceOnly` and source-only generic selection.
* Existing lit and C++ target artifact tests still exercise the generic source
  front door as a supported fail-closed route, even though this campaign now
  requires the command/API front door itself to disappear.
* The lowering/runtime spec states direct descriptor-to-C and direct C
  semantic exporters are deleted routes until a materialized EmitC route exists.

## Requirements

* Delete the `tcrv-export-target-source-artifact` translation registration.
* Delete the `exportTargetSourceArtifact` public Target API and implementation.
* Delete `ArtifactSelectionMode::SourceOnly` and source-only generic route
  selection authority.
* Preserve non-source generic artifact export and header export behavior unless
  directly required by source-front-door deletion.
* Rewrite focused tests so they prove the source front door is absent or no
  longer an API authority, rather than proving source export through the generic
  source command.
* Keep source artifact route strings only where they are historical negative
  evidence, bundle metadata, or non-front-door file naming semantics.

## Acceptance Criteria

* [x] `tcrv-export-target-source-artifact` is not a registered
  `tcrv-translate` command.
* [x] `exportTargetSourceArtifact` is absent from the public Target artifact
  export API and implementation.
* [x] `ArtifactSelectionMode::SourceOnly` is absent from the Target artifact
  exporter.
* [x] Tests under `test/Target/ArtifactExport/` and
  `test/Target/TargetArtifactExportTest.cpp` no longer protect direct C
  semantic source export as legal behavior.
* [x] Focused scans for `tcrv-export-target-source-artifact`,
  `exportTargetSourceArtifact`, and `ArtifactSelectionMode::SourceOnly` leave
  no active command/API/selection authority.
* [x] Focused build/lit or unit checks for affected translate/target artifact
  export behavior pass, or failures are recorded as expected deletion gaps
  without restoring the deleted route.

## Non-Goals

* Do not implement a replacement EmitC route.
* Do not add RVV lowering, scalar lowering, artifact formats, compatibility
  wrappers, quarantines, descriptor fallbacks, helper commands, or broad
  evidence matrices.
* Do not repair deleted direct-C routes by adding new source-export authority.

## Technical Notes

* Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
* Relevant code: `tools/tcrv-translate/tcrv-translate.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`.
* Relevant tests: `test/Target/ArtifactExport/`,
  `test/Target/TargetArtifactExportTest.cpp`, and any focused translate route
  absence coverage touched by this deletion.
