# Metadata diagnostic artifact-kind authority erasure

## Goal

Delete `metadata-diagnostic` as current supported construction/emission artifact
authority. This is a Wrong Logic Deletion Campaign round: remove metadata-only
artifact-kind authority from common plugin validation, Toy/TensorExtLite
construction manifests, plugin emission-plan surfaces, related tests, and any
spec wording that still protects it as an active artifact route before any
rebuild or replacement route.

## Why Now

Commit `880764f` removed the Toy template target artifact exporter and archived
that task with a clean worktree, but live inspection still shows
`metadata-diagnostic` as supported artifact authority in common
construction/emission code and in Toy/TensorExtLite construction protocol
surfaces. Those metadata-only routes are old-logic residue: they describe
diagnostic metadata as if it were a current construction or emission artifact
kind.

The current supported target artifact authority must be limited to rebuilt
materialized routes such as the RVV object/header bundle path. Unsupported
diagnostic metadata may still exist only as unsupported diagnostics, not as a
supported artifact kind or production route.

## Requirements

- Remove `metadata-diagnostic` from active common validators for supported
  construction artifact kinds and supported emission artifact kinds.
- Remove Toy construction protocol route manifest fields, plugin emission-plan
  outputs, and tests that publish or verify `metadata-diagnostic` as an active
  artifact kind.
- Remove TensorExtLite construction protocol route manifest fields, plugin
  emission-plan outputs, and tests that publish or verify `metadata-diagnostic`
  as an active artifact kind.
- Remove or rewrite tests that assert metadata-only artifact candidates,
  metadata-diagnostic target-route support, or manifest equality against
  `metadata-diagnostic` as production authority.
- Update directly related spec wording if it still presents
  `metadata-diagnostic` as a supported compiler emission artifact kind.
- Preserve real materialized EmitC in-memory route materialization and the
  existing RVV materialized EmitC object/header/bundle route where they do not
  depend on `metadata-diagnostic` artifact authority.
- Preserve generic fail-closed behavior and diagnostics for unsupported
  emission/construction routes.

## Acceptance Criteria

- [x] `metadata-diagnostic` is no longer accepted by active common construction
  artifact-kind validators.
- [x] `metadata-diagnostic` is no longer accepted by active common emission
  artifact-kind validators for supported plans.
- [x] Toy no longer publishes or verifies `metadata-diagnostic` as an active
  construction route or emission-plan artifact kind.
- [x] TensorExtLite no longer publishes or verifies `metadata-diagnostic` as an
  active construction route or emission-plan artifact kind.
- [x] Tests no longer protect metadata-only artifact candidates,
  metadata-diagnostic target-route support, or manifest equality against
  `metadata-diagnostic` as production authority.
- [x] Unsupported diagnostics, if present, remain unsupported diagnostics only
  and do not become supported artifact authority.
- [x] Focused scans over common plugin construction/emission code,
  Toy/TensorExtLite plugin code, related tests, and target artifact export tests
  show no active supported `metadata-diagnostic` artifact kind.
- [x] Focused C++ tests for touched common construction protocol, Toy plugin,
  TensorExtLite plugin, and target artifact export surfaces pass, or any
  remaining failures are reported as missing rebuild gaps without restoring the
  metadata artifact path.

## Out Of Scope

- No replacement Toy/TensorExtLite exporter.
- No new target artifact route.
- No new common lower-to-EmitC pass.
- No plugin template, RVV family expansion, compatibility wrapper, legacy mode,
  descriptor adapter, direct C semantic exporter, source-export route, Python
  compiler-core behavior, or core extension-specific branch.
- No deletion of the rebuilt RVV materialized EmitC object/header/bundle path
  unless a direct dependency on `metadata-diagnostic` artifact authority is
  discovered.
- No broad test matrix or unrelated cleanup beyond the artifact-kind authority
  deletion.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`.
- Previous completed task:
  `.trellis/tasks/archive/2026-05/05-17-toy-template-target-artifact-route-erasure/prd.md`.
- Direction Brief read set includes:
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  Toy/TensorExtLite construction protocol and plugin files, and focused C++
  tests under `test/Plugin` and `test/Target`.
- If deletion exposes build/test failures, treat those as missing
  new-architecture gaps. Do not restore the metadata-only artifact path to make
  checks pass.

## Verification

- Built focused C++ targets:
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-plugin-emission-plan-test`,
  `tianchenrv-toy-extension-plugin-test`,
  `tianchenrv-tensorext-lite-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Ran focused C++ tests:
  `./build/bin/tianchenrv-construction-protocol-common-test`,
  `./build/bin/tianchenrv-plugin-emission-plan-test`,
  `./build/bin/tianchenrv-toy-extension-plugin-test`,
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`, and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Ran focused lit:
  `/usr/bin/python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'tensorext-lite-(first-slice-materialization|target-artifact-unsupported)'`
  from `build/test`.
- Ran `git diff --check`.
- Ran `cmake --build build --target check-tianchenrv`; all 99 lit tests passed.
- Scanned production plugin/target code with
  `rg -n "metadata-diagnostic" include/TianChenRV/Plugin lib/Plugin include/TianChenRV/Target lib/Target`;
  no matches.
- Scanned tests with
  `rg -n "metadata-diagnostic" test/Plugin test/Target test/Conversion`;
  remaining matches are negative rejection coverage only.
