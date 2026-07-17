# RVV Smoke-Probe Route Fixture Erasure

## Goal

Delete the active RVV smoke-probe direct-C route fixture surface from tests and
specs. This round removes route-protecting references to the deleted
`tcrv-export-rvv-smoke-probe-c` command and the former RVV standalone source
route identity so future RVV executable work must return through explicit
extension-family ops, common EmitC materialization, MLIR C/C++ emission,
RVV intrinsic/runtime ABI validation, and real `ssh rvv` evidence.

## What I Already Know

- The previous committed round is `716ce68 chore(offload): erase descriptor route residue`.
- The current worktree started clean on `main`.
- The current campaign rule is deletion before rebuild.
- Active residue was reported in:
  - `test/Target/RVVSmokeProbe/rvv-smoke-probe-route-deleted.mlir`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Initial active ref scan confirmed route-name hits in the two tests and the
  three active specs when hidden `.trellis/spec` files are included.

## Requirements

- Remove active tests that invoke or assert lookup absence for
  `tcrv-export-rvv-smoke-probe-c` as a named fixture.
- Remove or rewrite active spec/testing text that preserves
  `rvv-smoke-probe-standalone-c-source` or RVV `standalone-c-source` as a
  named compiler route contract.
- Keep target route/exporter tests focused on the current allowed registry
  shape: no built-in target translate routes, no deleted direct-C artifact
  families, and only current supported metadata route registration.
- Preserve the deletion-campaign architecture boundary: no new RVV emission,
  no source route, no wrapper, no alias, no compatibility mode, and no
  replacement smoke-probe exporter.

## Acceptance Criteria

- [x] Active tests no longer invoke, look up, or FileCheck
  `tcrv-export-rvv-smoke-probe-c`.
- [x] Active specs no longer preserve `rvv-smoke-probe-standalone-c-source` or
  RVV `standalone-c-source` as named route fixtures.
- [x] Target artifact/translate registry coverage still proves the current
  allowed registry shape without the old RVV smoke-probe route-name fixture.
- [x] No new RVV emission route, source route, wrapper, alias, compatibility
  path, or probe-export replacement is added.
- [x] Focused active ref scan over active repo surfaces, excluding
  `artifacts/tmp`, `.trellis/tasks/archive`, and `.trellis/workspace`, leaves no
  route-authority hits for
  `tcrv-export-rvv-smoke-probe-c|rvv-smoke-probe-standalone-c-source|standalone-c-source`.
- [x] Focused build/test coverage passes for `tianchenrv-target-artifact-export-test`,
  `tcrv-translate`, and any affected lit tests.
- [x] `git diff --check` and Trellis task validation pass.

## Out Of Scope

- RVV rebuild.
- Common EmitC implementation.
- Executable plugin template.
- New RVV target artifact route.
- `ssh rvv` runtime evidence as the main result.
- Replacement smoke-probe exporter.
- Edits to `artifacts/tmp`, archived Trellis tasks, or supervisor prompt
  guardrails just to reduce grep counts.
- Converting the old smoke-probe route into legacy-mode or compatibility
  negative coverage.

## Technical Notes

- Specs read for this round:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Initial scan command:

```bash
rg -n --hidden "tcrv-export-rvv-smoke-probe-c|rvv-smoke-probe-standalone-c-source|standalone-c-source" .trellis/spec test lib include tools --glob '!artifacts/tmp/**' --glob '!.trellis/tasks/archive/**' --glob '!.trellis/workspace/**'
```

- Completion scan over `.trellis/spec test lib include tools CMakeLists.txt`
  returned no matches for the deleted route-name patterns.
- Full scan before archive returned matches only in this PRD, where the strings
  are deletion-campaign governance input rather than compiler route authority.
- Checks run:
  - `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-translate -j2`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `build/bin/tcrv-translate --help-hidden` with a no-match check for the old
    smoke-probe route option
  - `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport/target-artifact-export-registry.test Target/ArtifactExport/target-source-artifact-routes.test`
  - `git diff --check`
  - `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-rvv-smoke-probe-route-fixture-erasure`
