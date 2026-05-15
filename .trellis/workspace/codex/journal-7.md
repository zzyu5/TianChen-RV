# Journal - codex (Part 7)

> Continuation from `journal-6.md` (archived at ~2000 lines)
> Started: 2026-05-15

---



## Session 76: Delete support-layer I32 RVV runtime ABI residue

**Date**: 2026-05-15
**Task**: Delete support-layer I32 RVV runtime ABI residue
**Branch**: `main`

### Summary

Deleted Support-owned I32/RVV runtime ABI contracts and helpers; rewired tests/specs to explicit finite-binary ABI primitives and explicit dispatch guard names; focused build/tests/ref-scans passed.

### Main Changes

- Removed `I32BinaryRuntimeABIContract`, `getI32BinaryRuntimeABIContract`, and
  I32 helper wrappers/defaults from Support headers and implementation.
- Removed Support-owned RVV/scalar dispatch runtime C ABI identity strings and
  direct-route labels from Support/Target tests.
- Kept generic finite-binary ABI primitives: explicit caller-owned contracts,
  role binding, mem-window/runtime-param mirror validation, and invocation
  contract formatting.
- Switched dispatch guard materialization to an explicit generic
  `dispatch_available` C name instead of the Support default `rvv_available`.
- Updated lowering/runtime spec to make plugin/target-owned ABI construction
  the only future source of runtime ABI identity.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(support): erase i32 rvv runtime abi residue |

### Testing

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/DispatchRuntimeGuard Target/RVVScalarDispatch` from `artifacts/tmp/tianchenrv-build/test`
- [OK] Focused ref-scans over Support and directly affected Target tests for deleted I32/RVV ABI authority terms
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-delete-support-i32-rvv-runtime-abi`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 78: Direct-C source artifact kind erasure

**Date**: 2026-05-15
**Task**: Direct-C Source Artifact Kind Erasure
**Branch**: `main`

### Summary

Deleted the remaining active `runtime-callable-c-source` /
`standalone-c-source` route-kind authority from plugin construction,
plugin emission-plan validation, target artifact export, and directly
protecting tests. Validation now fail-closes on generic source artifact-kind
shape without preserving the old direct-C route IDs as code branches or
fixtures.

### Main Changes

- Removed old-name-specific source artifact-kind constants and predicates from
  `lib/Plugin/Construction`, `lib/Plugin`, and `lib/Target`.
- Replaced old route-kind rejection with generic source-token artifact-kind
  fail-closed validation.
- Rewrote focused construction, emission-plan, target export, and lit tests to
  use generic source artifact fail-closed probes instead of deleted direct-C
  route names.
- Kept current metadata/header/object route behavior intact; no compatibility
  layer or new source route was added.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(target): erase direct-c source artifact kinds |

### Testing

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-construction-protocol-common-test tianchenrv-plugin-emission-plan-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-plugin-emission-plan-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport/target-source-artifact-routes.test` from `artifacts/tmp/tianchenrv-build/test`
- [OK] `rg -n "runtime-callable-c-source|standalone-c-source|deleted source artifact kind" lib/Plugin lib/Target test/Plugin test/Target` returned no matches
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-05-15-direct-c-source-artifact-kind-erasure`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 79: Offload descriptor route residue erasure

**Date**: 2026-05-15
**Task**: Offload Descriptor Route Residue Erasure
**Branch**: `main`

### Summary

Deleted the remaining active Offload descriptor-route identity from the
unsupported emission-plan diagnostic and directly protecting tests. Offload
still fails closed as metadata-only with no executable lowering or target
artifact route, but active code/tests no longer preserve the old descriptor
artifact export wording or `tcrv-export-offload-runtime-descriptor` route id.

### Main Changes

- Replaced the Offload unsupported emission-plan message with descriptor-free
  no-executable-route wording.
- Updated Offload plugin, execution-planning, and emission-manifest tests to
  assert the current unsupported message.
- Replaced the deleted Offload descriptor route-id fixture in the target
  artifact export test with a route-id-free assertion that Offload contributes
  no target artifact exporters.
- Kept unrelated plugin-owned Toy metadata route registration coverage intact.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(offload): erase descriptor route residue |

### Testing

- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-offload-descriptor-route-residue-erasure`
- [OK] `cmake --build build --target tianchenrv-offload-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-offload-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/ExecutionPlanning/execution-planning-pipeline-offload.mlir Target/EmissionManifest/emission-manifest-offload-pipeline.mlir` from `build/test`
- [OK] Focused active ref-scans for deleted Offload descriptor wording and route id returned no matches in `lib`, `test`, and active specs.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 77: Delete common EmitC source-authority exporter residue

**Date**: 2026-05-15
**Task**: Delete common EmitC source-authority exporter residue
**Branch**: `main`

### Summary

Deleted common Conversion/EmitC C++ source-authority APIs and tests; kept only generic in-memory EmitC materialization; focused build/test/ref-scan passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 78: RVV smoke-probe route fixture erasure

**Date**: 2026-05-15
**Task**: RVV Smoke-Probe Route Fixture Erasure
**Branch**: `main`

### Summary

Deleted the active RVV smoke-probe direct-C route fixture surface without adding
a replacement RVV emission path. The old standalone smoke-probe CLI/route name
is no longer invoked or looked up by active tests, and active specs now describe
the deletion boundary without preserving the old route id or standalone source
artifact kind as a fixture.

### Main Changes

- Removed `test/Target/RVVSmokeProbe/rvv-smoke-probe-route-deleted.mlir`.
- Reworked `TargetArtifactExportTest.cpp` registry coverage to assert current
  route registry shape without a smoke-probe route-name lookup.
- Updated emission/runtime, RVV plugin, and MLIR testing specs to stop naming
  the old smoke-probe route id or RVV standalone source route fixture.
- Confirmed no new RVV emission, source route, wrapper, alias, compatibility
  path, or probe-export replacement was added.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(rvv): erase smoke-probe route fixtures |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tcrv-translate --help-hidden` with no match for the deleted
  smoke-probe route option
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport/target-artifact-export-registry.test Target/ArtifactExport/target-source-artifact-routes.test` from `build/test`
- [OK] Focused active ref scan over `.trellis/spec`, `test`, `lib`, `include`,
  `tools`, and `CMakeLists.txt` returned no deleted smoke-probe route-name or
  standalone source route fixture hits.
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-rvv-smoke-probe-route-fixture-erasure`

### Status

[OK] **Completed**

### Next Steps

- Continue deletion-only cleanup only if another active direct-C/descriptor
  route fixture remains; do not start RVV rebuild from this task.
