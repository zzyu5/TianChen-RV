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


## Session 79: RVV microkernel direct route fixture erasure

**Date**: 2026-05-15
**Task**: RVV Microkernel Direct Route Fixture Erasure
**Branch**: `main`

### Summary

Deleted active RVV/scalar/dispatch direct microkernel route-name fixtures
without adding any replacement emission path. The remaining route-id hits are
limited to the current Trellis PRD's deletion target list; active source, spec,
and test surfaces no longer invoke or look up the historical route ids.

### Main Changes

- Removed direct RVV microkernel translate-option invocation from
  `Target/ArtifactExport/target-source-artifact-routes.test`.
- Reworked `Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir` to test
  unsupported lowering-boundary/emission-plan behavior without old direct route
  options.
- Removed named absence lookups for historical RVV/scalar/dispatch direct route
  ids from `TargetArtifactExportTest.cpp`.
- Updated lowering/runtime and MLIR testing specs to describe deleted route
  families instead of preserving old route ids as active contracts.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(rvv): erase microkernel direct route fixtures |

### Testing

- [OK] `cmake --build build --target tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tcrv-translate --help-hidden` has no match for the four
  historical route ids.
- [OK] Focused lit filter for `Target/ArtifactExport/target-source-artifact-routes`
  and `Target/RVVMicrokernel/rvv-microkernel-pipeline`: 2 passed.
- [OK] Focused lit filter for `Target/ArtifactExport/target-artifact-export-registry`:
  1 passed.
- [OK] Focused active ref scan excluding `artifacts/tmp`, `.trellis/tasks`,
  `.trellis/workspace`, and `.git` found no hits for the four historical route
  ids.
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-rvv-microkernel-direct-route-fixture-erasure`
- [WARN] Full `cmake --build build --target check-tianchenrv -- -v` ran and
  failed in 12 existing RVV planning/lowering/script tests outside this task's
  changed files; this round did not expand scope to repair that unrelated
  baseline.

### Status

[OK] **Completed**

### Next Steps

- Continue deletion-only cleanup if active direct-C route fixtures remain; do
  not start RVV/Common EmitC rebuild from this deletion task.


## Session 80: Scalar deleted microkernel fixture erasure

**Date**: 2026-05-15
**Task**: Scalar Deleted Microkernel Fixture Erasure
**Branch**: `main`

### Summary

Deleted active scalar deleted-microkernel fixture authority without adding a
replacement scalar execution path. The scalar fallback surface remains
metadata-only, and active tests/spec/docs no longer preserve exact historical
finite-family scalar microkernel op names as route anchors.

### Main Changes

- Removed `test/Dialect/Scalar/deleted-microkernel.mlir`.
- Removed `hasDeletedScalarMicrokernelOp` and the C++ assertions that scanned
  live scalar plugin IR for deleted `_microkernel` op-name absence.
- Kept scalar plugin coverage on positive metadata-only behavior through
  selected `tcrv_scalar.lowering_boundary`, legality, cost, readiness, and
  emission-plan checks.
- Updated scalar fallback and emission-runtime specs to describe deleted
  finite-family scalar microkernel syntax generically instead of listing exact
  historical op spellings.
- Updated README scalar fallback wording so it no longer treats an explicit
  scalar microkernel body as current attachment authority.
- Confirmed no replacement op, alias, wrapper, compatibility path, direct C
  exporter, descriptor path, route id, runtime lowering, or rebuild
  implementation was added.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(scalar): erase deleted microkernel fixtures |

### Testing

- [OK] `ninja -C build tcrv-opt tianchenrv-scalar-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-scalar-extension-plugin-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Plugin/scalar-extension-plugin.test Dialect/Scalar` from `build/test`
- [OK] Focused active ref scan excluding `.trellis/tasks`, `.trellis/workspace`,
  `.git`, `build`, and artifacts returned no hits for the exact deleted scalar
  microkernel op names, `hasDeletedScalarMicrokernelOp`, or
  `deleted-microkernel.mlir`.
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-scalar-deleted-microkernel-fixture-erasure`
- [WARN] `ninja -C build check-tianchenrv -- -v` was an invalid invocation in
  this build because `-v` was parsed as a target.
- [WARN] `cmake --build build --target check-tianchenrv -- -v` ran and failed
  in the existing 12 RVV planning/lowering/script tests outside this task's
  changed scalar surface: `Plugin/plugin-emission-plan.test`,
  `Scripts/rvv-probe-to-mlir.test`,
  `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  `Transforms/EmissionReadiness/emission-readiness-rvv-builtin.mlir`,
  `Transforms/EmissionReadiness/emission-readiness.test`,
  `Transforms/EmissionReadiness/materialize-emission-plans-rvv-builtin.mlir`,
  `Transforms/ExecutionPlanCoherence/rvv-capacity-stale-boundary-fails.mlir`,
  `Transforms/LoweringBoundary/rvv-i32m1-policy-capability-fails.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-compat.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-malformed.mlir`,
  `Transforms/PluginVariantLegality/plugin-variant-legality-pass-invalid.mlir`,
  and `Transforms/PluginVariantLegality/plugin-variant-legality-pass.mlir`.

### Status

[OK] **Completed**

### Next Steps

- Continue deletion-only cleanup if another active deleted scalar/direct-C
  route fixture remains; otherwise the next phase should be an explicit rebuild
  task rather than restoring scalar microkernel syntax.


## Session 81: RVV deferred runtime ABI fixture erasure

**Date**: 2026-05-15
**Task**: RVV Deferred Runtime ABI Fixture Erasure
**Branch**: `main`

### Summary

Erased active stale RVV deferred runtime-ABI/glue fixture authority from
tests and specs without adding a replacement runtime ABI or executable RVV
route. Remaining RVV unsupported emission-plan coverage now asserts the
current generic unsupported/no-runtime metadata boundary.

### Main Changes

- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to document
  generic unsupported runtime metadata instead of the old deferred RVV ABI/glue
  strings.
- Rewrote EmissionReadiness lit/C++ expectations from old deferred ABI/glue
  anchors to `unsupported-plugin-runtime-abi`,
  `unsupported-emission-runtime-abi`, and `no-runtime-glue-unsupported`.
- Aligned the retained EmissionReadiness RVV fixtures with the current selected
  vector-shape contract so they test the current unsupported plan path.
- Updated target manifest and target bundle guard fixtures to stop carrying the
  old deferred runtime ABI/glue strings.
- Deleted `test/Transforms/ExecutionPlanCoherence/rvv-capacity-stale-boundary-fails.mlir`
  because, after current selected-shape metadata was applied, it no longer
  produced its stated negative failure and retaining it would require unrelated
  execution-plan-coherence rebuild work.
- Added no RVV runtime ABI, wrapper, alias, compatibility layer, descriptor
  route, direct C exporter, source/header/object route, or rebuild path.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | chore(rvv): erase deferred runtime abi fixtures |

### Testing

- [OK] `ninja -C build tcrv-opt tianchenrv-emission-readiness-test`
- [OK] `build/bin/tianchenrv-emission-readiness-test`
- [OK] `build/bin/tcrv-opt test/Transforms/EmissionReadiness/materialize-emission-plans-rvv-builtin.mlir --tcrv-materialize-emission-plans --split-input-file | /usr/lib/llvm-20/bin/FileCheck test/Transforms/EmissionReadiness/materialize-emission-plans-rvv-builtin.mlir`
- [OK] `/usr/lib/llvm-20/bin/not build/bin/tcrv-opt test/Transforms/EmissionReadiness/emission-readiness-rvv-builtin.mlir --tcrv-check-emission-paths 2>&1 | /usr/lib/llvm-20/bin/FileCheck test/Transforms/EmissionReadiness/emission-readiness-rvv-builtin.mlir`
- [OK] `build/bin/tianchenrv-emission-readiness-test | /usr/lib/llvm-20/bin/FileCheck test/Transforms/EmissionReadiness/emission-readiness.test`
- [OK] `build/bin/tcrv-translate --tcrv-export-emission-manifest test/Target/EmissionManifest/emission-manifest-selected.mlir | /usr/lib/llvm-20/bin/FileCheck test/Target/EmissionManifest/emission-manifest-selected.mlir`
- [OK] Target artifact bundle guard missing-dir, invalid-dir, and unsupported
  no-bundle direct command checks.
- [WARN] `python3 -m lit` is unavailable in this environment (`No module named
  lit`); direct test commands and the build's lit runner were used instead.
- [WARN] `ninja -C build check-tianchenrv` was attempted and failed in 8
  existing baseline tests outside this owner: `Plugin/plugin-emission-plan.test`,
  `Scripts/rvv-probe-to-mlir.test`,
  `Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`,
  `Transforms/LoweringBoundary/rvv-i32m1-policy-capability-fails.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-compat.mlir`,
  `Transforms/LoweringBoundary/rvv-lowering-boundary-malformed.mlir`,
  `Transforms/PluginVariantLegality/plugin-variant-legality-pass-invalid.mlir`,
  and `Transforms/PluginVariantLegality/plugin-variant-legality-pass.mlir`.

### Status

[OK] **Completed**

### Next Steps

- Continue deletion-only cleanup for remaining active wrong-logic fixtures, or
  open a separate rebuild task for the existing RVV selected-shape legality
  baseline failures. Do not restore deferred runtime ABI/glue metadata.
