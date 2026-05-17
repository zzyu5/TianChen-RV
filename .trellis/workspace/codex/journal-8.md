# Journal - codex (Part 8)

> Continuation from `journal-7.md` (archived at ~2000 lines)
> Started: 2026-05-16

---



## Session 93: RVV construction-backed hardware artifact proof

**Date**: 2026-05-16
**Task**: RVV construction-backed hardware artifact proof
**Branch**: `main`

### Summary

Refreshed current-HEAD proof that RVV i32m1 add/sub/mul selected paths cross construction-checked EmitC, target object/header export, bundle metadata, and real ssh rvv correctness evidence without requiring source rewiring.

### Main Changes

- Created and archived Trellis task `05-16-rvv-construction-backed-hardware-artifact-proof` from the Hermes direction brief.
- Wrote the PRD around current-HEAD proof of the existing RVV i32m1 add/sub/mul route instead of expanding RVV coverage.
- Confirmed no source-code repair was required: RVV construction mapping, route provider validation, selected `tcrv_rvv.with_vl`, common selected EmitC artifact materialization, object/header export, and translate routes are already wired.
- Generated current add/sub/mul default and exact object/header artifacts plus bundle indexes under `artifacts/tmp/rvv_construction_backed_hardware_artifact_proof/20260516T125646Z`.
- Bundle metadata records route ids, callable component groups, `plugin-owned-runtime-abi`, and ordered `lhs`, `rhs`, `out`, `n` target-export ABI parameters.
- Linked and ran the generated add/sub/mul objects on `ssh rvv` through `rvv_i32m1_arithmetic_harness.c`; remote output was `tcrv_rvv_i32m1_arithmetic_current_head status=PASS n=4 add=[12,6,16,12] sub=[2,-12,24,0] mul=[35,-27,-80,36]`.
- Focused build, RVV plugin/unit tests, target artifact export test, construction/dialect tests, focused RVV lit set, full `check-tianchenrv`, `git diff --check`, and legacy/common-core scans passed.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | task(rvv): record construction-backed hardware proof |

### Testing

- [OK] Focused build for RVV construction protocol, RVV EmitC route provider,
  RVV plugin, RVV target support, target artifact export, `tcrv-opt`, and
  `tcrv-translate`.
- [OK] RVV plugin/unit tests, target artifact export test, construction
  protocol test, and RVV dialect test.
- [OK] Focused lit filter for RVV EmitC materialization, Target/RVV i32m1, and
  selected with_vl boundary tests: 17/17 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 passed.
- [OK] `ssh rvv` add/sub/mul external ABI harness passed.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 100: Toy source-front-door construction template proof

**Date**: 2026-05-17
**Task**: Toy source-front-door construction template proof
**Branch**: `main`

### Summary

Added a Toy-owned bounded source front door proving the common
`SourceFrontDoorPassRegistration` and
`--tcrv-source-artifact-front-door-pipeline` path is reusable by a non-RVV
extension family without common/core Toy or RVV semantic branches.

### Main Changes

- Added `--tcrv-toy-materialize-template-source-front-door` and registered it
  through the Toy plugin source-front-door hook.
- Materialized the Toy source marker into `tcrv.exec.kernel`,
  `toy.template` capability, `origin = "toy-plugin"` selected variant,
  `tcrv_toy.compute_skeleton`, selected diagnostic, Toy runtime ABI metadata,
  and the existing Toy EmitC/header artifact route.
- Added positive and negative lit coverage for Toy source front-door behavior,
  disabled built-ins, stale Toy seed metadata, stale selected-boundary residue,
  and Toy source-to-header export.
- Updated plugin-protocol spec with the durable Toy construction-template
  source-front-door contract.

### Testing

- [OK] `git diff --check`
- [OK] `./build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-plugin-registry-test`
- [OK] focused lit filter for Toy source-front-door, Toy target header,
  SourceFrontDoor negatives, and RVV source-front-door regression: 8/8 passed.
- [OK] `ninja -C build check-tianchenrv`: 105/105 tests passed.
- [OK] Targeted scans over changed Toy/common/tool surfaces showed no
  `SourceSeed`, source-seed public API, descriptor authority, source-export,
  direct-C route, or core Toy/RVV branch residue; remaining old seed hits are
  deleted-option negative tests only.

### Status

[OK] Completed; ready to archive and commit.


## Session 103: Bounded RVV vector-source front door promotion

**Date**: 2026-05-17
**Task**: Bounded RVV vector-source front door promotion
**Branch**: `main`

### Summary

Promoted the bounded RVV i32 add source path from source-seed naming into a
production-scoped vector-source front door that materializes the selected RVV
boundary, reaches the supported emission plan, and feeds the generated
object/header/bundle artifact route.

### Main Changes

- Renamed the common plugin source entry registration from
  `SourceSeedPassRegistration` to `SourceFrontDoorPassRegistration` and changed
  the public source artifact pipeline to
  `--tcrv-source-artifact-front-door-pipeline`.
- Renamed the RVV materializer to `RVVVectorSourceFrontDoor` with public pass
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door`.
- Moved positive RVV transform and target artifact lit coverage to
  `vector-source` test names and updated expected selected symbols to
  `@vector_source_*` / `tcrv_emitc_vector_source_*`.
- Added old `source-seed` pass/pipeline deletion checks so seed naming remains
  negative coverage rather than public authority.
- Updated plugin-protocol, variant-pipeline, and EmitC route specs to record
  the source front-door registry and current vector-source C ABI symbol.

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test tianchenrv-toy-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-plugin-registry-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] focused lit filter for RVV vector-source front door, source front-door
  negatives, and RVV vector-source target artifacts: 7/7 passed.
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 lit tests
  passed.
- [OK] refreshed `ssh rvv` C-ABI link/run proof for the renamed vector-source
  symbol under
  `artifacts/tmp/rvv_vector_source_front_door_link_run_abi_proof/20260517T022309Z/`.
  Remote output:
  `PASS tcrv_rvv_i32m1_vector_source_front_door_c_abi n=257`.
- [OK] targeted scans showed descriptor/direct-C/source-export residue only in
  `CHECK-NOT` assertions, and old source-seed names only in deleted-option
  negative tests.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-05-17-rvv-vector-source-front-door-promotion`

### Status

[OK] Completed, archived, and committed.


## Session 101: Common materialized EmitC header artifact foundation

**Date**: 2026-05-17
**Task**: Common materialized EmitC header artifact foundation
**Branch**: `main`

### Summary

Extracted a common target-layer materialized EmitC header artifact foundation
and rewired Toy plus TensorExtLite target-support exporters to use it as their
production declaration-only header path.

### Main Changes

- Added `MaterializedEmitCHeaderArtifactConfig`,
  `MaterializedEmitCHeaderArtifactMetadataEvidence`,
  `validateMaterializedEmitCHeaderArtifactCandidate`, and
  `exportMaterializedEmitCHeaderArtifact` in the common target artifact layer.
- Common validation now checks configured route/origin/artifact/emission/ABI
  fields, selected variant identity, ordered runtime ABI parameters, required
  artifact metadata, forbidden descriptor/direct-C/source-export/compute-body
  metadata, exactly one materialized `emitc.func`, and function arity.
- Toy and TensorExtLite target support now provide only plugin-local config and
  metadata evidence requirements before delegating to the common header helper.
- TensorExtLite emission plans now carry source-op sequence, source-role
  sequence, source op-interface, construction protocol, semantic role graph,
  and typed role realization evidence for the target header artifact.
- Updated lowering-runtime spec with the common header artifact foundation
  contract.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] focused lit filter for Toy target artifact, TensorExtLite target
  artifact, target artifact registry, RVV target artifact handoff, and
  TensorExtLite first-slice materialization: 12/12 passed.
- [OK] `git diff --check`
- [OK] targeted scans over common target, Toy target, TensorExtLite target, and
  Toy/TensorExtLite target tests showed forbidden strings only in fail-closed
  rejection logic or negative `CHECK-NOT` assertions, not as route authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 lit tests
  passed.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-17-common-materialized-emitc-header-foundation`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 101: Toy extension bundle target artifact bridge

**Date**: 2026-05-17
**Task**: Toy extension bundle target artifact bridge
**Branch**: `main`

### Summary

Added a Toy plugin-local target artifact bridge through the built-in
`ExtensionBundleRegistry` front door. The existing Toy selected
`tcrv_toy.compute_skeleton` boundary now supports a materialized EmitC route
and a declaration-only runtime ABI header artifact without adding Toy-specific
branches to tools, built-in target registration, or common target export.

### Main Changes

- Added `Target/Toy` target-support bundle APIs and implementation for the Toy
  materialized EmitC header artifact exporter.
- Rewired `ToyExtensionPlugin::configureTargetSupportExtensionBundle` to
  publish `tcrv_toy`, the selected `tcrv_toy.compute_skeleton` boundary, and
  Toy target artifact exporter bundle registration.
- Changed Toy emission planning from unsupported to a supported
  `runtime-callable-c-header` route that carries route provenance, source-op
  provenance, construction protocol role metadata, and ordered runtime ABI
  parameter evidence.
- Extended Toy EmitC route materialization with a bounded runtime ABI parameter
  (`size_t toy_value_count`) and kept the common EmitC materializer as the only
  source of C/C++ handoff evidence.
- Updated C++ and lit coverage for Toy positive export, stale provenance
  rejection, disabled built-ins fail-closed behavior, and RVV/TensorExtLite
  route preservation through the bundle registry.

### Testing

- [OK] `cmake --build build --target tianchenrv-toy-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] focused lit filter for Toy target artifacts, Toy execution planning,
  Toy EmitC materialization, TensorExtLite header artifact, RVV header
  artifact, and target artifact registry: 10/10 passed.
- [OK] `./build/bin/tcrv-translate --help | rg "tcrv-export-target-artifact|tcrv-export-target-header-artifact|tcrv-export-target-artifact-bundle"`
- [OK] Toy positive pipeline/header evidence probe prints origin plugin,
  selected variant, route id, runtime ABI name, source op/interface, and
  construction protocol.
- [OK] disabled built-ins probe fails closed before Toy materialization.
- [OK] targeted scans show no Toy-specific branches in `tools/tcrv-opt`,
  `tools/tcrv-translate`, `lib/Target/Builtin`, or
  `lib/Target/TargetArtifactExport.cpp`; descriptor/direct-C/source-export text
  appears only in rejection checks/messages.
- [OK] `git diff --check`

### Status

[OK] **Completed**

### Next Steps

- Toy object/runtime execution remains intentionally out of scope; next rebuild
  work can add object/native compilation only under an explicit Toy runtime ABI
  contract.


## Session 101: Extension bundle registry default plugin front door

**Date**: 2026-05-17
**Task**: Extension bundle registry default plugin front door
**Branch**: `main`

### Summary

Rewired the default built-in extension construction path so tools and target
front doors now build plugins through `plugin::ExtensionBundleRegistry`.
Target artifact exporter and target translate route registries remain
target-owned consumers, but the built-in default construction authority is now
the bundle catalog -> bundle registry -> enabled plugin registry path.

### Main Changes

- Added `registerBuiltinExtensionBundlePlugins` as the canonical built-in
  bundle front-door helper.
- Made `registerBuiltinExtensionPlugins` delegate through the bundle registry
  instead of looping the concrete built-in plugin catalog directly.
- Rewired `tcrv-opt` and `tcrv-translate` default initialization to keep an
  `ExtensionBundleRegistry`, populate plugins through it, and register dialects,
  passes, target artifact exporters, and target translate routes from the
  bundle-owned plugin registry.
- Added target artifact exporter and target translate route overloads that
  consume `ExtensionBundleRegistry + ExtensionPluginRegistry`.
- Preserved empty-registry behavior when `tcrv-opt` disables built-in plugins.
- Improved target artifact exporter and target translate route diagnostics so
  failures name the responsible bundle/plugin.
- Added focused C++ coverage for bundle-front-door plugin registration,
  target artifact exporter registration, target translate route registration,
  missing/disabled plugin fail-closed behavior, and error context.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `./build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-source-seed-artifact-front-door-pipeline`
- [OK] `./build/bin/tcrv-opt test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir --tcrv-disable-builtin-plugins --tcrv-rvv-materialize-i32m1-selected-boundary-seed 2>&1 | rg "Unknown command line argument|tcrv-rvv-materialize-i32m1-selected-boundary-seed"`
- [OK] `./build/bin/tcrv-translate --help | rg "tcrv-rvv-emitc-to-cpp|tcrv-export-target-artifact|tcrv-export-target-header-artifact|tcrv-export-target-artifact-bundle"`
- [OK] focused lit filter for source-seed target artifacts, RVV EmitC handoff,
  TensorExtLite target artifact, and RVV source seed: 5/5 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 100/100 lit tests
  passed.
- [OK] `git diff --check`
- [OK] targeted scans proving no concrete plugin catalog was reintroduced under
  `lib/Target/Builtin`, no `target::ExtensionBundle` resurrection occurred, no
  descriptor/direct-C/source-export/Python compiler-core diff was added, and no
  independent default `registerBuiltinExtensionPlugins` production loop remains.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-extension-bundle-registry-default-plugin-front-door`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 99: Built-in extension catalog target-layer decoupling

**Date**: 2026-05-17
**Task**: Built-in extension catalog target-layer decoupling
**Branch**: `main`

### Summary

Moved built-in extension bundle manifest ownership from target artifact export
into the built-in plugin catalog, then rewired target artifact export,
target translate route registration, `tcrv-opt`, and `tcrv-translate` to consume
that plugin/catalog front door.

### Main Changes

- Added `plugin::registerBuiltinExtensionBundles(...)` beside
  `plugin::registerBuiltinExtensionPlugins(...)`, with the concrete built-in
  RVV/Offload/Toy/Template/TensorExtLite/Scalar manifest list now owned in
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`.
- Reduced `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` to target
  artifact exporter aggregation only: it delegates catalog assembly to
  `Plugin/BuiltinExtensionPlugins` and continues to use
  `ExtensionBundleRegistry::registerTargetArtifactExportersForEnabledPlugins`.
- Removed target namespace built-in extension bundle helper declarations and
  migrated target translate, `tcrv-opt`, `tcrv-translate`, and target artifact
  tests to the plugin catalog API.
- Preserved current RVV object/header composite and TensorExtLite header route
  behavior; Toy, Template, Offload, and Scalar still contribute no current
  target artifact route authority.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] focused lit from `build/test` for TensorExtLite/RVV/Toy target artifact
  routes and registry front doors: 8/8 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 100/100 lit tests
  passed.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-built-in-extension-catalog-target-layer-decoupling`
- [OK] Targeted scans: target builtin artifact aggregation has no concrete
  plugin headers, no hard-coded extension family bundle list, and the source
  diff adds no descriptor/direct-C/source-export authority.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 99: Common extension target artifact bundle registration seam

**Date**: 2026-05-17
**Task**: Common extension target artifact bundle registration seam
**Branch**: `main`

### Summary

Moved plugin-owned target artifact exporter bundle discovery behind the common
extension-bundle front door so built-in target exporter registration no longer
directly includes or calls extension target-support bundle helpers.

### Main Changes

- Added a reusable
  `ExtensionBundleRegistry::registerTargetArtifactExporterBundles` seam and
  rewired enabled-plugin target exporter registration through it.
- Collapsed built-in extension bundle registration to one generic manifest
  spec list, with target-support metadata coming from plugin manifest hooks.
- Moved TensorExtLite target-support bundle configuration into the
  TensorExtLite plugin hook, and moved Offload/Template required dialect bundle
  metadata into their plugin hooks.
- Split TensorExtLite construction protocol and EmitC route provider sources
  into dedicated CMake sublibraries so TensorExtLite target support can depend
  on route providers without depending on the full TensorExtLite plugin.
- Added C++ coverage proving RVV/TensorExtLite plugin-owned target exporter
  bundles are collected through the common front door, Toy/Template/Offload/
  Scalar remain no-route where expected, and invalid plugin exporter bundle
  registration fails closed with generic diagnostics.

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-rvv-extension-plugin-test tcrv-translate -j2`.
- [OK] Focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`, and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit target artifact routes: 8/8 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`: 100/100 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Trellis validation:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-common-extension-target-artifact-bundle-registration-seam`.
- [OK] Targeted scans: no extension target-support header/direct helper calls
  remain in built-in target exporter code; descriptor/direct-C/source-export
  matches are negative tests or deletion-rule checks.
- [WARN] `clang-format` was not available in PATH.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 99: TensorExtLite materialized EmitC target artifact bridge

**Date**: 2026-05-17
**Task**: TensorExtLite materialized EmitC target artifact bridge
**Branch**: `main`

### Summary

Rebuilt the TensorExtLite first-slice selected target artifact bridge as a
materialized-EmitC-derived declaration header route. The route now goes from
explicit TensorExtLite role ops through the plugin-owned common EmitC lowerable
route and the common selected EmitC artifact front door, without restoring
metadata diagnostic, descriptor, source-export, or object/runtime authority.

### Main Changes

- Changed TensorExtLite construction/emission artifact kind to
  `runtime-callable-c-header` for the bounded first-slice route.
- Made TensorExtLite emission planning produce a supported selected header
  artifact candidate with route ID, emission kind, runtime ABI ownership,
  lowering-boundary metadata, required capabilities, and EmitC route/role
  provenance.
- Added a TensorExtLite target support bundle that registers a standalone
  header exporter through the built-in extension bundle registry.
- Implemented the TensorExtLite header exporter through the common selected
  EmitC artifact front door, then prints only a declaration header and bounded
  provenance comments.
- Updated C++ and lit coverage for positive header export, default object
  fail-closed behavior, missing route provenance, stale descriptor-like
  metadata, and metadata-diagnostic rejection.
- Updated lowering-runtime specs to record that TensorExtLite now has exactly
  one materialized-EmitC-derived header route, while object/runtime evidence
  remains future rebuild work.

### Testing

- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-05-17-tensorextlite-materialized-emitc-target-artifact-bridge`
- [OK] `cmake --build build --target tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused TensorExtLite lit filter:
  `tensorext-lite-(first-slice-materialization|target-artifact-(unsupported|header))`,
  5/5 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 100/100 lit
  tests passed.
- [OK] Targeted scans over touched plugin/target/test/spec surfaces found no
  production resurrection of `metadata-diagnostic`, descriptor/direct-C/
  source-export authority, or RVV route leakage.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 99: Metadata-diagnostic artifact-kind authority erasure

**Date**: 2026-05-17
**Task**: Metadata diagnostic artifact-kind authority erasure
**Branch**: `main`

### Summary

Deleted `metadata-diagnostic` as a current supported construction/emission and
target artifact kind. Toy and TensorExtLite now keep only unsupported diagnostic
artifact-kind metadata for construction surfaces without publishing target
artifact authority, and TensorExtLite emission planning fails closed while
separate in-memory EmitC route materialization remains covered.

### Main Changes

- Removed `metadata-diagnostic` from common construction, plugin emission-plan,
  and target artifact exporter validators.
- Rewrote Toy and TensorExtLite construction route metadata to use
  `unsupported-emission-diagnostic` instead of metadata-only artifact authority.
- Rewrote TensorExtLite emission planning from a supported metadata diagnostic
  artifact to an unsupported plan with bounded unsupported runtime ABI ownership
  metadata.
- Updated target artifact tests to reject metadata diagnostic exporter
  registration and to keep bundle/default artifact coverage on object/header
  materialized kinds.
- Updated lit coverage so TensorExtLite target artifact export finds no
  supported route and combined unsupported-plan plus EmitC-route materialization
  fails closed.
- Added emission-runtime spec coverage for the metadata-diagnostic artifact-kind
  erasure contract and removed stale supported-plan wording from the EmitC route
  spec.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Built focused C++ targets:
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-plugin-emission-plan-test`,
  `tianchenrv-toy-extension-plugin-test`,
  `tianchenrv-tensorext-lite-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] Ran the five focused C++ test executables.
- [OK] Focused lit filter:
  `tensorext-lite-(first-slice-materialization|target-artifact-unsupported)`.
- [OK] `git diff --check`.
- [OK] Production scan found no `metadata-diagnostic` under
  `include/TianChenRV/Plugin`, `lib/Plugin`, `include/TianChenRV/Target`, or
  `lib/Target`.
- [OK] Test scan found `metadata-diagnostic` only in negative rejection cases.
- [OK] `cmake --build build --target check-tianchenrv`: 99/99 lit tests
  passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 99: Toy template target artifact route erasure

**Date**: 2026-05-17
**Task**: Toy template target artifact route erasure
**Branch**: `main`

### Summary

Deleted the remaining Toy metadata/source target artifact route, removed the
Toy target support bundle and direct translate route exposure, downgraded Toy
emission planning to an unsupported non-artifact diagnostic, and kept the Toy
EmitC materialization helper as non-production target-artifact behavior.

### Main Changes

- Removed `TianChenRVToyTarget`, `ToyTargetSupportBundle`, the Toy target
  exporter registration hook, and the Toy target translate route hook.
- Deleted `test/Target/Toy/*` fixtures that protected
  `tcrv-toy-template-artifact`, Toy `metadata-diagnostic` export, and
  `materialized_emitc_cpp_source` positive target artifact output.
- Updated built-in target artifact and translate route tests so the current
  production route set is RVV materialized EmitC object plus RVV header
  composite, with no Toy target artifact route.
- Changed Toy emission plans from supported metadata artifact candidates to
  unsupported diagnostics that state the Toy target artifact route is deleted.
- Refreshed the lowering-runtime spec so supported built-in target artifact
  routes are limited to explicit rebuilt materialized routes and Toy metadata
  exporters cannot stand in for compiler-owned lowering.

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-toy-extension-plugin-test tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] focused lit filter:
  `execution-planning-pipeline-toy|source-seed-target-artifact-object|tensorext-lite-target-artifact-unsupported|toy-template-materialization|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized`,
  9/9 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 99/99 lit tests passed.
- [OK] `git diff --check`
- [OK] `tcrv-translate --help` scan shows no `tcrv-toy-template-artifact`.
- [OK] targeted residue scan over Toy target, built-in target registration,
  tcrv-translate, Toy target lit, and TargetArtifactExportTest is clean for
  the deleted route strings and positive Toy metadata/source artifact outputs.

### Self-Repair

- First full `check-tianchenrv` failed because Toy still published a supported
  `metadata-diagnostic` emission plan after the exporter was deleted. Fixed by
  making the Toy emission plan unsupported rather than restoring the old target
  artifact exporter.

### Spec Update

- Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` to
  reflect the current supported target artifact route boundary after Toy
  deletion.

### Status

[OK] Completed and ready to archive.


## Session 98: RVV unseeded vector-loop selected-boundary materializer

**Date**: 2026-05-17
**Task**: `05-17-rvv-unseeded-vector-loop-selected-boundary-materializer`
**Branch**: `main`

### Summary

Moved the bounded RVV i32m1 add source front door from
`tcrv_rvv.lowering_seed` authorization to RVV-owned structural source
recognition, while preserving the existing selected multi-VL EmitC
object/header/bundle route.

### Main Changes

- Reworked `RVVSelectedBoundarySeed.cpp` to recognize one unseeded source
  function with positional `lhs/rhs/out/n`, `scf.for 0 to n step 4`,
  `vector<4xi32>` load/load/add/store, and no extra source compute.
- Rejected stale `tcrv_rvv.lowering_seed` metadata as route authority; it now
  only appears in negative tests/spec text and the rejection diagnostic.
- Kept the public RVV source materialization pass/front-door name as plugin
  plumbing, but updated descriptions and generated policy metadata to
  `source-pattern-*`.
- Updated positive RVV transform, SourceSeed pipeline, and target
  object/header/bundle fixtures to consume the unseeded source pattern.
- Added/updated negative coverage for wrong ABI/order, stale seed metadata,
  wrong loop bounds/step/vector shape, non-add arithmetic, extra loop ops,
  missing store, and stale selected-boundary residue.
- Updated RVV and variant-pipeline specs so the source body, not seed metadata,
  is the durable positive authority.

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [OK] Focused C++:
  `./build/bin/tianchenrv-rvv-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- [OK] Focused lit:
  `rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door-pipeline|source-seed-target-artifact-(header|object)`,
  7/7 passed.
- [OK] Focused explicit RVV route lit:
  `rvv-first-slice-materialization|i32m1-(add|sub|mul)|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized`,
  8/8 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 106/106 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Artifact evidence under
  `artifacts/tmp/rvv_unseeded_vector_loop_selected_boundary_materializer/20260516T202408Z`.
  `llvm-readobj` reports `Format: elf64-littleriscv`, `Arch: riscv64`, and
  `Type: Relocatable`.
- [OK] Real `ssh rvv` unseeded source-path compile/link/run:
  `vlmax_e32m1=4`, `n=4/5/11` all PASS, ending with
  `unseeded_multi_vl_tail_status=PASS`.
- [OK] Targeted scans: no positive RVV transform/target fixture still requires
  `tcrv_rvv.lowering_seed`; remaining hits are stale-metadata rejection
  tests/spec/code. No descriptor/direct-C/source-export route authority was
  restored in touched code/test surfaces.

### Self-Repair

- Re-ran artifact `readobj` with `/usr/lib/llvm-20/bin/llvm-readobj` after the
  shell PATH lacked `llvm-readobj`.
- Kept public pass/front-door names stable while removing seed metadata as
  production route authority.

### Status

[OK] Completed; ready to archive and commit.


## Session 98: RVV runtime AVL multi-VL EmitC loop

**Date**: 2026-05-17
**Task**: RVV runtime AVL multi-VL EmitC loop
**Branch**: `main`

### Summary

Replaced the bounded RVV i32m1 arithmetic one-VL artifact limit with a
materialized runtime multi-VL EmitC loop through the existing RVV
extension-family ops and common EmitC route.

### Main Changes

- Added structured loop payload support to `TCRVEmitCLowerableRoute` and
  materialized it as `emitc.for`.
- Rewired the RVV i32m1 add/sub/mul EmitC route to loop over runtime `n`,
  recompute per-chunk VL from remaining AVL, advance lhs/rhs/out by `offset`,
  and place RVV intrinsic load/compute/store calls inside the loop.
- Updated RVV runtime AVL/VL artifact metadata from
  `multi_vl = unsupported` to supported multi-VL metadata for the materialized
  loop route.
- Hardened target artifact export so multi-VL claims require materialized
  `emitc.for` route provenance.
- Updated header/object/bundle tests, materialization tests, target artifact
  negatives, and Trellis specs for the durable multi-VL boundary.

### Git Commits

| Hash | Message |
|------|---------|
| `this-commit` | (see git log) |

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-emitc-lowerable-interface-test tcrv-opt tcrv-translate -j2`.
- [OK] Focused C++ tests:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-emitc-lowerable-interface-test`.
- [OK] Focused lit filter:
  `rvv-first-slice-materialization|rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|source-seed-target-artifact-header|source-seed-target-artifact-object`,
  11/11 selected tests passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 106/106 lit
  tests passed.
- [OK] Artifact evidence under
  `artifacts/tmp/rvv_runtime_avl_multivl_emitc_loop/20260516T200010Z`.
  Generated C++ contains `emitc.for` lowered to a C++ `for`, `n - offset`,
  pointer advancement by `offset`, and repeated `__riscv_vsetvl_e32m1`.
  `llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- [OK] Real `ssh rvv` compile/run from generated header/object:
  `vlmax_e32m1=4`, `n=4 PASS`, `n=5 PASS`, `n=11 PASS`,
  `multi_vl_tail_status=PASS`.
- [OK] Targeted scans: no stale one-VL unsupported markers and no restored
  descriptor/direct-C/source-export authority in implementation surfaces. The
  only remaining `descriptor` hit is target preflight rejecting descriptor-local
  or hardcoded element-count residue.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 98: RVV runtime AVL/VL ABI artifact closure

**Date**: 2026-05-17
**Task**: RVV runtime AVL/VL ABI artifact closure
**Branch**: `main`

### Summary

Closed the runtime AVL/VL ABI metadata handoff for the existing bounded RVV
i32m1 materialized EmitC artifact route. The selected emission plan, target
candidate validation, declaration-only header, bundle index, and `ssh rvv`
evidence now agree that ABI parameter `n` is runtime AVL, `tcrv_rvv.setvl`
produces the VL consumed by `tcrv_rvv.with_vl`, and the current artifact is a
one-VL i32m1 arithmetic slice with multi-VL support explicitly unsupported.

### Main Changes

- Added bounded-slice and multi-VL support metadata to the RVV i32m1 artifact
  metadata contract.
- Published the full RVV config/runtime AVL/VL contract from the supported RVV
  emission-plan diagnostic.
- Made RVV object/header/bundle candidate preflight reject missing or stale
  runtime AVL/VL metadata and descriptor/hardcoded element-count residue.
- Added runtime AVL/VL boundary comments to the generated declaration-only
  header from the same selected candidate metadata.
- Extended C++ and lit coverage for positive plan/header/bundle metadata and
  fail-closed stale/missing metadata cases.

### Git Commits

| Hash | Message |
|------|---------|
| `this-commit` | (see git log) |

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- [OK] Focused C++:
  `./build/bin/tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|source-seed-target-artifact-header|source-seed-target-artifact-object|rvv-first-slice-vl-contract-negative|rvv-first-slice-config-vl-contract-negative|rvv-first-slice-materialization-missing-abi|rvv-first-slice-materialization-negative`,
  8/8 selected tests passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 106/106 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Artifact evidence under
  `artifacts/tmp/rvv_runtime_avl_vl_abi_artifact_closure/20260516T192409Z`.
  `llvm-readobj -h rvv_target_artifact.o` reports `Format:
  elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- [OK] Real `ssh rvv` header/object compile-run:
  `tcrv_rvv_runtime_avl_vl_header_bundle status=PASS runtime_avl_values=2,4 bounded_slice=one-vl-i32m1-arithmetic`.
- [OK] Targeted scans found no restored descriptor route authority, direct-C
  semantic exporter, source-export route, old RVV i32m1 route ids, or
  descriptor-local/hardcoded artifact element-count metadata in touched
  surfaces; matches were only negative guard text.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 97: RVV materialized EmitC header and bundle ABI packaging bridge

**Date**: 2026-05-17
**Task**: RVV materialized EmitC header and bundle ABI packaging bridge
**Branch**: `main`

### Summary

Rebuilt the RVV declaration-only header route and object+header artifact bundle
around the existing selected materialized EmitC object bridge, without
restoring descriptor/direct-C/source-export route authority.

### Main Changes

- Added `rvv-i32m1-arithmetic-emitc-route-family.header` as a composite header
  artifact route matched from the existing RVV materialized EmitC object
  candidate.
- Put the RVV object route and header route into the shared bundle component
  group `rvv-i32m1-arithmetic-materialized-emitc-bundle.v1`.
- Header export now materializes the selected EmitC module, verifies the single
  `emitc.func` boundary and ordered runtime ABI arity, then emits only a
  callable declaration with `<stddef.h>` and `<stdint.h>`.
- Extended composite bundle metadata with `handoffKind` so the bundle index
  records the same materialized object handoff identity for the header
  component.
- Updated RVV/lowering-runtime specs to document the bounded active
  object/header/bundle route while keeping historical header/bundle route ids
  deleted.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- [OK] Focused C++:
  `./build/bin/tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter:
  `source-seed-target-artifact-header|source-seed-target-artifact-object|rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized`,
  6/6 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 102/102 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Artifact evidence under
  `artifacts/tmp/rvv_materialized_emitc_header_bundle_bridge/20260516T181607Z`.
  `/usr/lib/llvm-20/bin/llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- [OK] Real `ssh rvv` header+object compile/link/run:
  `tcrv_rvv_materialized_emitc_header_bundle status=PASS n=4 add=[12,6,16,12]`.
- [OK] Targeted scans: no restored
  `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, old
  `getRVVI32M1Arithmetic{Object,Header,Bundle,Target}*RouteID`, historical
  `tcrv-rvv-i32m1-{add,sub,mul}` header/object/bundle route strings, or
  descriptor/direct-C/source-export/source-authority residue in RVV
  target/plugin/test surfaces; generated headers contain no RVV intrinsic body,
  `main`, runtime logs, ssh text, credentials, or performance/correctness
  claims.

### Self-Repair

- The first local readobj evidence command used `llvm-readobj` from the normal
  shell `PATH`; regenerated the readobj evidence with
  `/usr/lib/llvm-20/bin/llvm-readobj`.

### Status

[OK] Complete; ready to archive and commit.


## Session 98: RVV target descriptor-route authority erasure

**Date**: 2026-05-17
**Task**: `rvv-target-descriptor-route-erasure`
**Branch**: `main`

### Summary

Deleted RVV target-support descriptor/table authority for finite i32m1
add/sub/mul object/header/translate artifact routes. RVV selected artifact
planning now reports an unsupported deleted-route diagnostic; the retained RVV
EmitC-to-C++ route only accepts an already materialized EmitC module.

### Main Changes

- Created Trellis task `05-17-rvv-target-descriptor-route-erasure` and wrote a
  deletion-only PRD from the supplied Direction Brief.
- Replaced `lib/Target/RVV/RVVTargetSupportBundle.cpp` with a minimal
  materialized EmitC-to-C++ translate route registration; no RVV artifact
  exporter bundle is installed.
- Removed RVV construction-protocol object/header/translate route fields and
  updated construction tests accordingly.
- Changed RVV emission readiness/planning to fail closed after validating the
  explicit typed RVV body, instead of naming a supported object/header route.
- Deleted old RVV target object/header/bundle lit fixtures and rewrote the
  source-seed fixture to assert unsupported/deleted-route diagnostics and
  target artifact export failure.
- Updated lowering-runtime and RVV plugin specs to stop treating finite RVV
  object/header target routes as current production behavior.

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tcrv-opt tcrv-translate -j2`.
- [OK] Focused C++:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized|rvv-first-slice-materialization|source-seed-artifact-front-door`,
  13/13 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 100/100 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Targeted scans found no RVV target descriptor/table names, old finite
  RVV target route ids, construction object/header/translate route fields, or
  RVV target artifact direct exporter residue in the touched target/test
  surfaces. Retained `riscv_vector.h` and `__riscv_vadd_vv_i32m1` evidence is
  limited to the materialized EmitC-to-C++ handoff fixture.

### Self-Repair

- Narrowed the source-seed fixture after the initial lit run correctly showed
  that EmitC materialization cannot select a supported non-fallback emission
  plan once RVV artifact route authority is deleted.

### Spec Update Judgment

Spec updates were required because the current RVV target artifact route state
changed from supported finite object/header export to unsupported deleted-route
diagnostics pending a non-descriptor rebuild.

### Status

[OK] Completed and ready to archive.


## Session 97: Registry-composed source-seed-to-artifact front door

**Date**: 2026-05-16
**Task**: Registry-composed source-seed-to-artifact front door
**Branch**: `main`

### Summary

Added an explicit `tcrv-opt` source-seed artifact front-door pipeline. The new
pipeline collects plugin-registered source-seed materialization passes in
registry order, then runs generic legality/capability, emission-plan, and
execution-plan coherence gates so bounded RVV and Toy source seeds can feed the
existing target artifact routes without direct Toy/RVV wiring in public tools or
common transforms.

### Main Changes

- Created Trellis task
  `05-16-registry-composed-source-seed-front-door` from the supplied Direction
  Brief and wrote the PRD around a registry-composed front door.
- Added `--tcrv-source-seed-artifact-front-door-pipeline`.
- The pipeline runs enabled plugin source-seed passes, then
  `tcrv-check-hart-parallel-capabilities`,
  `tcrv-verify-plugin-variant-legality`,
  `tcrv-check-capability-requires`, `tcrv-materialize-emission-plans`, and
  `tcrv-check-execution-plan-coherence`.
- Kept the ordinary `tcrv-execution-planning-pipeline` unchanged; source-seed
  outputs already contain selected variant/boundary surfaces and should not be
  sent through proposal/selection again.
- Added RVV/Toy positive lit coverage proving both source seeds exercise the
  same pipeline and feed the existing target artifact routes.
- Added negative lit coverage for disabled built-ins, unsupported source seed
  markers, stale residue, and mixed incompatible RVV+Toy seed inputs.
- Updated the variant-pipeline spec with the durable public contract for the
  source-seed artifact front-door pipeline.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVTransforms TianChenRVPlugin
  TianChenRVRVVPlugin TianChenRVToyPlugin tcrv-opt tcrv-translate
  tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test
  tianchenrv-toy-extension-plugin-test -j2`.
- [OK] C++ tests:
  `tianchenrv-plugin-registry-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-toy-extension-plugin-test`.
- [OK] Focused lit from `build/test`:
  `source-seed-artifact-front-door|rvv-i32m1-selected-boundary-seed|toy-template-selected-boundary-seed|i32m1-add-object-artifact`,
  8/8 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 109/109
  passed.
- [OK] `git diff --check`.
- [OK] Tool direct seed-factory scan returned no matches under
  `tools/tcrv-opt` and `tools/tcrv-translate`.
- [OK] Common/core source-seed semantic branch scan returned no matches under
  `include/TianChenRV/Transforms`, `lib/Transforms`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`, and
  `lib/Plugin/ExtensionPlugin.cpp`.
- [OK] RVV front-door artifacts generated under
  `artifacts/tmp/rvv_selected_boundary_seed_frontdoor/20260516T143133Z`.
- [OK] `ssh rvv` link/run printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- Avoided composing source seeds with `tcrv-execution-planning-pipeline` after a
  live probe showed the proposal stage correctly rejects already selected seed
  variants.
- Avoided unconditional selected lowering-boundary materialization after a live
  probe showed it duplicates the Toy seed boundary.
- Repaired the disabled-builtins negative test to expect the final generic
  coherence fail-closed diagnostic.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 95: Plugin-owned source seed registration interface

**Date**: 2026-05-16
**Task**: Plugin-owned source seed registration interface
**Branch**: `main`

### Summary

Moved the RVV bounded selected-boundary source seed pass behind a common
extension-plugin source-seed registration surface. `tcrv-opt` now registers the
seed by collecting enabled plugin registrations instead of directly including
or invoking the RVV pass factory.

### Main Changes

- Created and archived Trellis task
  `05-16-plugin-owned-source-seed-registration-interface` from the supplied
  Direction Brief.
- Added `SourceSeedPassRegistration` and
  `ExtensionPluginRegistry::collectSourceSeedPasses` as common pass
  descriptor/factory plumbing.
- Added `ExtensionPlugin::registerSourceSeedPasses` with default no-op
  behavior, and moved the existing RVV i32m1 selected-boundary seed factory
  behind `RVVExtensionPlugin`.
- Removed direct `RVVSelectedBoundarySeed.h` include and direct RVV seed pass
  factory registration from `tools/tcrv-opt/tcrv-opt.cpp`.
- Removed the direct `TianChenRVRVVPlugin` link from `tcrv-opt`; the tool now
  reaches built-in RVV behavior through built-in plugin/target registration.
- Added registry and RVV plugin C++ coverage for source-seed pass registration,
  plus lit coverage proving `--tcrv-disable-builtin-plugins` leaves the RVV
  seed pass unregistered.
- Promoted the source-seed pass provider interface rules into
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | plugin(registry): route source seeds through plugins |

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVPlugin TianChenRVRVVPlugin tcrv-opt
  tcrv-translate tianchenrv-plugin-registry-test
  tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] C++ tests:
  `tianchenrv-plugin-registry-test`,
  `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit from `build/test`:
  `rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary`,
  19/19 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 104/104
  passed.
- [OK] `git diff --check`.
- [OK] Public-tool direct RVV seed wiring scan returned no matches for
  `RVVSelectedBoundarySeed` or
  `createMaterializeRVVI32M1SelectedBoundarySeedPass` under `tools/`.

### Hardware Evidence

Not rerun. This round changed pass registration ownership only; selected
boundary output, emission-plan/EmitC route semantics, target artifact semantics,
and runtime ABI shape are unchanged and covered by focused lit. The previous
seed task already produced `ssh rvv` correctness evidence for the same route.

### Self-Repair

- First focused build exposed an incomplete `mlir::Pass` type in
  `RVVExtensionPlugin.cpp`; added `mlir/Pass/Pass.h` and reran successfully.
- First lit attempt used `lit.py build/test` from repo root and hit this repo's
  generated relative `lit.site.cfg.py` path. Reran from `build/test`, which is
  the project convention, and the focused set passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 94: Bounded MLIR-to-RVV selected-boundary seed

**Date**: 2026-05-16
**Task**: Bounded MLIR-to-RVV selected-boundary seed
**Branch**: `main`

### Summary

Added an RVV-owned bounded vector i32 add source seed pass that materializes selected RVV i32m1 boundary IR, covered positive/negative lit, generated target artifacts, and passed ssh rvv correctness plus full check-tianchenrv.

### Main Changes

- Created and archived Trellis task
  `05-16-bounded-mlir-to-rvv-selected-boundary-seed` from the supplied
  Direction Brief.
- Added RVV plugin-owned pass
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed`, registered through
  `tcrv-opt`, for one explicitly marked `func`/`scf`/`vector`/`arith` i32 add
  source shape.
- The pass materializes `tcrv.exec.kernel`, a selected `origin = "rvv-plugin"`
  variant, explicit `lhs`/`rhs`/`out`/`n` runtime ABI bindings,
  `tcrv_rvv.setvl`, selected `tcrv_rvv.with_vl`, and RVV
  `i32_load` / `i32_add` / `i32_store`.
- Added positive and fail-closed lit coverage for selected-boundary output,
  emission-plan/EmitC route consumption, missing ABI operands, unsupported
  dtype/rank/vector shape, malformed source body, and stale
  `tcrv.exec`/`tcrv_rvv` residue.
- Promoted the bounded seed command/input/output/error contract into
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Generated seed object/header artifacts under
  `artifacts/tmp/rvv_selected_boundary_seed/20260516T132508Z` and ran a real
  `ssh rvv` link/run harness.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build for RVV construction protocol, RVV EmitC route provider,
  RVV plugin, transforms, `tcrv-opt`, `tcrv-translate`, and focused C++ tests.
- [OK] C++ tests:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-rvv-dialect-test`.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary`,
  19/19 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 104/104
  passed.
- [OK] `ssh rvv` seed harness printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 96: Toy source-seed plugin-template proof

**Date**: 2026-05-16
**Task**: Toy source-seed plugin-template proof
**Branch**: `main`

### Summary

Added Toy as the second plugin-owned source-seed registry consumer. The new
bounded Toy seed is registered through `ToyExtensionPlugin::registerSourceSeedPasses`,
materializes the existing Toy selected `tcrv_toy.compute_skeleton` boundary,
and feeds the existing Toy EmitC and target artifact route without direct
`tcrv-opt` wiring or common/core Toy semantic branches.

### Main Changes

- Created Trellis task `05-16-toy-source-seed-plugin-template-proof` from the
  supplied Direction Brief and wrote the PRD around a Toy-only second consumer.
- Added `--tcrv-toy-materialize-template-selected-boundary-seed` under the Toy
  plugin.
- The seed accepts one explicit source-only marker,
  `tcrv_toy.lowering_seed = "template_compute"`, on a zero-argument,
  zero-result `func.func` with one empty return.
- The seed materializes a Toy `tcrv.exec.kernel`, available `toy.template`
  capability, selected `origin = "toy-plugin"` variant with the existing Toy
  construction metadata, selected diagnostic, and `tcrv_toy.compute_skeleton`.
- Added Toy plugin/unit coverage for source-seed registration metadata and
  factory construction.
- Added lit coverage for Toy seed positive boundary, EmitC, target artifact,
  built-in-disabled fail-closed behavior, unsupported marker/shape/body, stale
  `tcrv.exec`, and stale `tcrv_toy` residue.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVToyPlugin tcrv-opt tcrv-translate
  tianchenrv-toy-extension-plugin-test tianchenrv-plugin-registry-test -j2`.
- [OK] C++ tests:
  `tianchenrv-toy-extension-plugin-test`,
  `tianchenrv-plugin-registry-test`.
- [OK] Focused lit from `build/test`:
  `toy-template-selected-boundary-seed|toy-template-target-artifact|toy-template-materialization|rvv-i32m1-selected-boundary-seed`,
  14/14 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 106/106
  passed.
- [OK] `git diff --check`.
- [OK] Tool direct seed-factory scan returned no matches under
  `tools/tcrv-opt` and `tools/tcrv-translate`.
- [OK] Common/core source-seed semantic branch scan returned no matches under
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`, `include/TianChenRV/Transforms`, and
  `lib/Transforms`.

### Self-Repair

- Fixed zero-argument MLIR source syntax in the new Toy lit input.
- Repaired Toy seed variant metadata to use the existing Toy legality
  attributes (`tcrv_toy.*`) rather than construction-protocol helper metadata
  names.
- Replaced an invalid unknown-op Toy residue test with a legal
  `tcrv_toy.compute_skeleton` residue.

### Spec Update Judgment

No spec update was needed. Existing plugin-protocol specs already define the
source-seed registry rule; this round proved a second consumer without adding a
new durable contract.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 98: RVV selected-boundary runtime-VL contract

**Date**: 2026-05-16
**Task**: RVV selected-boundary config and runtime-VL contract
**Branch**: `main`

### Summary

Defined and enforced one explicit RVV i32m1 config/runtime-VL contract across
the existing source-seed add route, selected emission-plan diagnostics, target
artifact candidates, generated headers, and bundle metadata. The RVV-specific
contract stays in RVV-owned code; common emission readiness, coherence, and
target export only carry generic bounded key/value artifact metadata.

### Main Changes

- Created Trellis task
  `05-16-rvv-selected-boundary-runtime-vl-contract` from the supplied
  Direction Brief and wrote the PRD around the existing RVV i32m1 source-seed
  proof path.
- Added a small generic `ArtifactMetadataEntry` support type and threaded
  artifact metadata through `VariantEmissionPlan`, `tcrv.exec.diagnostic`,
  target artifact candidates, and target artifact bundle records.
- Extended `RVVConfigContract` with the exact i32m1 metadata contract:
  SEW32, LMUL m1, tail/mask agnostic policy, runtime AVL from ABI `n`,
  `setvl`/`with_vl` boundary identity, same-VL dataflow uses, and callable ABI
  order `lhs,rhs,out,n`.
- Made RVV emission plans emit the metadata and made RVV target artifact export
  validate it before producing object/header/bundle artifacts.
- Updated RVV source-seed, selected-dispatch, target artifact, bundle, and C++
  target export tests for positive propagation plus fail-closed missing or
  mismatched metadata.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin
  TianChenRVTarget tcrv-opt tcrv-translate
  tianchenrv-target-artifact-export-test -j2`.
- [OK] Focused RVV/plugin build:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-construction-protocol-common-test
  tianchenrv-rvv-dialect-test -j2`.
- [OK] C++ tests:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-rvv-dialect-test`.
- [OK] Focused lit from `build/test`:
  `rvv-i32m1-selected-boundary-seed|i32m1-add-object-artifact|i32m1-selected-dispatch-artifact|i32m1-sub-selected-dispatch-artifact|i32m1-mul-selected-dispatch-artifact|i32m1-object-stale-route-op|i32m1-object-missing-contract-metadata|i32m1-artifact-ambiguous-selected|source-seed-artifact-front-door`,
  12/12 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110
  passed.
- [OK] `git diff --check`.
- [OK] Generated RVV source-seed artifacts under
  `artifacts/tmp/rvv_selected_boundary_runtime_vl_contract/20260516T150644Z`.
- [OK] `ssh rvv` link/run passed:
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.
- [OK] Changed-surface scans showed no new descriptor/direct-C/Python
  compiler-core route and no RVV semantic branch in common/core orchestration.

### Self-Repair

- Kept common metadata parsing and printing extension-neutral after rejecting
  RVV-specific handling in common transforms/target export as the wrong
  ownership boundary.
- Updated existing hand-authored RVV emission-plan fixtures so they model the
  same explicit contract as plugin-generated emission plans.
- Added a missing-contract RVV target export negative test to prove fail-closed
  behavior for stale or incomplete artifact candidates.

### Spec Update Judgment

No spec update was needed. Existing RVV plugin, EmitC route, variant pipeline,
and MLIR testing specs already cover this boundary and evidence policy.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 95: Bounded source-vector-to-RVV selected-boundary lowering

**Date**: 2026-05-16
**Task**: Bounded source-vector-to-RVV selected-boundary lowering
**Branch**: `main`

### Summary

Made the RVV i32m1 add source seed consume validated source shape and source-derived ABI provenance before materializing the selected boundary.

### Main Changes

### Main Changes

- Created and archived Trellis task `05-16-05-16-bounded-source-vector-to-rvv-selected-boundary-lowering` from the supplied Direction Brief.
- Reworked the RVV selected-boundary source seed matcher to return a recognized `BoundedI32AddSourceSeed` before materialization.
- Materialization now consumes source-derived ABI argument mapping and emits `tcrv_rvv.runtime_abi_value` purpose provenance: `source-arg-0:lhs`, `source-arg-1:rhs`, `source-arg-2:out`, and `source-arg-3:n`.
- Tightened source-shape validation to reject loop-carried `scf.for` values before RVV boundary creation.
- Expanded negative lit coverage for unsupported marker, missing/extra `n`, wrong arithmetic op, wrong output buffer use, unsupported loop bounds/step, loop-carried values, unrelated/empty body, and stale `tcrv.exec`/`tcrv_rvv` residue.
- Updated the variant-pipeline spec with the source-argument provenance and loop-carried rejection contract.

### Testing

- [OK] Focused build for RVV dialect/plugin/target, `tcrv-opt`, `tcrv-translate`, and RVV/target C++ tests.
- [OK] C++ tests: `tianchenrv-rvv-extension-plugin-test`, `tianchenrv-construction-protocol-common-test`, `tianchenrv-rvv-dialect-test`, `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter: `rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door|i32m1-add-object-artifact`, 6/6 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110 passed.
- [OK] `git diff --check`.
- [OK] Changed-surface scans: no common/core/tooling files changed; no descriptor/direct-C/source-export/Python compiler-core route terms in changed RVV seed/test diff.
- [OK] New artifacts under `artifacts/tmp/bounded_source_vector_to_rvv_selected_boundary/20260516T153540Z`.
- [OK] Real `ssh rvv` link/run: `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- Fixed a const erase compile error in the matched seed cleanup loop.
- Fixed the loop-carried negative test to reach the intended `scf.for iter_args` source-shape gate.

### Status

[OK] Completed and archived.


## Session 97: Common materialized EmitC handoff for RVV source dispatch

**Date**: 2026-05-17
**Task**: `rvv-source-dispatch-emitc-handoff`
**Branch**: `main`

### Summary

Made the existing bounded RVV i32m1 add source-seed selected-dispatch path
prove a common materialized MLIR EmitC handoff before MLIR EmitC C/C++ emission
and RVV target artifact packaging.

### Main Changes

- Created and archived Trellis task
  `05-17-rvv-source-dispatch-emitc-handoff` from the supplied Direction Brief.
- Added target-side validation in `lib/Target/TargetArtifactExport.cpp` so
  selected EmitC artifact routes must materialize an EmitC function boundary
  with route source-op and call source-op provenance before C/C++ emission or
  object packaging.
- Kept the common helper extension-neutral; RVV intrinsic/header/runtime details
  remain in RVV-owned route and target-support code.
- Extended `test/Target/TargetArtifactExportTest.cpp` to prove common selected
  EmitC source keeps route provenance and rejects route builders that omit route
  source-op provenance.
- Extended the RVV source-seed selected-dispatch fixture to run through
  materialized EmitC into the MLIR EmitC C/C++ emitter before object/header/
  bundle export checks.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  selected artifact handoff contract.

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- [OK] C++ tests:
  `tianchenrv-emission-readiness-test`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-dialect-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter from `build/test`:
  `rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|i32m1-add-object-artifact|i32m1-selected-dispatch-artifact|i32m1-object-stale-route-op|i32m1-object-missing-contract-metadata|i32m1-artifact-ambiguous-selected|source-seed-artifact-front-door`,
  11/11 selected tests passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Changed-surface scan: no Python files changed; no descriptor/direct-C/
  source-export route was added; common target code only adds generic EmitC
  handoff validation.
- [OK] Refreshed artifacts under
  `artifacts/tmp/source_seed_emitc_handoff/20260516T163726Z`.
- [OK] Real `ssh rvv` link/run:
  `tcrv_rvv_i32m1_source_emitc_handoff status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- `clang-format` was unavailable, so the touched C++ line wrap was manually
  normalized.
- Re-ran the final full check with the clean target command after an earlier
  redundant build-tool suffix.

### Spec Update Judgment

Spec updated because this round changed the executable target/export handoff
contract: selected artifact routes that emit/package from EmitC must validate
materialized route provenance before emitter/toolchain steps.

### Status

[OK] Completed and archived.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 96: RVV source-seed selected-dispatch artifact path

**Date**: 2026-05-16
**Task**: `rvv-source-seed-selected-dispatch-artifact`
**Branch**: `main`

### Summary

Carried the bounded RVV i32m1 add source seed through a selected
`tcrv.exec.dispatch` envelope and into the existing RVV EmitC/object/header/
bundle artifact path.

### Main Changes

- Created Trellis task `05-16-rvv-source-seed-selected-dispatch-artifact` from
  the supplied Direction Brief and repaired its PRD before implementation.
- Updated the RVV source-seed materializer to emit `@rvv`,
  `@scalar_fallback`, source-derived `@seed_rvv_i32_add`, conservative
  `@seed_scalar_fallback`, `tcrv.exec.case`, and `tcrv.exec.fallback`.
- Kept fallback as selection/envelope metadata only; no scalar fallback compute
  semantics were added.
- Generalized emission readiness to allow optional lowering-boundary candidates
  for dispatch-case references while keeping direct static selected markers
  boundary-required.
- Updated EmitC lowerable materialization to select the supported
  selected-emission diagnostic when the module also contains an unsupported
  fallback variant.
- Extended the RVV source-seed fixture to prove dispatch/case/fallback
  organization, emission-plan metadata, EmitC route materialization,
  object/header export, and bundle index selection.

### Testing

- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, and touched
  RVV/plugin/target C++ test executables.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door|i32m1-(add-object-artifact|selected-dispatch-artifact|sub-selected-dispatch-artifact|mul-selected-dispatch-artifact)|rvv-first-slice-materialization|emitc-to-cpp-handoff|toy-template-selected-boundary-seed`,
  18/18 passed.
- [OK] C++ tests:
  `tianchenrv-emission-readiness-test`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-dialect-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Changed-surface scans: no Python files changed; common/core transform
  diff has no RVV/scalar/descriptor/direct-C/source-export/Python compiler-core
  terms; RVV fixture intrinsic header assertion is the expected EmitC route
  evidence.
- [OK] New artifacts under
  `artifacts/tmp/source_seed_selected_dispatch_artifact/20260516T161432Z`.
- [OK] Real `ssh rvv` link/run:
  `tcrv_rvv_i32m1_source_selected_dispatch status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- Removed an accidental scalar-plugin static link dependency from the RVV seed
  implementation and replaced it with local fallback identity constants.
- Fixed FileCheck ordering in the RVV source-seed fixture.
- Relabeled the temporary RVV harness output to name this round's
  source-selected-dispatch evidence path.

### Spec Update Judgment

No spec update was needed. Existing variant-pipeline, EmitC route,
plugin-protocol, and MLIR testing specs already cover this bounded path.

### Status

[OK] Completed and archived.


## Session 96: RVV materialized EmitC target artifact bridge

**Date**: 2026-05-17
**Task**: RVV materialized EmitC target artifact bridge
**Branch**: `main`

### Summary

Rebuilt the bounded RVV selected target artifact bridge through provider-owned EmitC route materialization, MLIR EmitC C/C++ emission, clang RISC-V object packaging, focused checks, and ssh rvv compile/run evidence.

### Main Changes

- Rebuilt the RVV selected emission plan as a supported family-level route:
  `rvv-i32m1-arithmetic-emitc-route-family`.
- Registered one Target/RVV materialized EmitC target artifact exporter that
  consumes selected emission-plan candidates and provider-owned
  `TCRVEmitCLowerableRoute` payloads; old descriptor/table route ids and
  per-op target route tables remain absent.
- Reused the common selected EmitC artifact bridge to materialize verified
  MLIR EmitC, invoke the MLIR EmitC C/C++ emitter, and package a RISC-V RVV
  relocatable object with clang.
- Added RVV target artifact preflight for origin, emission kind, artifact kind,
  lowering boundary, runtime ABI identity, ordered ABI parameters, and
  `rvv_emitc_lowerable_route` provenance.
- Updated RVV/lowering-runtime specs so historical direct/descriptor routes
  remain deleted while the bounded explicit RVV i32m1 object route is the only
  active rebuilt artifact authority.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tcrv-opt tcrv-translate -j2`.
- [OK] Focused C++ tests:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-construction-protocol-common-test`.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|source-seed-target-artifact-object|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized|source-seed-artifact-front-door|toy-template-target-artifact`,
  14/14 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 101/101 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Targeted scans: no restored
  `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, old
  `getRVVI32M1Arithmetic*Target*RouteID`, old
  `tcrv-rvv-i32m1-{add,sub,mul}` target route strings, deleted-route
  diagnostic text, source-authority APIs, or direct C semantic exporter residue
  in touched code/test surfaces.
- [OK] Artifact evidence under
  `artifacts/tmp/rvv_materialized_emitc_target_artifact_bridge/20260516T174758Z-ok`.
  `llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- [OK] Real `ssh rvv` compile/run:
  `tcrv_rvv_materialized_emitc_target_artifact status=PASS n=4 add=[12,6,16,12]`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 97: TensorExtLite common EmitC route activation

**Date**: 2026-05-17
**Task**: TensorExtLite common EmitC route activation
**Branch**: `main`

### Summary

Activated one plugin-owned TensorExtLite explicit role sequence through the common TCRVEmitCLowerable EmitC materialization route, while keeping target artifact export unsupported.

### Main Changes

- Added TensorExtLite typed first-slice role ops for config, load_frag, tile_mma, and store_frag.
- Added a plugin-local TensorExtLite EmitC route provider that validates selected explicit role bodies, origin, role uniqueness, order, role interfaces, and call-opaque mapping.
- Rewired TensorExtLite emission readiness and emission-plan metadata from passive no-active-route diagnostics to the active common EmitC route.
- Added C++ and lit coverage for positive route materialization plus fail-closed missing body, stale no-active metadata, and target artifact export rejection.
- Updated the EmitC route spec with the TensorExtLite MLIR-EmitC-only base case and non-claim boundary.
- Checks: focused TensorExtLite/Toy/RVV plugin build and tests, focused TensorExtLite/Toy/RVV lit selection, full check-tianchenrv, git diff --check, and targeted residue scans for descriptor/direct-C/source-export/core TensorExtLite branch leakage.


### Git Commits

| Hash | Message |
|------|---------|
| `this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 98: Toy source-seed metadata authority erasure

**Date**: 2026-05-17
**Task**: Toy source-seed metadata authority erasure
**Branch**: `main`

### Summary

Deleted the Toy metadata-seed selected-boundary route, removed the Toy source-seed pass and positive seed tests, updated SourceSeed/Toy negative coverage, refreshed the source-seed spec, and verified focused plus full check-tianchenrv.

### Main Changes

- Removed the Toy source-seed public pass registration, build entry, header, and
  implementation for `tcrv-toy-materialize-template-selected-boundary-seed`.
- Deleted Toy lit tests that treated `tcrv_toy.lowering_seed =
  "template_compute"` as a positive selected-boundary, EmitC, or artifact
  source.
- Added negative coverage proving the deleted Toy pass option is unknown and
  stale Toy seed metadata fails closed through the common source-seed front
  door without materializing `tcrv_toy.compute_skeleton`.
- Updated SourceSeed/RVV-focused tests so RVV unseeded structural source
  materialization remains covered while Toy seed success is no longer required.
- Refreshed the variant-pipeline spec to state that plugins without an active
  source materializer contribute no source-seed boundaries.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-toy-extension-plugin-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] focused lit set for Toy deletion, SourceSeed fail-closed, RVV source
  materializer, and RVV source-seed target artifacts: 8/8 passed.
- [OK] `cmake --build build --target tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-plugin-registry-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused plugin lit set: 3/3 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 105/105 lit tests
  passed.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-toy-source-seed-authority-erasure`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 100: Extension bundle interface target-layer extraction

**Date**: 2026-05-17
**Task**: Extension bundle interface target-layer extraction
**Branch**: `main`

### Summary

Moved the generic extension bundle and bundle registry interface out of target
artifact export into the plugin/common layer, rewired plugin target-support
bundle hooks to use the rehomed type, and kept target artifact export as a
generic consumer of plugin-owned bundle registries.

### Main Changes

- Added `TianChenRV/Plugin/ExtensionBundle.h` and
  `lib/Plugin/ExtensionBundle.cpp` for `plugin::ExtensionBundle`,
  `plugin::ExtensionBundleRegistry`, and extension plugin registration callback
  ownership.
- Removed generic bundle class and callback definitions from
  `TargetArtifactExport.h`; target keeps exporter registries and exposes only a
  target-side helper to consume enabled extension bundles.
- Rewired built-in plugin catalog registration and all
  `configureTargetSupportExtensionBundle` overrides to use the plugin/common
  bundle interface.
- Removed the direct `TianChenRVTarget` link from `TianChenRVBuiltinPlugins`
  and added the bundle implementation to `TianChenRVPlugin`.
- Updated target artifact tests and plugin-protocol spec coverage for the new
  long-term bundle ownership contract.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test tcrv-translate tcrv-opt -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] focused lit filter for TensorExtLite target artifact, source-seed target
  artifact, EmitC handoff, Toy/Template target artifact, target artifact
  registry, and no-viable bundle front doors: 8/8 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 100/100 lit tests
  passed.
- [OK] `git diff --check`
- [OK] targeted scans proving `lib/Target/Builtin` has no concrete plugin
  catalog list, `TargetArtifactExport.h` no longer defines generic bundle
  classes, plugin/builtin no longer depends on target bundle types, and no
  descriptor/direct-C/source-export route authority was introduced in code
  diff.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-17-extension-bundle-interface-target-layer-extraction`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 102: RVV common materialized EmitC header adoption

**Date**: 2026-05-17
**Task**: RVV common materialized EmitC header adoption
**Branch**: `main`

### Summary

Migrated the selected RVV i32m1 materialized EmitC header artifact onto the
common target-layer `MaterializedEmitCHeaderArtifact` helper while preserving
the RVV object exporter and coherent object+header bundle composition.

### Main Changes

- Extended the common materialized EmitC header helper so object-backed
  selected materialized EmitC candidates can emit declaration-only headers
  when a route-local candidate validation callback owns the runtime ABI
  contract.
- Added dynamic runtime ABI identity mode for route families such as RVV
  add/sub/mul, where the selected candidate carries the op-specific ABI name
  and RVV-local preflight validates it.
- Replaced the RVV-local header declaration renderer and duplicated EmitC
  function boundary/arity checks with an RVV-local common header config plus
  `exportMaterializedEmitCHeaderArtifact`.
- Kept RVV-specific object packaging, runtime AVL/VL metadata validation,
  arithmetic route/op consistency checks, and bundle component metadata in RVV
  target support.
- Updated RVV header and bundle lit expectations to check common helper
  evidence: origin plugin, selected variant, selected route, runtime ABI
  kind/name, ordered ABI parameters, and required RVV provenance metadata.
- Updated the lowering-runtime spec for object-backed common header helper
  usage and dynamic runtime ABI identity preflight.

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused RVV lit filter for RVV target artifacts and source-seed
  materialization: 6/6 passed.
- [OK] focused Toy/TensorExtLite target artifact lit filter: 4/4 passed.
- [OK] `cmake --build build --target tianchenrv-toy-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] targeted scan over RVV target/plugin/translate/tests and common target
  files showed no stale RVV local header renderer and no descriptor/direct-C/
  source-export route authority beyond fail-closed rejection text and
  `CHECK-NOT` assertions.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 lit tests
  passed.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-17-rvv-common-materialized-emitc-header-adoption`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 99: RVV generated artifact bundle C ABI proof

**Date**: 2026-05-17
**Task**: RVV generated artifact bundle C ABI proof
**Branch**: `main`

### Summary

Made the selected RVV source-seed object/header bundle externally consumable as a true C ABI product and proved it with a C harness on ssh rvv.

### Main Changes

### Main Changes

- Added `emitExternC` to the common EmitC route materializer and enabled it for selected target artifact materialization so runtime-callable object exports define unmangled C symbols.
- Updated the common materialized EmitC header exporter to wrap runtime-callable declarations in C++ `extern "C"` guards while keeping the header valid C.
- Strengthened RVV source-seed target artifact lit coverage to assert the generated object exports the unmangled `tcrv_emitc_seed_kernel_seed_rvv_i32_add` symbol and that the header carries extern guards.
- Recorded the durable C ABI linkage rule in lowering-runtime and testing specs.
- Generated evidence under `artifacts/tmp/rvv_generated_artifact_bundle_link_run_abi_proof/20260517T014226Z/` with bundle artifacts, harness source, command record, local readobj/nm output, and `ssh rvv` compile/link/run log.

### Evidence

- Remote command compiled a C harness with `clang -O2 -march=rv64gcv -mabi=lp64d`, linked the generated bundle object, and ran on `ssh rvv`.
- Remote output: `PASS tcrv_rvv_i32m1_add_bundle_c_abi n=257`.
- Local `llvm-nm -g` output for the generated object contains unmangled `tcrv_emitc_seed_kernel_seed_rvv_i32_add`.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit filter for source-seed RVV target artifacts plus Toy/TensorExtLite target artifact headers: 4/4 passed.
- [OK] `cmake --build build --target tianchenrv-emitc-lowerable-interface-test tianchenrv-toy-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [OK] `./build/bin/tianchenrv-emitc-lowerable-interface-test && ./build/bin/tianchenrv-toy-extension-plugin-test && ./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `git diff --check`
- [OK] targeted scans showed no descriptor/direct-C/source-export/legacy RVV route authority introduced; remaining hits are rejection text or CHECK-NOT assertions.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 lit tests passed.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-17-rvv-generated-artifact-bundle-link-run-abi-proof`

### Status

[OK] Completed and archived.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 100: TensorExtLite source-front-door construction template

**Date**: 2026-05-17
**Task**: TensorExtLite source-front-door construction template
**Branch**: `main`

### Summary

Added a TensorExtLite-owned source-front-door pass and registry hook that materializes the fragment-MMA source marker into selected TensorExtLite role ops, a selected lowering-boundary marker, emission-plan provenance, and the existing declaration-only header artifact route; updated plugin/lowering specs and focused C++/lit coverage.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
