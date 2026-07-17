# TensorExtLite production construction-template artifact adapter consumption

## Goal

Migrate the existing TensorExtLite target object/header/bundle artifact workflow to the common `ConstructionTemplateArtifactAdapterConfig` and `registerConstructionTemplateArtifactAdapterExporters` production path, then delete the TensorExtLite-local materialized EmitC header/object-bundle config assembly and direct source/object/header helper layer that the common adapter replaces.

This round proves the construction-template artifact adapter is a reusable extension-family target surface, not an RVV-only cleanup.

## Background

Commit `56e8407` made RVV consume the production construction-template adapter for object/header/bundle packaging while preserving RVV-local validation, manifest mapping, EmitC route building, and RISC-V object packaging. Bounded inspection in the task brief reports that TensorExtLite still has the corresponding one-off pattern in `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`: local `MaterializedEmitCHeaderArtifactConfig`, local `MaterializedEmitCObjectBundleArtifactConfig`, direct `emitSelectedEmitCArtifactCppSource` object/header helpers, and direct `registerMaterializedEmitCObjectBundleArtifactExporters` registration.

## Scope

- Rewire the default TensorExtLite target-support bundle so object, declaration-only header, and coherent object-bundle exporters are registered through the common construction-template artifact adapter.
- Keep TensorExtLite-local ownership only for plugin validation, construction/runtime metadata, role-sequence provenance, EmitC route construction, source-front-door handling, and any route-local object packager callback.
- Delete or rewrite obsolete TensorExtLite-local target-support assembly that duplicates the common adapter.
- Preserve positive source-front-door and already-materialized role-sequence export behavior through the same selected TensorExtLite candidate.
- Preserve fail-closed behavior for invalid or mixed candidate surfaces.

## Non-Goals

- No new TensorExtLite feature, tensor/linalg/tile lowering, vendor runtime, performance claim, or hardware evidence claim.
- No new RVV coverage or RVV behavior change.
- No new generic lowering pass.
- No compatibility wrapper, legacy mode, descriptor adapter, or parallel old TensorExtLite bespoke path.
- No descriptor-driven computation, direct C/source-export semantic route, Python compiler-core behavior, or common/core branching on TensorExtLite/RVV/Toy names.

## Requirements

1. The production/default TensorExtLite target-support bundle must call the common adapter registration path for object/header/bundle exporters.
2. The common adapter must stay family-neutral and must not learn TensorExtLite role semantics, fragment layout, intrinsic/runtime names, descriptor strings, or compute branches.
3. TensorExtLite must provide adapter configuration through bounded callbacks and family-owned validation/metadata hooks rather than bespoke target-support exporter assembly.
4. The selected TensorExtLite boundary must drive object, header, and bundle artifacts from one coherent selected candidate.
5. Source-front-door inputs and already-materialized TensorExtLite role-sequence inputs must both reach export through the selected materialized EmitC route.
6. Invalid inputs must fail closed, including disabled built-ins, malformed source marker, stale selected-boundary residue, missing or reordered role ops, missing route provenance, missing runtime ABI metadata, descriptor/direct-C/source-export residue, and mixed RVV/Toy/TensorExtLite candidate surfaces.
7. Targeted scans must show no remaining TensorExtLite-local uses of `MaterializedEmitCHeaderArtifactConfig`, `MaterializedEmitCObjectBundleArtifactConfig`, direct `registerMaterializedEmitCObjectBundleArtifactExporters`, or direct `emitSelectedEmitCArtifactCppSource` target-support path outside the common adapter and allowed non-semantic EmitC translation helpers.

## Acceptance Criteria

- [ ] TensorExtLite object/header/bundle exporters are registered through `registerConstructionTemplateArtifactAdapterExporters` in the default target-support bundle.
- [ ] TensorExtLite-local materialized EmitC header/object-bundle config assembly and direct source/object/header helper layer are deleted or rewritten.
- [ ] Positive C++ tests cover TensorExtLite exporter registration, adapter config validation, and object/header/bundle candidate validation.
- [ ] Negative C++ tests cover fail-closed metadata/provenance/role-order and mixed-surface cases relevant to the migrated path.
- [ ] Focused lit/FileCheck tests cover TensorExtLite source-front-door and already-materialized role-sequence inputs reaching object/header/bundle export through the selected materialized EmitC route.
- [ ] Focused checks pass or any remaining failure is recorded as a missing new-architecture gap rather than restored old-path compatibility.

## Expected Evidence

- Relevant TensorExtLite plugin tests.
- Target artifact export tests.
- Focused lit filters for TensorExtLite and target artifact paths.
- `git diff --check`.
- `check-tianchenrv` if practical for this round.
- Targeted residue scans over TensorExtLite target/plugin/translate/tests for deleted bespoke symbols and direct-exporter paths.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/tasks/archive/2026-05/05-18-rvv-production-construction-template-artifact-adapter/prd.md`
- `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h`
- `lib/Target/ConstructionTemplateArtifactAdapter.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `include/TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h`
- `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`
- `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`
- `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`
- `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`
- `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`
- `test/Target/TargetArtifactExportTest.cpp`
- Relevant TensorExtLite lit tests under `test/Target` and `test/Transforms`.

## Definition of Done

- Implementation follows the Trellis specs and keeps compiler implementation in the C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck stack.
- Obsolete TensorExtLite target-support code is removed instead of preserved as a parallel path.
- Checks and targeted scans are recorded in the final report.
- Task status is updated truthfully; if complete, the Trellis task is finished/archived and one coherent commit is created.

## Completion Notes

- Rewired `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp` so the default TensorExtLite object/header/bundle exporter registration consumes `ConstructionTemplateArtifactAdapterConfig` and `registerConstructionTemplateArtifactAdapterExporters`.
- Replaced TensorExtLite-local materialized EmitC header config, object/header bundle config, direct `emitSelectedEmitCArtifactCppSource` object/C++ export composition, and direct `registerMaterializedEmitCObjectBundleArtifactExporters` registration with common adapter calls.
- Preserved TensorExtLite-owned validation for stale source-front-door metadata, selected lowering-boundary conformance, construction protocol metadata, role-sequence provenance, runtime ABI metadata, and the local clang RISC-V object packager callback.
- Added focused C++ coverage that builds a TensorExtLite construction-template adapter config from real TensorExtLite protocol constants, validates the config, registers object/header exporters through the common adapter, validates object/header candidates, and rejects missing packager, missing route-local validator, fallback-only role, and mixed plugin origin.
- Spec-update review: no `.trellis/spec/` edit was required because `.trellis/spec/lowering-runtime/emitc-route.md` already contains the durable construction-template adapter contract, including RVV/TensorExtLite consumption, object-backed header composite behavior, dynamic/static runtime ABI identity, and fail-closed residue rules.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-tensorextlite-construction-template-adapter`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'TensorExtLite|tensorext-lite|TargetArtifactBundleExport'` passed 17/123 selected tests.
- `cmake --build build --target check-tianchenrv -j2` passed 123/123 tests.
- `git diff --check`

## Residue Scans

- TensorExtLite target/plugin/translate/tests scan for `MaterializedEmitCHeaderArtifactConfig`, `MaterializedEmitCObjectBundleArtifactConfig`, `registerMaterializedEmitCObjectBundleArtifactExporters`, and `emitSelectedEmitCArtifactCppSource` returned no matches.
- Common adapter family-branch scan over `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h` and `lib/Target/ConstructionTemplateArtifactAdapter.cpp` returned no TensorExtLite/RVV/Toy/IME/Offload, descriptor, direct-C, source-export, compute-body, fragment, or intrinsic matches.
- TensorExtLite target/plugin/test forbidden-residue scan found only negative FileCheck assertions and source-front-door stale-residue rejection text.
