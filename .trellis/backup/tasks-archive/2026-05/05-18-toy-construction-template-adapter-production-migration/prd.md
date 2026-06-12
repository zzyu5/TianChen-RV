# Toy construction-template adapter production migration

## Goal

Migrate the Toy target-support production object/header/bundle exporters from
Toy-local materialized EmitC config assembly to the common
`ConstructionTemplateArtifactAdapterConfig` and
`registerConstructionTemplateArtifactAdapterExporters` path.

This round makes Toy the remaining concrete proof that the construction-template
adapter is the default reusable target-artifact surface for simple plugins,
while Toy keeps plugin-owned construction protocol, EmitC route, validation, and
object packaging callbacks.

## Background

Recent commits made Template, RVV, and TensorExtLite consume the common
construction-template target artifact adapter for their production object,
header, and bundle paths. Current Toy target support still locally builds
`MaterializedEmitCHeaderArtifactConfig` and
`MaterializedEmitCObjectBundleArtifactConfig`, directly calls
`emitSelectedEmitCArtifactCppSource`, and directly registers
`registerMaterializedEmitCObjectBundleArtifactExporters`.

That leaves the sample/simple extension path partially hand-specialized instead
of demonstrating the reusable construction-template target artifact surface.

## Scope

- Rewire the default Toy target-support bundle so object, declaration-only
  header, and coherent object/header bundle exporters are registered through
  the common construction-template adapter.
- Build one Toy-owned `ConstructionTemplateArtifactAdapterConfig` carrying Toy
  construction protocol metadata, archetype/semantic role graph evidence,
  source op/role/interface provenance, EmitC route metadata, runtime ABI
  metadata, selected-boundary validation, header metadata evidence, and the Toy
  object packager callback.
- Delete or rewrite obsolete Toy-local materialized EmitC header config,
  object-bundle config, direct C++ source export composition, and direct bundle
  exporter registration.
- Preserve Toy's existing object/header/bundle route identifiers and default
  plugin exporter bundle registration.
- Keep Toy-local validation and object packaging callbacks local to Toy.

## Non-Goals

- No new Toy dialect feature, Toy computation semantic, Toy operation, runtime
  claim, correctness claim, or performance claim.
- No new extension plugin or executable family.
- No RVV, TensorExtLite, Template, IME, Offload, or source-front-door behavior
  broadening.
- No descriptor-driven computation, descriptor compatibility layer, direct C
  semantic exporter, source-export route, legacy mode, or fallback old Toy path.
- No Python compiler-core behavior.
- No extension-specific semantic branch in common/core target code.

## Requirements

1. Toy production/default object/header/bundle exporters must use
   `ConstructionTemplateArtifactAdapterConfig` and
   `registerConstructionTemplateArtifactAdapterExporters`.
2. Toy target support must no longer locally build
   `MaterializedEmitCHeaderArtifactConfig` or
   `MaterializedEmitCObjectBundleArtifactConfig`.
3. Toy object export must call
   `exportConstructionTemplateObjectArtifact`; Toy header export must call
   `exportConstructionTemplateHeaderArtifact`.
4. Toy target support must no longer directly call
   `emitSelectedEmitCArtifactCppSource` or
   `registerMaterializedEmitCObjectBundleArtifactExporters`.
5. Toy adapter config must include route/source-op metadata, runtime ABI
   metadata, selected variant identity, semantic role graph evidence, typed role
   realization evidence, header evidence, component group, handoff kind, and
   the Toy object packager callback.
6. Invalid Toy candidates must fail closed for stale route/source-front-door
   metadata, missing construction protocol/route/role/runtime ABI evidence,
   missing packager or route-local validator, fallback-only roles, mixed plugin
   origin, malformed selected boundary, and forbidden metadata that attempts to
   treat descriptor/direct-C/source-export/compute-body metadata as compute
   authority.
7. Targeted scans must show no Toy-local local materialized EmitC config
   assembly or direct source/bundle exporter calls remain under Toy target
   support, plugin, translate, or tests, except common adapter references and
   negative test text where applicable.

## Acceptance Criteria

- [ ] Toy object/header/bundle exporters are registered through
      `registerConstructionTemplateArtifactAdapterExporters`.
- [ ] Toy target support contains one Toy adapter config and no Toy-local
      `MaterializedEmitCHeaderArtifactConfig` or
      `MaterializedEmitCObjectBundleArtifactConfig` builders.
- [ ] Toy target support no longer directly calls
      `emitSelectedEmitCArtifactCppSource`.
- [ ] Toy target support no longer directly calls
      `registerMaterializedEmitCObjectBundleArtifactExporters`.
- [ ] Existing Toy object/header/bundle route IDs remain coherent and registered
      by the default Toy exporter bundle.
- [ ] Focused C++ tests cover Toy adapter config validation, Toy default
      exporter registration, object/header/bundle selection, and fail-closed
      malformed candidates.
- [ ] Focused lit or existing target-artifact tests cover Toy object/header/
      bundle behavior if present.
- [ ] Focused checks and residue scans pass, or any failure is recorded as a
      new-architecture adapter gap without restoring old Toy compatibility.

## Expected Evidence

- `python3 ./.trellis/scripts/task.py validate ...`
- Focused C++ build for `tcrv-opt`, `tcrv-translate`, and target/plugin test
  binaries affected by the migration.
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused Toy/template plugin tests affected by the migration.
- Focused lit/FileCheck filter for Toy and target artifact bundle export tests
  if available.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` if practical.
- Targeted residue scans over Toy target/plugin/translate/tests for deleted
  local config builders and direct source/exporter calls, plus descriptor/
  direct-C/source-export residue.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/tasks/archive/2026-05/05-18-tensorextlite-construction-template-adapter/prd.md`
- `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h`
- `lib/Target/ConstructionTemplateArtifactAdapter.cpp`
- `lib/Target/Toy/ToyTargetSupportBundle.cpp`
- `lib/Target/Template/TemplateTargetSupportBundle.cpp`
- `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`
- `test/Target/TargetArtifactExportTest.cpp`

## Definition of Done

- Implementation follows the Trellis specs and keeps compiler implementation in
  the C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck stack.
- Obsolete Toy target-support code is removed instead of preserved as a parallel
  path.
- Checks and targeted scans are recorded in the final report.
- Task status is updated truthfully; if complete, the Trellis task is finished
  and archived.
- One coherent commit is created for the completed round.

## Completion Notes

- Rewired `lib/Target/Toy/ToyTargetSupportBundle.cpp` so Toy production object,
  declaration-only header, and coherent object/header bundle exporters consume
  `ConstructionTemplateArtifactAdapterConfig` and
  `registerConstructionTemplateArtifactAdapterExporters`.
- Deleted Toy-local `MaterializedEmitCHeaderArtifactConfig`,
  `MaterializedEmitCObjectBundleArtifactConfig`, direct
  `emitSelectedEmitCArtifactCppSource` object export composition, and direct
  `registerMaterializedEmitCObjectBundleArtifactExporters` registration.
- Preserved Toy-local construction protocol verification, selected direct
  variant validation, plugin-owned EmitC route building, and the local clang++
  relocatable object packager callback.
- Extended Toy artifact metadata evidence so the adapter config and selected
  emission plan carry Toy construction protocol, extension archetype, semantic
  role graph, common interface realization, typed role realization, EmitC route
  mapping, source op/role/interface provenance, runtime ABI fields, and evidence
  profile.
- Added focused C++ coverage for Toy construction-template adapter config
  validation, common adapter exporter registration, object/header candidate
  validation, object-backed header matching, missing packager, missing
  route-local validator, fallback-only role, mixed plugin origin, malformed
  lowering boundary, missing construction protocol/source-role/runtime ABI
  metadata, and forbidden source-export compute metadata.
- Updated focused Toy plugin and Toy materialized artifact tests to assert the
  expanded construction-template evidence surface.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-toy-construction-template-adapter-production-migration`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-toy-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`
- `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-toy-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Toy|toy|TargetArtifactBundleExport'` passed 16/123 selected tests.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 123/123 tests.

## Residue Scans

- Toy target/plugin/test scan for `MaterializedEmitCHeaderArtifactConfig`,
  `MaterializedEmitCObjectBundleArtifactConfig`,
  `emitSelectedEmitCArtifactCppSource`, and
  `registerMaterializedEmitCObjectBundleArtifactExporters` returned no matches.
- Toy descriptor/direct-C/source-export residue scan found only negative
  FileCheck / implicit-check-not assertions in Toy and target artifact tests.
- Common adapter family-branch scan over
  `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h` and
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp` returned no Toy/RVV/
  TensorExt/IME/Offload names and no descriptor/direct-C/source-export/
  compute-body/intrinsic/fragment residue.

## Spec Update Review

No `.trellis/spec/` edit was required. The existing lowering-runtime
construction-template adapter contract already covers Toy consuming the common
object-backed header and bundle helper after its selected typed role-op path
materializes through the plugin-owned EmitC route.
