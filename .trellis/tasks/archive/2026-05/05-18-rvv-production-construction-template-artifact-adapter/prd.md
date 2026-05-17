# RVV production construction-template artifact adapter consumption

## Goal

Migrate RVV target-support object/header/bundle registration and export
plumbing onto the production `ConstructionTemplateArtifactAdapter`, making RVV
the first real hardware extension-family consumer of the common
construction-template artifact workflow while preserving the current selected
RVV materialized EmitC route and callable artifact ABI.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Session-start `git status --short` was clean.
- Current HEAD at session start is
  `15ef25b plugin: productize construction template artifact adapter`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Direction Brief before source edits.
- Previous task
  `.trellis/tasks/archive/2026-05/05-18-production-construction-template-artifact-adapter/prd.md`
  productized `ConstructionTemplateArtifactAdapter` and rewired Template target
  support to consume it.
- Current RVV target support still hand-builds selected EmitC route config,
  materialized header config, object/header bundle config, object export, and
  header export locally in `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- RVV differs from Template because the selected arithmetic route owns dynamic
  runtime ABI identity (`add/sub/mul` ABI names) and accepts source-front-door
  selected variant names, so the common adapter must preserve dynamic candidate
  identity instead of forcing a static variant or runtime ABI name.

## Requirements

- Rewire RVV target support so object/header/bundle exporters register through
  `ConstructionTemplateArtifactAdapter`.
- RVV target support must construct one adapter config from
  `RVVConstructionManifest`, `RVVI32M1ArithmeticTargetArtifactMapping`, RVV
  candidate validation, RVV EmitC route builder, RVV metadata evidence, and the
  RVV RISC-V clang object packager callback.
- Keep RVV route-local authority only for RVV candidate preflight, RVV
  construction/runtime metadata validation, RVV manifest and target mapping
  values, RVV EmitC route construction, and RISC-V object packaging.
- Collapse or delete the bespoke RVV header/object/bundle config and exporter
  composition now duplicated by the adapter.
- Preserve the selected RVV materialized EmitC route id, header route id,
  object artifact kind, header artifact kind, origin plugin, emission kind,
  lowering boundary, runtime ABI kind, runtime glue role, ordered ABI
  parameters, bundle component group, handoff kind, and existing fail-closed
  metadata rejection semantics.
- Extend the production adapter only in an extension-agnostic way if needed
  for RVV dynamic selected variant and dynamic runtime ABI identity. Do not add
  RVV branches or intrinsic knowledge to common target code.
- Preserve RVV object packaging through the MLIR EmitC C/C++ emitter output and
  clang RISC-V relocatable object callback.

## Acceptance Criteria

- RVV object/header/bundle exporter registration calls
  `registerConstructionTemplateArtifactAdapterExporters`.
- RVV object and header export callbacks call
  `exportConstructionTemplateObjectArtifact` and
  `exportConstructionTemplateHeaderArtifact`.
- The adapter supports RVV dynamic selected variant and runtime ABI identity
  without letting common target code infer RVV semantics.
- Positive RVV source-seed/materialized artifact tests still produce object,
  header, and bundle outputs tied to the selected variant, origin plugin,
  materialized EmitC route, runtime ABI name, and ordered ABI parameters.
- Negative RVV tests still reject fallback-only selections, missing
  materialized EmitC provenance, mismatched runtime ABI parameters, stale
  descriptor/direct-C/source-export metadata, wrong route identity, missing
  runtime AVL/VL evidence, and multiple selected supported candidates.
- Focused scans show no descriptor route authority, no direct-C semantic
  exporter, no source-export route, no stale compatibility route id, and no
  duplicate bespoke RVV adapter surface left behind except route-local
  validator and object packager callback.
- `git diff --check` passes.

## Definition Of Done

- PRD and task context describe this round truthfully.
- RVV production target-support artifact registration/export plumbing consumes
  the common construction-template adapter.
- Focused build/tests for RVV target artifact export and touched target/plugin
  surfaces pass, or blockers are recorded precisely.
- Focused lit for RVV object/header/bundle and materialized EmitC handoff
  surfaces passes.
- `ssh rvv` object/header/bundle evidence is refreshed if the remote is
  reachable and the local generated bundle is available.
- Trellis task is finished/archived when complete.
- One coherent commit is created if the task is complete.

## Out Of Scope

- New RVV arithmetic coverage, new SEW/LMUL families, generic RVV lowering, or
  high-level frontend lowering.
- New route IDs, descriptor adapters, legacy wrappers, source-export routes,
  direct C compute-body generation, scalar fallback compute, or Python
  compiler-core behavior.
- Common/core RVV semantic branches, common target knowledge of RVV intrinsic
  names, or adapter family-specific switches.
- Runtime correctness or performance claims without real `ssh rvv` evidence.

## Technical Approach

Make `ConstructionTemplateArtifactAdapterConfig` express the already-supported
lower-level dynamic modes from `MaterializedEmitCHeaderArtifactConfig`:
dynamic selected variant by leaving `selectedVariant` empty, and dynamic
runtime ABI identity via an explicit adapter flag forwarded to the header
config. Then replace RVV's bespoke materialized header config, object bundle
config, and direct common-helper calls with one RVV adapter config. Keep the
RVV candidate validator and clang RISC-V object packager in RVV target support.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous adapter task PRD read:
  `.trellis/tasks/archive/2026-05/05-18-production-construction-template-artifact-adapter/prd.md`.
- Relevant journal entries read:
  `.trellis/workspace/codex/journal-9.md` Session 121 production adapter notes
  and RVV materialized EmitC/target artifact notes.
- Main source surfaces inspected before implementation:
  `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h`,
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp`,
  `lib/Target/Template/TemplateTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`, and `test/Target/RVV/`.

## Completion Notes

- Extended the production `ConstructionTemplateArtifactAdapterConfig` with
  `allowDynamicRuntimeABIIdentity` and forwarded it to the common
  materialized EmitC header/object bundle helper.
- Relaxed adapter-level static `selectedVariant`, `runtimeABI`,
  `runtimeABIName`, and `externalABIName` requirements so callers can consume
  the lower-level dynamic selected-candidate identity already supported by the
  common target artifact layer.
- Rewired RVV target support to build one
  `ConstructionTemplateArtifactAdapterConfig` from the RVV construction
  manifest, RVV target artifact mapping, RVV selected candidate validator, RVV
  metadata evidence, RVV EmitC route builder, and the RISC-V clang object
  packager callback.
- Deleted RVV-local construction of
  `MaterializedEmitCHeaderArtifactConfig`,
  `MaterializedEmitCObjectBundleArtifactConfig`, direct
  `emitSelectedEmitCArtifactCppSource` object export composition, and direct
  `registerMaterializedEmitCObjectBundleArtifactExporters` registration.
- Preserved RVV route-local validation for selected route identity,
  fallback-only rejection, runtime ABI name/kind/glue/ordered parameters,
  construction metadata, runtime AVL/VL evidence, forbidden
  descriptor/direct-C/source-export/compute-body metadata, and multiple
  selected supported candidates.
- No descriptor adapter, direct-C/source-export compatibility path, Python
  compiler-core behavior, common RVV semantic branch, or common target
  knowledge of RVV intrinsic names was added.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  construction-template adapter dynamic identity contract.

## Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-production-construction-template-artifact-adapter`
- `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- Initial lit attempt from the source test directory failed because
  `tianchenrv_obj_root` is only injected by the build test config; reran from
  `build/test`.
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Target/RVV|RVV'` passed 30/123 selected tests.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- RVV duplicate bespoke adapter surface scan:
  `rg -n "MaterializedEmitCHeaderArtifactConfig|MaterializedEmitCObjectBundleArtifactConfig|registerMaterializedEmitCObjectBundleArtifactExporters|exportMaterializedEmitCHeaderArtifact|emitSelectedEmitCArtifactCppSource|getRVVI32M1ArithmeticHeaderArtifactConfig|getRVVI32M1ArithmeticObjectBundleConfig" lib/Target/RVV include/TianChenRV/Target/RVV`
  returned no matches.
- Common adapter family-branch scan over
  `include/TianChenRV/Target/ConstructionTemplateArtifactAdapter.h` and
  `lib/Target/ConstructionTemplateArtifactAdapter.cpp` returned no concrete
  family-name matches.
- Forbidden-residue scan over RVV target support, common adapter,
  `tools/tcrv-translate`, and `test/Target/RVV` found only RVV fail-closed
  metadata rejection strings and `implicit-check-not` negative assertions in
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` passed 123/123 tests.
- Spec update review performed via `trellis-update-spec`; the API signature
  change required a targeted lowering-runtime spec update.
- First `rvv_generated_bundle_abi_e2e.py` run was blocked by missing
  `llvm-readobj` on PATH; reran with `/usr/lib/llvm-20/bin/llvm-readobj`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind add --runtime-count 7 --runtime-count 16 --run-id rvv-adapter-consumption-add-20260518 --overwrite --timeout 120 --connect-timeout 10 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  passed with real `ssh rvv` evidence.
- RVV evidence artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/rvv-adapter-consumption-add-20260518/add/evidence.json`.
  It records `status=success`, `ssh_evidence=true`,
  `remote_compile_succeeded=true`, `remote_run_succeeded=true`,
  `ssh_target=rvv`, selected variant `vector_source_rvv_i32_add`,
  runtime ABI `rvv-i32m1-add-callable-c-abi.v1`, unmangled object symbol
  `tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add`, and remote
  output `PASS op=add counts=7,16`.
