# RVV vector-source front-door artifact bridge

## Goal

Prove the bounded RVV source-front-door to materialized target-artifact bridge
for one existing source-level `i32m1` vector arithmetic pattern, using the
current production/default path:

```text
source MLIR func/scf/vector/arith
  -> RVV plugin source front door
  -> selected tcrv.exec + tcrv_rvv boundary
  -> RVV plugin-owned EmitC route
  -> construction-template object/header/bundle exporters
  -> local object/header/bundle checks
  -> ssh rvv ABI harness evidence
```

The smallest bounded positive path is the existing add fixture. Sub and mul may
remain broader parity coverage, but this round's required proof is one coherent
add path from source to target artifacts and real RVV execution evidence.

## Current Repo Evidence

- Current HEAD already contains `RVVVectorSourceFrontDoor.cpp`, which accepts
  source-level `func` / `scf.for` / `vector.load` /
  `arith.addi|subi|muli` / `vector.store` inputs and materializes selected
  `tcrv.exec` and `tcrv_rvv` bodies.
- The RVV plugin already registers
  `tcrv-rvv-materialize-i32m1-vector-source-front-door` and participates in the
  shared `tcrv-source-artifact-front-door-pipeline`.
- `RVVEmitCRouteProvider.cpp` already builds the bounded RVV i32m1
  materialized EmitC route from explicit `tcrv_rvv` ops and ABI roles.
- `RVVTargetSupportBundle.cpp` already registers the RVV object, declaration
  header, and coherent object/header bundle routes through
  `ConstructionTemplateArtifactAdapterConfig`.
- Existing materialized-target tests start from selected/materialized RVV IR,
  while source-front-door target bundle tests cover a one-command bundle path.
  The remaining acceptance gap for this task is focused source-level target
  artifact lit coverage and a fresh artifact/ssh evidence directory for the
  selected add path.

## Scope

- Add focused lit coverage that starts from source-level RVV add MLIR and runs
  the existing source-artifact pipeline into:
  - `tcrv-translate --tcrv-export-target-artifact`;
  - `tcrv-translate --tcrv-export-target-header-artifact`;
  - `tcrv-translate --tcrv-export-target-artifact-bundle`.
- Assert selected RVV variant identity, route, runtime ABI ordering, runtime
  ABI metadata, materialized EmitC provenance, object symbol, header metadata,
  bundle index coherence, and absence of descriptor/direct-C/source-export
  residue.
- Keep existing production/default code path intact unless focused checks expose
  a real defect.
- Produce a small artifact directory for the add path with source MLIR,
  selected/materialized plan, emitted header, bundle index, object readobj
  output, generated harness, and ssh RVV compile/run evidence.

## Non-Goals

- No new RVV dtype, SEW, LMUL, shape, source language, performance matrix, or
  generic vector lowering.
- No descriptor adapter, descriptor-driven computation, direct C semantic
  exporter, source-export route, compatibility wrapper, or legacy route id.
- No Python compiler-core behavior. Python evidence tooling may only drive
  generated artifact checks and ssh RVV execution.
- No new extension family and no core/shared RVV semantic branch.
- No header or bundle route may synthesize RVV compute bodies; headers remain
  declarations and bounded metadata only.

## Requirements

1. The source-level add fixture must reach target object export through
   `tcrv-opt --tcrv-source-artifact-front-door-pipeline` piped to
   `tcrv-translate --tcrv-export-target-artifact`.
2. The same source-level add fixture must reach declaration-only header export
   through the same source-artifact pipeline and
   `--tcrv-export-target-header-artifact`.
3. The same source-level add fixture must reach coherent bundle export through
   the same source-artifact pipeline and
   `--tcrv-export-target-artifact-bundle`.
4. The tests must prove the selected variant is
   `@vector_source_rvv_i32_add`, the route is
   `rvv-i32m1-arithmetic-emitc-route-family`, the runtime ABI is
   `rvv-i32m1-add-callable-c-abi.v1`, and the ABI parameter order is
   `lhs, rhs, out, n`.
5. Object checks must prove a RISC-V relocatable object and the unmangled
   callable symbol:
   `tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add`.
6. Header and bundle checks must reject descriptor, direct-C, source-export,
   RVV direct microkernel, intrinsic, callable body, and self-check residue.
7. Negative coverage for stale seed, pre-existing `tcrv.exec`/`tcrv_rvv`
   residue, wrong ABI order, unsupported dtype/shape/op, fallback-only
   selection, missing materialized EmitC provenance, stale runtime ABI order,
   descriptor-like metadata, disabled plugins, and no matching source
   materialization must be present in existing focused tests or added if absent.
8. Fresh evidence must include local artifact generation and real `ssh rvv`
   compile/run output for the generated object/header/harness.

## Acceptance Criteria

- [ ] Source-level add MLIR exports a RISC-V relocatable object through the
      source-artifact pipeline and generic target artifact exporter.
- [ ] Source-level add MLIR exports a declaration-only callable header through
      the source-artifact pipeline and generic header exporter.
- [ ] Source-level add MLIR exports a coherent object/header bundle through the
      source-artifact pipeline and generic bundle exporter.
- [ ] Focused lit checks assert selected variant, route, runtime ABI,
      construction protocol metadata, materialized EmitC provenance, object
      symbol, header metadata, bundle index fields, and forbidden residue
      absence.
- [ ] Existing negative coverage covers the required fail-closed cases without
      adding descriptor/direct-C compatibility behavior.
- [ ] A fresh artifact directory records source, selected/materialized plan,
      generated artifacts, readobj output, harness, and ssh RVV evidence.
- [ ] Focused C++/lit checks, `git diff --check`, targeted residue scans, and
      `check-tianchenrv` if practical are run and recorded.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-05-18-rvv-vector-source-artifact-bridge`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- Focused lit filter covering RVV source front door, RVV target artifact, and
  source artifact bundle front door tests.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind add ...`
- `git diff --check`
- Targeted scans over touched RVV/plugin/target/test surfaces for descriptor,
  direct-C semantic exporter, source-export route, legacy route ids, and core
  extension-specific branches.
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/tasks/archive/2026-05/05-18-toy-construction-template-adapter-production-migration/prd.md`
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `lib/Target/ConstructionTemplateArtifactAdapter.cpp`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`
- `test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir`
- `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`
- `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`
- `test/Target/TargetArtifactExportTest.cpp`

## Definition of Done

- The proof follows the current C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck
  stack and does not add Python compiler-core behavior.
- Obsolete descriptor/direct-C/source-export logic is not restored.
- The Trellis task status, completion notes, checks, and artifact evidence are
  recorded truthfully.
- The task is finished and archived when complete.
- One coherent commit is created.

## Completion Notes

- Confirmed the production RVV path already exists in current HEAD:
  source-level vector MLIR is materialized by the RVV plugin source front door,
  planned through the shared source-artifact front-door pipeline, routed through
  the RVV plugin-owned EmitC route, and consumed by the RVV
  construction-template object/header/bundle target exporters.
- Added `test/Target/RVV/vector-source-target-artifact-exporters.mlir` so the
  source-level add fixture directly proves all three target exporter surfaces:
  relocatable object, declaration-only header, and coherent object/header
  bundle.
- The new lit coverage asserts `@vector_source_rvv_i32_add`,
  `rvv-i32m1-arithmetic-emitc-route-family`,
  `rvv-i32m1-add-callable-c-abi.v1`, ABI parameter order
  `lhs, rhs, out, n`, RVV construction protocol metadata, materialized EmitC
  provenance, object symbol
  `tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add`, header metadata,
  bundle index fields, and absence of descriptor/direct-C/source-export/RVV
  direct microkernel residue.
- Generated fresh evidence under
  `artifacts/tmp/rvv_vector_source_frontdoor_artifact_bridge/20260518T-rvv-vector-source-frontdoor-bridge-add`.
  The evidence directory contains the source MLIR, selected/materialized plan,
  generated object/header/bundle, bundle index, readobj outputs, harness, and
  real `ssh rvv` compile/run output for runtime counts 7, 16, and 23.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-05-18-rvv-vector-source-artifact-bridge`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVV.*vector-source|vector-source-target-artifact|source-artifact-bundle-front-door-rvv|source-artifact-bundle-front-door-fail-closed'` passed 7/124 selected tests.
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -av . --filter 'vector-source-target-artifact-exporters'` passed 1/124 selected tests and showed the exact object/header/bundle commands.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_vector_source_frontdoor_artifact_bridge --run-id 20260518T-rvv-vector-source-frontdoor-bridge-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `git diff --check`
- Targeted residue scan over RVV plugin/target/test surfaces found only
  negative FileCheck assertions and RVV target fail-closed rejection text for
  descriptor/direct-C/source-export metadata; no restored direct exporter or
  legacy RVV route authority was added.
- Common/core branch scan over transform, target artifact, construction
  adapter, and translate surfaces returned no new RVV semantic branches.
- `cmake --build build --target check-tianchenrv -j2` passed 124/124 tests.

## Spec Update Review

No `.trellis/spec/` edit was required. The existing RVV plugin,
lowering-runtime, EmitC route, plugin registry, and variant-pipeline specs
already require this source-artifact bridge shape and its lit/evidence
coverage. This round added the missing focused proof and fresh evidence rather
than changing a durable contract.
