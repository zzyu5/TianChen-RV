# RVV construction-template metadata closure on real hardware path

## Goal

Close the existing bounded RVV i32m1 arithmetic materialized EmitC
object/header/bundle route against the executable extension-family
construction template proven by TensorExtLite. The production path must carry
one coherent RVV construction protocol record through:

```text
selected RVV extension-family ops
  -> RVV construction protocol record
  -> RVV-owned EmitC route
  -> materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter
  -> RVV object/header/bundle artifacts
  -> ssh rvv evidence for runtime claims
```

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `bdb849d plugin: close tensorextlite construction template`; worktree clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- Commit `bdb849d` closed TensorExtLite by making emission plans,
  declaration-only headers, and object/header bundles consume the same complete
  construction protocol metadata record.
- Current RVV already has a construction manifest with protocol version,
  archetype, semantic role graph, interface realization, typed role
  realization, EmitC mapping, evidence profile, per-op EmitC route ids,
  runtime ABI names/contracts, and ordered ABI parameters.
- Current RVV emission plans and target exporters are still mostly
  route/runtime/header oriented: they emit `rvv_emitc_lowerable_route`,
  `rvv_arithmetic_op`, and RVV config/runtime-VL metadata, but they do not yet
  expose the complete construction-template record as the target artifact
  contract.
- The Direction Brief path
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir` is stale in the
  current repository; the active RVV source-seed fixtures are
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir`.

## Boundaries

- Keep the existing RVV i32m1 add/sub/mul family and materialized EmitC route.
- Do not add new SEW/LMUL/dtype/op coverage, a new RVV family, a new route, a
  generic RVV backend, or scalar fallback compute.
- Do not restore descriptor tables, old route ids, direct C semantic exporters,
  source-export paths, compatibility aliases, Python compiler-core logic, or
  common/core RVV semantic branches.
- RVV headers remain declaration-only. They may carry metadata comments and
  callable declarations, but must not embed RVV intrinsic compute bodies.
- Runtime correctness/performance claims require real `ssh rvv` evidence. If
  generated code is unchanged, reuse/refresh evidence only with a clear
  statement of what was revalidated.

## Requirements

- RVV emission-plan metadata must consume and emit the construction manifest
  fields: protocol version, extension archetype, semantic role graph,
  common-interface realization, typed-role realization, EmitC route mapping,
  evidence profile, selected per-op route, source-op provenance, runtime ABI
  contract/name, ordered runtime ABI parameters, and materialized object
  handoff.
- RVV target object/header/bundle validation must fail closed when the selected
  candidate is missing construction-template metadata, has stale route-only
  metadata, lacks source-op provenance, has mismatched runtime ABI identity or
  parameters, represents fallback-only selection, or exposes multiple selected
  supported candidates.
- RVV header artifact evidence must be derived from the same selected
  materialized EmitC candidate as the object route and must preserve the same
  construction record.
- RVV object/header bundle index must tie the same selected variant, owner
  plugin, construction protocol record, EmitC route, runtime ABI name, ordered
  ABI params, component group, object handoff, and header declaration.
- Existing negative checks for descriptor/direct-C/source-export residue must
  remain fail-closed; broaden them where the RVV construction-template closure
  creates new observable metadata.

## Acceptance Criteria

- [x] RVV construction protocol C++ coverage proves the manifest, typed role
      graph, per-op EmitC route mapping, runtime ABI contracts, ordered ABI
      parameters, and construction artifact metadata validate together.
- [x] RVV emission plans for source-front-door selected add/sub/mul carry the
      complete construction-template metadata record, not just route/runtime
      metadata.
- [x] RVV materialized object/header/bundle target artifact exports preserve
      the same construction-template record and reject stale/missing/mismatched
      construction metadata.
- [x] Header FileCheck coverage proves declaration-only headers carry the
      construction metadata and reject RVV intrinsic body/direct-C/source-export
      residue.
- [x] Bundle FileCheck coverage proves object and header records share the
      selected variant, owner plugin, runtime ABI name, ordered ABI parameter
      count/entries, component group, handoff kind, and construction metadata.
- [x] Focused scans over RVV plugin/target/translate/tests show no descriptor
      route authority, direct-C/source-export path, compatibility alias,
      Python compiler-core logic, or RVV semantic branch in core/common
      orchestration.
- [x] Focused build/lit/C++ checks pass for the touched RVV plugin and target
      artifact surfaces, or any failure is recorded as a precise rebuild gap.
- [x] Real `ssh rvv` compile/run evidence for the bundled RVV path is refreshed
      if generated code changes. If only metadata/validation changes, the
      report must say why previous object/runtime proof still covers the
      unchanged generated code and what local artifact evidence was rerun.
- [x] Trellis task status, journal, archive, and final report truthfully
      distinguish construction-template metadata closure from new runtime or
      performance claims.

## Completion Evidence

- Added RVV construction-template artifact metadata APIs under
  `RVVConstructionProtocol`: selected EmitC route, selected arithmetic op,
  bounded source-op/source-role/interface provenance, construction protocol,
  extension archetype, semantic role graph, bounded common-interface
  realization summary, bounded typed-role realization summary, EmitC route
  mapping, evidence profile, runtime ABI contract, bundle component group, and
  object handoff.
- Added RVV construction target artifact mapping API for header route,
  header artifact kind, bundle component group, object handoff kind, and
  EmitC-to-C++ translate route. RVV target support now consumes this mapping
  instead of keeping those route constants as target-local authority.
- Rewired RVV emission-plan construction so source-front-door add/sub/mul
  plans emit the complete construction-template metadata before the existing
  RVV config/runtime-VL metadata.
- Rewired RVV object/header/bundle validation so selected candidates must
  pass construction-template metadata verification and the existing runtime
  AVL/VL metadata verification. Negative C++ coverage now rejects missing
  construction protocol metadata, stale route-only metadata, stale source-op
  provenance, mismatched runtime ABI parameters, fallback-only selection,
  descriptor element-count residue, direct-C/source-export/compute-body
  residue, and ambiguous selected candidates.
- Updated RVV header and bundle lit coverage so declaration-only headers and
  bundle indexes expose the construction-template record while still rejecting
  RVV intrinsic bodies, descriptor/direct-C/source-export text, stale
  microkernel route names, and source-like compute bodies.
- The generated RVV compute C++/object logic was not changed; this round
  changed metadata propagation/validation and header/index evidence. Real
  `ssh rvv` bundle ABI evidence was still refreshed for current HEAD.

## Checks

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-17-rvv-construction-template-metadata-closure`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-construction-protocol-common-test`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `./build/bin/tianchenrv-target-artifact-export-test`
- [x] Focused lit from `build/test` passed 6/6 for
      `rvv-i32m1-vector-source-front-door*`,
      `Target/RVV/vector-materialized-target-artifact-exporters.mlir`, and
      `Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`.
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-rvv-construction-template-metadata-closure-dry --overwrite --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id codex-rvv-construction-template-metadata-closure-ssh --overwrite --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- [x] `ssh rvv` evidence passed add/sub/mul for counts `1,7,16,17,257` with
      `tcrv_rvv_generated_bundle_abi_{add,sub,mul}_ok` and
      `PASS op={add,sub,mul}` markers.
- [x] Focused descriptor/direct-C/source-export residue scan over RVV
      plugin/target/tests left only fail-closed rejection logic, negative
      fixtures, and `implicit-check-not` assertions.
- [x] Core/common RVV branch scan over `lib/Transforms`, `lib/Conversion`,
      `include/TianChenRV/Transforms`, and `include/TianChenRV/Conversion`
      left only prohibitive `Passes.td` text.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed.

## Out Of Scope

- No new RVV source front door beyond the existing vector i32m1 add/sub/mul
  fixtures.
- No new arithmetic op, dtype, SEW/LMUL, policy, or hardware family.
- No generic RVV backend, independent backend dialect, or new lowering route.
- No descriptor adapter, descriptor-driven computation, direct C semantic
  exporter, source-export path, scalar fallback compute, compatibility wrapper,
  or Python compiler-core implementation.
- No broad test matrix unless focused checks expose a cross-layer break that
  cannot be isolated.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous proof PRD read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-tensorextlite-executable-plugin-construction-template-closure/prd.md`.
- Initial implementation files inspected:
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp`.
- Initial test files inspected:
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir`, and
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`.

## Definition Of Done

- RVV construction-template metadata is implemented in the production
  emission/header/bundle route.
- Focused C++/lit checks and residue scans are recorded.
- Trellis context validates.
- Task is finished/archived if acceptance criteria are met.
- One coherent commit is created if the round completes.
