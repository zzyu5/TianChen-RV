# TensorExtLite executable plugin construction template closure

## Goal

Close one coherent TensorExtLite executable plugin-template slice so future
extension-family work can copy a real construction workflow instead of a
metadata-only manifest or an RVV-only pattern. The bounded workflow is:

```text
TensorExtLite source marker
  -> plugin-owned selected TensorExtLite extension-family role ops
  -> construction protocol record
  -> plugin-owned EmitC route
  -> materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter
  -> target object/header/bundle artifact
```

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round: HEAD
  `0eb8c4b test: prove rvv generated bundle c abi`; worktree clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- The previous archived RVV task
  `.trellis/tasks/archive/2026-05/05-17-rvv-generated-bundle-external-c-abi-proof/prd.md`
  proved generated RVV object/header/bundle ABI consumption with real
  `ssh rvv` evidence, but it did not make TensorExtLite the reusable
  non-RVV plugin construction template.
- Current TensorExtLite already has a source front door, selected variant,
  explicit role ops, plugin-owned EmitC route provider, materialized EmitC lit
  tests, and local RISC-V object/header/bundle target artifact tests.
- Current TensorExtLite construction artifact metadata records source op/role
  provenance plus protocol, semantic role graph, and typed role realization.
  The brief requires the target artifact record to carry the complete
  construction protocol record, including extension archetype, common interface
  realization, EmitC route mapping, and evidence profile.

## Boundaries

- The module owner is the bounded TensorExtLite first slice: fragment-MMA
  source marker, explicit TensorExtLite role ops, common construction protocol,
  materialized EmitC, and target artifact object/header/bundle.
- Keep the route plugin-owned and common-interface-driven. Core/common passes
  may consume generic interfaces and generic target artifact helpers only.
- Use the existing bounded TensorExtLite fragment-MMA source front door and
  first-slice ops. Do not add a new frontend, new TensorExt feature, or new
  hardware/runtime claim.
- Python is not part of the compiler implementation in this round. It may only
  be used indirectly by existing lit/build tooling.
- Local TensorExtLite object packaging is a compile/package proof for generated
  C++ and RISC-V relocatable object shape. It is not TensorExtLite runtime
  correctness, performance, or real hardware execution evidence.

## Requirements

- Positive source-front-door evidence must show the marker reaches an explicit
  selected TensorExtLite variant body with configure/load_frag/tile_mma/
  store_frag role ops.
- Positive construction evidence must expose a complete construction protocol
  record:
  protocol version, extension archetype, semantic role graph, common interface
  realization, typed role realization, EmitC route mapping, and evidence
  profile.
- Positive EmitC evidence must show the explicit role body builds a
  plugin-owned `TCRVEmitCLowerableRoute` and materializes a MLIR EmitC module
  through the common materializer.
- Positive target evidence must show the production bundle front door emits a
  coherent object/header bundle with matching selected variant, owner plugin,
  route ids, runtime ABI identity, zero-argument ABI boundary, component group,
  object handoff kind, and the complete construction protocol metadata.
- Negative coverage must fail closed for stale source metadata, missing
  extension-family body, stale route mapping, mismatched construction protocol
  metadata, non-materialized EmitC, and descriptor/direct-C/source-export
  residue.
- Focused scans must show no descriptor-driven compute authority, no direct C
  semantic exporter, no source-export route, no Python compiler-core behavior,
  and no TensorExtLite/RVV/Toy/Offload semantic branch added in core/common
  orchestration.

## Acceptance Criteria

- [x] TensorExtLite source-front-door lit coverage proves source marker ->
      selected variant -> explicit TensorExtLite role body -> emission plan.
- [x] TensorExtLite construction C++ coverage proves manifest, typed role
      realization, role steps, EmitC route mapping, target artifact mapping,
      and complete artifact metadata validate together.
- [x] TensorExtLite emission plan artifact metadata includes the complete
      construction protocol record required by the Direction Brief.
- [x] TensorExtLite header and bundle artifact metadata include the same
      complete construction protocol record and reject stale/mismatched records
      through existing metadata validation paths.
- [x] TensorExtLite materialized EmitC lit coverage still proves common EmitC
      module materialization and fail-closed missing-body/stale-route cases.
- [x] TensorExtLite target artifact lit coverage still proves local RISC-V
      relocatable object, declaration-only header, object/header bundle
      coherence, and fail-closed non-materialized/stale inputs.
- [x] Focused build/test/lit checks pass for changed TensorExtLite/plugin/
      target surfaces, or any failure is recorded as a precise rebuild gap.
- [x] Trellis task status, journal, archive, and final report truthfully
      distinguish compiler/workflow closure from runtime/hardware claims.

## Completion Evidence

- Extended TensorExtLite fragment-MMA artifact metadata from 8 to 12 entries,
  adding extension archetype, common interface realization, EmitC route
  mapping, and evidence profile alongside the existing route/source/protocol/
  role metadata.
- Rewired TensorExtLite emission-plan construction to emit the complete
  construction protocol metadata record and validate it through
  `verifyTensorExtLiteFragmentMmaArtifactMetadata`.
- Rewired TensorExtLite header evidence and object/header bundle export so the
  declaration-only header and bundle index preserve the same complete
  construction record.
- Updated TensorExtLite C++ plugin tests, target artifact export C++ fixtures,
  source-front-door lit, materialized EmitC lit, target header/object/bundle
  lit, and local runtime ABI evidence checks for the complete record.
- Self-repair: first full `check-tianchenrv` run failed because
  `TargetArtifactExportTest.cpp` had a handwritten TensorExtLite emission-plan
  fixture still using the old 8-entry metadata set. The fixture was updated to
  the 12-entry contract, then the target-artifact test and full check passed.
- Local ABI evidence:
  `artifacts/tmp/tensorextlite_runtime_abi_e2e/codex-tensorextlite-template-closure`.
  It consumed `--tcrv-source-artifact-bundle-front-door`, verified the
  generated object/header bundle metadata, compiled a native ABI harness
  against generated C++ and header surfaces, and recorded call trace
  `configure,load_frag,tile_mma,store_frag`.

## Checks

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-17-05-17-tensorextlite-executable-plugin-construction-template-closure`
- [x] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-tensorext-lite-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [x] Focused lit from `build/test` passed 12/12 for TensorExtLite source
      front door, EmitC materialization, target artifact, bundle, and runtime
      ABI harness coverage.
- [x] `python3 -m py_compile scripts/tensorextlite_runtime_abi_e2e.py`
- [x] `python3 scripts/tensorextlite_runtime_abi_e2e.py --artifact-root artifacts/tmp/tensorextlite_runtime_abi_e2e --run-id codex-tensorextlite-template-closure --input test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir --materialized-input test/Target/TensorExtLite/tensorext-lite-target-artifact-header.mlir --tcrv-translate build/bin/tcrv-translate --clangxx clang++ --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [x] `cmake --build build --target tianchenrv-target-artifact-export-test -j2 && ./build/bin/tianchenrv-target-artifact-export-test`
- [x] Focused residue scan over touched TensorExtLite/plugin/target/script
      surfaces: remaining descriptor/direct-C/source-export matches are
      rejection checks, negative fixtures, or `implicit-check-not` assertions.
- [x] Core/common branch scan over `lib/Transforms`, `lib/Conversion`,
      `include/TianChenRV/Transforms`, and `include/TianChenRV/Conversion`:
      remaining matches are prohibitive documentation text in `Passes.td`, not
      extension-specific semantic branches.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2` -> 122/122 passed
      after the handwritten target artifact fixture self-repair.

## Out of Scope

- No RVV family coverage or new RVV evidence scripts.
- No broad smoke matrix.
- No new TensorExtLite feature beyond the existing first slice.
- No standalone report tooling.
- No metadata-only manifest as completion.
- No descriptor adapter, descriptor-driven computation, direct C semantic
  exporter, source-export route, fake intrinsic header, source-like skeleton
  printer, compatibility wrapper, scalar fallback compute, or Python
  compiler-core behavior.
- No core orchestration branch on TensorExtLite/RVV/Toy/Offload semantics.
- No TensorExtLite runtime correctness, performance, or hardware execution
  claim.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Previous proof PRD read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-generated-bundle-external-c-abi-proof/prd.md`.
- Implementation files inspected:
  `include/TianChenRV/Plugin/ConstructionProtocol.h`,
  `lib/Plugin/Construction/ConstructionProtocol.cpp`,
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`, and
  `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`.
- Test files inspected:
  `test/Transforms/TensorExtLite/`,
  `test/Conversion/EmitC/tensorext-lite-first-slice-materialization*.mlir`,
  and `test/Target/TensorExtLite/`.

## Definition of Done

- Focused TensorExtLite compiler/workflow closure is implemented.
- Focused C++/lit checks and residue scans are recorded.
- Trellis context validates.
- Task is finished/archived if acceptance criteria are met.
- One coherent commit is created if the round completes.
