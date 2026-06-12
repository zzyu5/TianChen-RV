# TensorExtLite Executable Construction Protocol Consumption

## Goal

Make TensorExtLite's existing source-front-door, ordered role sequence, EmitC
route, runtime ABI metadata, target artifact route identity, header/bundle
coherence, and evidence profile consume one TensorExtLite-owned construction
protocol instead of preserving duplicated hand-wired declarations across the
plugin, EmitC route provider, target artifact bundle, and tests.

This is a code-consuming construction-template milestone. The existing
fragment-MMA-like first slice must remain behaviorally unchanged:

```text
TensorExtLite source marker
  -> TensorExtLite source front door
  -> selected TensorExtLite variant
  -> configure -> load_frag -> tile_mma -> store_frag role ops
  -> TensorExtLite EmitC route
  -> common materialized EmitC C/C++ emitter
  -> TensorExtLite object/header/bundle artifacts
```

## Current Repository Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- This is a continuation retry after a transient Codex/API failure. The
  previous attempt created `.trellis/tasks/05-17-tensorextlite-construction-protocol-consumption/`
  but did not create `prd.md`, edit compiler code, run checks, archive, or
  commit.
- Starting HEAD is `a53599c trellis: archive TensorExtLite source artifact closure`.
- The previous archived task proves the current TensorExtLite source-to-object,
  header, and bundle production path is already closed in HEAD. Repeating that
  evidence-only closure is not the goal.
- Current code already has
  `TensorExtLiteConstructionProtocol.{h,cpp}` with a manifest, typed-role
  realization, route mapping, and verifier.
- Current TensorExtLite source-front-door, EmitC route provider, target support
  bundle, and focused tests still contain duplicated role/route/artifact
  strings that should be sourced from the construction protocol where that
  protocol is the stable owner.

## Requirements

- Keep the work bounded to TensorExtLite's existing fragment-MMA first slice.
- Do not add new TensorExtLite math semantics, new extension families, RVV
  behavior, descriptor adapters, direct C semantic exporters, source-export
  routes, compatibility aliases, Python compiler-core behavior, or common/core
  family-specific branches.
- Keep all compiler behavior in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Preserve the default production behavior and public route identity: existing
  TensorExtLite source-front-door inputs must still reach materialized EmitC
  object/header/bundle artifacts.
- Make at least two real TensorExtLite consumers derive role/route/artifact
  identity from the construction protocol rather than local duplicate
  declarations. Priority consumers are the source front door, EmitC route
  provider, target artifact bundle, and C++ tests.
- Make stale or mismatched construction protocol data fail before artifact
  export. Existing stale route, missing role, reordered role, missing boundary,
  runtime ABI, and header/bundle mismatch negatives should continue to fail
  closed.
- Delete or rewrite tests/helpers that only protect hand-wired TensorExtLite
  constants when a protocol-derived helper is the correct authority.

## Acceptance Criteria

- [x] TensorExtLite construction protocol exposes reusable code-level access to
      the ordered role realization, route callee mapping, artifact metadata
      keys/values, route source-op/source-role strings, and zero-argument
      runtime ABI signature needed by the existing path.
- [x] TensorExtLite source-front-door role materialization consumes typed-role
      realization data from the construction protocol instead of a local
      duplicate role-spec table.
- [x] TensorExtLite EmitC route construction consumes protocol role/route
      mapping for role lookup, call-opaque callee mapping, source-role order,
      and route identity.
- [x] TensorExtLite target artifact/header/bundle validation consumes protocol
      route/artifact metadata and fails closed on stale route, role sequence,
      runtime ABI, or metadata evidence mismatch before artifact export.
- [x] Focused C++ tests prove protocol-derived plugin proposal metadata,
      ordered role materialization, EmitC route mapping, runtime ABI metadata,
      header metadata, bundle coherence, and mismatch failures.
- [x] Focused lit tests for TensorExtLite source-front-door, EmitC
      materialization, object/header/bundle artifact export, and negative
      unsupported/missing/mismatched cases still pass.
- [x] Targeted scans over TensorExtLite plugin/target/tests and common target
      surfaces show no descriptor route authority, direct C semantic exporter,
      source-export route, Python compiler-core path, or common/core
      extension-specific semantic branch was introduced.
- [x] `git diff --check` passes.
- [x] Focused build targets and tests pass; run `check-tianchenrv` if
      practical.

## Out Of Scope

- New RVV work or RVV refactor.
- New TensorExtLite operations, tensor/tile semantics, runtime execution,
  correctness, or performance claims.
- Another evidence-only harness, broad smoke matrix, status report, metadata
  checklist, or PRD-only template.
- Descriptor-driven computation, descriptor-to-C export, direct C semantic
  exporter, source-export route, compatibility alias, or Python compiler-core
  implementation.
- Moving TensorExtLite behavior into common/core orchestration through a
  TensorExtLite/Toy/RVV name branch.

## Technical Notes

- Required specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/*`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-17-tensorextlite-source-artifact-closure/prd.md`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/TensorExtLite/TensorExtLiteConstructionProtocol.h`,
  `lib/Plugin/TensorExtLite/Construction/TensorExtLiteConstructionProtocol.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteSourceFrontDoor.cpp`,
  `lib/Plugin/TensorExtLite/EmitC/TensorExtLiteEmitCRouteProvider.cpp`,
  `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`, and
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`.
- Primary test surfaces:
  `test/Plugin/TensorExtLiteExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Transforms/TensorExtLite/`,
  `test/Conversion/EmitC/tensorext-lite-first-slice-materialization*.mlir`,
  and `test/Target/TensorExtLite/`.

## Completion Summary

- Added protocol-owned role-step, route, runtime ABI, artifact metadata, header,
  bundle, and evidence-profile accessors for the existing TensorExtLite
  fragment-MMA first slice.
- Rewired production consumers so the TensorExtLite source front door, EmitC
  route provider, plugin emission plan, and target object/header/bundle support
  consume the construction protocol instead of local duplicate declarations.
- Rewrote focused C++ tests so protocol role graph, source provenance, route
  callees, runtime ABI, artifact metadata, header metadata, bundle coherence,
  and stale metadata failure are tested through protocol-derived values.
- Preserved the existing default source-to-artifact behavior: source marker to
  role ops to materialized EmitC to object/header/bundle remains unchanged.
- Targeted scans found no descriptor route authority, direct C semantic
  exporter, source-export route, Python compiler-core path, or common/core
  TensorExtLite semantic branch introduced.
- Remaining future work is construction-template expansion for additional
  extension slices or families, not more artifact-path evidence for the already
  closed TensorExtLite first slice.

## Validation

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-construction-protocol-common-test -j2`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='TensorExtLite|tensorext-lite' /home/kingdom/phdworks/TianchenRV/build/test`
- `cmake --build build --target check-tianchenrv -j2`
- `git diff --check`
- Targeted residue scans over TensorExtLite plugin/target/tests and common
  target surfaces.
