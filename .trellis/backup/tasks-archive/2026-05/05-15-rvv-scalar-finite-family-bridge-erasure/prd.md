# RVV-scalar finite-family bridge erasure

## Goal

Delete the obsolete RVV-derived scalar/RVV-scalar finite-family bridge as an
active source of scalar callable microkernel, RVV-scalar dispatch,
selected-binary metadata, and runtime ABI authority. Scalar fallback must remain
metadata-only unless a future task rebuilds an executable route through real
extension-family ops and the common EmitC route.

## Why Now

The previous deletion round removed scalar and RVV-scalar direct-C route shells.
The remaining bridge is no longer a production route API. It derives scalar and
RVV-scalar family authority from `RVVBinaryFamilyRecord` through
`RVVScalarBinaryFamily.h` and feeds scalar plugin callable microkernel handling
and runtime ABI/window materialization. That is old finite-family /
descriptor-style authority and must be deleted before rebuild.

## Scope

- Delete `include/TianChenRV/Target/RVVScalarBinaryFamily.h` as an active
  bridge header.
- Remove direct includes, aliases, and CMake/header references for that bridge.
- Remove scalar plugin code paths that derive scalar microkernel families or
  runtime ABI/mem-window facts from RVV family records.
- Delete or rewrite tests that protect bridge-derived scalar callable,
  RVV-scalar dispatch, selected-binary metadata, or runtime ABI behavior.
- Keep scalar fallback metadata-only behavior where it remains independent of
  the finite-family bridge.

## Non-Goals

- Do not implement a new scalar EmitC route.
- Do not implement a new RVV-scalar dispatch route.
- Do not implement RVV lowering, scalar lowering, runtime ABI model, artifact
  emitter, compatibility wrapper, legacy mode, or replacement family registry.
- Do not delete unrelated generic runtime ABI utilities, generic target
  artifact registry code, or scalar fallback metadata-only proposal behavior.
- Do not add tests for the old bridge or broaden finite RVV/scalar family
  coverage.

## Requirements

- `RVVScalarBinaryFamily`, `RVVScalarBinaryFamilyRecord`,
  `ScalarBinaryMicrokernelRecord`, `RVVScalarDispatchFamilyRecord`,
  `getRVVScalarBinaryRegistrationRecords`, scalar selected-binary metadata
  helpers, and bridge-derived runtime ABI helper calls must not exist as active
  code.
- Scalar plugin emission/readiness must not carry a special unsupported direct-C
  microkernel branch keyed by explicit scalar microkernel detection.
- Scalar lowering-boundary materialization must not synthesize or preserve
  RVV-derived callable microkernel/runtime ABI authority.
- Typed scalar dialect ops may remain as dialect surface or future-route IR if
  still used elsewhere, but they must not be promoted by the scalar fallback
  plugin into current executable or direct-C route authority.
- Any exposed future scalar executable need should be reported as a rebuild gap,
  not repaired by restoring the bridge.

## Acceptance Criteria

- [ ] Focused reference scans show no active code references to the bridge
      symbols listed above.
- [ ] `include/TianChenRV/Target/RVVScalarBinaryFamily.h` is removed or reduced
      to no active bridge authority; preferred result is deletion.
- [ ] Scalar plugin no longer includes the bridge header and no longer derives
      scalar microkernel family specs from RVV records.
- [ ] Tests protecting explicit scalar callable direct-C unsupported emission
      or RVV-scalar dispatch bridge records are removed or rewritten.
- [ ] Focused build/test targets for scalar plugin, runtime ABI callable plan,
      target artifact export, and touched lit tests pass, or failures are
      recorded as expected deletion gaps without restoring wrong logic.
- [ ] Trellis context validates, task is finished/archived if complete, and the
      work is committed as one coherent deletion/refactor commit.

## Relevant Specs

- `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

## Initial Technical Findings

- `include/TianChenRV/Target/RVVScalarBinaryFamily.h` defines the bridge record
  types, scalar selected-binary metadata helpers, RVV-derived family lookup, and
  runtime ABI helper calls.
- `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` includes the bridge and maps
  typed scalar microkernel ops to bridge-derived RVV records before returning
  deleted direct-C unsupported plans or inserting ABI boundary ops.
- `test/Plugin/ScalarExtensionPluginTest.cpp`,
  `test/Plugin/RVVBinaryPlanningTest.cpp`, and
  `test/Target/TargetArtifactExportTest.cpp` include or reference bridge
  behavior that must be deleted or rewritten.
