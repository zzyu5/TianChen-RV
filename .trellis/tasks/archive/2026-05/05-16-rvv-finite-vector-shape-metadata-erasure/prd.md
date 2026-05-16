# RVV finite vector-shape metadata erasure

## Goal

Erase RVV finite vector-shape metadata catalogs and runtime-length metadata
helpers as route/config authority. This round keeps the deletion campaign
honest: future RVV executable rebuild must carry SEW, LMUL, tail, mask, AVL,
and VL through explicit `tcrv_rvv` ops and common lowering/export interfaces,
not through target-owned descriptor helper catalogs or comment/expression
formatters.

## Why

The previous completed round deleted support-layer finite-binary callable ABI
plan authority, but the RVV target helper surface still contains finite i32/i64
shape catalogs, selected vector-shape capability IDs, selected vector attrs,
vector type/suffix/setvl suffix strings, runtime-VL boundary comment text, and
remaining-AVL C expression formatting. `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
still advertises finite vector-shape capabilities from that catalog, and tests
still protect the selected-shape selector and runtime-length contract. That is
descriptor-style finite-family route/config authority rather than explicit
extension-family ops materialized through EmitC.

## Scope

- Delete RVV finite vector-shape target helper APIs and catalogs from
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`.
- Delete RVV runtime-length metadata/comment/expression helper APIs from
  `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`.
- Rewire or delete direct consumers in the RVV plugin capability advertisement
  path without introducing replacement vector-shape metadata schemas.
- Rewrite or remove tests that protect selected-vector-shape selector capability
  lookup, selected vector attr helper authority, runtime-VL boundary comments,
  or remaining-AVL C expression formatting.
- Remove direct CMake references if any deleted test/source target disappears.
- Rewrite directly related spec wording if it preserves finite RVV target helper
  catalogs or runtime-length metadata helpers as active authority.
- Run focused active-surface ref-scans and targeted build/check commands after
  deletion.

## Non-goals

- No Common EmitC rebuild.
- No new RVV lowering.
- No new runtime ABI implementation.
- No replacement vector-shape schema.
- No new selected-boundary metadata layer.
- No source artifact route.
- No `ssh rvv` evidence work.
- No compatibility wrappers.
- No broad redesign of neutral capability/plugin registry code.
- No restoration of finite shape or runtime-length helper authority to keep
  tests green.

## Requirements

- No active surface defines or consumes finite RVV vector-shape config structs,
  finite i32/i64 shape catalogs, finite lookup helpers, selected-shape
  capability ID helpers, selected vector attr/suffix helper authority,
  `RVVRuntimeLengthContract`, runtime-VL boundary comment formatting, or
  remaining-AVL C expression formatting.
- RVV plugin tests no longer protect selected-vector-shape selector capability
  lookup.
- Target artifact tests no longer protect runtime-length comments or C
  expression formatting.
- The base RVV capability and explicit `tcrv_rvv` dialect/op surfaces may
  remain, but they must not depend on finite target-helper catalogs or
  replacement metadata descriptors.
- If deletion exposes missing new-architecture gaps, report the gap and keep
  deletion truthful rather than restoring obsolete helpers.

## Acceptance Criteria

- [ ] Focused active-surface ref-scan finds no live reference to
  `RVVVectorShapeConfig`, `RVVSelectedVectorShapeConfig`,
  `RVVI32VectorShapeConfig`, `getI32M1VectorShapeConfig`,
  `getI32M2VectorShapeConfig`, `getI64M1VectorShapeConfig`,
  `getFiniteI32VectorShapeConfigs`, `getFiniteI64VectorShapeConfigs`,
  `lookupFiniteI32VectorShapeConfigByShapeID`,
  `lookupFiniteI64VectorShapeConfigByShapeID`,
  `getRVVI32BinarySelectedVectorShapeCapabilityID`, `selected_vector_shape`,
  `selected_vector_type`, `selected_vector_suffix`, `selected_setvl_suffix`,
  `RVVRuntimeLengthContract`, `validateRVVRuntimeLengthContract`,
  `formatRuntimeVLBoundaryCommentBody`,
  `formatRemainingAVLOperandExpression`, or
  `selected_runtime_vl_boundary` outside archived tasks, workspace notes,
  build artifacts, temporary artifacts, and git metadata.
- [ ] `RVVExtensionPlugin` no longer advertises finite vector-shape
  capabilities from target helper catalogs.
- [ ] RVV plugin tests no longer encode selected vector-shape selector
  capability lookup or selected vector attr helper authority.
- [ ] Target artifact tests no longer encode runtime-length comment bodies or
  remaining-AVL C expression formatting.
- [ ] Affected plugin/target/support targets that still exist build.
- [ ] Remaining relevant affected plugin and target artifact tests run.
- [ ] `ninja -C build check-tianchenrv` is attempted after focused checks.
- [ ] `git diff --check`, Trellis validation, task finish/archive, clean
  status, and one coherent commit are completed if the task is complete.

## Read First

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `include/TianChenRV/Target/RVV/RVVVectorShape.h`
- `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `test/Target/TargetArtifactExportTest.cpp`
- Direct CMake/spec consumers found by focused ref-scan only.

## Technical Notes

- Initial repo state: `/home/kingdom/phdworks/TianchenRV`, clean worktree,
  HEAD `751bd28 chore(support): erase callable abi plan authority`.
- There is no pre-existing active Trellis task; this task was created from the
  Direction Brief as a deletion-only Wrong Logic Deletion Campaign round.
- Durable project rules from `.trellis/spec/index.md`: compiler implementation
  stays in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck; Python is tooling only;
  descriptor-driven computation is deletion target/fail-closed debt; future
  executable work goes through extension family ops and common EmitC lowering.
- Memory-derived campaign context used only as guardrail: deletion before
  rebuild, no compatibility wrappers, and no mixing RVV rebuild work into this
  deletion round.
