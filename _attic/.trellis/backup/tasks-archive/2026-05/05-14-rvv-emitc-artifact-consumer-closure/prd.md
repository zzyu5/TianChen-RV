# RVV EmitC Artifact Consumer Closure

## Goal

Close the production artifact/export consumer path for the selected RVV EmitC
body emitter. The migrated dynamic `i32-vadd` and `i32-vsub` routes must carry
compiler-selected RVV family/body authority through generated source, header,
object, and bundle artifacts rather than stopping at RVV target body emission
or selected metadata annotation.

## Why Now

The previous committed slice at `7c553ce` made the migrated RVV body emitter
consume selected EmitC mapping from materialized selected state. The next
bottleneck is artifact ownership: default TargetArtifactExport and
TargetArtifactBundleExport paths must prove that their generated artifacts
consume the selected EmitC body mapping and fail closed when only legacy,
descriptor-only, or stale direct target string authority is present.

## Scope

- Existing migrated `vector-dynamic-i32-vadd` and
  `vector-dynamic-i32-vsub` selected routes only.
- Production artifact/export handoff for generated C source, C header,
  relocatable object, and target artifact bundle records.
- RVV selected emission planning consumption, RVV target compatibility handoff,
  and e2e plumbing only where needed to make generated artifacts consume the
  selected EmitC body route.
- Focused fail-closed coverage for missing or stale selected EmitC body
  authority before source/header/object/bundle output.

## Non-Goals

- No new dtype, i64 expansion, LMUL matrix, third operation, broad family
  matrix, broad smoke suite, or performance tuning.
- No descriptor-to-C production exporter and no descriptor element count or
  descriptor vector shape as authoritative compute/config/runtime control.
- No computation semantics in `tcrv.exec`.
- No RVV semantic branches in generic core orchestration or generic transforms.
- No GCC/vendor compiler default route.
- No Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin work
  except narrow regressions caused by shared validation.
- No runtime, correctness, or performance claims beyond focused evidence
  actually run for this finite selected vadd/vsub route.

## Requirements

- TargetArtifactExport and TargetArtifactBundleExport for vadd/vsub consume
  the selected RVV EmitC body emission output, not only selected metadata or a
  direct target string fallback.
- Generated C bodies preserve op-family identity, source identity, selected
  SEW/LMUL/tail/mask config, runtime length role data, and
  clang/LLVM-compatible RVV intrinsics through `riscv_vector.h`.
- Generated header/object/bundle artifacts are matched from the same
  validated callable source candidate and cannot bypass selected EmitC body
  mapper authority.
- Direct RVV target body emission remains either factored behind the selected
  EmitC body mapper or retained only as a compatibility path with fail-closed
  selected-state checks.
- Descriptor-only compute/config/runtime authority remains quarantined on the
  migrated default route.
- Core `tcrv.exec` and generic transforms stay orchestration-only.

## Acceptance Criteria

- [x] vadd and vsub default artifact source routes prove generated body C came
      from the selected RVV EmitC body mapping.
- [x] vadd and vsub header/object routes preflight the same selected callable
      body candidate rather than bypassing body-mapper validation.
- [x] vadd and vsub bundle export records preserve selected EmitC body route
      metadata, selected config, runtime length authority, component metadata,
      and generated source/header/object outputs.
- [x] Missing selected EmitC body mapping fails before source/header/object or
      complete bundle output.
- [x] Descriptor-only production attempts, stale op identity, vadd/vsub ABI
      mismatch, missing/conflicting selected config, and missing/stale runtime
      length role data remain fail-closed.
- [x] Focused vadd/vsub VectorToExec, TargetArtifactExport,
      TargetArtifactBundleExport, RVVMicrokernel, RVVScalarDispatch, object,
      selected-quarantine, and `rvv-microkernel-bundle-e2e.test` regressions
      pass or exact blockers are recorded.
- [x] Bounded reference scan over touched RVV plugin/target/export files shows
      descriptor-driven or direct string-owned C emission is not the migrated
      default artifact route.
- [x] If generated RVV C/object behavior changes, focused `ssh rvv` evidence is
      collected through the existing e2e path; otherwise no new runtime or
      correctness claim is made.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean worktree, and one coherent
      commit complete the round if finished.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/architecture/design-boundaries.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-14-rvv-emitc-body-emission-production-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-14-rvv-extension-family-emitc-emission-route/prd.md`

Likely implementation surface:

- `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
- `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
- `lib/Target/RVV/RVVMicrokernel.cpp`
- `lib/Target/RVV/RVVScalarDispatch.cpp`
- `include/TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h`
- `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `test/Target/TargetArtifactExportTest.cpp`
- `test/Target/TargetArtifactBundleExport/`
- `test/Transforms/VectorToExec/`
- `test/Scripts/rvv-microkernel-bundle-e2e.test`
- `scripts/rvv_microkernel_e2e.py`
- `tools/tcrv-translate/tcrv-translate.cpp`

## Definition Of Done

The finite migrated dynamic `i32-vadd`/`i32-vsub` selected route has an active
artifact/export consumer closure: default generated source/header/object/bundle
artifacts consume the selected RVV EmitC body mapper, descriptor-only authority
remains quarantined, generic orchestration remains target-neutral, focused
checks pass or exact blockers are recorded, the task is archived, and one
coherent commit records the work.

## Implementation Summary

- `TargetArtifactExport.cpp` now applies registered composite
  `TargetArtifactRouteMetadata` to matched single-candidate composite routes
  during generic selection, bundle record collection, and exact composite route
  export. This closes the generic artifact/export consumer gap without adding
  RVV semantic branches to core orchestration.
- `RVVMicrokernel.cpp` now registers header/object composite routes with the
  same selected-plan requirements as the source route, including typed RVV
  family identity, selected config, runtime VL/AVL authority, and selected
  EmitC route metadata. Headers also record declaration-only proof that the
  selected body mapping was validated before source/header/object export.
- `RVVScalarDispatch.cpp` now prints a sanitized dispatch header/source summary
  showing that the selected RVV EmitC body mapping was consumed through the
  embedded selected RVV source artifact, while keeping dispatch headers free of
  raw RVV intrinsic/include names.
- Focused tests now assert selected EmitC body mapping reaches direct RVV
  source/header/object bundle artifacts and vector-dynamic vadd/vsub dispatch
  source/header/object bundle artifacts.

## Validation Summary

- `git diff --check`
- `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-header\\.mlir|rvv-microkernel-family-sub\\.mlir|target-artifact-bundle-positive\\.mlir|target-artifact-bundle-rvv-vsub\\.mlir|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle\\.mlir|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle\\.mlir'` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-bundle-e2e\\.test|rvv-microkernel-e2e\\.test'` from `build/test`
- Bounded ref-scan over touched RVV plugin/target/export files found selected
  EmitC route metadata and composite validation as active artifact/export
  gates; descriptor-to-C remains diagnostic/quarantined and direct selected
  config compatibility is bounded to non-selected-plan compatibility.
- No generated RVV C behavior changed beyond comments/metadata validation, so
  this round makes no new ssh-rvv runtime, correctness, or performance claim.
