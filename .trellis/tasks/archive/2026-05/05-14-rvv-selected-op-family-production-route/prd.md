# RVV selected op-family production route

## Goal

Make the selected RVV op-family artifact route production-reusable by carrying
`vector-dynamic-i32-vsub` through the same materialized `tcrv_rvv` source
identity, selected SEW/LMUL/tail/mask config, runtime AVL/VL authority, selected
emission planning, scalar dispatch, and source/header/object bundle route that
is already proven for `vector-dynamic-i32-vadd`.

This round must not close as another script-only or test-only evidence pass. If
the vsub route is already mostly present, the implementation must remove or
factor any remaining vadd-specialized production assumption in the migrated
op-family path and add focused evidence for the second family member.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round: clean worktree, HEAD
  `daa4dbc test(rvv): close compiler artifact e2e evidence`.
* No `.trellis/.current-task` existed before this task; this task was created
  from the Hermes Direction Brief and started as the current Trellis task.
* The previous compiler-produced artifact closure changed only Trellis state,
  journal state, and `test/Scripts/rvv-microkernel-bundle-e2e.test`.
* Existing code and tests already show a broad finite RVV binary family table
  for i32/i64 add/sub/mul, plus `vector-dynamic-i32-vsub` lowering and bundle
  tests.
* A real production gap remains in
  `lib/Target/RVV/RVVMicrokernel.cpp`: target-artifact candidate preflight
  currently requires selected lowering-boundary source identity only for the
  i32-vadd family. A planned i32-vsub route can have op-owned source identity
  stripped from both `tcrv_rvv.lowering_boundary` and
  `tcrv_rvv.i32_vsub_microkernel` while still exporting a complete bundle.

## Requirements

* Keep scope bounded to one second existing i32 family member:
  `vector-dynamic-i32-vsub`, plus vadd regression.
* Do not add a dtype, i64 expansion, LMUL matrix, broad op matrix, third
  operation, or performance tuning objective.
* Operation identity must be represented as selected RVV op-family state, not
  as a vadd-only preflight or route assumption.
* The upstream selected vsub route must materialize a concrete
  `tcrv_rvv.i32_vsub_microkernel` with selected source identity, selected
  config, runtime AVL/VL role data, and runtime ABI role state.
* Selected emission planning and artifact export must consume the same
  `RVVSelectedConfigContract` and `RVVRuntimeLengthContract` path as vadd.
* Generated source must use clang/LLVM-compatible RVV C intrinsics through
  `riscv_vector.h`, specifically `__riscv_vsub_vv_i32m1` for this bounded
  dynamic vsub slice.
* Stale or descriptor-only op/config/runtime authority and vadd/vsub ABI or
  source-identity mismatches must fail closed before source/header/object or a
  complete bundle index is emitted.
* Core `tcrv.exec` must remain orchestration-only. Generic transforms and
  generic artifact routing must not gain RVV semantic branches.
* Python may be used only for runner/test mutation support and evidence
  orchestration, not compiler semantics.

## Acceptance Criteria

* [x] Production target-artifact preflight no longer has an i32-vadd-only
      requirement for selected-boundary source identity in the finite RVV binary
      family route.
* [x] Removing op-owned selected source identity from both the
      `vector-dynamic-i32-vsub` RVV lowering boundary and microkernel fails
      before bundle completion; this must not rely on descriptor-only fallback
      or stale selected-plan metadata.
* [x] Positive vsub bundle evidence still proves the selected path carries
      `i32-vsub`, `tcrv_rvv.i32_vsub_microkernel`,
      `tcrv_rvv.i32_sub`, `frontend-lowering`, selected config, runtime AVL/VL
      role data, source/header/object records, and `__riscv_vsub_vv_i32m1`.
* [x] The existing vadd compiler-produced bundle/e2e coverage remains green.
* [x] Focused fail-closed coverage covers at least missing vsub op-owned source
      identity plus existing stale runtime/config/source checks.
* [x] A bounded ref-scan confirms the changed path removes a vadd-only
      hardcode in the migrated RVV target artifact preflight and adds no RVV
      semantic branch to core `tcrv.exec` or generic transforms.
* [x] Focused build/lit/script checks pass for touched RVV target/export,
      vector-to-exec, target artifact bundle, and e2e surfaces.
* [x] `ssh rvv` evidence is collected if generated source/object behavior
      changes. If the change is fail-closed preflight/test plumbing only, the
      final report must not claim runtime correctness or performance.
* [x] `git diff --check`, staged diff check, Trellis validation before finish
      and after archive, final clean worktree, and one coherent commit complete
      the round if finished.

## Definition Of Done

The finite RVV binary selected route treats vsub as the second member of the
same selected op-family production contract rather than as a path that can
inherit vadd-only preflight guarantees. The compiler-produced vsub route must
export only when selected source identity, selected config, runtime length, and
runtime ABI contracts agree from materialization through bundle records.

## Out Of Scope

* New dtype, i64 expansion, LMUL matrix, broad family matrix, third operation,
  broad smoke suite, or performance work.
* Descriptor-to-C production export or descriptor element count/vector shape as
  authoritative compute, config, or runtime control.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic core orchestration.
* GCC/vendor compiler as the default route.
* Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin changes except
  narrow regressions caused by shared validation.
* Runtime, correctness, or performance claims beyond focused evidence actually
  run for this finite selected vsub route.

## Technical Notes

Specs read before PRD:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/capability-model/capability-contract.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-14-05-14-rvv-compiler-produced-artifact-e2e-closure/prd.md`
* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-bundle-full-state-closure/prd.md`
* `.trellis/workspace/codex/journal-5.md` latest RVV compiler-produced artifact
  evidence entry.

Likely implementation surface:

* `lib/Target/RVV/RVVMicrokernel.cpp`
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`
* `test/Scripts/rvv-microkernel-bundle-e2e.test`
* `scripts/rvv_microkernel_e2e.py` only if runner front-door support is needed.

Initial diagnostic command proving the vsub gap before the fix:

```text
./build/bin/tcrv-opt test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline |
  python3 -c '<strip selected_binary_* and emitc_* identity attrs from tcrv_rvv.lowering_boundary and tcrv_rvv.i32_vsub_microkernel>' |
  ./build/bin/tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=<tmp>
```

Observed before the production fix: bundle export completed and emitted source,
header, object, and index despite missing vsub op-owned source identity.

## Implementation Summary

* `lib/Target/RVV/RVVMicrokernel.cpp` now requires selected lowering-boundary
  source identity for finite typed RVV binary artifact routes by dtype instead
  of only for the i32-vadd family.
* `scripts/rvv_microkernel_e2e.py` accepts
  `--lower-vector-i32-vsub-frontend` and routes it through the same selected
  binary lowering plus execution-planning path used by the selected RVV
  family route.
* `test/Scripts/rvv-microkernel-bundle-e2e.test` now covers the compiler-
  produced dynamic vector i32-vsub bundle route, selected kernel identity,
  generated source/header/object records, RVV intrinsic source, runtime counts,
  and external caller ABI expectation.
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir`
  now fails closed when op-owned source identity is stripped from the vsub
  lowering boundary and microkernel before bundle export.

## Validation Summary

* Focused RVV dialect/plugin/target/export builds passed, including
  `TianChenRVRVVDialect`, `TianChenRVRVVPlugin`, `TianChenRVRVVTarget`,
  `TianChenRVScalarTarget`, `tcrv-opt`, `tcrv-translate`, and focused C++ test
  binaries.
* Focused C++ tests passed:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`, and
  `tianchenrv-rvv-extension-plugin-test`.
* Focused lit tests passed for vector-dynamic vsub target-artifact export,
  vadd target-artifact regression, vector-dynamic vsub lowering, and
  `rvv-microkernel-bundle-e2e.test`.
* `python3 scripts/rvv_microkernel_e2e.py --self-test` passed.
* Manual dry-run vsub bundle probes succeeded for both plan-and-export and
  two-step selected planning/export routes, with runtime counts `7,16,23`,
  source route `tcrv-export-rvv-i32-vsub-microkernel-c`, selected kernel
  `frontend_vector_dynamic_i32_vsub`, and no ssh/runtime/performance claim.
* Manual fail-closed probe after the production fix rejected a planned vsub
  bundle with stripped selected source identity before writing a complete
  bundle index.
* Bounded ref-scan found no changed generic transform or core `tcrv.exec`
  sources. Remaining vadd references in `RVVMicrokernel.cpp` are existing
  vadd accessors/export wrappers, not selected-route preflight hardcoding.
