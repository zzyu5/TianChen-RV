# RVV selected variant end-to-end route closure

## Goal

Close one coherent production/default selected-variant route for
`vector-dynamic-i32-vadd`: upstream Vector/Linalg-style source lowering must
materialize a selected RVV extension-family path whose op-owned source
identity, selected SEW/LMUL/tail/mask config, runtime AVL/VL role data, and
IR-backed callable ABI state are the same state consumed by RVV planning and
target artifact bundle/source/header/object export.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round: clean worktree, HEAD
  `c6a0b94 feat(rvv): carry selected source identity on microkernels`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* The previous completed task made materialized RVV microkernel ops carry
  op-owned selected source identity. That closes the source-identity slice but
  is not sufficient for this round.
* Prior selected-config and runtime-length tasks established
  `RVVBinarySelectedConfigContract` and `RVVRuntimeLengthContract` as the
  target-owned authority for selected vector config and runtime element-count /
  AVL / VL metadata.
* The current task must prove the compiler-produced dynamic vadd route reaches
  generated RVV artifact output through those contracts, not through stale
  descriptor-local compute/config/runtime assumptions.

## Requirements

* Keep the migrated slice bounded to the existing
  `vector-dynamic-i32-vadd` / RVV finite binary `i32-vadd` route. Do not add a
  new dtype, family, LMUL matrix, backend, or performance expansion.
* The upstream selected route must materialize a concrete
  `tcrv_rvv.i32_vadd_microkernel` with:
  * op-owned selected source identity;
  * selected vector-shape config for SEW 32, selected LMUL, tail policy, mask
    policy, vector type, vector suffix, and setvl suffix;
  * runtime AVL/VL control-plane body through `tcrv_rvv.setvl` and
    `tcrv_rvv.with_vl`;
  * IR-backed lhs/rhs/out `tcrv.exec.mem_window` roles and runtime element-count
    `tcrv.exec.runtime_param` role.
* RVV plugin legality/planning must consume the materialized selected
  microkernel/state through plugin-owned contracts and must not make
  descriptor-owned compute semantics authoritative.
* Target artifact source/header/object and bundle export must consume the same
  selected source identity, selected config, runtime length, runtime ABI, and
  source-tail authority state. Missing or mismatched fields must fail before
  artifact bytes or bundle records are emitted.
* Descriptor fields may only remain as bounded legacy mirror/capacity metadata
  after typed source authority, selected-config authority, and runtime-length
  authority are validated.
* Dynamic source-tail authority must remain explicit: runtime `%n` is the
  source upper bound, runtime element-count ABI parameter, and AVL source, while
  selected RVV tail/mask policy remains compile-time config.
* Core `tcrv.exec` and generic orchestration must remain target-neutral. Do not
  add RVV semantic branches to core passes, generic transforms, or shared route
  dispatch.
* Compiler production logic remains C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck. Python may only be used for existing tooling/evidence
  helpers.

## Acceptance Criteria

* [ ] `vector-dynamic-i32-vadd` lowering plus the existing execution-planning
      pipeline materializes `tcrv_rvv.lowering_boundary` and a concrete
      `tcrv_rvv.i32_vadd_microkernel` carrying op-owned source identity,
      selected config, runtime AVL/VL body, and runtime ABI role state.
* [ ] RVV selected planning/export validates and consumes the same
      `RVVBinarySelectedConfigContract` and `RVVRuntimeLengthContract` data for
      direct source/header/object and target artifact bundle records.
* [ ] Generated RVV artifact output exposes selected source identity, selected
      config authority, runtime element-count C name, runtime AVL source/role,
      runtime VL source/scope, dynamic source-tail authority, descriptor-local
      capacity metadata, and `tcrv_rvv.setvl` provenance without treating
      descriptor-local element count as runtime control.
* [ ] Plan-and-export bundle coverage proves the compiler-produced dynamic vadd
      route reaches generated RVV artifact output rather than only a
      hand-authored target fixture.
* [ ] Fail-closed coverage rejects missing, stale, partial, or conflicting
      selected source identity, selected config, runtime length role data,
      dynamic source-tail metadata, and descriptor-only production attempts
      before source/header/object or bundle output.
* [ ] Focused regressions for op-owned source identity, selected-config
      consumption, runtime-length consumption, op-owned object artifact
      evidence, RVV microkernel export, scalar dispatch, and target bundle
      export still pass.
* [ ] A bounded ref-scan confirms no core `tcrv.exec` / generic transform RVV
      semantic branch and no descriptor-only compute/config/runtime default
      production route for the migrated slice.
* [ ] If generated RVV source/object semantics change, focused `ssh rvv`
      compile/run evidence is collected through the existing e2e path. If this
      round changes only validation/metadata plumbing, no RVV runtime,
      correctness, or performance claim is made.
* [ ] `git diff --check`, staged diff check, Trellis validation before finish
      and after archive, final clean worktree, and one coherent commit complete
      the round if finished.

## Definition Of Done

The default compiler-produced `vector-dynamic-i32-vadd` selected RVV route is a
single connected production path: source lowering creates the exec/runtime ABI
surface, RVV plugin planning materializes the selected `tcrv_rvv` op and state,
and target artifact export consumes that exact selected state for direct and
bundle output. Descriptor-only authority remains rejected or quarantined, and
all claims are bounded to the evidence actually run.

## Out Of Scope

* New RVV family, i64 expansion, new dtype matrix, broad LMUL matrix, broad
  smoke matrix, or performance tuning.
* Descriptor-to-C production export, descriptor element count as runtime trip
  count, descriptor vector shape as selected config authority, or descriptor
  metadata as compute authority.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic core orchestration.
* GCC/vendor compiler as the default route.
* Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin changes
  except narrow regressions caused by shared validation.
* Runtime/correctness/performance claims without focused real `ssh rvv`
  evidence.

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

* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-materialization-implementation/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-artifact-consumption/prd.md`
* `.trellis/workspace/codex/journal-5.md` selected RVV route/config/runtime
  sections.

Likely implementation surface to inspect:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir`
* `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
