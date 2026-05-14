# RVV selected variant bundle full-state closure

## Goal

Finish one coherent production/default selected-variant bundle route for
`vector-dynamic-i32-vadd`: the compiler-produced selected RVV path must carry
and consume the same full materialized RVV state through selected emission
planning and target artifact bundle/source/header/object export. Full state
means op-owned source identity, selected SEW/LMUL/tail/mask config, and
runtime AVL/VL length authority from the materialized `tcrv_rvv` microkernel
path, not descriptor-adjacent assumptions.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round: clean worktree, HEAD
  `29a5903 fix(supervisor): resume review after codex transient`.
* The requested HEAD mismatch is real: HEAD is a supervisor-resume fix, while
  the latest RVV owner commit in the current log is
  `4df5aaa feat(rvv): carry selected source identity into bundle export`.
* `git show --name-status --stat --oneline --decorate HEAD` touches only
  supervisor task/spec/script files. `git show` for `4df5aaa` touches the RVV
  selected emission planning, selected config contract, RVV target export,
  scalar dispatch, and vector-dynamic bundle test owners.
* No `.trellis/.current-task` existed before this task; this task was created
  from the Hermes Direction Brief and set as current.
* The archived source-identity tasks closed the op-owned source identity slice,
  but this round must not stop at source identity. Closure requires selected
  config and runtime length authority to be consumed by the production bundle
  path as the same materialized selected state.

## Requirements

* Keep the module bounded to the existing `vector-dynamic-i32-vadd` /
  finite RVV binary `i32-vadd` selected route. Do not add new dtypes, families,
  LMUL matrices, route families, or performance work.
* The upstream selected route must still materialize a concrete
  `tcrv_rvv.i32_vadd_microkernel` from the compiler-produced path, carrying
  op-owned source identity.
* Selected emission planning must carry selected vector config as a
  contract-owned state group: selected SEW, selected LMUL, selected tail
  policy, selected mask policy, selected vector type, vector suffix, and setvl
  suffix.
* Selected emission planning must carry runtime length authority as a
  contract-owned state group: runtime element-count C name/role, runtime AVL
  source/role, runtime VL source/scope, descriptor-local element-count
  metadata, and dynamic source-tail authority where applicable.
* Artifact bundle, source, header, object, microkernel, and scalar-dispatch
  export must consume that same selected state through
  `RVVSelectedConfigContract` and `RVVRuntimeLengthContract`.
* Descriptor-only or stale descriptor-driven source/config/runtime authority
  must fail closed, be bypassed, or remain explicitly quarantined for the
  migrated vadd slice before source/header/object/bundle output.
* The test evidence must prove a compiler-produced vadd route reaches
  generated RVV artifact bundle output with identity/config/runtime data, not
  only a hand-authored target fixture.
* Core `tcrv.exec` and generic orchestration remain target-neutral. Do not add
  RVV semantic branches in core transforms or generic route dispatch.
* Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only be used for existing tooling, runners, artifact
  parsing, and validation helpers.

## Acceptance Criteria

* [x] Current repo state is reconciled before source changes: supervisor HEAD
      mismatch is documented and the RVV owner commit `4df5aaa` is inspected
      as the relevant prior implementation slice.
* [x] The compiler-produced `vector-dynamic-i32-vadd` route materializes
      `tcrv_rvv.lowering_boundary` and one concrete
      `tcrv_rvv.i32_vadd_microkernel` with op-owned source identity, selected
      config, runtime AVL/VL body, and runtime ABI role state.
* [x] `RVVBinarySelectedEmissionPlanning` and downstream target export owners
      consume selected config through `RVVSelectedConfigContract`, not through
      stale descriptor-local config or duplicated string authority.
* [x] The same selected path consumes runtime length through
      `RVVRuntimeLengthContract`, including runtime `n`/AVL/VL role metadata
      and descriptor-local element-count quarantine.
* [x] Generated source/header/object and target artifact bundle records expose
      source identity, selected config authority, runtime element-count C name,
      runtime AVL source/role, runtime VL source/scope, dynamic source-tail
      authority, descriptor-local capacity metadata, and `tcrv_rvv.setvl`
      provenance consistently.
* [x] Fail-closed coverage rejects missing, partial, stale, or conflicting
      selected source identity, selected config, runtime length role data, and
      descriptor-only production attempts before artifact bytes or complete
      bundle records are emitted.
* [x] Focused regressions for op-owned object artifact evidence, runtime-length
      consumption, selected-config consumption, selected-source bundle export,
      RVV microkernel export, scalar dispatch, VectorToExec, and LinalgToExec
      still pass.
* [x] A bounded scan confirms no new RVV semantic branch in core `tcrv.exec` or
      generic transforms, and no descriptor-only compute/config/runtime control
      is the default production route for the migrated slice.
* [x] If generated RVV source/object semantics change, collect focused
      `ssh rvv` compile/run evidence through the existing e2e path. If changes
      are validation/metadata/plumbing only, make no RVV runtime, correctness,
      or performance claim.
* [x] `git diff --check`, staged diff check, Trellis validation before finish
      and after archive, final clean worktree, and one coherent commit complete
      the round if finished.

## Definition Of Done

The default compiler-produced `vector-dynamic-i32-vadd` selected route is a
single connected production path: source lowering creates the exec/runtime ABI
surface, RVV plugin materialization creates the selected `tcrv_rvv` op and
state, selected emission planning packages that same state, and target artifact
export consumes it for source/header/object/bundle output. Descriptor-only
authority remains rejected or quarantined, and the report only claims evidence
actually run.

## Out Of Scope

* New RVV family, i64 expansion, new dtype matrix, LMUL matrix, broad smoke
  matrix, or performance tuning as the main result.
* Descriptor-to-C production export or descriptor element count/vector shape as
  authoritative compute, selected config, or runtime control.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic core orchestration.
* GCC/vendor compiler as the default route.
* Supervisor-loop work as the module result.
* Template, Toy, TensorExtLite, IME, Offload, or unrelated plugin changes
  except narrow regressions caused by shared validation.
* Runtime, correctness, or performance claims without focused real `ssh rvv`
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

* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-route-closure/prd.md`
* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-materialization-implementation/prd.md`
* `.trellis/workspace/codex/journal-5.md` selected RVV route/config/runtime
  sections.

Likely implementation surface:

* `include/TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/RVV/RVVScalarDispatch.cpp`
* `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
* Focused RVV microkernel, scalar-dispatch, VectorToExec, and LinalgToExec
  tests touched by the previous selected-route rounds.

## Implementation Summary

Completed in this round:

* Reconciled the run-state mismatch: current HEAD is the supervisor-resume
  commit `29a5903`, while the relevant RVV owner work is `4df5aaa`.
* Confirmed the current compiler-produced dynamic vector vadd route already
  carries source identity, selected config, runtime AVL/VL metadata, dynamic
  source-tail authority, and descriptor-local capacity into the selected
  emission plan and generated bundle records.
* Added dispatch-bundle metadata formatting helpers to
  `RVVBinarySelectedConfigContract` so the RVV+scalar bundle path serializes
  selected vector config, runtime VL boundary, and selected source identity
  through the same target-owned selected config/runtime-length contract object.
* Rewired `RVVScalarDispatch` bundle metadata construction to use those
  contract helpers instead of duplicating selected config/runtime formatting in
  the dispatch exporter.
* Extended the compiler-produced `vector-dynamic-i32-vadd` bundle lit test
  with fail-closed selected-state mutations:
  stale selected LMUL, missing selected LMUL metadata, stale runtime AVL
  source, and missing runtime VL source. Existing source-identity and
  descriptor-only-style source-kind rejection remains covered in the same test.

No core `tcrv.exec` dialect, generic orchestration pass, or descriptor-to-C
production route was added.

## Validation Summary

Repo-state reconciliation:

* `pwd && git status --short && git log --oneline -8`
* `git show --name-status --stat --oneline --decorate HEAD`
* `git show --name-status --stat --oneline --decorate 4df5aaa`

Focused build:

* `cmake --build build --target TianChenRVRVVTarget TianChenRVScalarTarget tcrv-translate tcrv-opt -j2`
* `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test -j2`

Focused C++ tests:

* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`

Focused lit:

* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel Target/RVVScalarDispatch`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport Transforms/VectorToExec Transforms/LinalgToExec`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target`

Other checks:

* Bounded core-neutrality scan:
  `rg -n "rvv|RVV|tcrv_rvv|selected_vector|runtime_avl|runtime_vl|descriptor_element_count" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-selected-variant-bundle-full-state-closure`

No `ssh rvv` evidence was run because generated RVV source/object semantics,
runtime behavior, correctness, and performance claims were not changed. This
round changed contract-owned metadata plumbing and fail-closed test coverage
only.

## Spec Update Judgment

No `.trellis/spec/` update is needed for this round. The relevant contracts are
already specified in the RVV plugin, lowering/runtime, EmitC route, capability
layering, and MLIR testing specs. This implementation makes production code and
tests conform more tightly to those existing specs rather than adding a new
architectural rule, command, route, or payload shape.
