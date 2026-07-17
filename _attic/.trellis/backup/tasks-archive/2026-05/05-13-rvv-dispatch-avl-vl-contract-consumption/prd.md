# RVV dispatch AVL/VL contract consumption

## Goal

Make the RVV+scalar dispatch artifact route consume the already selected direct
RVV runtime AVL/VL contract as production authority for the embedded RVV
component. The dispatch source/header/object and bundle paths must fail before
artifact output when the selected RVV dispatch-case component has missing,
stale, duplicated, or conflicting runtime-element-count binding, and they must
publish the consumed contract as metadata without changing runtime behavior.

## What I Already Know

* Current HEAD at task start is
  `f0dc6a5 fix(rvv): bind direct runtime avl contract`; the worktree was clean.
* No `.trellis/.current-task` existed before this task was created from the
  Hermes direction brief.
* The archived prior task
  `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-avl-vl-boundary-authority/prd.md`
  added a direct `RVVBinarySelectedConfigContract` after IR-backed callable ABI
  resolution.
* Direct RVV source now records `selected_binary_config` and the resolved
  runtime-element-count C ABI name, including custom names such as `len`.
* The remaining module boundary is the RVV+scalar composite dispatch path: it
  embeds the selected RVV callable and scalar fallback, then emits dispatch
  source/header/object and bundle artifacts.
* The lowering/runtime spec requires RVV+scalar dispatch export to resolve
  callable parameters from the same IR-backed callable ABI plan for both
  branches and to keep dispatch availability as a separate guard parameter.
* The direct RVV contract is selected compile-time config plus descriptor-local
  capacity metadata plus runtime-element-count ABI name. Runtime `n`/AVL is
  still a runtime ABI/control value, while descriptor-local element count is
  metadata only.

## Requirements

* Dispatch export must consume the selected RVV dispatch-case component's direct
  selected-config/runtime AVL contract before source/header/object/bundle
  output.
* Dispatch preflight must fail closed when the selected RVV dispatch-case
  component has missing, stale, duplicated, or conflicting
  runtime-element-count binding.
* The RVV component runtime count must be tied to the dispatch runtime element
  count parameter, while dispatch availability remains a separate guard.
* Dispatch runtime ABI parameter order must remain explicit:
  `lhs`, `rhs`, `out`, runtime element count, dispatch availability guard.
* Generated dispatch source and bundle index must expose the selected RVV
  component contract as metadata, including runtime-element-count C name,
  selected vector config, dispatch role, and descriptor-local capacity.
* Source, header, object, and bundle routes must reuse the same production
  preflight and must not parse generated source comments as authority.
* Header/object helpers must keep using the same dispatch ABI plan as source.
* This task must not add new arithmetic families, dtype matrices, descriptor
  compute, generic RVV lowering, or Python compiler internals.

## Acceptance Criteria

* [ ] PRD and Trellis context are created before source edits.
* [ ] RVV+scalar dispatch source/header/object routes consume the selected RVV
      component's direct selected-config/runtime AVL contract.
* [ ] Missing, stale, duplicated, or conflicting RVV component
      runtime-element-count binding fails before dispatch artifact output.
* [ ] Dispatch source metadata exposes the RVV component
      `selected_binary_config` or equivalent contract fields:
      runtime-element-count C name, selected vector config, dispatch role, and
      descriptor-local capacity.
* [ ] Bundle/index metadata exposes the same contract fields for the dispatch
      external ABI group without moving RVV/scalar semantics into generic code.
* [ ] Dispatch callable runtime count is tied to the dispatch runtime count
      parameter, and dispatch availability remains a distinct target/export
      guard parameter.
* [ ] Focused positive FileCheck coverage proves dispatch consumption of the
      RVV runtime AVL contract.
* [ ] Focused negative coverage proves stale or missing component AVL binding
      fails closed.
* [ ] Existing direct RVV runtime-ABI role binding coverage remains valid.
* [ ] Focused build, C++ target artifact export test, focused lit, exact manual
      artifact commands, `git diff --check`, `git diff --cached --check`, and
      Trellis validation pass.
* [ ] If generated executable behavior changes, one bounded `ssh rvv` run
      confirms correctness. If only validation/provenance changes, final notes
      state why runtime behavior is unchanged.
* [ ] Task is finished/archived and one coherent commit records the round if
      complete.

## Definition Of Done

* Compiler implementation remains C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Python remains limited to evidence orchestration or artifact parsing.
* No descriptor-driven computation or descriptor-to-C fallback is added.
* No computation semantics are added to `tcrv.exec`.
* No RVV-specific semantic branch is added to shared core orchestration.
* No report-only, helper-only, or broad-smoke-only round is treated as complete.
* Runtime or correctness claims use real `ssh rvv` evidence when behavior
  changes; otherwise this round claims only compiler validation/provenance.

## Out Of Scope

* New arithmetic families, dtype expansion, performance claims, or broad test
  matrices.
* New RVV hardware proof unless generated executable behavior changes.
* Generic tensor/tile IR, generic linalg lowering, or generic RVV vector
  modeling.
* Descriptor-driven compute or descriptor-to-C replacement paths.
* Python implementation of compiler core, passes, dialects, plugin registry,
  capability model, lowering, or emission.

## Technical Notes

* Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/guides/index.md`, and
  `.trellis/spec/guides/capability-first-design-guide.md`.
* Prior PRD read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-avl-vl-boundary-authority/prd.md`.
* Initial source/test surfaces to inspect:
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `lib/Target/TargetArtifactBundleExport.cpp`,
  `test/Target/RVVScalarDispatch/`,
  `test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir`,
  and `test/Target/TargetArtifactBundleExport/`.

## Completion Notes

* Added
  `rvv::resolveRVVMicrokernelSelectedConfigContractAuthority()` so downstream
  target routes can consume the same validated direct RVV selected-config/
  runtime AVL contract that direct RVV source/header/object export uses.
* Rewired RVV+scalar dispatch selected-config construction to consume that
  direct RVV component authority. Dispatch now preserves the RVV component
  descriptor-local capacity from the typed direct microkernel record, validates
  the direct runtime element-count C name against the dispatch runtime count
  ABI parameter, and keeps dispatch availability as a separate guard.
* Extended composite target-artifact bundle metadata so RVV+scalar dispatch
  source/header/object bundle records carry dispatch-specific selected-plan
  metadata for the consumed RVV contract:
  runtime element-count C name, selected vector config, selected dispatch role,
  and descriptor-local element count as bounded component capacity metadata.
* Updated dispatch source/FileCheck coverage to prove the dispatch-level
  `selected_binary_config` now includes `descriptor_element_count=16`,
  `runtime_element_count_c_name`, and the separate dispatch guard C name.
* Added stale RVV component runtime-count metadata negative coverage that fails
  before dispatch source output.
* Updated the bundle export lit test and C++ target artifact export test to
  assert the new dispatch contract metadata is preserved in bundle metadata.
* Updated `.trellis/spec/lowering-runtime/emission-runtime-contract.md` with
  the executable RVV+scalar dispatch bundle contract.
* Self-repair performed: the first focused lit run failed because the new
  FileCheck lines expected the selected-config summary before the emitted
  runtime-guard comment. The check order was corrected and the same focused lit
  suite then passed.
* Generated executable behavior did not change. The C function bodies,
  dispatcher ABI order, embedded RVV/scalar calls, header prototype, and object
  route remain the same; this round changed production validation and metadata
  provenance only. No new runtime/correctness claim is made, so no new
  `ssh rvv` run was collected.

## Manual Artifact Evidence

* Artifact directory:
  `artifacts/tmp/rvv_dispatch_avl_vl_contract_consumption/`.
* Dispatch source:
  `build/bin/tcrv-opt test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c > artifacts/tmp/rvv_dispatch_avl_vl_contract_consumption/dispatch_i32_vadd.c`
* Dispatch header:
  `build/bin/tcrv-opt test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header > artifacts/tmp/rvv_dispatch_avl_vl_contract_consumption/dispatch_i32_vadd.h`
* Dispatch object:
  `build/bin/tcrv-opt test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object > artifacts/tmp/rvv_dispatch_avl_vl_contract_consumption/dispatch_i32_vadd.o`
* Bundle:
  `build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_dispatch_avl_vl_contract_consumption/bundle test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_dispatch_avl_vl_contract_consumption/bundle.stdout`
* Source evidence observed with `rg`: dispatch source contains one
  dispatch-level `selected_binary_config` with
  `runtime_element_count_c_name=n`,
  `dispatch_availability_c_name=rvv_available`,
  `descriptor_element_count=16`, and `selected_role=dispatch case`.
  The embedded direct RVV source still records `control_plane_runtime_avl`
  mapping the body index argument to runtime `n`.
* Header evidence: generated prototype is
  `void tcrv_dispatch_i32_vadd_dispatch_vadd(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n, int rvv_available);`
* Object evidence: `file` reports an ELF 64-bit little-endian RISC-V
  relocatable object, and `llvm-readobj` reports `Format: elf64-littleriscv`,
  `Arch: riscv64`, `Type: Relocatable`, and symbol
  `tcrv_dispatch_i32_vadd_dispatch_vadd` with no `main` symbol.
* Bundle index evidence: each source/header/object record in
  `rvv-scalar-i32-vadd-dispatch-external-abi.v1` carries
  `tcrv_rvv.dispatch_contract_runtime_element_count_c_name`,
  `tcrv_rvv.dispatch_contract_selected_vector_config`,
  `tcrv_rvv.dispatch_contract_selected_role`, and
  `tcrv_rvv.dispatch_contract_descriptor_element_count`.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* Initial lit attempt from repo root failed because `build/test/lit.site.cfg.py`
  expects to be run from `build/test`; rerun from `build/test` below.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVScalarDispatch|RVVMicrokernel|TargetArtifactBundleExport|target-source-artifact-routes|target-artifact-export-registry'`
  with 61/61 selected tests passed after the FileCheck ordering self-repair.
* Manual artifact source/header/object/bundle commands listed above.
* `git diff --check`
