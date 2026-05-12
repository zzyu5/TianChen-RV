# RVV runtime AVL/VL boundary authority

## Goal

Harden the existing bounded RVV `i32-vadd` selected-boundary and artifact path
so runtime element count, AVL, VL, and compile-time RVV config have explicit
IR/exporter authority from selected IR through typed `tcrv_rvv.setvl`,
`tcrv_rvv.with_vl`, typed body ops, and generated EmitC source/header/object
artifacts.

This task follows the completed RVV+scalar dispatch runtime ABI artifact proof.
The next useful boundary is semantic maturity inside the RVV path: runtime
`n`/AVL/VL must come from runtime ABI/control IR, while SEW/LMUL/tail/mask must
remain selected compile-time RVV config validated against the selected RVV
capability/config contract.

## What I Already Know

* Current HEAD is `1a69b3c fix(rvv): validate dispatch runtime abi artifact path`
  and the worktree was clean at task creation.
* No `.trellis/.current-task` existed before this task was created from the
  Hermes brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-rvv-scalar-dispatch-runtime-abi-artifact/prd.md`
  proved RVV+scalar `i32-vadd` source/header/object bundle generation and real
  `ssh rvv` external ABI execution for scalar and RVV branches.
* The lowering/runtime spec says runtime SSA/control values such as AVL, VL,
  pointer arguments, length `n`, and dispatch guards may be emitted only as real
  IR/control fields or generated ABI parameters.
* The same spec says SEW, LMUL, tail policy, mask policy, vector type, intrinsic
  suffix, and setvl suffix are selected compile-time RVV config, not runtime ABI
  values or descriptor-local element count.
* `tcrv_rvv.setvl` consumes a runtime AVL SSA value and produces
  `!tcrv_rvv.vl`; `tcrv_rvv.with_vl` consumes that VL token and scopes the
  finite body dataflow.
* Current RVV materialization already creates a one-argument microkernel body,
  uses that block argument as `setvl` AVL, and threads the resulting VL token
  into `with_vl`, loads, arithmetic, and store.
* Existing direct and dispatch tests already exercise several stale or missing
  runtime-element-count failures. This round must verify the actual production
  authority boundary and harden any first gap found, not add metadata-only
  comments.

## Requirements

* Runtime element count used as AVL must be resolved from the runtime
  ABI/control boundary, not from descriptor-only metadata or target-local
  inference.
* The selected typed RVV body must expose exactly one runtime index body
  argument for `n`/AVL, exactly one `tcrv_rvv.setvl`, exactly one
  `tcrv_rvv.with_vl`, and a valid VL dataflow chain.
* `tcrv_rvv.setvl` must use the runtime body argument as AVL; it must not use
  `tcrv_rvv.element_count`, a constant, or detached selected-plan text.
* `tcrv_rvv.with_vl` and all finite body ops must consume the VL token produced
  by `setvl`.
* SEW/LMUL/tail/mask and intrinsic suffix selection must remain selected
  compile-time RVV config validated from the selected RVV config/capability
  contract.
* Generated source/header/object provenance must make visible:
  runtime ABI parameter source for `n`,
  AVL source,
  `setvl` and `with_vl` VL dataflow,
  compile-time config source,
  typed family/body authority,
  and descriptor-local `element_count` as metadata only.
* Stale, missing, duplicated, or conflicting runtime AVL/VL binding must fail
  closed before source, header, object, or bundle export.
* Existing direct RVV and RVV+scalar dispatch `i32-vadd` artifact paths must
  still pass focused tests.
* Any production change affecting generated runtime behavior must receive one
  bounded real `ssh rvv` correctness run.

## Acceptance Criteria

* [ ] PRD and Trellis context are created for this task before source edits.
* [ ] Direct RVV `i32-vadd` source/header/object artifact routes validate the
      runtime `n`/AVL source through IR-backed callable ABI plus typed
      `setvl`/`with_vl` dataflow, not descriptor-only metadata.
* [ ] Missing, stale, duplicated, or conflicting runtime-element-count binding
      fails before direct artifact output.
* [ ] Generated direct RVV source visibly records callable
      `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` authority,
      `setvl` AVL input source, `with_vl` VL scope, and selected compile-time
      RVV config.
* [ ] Header/object helpers for the same direct RVV callable candidate reuse
      the same source-candidate preflight and cannot bypass runtime AVL/VL
      validation.
* [ ] RVV+scalar dispatch `i32-vadd` still preflights both callable components
      against the direct callable ABI contract and preserves explicit dispatch
      guard authority.
* [ ] Focused FileCheck coverage exists for positive runtime AVL/VL source
      provenance and fail-closed stale/missing AVL binding.
* [ ] Focused build, `tianchenrv-target-artifact-export-test`, focused lit,
      exact manual artifact commands, `git diff --check`, `git diff --cached
      --check`, and Trellis validation pass.
* [ ] If generated runtime behavior changes, one bounded `ssh rvv` run confirms
      correctness; otherwise the final report explains why existing generated
      behavior was unchanged.
* [ ] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* No descriptor-driven computation or descriptor-to-C fallback is introduced.
* No computation semantics are added to `tcrv.exec`.
* No RVV-specific branches are added to shared core orchestration.
* No new arithmetic family, dtype matrix, broad smoke matrix, performance
  claim, or report-only evidence task is treated as completion.
* Python remains bounded to evidence orchestration and artifact parsing.
* Generated runtime/binary artifacts remain under `artifacts/tmp` or remote
  scratch space and are not committed.
* Any RVV runtime/correctness claim is backed by real `ssh rvv` evidence and
  scoped only to the bounded path actually run.

## Out Of Scope

* New arithmetic families or dtype expansion.
* Generic linalg lowering, generic tensor/tile IR, or arbitrary RVV vector
  modeling.
* Performance measurement.
* Descriptor-driven computation or direct descriptor-to-C replacement paths.
* Python implementation of compiler internals.
* Broad hardware/runtime matrix validation.

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
  `.trellis/tasks/archive/2026-05/05-13-rvv-scalar-dispatch-runtime-abi-artifact/prd.md`.
* Initial source surfaces inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/RVVScalarDispatch/`, and
  `test/Target/EmissionManifest/`.
* First implementation hypothesis: direct RVV source candidate validation is
  already close, but this task should close any remaining bypass where selected
  metadata can claim runtime AVL/VL authority without the typed body and
  callable ABI agreeing.

## Completion Notes

* Added an explicit `RVVBinarySelectedConfigContract` to the direct
  `RVVMicrokernelRecord` after IR-backed callable ABI resolution.
* The contract is built from the static RVV finite-family registry entry, the
  selected vector-shape config, selected variant/role, descriptor-local bounded
  element count, and the actual runtime-element-count C ABI name resolved from
  `tcrv.exec.runtime_param`.
* The direct source/header/object record construction now fails before artifact
  output if the IR-backed runtime-element-count `tcrv.exec.runtime_param`
  `c_name` disagrees with the selected callable ABI parameter.
* Generated direct RVV source now prints one `selected_binary_config` comment
  that carries dtype, family, operator, SEW, LMUL, tail/mask policy,
  vector/setvl suffixes, selected path, descriptor-local capacity, and
  `runtime_element_count_c_name`.
* Generated direct RVV source now names the actual runtime count parameter in
  `control_plane_runtime_avl` and `dataflow_abi_roles`. The custom `len`
  runtime-count fixture proves this is not hard-coded to `n`.
* Focused FileCheck coverage was updated for default `n` and custom `len`
  runtime count names.
* Self-repair performed: the first implementation stored a selected-config
  contract pointer to `record.descriptor.family`, which became dangling after
  `RVVMicrokernelRecord` moved and caused `tcrv-translate` OOM in self-check
  lit routes. The fix rebuilds the contract from
  `lookupRVVBinaryFamilyRegistrationByID`, so the contract points at the
  stable static family registry entry.
* Spec update judgment: no `.trellis/spec/` changes were needed. The existing
  lowering-runtime and RVV plugin specs already define the runtime
  ABI/control-vs-compile-time-config layering this round enforced.
* Generated runtime behavior did not change: only record validation and source
  provenance comments changed. No new `ssh rvv` correctness claim is made in
  this task.

## Manual Artifact Evidence

* Artifact directory:
  `artifacts/tmp/rvv_runtime_avl_vl_boundary_authority_manual/`.
* Direct source:
  `build/bin/tcrv-translate --tcrv-export-rvv-microkernel-c test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir > artifacts/tmp/rvv_runtime_avl_vl_boundary_authority_manual/rvv_i32_vadd_len.c`
* Direct header:
  `build/bin/tcrv-translate --tcrv-export-rvv-microkernel-header test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir > artifacts/tmp/rvv_runtime_avl_vl_boundary_authority_manual/rvv_i32_vadd_len.h`
* Direct object:
  `build/bin/tcrv-translate --tcrv-export-rvv-microkernel-object test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir > artifacts/tmp/rvv_runtime_avl_vl_boundary_authority_manual/rvv_i32_vadd_len.o`
* Generic source front door:
  `build/bin/tcrv-translate --tcrv-export-target-source-artifact test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir > artifacts/tmp/rvv_runtime_avl_vl_boundary_authority_manual/rvv_i32_vadd_len.generic.c`
* Source evidence observed with `rg`:
  `selected_binary_config` records
  `runtime_element_count_c_name=len`;
  `control_plane_runtime_avl` records runtime `len` as the AVL source;
  `dataflow_abi_roles` records runtime `len` as the runtime element-count ABI
  parameter; `callable_runtime_param[0]` and
  `runtime_abi_parameter[3]` both record `c_name=len`; and the generated source
  still uses `tcrv_rvv.setvl` / `__riscv_vsetvl_e32m1`.
* Header evidence:
  `void tcrv_rvv_i32_vadd_microkernel_abi_names_rvv_first_slice(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t len);`
* Object evidence:
  `file artifacts/tmp/rvv_runtime_avl_vl_boundary_authority_manual/rvv_i32_vadd_len.o`
  reported `ELF 64-bit LSB relocatable, UCB RISC-V, RVC, double-float ABI`.
  `/usr/lib/llvm-20/bin/llvm-readobj --file-headers --symbols ...` reported
  `Format: elf64-littleriscv`, `Arch: riscv64`, `Type: Relocatable`, and the
  exported symbol
  `tcrv_rvv_i32_vadd_microkernel_abi_names_rvv_first_slice`; no `main` symbol
  was present.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|RVVScalarDispatch|EmissionManifest|TargetArtifactBundleExport|target-source-artifact-routes|target-artifact-export-registry'`
  with 73/73 selected tests passed after the contract-lifetime self-repair.
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-runtime-avl-vl-boundary-authority`
* `git diff --check`
* `git diff --cached --check`
