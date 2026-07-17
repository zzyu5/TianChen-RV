# RVV op-owned object artifact evidence closure

## Goal

Close one bounded RVV i32-vadd artifact route end to end: the selected
`tcrv_rvv` op/source identity and selected-boundary state must drive
source/header/object export, the object route must be exercised as a real
RISC-V relocatable artifact, and focused `ssh rvv` evidence must compile and
run the generated object through the external runtime ABI without broad
correctness or performance claims.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round: clean worktree, HEAD
  `341699e feat(rvv): require op-owned artifact source identity`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-rvv-selected-boundary-extension-op-production-route/prd.md`
  made selected lowering boundaries carry typed RVV binary source identity.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-artifact-default-emission/prd.md`
  made default i32-vadd source/header/object preflight require that identity.
* Existing runner tooling `scripts/rvv_microkernel_e2e.py` already has a
  bounded external-ABI route that can export source/header/object artifacts and
  optionally run them on `ssh rvv`.

## Requirements

* Keep the migrated slice to the existing default RVV i32-vadd family/config,
  preferably i32m1 unless code evidence shows a different already-supported
  i32 slice is the only landed source-identity route.
* RVV plugin planning and target export must carry the selected source
  identity plus selected SEW, LMUL, tail policy, mask policy, AVL/VL authority,
  runtime ABI kind/name/glue role, and ordered runtime ABI parameter roles into
  the source/header/object artifact handoff.
* Generated C source must remain on the clang-compatible `riscv_vector.h`
  intrinsic route and derive arithmetic from typed RVV op/source authority, not
  from descriptor-only metadata.
* Header export must expose only the declaration for the same selected
  external ABI surface as the source/object route.
* Object export must compile the validated source for the selected route into a
  non-empty RISC-V ELF relocatable object and must not hide a `main` or
  self-check harness in the default library object.
* Descriptor-only computation attempts must remain rejected, bypassed, or
  explicitly quarantined from the default production route for this slice.
* Add or update fail-closed coverage for missing, stale, or wrong selected
  source identity; stale selected boundary/config; wrong policy/config;
  missing runtime ABI role data; and descriptor-only production attempts.
* Keep `tcrv.exec`, generic transforms, shared construction code, Template,
  Toy, and TensorExtLite free of RVV semantic branches.

## Acceptance Criteria

* [x] The bounded i32-vadd route emits source, header, and object artifacts
      from the same op-owned selected source identity and selected-boundary
      state.
* [x] Generated source exposes selected source identity, selected vector config,
      runtime AVL/VL authority, and ordered runtime ABI role data.
* [x] Generated header prototype exactly matches the runtime ABI role contract.
* [x] Generated object is non-empty RISC-V ELF relocatable output produced from
      the validated RVV source route.
* [x] Focused `ssh rvv` evidence compiles and runs both source-built and
      generated-object external callers for bounded runtime counts, or records
      the exact concrete blocker if remote object execution is unavailable.
* [x] Missing/stale/wrong source identity, stale boundary/config, wrong
      policy/config, missing ABI role data, and descriptor-only production
      attempts fail before artifact output.
* [x] Core-pass neutrality is checked: no RVV semantic branches are added to
      `tcrv.exec`, generic transforms, or common construction code.
* [x] Focused C++/TableGen/build/lit checks pass for touched RVV dialect,
      plugin, target, generated headers, `tcrv-opt`, `tcrv-translate`, and
      affected artifact/export tests.
* [x] `git diff --check`, staged diff check, Trellis validation, archive, and a
      coherent commit complete the round if the task is finished.

## Definition Of Done

* Source/header/object generation for the migrated slice is covered by focused
  lit/FileCheck or C++ tests.
* The descriptor-only compute path is demonstrably not the default production
  path for this migrated slice.
* Focused `ssh rvv` compile/run evidence is captured in an artifact directory
  and summarized in the final report, unless blocked by a concrete toolchain or
  harness issue.
* Task status, PRD completion notes, workspace journal, archive state, and git
  commit truthfully reflect what was finished.

## Out Of Scope

* New RVV family, dtype, LMUL, arithmetic coverage, broad smoke matrix, or
  performance expansion as the main result.
* Descriptor-to-C production export or descriptor-owned computation semantics.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic transforms or common construction code.
* GCC or vendor compiler default route.
* Runtime/correctness/performance claims beyond the focused artifact evidence
  actually run.
* Template/Toy/TensorExtLite changes except narrowly necessary shared-interface
  regression fixes.

## Technical Approach

First inspect the current RVV source/header/object route, selected-boundary
identity validation, object exporter, and existing e2e runner. If the route
already emits real object artifacts, tighten the production contract where
evidence shows a gap: prefer runtime ABI role preflight, selected config
coherence, artifact output comments/index metadata, or focused fail-closed
coverage over helper-only changes. Then run the bounded local build/lit checks
and focused `ssh rvv` external-ABI evidence for the generated object route.

## Technical Notes

Specs and prior context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-artifact-default-emission/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-boundary-extension-op-production-route/prd.md`
* `.trellis/workspace/codex/journal-5.md` recent RVV artifact entries.

Likely implementation surface:

* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `test/Target/RVVMicrokernel/`
* `test/Target/RVVScalarDispatch/`
* `test/Target/TargetArtifactExportTest.cpp`
* `scripts/rvv_microkernel_e2e.py` only as runner/evidence tooling.

## Completion Summary

This round closed the bounded RVV i32-vadd object artifact evidence path
without adding core RVV branches:

* `lib/Target/RVV/RVVMicrokernel.cpp` now appends an object-only
  `.rodata.tianchenrv.rvv_artifact` section before compiling the validated RVV
  library source into a RISC-V ELF relocatable object.
* That section is derived from the same `RVVMicrokernelRecord` as source/header
  export: selected source kind, dtype/family/operator, executable microkernel
  op, EmitC source op/interface, selected vector config, runtime AVL/VL
  authority, runtime ABI kind/name/glue role, and ordered runtime ABI
  parameters.
* The default source/header outputs remain library-style callable artifacts
  without hidden `main` or self-check harnesses.
* `scripts/rvv_microkernel_e2e.py` now validates the generated object evidence
  payload before attempting external ABI ssh evidence.
* `test/Target/RVVMicrokernel/rvv-microkernel-object.mlir` now checks the
  direct and generic object section payload and adds an object-route
  fail-closed case for missing selected-boundary source identity.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` documents the
  bounded object evidence section as compiler-artifact provenance only, with no
  runtime logs, hardware success text, artifact paths, credentials,
  correctness claims, or performance claims.

Migrated slice:

* Family/config: RVV `i32-vadd`, `i32m1`, SEW 32, LMUL m1, tail/mask agnostic.
* Selected source identity consumed in object evidence:
  `selected_binary_source_kind`, `i32`, `i32-vadd`, `add`,
  `tcrv_rvv.i32_vadd_microkernel`, `tcrv_rvv.i32_add`,
  `TCRVEmitCLowerableOpInterface`.
* Runtime ABI role data carried:
  `lhs`, `rhs`, `out`, `n` with roles `lhs-input-buffer`,
  `rhs-input-buffer`, `output-buffer`, and `runtime-element-count`.

## Evidence

Focused `ssh rvv` evidence:

* Artifact directory:
  `artifacts/tmp/rvv_op_owned_object_artifact_evidence_closure/20260513T-rvv-op-owned-object-artifact-evidence-closure-i32-vadd/`
* Evidence JSON:
  `artifacts/tmp/rvv_op_owned_object_artifact_evidence_closure/20260513T-rvv-op-owned-object-artifact-evidence-closure-i32-vadd/evidence.json`
* Source-built external caller:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16`
* Generated-object external caller:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16`
* Object evidence section:
  `.rodata.tianchenrv.rvv_artifact`,
  schema `rvv-op-owned-object-artifact.v1`, 32 validated fields.
* Generated object hash:
  `c2eecf7e252614ad7c81c476c745f1fcda625f1d9f380f94dc0a2dcb6149e3d4`

## Checks Run

* `cmake --build build --target TianChenRVRVVTarget tcrv-translate -j2`
* `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-object.mlir` from `build/test`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-object.mlir Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVMicrokernel/rvv-microkernel-header.mlir Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir Target/RVVMicrokernel/rvv-microkernel-stale-boundary-fails.mlir Target/RVVMicrokernel/rvv-microkernel-missing-policy-fails.mlir Target/RVVMicrokernel/rvv-microkernel-selected-shape-metadata-fails.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'` from `build/test`
* `python3 scripts/rvv_microkernel_e2e.py --artifact-root artifacts/tmp/rvv_op_owned_object_artifact_evidence_closure --run-id 20260513T-rvv-op-owned-object-artifact-evidence-closure-i32-vadd --overwrite --expect-selected-kernel rvv_microkernel_manifest`
* Core neutrality scan: `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
* Descriptor quarantine scan: `rg -n "lowering_descriptor|descriptor-only|descriptor_compute_authority|descriptor metadata is non-authoritative" lib/Target/RVV/RVVMicrokernel.cpp lib/Plugin/RVV/RVVBinaryPlanning.cpp lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp test/Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir`
* `git diff --check`
