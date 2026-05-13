# RVV runtime length artifact consumption

## Goal

Complete the bounded RVV runtime-length migration for the existing i32-vadd
i32m1 microkernel artifact route by making target source/header/object export
and the e2e object evidence path consume `RVVRuntimeLengthContract` as the
production source for runtime `n` / AVL / VL / `vsetvl` behavior. The runtime
element-count ABI parameter remains IR-backed through `tcrv.exec.runtime_param`;
`tcrv_rvv.setvl` and `tcrv_rvv.with_vl` remain the runtime AVL/VL control
surface; descriptor-local `element_count` remains bounded component-capacity
metadata only.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round: clean worktree, HEAD
  `499064e feat(rvv): materialize runtime length contract`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* The archived runtime AVL/VL ABI task introduced
  `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`, rewired selected
  config contracts, and added planning/export validation coverage.
* Current exporter code already builds an `RVVBinarySelectedConfigContract`
  from the selected microkernel record, validates selected-plan runtime
  metadata, and prints runtime-length comments.
* The remaining production gap is narrower: the actual EmitC route still forms
  the runtime AVL operand from the runtime ABI parameter directly, and object
  evidence writes runtime AVL/VL constants directly instead of using the
  runtime-length contract object. The e2e script validates source/header/object
  evidence, but object runtime-length fields are not yet part of the required
  evidence signature.

## Requirements

* Keep the migrated slice to the existing default RVV `i32-vadd`, `i32m1`,
  SEW 32, LMUL m1, tail/mask agnostic runtime-callable route.
* Source/header/object export must validate and consume
  `RVVRuntimeLengthContract` before generating artifact bytes.
* The generated source route must derive its `__riscv_vsetvl_*` runtime AVL
  operand from the contract's runtime element-count C name, not from
  descriptor-local `element_count` or a separate ad hoc parameter lookup.
* Object evidence must serialize runtime-length fields from the contract:
  runtime element-count C name, runtime AVL source/role, runtime VL
  source/scope, and descriptor-local element count.
* The e2e object validator must require those runtime-length evidence fields
  and cross-check them against generated source/runtime ABI evidence.
* Missing, stale, conflicting, or descriptor-only runtime length data must fail
  before source/header/object output for the migrated route.
* Core `tcrv.exec`, generic transforms, and common construction code must not
  gain RVV semantic branches.
* Python remains runner/evidence tooling only; compiler production logic stays
  in C++/MLIR/LLVM/TableGen/CMake/lit.

## Acceptance Criteria

* [x] `RVVRuntimeLengthContract` exposes the bounded production helpers needed
      by the RVV exporter to form and validate runtime AVL/VL emission.
* [x] RVV microkernel source export passes the selected runtime-length contract
      into the EmitC lowerable route and verifies the runtime-element-count ABI
      parameter mirrors the contract before emitting C.
* [x] Generated source still uses clang-compatible `riscv_vector.h` RVV
      intrinsics; the EmitC route provenance records
      `expression=<runtime n> - offset, c_type=size_t` for the migrated
      i32-vadd/i32m1 route and the MLIR Cpp emitter feeds the derived temporary
      into `__riscv_vsetvl_e32m1`.
* [x] Header and object export reuse the same selected config/runtime-length
      contract as source export.
* [x] Object evidence includes contract-derived
      `runtime_element_count_c_name`, `runtime_avl_source`,
      `runtime_avl_role`, `runtime_vl_source`, `runtime_vl_scope`, and
      `descriptor_element_count`.
* [x] e2e object validation requires those fields and rejects object evidence
      that lacks or disagrees with the source/runtime ABI contract.
* [x] Focused lit/C++ tests cover runtime-length source/header/object evidence
      and fail-closed metadata cases.
* [x] Focused build/tests, `git diff --check`, Trellis validation, archive, and
      one coherent commit complete the round if finished.

## Definition Of Done

* The production RVV artifact exporter consumes the runtime-length contract in
  the emitted control path, not only in comments or helper tests.
* Source/header/object evidence all report the same runtime-length authority.
* Descriptor-only `element_count` remains quarantined from runtime AVL/VL
  control.
* The final report states migrated slice, runtime-length contract consumption,
  selected boundary/source identity preservation, generated `vsetvl` behavior,
  fail-closed cases, checks, `ssh rvv` evidence or blocker, archive state, and
  commit hash.

## Out Of Scope

* New RVV family, dtype, LMUL matrix, i64 expansion, vsub/vmul expansion, broad
  smoke matrices, or performance tuning as the main result.
* Descriptor-to-C production export or descriptor `element_count` as runtime
  control authority.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic transforms or common construction code.
* GCC/vendor compiler as default route.
* Template/Toy/TensorExtLite changes except narrow regressions caused by shared
  artifact interfaces.
* Runtime/correctness/performance claims beyond focused evidence actually run.

## Technical Approach

Add small production helpers to `RVVRuntimeLengthContract`, then thread the
contract into the `RVVBinaryEmitCLowerable` source route so `vsetvl` operand
construction and validation use the contract's runtime element-count C name.
Use the same contract when serializing object evidence. Extend focused lit and
the e2e script to require the runtime-length evidence fields in source and
object artifacts. Keep the change local to RVV target/export and runner
validation.

## Implementation Summary

* Added `RVVRuntimeLengthContract::formatRemainingAVLOperandExpression()` so
  the target exporter has a contract-owned way to form the remaining-AVL
  operand from the runtime element-count C name and loop index.
* Threaded the selected `RVVRuntimeLengthContract` into
  `RVVBinaryEmitCLowerable` and fail closed if the IR-backed callable runtime
  ABI parameter does not match the contract's runtime element-count C name.
* Emitted stable `emitc.call_opaque_operand[...]` provenance for generated
  EmitC calls. This is the source-level evidence that the selected
  runtime-length contract drives the `__riscv_vsetvl_e32m1` AVL operand even
  though the MLIR Cpp emitter materializes `len - offset` into a temporary.
* Changed object evidence serialization to read runtime element-count C name,
  runtime AVL source/role, runtime VL source/scope, and descriptor-local
  element count from the selected runtime-length contract.
* Extended `scripts/rvv_microkernel_e2e.py` to parse
  `selected_runtime_vl_boundary`, require the generated `setvl` operand
  provenance, reject descriptor-count `setvl` calls, and require matching
  runtime-length fields in the object evidence section.
* Extended focused FileCheck/C++ tests for source/header/object runtime-length
  evidence and contract-derived remaining-AVL operand formatting.

## Evidence

Migrated slice:

* RVV `i32-vadd`, `i32m1`, SEW 32, LMUL m1, tail agnostic, mask agnostic,
  runtime-callable artifact route.

Generated source/object evidence:

* Selected runtime boundary keeps op-owned identity
  `selected_kernel=rvv_microkernel_manifest`.
* Runtime element-count ABI C name is `n` for the default e2e route and `len`
  in the focused ABI role-binding lit fixture.
* `selected_runtime_vl_boundary` reports
  `runtime_avl_source=runtime-element-count-abi-parameter`,
  `runtime_avl_role=runtime-element-count`,
  `runtime_vl_source=tcrv_rvv.setvl`,
  `runtime_vl_scope=tcrv_rvv.with_vl`, and
  `descriptor_element_count=16`.
* Object evidence section `.rodata.tianchenrv.rvv_artifact` uses schema
  `rvv-op-owned-object-artifact.v1` and validated 38 fields in the e2e run.

`ssh rvv` evidence:

* Artifact directory:
  `artifacts/tmp/rvv_runtime_length_artifact_consumption/20260513T-rvv-runtime-length-artifact-consumption-i32-vadd/`
* Evidence JSON:
  `artifacts/tmp/rvv_runtime_length_artifact_consumption/20260513T-rvv-runtime-length-artifact-consumption-i32-vadd/evidence.json`
* Mode/status: `ssh`, `success`.
* Runtime element counts: `7,16`.
* Source-built external caller marker:
  `tcrv_rvv_microkernel_external_abi_ok`.
* Generated-object external caller marker:
  `tcrv_rvv_microkernel_external_abi_ok`.
* Generated object hash:
  `da234107a9a7b79047150c7ac972a8860b2fc3d302b0daebcd154f093ddc5311`.
* Generated source hash:
  `59643c414baa27c7840cb404f0a399ccf7d2163bc1ced69d7a98cc4a94de3b47`.

## Checks Run

* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-(runtime-abi-role-binding|object|auto-materialization|descriptor-only-production-rejects)'` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-(runtime-abi-role-binding|object|auto-materialization|header|descriptor-only-production-rejects|stale-boundary-fails|missing-policy-fails|selected-shape-metadata-fails)|rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding'` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'` from `build/test`
* `python3 scripts/rvv_microkernel_e2e.py --artifact-root artifacts/tmp/rvv_runtime_length_artifact_consumption --run-id 20260513T-rvv-runtime-length-artifact-consumption-i32-vadd --overwrite --expect-selected-kernel rvv_microkernel_manifest`
* Core neutrality scan: `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
* Descriptor quarantine scan: `rg -n "lowering_descriptor|descriptor-only|descriptor_compute_authority|descriptor metadata is non-authoritative|descriptor_element_count|element_count" lib/Target/RVV/RVVMicrokernel.cpp lib/Plugin/RVV/RVVBinaryPlanning.cpp lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp test/Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir`
* `git diff --check`

Self-repair notes:

* A first direct-source lit invocation failed because `test/lit.cfg.py` requires
  generated site config fields such as `tianchenrv_obj_root`; reran lit from
  `build/test` with the generated `lit.site.cfg.py`.
* Adjusted the FileCheck order in
  `rvv-microkernel-runtime-abi-role-binding.mlir` to match the actual emitted
  metadata order, where EmitC route provenance appears before callable ABI
  comments.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/capability-model/capability-contract.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-avl-vl-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-op-owned-object-artifact-evidence-closure/prd.md`

Likely implementation surface:

* `include/TianChenRV/Target/RVV/RVVRuntimeLengthContract.h`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `scripts/rvv_microkernel_e2e.py`
* focused tests under `test/Target/RVVMicrokernel/`
* `test/Target/TargetArtifactExportTest.cpp` if C++ route/evidence assertions
  need extension.
