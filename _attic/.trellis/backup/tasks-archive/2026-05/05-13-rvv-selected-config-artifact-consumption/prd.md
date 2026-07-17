# RVV selected config artifact consumption

## Goal

Make the bounded op-owned RVV `i32-vadd` / `i32m1` microkernel artifact route
consume the selected RVV compile-time config contract as the production source
for emitted RVV C vector type, intrinsic suffixes, load/store/arithmetic
intrinsic names, and `__riscv_vsetvl_*` spelling. Runtime `n` / AVL / VL
remains governed by `RVVRuntimeLengthContract`; descriptor-local element count
remains bounded component-capacity metadata and must not become config or
runtime-control authority.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round: clean worktree, HEAD
  `0c5d499 feat(rvv): consume runtime length in artifact export`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* The previous runtime-length task completed the neighboring runtime boundary:
  source/header/object and e2e object evidence now consume
  `RVVRuntimeLengthContract` for runtime element-count C name, runtime AVL
  source/role, runtime VL source/scope, and descriptor-local count evidence.
* `RVVBinarySelectedConfigContract` already records family, selected
  vector-shape config, selected path, runtime-length contract, and source extent
  contracts.
* Current target export still forms several production emission decisions
  through `RVVBinaryIntrinsicDescriptor` and `RVVIntrinsicConfig` built from the
  selected body/config verifier. That is useful cross-check evidence, but this
  round's migration should make the selected config contract the object passed
  into the EmitC/source/header/object artifact emitter for vector type,
  suffixes, and intrinsic name formation.
* Current `RVVBinaryMicrokernelBodyVerifier` already fails closed when
  `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` SEW/LMUL/policy, typed vector tokens,
  dataflow op family, or runtime ABI role bindings disagree with the selected
  descriptor/config. This should remain as a producer-side/body cross-check,
  not become the final artifact spelling authority.

## Requirements

* Keep the migrated slice bounded to the existing op-owned RVV `i32-vadd`,
  `i32m1`, SEW 32, LMUL m1, tail/mask agnostic runtime-callable source/header/
  object route.
* Add a selected-config emission surface that derives:
  * selected C vector type;
  * selected vector suffix;
  * selected `setvl` suffix and full `__riscv_vsetvl_*` intrinsic spelling;
  * load/store RVV intrinsic names;
  * arithmetic RVV intrinsic name by combining the family-owned suffix-free
    arithmetic prefix with the selected vector suffix.
* Rewire the RVV microkernel source/object production path so EmitC route
  construction and object evidence consume that selected-config emission surface
  for the migrated slice.
* Preserve `RVVRuntimeLengthContract` as the source for runtime element-count C
  name and `n - offset` AVL operand formation.
* Preserve typed body, boundary source identity, selected-plan metadata, and
  runtime ABI validation as fail-closed cross-checks before source/header/object
  bytes are emitted.
* Missing, stale, partial, descriptor-only, or conflicting selected SEW/LMUL/
  tail/mask/vector-type/suffix/setvl metadata must fail before artifact output.
* Descriptor metadata may only be validated as a legacy mirror after typed
  source authority and selected config authority are established.
* Core `tcrv.exec`, generic transforms, and common construction code must not
  gain RVV semantic branches.
* Python may only remain runner/evidence tooling; compiler production logic
  stays in C++/MLIR/TableGen/CMake/lit.

## Acceptance Criteria

* [x] `RVVBinarySelectedConfigContract` exposes a bounded selected-config
      emission view/helper used by the target exporter to derive selected
      vector type, vector suffix, setvl suffix, full `vsetvl`, load, store, and
      arithmetic intrinsic names.
* [x] RVV microkernel source export passes the selected-config emission view
      into the EmitC lowerable route; the route no longer relies on a separate
      descriptor/body-derived intrinsic config as its production spelling
      authority for the migrated `i32-vadd`/`i32m1` slice.
* [x] Generated source still contains clang-compatible `riscv_vector.h`,
      `vint32m1_t`, `__riscv_vsetvl_e32m1`, `__riscv_vle32_v_i32m1`,
      `__riscv_vadd_vv_i32m1`, and `__riscv_vse32_v_i32m1`, with comments
      showing selected config authority and runtime-length authority separately.
* [x] Header and object export reuse the same selected config contract as the
      source route; object evidence records selected vector config fields from
      the contract.
* [x] Focused fail-closed tests cover stale selected vector type/suffix/setvl
      metadata, missing selected config metadata, wrong policy/config, and
      descriptor-only config production attempts before source/header/object
      output.
* [x] Runtime-length artifact consumption, op-owned object artifact evidence,
      and existing RVV microkernel/scalar-dispatch focused regressions still
      pass.
* [x] A bounded scan confirms no changes under core `tcrv.exec` or generic
      transform code and no descriptor-only config/control default production
      route for the migrated slice.
* [x] Focused build/lit/C++ checks, `git diff --check`, staged diff check,
      Trellis validation before finish and after archive, and one coherent
      commit complete the round if finished.

## Definition Of Done

* The production artifact exporter consumes selected compile-time RVV config,
  not descriptor vector shape or body-derived intrinsic config, as the final
  source for emitted RVV type/intrinsic/vsetvl spelling in the migrated slice.
* Runtime length remains a separate runtime ABI/control contract and descriptor
  element count remains quarantined component-capacity metadata.
* The final report states migrated slice, selected config consumed by exporter,
  runtime length contract preservation, op-owned source identity preservation,
  generated RVV type/intrinsic/vsetvl behavior, fail-closed cases, checks,
  `ssh rvv` evidence or exact blocker, archive status, and commit hash.

## Out Of Scope

* New RVV family, i64 expansion, new dtype matrix, broad LMUL matrix,
  performance tuning, or broad smoke matrices as the main result.
* Descriptor-to-C production export or descriptor element count / descriptor
  vector shape as config or runtime control authority.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic transforms or common construction code.
* GCC/vendor compiler as the default route.
* Template/Toy/TensorExtLite changes except narrow regressions caused by shared
  target/export interfaces.
* Runtime/correctness/performance claims beyond focused evidence actually run.

## Technical Approach

Introduce a small selected-config emission view in the RVV target contract
layer, probably adjacent to `RVVBinarySelectedConfigContract`, and derive it
from the already validated selected family plus `RVVVectorShapeConfig`. Thread
that view through `RVVMicrokernelRecord` and `RVVBinaryEmitCLowerable` so
generated C source uses the selected config contract for C type and intrinsic
spelling. Keep `RVVBinaryMicrokernelBodyVerifier` validating body consistency
and producing cross-check facts, but make the final emission route compare
those facts to the selected-config view rather than treating them as the
authority.

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

* `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-length-artifact-consumption/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-avl-vl-abi-boundary/prd.md`
* `.trellis/workspace/codex/journal-5.md` entries for RVV selected config,
  runtime VL boundary, op-owned object artifact evidence, and runtime-length
  contract migration.

Likely implementation surface:

* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`
* `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `scripts/rvv_microkernel_e2e.py`
* focused tests under `test/Target/RVVMicrokernel/`
* `test/Target/TargetArtifactExportTest.cpp` if the C++ route/evidence
  assertions need extension.

## Implementation Summary

Completed for the bounded `i32-vadd` / `i32m1` slice and kept adjacent
`i32-vsub`, `i32-vmul`, `i32m2`, and `i64m1` tests consistent where they share
the generic binary RVV route:

* Added `RVVBinarySelectedConfigEmissionView` and
  `buildRVVBinarySelectedConfigEmissionView()` in
  `RVVSelectedConfigContract.h`.
* Added selected-config contract helpers for full `__riscv_vsetvl_*`,
  load/store, and arithmetic intrinsic spelling, plus emitted authority
  comments.
* Rewired `RVVMicrokernel.cpp` so the EmitC lowerable route, vector type
  mapping, call opaque callee selection, self-check harness first-vl probe, and
  object evidence consume `record.selectedConfigEmission`.
* Kept `RVVIntrinsicConfig` only as body-verifier cross-check evidence before
  artifact emission. The source route now errors if selected config emission
  fields disagree with the verified `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` /
  dataflow body.
* Extended e2e source validation to require
  `selected_config_emission_authority` with vector type/suffix, setvl suffix,
  full intrinsic names, tail/mask policy, and
  `source=RVVBinarySelectedConfigContract`.
* Extended focused lit and C++ tests to assert selected-config-driven emitted
  RVV type/intrinsic/vsetvl behavior.

## Evidence

Local checks:

* `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test tcrv-translate tcrv-opt -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir Target/RVVMicrokernel/rvv-microkernel-selected-shape-metadata-fails.mlir Target/RVVMicrokernel/rvv-microkernel-control-body-policy-mismatch-fails.mlir Target/RVVMicrokernel/rvv-microkernel-object.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel Target/RVVScalarDispatch`
* `git diff --check`
* `git diff --cached --check`

The first attempted lit invocation from the repo root failed before executing
tests because `lit.site.cfg.py` could not resolve `../../test/lit.cfg.py`; the
same test list passed from `build/test`, which is the correct lit workdir for
this build tree.

`ssh rvv` evidence:

* Command:
  `python3 scripts/rvv_microkernel_e2e.py --artifact-root artifacts/tmp/rvv_selected_config_artifact_consumption --run-id 20260513T-rvv-selected-config-artifact-consumption-i32-vadd --overwrite --arithmetic-family i32-vadd --vector-shape i32m1 --expect-selected-kernel rvv_microkernel_manifest --runtime-count 7 --runtime-count 16`
* Artifact directory:
  `artifacts/tmp/rvv_selected_config_artifact_consumption/20260513T-rvv-selected-config-artifact-consumption-i32-vadd`
* Result: status `success`, mode `ssh`, target `rvv`, compile flags
  `-O2 -march=rv64gcv -mabi=lp64d`.
* Remote source-built caller stdout:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16`.
* Remote generated-object caller stdout:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16`.

Core neutrality / descriptor quarantine:

* Changed files are limited to RVV target contract/export code, the RVV e2e
  evidence script, RVV microkernel tests, and the target artifact C++ test.
* No `tcrv.exec` dialect, generic transform, or common construction code was
  changed.
* Descriptor-local element count remains capacity metadata and
  `RVVRuntimeLengthContract` still owns runtime AVL/VL naming and
  `len - offset` / `n - offset` formation.
