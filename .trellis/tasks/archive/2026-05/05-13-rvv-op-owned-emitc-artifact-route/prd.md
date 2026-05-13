# RVV op-owned EmitC artifact production route

## Goal

Turn the existing dynamic RVV finite-binary source/header/object artifact
production route from "authority locked by tests and comments" into a route
whose default compute emission is owned by materialized `tcrv_rvv` extension
ops and the shared `TCRVEmitCLowerableOpInterface` / common EmitC lowering
payload. Dynamic i32-vsub is the primary path; dynamic i32-vadd is the required
regression.

The point of this round is structural production wiring, not another report or
annotation round. Descriptor mirrors may remain only as compatibility
diagnostics after op-owned authority has been validated.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial worktree state was clean.
* Initial HEAD was `9c2fb35 fix(rvv): lock extension-op emitc source authority`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the supplied worker direction before source edits.
* The previous completed task added useful generated artifact comments and lit
  checks around extension-op EmitC source authority, but its own PRD says the
  production route was already mostly correct before that round.
* The next bottleneck is therefore the production route itself: direct source,
  header, object, and plan-and-export bundle artifact generation must consume
  materialized `tcrv_rvv` ops and the common EmitC lowerable route as compute
  authority.
* Current real RVV hardware evidence, when making runtime/correctness claims,
  must come from `ssh rvv`.

## Scope

In scope:

* Existing dynamic i32m1 vsub direct source/header/object artifact route.
* Existing dynamic i32m1 vsub plan-and-export bundle route.
* Dynamic i32m1 vadd as the regression for the same op-owned route.
* RVV selected family/config/runtime-VL contract consumption where it feeds
  shape, SEW, LMUL, tail policy, mask policy, ABI names, runtime AVL/VL, and
  operand ordering.
* Common EmitC lowerable materialization as the source of intrinsic spelling
  and typed source-op provenance for artifact emission.
* Fail-closed behavior for stale descriptor-only or mismatched descriptor
  cases before direct or bundle artifact emission.

Out of scope:

* New vmul, i64, LMUL, dtype, vector-shape, or extension-family expansion.
* Generic plugin framework rewrite or route-registration-only cleanup as the
  main result.
* Descriptor-driven computation or direct descriptor-to-C export as the
  default path.
* Moving computation semantics into `tcrv.exec`.
* Python implementation of compiler core, dialects, passes, plugin registry,
  lowering, or emission.
* Performance claims or broad benchmark packaging.

## Requirements

* Compiler implementation remains in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Default artifact emission must derive add/sub semantics from the verified
  materialized RVV family body or common EmitC lowerable route payload, not
  from `tcrv_rvv.lowering_descriptor` arithmetic strings.
* `tcrv_rvv.i32_sub` must select `__riscv_vsub_vv_i32m1`; `tcrv_rvv.i32_add`
  must select `__riscv_vadd_vv_i32m1`.
* Operand ordering, selected shape, tail/mask policy, runtime AVL/VL boundary,
  and ABI parameter names must be obtained from the materialized op route or
  the shared selected-config/runtime ABI contract.
* Descriptor mirrors cannot independently select add vs sub, shape, runtime
  element-count authority, ABI names, or intrinsic spelling.
* Descriptor-only compute emission must be unsupported or fail closed before
  source/header/object/bundle output.
* Direct export and plan-and-export bundle export must share the same op-owned
  source authority.
* Existing plugin-manifest route activation, selected-config fail-closed
  validation, finite family registry behavior, fixed-vector vadd extent
  enforcement, and dynamic-tail/runtime-VL authority must stay intact.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as current before implementation.
* [x] Direct dynamic i32-vsub source/header emission obtains intrinsic
      semantics, operand order, ABI names, selected shape, and runtime VL
      metadata from materialized `tcrv_rvv` ops or the shared selected-config
      contract.
* [x] Direct dynamic i32-vsub object emission uses the same op-owned route and
      cannot fall back to descriptor-selected computation.
* [x] Dynamic i32-vadd remains a regression with the same route and emits
      `__riscv_vadd_vv_i32m1` from `tcrv_rvv.i32_add`.
* [x] Plan-and-export bundle emission uses the same op-owned authority as
      direct artifact export for vsub and vadd.
* [x] Descriptor/operator mismatches cannot affect emitted vsub/vadd semantics
      and fail before artifact emission.
* [x] Descriptor-only compute materialization is rejected or bypassed by the
      default route and is not required for successful source/header/object
      emission.
* [x] Tests or C++ coverage prove the production artifact route consumes the
      common EmitC lowerable materialization payload, not an independent
      descriptor-to-C exporter.
* [x] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`,
      `tianchenrv-target-artifact-export-test`, and any touched RVV/EmitC
      tests.
* [x] Focused lit/C++ checks cover dynamic vector vsub/vadd, RVV microkernel
      materialization, selected-config fail-closed regressions,
      `TargetArtifactBundleExport`, `EmissionManifest`, family registry,
      fixed-vector vadd, dynamic-tail/runtime-VL authority, plugin route
      activation, and representative linalg compatibility if touched.
* [x] Exact direct and bundle artifact commands are recorded for dynamic
      i32-vsub plus at least one i32-vadd regression.
* [x] Fresh `ssh rvv` evidence is collected for dynamic i32-vsub if emitted
      source/object behavior changes. If the round only changes fail-closed
      wiring with no runtime artifact delta, the report states the narrower
      claim.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If incomplete, it remains open with the exact next
      continuation point.

## Definition Of Done

The default dynamic i32-vsub artifact route is materially op-owned: direct and
bundle source/header/object emission consume materialized RVV extension ops and
the common EmitC lowerable payload for compute semantics. Dynamic i32-vadd
remains a regression, descriptor mirrors are compatibility diagnostics only,
and stale descriptor-only or mismatched descriptor cases fail closed before
artifact emission.

## Technical Approach

1. Inspect the current RVV direct and bundle artifact export route from source
   lowering through selected microkernel materialization, common EmitC
   materialization, target artifact export, and object production.
2. Identify any production branch that still chooses add/sub semantics, operand
   roles, shape, runtime count, ABI names, or intrinsic spelling from
   descriptor-local metadata rather than materialized op route contracts.
3. Rewire the narrow dynamic i32-vsub/vadd path so direct and bundle artifacts
   share one op-owned authority surface.
4. Add focused fail-closed and regression coverage around descriptor mismatch,
   descriptor-only input, direct artifacts, and bundle artifacts.
5. Run bounded local checks and fresh `ssh rvv` evidence if generated runtime
   artifacts changed.

## Completion Notes

Production authority change:

* Added a finite-family registry lookup by materialized RVV arithmetic op name.
* Rewired arithmetic `emitc.call_opaque` callee selection in
  `RVVMicrokernel.cpp` so add/sub/mul intrinsic spelling is derived from the
  typed compute step's `sourceOpName`, `sourceOpRole=compute`, and
  `TCRVEmitCLowerableOpInterface` provenance.
* The selected RVV record and typed source-op family must agree before the
  arithmetic intrinsic callee is chosen. Missing source provenance, unknown
  source ops, or selected-family/source-family disagreement now fail before
  source/header/object emission.
* Descriptor mirror metadata remains emitted as compatibility diagnostics, but
  it no longer independently chooses arithmetic intrinsic spelling for the
  production RVV EmitC route.

Artifact evidence root:

* `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z`

Direct artifact commands:

```bash
build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h
build/bin/tcrv-translate --tcrv-export-rvv-i32-vsub-microkernel-object artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o
build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-rvv-microkernel-object artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.o
```

Bundle artifact commands:

```bash
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/bundle/vector_dynamic_i32_vsub test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/bundle/vector_dynamic_i32_vsub/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/bundle/vector_dynamic_i32_vadd test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/bundle/vector_dynamic_i32_vadd/stdout.txt
```

Local checks:

```bash
cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
./build/bin/tianchenrv-target-artifact-export-test
./build/bin/tianchenrv-i32-binary-family-registry-test
./build/bin/tianchenrv-rvv-binary-planning-test
./build/bin/tianchenrv-rvv-extension-plugin-test
./build/bin/tianchenrv-rvv-selected-lowering-boundary-test
./build/bin/tianchenrv-emitc-lowerable-interface-test
python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Target/RVVMicrokernel Target/EmissionManifest Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vsub-to-exec.mlir
```

The focused lit run passed 50/50 with the existing warning that
`Transforms/LinalgToExec/linalg-i32-vsub-to-exec.mlir` contains no tests.

Hardware evidence:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/e2e --run-id 20260513T070822Z-rvv-op-owned-emitc-artifact-route-vsub --overwrite --timeout 120
```

Result: success on `ssh rvv`; artifact dir
`artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/e2e/20260513T070822Z-rvv-op-owned-emitc-artifact-route-vsub`;
runtime counts `7`, `16`, and `23`; selected kernel
`frontend_vector_dynamic_i32_vsub`.

## Technical Notes

Specs and context to read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/core-dialect/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-extension-op-emitc-source-authority/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-05-13-extension-manifest-target-support-route-activation/prd.md`
* `.trellis/workspace/codex/journal-5.md`

Initial code inspection targets:

* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/RVV/RVVScalarDispatch.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `tools/tcrv-translate/tcrv-translate.cpp`
* dynamic vector add/sub tests under `test/Transforms/VectorToExec/`
* RVV microkernel and target artifact bundle tests
