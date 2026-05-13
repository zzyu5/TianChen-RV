# RVV i32-vsub materialized artifact parity

## Goal

Make the source-fronted `linalg.generic` i32 subtraction route demonstrate the
same materialized selected-body and IR-backed runtime ABI artifact authority
that the current i32-vadd route demonstrates. The route must remain owned by
the finite-binary RVV family/plugin registry, not by copy/paste vsub special
casing in shared orchestration.

The production path for this task is:

```text
linalg.generic i32 subtract
  -> finite-binary RVV frontend lowering
  -> selected RVV variant/config/runtime contract
  -> materialized tcrv_rvv.i32_vsub_microkernel body
  -> source/header/object target artifact bundle export
  -> bounded ssh rvv compile/run evidence for multiple runtime counts
```

## What I Already Know

* Initial repository state for this task: `pwd` was
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `d374306 fix(rvv): expose runtime ABI authority in header artifacts`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The prior archived vadd task made generated headers expose selected-body
  authority, selected config/runtime AVL/VL, callable ABI provenance, runtime
  ABI parameter roles, and fresh RVV hardware compile/run evidence.
* The RVV specs define i32 add/sub/mul as one bounded finite binary family.
  Typed family/body authority must select computation, dtype, shape, route id,
  callable ABI, required capabilities, selected vector config, and emitted body.
  Legacy descriptors may only mirror or cross-check that authority.
* `tcrv_rvv.i32_vsub_microkernel` must carry one runtime index body argument,
  one `tcrv_rvv.setvl`, one matching `tcrv_rvv.with_vl`, two
  `tcrv_rvv.i32_load` ops, one `tcrv_rvv.i32_sub`, and one
  `tcrv_rvv.i32_store`.
* Source/header/object and bundle routes must validate stale or mismatched
  selected body, selected vector shape, and runtime ABI role binding before
  artifact output.
* Runtime/correctness claims for RVV require fresh `ssh rvv` evidence. Local
  lit/build evidence is compiler/export coverage, not hardware correctness.

## Requirements

* The source-fronted i32-vsub route must lower from the linalg subtraction body
  through the finite-binary frontend into an exec kernel with lhs/rhs/out memory
  roles and runtime element-count ABI boundary.
* The selected RVV path must materialize a family-correct
  `tcrv_rvv.i32_vsub_microkernel` body from the selected config/runtime
  contract, including selected `setvl`, matching `with_vl`, two loads, one
  subtract op, and one store.
* Generated source must use RVV intrinsics derived from the materialized
  `tcrv_rvv` body, including `__riscv_vsub_vv_*`, not stale vadd descriptors,
  names, markers, or caller arithmetic.
* Generated header, object, and bundle metadata must agree with source export on
  selected variant, selected vector shape, runtime AVL/VL boundary, runtime ABI
  parameter roles, callable signature, route id, runtime ABI kind/name, and
  selected component metadata.
* Stale or mismatched selected body, selected vector shape, descriptor mirror,
  or runtime ABI role binding must fail before source/header/object/bundle
  output.
* If existing behavior mostly works, this round must still make the
  family-owned compiler/export contract explicit in production code or focused
  tests; report-only closure is not sufficient.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as the current Trellis task.
* [x] One coherent compiler/export change or focused test contract proves i32
      vsub selected-body/runtime-ABI parity through the family-owned route.
* [x] Pipeline output for `linalg-i32-vsub-to-rvv-artifact.mlir` shows selected
      i32-vsub config/runtime metadata and a materialized
      `tcrv_rvv.i32_vsub_microkernel` with selected `setvl`, matching
      `with_vl`, two loads, `tcrv_rvv.i32_sub`, and one store.
* [x] Source export uses the selected RVV vsub intrinsic from the materialized
      body and rejects stale vadd/body/descriptor mismatches before output.
* [x] Header export records declaration-only selected-body and runtime ABI role
      authority for i32-vsub with the same ordered ABI signature as the source
      callable definition.
* [x] Object and bundle routes agree with source/header on selected family,
      route id, selected vector shape, runtime ABI kind/name, runtime ABI
      parameter roles, component metadata, and callable symbol.
* [x] Source-fronted plan-and-export bundle front door is exercised for i32-vsub
      and does not rely on hand-authored selected-path diagnostics.
* [x] Exact `tcrv-opt`/`tcrv-translate` source/header/object or bundle artifact
      commands for `linalg-i32-vsub-to-rvv-artifact.mlir` are recorded.
* [x] Focused build covers touched C++ targets, including `TianChenRVTarget`,
      `TianChenRVTransforms`, `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] `tianchenrv-target-artifact-export-test` runs if target/export code
      changes.
* [x] Focused lit covers applicable files under `Transforms/LinalgToExec`,
      `Target/RVVMicrokernel`, `Target/TargetArtifactBundleExport`,
      `Target/EmissionManifest`, and `Transforms/LoweringBoundary`.
* [x] Fresh `ssh rvv` evidence proves the generated bounded i32-vsub artifact
      for multiple runtime counts.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round if
      complete.

## Completed This Round

* Tightened the source-fronted i32-vsub artifact lit contract so
  `linalg-i32-vsub-to-rvv-artifact.mlir` now proves two load roles, selected
  `tcrv_rvv.i32_sub` body authority, selected runtime AVL/VL metadata, source
  and header runtime ABI roles, callable signature parity, and a stale runtime
  ABI role fail-closed case before header output.
* Tightened the source-fronted plan-and-export vsub bundle lit contract so the
  bundle index and generated dispatch source explicitly preserve ordered
  runtime ABI parameters, selected vsub config, runtime AVL/VL selected-plan
  metadata, and dispatch callable ABI metadata.
* Generated direct source/header/object artifacts from the source-fronted vsub
  fixture under
  `artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/direct`.
* Generated a local plan-and-export bundle under
  `artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/bundle`.
* Collected fresh source-fronted direct RVV i32-vsub bundle evidence under
  `artifacts/tmp/rvv_microkernel_bundle_e2e/20260513T-materialized-i32-vsub-artifact-parity`.
  The real `ssh rvv` run compiled and ran both the source-built object and the
  bundle object external callers, each printing
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.

## Exact Artifact Commands

```bash
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-source-artifact > artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/direct/frontend_i32_vsub.c
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-header-artifact > artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/direct/frontend_i32_vsub.h
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact > artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/direct/frontend_i32_vsub.o
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/bundle test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_i32_vsub_materialized_artifact_parity/bundle/stdout.txt
```

## Evidence Commands

```bash
python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --vector-shape i32m2 --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i32_vsub --runtime-count=7 --runtime-count=16 --runtime-count=23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --run-id 20260513T-materialized-i32-vsub-artifact-parity-dry-run --overwrite --timeout 120
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --vector-shape i32m2 --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i32_vsub --runtime-count=7 --runtime-count=16 --runtime-count=23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --run-id 20260513T-materialized-i32-vsub-artifact-parity --overwrite --timeout 120
```

## Checks Run

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'linalg-i32-vsub-to-rvv-artifact|plan-linalg-i32-vsub-and-export-target-artifact-bundle'`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|TargetArtifactBundleExport|EmissionManifest|LoweringBoundary|LinalgToExec'`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-i32-vsub-materialized-artifact-parity`
* `git diff --check`
* `git diff --cached --check`

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Python remains limited to tooling, artifact parsing, evidence runners, and
  small support scripts.
* Computation semantics do not move into descriptors, Python, generated
  harnesses, reports, or `tcrv.exec`.
* Shared core orchestration does not gain RVV-family semantic branches outside
  existing plugin/target/export interfaces.
* The selected `tcrv_rvv` body and IR/runtime ABI role contract remain the
  authority for generated artifacts; descriptor metadata is only a bounded
  mirror/cross-check.
* Runtime evidence is bounded to the generated i32-vsub callable artifact and
  is not reported as performance evidence, generic RVV backend support, or
  broad lowering coverage.

## Out Of Scope

* Generic RVV backend claims, performance claims, broad dtype/op matrices, or a
  new family expansion beyond one i32-vsub parity submodule.
* Moving away from extension family ops to the current EmitC/RVV intrinsic
  source and native compiler route.
* Descriptor-driven computation, direct descriptor-to-C export as production
  authority, or Python compiler-core modeling.
* Moving compute semantics into `tcrv.exec`, generated harnesses, or ad hoc
  direct exporters.
* Claiming runtime correctness or hardware behavior without fresh `ssh rvv`
  compile/run evidence.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-materialized-i32-vadd-artifact-runtime-abi-proof/prd.md`.
* Initial source inspection should focus on:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir`,
  `test/Target/TargetArtifactBundleExport/target-artifact-bundle-rvv-vsub.mlir`,
  `test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir`, and
  `test/Target/RVVMicrokernel/rvv-microkernel-header.mlir`.
