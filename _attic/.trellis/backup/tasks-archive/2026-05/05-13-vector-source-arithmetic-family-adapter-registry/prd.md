# Vector source arithmetic-family adapter registry

## Goal

Replace the current vector-source vadd-only arithmetic handling with a bounded
shared arithmetic-family adapter, then prove that adapter by adding exactly one
new dynamic vector `i32-vsub` source route. The intended production shape is a
single neutral source-frontdoor owner that can recognize marked MLIR
vector/SCF source bodies, cross-check the marker against the body arithmetic,
and hand the selected family into the existing RVV finite-binary path without
turning this into a generic vector backend.

The target route for this round is:

```text
marked MLIR vector/SCF source body
  -> shared source-frontdoor arithmetic-family adapter
  -> source-tail / active-lane authority cross-check
  -> tcrv.exec ABI / runtime AVL contract
  -> selected RVV i32-vsub family path
  -> RVV vsub microkernel materialization
  -> EmitC / source / header / object / bundle export
  -> fresh ssh rvv evidence for runtime counts 7, 16, 23
```

This is a bounded family-adapter round, not a generic vector compiler rewrite.

## What I Already Know

* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* The worktree was clean at session start and `HEAD` was
  `1707696 fix(vector): make dynamic tail authority explicit`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief before source edits.
* The previous dynamic vector vadd round made source `%n` tail authority
  explicit and carried it through lowering, selected metadata, generated
  artifacts, dispatch, and `ssh rvv` evidence.
* The previous source-frontend split round already introduced a neutral
  production source-frontdoor owner with reusable adapter boundaries.
* Repository evidence already has RVV and linalg `vsub` artifact support, but
  the vector source route still rejects `i32-vsub` as an unsupported source
  family.

## Scope

The production change must do one coherent thing:

* introduce a shared arithmetic-family adapter boundary for the vector source
  front door;
* add one bounded dynamic vector `i32-vsub` source route through that adapter;
* keep `i32-vadd` behavior unchanged;
* keep fixed-vector source contracts intact, including the existing `n == 16`
  enforcement and the current dynamic transfer-tail authority model;
* preserve the existing RVV finite-binary family path rather than inventing a
  parallel backend or descriptor-driven source exporter.

If the full sibling-family refactor proves too large, the minimum acceptable
result is:

* shared adapter boundary in production code;
* dynamic vector `i32-vsub` direct lowering and plan-and-export bundle route;
* truthful staging of any fixed-vector `vsub` route for a later task.

## Requirements

* Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python stays limited to tooling, evidence runners, artifact
  parsing, and small support scripts.
* The adapter must cross-check marker/body arithmetic in both directions:
  `arith.addi` routes stay on the `i32-vadd` family and `arith.subi` routes
  stay on the `i32-vsub` family.
* Marker/body mismatches must fail closed before source/header/object/bundle
  output.
* Family metadata, selected config, runtime AVL, active-lane authority, and
  source-tail policy must be carried consistently through lowering, selection,
  and export.
* Selected planning must materialize the existing RVV `i32-vsub` family ops and
  the generated source must use the RVV `vsub` intrinsic route, not a stale
  `vadd` path or descriptor-driven computation.
* Direct export and plan-and-export bundle export must both use the same
  neutral source-frontdoor route.
* Existing fixed and dynamic `vadd` behavior must remain unchanged.

## Acceptance Criteria

* [x] Trellis task, PRD, implement context, and check context exist before
      source edits, and the task is started as current.
* [x] A shared source-frontdoor arithmetic-family adapter exists in production
      code and is used by the dynamic vector `i32-vsub` route.
* [x] The dynamic vector `i32-vsub` source pattern is accepted only when the
      body uses `arith.subi` and the marker/body family cross-check agrees.
* [x] The reverse mismatch cases fail closed: `vadd` marker with `subi` body
      and `vsub` marker with `addi` body both reject before artifact output.
* [x] Selected planning materializes the existing RVV `i32-vsub` family ops,
      and generated source uses the `vsub` RVV intrinsic route.
* [x] Direct export and plan-and-export bundle export both produce artifacts
      through the neutral production path for the dynamic vector `i32-vsub`
      route.
* [x] Existing fixed and dynamic `i32-vadd` behavior remains unchanged.
* [x] Focused build covers the touched compiler targets, including
      `TianChenRVTransforms`, `TianChenRVTarget`, `tcrv-opt`, `tcrv-translate`,
      and `tianchenrv-target-artifact-export-test` as applicable.
* [x] Focused lit covers the new dynamic vector `i32-vsub` path, existing
      dynamic/fixed `vadd` regressions, vector invalid mismatch diagnostics,
      relevant `RVVMicrokernel`, `TargetArtifactBundleExport`,
      `EmissionManifest`, `LinalgToExec`, and `VectorToExec` coverage.
* [x] Exact direct and bundle artifact commands are recorded for the dynamic
      vector `i32-vsub` input.
* [x] Fresh `ssh rvv` evidence proves the generated dynamic vector `i32-vsub`
      artifact contract for runtime counts `7`, `16`, and `23`.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] One coherent commit records the completed round if the task finishes.

## Definition Of Done

* The new shared adapter is production code, not helper-only scaffolding.
* The adapter is consumed by both direct lowering and plan-and-export in the
  same round.
* The change does not weaken fixed-vector or dynamic transfer-tail authority.
* Existing RVV selected-body, common EmitC, source/header/object, bundle, and
  dispatch semantics remain behaviorally equivalent for the covered routes.
* The final report truthfully distinguishes local compiler/export validation
  from fresh `ssh rvv` runtime evidence.

## Completed This Round

* Added dynamic vector `i32-vsub` source-frontdoor support through the neutral
  `--tcrv-lower-source-rvv-binary-to-exec` production path.
* Kept the deprecated `--tcrv-lower-vector-rvv-i32-vadd-to-exec` adapter
  vadd-only while letting the neutral production path use the new
  arithmetic-family adapter.
* Added bounded dynamic vector source-kind support for
  `mlir-vector-scf-runtime-i32-vsub.v1` while preserving the existing dynamic
  vadd source-tail/active-lane authority and fixed-vector vadd source-extent
  contract.
* Added marker/body cross-checks for dynamic vector add/sub: `i32-vsub` marker
  with `arith.addi` body and `i32-vadd` marker with `arith.subi` body both fail
  closed.
* Added direct dynamic vector `i32-vsub` lit coverage proving selected
  `tcrv_rvv.i32_vsub_microkernel`, `tcrv_rvv.i32_sub`, vsub source-kind
  metadata, runtime `%n` AVL authority, and generated `__riscv_vsub_vv_i32m1`
  source output.
* Added plan-and-export target artifact bundle coverage for dynamic vector
  `i32-vsub`, including source/header/object bundle records and RVV+scalar
  dispatch source/header/object output through the neutral front door.
* Regenerated local direct source/header/object artifacts and a plan-and-export
  bundle under `artifacts/tmp/vector_source_arithmetic_family_adapter_registry/`.
* Collected fresh `ssh rvv` evidence for runtime counts `7`, `16`, and `23`
  using the plan-and-export direct RVV bundle route.

## Exact Artifact Commands

```bash
mkdir -p artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub artifacts/tmp/vector_source_arithmetic_family_adapter_registry/bundle/vector_dynamic_i32_vsub

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o

build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/vector_source_arithmetic_family_adapter_registry/bundle/vector_dynamic_i32_vsub test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/vector_source_arithmetic_family_adapter_registry/bundle/vector_dynamic_i32_vsub/stdout.txt
```

## Dynamic RVV Evidence

Command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/vector_source_arithmetic_family_adapter_registry/e2e --run-id 20260513T-vector-dynamic-i32-vsub-adapter-registry --overwrite --timeout 120
```

Result:

* Evidence file:
  `artifacts/tmp/vector_source_arithmetic_family_adapter_registry/e2e/20260513T-vector-dynamic-i32-vsub-adapter-registry/evidence.json`.
* `status = success`, `ssh_evidence.success = true`,
  `compiler_path_context.selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`.
* Both remote runs printed
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`:
  `logs/013_ssh_run_bundle_source_external_caller.log` and
  `logs/016_ssh_run_bundle_index_object_external_caller.log`.

## Checks Run

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vsub-to-exec|vector-dynamic-i32-binary-invalid|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle'`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vadd-to-exec|vector-i32-vadd-to-exec|vector-i32-vadd-invalid|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-i32-vadd-and-export-target-artifact-bundle|plan-linalg-i32-vsub-and-export-target-artifact-bundle|target-artifact-bundle-rvv-vsub'`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'VectorToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest|LinalgToExec'`
* `python3 scripts/rvv_microkernel_e2e.py --self-test`
* Direct and bundle artifact commands listed above.
* Dynamic vector `ssh rvv` evidence command listed above.
* `file` confirmed the direct object and bundle object are RISC-V ELF
  relocatables.
* `git diff --check`
* `git diff --cached --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-vector-source-arithmetic-family-adapter-registry`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-13-vector-source-arithmetic-family-adapter-registry`

## Self-Repair

* The first runtime-evidence attempt used the dispatch bundle lit fixture, but
  `scripts/rvv_microkernel_e2e.py` expects the direct RVV microkernel bundle
  route for this evidence mode. I reran the evidence with the direct dynamic
  vector `i32-vsub` input and the same plan-and-export bundle front door.

## Out Of Scope

* A generic MLIR vector backend.
* A broad family matrix or new dtype expansion.
* `i64` or `vmul` expansion.
* Performance claims.
* Descriptor-driven computation or direct descriptor-to-C export as the
  production authority.
* Weakening the fixed-vector `n == 16` contract or the dynamic transfer-tail
  contract.

## Technical Notes

* Specs read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant prior task context:
  `.trellis/tasks/archive/2026-05/05-13-05-13-source-frontend-lowering-ownership-split/prd.md`,
  `.trellis/tasks/archive/2026-05/05-13-rvv-i32-vsub-materialized-artifact-parity/prd.md`,
  and `.trellis/tasks/archive/2026-05/05-13-rvv-materialized-i32-vadd-artifact-runtime-abi-proof/prd.md`.
* Initial source inspection should focus on the shared source-frontdoor and
  RVV selection/emission path around:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir`,
  `test/Transforms/VectorToExec/vector-i32-vadd-invalid.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir`,
  and `test/Target/TargetArtifactBundleExport/target-artifact-bundle-rvv-vsub.mlir`.
