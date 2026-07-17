# RVV finite-binary family contract registry

## Goal

Complete the bounded RVV finite-binary family contract registry and migrate the
current production add/sub source-frontdoor and RVV target/export paths to use
that one registry for family identity, marker validation, selected metadata,
microkernel body identity, runtime ABI names, artifact routes, and EmitC
intrinsic identity.

The production route for this round is:

```text
marked linalg or vector source body
  -> RVV finite-binary family contract registry lookup
  -> shared source-frontdoor ABI / source-authority materialization
  -> selected RVV family metadata
  -> typed tcrv_rvv microkernel materialization
  -> common EmitC / target artifact source-header-object / bundle export
  -> focused local regression and fresh ssh rvv evidence
```

This is a registry migration round for existing add/sub behavior, not a new
family, dtype, shape, backend, or performance milestone.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* The worktree was clean at session start.
* Initial `HEAD` was `64d4136 feat(vector): add dynamic vsub source adapter`.
* No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes direction brief before source edits.
* The previous dynamic vector task proved `i32-vsub` through the neutral source
  front door, selected RVV ops, EmitC artifacts, bundle export, and `ssh rvv`
  counts `7`, `16`, and `23`.
* Current code already has `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  but the source-frontdoor path still keeps a separate frontend contract table
  and dynamic add/sub acceptance logic in
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h` and
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`.
* That split duplicates family ids, marker strings, dynamic vector source-kind
  acceptance, and add/sub validation rules across source support and RVV
  target/plugin code.

## Scope

In scope:

* Make the RVV family registry the contract source for current production
  `i32-vadd` and `i32-vsub` source-frontdoor family inference and marker
  validation.
* Keep existing linalg family behavior working; do not regress already
  production linalg add/sub/mul or i64 compatibility while migrating the
  current add/sub source-frontdoor path.
* Keep the support frontend lowering helpers as ABI/source-authority carriers,
  but stop treating their local add/sub strings as the independent source of
  family truth for RVV source lowering.
* Ensure selected planning, RVV materialization, target artifact export,
  direct export, and plan-and-export bundle continue to derive route/ABI/
  intrinsic metadata from the RVV family descriptor.
* Preserve fixed-vector `i32-vadd` extent enforcement and dynamic vector
  runtime-tail authority exactly.

Out of scope:

* No new `vmul`, `i64`, LMUL, shape, generic vector, or broad matrix expansion
  as the main result.
* No descriptor-driven computation or direct descriptor-to-C exporter.
* No movement of arithmetic compute semantics into `tcrv.exec`.
* No runtime/performance claim beyond required correctness/evidence checks.
* No report-only, helper-only, prompt-only, or broad smoke-test closeout.

## Requirements

* Compiler implementation remains C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck. Python remains evidence/tooling only.
* Source-frontdoor marker/body inference must query a shared RVV finite-binary
  family contract for accepted family id, frontend marker, source arithmetic op
  identity, dtype, source kind, and ABI C type spellings.
* `i32-vadd` marker with `arith.subi` body and `i32-vsub` marker with
  `arith.addi` body must still fail closed before artifact output.
* Dynamic vector `i32-vadd` and `i32-vsub` must keep distinct
  `tcrv_frontend_source_kind` values while sharing the same source-tail and
  runtime-AVL authority contract.
* Selected RVV planning and materialization must continue to use the same RVV
  family descriptor for family id, lowering descriptor mirror, microkernel op
  kind, arithmetic op kind, runtime ABI fields, artifact route ids, and EmitC
  intrinsic prefix.
* Direct source/header/object export and plan-and-export bundle export must
  remain equivalent for dynamic vector add/sub.
* Fixed-vector vadd `n == 16` enforcement and dynamic tail authority must not
  be weakened.

## Acceptance Criteria

* [x] The Trellis task has PRD, implement context, and check context before
      source edits, and it is started as current before implementation.
* [x] Source-frontdoor linalg and vector family inference uses the RVV family
      registry for family identity instead of independently formatting add/sub
      family strings.
* [x] Dynamic vector add/sub marker validation and source-kind selection are
      registry-backed, with unsupported dynamic vector families still rejected
      as non-generic-vector-backend cases.
* [x] Existing selected planning and target/export consumers are either already
      registry-backed or are migrated so add/sub route/ABI/intrinsic decisions
      are not duplicated outside the registry.
* [x] Duplicate add/sub string switches in the touched source/target path are
      deleted or reduced to adapter-local parsing with explicit comments.
* [x] Marker/body mismatch diagnostics remain fail-closed both directions.
* [x] Dynamic vector vadd and vsub artifacts still emit the correct
      `__riscv_vadd_vv_i32m1` and `__riscv_vsub_vv_i32m1` intrinsics through
      the neutral production route.
* [x] Direct export and plan-and-export bundle export remain equivalent for
      representative dynamic vector add/sub commands.
* [x] Relevant linalg add/sub regressions, fixed-vector vadd enforcement, and
      dynamic tail authority regressions pass.
* [x] Focused C++ build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] Focused lit covers dynamic vector vadd/vsub, vector binary invalid
      diagnostics, fixed-vector vadd regression, relevant linalg add/sub,
      RVV microkernel materialization, `TargetArtifactBundleExport`, and
      `EmissionManifest`.
* [x] Exact direct and bundle artifact commands are recorded for dynamic vector
      vadd/vsub, or a justified representative pair if source output is
      unchanged for one family.
* [x] Fresh `ssh rvv` evidence is collected for at least dynamic vector
      `i32-vsub` after migration; dynamic `i32-vadd` evidence is collected too
      if emitted vadd artifact code changes.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] If complete, the task is finished, archived, and committed as one
      coherent commit. If not complete, the task remains open with an exact
      continuation point.

## Definition Of Done

* The registry is production authority for the migrated add/sub family contract,
  not helper-only scaffolding.
* Source-frontdoor and target/export behavior consume the same family contract
  for the migrated path.
* The implementation deletes or narrows duplicate family branches in the
  touched path without broad unrelated refactors.
* Local regression and remote RVV evidence are reported with clear boundaries.

## Completed This Round

* Added source-frontdoor contract fields to
  `RVVBinaryFamilyDescriptor`: source arithmetic op name and dynamic vector
  source-kind metadata.
* Added RVV registry helpers for frontend contract lookup, source arithmetic
  inference, dynamic vector family acceptance, source-kind selection, and
  supported marker formatting.
* Migrated linalg and vector source-frontdoor family inference in
  `LowerSourceRVVBinaryToExec.cpp` from local family string construction to
  `lookupRVVBinaryFamilyRegistrationByFrontendSource`.
* Migrated dynamic vector marker validation and source-kind materialization to
  the RVV registry descriptor.
* Removed now-unused support-layer frontend marker/family lookup helpers so
  `FiniteBinaryFrontendLowering.h` remains the ABI/source-authority carrier
  rather than a second source of family truth.
* Preserved the legacy `--tcrv-lower-vector-rvv-i32-vadd-to-exec` adapter as
  vadd-only, while the neutral source path remains add/sub-capable.
* Confirmed dynamic vector vadd and vsub still emit the correct RVV intrinsic
  bodies and RISC-V relocatable artifacts.

## Exact Artifact Commands

```bash
mkdir -p artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vadd artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vsub

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.planned.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/vector-dynamic-i32-vadd.o

build/bin/tcrv-opt test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --tcrv-lower-source-rvv-binary-to-exec --tcrv-execution-planning-pipeline > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir
build/bin/tcrv-translate --tcrv-export-target-source-artifact artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.c
build/bin/tcrv-translate --tcrv-export-target-header-artifact artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.h
build/bin/tcrv-translate --tcrv-export-target-artifact artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.planned.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/vector-dynamic-i32-vsub.o

build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vadd test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vadd/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vsub test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vsub/stdout.txt
```

Artifact checks:

* Direct vadd source contains `__riscv_vadd_vv_i32m1`.
* Direct vsub source contains `__riscv_vsub_vv_i32m1`.
* Direct vadd/vsub objects and bundle vadd/vsub dispatch objects are RISC-V
  ELF relocatables.

## Dynamic RVV Evidence

Command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family i32-vsub --input test/Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir --expect-selected-kernel=frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --artifact-root artifacts/tmp/rvv_finite_binary_family_contract_registry/e2e --run-id 20260513T-rvv-finite-binary-family-contract-registry-vsub --overwrite --timeout 120
```

Result:

* Evidence file:
  `artifacts/tmp/rvv_finite_binary_family_contract_registry/e2e/20260513T-rvv-finite-binary-family-contract-registry-vsub/evidence.json`.
* `status = success`, `ssh_evidence = true`,
  `selected_kernel = frontend_vector_dynamic_i32_vsub`,
  `runtime_element_counts = [7, 16, 23]`.

## Checks Run

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-finite-binary-family-contract-registry`
* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'vector-dynamic-i32-vadd-to-exec|vector-dynamic-i32-vsub-to-exec|vector-dynamic-i32-binary-invalid|vector-i32-vadd-to-exec|vector-i32-vadd-invalid|linalg-i32-vadd-to-exec|linalg-i32-vsub-to-exec|plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle|plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle|plan-linalg-i32-vadd-and-export-target-artifact-bundle|plan-linalg-i32-vsub-and-export-target-artifact-bundle|RVVMicrokernel|EmissionManifest'`
  with 54 selected tests passed.
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest'`
  with 80 selected tests passed.
* Direct and bundle artifact commands listed above.
* `ssh rvv` evidence command listed above.
* `file` confirmed generated direct and bundle objects are RISC-V ELF
  relocatables.
* `git diff --check`
* `git diff --cached --check`
* Trellis validation before finish/archive and after archive.

## Self-Repair

* The first local direct artifact probe printed planned MLIR to the terminal
  instead of writing task artifacts. I reran the same direct vadd/vsub
  pipelines with explicit output paths and then generated source/header/object
  artifacts from those planned files.
* After the first green build, diff review showed the old unused support-layer
  family lookup/format helpers still formed a second registry surface. I
  removed those helpers, reran the focused build, C++ target-artifact test, and
  the broad focused lit group.

## Spec Update Judgment

No `.trellis/spec/` update was needed. The existing RVV plugin, EmitC route,
emission-runtime, plugin-protocol, and testing specs already require this
registry-backed family ownership and already distinguish intrinsic prefix
ownership from selected vector-shape suffix ownership.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-13-vector-source-arithmetic-family-adapter-registry/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-05-13-source-frontend-lowering-ownership-split/prd.md`

Initial code inspection focused on:

* `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`
* `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
* `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`
* `include/TianChenRV/Target/RVV/RVVVectorShape.h`
* `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
* `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
* `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`
* `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `lib/Target/Builtin/RVVScalarDispatch.cpp`
