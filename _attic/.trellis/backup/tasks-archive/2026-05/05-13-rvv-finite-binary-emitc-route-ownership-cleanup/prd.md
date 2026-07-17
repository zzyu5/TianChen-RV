# RVV finite-binary EmitC route ownership cleanup

## Goal

Clean up the bounded RVV finite-binary production emission owner so the
already-proved i32-vadd and i32-vsub artifact paths visibly derive executable
semantics from selected/materialized `tcrv_rvv` family bodies and the shared
EmitC lowerable route, not from descriptor text or i32-vadd-shaped local
authority names.

The production path for this task is:

```text
selected finite RVV binary config
  -> materialized tcrv_rvv finite binary microkernel body
  -> RVV binary body verifier / TCRVEmitCLowerableOpInterface route
  -> common EmitC call-opaque source route
  -> RVV source/header/object/bundle target artifacts
```

## What I Already Know

* Initial repository state for this task: `pwd` was
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `835115e test(rvv): prove i32-vsub artifact parity`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The archived i32-vadd and i32-vsub PRDs show source/header/object/bundle
  export and fresh `ssh rvv` correctness evidence already work for bounded
  representative routes.
* The relevant specs require selected typed RVV family/body authority before
  descriptors are considered, generated `TCRVEmitCLowerableOpInterface`
  provenance before EmitC route construction, and fail-closed artifact export
  when selected body, descriptor mirror, runtime ABI, or selected vector shape
  metadata disagree.
* Current production code already has shared binary family descriptors,
  selected config contracts, body materialization, body verification, and EmitC
  route metadata. Inspection found remaining vadd-shaped local authority names
  in generic binary surfaces, including `kRVVI32VAddLoweringDescriptorAttrName`,
  `kRVVI32VAddElementCountAttrName`, and `RVVI32VAddDataflow*` aliases in the
  target/export path.
* The default automatic first-slice proposal may remain an explicit bounded
  i32-vadd default when no frontend or typed body authority is present, but
  descriptor attributes and binary dataflow route objects must not be named or
  interpreted as vadd-only authority once the selected route is generic.

## Requirements

* Rename, factor, delete, or explicitly quarantine vadd-shaped production
  authority names on generic RVV finite-binary planning/export surfaces.
* Keep the selected/materialized `tcrv_rvv` microkernel body as the source of
  arithmetic semantics for source/header/object/bundle artifact generation.
* Preserve descriptor fields only as selected config, bounded element-count, or
  legacy mirror/cross-check metadata after typed family/body or selected-source
  authority is established.
* Ensure both i32-vadd and i32-vsub still export source/header/object/bundle
  artifacts through the same production route and continue to fail before
  artifact output when the selected body is missing, stale, or mismatched.
* Keep shared generic passes target-neutral: no new RVV/dtype/family semantic
  branches in core orchestration and no computation semantics moved into
  `tcrv.exec`, descriptors, Python, reports, or generated harnesses.
* If inspection proves the route is already behaviorally clean, make the
  contract explicit through the smallest production assertion or boundary test
  that protects selected-body/EmitC authority.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as the current Trellis task.
* [x] Generic RVV finite-binary descriptor/element-count/dataflow authority
      surfaces no longer use misleading i32-vadd-shaped local names unless the
      name is explicitly limited to the default i32-vadd fallback proposal.
* [x] Source/header/object/bundle artifact paths still validate a materialized
      selected `tcrv_rvv` body and generated EmitC lowerable op-interface
      provenance before output.
* [x] Focused positive coverage demonstrates i32-vadd and i32-vsub remain on
      the same selected-family/EmitC route without adding a new op matrix row.
* [x] Focused negative coverage proves stale descriptor/body authority fails
      before source/header/object or bundle artifact output.
* [x] Exact representative artifact commands are recorded for i32-vadd and
      i32-vsub source/header/object or bundle export.
* [x] Focused build covers touched targets, including `TianChenRVTarget`,
      `TianChenRVTransforms`, `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] `tianchenrv-target-artifact-export-test` runs if target/export code
      changes.
* [x] Focused lit covers applicable RVV microkernel, target artifact bundle,
      emission manifest, lowering boundary, and linalg-to-exec tests.
* [x] Fresh `ssh rvv` evidence is collected only if executable source/object
      generation semantics change; otherwise the final report states why it was
      not needed.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round
      if complete.

## Completed This Round

* Renamed generic finite-binary production authority identifiers that still
  looked vadd-specific:
  `kRVVI32VAddLoweringDescriptorAttrName` became
  `kRVVLegacyLoweringDescriptorAttrName`,
  `kRVVI32VAddElementCountAttrName` became `kRVVElementCountAttrName`, and
  target/export `RVVI32VAddDataflow*` aliases were removed in favor of the
  existing `RVVBinaryDataflow*` verifier types.
* Kept the explicit default i32-vadd source kind limited to the bounded
  fallback proposal path where no frontend or typed body authority exists.
* Tightened `linalg-i32-vadd-to-exec.mlir` so the generated source route checks
  selected `tcrv_rvv.i32_add` dataflow emission, generated
  `TCRVEmitCLowerableOpInterface` provenance, route source-op ordering, and
  `emitc.call_opaque` compute mapping.
* Tightened `linalg-i32-vsub-to-rvv-artifact.mlir` with a source-fronted stale
  descriptor negative case proving a vadd descriptor mirror beside a selected
  vsub body fails through plugin legality before target source artifact output.
* Generated representative direct source/header/object artifacts and
  plan-and-export bundles for i32-vadd and i32-vsub under
  `artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/`.
* No fresh `ssh rvv` run was collected because this round changed production
  local names and FileCheck contract coverage only; generated executable
  source/object semantics and runtime arithmetic did not change.

## Exact Artifact Commands

```bash
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-source-artifact > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_direct/frontend_i32_vadd.c
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-header-artifact > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_direct/frontend_i32_vadd.h
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_direct/frontend_i32_vadd.o
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-source-artifact > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_direct/frontend_i32_vsub.c
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-header-artifact > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_direct/frontend_i32_vsub.h
build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir --tcrv-lower-linalg-rvv-binary-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_direct/frontend_i32_vsub.o
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_bundle test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_bundle/stdout.txt
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_bundle test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir > artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_bundle/stdout.txt
```

## Checks Run

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'linalg-i32-vadd-to-exec|linalg-i32-vsub-to-rvv-artifact'`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|TargetArtifactBundleExport|EmissionManifest|LoweringBoundary|LinalgToExec'`
* `file artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_direct/frontend_i32_vadd.o artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_direct/frontend_i32_vsub.o artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vadd_bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o artifacts/tmp/rvv_finite_binary_emitc_route_ownership_cleanup/vsub_bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vsub-dispatch-object.o`
* `git diff --check`
* `git diff --cached --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-finite-binary-emitc-route-ownership-cleanup`

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Python remains limited to tooling, artifact parsing, evidence runners, and
  small support scripts.
* Computation semantics stay in selected/materialized extension-family ops and
  the common EmitC route, not descriptor strings or core exec metadata.
* Descriptor metadata remains bounded mirror/config metadata and cannot select
  arithmetic semantics without typed RVV family/body authority.
* Runtime/correctness claims remain bounded to explicit `ssh rvv` evidence and
  are not reported as performance or generic RVV backend support.

## Out Of Scope

* Broad add/sub/mul/i64 matrix expansion, new arithmetic family rows, generic
  RVV backend claims, or performance claims.
* Replacing the current extension family ops -> EmitC -> RVV intrinsic source
  route with MLIR vector, LLVM scalable vector, inline asm, or backend patches.
* Python compiler-core implementation, direct descriptor-to-C compute export,
  or moving arithmetic semantics into `tcrv.exec` or shared core passes.
* Helper-only, report-only, or broad smoke/evidence packaging as the main
  achievement.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-i32-vsub-materialized-artifact-parity/prd.md`,
  `.trellis/tasks/archive/2026-05/05-13-rvv-materialized-i32-vadd-artifact-runtime-abi-proof/prd.md`,
  and `.trellis/workspace/codex/journal-4.md` sessions 50-52.
* Initial source inspection covered:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`,
  `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryVariantLegality.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`, and
  `lib/Target/RVV/RVVMicrokernel.cpp`.
