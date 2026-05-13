# Bounded MLIR vector-to-RVV materialization front door

## Goal

Create the first bounded MLIR vector-dialect frontend route for TianChen-RV:
one exact i32 vector-add source wrapper lowers into the existing `tcrv.exec`
runtime ABI boundary, then reuses the RVV extension-family proposal,
selected/materialized `tcrv_rvv` microkernel body, common EmitC route,
source/header/object or bundle artifact export, and fresh `ssh rvv` evidence.

The production path for this task is:

```text
marked vector/arith i32 add source wrapper
  -> tcrv.exec.kernel with finite binary ABI roles
  -> RVV plugin selected finite i32-vadd variant
  -> materialized tcrv_rvv.i32_vadd_microkernel body
  -> common EmitC lowerable route
  -> RVV intrinsic source/header/object or bundle artifacts
  -> ssh rvv external ABI correctness evidence
```

## What I Already Know

* Initial repository state for this task: `pwd` is
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `51604fb fix(rvv): clarify finite binary emitc authority`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* Current `registerAllDialects` registers arith, func, linalg, and exec but not
  MLIR vector, so a real vector frontend requires vector dialect registration
  before `tcrv-opt` / `tcrv-translate` can parse vector input.
* Existing `--tcrv-lower-linalg-rvv-binary-to-exec` already centralizes the
  finite binary frontend ABI contract in
  `FiniteBinaryFrontendLowering.h`, including lhs/rhs/output mem_windows and
  runtime `n` parameter roles.
* Existing RVV plugin and target code already own selected vector shape,
  selected/materialized `tcrv_rvv.i32_vadd_microkernel`, runtime ABI
  validation, common EmitC route provenance, artifact source/header/object
  export, bundle export, and `ssh rvv` e2e tooling for source-fronted i32-vadd.
* The prior archived route-authority task made descriptor metadata
  non-authoritative mirror/config metadata and tightened source-fronted vadd/vsub
  artifact route checks.

## Requirements

* Add a real C++/MLIR frontend transform for exactly one vector/arith source
  pattern: i32 vector transfer-read of lhs and rhs, `arith.addi` on the vector
  values, vector transfer-write to output, and return.
* Keep the vector frontend bounded to i32-vadd and one representative static
  vector source shape. Do not introduce a generic MLIR vector backend.
* Reuse the existing finite binary frontend ABI contract and existing RVV
  plugin/materialization/export route. Do not add descriptor-driven computation
  or direct descriptor-to-C export.
* Preserve the same runtime ABI roles, selected vector config, AVL/VL surface,
  callable boundary, and generated source/header/object or bundle artifact
  authority already enforced for linalg-fronted i32-vadd.
* Make stale or missing selected/materialized body authority fail before target
  artifact output through the existing RVV plugin/export gates.
* Update the plan-and-export target artifact bundle front door so vector-fronted
  input can be consumed without hand-authored selected-path metadata.
* Keep shared orchestration target-neutral: no new RVV-specific semantic branch
  in core planning or artifact routing.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as the current Trellis task.
* [x] `tcrv-opt` accepts a checked-in vector/arith i32-vadd source fixture
      through a new bounded C++/MLIR transform pass and emits a parseable
      `tcrv.exec.kernel` with the finite binary runtime ABI mem_window and
      runtime_param roles.
* [x] Running the new vector frontend pass plus
      `--tcrv-execution-planning-pipeline` produces a selected RVV variant and
      a materialized `tcrv_rvv.i32_vadd_microkernel` with structured
      `setvl` / `with_vl` / load / add / store body.
* [x] Generated vector-fronted source uses RVV C intrinsics from the
      materialized `tcrv_rvv` body through the common EmitC route, including
      `TCRVEmitCLowerableOpInterface` route provenance.
* [x] Plan-and-export bundle coverage proves a vector-fronted input reaches
      source/header/object bundle artifacts without hand-authored selected
      metadata.
* [x] Negative coverage proves unsupported vector patterns or stale legacy
      descriptor metadata do not create an exec kernel or complete artifact.
* [x] Focused build covers touched targets, including `TianChenRVTransforms`,
      `TianChenRVTarget`, `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [x] Focused lit covers the new vector frontend path plus representative
      `RVVMicrokernel`, `TargetArtifactBundleExport`, `EmissionManifest`,
      `LoweringBoundary`, and `LinalgToExec` regression coverage.
* [x] Exact direct or bundle artifact commands for the vector-fronted input are
      recorded.
* [x] Fresh `ssh rvv` evidence proves the vector-fronted artifact for multiple
      runtime counts.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round
      if complete.

## Completed This Round

* Added the bounded `--tcrv-lower-vector-rvv-i32-vadd-to-exec` pass. It
  recognizes exactly one `func.func` vector/arith source wrapper with three
  `memref<?xi32>` arguments, zero index, zero i32 padding, two
  `vector.transfer_read` operations, one `arith.addi` over `vector<16xi32>`,
  one `vector.transfer_write`, and `func.return`.
* Registered the MLIR vector dialect in TianChen-RV tool dialect setup so
  checked-in vector-dialect source IR is parsed by `tcrv-opt` and
  `tcrv-translate`.
* Reused the existing finite binary frontend ABI contract to emit the same
  `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` runtime callable boundary
  as the linalg i32-vadd frontend.
* Wired the plan-and-export bundle front door to run bounded vector frontend
  lowering before the existing linalg frontend lowering, then reuse the same
  execution-planning and artifact-export pipeline.
* Added positive lit coverage for vector-to-exec lowering, RVV selected
  variant planning, materialized `tcrv_rvv.i32_vadd_microkernel` body, common
  EmitC route provenance, and generated RVV intrinsic source.
* Added plan-and-export bundle coverage proving a vector-fronted source fixture
  reaches source/header/object artifacts without hand-authored selected-path
  metadata.
* Added fail-closed vector negative coverage for non-vadd markers, stale legacy
  descriptor metadata, and unsupported vector arithmetic body shape.
* Updated the existing linalg plan-and-export marker-mismatch diagnostic
  expectation after the front-door failure text was generalized from bounded
  linalg frontend lowering to bounded source frontend lowering.
* Generated direct vector-fronted source/header/object artifacts and a
  vector-fronted plan-and-export bundle under
  `artifacts/tmp/bounded_vector_to_rvv_frontdoor/`.
* Collected fresh `ssh rvv` evidence through the C++ plan-and-export bundle
  front door under
  `artifacts/tmp/rvv_microkernel_bundle_e2e/20260513T-bounded-vector-to-rvv-frontdoor/`.
  Both source-built and bundle-object remote callers printed
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`.

## Exact Artifact Commands

```bash
build/bin/tcrv-opt test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-source-artifact > artifacts/tmp/bounded_vector_to_rvv_frontdoor/direct/frontend_vector_i32_vadd.c
build/bin/tcrv-opt test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-header-artifact > artifacts/tmp/bounded_vector_to_rvv_frontdoor/direct/frontend_vector_i32_vadd.h
build/bin/tcrv-opt test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --tcrv-lower-vector-rvv-i32-vadd-to-exec --tcrv-execution-planning-pipeline | build/bin/tcrv-translate --tcrv-export-target-artifact > artifacts/tmp/bounded_vector_to_rvv_frontdoor/direct/frontend_vector_i32_vadd.o
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/bounded_vector_to_rvv_frontdoor/bundle test/Target/TargetArtifactBundleExport/plan-vector-i32-vadd-and-export-target-artifact-bundle.mlir > artifacts/tmp/bounded_vector_to_rvv_frontdoor/bundle/stdout.txt
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --input test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --expect-selected-kernel=frontend_vector_i32_vadd --runtime-count=7 --runtime-count=16 --runtime-count=23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --run-id 20260513T-bounded-vector-to-rvv-frontdoor --overwrite --timeout 120
```

## Checks Run

* `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'VectorToExec|plan-vector-i32-vadd'`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'VectorToExec|plan-vector-i32-vadd|RVVMicrokernel|TargetArtifactBundleExport|EmissionManifest|LoweringBoundary|LinalgToExec'`
* `file artifacts/tmp/bounded_vector_to_rvv_frontdoor/direct/frontend_vector_i32_vadd.o artifacts/tmp/bounded_vector_to_rvv_frontdoor/bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-scalar-i32-vadd-dispatch-object.o`
* `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --input test/Transforms/VectorToExec/vector-i32-vadd-to-exec.mlir --expect-selected-kernel=frontend_vector_i32_vadd --runtime-count=7 --runtime-count=16 --runtime-count=23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --run-id 20260513T-bounded-vector-to-rvv-frontdoor --overwrite --timeout 120`
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-bounded-vector-to-rvv-frontdoor`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-13-bounded-vector-to-rvv-frontdoor`
* `git diff --cached --check`

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Python remains limited to tooling, artifact parsing, evidence runners, and
  small support scripts.
* Computation authority for emitted RVV code remains the selected/materialized
  `tcrv_rvv` family body and common EmitC route, not descriptors, Python, or
  `tcrv.exec` compute semantics.
* Runtime/correctness claims are bounded to named `ssh rvv` evidence and are not
  reported as performance or generic vector/RVV backend support.

## Out Of Scope

* Generic MLIR vector lowering, scalable-vector maturity, broad dtype/op/shape
  matrices, performance claims, LLVM/RISC-V backend lowering, or inline asm.
* Descriptor-driven computation, direct descriptor-to-C export, or moving
  arithmetic semantics into `tcrv.exec`.
* Adding unrelated RVV finite-family rows or expanding existing linalg source
  matrices unless required to protect the vector i32-vadd path.
* Helper-only, report-only, prompt-only, or broad smoke-test closeout as the
  main achievement.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/implementation-stack/index.md`,
  `.trellis/spec/implementation-stack/compiler-stack-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/{capability-first-design-guide,plugin-locality-review-guide,compute-boundary-review-guide}.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-finite-binary-emitc-route-ownership-cleanup/prd.md`
  and `.trellis/workspace/codex/journal-4.md` sessions 49-53.
* Initial source inspection covered:
  `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`,
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Transforms/Passes.{h,td}`,
  `lib/InitTianChenRVDialects.cpp`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVBinarySelectedLoweringBoundary.h`,
  `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`, and
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir`.
