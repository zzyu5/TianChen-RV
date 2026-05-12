# RVV materialized i32-vadd artifact runtime ABI proof

## Goal

For the bounded source-fronted `i32-vadd` first slice, make the materialized
selected RVV body and the IR-backed runtime ABI role contract the authority for
generated source/header/object artifacts, then prove the generated artifact
path is executable on the RVV target when this round claims runtime behavior.

The production path is:

```text
linalg/source i32-vadd
  -> execution planning
  -> selected/materialized tcrv_rvv body
  -> runtime ABI-aware source/header/object target artifacts
  -> bounded ssh rvv compile/run evidence when runtime behavior is claimed
```

## What I Already Know

* Initial repository state for this task: `pwd` was
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `d859525 fix(rvv): require materialized selected body for export`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The immediately preceding selected-body task
  `.trellis/tasks/archive/2026-05/05-13-rvv-selected-variant-materialization/prd.md`
  made RVV export fail when the selected `tcrv_rvv` body is stripped, and its
  lit coverage proves concrete `tcrv_rvv.i32_vadd_microkernel`
  materialization.
* Earlier source-fronted proof tasks already demonstrated that generated
  `i32-vadd` source/header/object artifacts can execute on `ssh rvv`, including
  source-fronted bundle front-door evidence for runtime counts `7, 16, 23`.
  This task must not merely repackage those artifacts; it should make the
  compiler/export contract more explicit when production behavior already
  works.
* Relevant production surfaces from the brief are
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `include/TianChenRV/Support/RuntimeABIContract.h`,
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `test/Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir`,
  `test/Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir`,
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir`,
  `test/Target/EmissionManifest/`, and
  `test/Target/TargetArtifactExportTest.cpp`.
* The specs require compiler behavior to remain in C++/MLIR/LLVM/TableGen/
  CMake/lit, shared generic routing to stay target-neutral, and RVV runtime or
  correctness claims to use fresh `ssh rvv` evidence.

## Requirements

* The source-fronted and direct artifact routes must agree on selected variant,
  selected vector shape, runtime AVL/VL, emitted C signature, and generated
  selected `tcrv_rvv` body authority.
* Generated source/header/object artifacts must expose runtime element-count
  and buffer roles from the IR/runtime ABI contract, not descriptor-only
  mirrors.
* Target artifact export must fail before output when runtime ABI role binding
  is stale, missing, spoofed, or inconsistent with the selected executable
  boundary.
* Generated C source must contain RVV intrinsics derived from the materialized
  selected `tcrv_rvv.i32_vadd_microkernel` body.
* Source-fronted plan-and-export bundle evidence must remain on the generic
  target artifact front door, not only direct RVV compatibility helper routes.
* If executable C/header/object semantics change or this round raises a fresh
  runtime correctness claim, it must include a fresh `ssh rvv` compile/run log
  for the generated bounded i32-vadd artifact.
* If the artifact already works without production changes, this task must
  still improve the compiler/export contract explicitly rather than only add
  reports, broad smoke tests, or prompt/docs changes.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits, and the task is started as the current Trellis task.
* [x] One bounded compiler/export behavior change makes the selected
      materialized RVV body plus runtime ABI role contract more explicit in the
      source/header/object artifact path.
* [x] Source-fronted and direct artifact routes agree on selected variant,
      selected vector shape, runtime AVL/VL, runtime ABI parameter roles, and
      emitted C signature for `i32-vadd`.
* [x] Stale or missing runtime ABI role binding fails before source, header,
      object, or bundle output.
* [x] Generated source visibly contains RVV intrinsics derived from the
      materialized selected `tcrv_rvv` body and records runtime ABI role
      provenance from IR-backed callable boundaries.
* [x] Exact source-fronted `tcrv-opt ... --tcrv-execution-planning-pipeline |
      tcrv-translate ...` command is recorded.
* [x] Focused build includes touched targets, at minimum
      `TianChenRVTarget`, `TianChenRVTransforms`, `tcrv-opt`,
      `tcrv-translate`, and `tianchenrv-target-artifact-export-test` as
      applicable.
* [x] `tianchenrv-target-artifact-export-test` runs if target/export code
      changes.
* [x] Focused lit covers `Target/RVVMicrokernel`,
      `Target/TargetArtifactBundleExport`, `Target/EmissionManifest`,
      `Transforms/LoweringBoundary`, and `Transforms/LinalgToExec` as
      applicable.
* [x] Fresh `ssh rvv` evidence is included because runtime behavior and
      executable artifact semantics are claimed in this round.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] The task is finished/archived and one coherent commit records the round
      if complete.

## Completed This Round

* The RVV microkernel header now emits declaration-only provenance comments
  that name the selected materialized body, the runtime ABI roles, and the
  selected runtime element-count boundary before the C include guard.
* The i32-vadd runtime ABI role-binding lit coverage now checks the direct
  header exporter and fails closed on stale lhs role binding before artifact
  output.
* The direct source/header/object generation path and the target-artifact
  bundle front door were both regenerated for the bounded `i32-vadd` slice.
* Fresh `ssh rvv` evidence validated the generated source-and-bundle external
  caller path with runtime counts `7, 16, 23`.

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
* Python remains limited to tooling, artifact parsing, and bounded evidence
  runners.
* Computation semantics do not move into descriptors, Python, reports,
  generated harnesses, or `tcrv.exec`.
* Shared core orchestration does not gain RVV-family semantic branches outside
  existing plugin/target/export interfaces.
* Descriptor metadata remains a bounded legacy mirror/cross-check surface, not
  the authority for selected computation, ABI identity, or emitted body.
* Runtime evidence, if collected, is bounded to the generated source-fronted
  `i32-vadd` callable artifact and is not reported as performance evidence,
  broad RVV support, or generic backend coverage.

## Out Of Scope

* Generic RVV backend claims, performance claims, broad dtype/family matrices,
  new arithmetic families, or broad smoke matrix expansion.
* Moving away from extension family ops -> EmitC/RVV intrinsic source ->
  runtime C/C++ artifacts.
* Descriptor-driven computation, direct descriptor-to-C export as production
  authority, or Python compiler-core modeling.
* Editing route claims to imply runtime correctness or hardware execution
  without fresh `ssh rvv` evidence.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/index.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-rvv-selected-variant-materialization/prd.md`,
  `.trellis/tasks/archive/2026-05/05-13-source-fronted-rvv-runtime-artifact-proof/prd.md`,
  `.trellis/tasks/archive/2026-05/05-13-source-fronted-rvv-target-artifact-front-door/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-13-plugin-owned-rvv-artifact-route-authority/prd.md`.
* Relevant journal context read:
  `.trellis/workspace/codex/journal-4.md` sessions 47-51 and adjacent RVV
  runtime ABI / route authority summaries.
* Initial source inspection should focus on:
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `include/TianChenRV/Support/RuntimeABIContract.h`,
  `include/TianChenRV/Support/RuntimeABICallablePlan.h`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/TargetArtifactBundleExport.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/TargetArtifactBundleExport/`,
  `test/Target/EmissionManifest/`,
  `test/Transforms/LoweringBoundary/`,
  `test/Transforms/LinalgToExec/`, and
  `test/Target/TargetArtifactExportTest.cpp`.
