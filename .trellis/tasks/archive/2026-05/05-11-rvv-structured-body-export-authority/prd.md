# RVV structured microkernel body export authority

## Goal

Make the typed `tcrv_rvv.*_microkernel` structured body the checked source of
truth for bounded RVV direct microkernel artifact export. For the already
supported frontend-selected `i32-vmul` slice, target-owned export must inspect
and validate the body sequence before emitting RVV C/header/object metadata,
rather than allowing descriptor metadata alone to determine generated
intrinsics.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state was clean on `main` at
  `30eb357 test(rvv): enforce e2e evidence manifest authority`.
- No `.trellis/.current-task` existed before this task was created.
- The immediately previous task
  `.trellis/tasks/archive/2026-05/05-11-rvv-e2e-evidence-manifest-authority/`
  completed and must remain archived.
- Prior bounded evidence tasks already proved direct and dispatch
  `i32-vmul` RVV runtime correctness through fresh `ssh rvv` evidence.
- This task intentionally moves ownership back into C++/MLIR RVV target
  emission authority. Runner/evidence updates are allowed only as validation
  around compiler-derived artifacts.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only run existing e2e validation, parse artifacts, and capture
  evidence; it must not implement compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, or route selection.
- Preserve TianChen-RV boundaries: `tcrv.exec` remains compute-free; RVV
  computation and RVV artifact emission remain plugin/target-owned.
- The RVV direct export path must inspect typed `tcrv_rvv` body operations for
  the bounded slice before emitting source.
- The bounded body authority check must recognize the expected sequence:
  `setvl`, `with_vl`, two loads, matching arithmetic op, and store.
- Descriptor family and typed arithmetic op must agree. For example,
  selected `i32-vmul` must fail if the structured body contains
  `tcrv_rvv.i32_add` or `tcrv_rvv.i32_sub`.
- Selected SEW, LMUL, policy, vector type, and `setvl`/`with_vl` dataflow token
  types must agree. Stale `i32m1` body under selected `i32m2`, or stale `i32m2`
  body under selected `i32m1`, must fail closed.
- Runtime AVL/VL layering must remain correct: runtime AVL is SSA/control;
  selected SEW/LMUL/policy are compile-time config; VLEN/vlenb are target
  capability facts; and `element_count` remains descriptor-local metadata.
- Load/store buffer roles remain ABI-role metadata and must be verified enough
  to reject swapped or missing required roles for the bounded body.
- Source comments, manifest fields, and diagnostics may expose body-authority
  status, but comments alone are not the implementation. The
  exporter/verifier must actually reject mismatched bodies.
- If existing code already has most of the boundary, codify it with focused
  fail-closed tests and add only the smallest missing C++ behavior.

## Acceptance Criteria

- Direct RVV `i32-vmul` artifact export inspects typed `tcrv_rvv` body ops
  before emitting RVV source/header/object metadata.
- Accepted `i32-vmul` export is derived from a body containing selected
  `setvl`, `with_vl`, two loads, `i32_vmul`, and store with coherent roles and
  vector config.
- Mismatched arithmetic body fails closed with a clear diagnostic.
- Mismatched selected vector config versus body `setvl`/`with_vl` token/vector
  types fails closed with a clear diagnostic.
- Missing or swapped required load/store ABI roles fail closed for the bounded
  body.
- Focused C++ tests cover accepted body-authority export and rejected stale or
  mismatched typed bodies.
- Focused lit/FileCheck coverage proves generated direct RVV `i32-vmul` source
  still contains the expected intrinsic family and body-derived metadata after
  verifier/export enforcement.
- A focused direct dry-run through `scripts/rvv_microkernel_e2e.py` succeeds
  for frontend-selected `i32-vmul`.
- A fresh `ssh rvv` direct `i32-vmul` validation succeeds through the updated
  path, with the claim limited to that exact slice and no performance claim.
- `git diff --check` and Trellis task validation pass before finish/archive.

## Non-Goals

- No new RVV family, dtype, LMUL, operation matrix, performance benchmark,
  throughput/latency claim, or generic RVV backend claim.
- No MLIR vector/scalable-vector lowering route.
- No changes to IME, AME, Sophgo/offload, Template, Toy, or scalar behavior
  except untouched rebuild coverage where required.
- No compute semantics in `tcrv.exec`.
- No Python implementation of compiler internals.
- No helper-only, report-only, smoke-only, metadata-only, or runner-only
  closeout.

## Validation Plan

- Build focused targets:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test` if present, `tcrv-opt`, and
  `tcrv-translate`.
- Run focused C++ tests affected by body verifier/export ownership:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-rvv-binary-planning-test`.
- Run focused lit/FileCheck coverage for the direct RVV `i32-vmul` source path,
  including `test/Scripts/rvv-microkernel-e2e.test`.
- Run focused direct dry-run:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i32_vmul`.
- Run one fresh `ssh rvv` direct `i32-vmul` validation through the updated path
  and keep the evidence claim bounded to that exact run.
- Run `git diff --check`.
- Run `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-structured-body-export-authority`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical after focused checks pass.

## Technical Notes

- Specs to read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/experiment-reference.md`.
- Prior task context to read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-e2e-evidence-manifest-authority/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-i32-vmul-frontend-artifact-ssh-evidence/prd.md`.
- Primary implementation/test surfaces:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVVerifier.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`,
  `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`,
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `test/Plugin/RVVBinaryPlanningTest.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Dialect/RVV/RVVDialectTest.cpp`,
  `test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir`, and
  `test/Scripts/rvv-microkernel-e2e.test`.

## Definition Of Done

The task is done when the bounded direct RVV `i32-vmul` artifact export path
enforces typed RVV structured body authority in C++/MLIR target or verifier
code, focused mismatch tests fail closed, focused source/lit/dry-run checks
pass, fresh `ssh rvv` evidence validates the exact updated slice, the Trellis
task validates and archives, and one coherent commit records the work.

If unfinished, leave this task open and record the exact continuation point:
PRD repair, body verifier, exporter consumption, descriptor/body mismatch
diagnostics, selected config mismatch diagnostics, C++ tests, lit source check,
direct dry-run, ssh run, full check, Trellis validation, or commit.

## Completion Notes

- Implemented the smallest missing C++ target/export boundary in
  `lib/Target/RVV/RVVMicrokernel.cpp`: selected
  `tcrv_rvv.lowering_descriptor` is now resolved and checked against the typed
  `tcrv_rvv.*_microkernel` body family before artifact export can emit source,
  header, or object-derived records.
- Existing target-owned structured body verification already covered
  `setvl`, `with_vl`, the two load ops, arithmetic op, store op, SSA/dataflow
  chain, selected SEW/LMUL/policy/vector metadata, runtime AVL/VL layering,
  and load/store ABI roles. This round closed the remaining descriptor/body
  family mismatch that allowed stale selected descriptor metadata to coexist
  with a different typed microkernel family.
- Added C++ fail-closed coverage in
  `test/Target/TargetArtifactExportTest.cpp` for a selected
  `i32-vmul` descriptor paired with a typed
  `tcrv_rvv.i32_vadd_microkernel` body. The exporter now rejects it before
  emitting source.
- Added lit/FileCheck coverage in
  `test/Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir` proving that
  mutating the selected descriptor after materialization fails closed, and
  strengthened
  `test/Transforms/LinalgToExec/linalg-i32-vmul-to-rvv-artifact.mlir` to check
  body-derived dataflow/config source comments.
- Validation completed:
  focused build of `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`, `tcrv-opt`, and `tcrv-translate`;
  focused C++ tests; focused lit via the build-tree site config; direct
  dry-run `codex-body-authority-vmul-dry-run-20260511`; fresh `ssh rvv`
  direct `i32-vmul` run `codex-body-authority-vmul-ssh-20260511`;
  `git diff --check`; Trellis task validation; and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  with 205/205 tests passing.
- Runtime claim is limited to the exact fresh direct `i32-vmul` run above:
  generated RVV direct helper artifact handoff plus header/object external
  caller correctness on `ssh rvv`; no performance or generic RVV backend claim
  is made.
