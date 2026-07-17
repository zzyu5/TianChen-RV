# RVV i64 binary sub/mul frontend artifact and ssh evidence

## Goal

Add `i64-vsub` and `i64-vmul` as first-class finite RVV binary families, then
prove they travel through the real marked linalg frontend, RVV plugin
selection/materialization, target artifact export, and real `ssh rvv`
correctness evidence for compiler-generated frontend artifacts.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting state was clean on `main` at commit `75b2cea`.
- No current Trellis task existed, so this task owns the bounded sub/mul module.
- The archived `05-10-rvv-binary-family-registry-owner` task completed the
  RVV-owned registry for `i32-vadd`, `i32-vsub`, `i32-vmul`, and `i64-vadd`
  only. It explicitly did not add i64 sub/mul or ssh evidence.
- The archived i64-vadd frontend/evidence tasks proved the current i64 add
  path and runner shape; this task must reuse that route style instead of
  creating a hand-written standalone C path.
- Current `LowerLinalgI32BinaryToExec.cpp` already looks up frontend lowering
  through `RVVBinaryFamilyRegistry`, so extending the registry should be the
  frontend owner if the downstream RVV dialect/plugin/export paths also accept
  the new families.
- Current RVV dialect and microkernel/export code still contain i64-vadd-only
  typed surfaces (`tcrv_rvv.i64_add`, `tcrv_rvv.i64_vadd_microkernel`) that
  must become bounded i64 add/sub/mul siblings inside the RVV target/plugin
  layer.

## Boundaries

- Compiler behavior remains C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Python may only orchestrate compiler tools, parse compiler-emitted metadata,
  generate bounded callers, run remote evidence, and write sanitized evidence.
- `tcrv.exec` remains compute-free. i64 sub/mul dataflow belongs only in the
  RVV extension dialect/plugin/target exporter.
- Generic plugin, selection, readiness, emission-plan, and target-artifact
  front doors must remain target-neutral; do not add RVV/dtype/family branches
  in shared core orchestration.
- No generic linalg lowering, new dtype, new LMUL, masks, scalar fallback,
  offload, IME/AME, performance claim, or broad smoke matrix.
- Do not commit `artifacts/tmp`, generated objects/sources, raw ssh logs,
  credentials, private SSH material, or build outputs.

## Requirements

1. Extend `RVVBinaryFamilyRegistry` with finite `i64-vsub` and `i64-vmul`
   descriptors, including family ids, frontend lowering strings, lowering
   descriptors, route ids, header/object route ids, runtime ABI identities,
   glue roles, external ABI component groups, C operator/type metadata,
   arithmetic intrinsic prefixes, function/header stems, and RVV dialect op
   names.
2. Keep `RVVBinaryDescriptor` and i64m1 shape-derived intrinsic construction
   registry-driven; full arithmetic intrinsics must be emitted by combining the
   family prefix with selected vector suffix.
3. Add bounded RVV dialect/plugin-local i64 sub/mul dataflow and microkernel
   surfaces as siblings of the existing i64 add slice. They must remain finite
   extension-dialect ops, not `tcrv.exec` compute semantics.
4. Allow marked `linalg.generic` `memref<?xi64>` wrappers using
   `arith.subi` / `arith.muli` and `tcrv_frontend_lowering = "i64-vsub"` /
   `"i64-vmul"` to lower to exec ABI boundaries and selected RVV artifacts.
5. Ensure RVV plugin proposal, selected lowering-boundary materialization,
   explicit microkernel validation, readiness, and emission-plan metadata are
   derived from the selected RVV family descriptor for all i64 add/sub/mul
   siblings.
6. Ensure target source/header/object exporters register and validate the new
   routes from descriptor metadata and emit `__riscv_vsub_vv_i64m1` and
   `__riscv_vmul_vv_i64m1` with the existing i64m1 load/store/vsetvl
   intrinsics.
7. Extend the evidence runner only as tooling so it can drive compiler-generated
   i64-vsub and i64-vmul frontend artifacts through the same bounded evidence
   contract as i64-vadd.
8. Preserve existing i32 add/sub/mul and i64-vadd spellings, comments, route
   identities, ABI role binding, and focused regression coverage.

## Acceptance Criteria

- [x] RVV registry lookup by family id, frontend lowering, and lowering
      descriptor succeeds for `i64-vsub` and `i64-vmul`, including route ids,
      runtime ABI names, scalar C types, and i64m1 selected-shape hooks.
- [x] `RVVBinaryDescriptor` derives i64-vsub/i64-vmul ABI and intrinsic names
      from registry plus i64m1 shape metadata.
- [x] RVV plugin tests prove proposal, materialization, lowering descriptor,
      selected i64m1 metadata, microkernel op, and emission plan for i64-vsub
      and i64-vmul.
- [x] Target artifact export tests prove source/header/object routes,
      runtime ABI role requirements, and emitted intrinsic spelling for the new
      i64 families.
- [x] lit/FileCheck covers marked linalg i64-vsub and i64-vmul frontend to RVV
      artifact paths, plus a focused fail-closed missing-i64m1-capability case.
- [x] Existing focused i32 and i64-vadd regression tests still pass.
- [x] Real `ssh rvv` evidence passes for compiler-generated frontend artifacts
      for both `i64-vsub` and `i64-vmul`, with sanitized evidence under
      `artifacts/tmp` and no committed artifacts.
- [x] Trellis task validation passes and the task is finished/archived only
      after module behavior and evidence are complete.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit/FileCheck for new i64-vsub/i64-vmul linalg artifact tests plus
  existing i64-vadd and i32 add/sub/mul regressions.
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py` if the runner changes.
- Focused script/lit dry-run coverage for i64-vsub and i64-vmul evidence
  modes.
- Real evidence:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vsub --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir --run-id <bounded-id> --overwrite --timeout 120 --ssh-target rvv`
- Real evidence:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vmul --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir --run-id <bounded-id> --overwrite --timeout 120 --ssh-target rvv`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if practical.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence`

## Out Of Scope

- No new i16, i8, floating-point, mask, LMUL, vector-shape search, scalar
  fallback, offload, IME/AME, or performance/benchmark work.
- No generic arbitrary linalg lowering beyond the finite marked binary cases.
- No Python implementation of compiler descriptor lookup, legality,
  materialization, lowering, or emission.
- No compute semantics in `tcrv.exec`.
- No hand-written standalone RVV C evidence that bypasses compiler-generated
  frontend artifacts.
- No broad report-only, prompt-only, dashboard-only, or smoke-matrix closeout.

## Technical Notes

- Required specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/design-boundaries.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Required archive context read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-binary-family-registry-owner/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-linalg-i64-vadd-frontend-rvv-ssh-evidence/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vadd-ssh-evidence/prd.md`
- Primary implementation surfaces:
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVV/RVVBinaryDescriptor.h`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Transforms/LowerLinalgI32BinaryToExec.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `scripts/rvv_microkernel_e2e.py`
  - `test/Transforms/LinalgToExec/`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Scripts/rvv-microkernel-e2e.test`
