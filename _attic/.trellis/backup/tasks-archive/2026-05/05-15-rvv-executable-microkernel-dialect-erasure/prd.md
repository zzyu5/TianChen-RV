# RVV executable microkernel dialect erasure

## Goal

Delete the remaining active RVV executable microkernel dialect wrapper surface so the future RVV compiler path must rebuild from explicit RVV extension-family control/dataflow ops plus a materialized EmitC route, not from preserved `tcrv_rvv.*_microkernel` wrapper ops, selected-path route metadata, or descriptor-local `element_count` bodies.

## Direction Brief Facts

- Previous commit `f66db51` erased RVV binary finite-family planning/export authority and left the worktree clean.
- The next old authority is in `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` and `lib/Dialect/RVV/IR/RVVDialect.cpp`.
- The active deletion boundary includes `I32VAddMicrokernelOp`, `I32VSubMicrokernelOp`, `I32VMulMicrokernelOp`, `I64VAddMicrokernelOp`, `I64VSubMicrokernelOp`, `I64VMulMicrokernelOp`, microkernel family spec structs, verifier branches that tie arithmetic/dataflow legality to enclosing microkernel op names, and tests/fixtures whose purpose is to preserve that executable surface.
- This is a Wrong Logic Deletion Campaign round: deletion before rebuild; do not add replacement architecture, compatibility modes, helper wrappers, new executable routes, or descriptor tests.

## Repo Facts Observed

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` still defines the six `tcrv_rvv.*_microkernel` ops with executable microkernel summaries, `source_kernel`, `selected_variant`, `origin`, `role`, `element_count`, `required_capabilities`, `required_march`, selected vector-shape metadata, and selected MABI metadata.
- `lib/Dialect/RVV/IR/RVVDialect.cpp` still contains `I32MicrokernelFamilySpec`, `I64MicrokernelFamilySpec`, enclosing microkernel arithmetic detection, microkernel structured-control verifiers, and `I32V*MicrokernelOp::verify` / `I64V*MicrokernelOp::verify` entry points.
- RVV dataflow verifiers currently require `i32_*` and `i64_*` dataflow ops to be nested under the deleted microkernel wrapper ops, which would keep the wrapper names as semantic authority.
- Focused tests under `test/Dialect/RVV`, `test/Transforms/VariantMaterialization`, `test/Transforms/VariantSelection`, `test/Transforms/ExecutionPlanning`, `test/Target/RVVScalarDispatch`, and `test/Target/ArtifactExport/Inputs` still contain hand-authored `tcrv_rvv.i32_vadd_microkernel` / `i32_vsub_microkernel` / `i32_vmul_microkernel` attachments.
- `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and `.trellis/spec/testing/mlir-testing-contract.md` still contain historical RVV microkernel wording that should be rewritten when it protects the deleted wrapper surface.

## Requirements

- Remove the six RVV ODS executable microkernel wrapper ops from the active dialect.
- Remove RVV C++ verifier code whose only purpose is to validate those wrapper ops or bind arithmetic semantics to enclosing wrapper op names.
- Keep non-executable RVV control/config/dataflow surfaces only when they do not claim direct artifact/exporter authority and do not depend on `tcrv_rvv.*_microkernel` wrapper names.
- Rewrite surviving RVV dataflow wording and verifier diagnostics so explicit dataflow ops are bounded RVV extension-family surfaces under `tcrv_rvv.with_vl`, not executable microkernel route authority.
- Delete or rewrite active RVV tests/fixtures that preserve the removed wrapper ops as parseable executable route, family, ABI, selected-boundary, or artifact authority.
- Update relevant Trellis specs when they still instruct agents/tests to preserve `tcrv_rvv.*_microkernel` wrapper authority.
- Keep failures truthful: if old wrapper deletion exposes missing new architecture, report that gap instead of restoring deleted logic.

## Non-Goals

- Do not implement new RVV lowering, common EmitC lowering, artifact emission, runtime ABI glue, intrinsic mapping, replacement family registry, compatibility wrapper, or legacy mode.
- Do not preserve old RVV microkernel ops for compatibility.
- Do not extend finite RVV families.
- Do not delete generic RVV capability/profile/vector-shape utilities, plugin identity, target support registration, or selected lowering-boundary metadata unless they are reachable only through deleted wrapper authority.
- Do not change scalar microkernel dialect surfaces in this RVV-only round.

## Acceptance Criteria

- [ ] Active ODS no longer defines `tcrv_rvv.i32_vadd_microkernel`, `tcrv_rvv.i32_vsub_microkernel`, `tcrv_rvv.i32_vmul_microkernel`, `tcrv_rvv.i64_vadd_microkernel`, `tcrv_rvv.i64_vsub_microkernel`, or `tcrv_rvv.i64_vmul_microkernel`.
- [ ] Active RVV C++ verifier code no longer contains `MicrokernelFamilySpec` structs, `I*MicrokernelOp::verify` entry points, or diagnostics requiring dataflow ops to be nested under RVV microkernel wrapper op names.
- [ ] Active RVV tests no longer parse, print, or depend on `tcrv_rvv.*_microkernel` wrapper ops.
- [ ] Remaining RVV dialect/control/dataflow tests prove only non-executable `setvl`, `with_vl`, policy/type, lowering-boundary metadata, and bounded dataflow surfaces.
- [ ] Focused ref-scans show intentional RVV leftovers only in deleted-route negative docs/tests or non-RVV scalar surfaces, not active RVV dialect wrapper authority.
- [ ] Focused build/lit/C++ tests for touched RVV dialect/plugin/variant/target surfaces run or any failures are reported as deletion gaps.
- [ ] `git diff --check` and `git diff --cached --check` pass before commit.

## Definition of Done

- Task status is updated truthfully.
- Relevant Trellis context files reference the PRD, specs, and touched code/test surfaces.
- One coherent commit records the deletion/refactor-only round when checks are complete enough to support it.
- Final report includes deleted RVV microkernel owners, tests deleted or rewritten, retained RVV surfaces and rationale, focused ref-scan results, checks, task status, commit hash, and clean worktree status.

## Technical Notes

- Required specs read first:
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- Primary implementation files:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
- Focused scan terms:
  - `tcrv_rvv.i32_vadd_microkernel`
  - `tcrv_rvv.i32_vsub_microkernel`
  - `tcrv_rvv.i32_vmul_microkernel`
  - `tcrv_rvv.i64_vadd_microkernel`
  - `tcrv_rvv.i64_vsub_microkernel`
  - `tcrv_rvv.i64_vmul_microkernel`
  - `MicrokernelOp`
  - `MicrokernelFamilySpec`
  - `executable microkernel`
  - `selected-path metadata`
  - `element_count`
