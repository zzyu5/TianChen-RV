# RVV family EmitC direct ssh evidence

## Goal

Produce fresh bounded hardware evidence for the direct typed RVV `i32-vmul`
family-op EmitC route. The evidence must prove that the selected finite
`i32-vmul-microkernel.v1` path reaches a generated RVV intrinsic C/C++ artifact
through verified `tcrv_rvv.i32_vmul_microkernel` body metadata, then compiles
and runs on the real `ssh rvv` host.

## Why Now

The archived parent task completed the family-op to EmitC route for i32 add.
The archived i32 binary continuation closed local i32 sub/mul FileCheck and
dry-run coverage at HEAD `15ff575`, but intentionally made no RVV runtime or
correctness claim. The remaining bottleneck is fresh hardware evidence for the
direct typed `i32-vmul` route, not another broad local matrix round.

## References

- Parent evidence only:
  `.trellis/tasks/archive/2026-05/05-11-rvv-family-ops-emitc-route-refactor/`
- i32 binary continuation evidence only:
  `.trellis/tasks/archive/2026-05/05-11-05-11-rvv-family-ops-emitc-route-refactor-i32-binary-complete/`
- Current task:
  `.trellis/tasks/05-11-rvv-family-emitc-direct-ssh-evidence/`

## Requirements

- Use the existing MLIR -> `tcrv-opt` -> `tcrv-translate` generated artifact
  path. Do not bypass the compiler with handwritten RVV C.
- Keep the module bounded to the direct `i32-vmul` RVV microkernel route.
- The evidence runner or focused lit coverage must fail closed unless the
  generated artifacts are tied to:
  - selected family/lowering descriptor `i32-vmul-microkernel.v1`;
  - executable typed body `tcrv_rvv.i32_vmul_microkernel`;
  - dataflow body
    `tcrv_rvv.i32_load -> tcrv_rvv.i32_load -> tcrv_rvv.i32_mul -> tcrv_rvv.i32_store`;
  - EmitC route metadata
    `tcrv_rvv.family_ops -> emitc.call_opaque -> RVV intrinsic C/C++`;
  - arithmetic call mapping
    `emitc.call_opaque[3]: __riscv_vmul_vv_i32m1 from tcrv_rvv.i32_mul`;
  - generated source using `riscv_vector.h` RVV C intrinsics, not descriptor-only
    arithmetic or scalar fallback computation.
- The sanitized evidence JSON and command summary must record generated
  artifact paths, hashes, command outcomes, and `ssh rvv` compile/run status
  without secrets, raw credentials, or performance claims.

## Acceptance Criteria

- [ ] `scripts/rvv_microkernel_e2e.py` validates direct `i32-vmul` source
      provenance through selected descriptor/family metadata, typed body
      metadata, EmitC route metadata, and arithmetic `emitc.call_opaque`
      mapping before accepting dry-run or ssh evidence.
- [ ] Focused script lit coverage asserts the direct `i32-vmul` evidence JSON
      records the descriptor/family, typed body, route, source ops, call mapping,
      generated direct helper artifacts, and no performance/runtime overclaim.
- [ ] Focused direct microkernel lit coverage for
      `test/Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir` still passes.
- [ ] One fresh non-dry-run direct `i32-vmul` run through
      `scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vmul` succeeds on
      `ssh rvv`.
- [ ] `git diff --check`, Trellis validation, focused build/test targets, and
      `check-tianchenrv` pass when the build tree is usable.

## Non-Goals

- No broad i32 add/sub, i32m2, i64, dispatch, scalar fallback, performance, or
  matrix coverage expansion.
- No new lowering architecture, LLVM scalable-vector path, inline asm path,
  GCC-default path, or descriptor-to-C computation path.
- No compiler core, dialect, pass, plugin registry, capability model, lowering,
  or emission implementation in Python. Python changes are runner/evidence
  validation only.
- No computation semantics in `tcrv.exec`.
- No performance claim. A successful run may claim only bounded correctness for
  this direct generated RVV `i32-vmul` helper artifact handoff and external
  caller route on `ssh rvv`.

## Minimal Validation Plan

- Build focused targets needed by the runner:
  `TianChenRVRVVTarget`, `tcrv-opt`, `tcrv-translate`, and touched focused tests
  if available in the current build tree.
- Run focused lit for
  `Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir`.
- Run focused lit for `Scripts/rvv-microkernel-e2e.test`.
- Run `python3 scripts/rvv_microkernel_e2e.py --dry-run
  --arithmetic-family=i32-vmul --run-id codex-i32-vmul-emitc-direct-dryrun
  --overwrite`.
- Run one fresh non-dry-run `ssh rvv` command for direct `i32-vmul`.
- Run `git diff --check`.
- Run Trellis validation on this task and reference archive paths as applicable.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if focused checks pass and the build tree is usable.

## Completion Boundary

Finish and archive only if the runner/test evidence is tied to the typed
family-op EmitC route and fresh `ssh rvv` compile/run evidence passes for this
exact direct `i32-vmul` route. If remote access or toolchain availability blocks
the run, keep the task open with sanitized failure logs and an exact continuation
point.
