# Add RVV VL Scope Region Op

## Goal

Add a plugin-local RVV VL scope region operation, `tcrv_rvv.with_vl`, as the bounded structural companion to `tcrv_rvv.setvl`. The op consumes the `!tcrv_rvv.vl` token produced by `setvl` and creates a single-block region for future RVV control/body work without adding arithmetic, memory, lowering, runtime ABI, export, correctness, or performance semantics.

## Requirements

- Add `tcrv_rvv.with_vl` in RVV ODS as plugin-local control-plane IR.
- Consume exactly one `!tcrv_rvv.vl` value and contain a single-block region.
- Use a region argument only if it is `!tcrv_rvv.vl` and verifier-tied to the consumed VL value.
- Preserve bounded first-slice configuration compatibility with `setvl`: SEW 32, LMUL `m1`, and `#tcrv_rvv.policy`, with no VLEN/vlenb, element_count, required_march, required_capabilities, pointer, memory, arithmetic, mask, reduction, vector-register, lowering, runtime ABI, export, correctness, or performance semantics.
- Update RVV dialect verifier logic to reject layering mistakes and validate visible `setvl` compatibility when attrs are present.
- Update the minimum relevant specs describing `with_vl` as input/control-plane IR only.
- Add focused lit/FileCheck coverage for valid round-trip and minimal negative verifier cases.

## Out Of Scope

- Python compiler internals.
- `tcrv.exec` compute ops or core-dialect extension-specific semantics.
- RVV arithmetic, load/store, reduction, masks, vector registers, LLVM/RISC-V lowering, runtime ABI, target artifact export, probes, hardware runs, correctness claims, or performance claims.
- Changes to RVV microkernel C export, scalar export, host dispatch export, smoke probes, artifact routes, or runtime evidence scripts except for direct compile-break compatibility, which is not expected.

## Acceptance Criteria

- `git diff --check` passes.
- CMake configure against LLVM/MLIR 20 passes.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.
- A coherent commit is created for ODS/C++ verifier/tests/spec updates.
- Worktree is clean after the commit and task archival.
