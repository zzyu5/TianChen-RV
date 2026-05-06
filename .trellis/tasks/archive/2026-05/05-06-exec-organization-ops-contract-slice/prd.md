# exec organization ops contract slice

## Goal

Complete the next bounded `tcrv.exec` ODS/C++ contract slice by adding core execution-organization ops for `region`, `hart_parallel`, `mem_window`, and `diagnostic`, with structural verifiers and lit/FileCheck coverage.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current `HEAD` is `adb57f8 feat: add exec capability verifier slice`.
- Existing `tcrv.exec` ODS covers target, capability, kernel, variant, dispatch, and fallback.
- Existing C++ verifiers check capability `id`/`kind`, variant `origin`/`requires`, capability references, and fallback target references.
- Stable specs name `region`, `hart_parallel`, `mem_window`, and diagnostics as core execution-organization surfaces.

## Requirements

- Add ODS definitions for `tcrv.exec.region`, `tcrv.exec.hart_parallel`, `tcrv.exec.mem_window`, and `tcrv.exec.diagnostic`.
- Keep op semantics structural and compute-free.
- Add C++ verifiers for non-empty required string attributes, positive explicit numeric fields, applicable symbol resolution, and required nesting.
- Avoid extension-specific target checks or hard-coded RVV/IME/Sophgo legality.
- Preserve the textual `tcrv.exec.*` operation family with concrete MLIR namespace `tcrv`.
- Extend lit/FileCheck positive and negative tests.

## Acceptance Criteria

- [ ] `tcrv.exec.region`, `tcrv.exec.hart_parallel`, `tcrv.exec.mem_window`, and `tcrv.exec.diagnostic` parse/print in lit tests.
- [ ] Negative tests cover malformed required attributes and non-positive hart counts.
- [ ] Verifiers remain structural and plugin-locality preserving.
- [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes after configure.
- [ ] `git diff --check` passes.
- [ ] Trellis task state is validated and archived before final report.

## Out Of Scope

- No plugin registry implementation.
- No RVV dialect, IME dialect, offload dialect, lowering, emission, runtime glue, or performance probe.
- No Python compiler internals.
- No generic tensor, tile, vector arithmetic, or scalar algorithm ops inside `tcrv.exec`.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`, `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant implementation files: `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`, `include/TianChenRV/Dialect/Exec/IR/ExecOps.h`, `lib/Dialect/Exec/IR/ExecOps.cpp`, `test/Dialect/Exec/basic.mlir`, `test/Dialect/Exec/verify.mlir`.
