# structured dispatch fallback contract

## Goal

Make `tcrv.exec` dispatch/fallback structure compiler-visible and testable before adding variant selection, cost modeling, lowering, emission, or runtime behavior.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- HEAD is `f7894a0 feat: add capability query legality pass`.
- The initial inspection showed no active Trellis task and only archived prior tasks.
- Existing `tcrv.exec.dispatch` is a single-block region container.
- Existing `tcrv.exec.fallback` references a variant but does not reject arbitrary variant-body nesting.
- The core implementation must remain C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
- Python is allowed only for workflow/support scripts, not compiler IR or pass logic.

## Requirements

- Add a structured `DispatchCaseOp` with textual form `tcrv.exec.case`.
- Each dispatch case must reference a `tcrv.exec.variant` in the enclosing `tcrv.exec.kernel` via `FlatSymbolRefAttr`.
- Optional `condition`, `guard`, and `policy` metadata must be generic non-empty strings when present.
- Dispatch cases must be nested directly in `tcrv.exec.dispatch`.
- Dispatch must be nested directly in a `tcrv.exec.kernel`, not in a variant body.
- Dispatch must contain at least one dispatch case and exactly one fallback.
- Dispatch should reject duplicate case targets when simple and deterministic.
- Fallback must reference an existing variant and must not be nested in a `tcrv.exec.variant` body.
- Do not interpret condition strings as RVV, IME, Sophgo, AME, or offload-specific logic.

## Acceptance Criteria

- [ ] Valid lit coverage includes at least two variants, one dispatch case, and one fallback.
- [ ] Negative lit coverage rejects unknown case target.
- [ ] Negative lit coverage rejects case outside dispatch.
- [ ] Negative lit coverage rejects missing or duplicate fallback.
- [ ] Negative lit coverage rejects empty `condition`, `guard`, or `policy` metadata when present.
- [ ] Negative lit coverage rejects fallback nested inside a variant body.
- [ ] Existing exec, capability, plugin, and capability-pass tests still pass.
- [ ] `cmake` configure and `check-tianchenrv` pass locally.
- [ ] `git diff --check` passes.
- [ ] Trellis task validates and is archived before commit.

## Out Of Scope

- Variant selection implementation.
- Cost model or tuning implementation.
- Lowering, emission, runtime ABI, or plugin implementation.
- RVV/IME/Sophgo/AME target-family branches in core dispatch logic.
- Python implementation of compiler internals.

## Technical Notes

- Relevant implementation files: `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`, `lib/Dialect/Exec/IR/ExecOps.cpp`.
- Relevant tests: `test/Dialect/Exec/basic.mlir`, `test/Dialect/Exec/verify.mlir`.
- Stable specs to update only for durable names/semantics: `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
