# Capability Model Attributes And Exec Verifier Slice

## Goal

Implement the first compiler-visible capability model slice for `tcrv.exec` using MLIR/TableGen/C++ and lit/FileCheck tests.

## Requirements

- Add a minimal structured capability/requirement representation for variants and capabilities.
- Keep the representation extension-neutral: core logic must not branch on RVV, IME, Sophgo, AME, or other concrete extension names.
- Verify `tcrv.exec.variant` has a non-empty `origin`.
- Verify `tcrv.exec.variant` has structured `requires`.
- Verify every required capability symbol resolves to a `tcrv.exec.capability` in the enclosing `tcrv.exec.kernel`.
- Verify `tcrv.exec.fallback` references an existing `tcrv.exec.variant` in the enclosing kernel/dispatch scope.
- Verify `tcrv.exec.capability` has well-formed structured `id` and `kind` fields.
- Preserve `tcrv.exec.target` as target/capability anchoring only; do not add compute semantics.
- Preserve textual first-dot compatibility: operations remain `tcrv.exec.*`, registered dialect namespace may remain `tcrv`.

## Non-Goals

- Do not implement a broad plugin registry, lowering, emission, or RVV runtime path.
- Do not add compute ops such as matmul, softmax, reduce, generic tile, or tensor compute to `tcrv.exec`.
- Do not implement core IR, dialects, capability model, verifier, passes, plugin registry, lowering, or emission in Python.
- Do not claim RVV correctness, runtime, or performance evidence.

## Evidence

- Positive lit/FileCheck test for kernel, target, capability, variant `requires`, dispatch, and fallback.
- Negative lit/FileCheck tests for missing variant `requires`, unknown required capability, missing/empty origin, malformed/missing capability fields, and bad fallback reference.
- Run CMake configure, `check-tianchenrv`, `git diff --check`, and `python3 .trellis/scripts/task.py validate`.
