# Generic Plugin-Owned Variant Metadata Preservation

## Goal

Implement a bounded C++/MLIR path that lets `VariantProposal` carry generic
plugin-owned MLIR attributes through proposal collection and materialization,
then exercise it with one typed, finite, non-compute RVV policy attribute.

## Requirements

- Add a generic `VariantProposal` attribute bag or named-attribute list.
- Validate proposal-carried attribute names generically before IR mutation when
  practical:
  - non-empty names only;
  - dialect-qualified/discardable names only;
  - no duplicates;
  - no collisions with core `tcrv.exec.variant` attributes such as `sym_name`,
    `origin`, `requires`, `condition`, `guard`, or `policy`;
  - non-null values only.
- Preserve valid plugin-owned attributes on generated `tcrv.exec.variant` ops
  in `VariantMaterialization`.
- Add a typed RVV dialect policy attribute under concrete namespace
  `tcrv_rvv`, without adding RVV compute ops or core `tcrv.exec` fields.
- Update `RVVExtensionPlugin` so available-RVV proposals carry the typed RVV
  policy attribute while keeping generic string decision metadata.
- Make RVV legality validation plugin-local: RVV-origin materialized variants
  must carry the expected typed RVV policy metadata.
- Keep selection/dispatch core code consuming only generic
  `condition`/`guard`/`policy` strings and capability metadata.
- Add C++ and lit/FileCheck coverage for positive and negative cases.

## Non-Goals

- No RVV arithmetic, memory, reduction, lowering, emission, runtime ABI,
  benchmark, correctness, or performance behavior.
- No RVV-specific fields or branches in `tcrv.exec` ODS or generic core
  selection/materialization semantics.
- No Python implementation of compiler internals.
- No `ssh rvv` evidence because this round makes compiler-local metadata and
  parser/printer claims only.

## Validation

- `git diff --check`
- `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- Focused RVV/materialization/selection binaries or lit filters as needed.
