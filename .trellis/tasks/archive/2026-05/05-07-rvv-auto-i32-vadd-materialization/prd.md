# RVV Plugin Auto I32 VAdd Materialization

## Goal

Implement one bounded RVV compiler slice: selected RVV paths with a finite
plugin-owned i32 vector-add microkernel descriptor should automatically
materialize a direct `tcrv_rvv.i32_vadd_microkernel` attachment through the RVV
plugin lowering-boundary path.

## Scope

- Keep the implementation in C++/MLIR/TableGen/CMake/lit.
- Keep RVV computation representation plugin-local under `tcrv_rvv` and the RVV
  plugin; do not add compute semantics to `tcrv.exec` or generic transforms.
- Add minimal RVV plugin-owned variant metadata for exactly this finite
  descriptor if existing selected variants do not already carry enough metadata.
- Preserve selected-path/export metadata on the materialized op:
  `source_kernel`, `selected_variant`, `origin`, `role`, `element_count`,
  `required_capabilities`, `required_march`, and `selected_mabi` when available.
- Reuse the existing `tcrv_rvv.i32_vadd_microkernel` op, RVV microkernel
  exporter, generic emission-plan path, and coherence-gated target artifact
  export.

## Non-Goals

- No generic RVV lowering, arbitrary vector lowering, runtime ABI integration,
  object generation, hardware correctness, or performance claim.
- No Python compiler internals.
- No new exporter route, smoke probe, remote probe, evidence helper, dashboard,
  broad matrix, scalar/offload/IME/Sophgo work, or generic target-family branch.
- Do not stage or rewrite pre-existing dirty supervisor-loop files.

## Required Evidence

- A lit/FileCheck positive test where an RVV-capable kernel fixture runs through
  the relevant pipeline and automatically materializes
  `tcrv_rvv.i32_vadd_microkernel` without a hand-authored microkernel input.
- A focused pipeline/export check proving the automatically materialized
  microkernel reaches the existing RVV source export route through the generic
  coherence-gated artifact path.
- One focused negative diagnostic for malformed finite RVV i32-vadd descriptor
  metadata.
- Run `git diff --check`, configure with LLVM/MLIR 20, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
