# Scalar Fallback Auto Microkernel Materialization For Dispatch Export

## Goal

Enable the existing bounded RVV+scalar i32-vadd host dispatch C export to run
from real post-planning MLIR without a hand-authored scalar microkernel fixture.
When the selected scalar fallback path is the current finite first slice, the
scalar plugin should materialize the matching plugin-local
`tcrv_scalar.i32_vadd_microkernel` attachment during selected
lowering-boundary materialization, so the existing emission-plan and dispatch
export routes can validate and consume it.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree was clean at `9122be0 feat: add RVV with_vl scope op`.
- Latest supervisor audit says the last round added only `tcrv_rvv.with_vl`
  control-plane structure and made no RVV runtime/hardware claim.
- Existing RVV pipeline already auto-materializes
  `tcrv_rvv.i32_vadd_microkernel` from a finite RVV descriptor.
- Existing scalar target export supports an explicit
  `tcrv_scalar.i32_vadd_microkernel`, but the scalar plugin currently only
  materializes `tcrv_scalar.lowering_boundary` and otherwise leaves scalar
  fallback as metadata-only.
- Existing RVV+scalar dispatch C export validates one selected RVV dispatch case
  plus one selected scalar fallback callable route, but current tests rely on a
  hand-authored scalar microkernel.

## Requirements

- Keep implementation in C++ / MLIR / TableGen / CMake / lit/FileCheck.
- Keep `tcrv.exec` compute-free; scalar fallback executable source semantics
  must stay plugin-local / target-owned under `tcrv_scalar` and scalar target
  export code.
- Add bounded scalar plugin-owned descriptor metadata for exactly the current
  i32-vadd fallback slice, if needed, on scalar fallback proposals:
  `tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"` and bounded
  `tcrv_scalar.element_count`.
- During scalar selected lowering-boundary materialization, validate the finite
  descriptor and materialize exactly one matching direct
  `tcrv_scalar.i32_vadd_microkernel` op for the selected scalar path.
- Preserve selected-path metadata on the materialized scalar microkernel:
  `source_kernel`, `selected_variant`, `origin`, `role`, `element_count`, and
  `required_capabilities`.
- Fail before emission if descriptor metadata is malformed, element count is
  out of the bounded range, or a matching scalar microkernel already exists for
  the selected path.
- Update scalar fallback specs and testing contract only where the durable
  compiler behavior changes.

## Non-Goals

- No generic scalar lowering, arbitrary scalar op family, object generation,
  linking, benchmark harness, runtime performance path, or correctness claim.
- No new RVV smoke/probe/evidence artifact and no `ssh rvv` claim.
- No changes to `tcrv.exec` compute semantics.
- No generic target-family branching in core transforms or generic artifact
  routing.
- No broad negative fixture matrix beyond focused descriptor/materialization
  coverage.

## Acceptance Criteria

- A lit/FileCheck test proves `tcrv-opt --tcrv-execution-planning-pipeline`
  materializes the scalar fallback microkernel without a hand-authored
  `tcrv_scalar.i32_vadd_microkernel` input.
- A pipeline-to-export lit/FileCheck test proves the existing
  `--tcrv-export-rvv-scalar-i32-vadd-dispatch-c` route can consume the
  automatically materialized scalar fallback microkernel.
- One focused negative diagnostic proves malformed scalar finite descriptor
  metadata fails before supported emission/export.
- `git diff --check` passes.
- CMake configure against LLVM/MLIR 20 passes.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passes.
- One coherent commit is created, task is archived, and the final worktree is
  clean.

## Technical Notes

- Relevant specs:
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Likely implementation files:
  scalar plugin materialization/readiness code, scalar plugin spec text, and
  focused lit/FileCheck fixtures under `test/Transforms` or `test/Target`.
