# Hermes: frontend linalg i32-vadd to TianChen-RV exec boundary

## Goal

Add one bounded, real compiler-owner slice that starts from hand-written/test
high-level MLIR and materializes the TianChen-RV execution boundary consumed by
the existing plugin planning pipeline.

The selected owner is frontend lowering, not scalar ABI resurrection, helper
packaging, smoke-only evidence, or report-only work.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Clean committed HEAD at takeover was `e44a972 chore: codify extension plugin integration steering`.
* The stale Trellis pointer to `.trellis/tasks/05-09-scalar-microkernel-external-abi-e2e` was cleared.
* Existing code already supports `tcrv.exec.kernel` planning, RVV/scalar proposal, selected lowering boundaries, emission-plan diagnostics, target artifact routes, headers, objects, bundles, and RVV+scalar dispatch for the bounded i32-vadd microkernel.
* There is no current `Conversion/` or high-level MLIR frontend lowering slice.
* Current plugin integration contract explicitly allows backend/plugin work from TianChen-RV MLIR and also allows a frontend owner to start from hand-written/test `linalg` inputs.

## Requirements

* Add a narrow C++/MLIR pass that consumes an explicitly marked, semantically checked `linalg.generic` i32 vector-add wrapper and creates one `tcrv.exec.kernel`.
* The produced kernel must carry the source-selected module target profile reference and the IR-backed callable ABI boundary:
  * `tcrv.exec.mem_window` for `lhs`, `rhs`, and `out`;
  * `tcrv.exec.runtime_param` for runtime element count `n`.
* The pass must not add generic `tcrv` compute ops and must not encode RVV/scalar extension semantics in core orchestration.
* The pass output must be usable by the existing `--tcrv-execution-planning-pipeline`.
* Register only the MLIR high-level dialects needed for this bounded frontend input in the public tool.
* Add focused lit/FileCheck coverage proving the high-level input lowers into exec/capability/ABI surfaces and then drives the existing RVV/scalar planning path.

## Acceptance Criteria

* `tcrv-opt` accepts the bounded linalg test input without `allow-unregistered-dialect`.
* `--tcrv-lower-linalg-i32-vadd-to-exec` emits a parseable `tcrv.exec.kernel` with `target = @...`, three ABI mem windows, and runtime `n`.
* Piping the lowering pass into `--tcrv-execution-planning-pipeline` materializes the existing RVV/scalar proposal path plus the selected execution surface, selected lowering boundary, and supported bounded emission plan that follow from the supplied capability profile.
* Negative coverage rejects an unsupported marked linalg body before creating a kernel.
* `cmake --build build --target check-tianchenrv` passes, or any local toolchain blocker is reported exactly.

## Out Of Scope

* No new scalar external ABI round.
* No generic linalg lowering.
* No generic tensor/tile/core compute dialect.
* No new RVV runtime/correctness/performance claim without `ssh rvv`.
* No new artifact helper whose only purpose is smoke/evidence packaging.

## Technical Notes

* The bounded source input is a wrapper `func.func` containing one marked `linalg.generic` with two i32 inputs, one i32 output, one `arith.addi`, and one `linalg.yield`.
* The target capability facts remain in `tcrv.exec.target` / capability providers. The lowering pass copies a selected target symbol reference onto the new `tcrv.exec.kernel`; it does not invent capabilities.
* The existing runtime ABI helper code can materialize and validate the exact i32-vadd ABI surfaces used by RVV/scalar exporters.
