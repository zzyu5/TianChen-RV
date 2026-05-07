# Scalar Fallback Explicit Microkernel Source Export

## Goal

Add the first bounded scalar fallback executable source-export slice through the
existing generic target artifact export route. The slice must prove that the
generic exporter registry is not RVV-shaped while keeping scalar computation
inside the `tcrv_scalar` extension dialect, scalar plugin, and scalar
target-owned exporter boundaries.

## Requirements

* Add exactly one explicit scalar fallback microkernel op if no equivalent live
  scalar op exists: `tcrv_scalar.i32_vadd_microkernel`.
* Keep the op bounded to a deterministic i32 vector-add standalone C self-check
  contract. Do not make it generic tensor, tile, loop, benchmark,
  bufferization, arbitrary scalar compute IR, runtime ABI integration, object
  generation, or linking support.
* Keep `tcrv.exec` compute-free. Concrete scalar behavior belongs only in
  `tcrv_scalar`, scalar plugin logic, and scalar target-owned export code.
* Add verifier checks for selected-path coherence, source kernel, selected
  variant, origin, role, required capabilities, available scalar fallback
  capability, element count, stale references, bounded string metadata, and
  secret-like or unsafe string rejection.
* Extend the scalar plugin locally so only an exact selected scalar fallback
  path with a matching `tcrv_scalar.lowering_boundary` and matching
  `tcrv_scalar.i32_vadd_microkernel` returns supported source-export readiness
  and emission-plan metadata.
* Preserve metadata-only behavior for generic scalar fallback paths without an
  explicit microkernel.
* Add a scalar target-owned exporter under the target layout to emit portable,
  deterministic C with no RVV headers, intrinsics, route ids, or claims.
* Register the scalar exporter in `TargetArtifactExporterRegistry` through the
  public generic `tcrv-translate --tcrv-export-target-source-artifact` route
  without adding scalar/RVV semantics to core generic validation.
* Preserve the existing RVV direct exporter and generic route. Do not weaken
  RVV fail-closed checks.

## Acceptance Criteria

* [ ] ODS/C++ verifier tests cover `tcrv_scalar.i32_vadd_microkernel` positive
  and negative cases.
* [ ] Scalar plugin tests prove the exact explicit scalar microkernel path is
  supported and generic scalar metadata-only fallback remains metadata-only.
* [ ] Target artifact export tests prove scalar post-planning MLIR exports
  deterministic portable C through `--tcrv-export-target-source-artifact`.
* [ ] Generated scalar C contains the expected scalar i32 add self-check marker
  and contains no RVV headers, intrinsics, RVV route ids, or RVV claims.
* [ ] Negative target export coverage includes scalar/RVV route spoofing,
  missing scalar boundary, missing scalar microkernel, stale selected variant,
  duplicate supported artifacts, unknown route id, unsupported artifact kind,
  and offload-only cases.
* [ ] Existing RVV route tests still pass.
* [ ] If local clang is available, generated scalar C compiles and runs under
  `artifacts/tmp` as local scalar self-check evidence only.
* [ ] Required project checks pass or are reported with exact blockers:
  `git diff --check`, CMake configure, and `check-tianchenrv`.

## Definition of Done

* Active C++/MLIR/TableGen/CMake/lit implementation is committed as one
  coherent commit.
* Trellis task state is archived and validated before final report.
* Generated sources, build outputs, local logs, and `artifacts/tmp` evidence
  remain uncommitted.
* Final report states no new RVV runtime/correctness/performance claim unless
  new `ssh rvv` evidence was intentionally collected.

## Technical Approach

Follow the existing RVV explicit microkernel shape, but keep scalar-specific
logic inside scalar-owned modules:

* Extend `ScalarOps.td` and `ScalarDialect.cpp` with the bounded scalar
  microkernel op and verifier.
* Add scalar plugin predicates that detect a matching selected
  `tcrv_scalar.i32_vadd_microkernel`; only then return a supported standalone C
  source export route.
* Add `include/TianChenRV/Target/Scalar` and `lib/Target/Scalar` exporter code,
  registering a scalar route with the generic registry.
* Update `tcrv-translate` to register both RVV and scalar target exporters.
* Add lit/C++ tests for dialect, plugin, and target artifact behavior.

## Out of Scope

* Generic scalar lowering, LLVM IR emission, object generation, linking,
  runtime ABI glue, bufferization, benchmarks, performance evidence, and
  arbitrary scalar source export.
* New Python compiler internals or scalar e2e helper scripts.
* New RVV runtime/correctness/performance claims or `ssh rvv` runs unless RVV
  generated source semantics are intentionally changed.
* Broad workspace hygiene work unrelated to this scalar source-export slice.

## Technical Notes

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial HEAD inspected: `9fef679 feat: add generic target source export route`.
* Initial worktree before task creation was clean and `predoc/` was absent.
* User requested a single serial full-access worker and explicitly forbade
  subagents, spawned agents, parallel agents, and multi-agent workflows.
* Required specs and implementation files were read before implementation.
