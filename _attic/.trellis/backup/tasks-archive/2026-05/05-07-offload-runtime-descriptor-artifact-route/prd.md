# Offload Runtime Descriptor Artifact Route

## Goal

Add a bounded, target-owned offload runtime handoff descriptor artifact route so the generic target artifact export path supports non-source artifacts without treating offload as a custom RISC-V ISA extension.

## Requirements

* Extend the C++ target artifact export surface only as needed to route supported non-source artifacts by artifact kind or explicit route/mode.
* Preserve existing `--tcrv-export-target-source-artifact` behavior and tests for RVV and scalar source exports.
* Add a target-owned offload artifact exporter under the existing target/exporter layout.
* Export a deterministic, FileCheck-friendly descriptor containing sanitized compiler-visible metadata: source kernel, selected variant, origin plugin, required capabilities, runtime ABI kind/name, emission kind, artifact kind, lowering boundary op name, and handoff reason.
* Register the offload descriptor exporter through `TargetArtifactExporterRegistry` using plugin-local route id / artifact kind metadata from the offload emission plan.
* Keep generic target artifact routing semantic-neutral: it may validate route id, artifact kind, origin, emission kind, selected path, lowering boundary, and emission-plan consistency, but must not hard-code RVV, scalar, IME, AME, Sophgo, or vendor execution semantics.
* Add the smallest offload-plugin-local emission metadata needed if the current plan lacks stable artifact kind or route id.
* Keep `tcrv_offload` as runtime-offload metadata only.
* Do not implement offload execution, runtime library calls, RPC, DMA, driver integration, object generation, linking, performance benchmarking, or real hardware correctness claims.
* Avoid Python compiler internals; Python may only remain helper/probe orchestration.

## Acceptance Criteria

* [ ] Positive lit/FileCheck coverage exports an offload descriptor from post-planning MLIR with selected offload path, matching `tcrv_offload.lowering_boundary`, runtime ABI metadata, and supported offload emission plan.
* [ ] Descriptor output includes source kernel, selected variant, origin plugin, required capability refs, runtime ABI kind/name, artifact kind, and handoff reason.
* [ ] Existing RVV and scalar source artifact routes still pass and remain source-only where applicable.
* [ ] Negative coverage fails closed for selected offload path without matching lowering boundary.
* [ ] Negative coverage fails closed for missing runtime ABI metadata.
* [ ] Negative coverage fails closed for stale selected variant or stale lowering boundary.
* [ ] Negative coverage fails closed for unknown route id.
* [ ] Negative coverage fails closed for unsupported artifact kind.
* [ ] Negative coverage prevents RVV/scalar routes from spoofing offload descriptor route and offload routes from spoofing RVV/scalar source exporters.
* [ ] Unsafe strings, secret-like strings, URLs, or raw credentials in metadata are rejected or redacted according to project conventions.
* [ ] Focused C++ registry/API tests are added or updated if the registry/API changes.
* [ ] `git diff --check` passes.
* [ ] CMake configure with LLVM/MLIR 20 passes.
* [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes.

## Definition of Done

* Active C++ target/export or plugin-local emission code is changed.
* Lit/C++ tests cover the new route and failure modes.
* Relevant specs/README/task state are updated only as needed.
* Trellis task is validated and archived before final commit if task tooling supports it.
* Generated descriptors, build outputs, artifacts/tmp evidence, logs, credentials, and stale prompt/design packs are not committed.
* Repository ends clean with one coherent commit.

## Out of Scope

* Offload hardware execution or correctness/performance evidence.
* Vendor runtime integration, RPC, DMA, driver calls, object generation, or linking model.
* Treating Sophgo/offload as a custom RISC-V ISA extension.
* Adding generic compute ops to `tcrv.exec`.
* Broad architecture rewrites or copied microkernel exporters.

## Technical Notes

* Required inspection files and commands are taken from the user handoff in this session.
* Primary stack remains C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Relevant spec layers: core dialect, plugin protocol, lowering runtime, extension plugins, testing, target artifact export behavior.
