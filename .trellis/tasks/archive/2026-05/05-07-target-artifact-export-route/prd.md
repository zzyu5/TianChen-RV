# target artifact export route for RVV microkernel

## Goal

Add a bounded C++ target artifact export routing surface so a generic `tcrv-translate` entry point can consume supported post-planning emission metadata and route to a target-owned exporter. The first concrete route is limited to the existing explicit RVV i32 vector-add microkernel standalone C source exporter.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD is expected and observed as `402a9c3 feat: add RVV microkernel e2e evidence helper`.
* `predoc/` is absent.
* Existing direct exporter is `tcrv-translate --tcrv-export-rvv-microkernel-c`, implemented in C++ under `include/TianChenRV/Target/RVV/` and `lib/Target/RVV/`.
* Existing emission manifest includes plugin-owned supported fields for the explicit RVV microkernel route.
* Python may only orchestrate helper/evidence behavior; compiler routing and export decisions must stay in C++/MLIR.

## Requirements

* Add the smallest generic C++ target artifact exporter abstraction under `include/TianChenRV/Target/` and `lib/Target/`.
* Register the existing RVV microkernel C exporter through target-owned RVV registration code.
* Add a generic `tcrv-translate` export entry point that consumes post-planning MLIR metadata and emits the selected supported source artifact.
* Preserve the existing RVV-specific translate flag for direct low-level testing.
* Fail closed for missing route metadata, unsupported artifact kind, unknown route id, ambiguous supported artifacts, stale selected path, missing lowering boundary, missing microkernel, and scalar/offload/non-RVV paths.
* Reuse the existing RVV C generation implementation; do not duplicate generated C logic.
* Update `scripts/rvv_microkernel_e2e.py` only to optionally use the generic route, keeping Python as orchestration.

## Acceptance Criteria

* [ ] Generic route exports deterministic C for the checked-in explicit RVV microkernel fixture.
* [ ] Direct RVV exporter and generic route agree on deterministic source for that fixture.
* [ ] Lit/FileCheck covers positive source content and equality behavior.
* [ ] Lit/FileCheck covers fail-closed negative cases for missing/unsupported/ambiguous/stale/unknown route conditions where representable.
* [ ] Existing emission manifest/RVV microkernel/helper tests continue to pass.
* [ ] `git diff --check` passes.
* [ ] CMake configure and `check-tianchenrv` pass, or any local toolchain blocker is reported exactly.
* [ ] If updated, the RVV e2e helper dry-run path passes.
* [ ] If `ssh rvv` is reachable and generic route is used, real bounded compile/run evidence is recorded under `artifacts/tmp` without broad claims.

## Out Of Scope

* Generic RVV lowering.
* Arbitrary RVV source export.
* Runtime ABI integration.
* Object generation, linking, bufferization, benchmark, or performance claims.
* Core branches that hard-code RVV/IME/offload/scalar semantics.
* Moving tracked tests into `artifacts/` or committing generated evidence.

## Technical Notes

* Relevant specs: plugin protocol interfaces/locality, lowering runtime emission contract, RVV plugin boundary, MLIR testing contract, validation evidence contract, implementation stack contract, and shared capability/plugin/compute guides.
* Generic exporter registry should be keyed by a stable route id already emitted by plugin-owned emission metadata when possible. Current route metadata can use the existing `lowering_pipeline = "tcrv-export-rvv-microkernel-c"` and `artifact_kind = "standalone-c-source"` fields to avoid widening manifest format unless code inspection proves a dedicated field is necessary.
* Generic tool registration may construct built-in target exporters, but shared route resolution must stay target-neutral.
