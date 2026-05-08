# Emission Manifest Target Artifact Handoff Bundle

## Goal

Make selected RVV microkernel and RVV+scalar dispatch executable paths expose a structured target-artifact handoff bundle in the emission manifest, so downstream build/runtime integrators can discover source, header, and object artifact routes from manifest data rather than route-specific `tcrv-translate` knowledge.

## Requirements

* Extend the C++ manifest/export path to report registry-derived target artifact records for supported selected paths.
* Keep generic manifest printing target-agnostic: it may serialize registry-provided route metadata but must not branch on RVV, scalar, IME, offload, vendor, dtype, shape, runtime, correctness, or performance semantics.
* Represent each available artifact record with stable fields for artifact kind, route/exporter id, owner, generic front-door selectability, direct helper route availability, runtime ABI kind/name when applicable, and bounded evidence role.
* Cover the bounded direct RVV i32-vadd microkernel path with separate source, header, and relocatable-object records sharing the same runtime ABI boundary.
* Cover the RVV+scalar i32-vadd dispatch selected surface with source, header, and relocatable-object records connected to the selected dispatch case/fallback surface while preserving existing dispatch case/fallback/guard manifest information.
* Keep artifact bundle discovery derived from existing selected paths, emission plans, target artifact registry/exporter metadata, runtime ABI metadata, and capability-backed route checks.
* Do not implement compiler manifest logic, artifact selection, runtime ABI modeling, or route discovery in Python.
* Update durable specs only where needed for the manifest handoff contract.

## Acceptance Criteria

* [ ] RVV microkernel manifest output includes source/header/object artifact bundle records with stable route identifiers/selectors and matching runtime ABI kind/name.
* [ ] RVV+scalar dispatch manifest output includes source/header/object dispatch artifact bundle records and retains selected surface, dispatch case, dispatch fallback, runtime ABI parameters, required capabilities, and preference fields.
* [ ] Unsupported or metadata-only paths do not fabricate target artifact records.
* [ ] Target artifact registry tests cover route metadata separation for source/header/object and the manifest-facing discovery helper.
* [ ] `git diff --check` passes.
* [ ] `cmake --build build --target tcrv-opt tcrv-translate -j2` passes.
* [ ] `cmake --build build --target check-tianchenrv -j2` passes.

## Definition of Done

* C++/MLIR/lit implementation is committed as one coherent compiler commit.
* Trellis task is validated and archived before final report.
* Worktree is clean after commit.
* No new RVV runtime/correctness/performance claim is made without `ssh rvv` evidence.

## Technical Approach

Add a target/export registry helper that collects manifest-safe target artifact bundle records from supported `TargetArtifactCandidate` data and registered single/composite exporter metadata. The emission manifest will consume that helper and print deterministic records. Route owner and direct helper availability stay target/export registry metadata; evidence role and generic front-door selector are derived from artifact kind generically.

## Out of Scope

* New RVV lowering, arbitrary kernel emission, IME/AME/Sophgo/offload implementation, runtime probing, linking, benchmarks, or performance claims.
* Python compiler-side manifest or route-selection logic.
* Hidden `main`, self-check harness, or executable evidence behavior in default source/header/object routes.

## Technical Notes

* Relevant contracts: `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/capability-model/capability-contract.md`.
* Main implementation surfaces: `include/TianChenRV/Target/TargetArtifactExport.h`, `lib/Target/TargetArtifactExport.cpp`, `include/TianChenRV/Target/EmissionManifest.h`, `lib/Target/EmissionManifest.cpp`, built-in target exporter registrations, and `tools/tcrv-translate/tcrv-translate.cpp`.
