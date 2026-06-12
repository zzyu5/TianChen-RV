# implement next compiler milestone from supervisor audit

## Goal

Implement one coherent compiler milestone after the latest supervisor run by strengthening the capability model as a real C++/MLIR compiler decision object. This round adds a bounded capability relation slice so plugin availability, variant materialization, and legality can be satisfied by structured capability provider relations rather than only exact capability id strings.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial required inspection showed `git status --short` clean before this task was created and HEAD at `428cec7 feat: add RVV scalar dispatch object export`.
* `.trellis/.current-task` did not exist before this task.
* `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0028-20260507T190425Z` exists but has no `repo_audit.md` or `review_input.md`; it appears to be an override prompt run directory, not a complete review artifact.
* The latest complete `repo_audit.md` / `review_input.md` pair is `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0027-20260507T182432Z`.
* r0027 added bounded RVV+scalar dispatch self-check object export and left the repo clean.
* Current code already has `tcrv.exec`, RVV/scalar/offload plugin slices, variant planning, selected lowering boundaries, emission plans, source/object artifact exports, and bounded RVV+scalar dispatch evidence.
* The capability spec explicitly names `provide`, `imply`, `conflict`, and dispatch condition relations, but the current `TargetCapabilitySet` primarily supports exact id/symbol/kind/property lookup.

## Assumptions

* This round should not launch Trellis subagents or spawned agents; all implementation and checks happen in this main Codex worker.
* The next owner should not be another standalone smoke/probe/export wrapper unless it is the only blocker. Capability relation support is a better compiler-core milestone because it affects plugin proposal, materialization, and legality.
* A bounded first slice can implement provider/implied capability id satisfaction without trying to complete all future conflict and dispatch-condition semantics.

## Requirements

* Add C++ `CapabilityDescriptor` / `TargetCapabilitySet` support for structured relation attributes carried by `tcrv.exec.capability`.
* Support at least `provides` and `implies` as capability-id relation lists; preserve explicit direct id behavior so an exact capability id remains authoritative when present.
* Expose relation-aware provider lookup and availability query APIs without changing the primary implementation stack.
* Use the relation-aware capability query in RVV plugin proposal/property view and variant legality so a profile capability can satisfy the RVV required capability id.
* Use the relation-aware provider lookup in plugin variant materialization so a proposal requiring capability id `rvv` can materialize `requires = [@rvv_profile]` when `@rvv_profile` provides or implies `rvv`.
* Keep `tcrv.exec` compute-free; relation attributes are capability metadata, not computation.
* Do not add Python compiler internals.

## Acceptance Criteria

* A focused C++ capability model test proves relation lists are parsed and relation-aware provider lookup works.
* A lit/FileCheck planning test proves RVV variant proposal/materialization/selection works when a profile capability provides `rvv` and there is no exact `id = "rvv"` capability.
* Malformed relation metadata is rejected by the `tcrv.exec.capability` verifier.
* Relevant specs document the bounded relation slice.
* `git diff --check`, targeted build/tests, and the project check target pass or exact blockers are documented.

## Definition of Done

* Tests added or updated for the changed compiler behavior.
* CMake build and lit/FileCheck checks pass where local toolchain support exists.
* Specs updated for durable relation semantics.
* One coherent commit is created.
* `.trellis/.current-task` is not left pointing at a completed top-level task.

## Out of Scope

* Generic RVV lowering to LLVM/RISC-V intrinsics.
* New object or smoke evidence routes.
* Broad negative fixture matrices.
* Full conflict resolution, full dispatch-condition semantics, or profile inheritance lattices.
* Any RVV runtime/correctness/performance claim beyond existing recorded evidence.

## Technical Notes

* Relevant specs: `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
* Relevant code: `include/TianChenRV/Support/CapabilityModel.h`, `lib/Support/CapabilityModel.cpp`, `lib/Dialect/Exec/IR/ExecOps.cpp`, `lib/Transforms/VariantMaterialization.cpp`, `lib/Plugin/RVV/RVVExtensionPlugin.cpp`.
* Relation syntax for this slice should use structured MLIR attributes such as `provides = ["rvv"]` and `implies = ["zvl128b"]`.
