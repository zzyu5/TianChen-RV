# codex worker review fallback compiler owner

## Goal

Advance the active TianChen-RV compiler path after the malformed Hermes review fallback by making module-level RVV target-profile capacity facts participate in RVV plugin compiler decisions. A kernel that binds `target = @profile` should be able to receive VLEN-derived RVV capacity metadata from that profile and carry it through proposal, cost metadata, selected lowering boundary, emission plan metadata, and the bounded i32-vadd microkernel descriptor.

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* The worktree was clean before this task was created.
* Latest committed HEAD before this task is `9d833e4 feat: consume module target profiles in planning`.
* Latest complete Hermes audit with `repo_audit.md` and `review_input.md` is `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0087-20260508T221245Z/`.
* The previous round made module-level `tcrv.exec.target` profiles visible to the active execution-planning pipeline for basic RVV capability, hart-count, and compile-run facts.
* Current RVV plugin logic consumes hart count and compile-run through `TargetCapabilitySet::lookupProviderByID`, but RVV vector capacity currently uses exact `lookupByID` for `rvv.vlenb_bytes` and `rvv.i32_m1_lane_count`.
* That means a single relation-provider module target profile can satisfy the basic RVV path but does not yet feed optional VLEN/i32-lane capacity into RVV proposal cost, element count, lowering-boundary metadata, or selected-plan metadata.

## Requirements

* RVV vector capacity lookup must accept exact capability providers and relation-provider capabilities from the active kernel capability-provider scope.
* The capacity pair remains all-or-nothing: if available `rvv.vlenb_bytes` and available `rvv.i32_m1_lane_count` are not both present, proposal/legality must fail or decline with the existing bounded diagnostic style.
* Parsed capacity facts must still validate `lanes = bytes / 4` for the current i32 M1 slice.
* Module-level target profiles remain explicit per-kernel inputs only through `kernel target = @profile`; unrelated module-level targets must not affect decisions.
* The change must preserve parameter layering: VLEN/i32-lane facts are hardware capability/profile facts, while `element_count` remains descriptor-local metadata and not runtime `n`, AVL, VL, or shape.
* Core orchestration must remain free of RVV-specific branches; RVV-specific interpretation stays in the RVV plugin.

## Acceptance Criteria

* A module-level `tcrv.exec.target` profile that provides `rvv`, `rvv.hart_count`, `rvv.probe.compile_run`, `rvv.vlenb_bytes`, and `rvv.i32_m1_lane_count` drives `tcrv_rvv.vlenb_bytes`, `tcrv_rvv.i32_m1_lanes`, and capacity-derived `tcrv_rvv.element_count` through `--tcrv-execution-planning-pipeline`.
* The resulting lowering boundary and RVV microkernel preserve the capacity-selected metadata.
* The supported emission-plan diagnostic serializes selected-plan metadata for those capacity facts without claiming runtime/performance evidence.
* Regression coverage shows that a relation-provider target profile can feed RVV cost/proposal metadata.
* Build/test checks pass through the project lit/FileCheck route.

## Out Of Scope

* No generic tensor/tile IR.
* No new high-level lowering.
* No new RVV arithmetic kernel beyond the bounded i32-vadd first slice.
* No new runtime probing or performance claim.
* No `ssh rvv` correctness/performance claim unless explicitly run and reported as such.

## Technical Notes

* Relevant specs: `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
* Relevant code paths inspected: `lib/Support/CapabilityModel.cpp`, `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, `lib/Transforms/VariantSelection.cpp`, `lib/Target/TargetArtifactExport.cpp`, `tools/tcrv-translate/tcrv-translate.cpp`.
* This PRD treats the user prompt as confirmation to choose and execute the highest-value coherent compiler owner without further Q&A.
