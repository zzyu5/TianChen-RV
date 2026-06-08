# RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

## Goal

Advance the RVV production-kernel capability campaign from the accepted packed i4/u4 product-reduction-dequant primitive and same-target no-win timing evidence into compiler-side, RVV-plugin-local resource-aware selected-body realization and planning. The macro owner is the RVV plugin compiler path that represents and consumes packed-resource and schedule facts before route planning and EmitC artifact generation.

## Campaign Gates

- [x] Gate 1: Resource-aware selected-body planning boundary for packed low-precision contraction is represented in RVV plugin/source and verified with focused tests.
- [ ] Gate 2: Route, statement, and artifact validation consume those resource facts without Common EmitC inventing RVV semantics.
- [ ] Gate 3: Generated artifact evidence reflects the new production path.
- [ ] Gate 4: Same-target measurement is rerun only after a production compiler/resource-aware change and honestly reports win, regression, or no-win.

## Current Round Slice

This round implements only a bounded Gate 1 production compiler slice: establish or strengthen the RVV plugin-local resource-aware schedule/realization boundary for packed low-precision product-reduction-dequant so selected-body realization or route planning carries explicit packed-resource facts and fails closed when those facts are stale, missing, or not consumed.

If repository evidence shows this Gate 1 boundary already exists and is complete, this round must identify the next concrete production-source blocker inside the same macro owner and close that slice. It must not switch to a generated-bundle, measurement-only, report-only, or unrelated route-family task.

## Repository Findings For This Round

- The previous packed-i4 campaign is archived and complete through primitive/resource facts, statement planning, target artifact validation, generated-bundle evidence, executable correctness, and same-target timing.
- Current source already has a Gate 1 foundation: `RVVContractionSelectedBodyRealizationOwner.cpp` consumes pass-produced low-precision resource facts, copies realization attrs to producer/consumer `with_vl`, emits `vsetvl_region_marker` ops, and builds a Gearbox cross-region handoff; `RVVEmitCContractionRouteFamilyPlanOwners.cpp` re-consumes realization attrs and realized marker/handoff structure before route acceptance.
- The remaining Gate 1 production-source gap is narrower: several verifier/route-collection helpers still encode the packed-i4 resource decision as "not grouped, therefore use the default two-region schedule". That is true only because the current packed-i4 region count equals the default byte count. The slice should make packed-i4 realization decision consumption explicit through shared resource-decision helpers and focused stale packed-resource tests.

## Requirements

- Change active RVV plugin/compiler owner code for the Gate 1 slice unless source evidence proves the boundary is already complete.
- Keep resource facts plugin-owned and structural: packed nibble handling, unpack placement, VL/LMUL/resource constraints, accumulator/dequant role preservation, and schedule/resource diagnostics must not be inferred from route ids, artifact names, benchmark names, descriptor residue, or Common EmitC semantics.
- Preserve the authority chain: selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-local realization/planning -> plugin-built route -> common EmitC materialization.
- Fail closed for stale, missing, or unconsumed packed-resource facts when the packed low-precision route would otherwise make an unmeasured or resource-blind executable claim.
- Add focused tests for resource fact propagation/consumption and fail-closed stale/missing packed-resource facts.
- Keep the macro task active after this slice unless all campaign gates are complete.

## Non-Goals

- No standalone generated-bundle or `ssh rvv` evidence seam as the main task.
- No q4/q8/llama.cpp route authority.
- No route id, artifact name, benchmark name, or descriptor residue as semantic authority.
- No high-level Linalg/Vector/StableHLO frontend work.
- No broad dtype/LMUL clone batch.
- No dashboard, report-only, prompt-only, or tooling-only closeout.
- No Common EmitC invention of packed low-precision semantics.
- No unrelated MAcc, mask, segment, compare/select, or reduction expansion outside the packed low-precision resource-aware realization boundary.

## Acceptance Criteria For This Round

- [x] PRD and Trellis context identify this as the macro campaign and this round as Gate 1.
- [x] Active RVV plugin/compiler owner code changes for packed low-precision resource-aware realization/planning, or the PRD/journal records exact source-backed evidence that the Gate 1 boundary was already complete and names the same-campaign substitute production slice.
- [x] Focused tests cover positive resource fact propagation/consumption and fail-closed stale or missing packed-resource facts.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Relevant packed i4 and non-packed product-reduction-dequant fixture behavior is checked with the same `tcrv-opt` / `tcrv-translate` pipelines and key output assertions; `FileCheck` / `llvm-lit` are unavailable in this environment.
- [x] Bounded old-authority scan over touched files and added diff lines is clean.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit is created for the slice.
- [x] `.trellis/.current-task` remains active unless all campaign gates, including post-change same-target measurement, are complete.

## Technical Notes

- Direction source: user/Hermes Direction Brief in this session.
- Previous campaign endpoint: commit `b0bec496` added packed i4 same-target timing evidence and recorded a no-win/regression baseline.
- Required first reads: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/testing/mlir-testing-contract.md`, archived task `.trellis/tasks/archive/2026-06/06-09-rvv-low-precision-packed-contraction-primitive-surface-campaign/`, `scripts/rvv_generated_bundle_same_target_measure.py`, `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`, `lib/Plugin/RVV/RVVGearboxSchedules.cpp`, `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and the packed/non-packed product-reduction-dequant fixture tests.

## Open Questions

No blocking user question is currently needed. The Direction Brief gives the owner, gates, non-goals, and minimum evidence; implementation choices should be derived from the repository and specs.

## Current Gate 1 Round Result

This slice strengthens the existing Gate 1 boundary instead of repeating the previous packed-i4 primitive/evidence campaign. The prior code already consumed pass-produced low-precision resource facts into selected-body realization and route-family validation. The production gap was that several verifier and route-collection helpers treated packed-i4 as the default non-grouped two-region schedule instead of consuming the packed realization decision explicitly.

The slice adds shared RVV Gearbox resource-decision helpers that derive:

- supported realization decision class;
- expected `vsetvl` region count;
- product-region marker index;
- dequant-region marker index;
- product phase.

The selected-body realizer, RVV dialect handoff verifier, route-family realization-structure validator, and route collector now use those helpers. Packed-i4 therefore carries an explicit schedule/resource decision through producer/consumer `with_vl`, `vsetvl_region_marker`, and `gearbox_cross_region_handoff` validation. Stale packed decision, stale packed region count, or stale packed `from_phase` fail closed before route acceptance.

Focused validation completed:

- built `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and `tianchenrv-target-artifact-export-test`;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- manually executed packed-i4 positive realized/header/CPP pipelines and negative stale packed decision, stale packed region count, and stale packed `from_phase` pipelines because this environment has no `FileCheck` / `llvm-lit`;
- manually executed non-packed product-reduction-dequant positive realized/header pipelines and missing-resource fail-closed pipeline;
- ran `git diff --check`;
- ran `git diff --cached --check`;
- ran a bounded added-line authority scan with no new legacy i32/source-front-door/descriptor/Common-EmitC authority matches.

The macro task remains active. Gate 2 is the next continuation point: make route/statement/artifact validation consume these explicit resource-decision schedule facts without relying on Common EmitC or metadata mirrors as semantic authority.
