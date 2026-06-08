# RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

## Goal

Advance the RVV production-kernel capability campaign from the accepted packed i4/u4 product-reduction-dequant primitive and same-target no-win timing evidence into compiler-side, RVV-plugin-local resource-aware selected-body realization and planning. The macro owner is the RVV plugin compiler path that represents and consumes packed-resource and schedule facts before route planning and EmitC artifact generation.

## Campaign Gates

- [x] Gate 1: Resource-aware selected-body planning boundary for packed low-precision contraction is represented in RVV plugin/source and verified with focused tests.
- [x] Gate 2: Route, statement, and artifact validation consume those resource facts without Common EmitC inventing RVV semantics.
- [x] Gate 3: Generated artifact evidence reflects the new production path.
- [ ] Gate 4: Same-target measurement is rerun only after a production compiler/resource-aware change and honestly reports win, regression, or no-win.

## Current Round Slice

This round implements a bounded Gate 3 generated-artifact evidence slice: the packed-i4 generated bundle evidence must be regenerated from the now-validated production route/statement/artifact path and must mirror provider-owned packed-resource schedule facts without becoming route or compute authority.

The slice focuses on the accepted signed packed-i4 product-reduction-dequant representative. The generated evidence must expose the selected low-precision resource candidate, operand form, storage/effective widths, packing layout, unpack intent, resource realization producer, resource realization decision, realized unroll factor, realized `vsetvl` region count, realized peak live-vector groups, product/dequant marker indices, product/dequant phases, runtime AVL source, producer/consumer scopes, runtime ABI order, provider target capability mirrors, packed low/high nibble statement payloads, generated header/object mirror agreement, and the packed harness oracle switch derived from validated provider metadata.

Gate 3 completion does not claim runtime correctness, performance, or llama.cpp/q4/q8 parity. The generated bundle is evidence that the production path preserved provider-owned facts; Gate 4 same-target measurement remains a later slice.

## Repository Findings For This Round

- Gate 1 is complete in commit `a2c7f126`: the selected-body realizer, RVV dialect handoff verifier, route-family realization-structure validator, and route collector consume shared Gearbox helper facts for expected region count, product/dequant marker indices, product phase, and realization decision.
- Gate 2 is complete in commit `104df15a`: route-family planning, statement-plan ownership, route metadata, target support bundle export, and target artifact validation consume packed-i4 realization schedule facts without Common EmitC inventing RVV semantics.
- The generated-bundle script already selected the packed scalar oracle from validated low-precision metadata, but its object/header verifier and evidence summary did not explicitly require the Gate 2 realization schedule mirror fields.
- The Gate 3 production-evidence gap is therefore to make generated bundle verification and dry-run evidence assert the same provider-owned schedule mirrors, then add a focused packed-i4 dry-run test that checks evidence JSON, bundle index metadata, emitted C++ statement shape, and harness oracle selection.

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

- [x] PRD and Trellis context identify this as the active macro campaign and this round as Gate 3, with Gate 2 reconciled complete from commit `104df15a`.
- [x] Generated-bundle verification treats low-precision realization schedule metadata as required object/header mirrors for product-dequant routes.
- [x] Packed-i4 generated evidence exposes realization producer/decision, realized unroll, realized `vsetvl` region count, realized peak live-vector groups, product/dequant region indices, product/dequant phases, runtime ABI order, and target capability mirrors.
- [x] Packed-i4 dry-run evidence checks the production pipeline from pre-realized selected body through materialized selected body, emitted RVV C++, target artifact bundle index, and external ABI harness.
- [x] Packed-i4 harness oracle selection remains driven by validated provider-owned `packed-i4-nibbles` metadata and records runtime `n` as packed input bytes.
- [x] Focused fail-closed checks cover stale or missing generated-artifact realization schedule mirrors in script verification and target artifact export.
- [x] Regression checks confirm the default unpacked product-dequant and dequant-clamp generated bundle paths still follow their own schedule facts.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Relevant packed-i4, unpacked product-dequant, and dequant-clamp dry-run pipelines are checked with `tcrv-opt` / `tcrv-translate`; `FileCheck` / `llvm-lit` availability is recorded honestly.
- [x] Bounded old-authority scan over touched files and added diff lines is clean.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit is created for the slice.
- [x] `.trellis/.current-task` remains active because Gate 4 remains incomplete.

## Current Gate 2 Round Result

This slice carries the Gate 1 packed-resource realization schedule facts into the Gate 2 production consumption path. `RVVLowPrecisionContractionResourceSelection` now retains the provider-owned realization producer, realization decision, realized unroll factor, realized `vsetvl` region count, realized peak live-vector groups, product/dequant region indices, and product/dequant phases. Route-family planning derives those facts from the same RVV Gearbox resource-decision helpers used by selected-body realization, and pass-fact reconstruction imports them from the typed body before route acceptance.

Direct-contraction statement planning now compares the provider plan against the route-family plan for the same schedule fields before constructing the packed-i4 statement payload. The packed-i4 owner still requires the explicit operand/resource facts: `packed-i4-nibbles`, signed source, storage width 8, effective width 4, two signed i4 nibbles per byte, sign-extension before widening product, unroll 1, two realized `vsetvl` regions, peak live vector groups 6, product region 1 with phase `load-product-reduce`, and dequant region 2 with phase `dequant-store`.

Route metadata and target artifact support bundle export those schedule facts as provider-owned mirrors. Target artifact route-family validation now rejects stale realization schedule mirrors before header artifact acceptance; the added packed-i4 stale-realization-decision check fails when the artifact payload is mutated to `artifact-name-derived-resource-decision`.

Focused validation completed:

- built `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and `tianchenrv-target-artifact-export-test`;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- manually executed packed-i4 PLAN, HEADER, and C++ emission pipelines because this environment has no `FileCheck` in `PATH`;
- manually executed stale packed decision, stale packed region count, stale packed `from_phase`, and stale artifact realization decision fail-closed pipelines;
- manually executed non-packed product-reduction-dequant header export and missing-resource fail-closed regression pipelines;
- ran `git diff --check`;
- ran `git diff --cached --check`;
- ran a bounded added-line authority scan; the only match was the intentional negative-test string `artifact-name-derived-resource-decision`.

Gate 2 is complete for this production-source consumption slice. The macro task remains active because Gate 3 generated artifact evidence and Gate 4 same-target measurement rerun are still unfinished. The next continuation point is Gate 3: regenerate and inspect the packed-i4 generated artifact evidence from the now-validated production route/statement/artifact consumption path, without treating generated artifacts as semantic authority.

## Current Gate 3 Round Result

This slice reconciles Gate 2 as complete from commit `104df15a` and closes Gate 3 generated-artifact evidence for the accepted signed packed-i4 product-reduction-dequant representative.

`scripts/rvv_generated_bundle_abi_e2e.py` now treats low-precision realization schedule mirrors as required product-dequant object/header metadata. The verifier checks the provider-owned realization producer, realization decision, realized unroll factor, realized `vsetvl` region count, realized peak live-vector groups, product/dequant region indices, product/dequant phases, runtime ABI order, and target capability mirrors. The product-dequant evidence summary now emits `generated_artifact_resource_schedule_evidence` with object/header agreement and expected-field mirrors so Gate 3 evidence is visible in JSON instead of only as raw bundle index text.

The focused packed-i4 dry-run test `rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequantize-f32-packed-i4-dry-run.test` regenerates the artifact from the packed fixture through the production materialize/export path. It checks evidence JSON, target artifact bundle index metadata, packed low/high nibble statement payloads, and the external ABI harness oracle switch derived from validated `packed-i4-nibbles` metadata. The default unpacked product-dequant and dequant-clamp dry-run paths still pass with their own non-packed schedule facts.

The generated-bundle schedule mirror contract is also recorded in the testing and RVV plugin specs so future Gate 3-style evidence checks require the same provider-owned schedule mirrors and missing/stale fail-closed behavior.

Focused validation completed:

- ran `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
- manually executed the packed-i4 generated-bundle dry-run with `--input test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`;
- manually executed default unpacked product-dequant and dequant-clamp generated-bundle dry-run regressions;
- manually confirmed packed-i4 stale artifact realization decision fails at target artifact export with the provider-selected realization decision diagnostic;
- ran `build/bin/tianchenrv-rvv-extension-plugin-test`;
- ran `build/bin/tianchenrv-target-artifact-export-test`;
- recorded that `FileCheck`, `llvm-lit`, and `llvm-readobj` are not available in this local environment, so the dry-run checks were executed directly with `--llvm-readobj ""` and targeted `rg` assertions over generated evidence;
- ran `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
- ran `git diff --check`;
- ran `git diff --cached --check`;
- ran a bounded added-line authority scan.

Gate 3 is complete for generated artifact evidence. The macro task remains active because Gate 4 same-target measurement rerun is still unfinished. The next continuation point is Gate 4: rerun same-target measurement after this production compiler/resource-aware evidence change and honestly report win, regression, or no-win.

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
