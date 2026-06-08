# RVV production-kernel capability campaign: resource-aware packed low-precision contraction realization

## Goal

Advance the RVV production-kernel capability campaign from the accepted packed i4/u4 product-reduction-dequant primitive and same-target no-win timing evidence into compiler-side, RVV-plugin-local resource-aware selected-body realization and planning. The macro owner is the RVV plugin compiler path that represents and consumes packed-resource and schedule facts before route planning and EmitC artifact generation.

## Campaign Gates

- [x] Gate 1: Resource-aware selected-body planning boundary for packed low-precision contraction is represented in RVV plugin/source and verified with focused tests.
- [x] Gate 2: Route, statement, and artifact validation consume those resource facts without Common EmitC inventing RVV semantics.
- [x] Gate 3: Generated artifact evidence reflects the new production path.
- [x] Gate 4: Same-target measurement is rerun only after a production compiler/resource-aware change and honestly reports win, regression, or no-win.

## Current Round Slice

This round implements the bounded Gate 4 same-target measurement slice: the accepted signed packed-i4 product-reduction-dequant generated RVV artifact is measured against the named scalar C packed-i4 baseline on the same `ssh rvv` target after the Gate 1-3 production compiler/resource-aware path changes.

The slice keeps the generated artifact tied to provider-owned packed-resource schedule facts and uses the packed scalar baseline only after object/header metadata validates `packed-i4-nibbles`. Acceptance is truthful same-target evidence with correctness guards before timing and an honest win/regression/no-win classification; a performance win is not required.

Gate 4 completion does not claim llama.cpp parity, q4/q8 route authority, or performance maturity. The measured result remains an evidence point for the current generated RVV artifact and named scalar baseline only.

## Repository Findings For This Round

- Gate 1 is complete in commit `a2c7f126`: the selected-body realizer, RVV dialect handoff verifier, route-family realization-structure validator, and route collector consume shared Gearbox helper facts for expected region count, product/dequant marker indices, product phase, and realization decision.
- Gate 2 is complete in commit `104df15a`: route-family planning, statement-plan ownership, route metadata, target support bundle export, and target artifact validation consume packed-i4 realization schedule facts without Common EmitC inventing RVV semantics.
- Gate 3 is complete in commit `56027bb6`: generated-bundle verification and dry-run evidence assert provider-owned schedule mirrors for the packed-i4 production path.
- The existing same-target measurement script already reuses generated-bundle ABI e2e artifact generation, selects packed-i4 timing only from validated provider metadata, and records correctness-before-timing plus raw `MEASURE`/`SUMMARY` records. The current production gap is to rerun that measurement after the Gate 1-3 resource-aware compiler changes, preserve the raw evidence path, and report the outcome without turning the measurement harness or baseline identity into route authority.

## Requirements

- Do not modify production compiler/export source unless same-target measurement reveals a precise production compiler or resource-aware artifact blocker.
- Keep resource facts plugin-owned and structural: packed nibble handling, unpack placement, VL/LMUL/resource constraints, accumulator/dequant role preservation, and schedule/resource diagnostics must not be inferred from route ids, artifact names, benchmark names, descriptor residue, or Common EmitC semantics.
- Preserve the authority chain: selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-local realization/planning -> plugin-built route -> common EmitC materialization.
- Fail closed for stale, missing, or unconsumed packed-resource facts when the packed low-precision route would otherwise make an unmeasured or resource-blind executable claim.
- Add focused tests for resource fact propagation/consumption and fail-closed stale/missing packed-resource facts.
- Finish/archive the macro task after this slice only if all campaign gates are complete and final quality verification passes.

## Non-Goals

- No standalone generated-bundle or `ssh rvv` evidence seam unrelated to the Gate 1-3 production compiler/resource-aware changes.
- No q4/q8/llama.cpp route authority.
- No route id, artifact name, benchmark name, or descriptor residue as semantic authority.
- No high-level Linalg/Vector/StableHLO frontend work.
- No broad dtype/LMUL clone batch.
- No dashboard, report-only, prompt-only, or tooling-only closeout.
- No Common EmitC invention of packed low-precision semantics.
- No unrelated MAcc, mask, segment, compare/select, or reduction expansion outside the packed low-precision resource-aware realization boundary.

## Acceptance Criteria For This Round

- [x] PRD and Trellis context identify this as the active macro campaign and this round as Gate 4, with Gates 1-3 reconciled complete through commit `56027bb6`.
- [x] Same-target measurement uses the production generated-bundle ABI e2e path for the packed-i4 generated RVV artifact and records selected input, selected variant, generated function, object/header paths, and object/header hashes.
- [x] Packed-i4 scalar baseline selection is driven by validated provider-owned `packed-i4-nibbles` metadata and uses baseline identity `scalar-c-reference/product-reduction-dequant-packed-i4-v1`.
- [x] Measurement runs both generated RVV artifact and scalar baseline on the same `ssh rvv` target with target profile, compile flags, input sizes, warmups, repeats, iterations, timing method, raw `MEASURE` records, parsed `SUMMARY` records, and correctness guards before timing.
- [x] The result is classified honestly as win, regression, or no-win; no llama.cpp parity, q4/q8 authority, or performance win is claimed without evidence.
- [x] Focused dry-run coverage checks the packed-i4 measurement harness metadata switch, baseline identity, timing structure, correctness-before-timing marker, and no descriptor/source-front-door/direct-C/q4/q8 authority drift.
- [x] Regression checks confirm the default unpacked product-dequant and dequant-clamp measurement dry-run paths keep their scalar baseline identities and do not enable the packed-i4 oracle.
- [x] Timing-specific self-test and Python compile check pass for `scripts/rvv_generated_bundle_same_target_measure.py`.
- [x] Relevant packed-i4 real measurement and default measurement dry-run commands are recorded with artifact paths.
- [x] Bounded old-authority scan over touched files and added diff lines is clean.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit is created for the slice.
- [x] `.trellis/.current-task` is ready to archive because all four campaign gates are complete after final verification.

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

## Current Gate 4 Round Result

This slice completes Gate 4 same-target measurement for the accepted signed
packed-i4 product-reduction-dequant representative after the Gate 1-3
production compiler/resource-aware changes. The measurement path reuses
`scripts/rvv_generated_bundle_same_target_measure.py`, which first regenerates
and validates the generated RVV object/header through the generated-bundle ABI
e2e path. The same-target harness selects
`scalar-c-reference/product-reduction-dequant-packed-i4-v1` only after the
generated bundle metadata validates provider-owned `packed-i4-nibbles` facts.

Raw timing evidence was collected on `ssh rvv` with remote `riscv64`, 64 CPUs,
`/usr/bin/clang`, Ubuntu clang 18.1.3, compile flags `-O2 -march=rv64gcv
-mabi=lp64d -I.`, timing method `clock_gettime(CLOCK_MONOTONIC_RAW)`, counts
`257,4096,65536`, patterns `0,1`, scales `-0.125,0.375`, warmups `2`,
repeats `5`, and iterations `8`.

Evidence artifacts:

- root:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/evidence.json`;
- per-op timing evidence:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/same_target_measurement_evidence.json`;
- raw target profile:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/remote_target_profile_stdout.txt`;
- raw timing stdout:
  `artifacts/tmp/gate4-same-target-measurement/gate4_packed_i4_same_target_measure_ssh/widening_product_reduce_dequantize_f32/remote_measure_run_stdout.txt`.

Generated artifact identity:

- selected input:
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir`;
- selected variant: `pre_realized_body_rvv_product_reduce_dequantize`;
- generated function:
  `tcrv_emitc_pre_realized_body_product_reduce_dequantize_kernel_pre_realized_body_rvv_product_reduce_dequantize`;
- object SHA-256:
  `5398112321bb4744f5b41b1ae54c7507152b4502ac4e277342278836c416d95d`;
- header SHA-256:
  `0211a71ee3c9a31feb607157aed37a4cbb53a8a2693baaa7efb504245ed1e44b`.

Parsed best-per-iteration timing summaries:

```text
n      pattern  scale   scalar ns/iter  generated ns/iter  speedup
257    0       -0.125   907.500         1187.500           0.764211
257    0        0.375   910.000         1185.000           0.767932
257    1       -0.125   907.500         1192.500           0.761006
257    1        0.375   907.500         1180.000           0.769068
4096   0       -0.125   13315.000       16565.000          0.803803
4096   0        0.375   13317.500       16565.000          0.803954
4096   1       -0.125   13315.000       16565.000          0.803803
4096   1        0.375   13315.000       16565.000          0.803803
65536  0       -0.125   212217.500      264760.000         0.801547
65536  0        0.375   213685.000      264787.500         0.807006
65536  1       -0.125   212180.000      265900.000         0.797969
65536  1        0.375   212182.500      264735.000         0.801490
```

All 12 timing summaries are no-win/regression signals for the generated RVV
artifact against the named scalar C packed-i4 baseline: best speedup ranges
from `0.761006` to `0.807006`. No performance win, llama.cpp parity, q4/q8
route authority, or benchmark-name authority is claimed.

Focused validation completed:

- ran `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`;
- ran `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`;
- ran the packed-i4 same-target measurement dry-run with the packed fixture
  override and `--llvm-readobj ""`;
- ran the packed-i4 same-target measurement on `ssh rvv` with the same fixture,
  counts `257,4096,65536`, warmups `2`, repeats `5`, iterations `8`, and
  `--llvm-readobj ""`;
- ran default product-dequant and dequant-clamp same-target measurement dry-run
  regression with `--llvm-readobj ""`; the per-op result keeps
  `scalar-c-reference/product-reduction-dequant-v1`,
  `packed_i4_reference_oracle=false`, and
  `scalar-c-reference/product-reduction-dequant-clamp-v1`;
- extended the Gate 4 same-target measurement dry-run lit test to check the
  packed-i4 metadata switch, packed scalar baseline identity, correctness guard,
  timing record shape, and packed harness structure;
- recorded that `FileCheck`, `llvm-lit`, and `llvm-readobj` are not available in
  this local environment, so dry-run checks were executed directly with
  `--llvm-readobj ""` and targeted `jq` / `rg` assertions over generated
  evidence and harness source.

All campaign gates are now complete. Final Trellis quality verification passed,
and the macro task is archived under
`.trellis/tasks/archive/2026-06/06-09-rvv-production-kernel-packed-resource-realization-campaign/`.

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
