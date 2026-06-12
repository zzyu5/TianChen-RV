# Stage2 RVV resource-selected widening product-reduce dequantize-f32 executable artifact ABI boundary

## Goal

Make one bounded Stage 2 low-precision direct-contraction workflow executable:
`widening_product_reduce_dequantize_f32` must flow from selected typed
`tcrv_rvv` body facts through the RVV-owned low-precision resource selection,
provider-built `TCRVEmitCLowerableRoute`, neutral Common EmitC materialization,
target artifact export, generated bundle ABI, and real `ssh rvv` correctness
evidence.

If current production code is already sufficient, close the exact executable
evidence blocker with focused generated-bundle/`ssh rvv` evidence and
fail-closed validation. Do not close this task as report-only, metadata-only,
or Trellis-only evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV resource-selected widening product-reduce dequantize-f32 executable artifact ABI boundary`

## Entry-Gate State

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Recent commits included HEAD
  `31bbcade rvv: seed low precision resource candidate`, followed by
  `2cabac98 chore: record journal`,
  `b9bb2569 rvv: repair product dequant family-plan mirror`,
  `2f483410 docs: record rvv stage2 resource-aware steering`, and
  `0df6b83a rvv: integrate widening product artifact contract core`.
- No `.trellis/.current-task` file existed, so this task was created from the
  Hermes brief before source edits.

## What I Already Know

- The previous archived task seeded
  `RVVLowPrecisionContractionResourceSelection` and threaded selected resource
  facts through provider planning, target validation, artifact metadata mirrors,
  header evidence, and stale metadata rejection.
- That previous seed intentionally made no runtime correctness or performance
  claim. This task must prove one concrete executable artifact path or expose
  the exact production executable-boundary blocker.
- Specs require q8/q4 and llama.cpp examples to remain pressure tests only.
  They must not become route ids, artifact names, helper names, benchmark-name
  authority, or handwritten intrinsic shortcuts.
- Common EmitC and target export are neutral consumers of provider-built facts.
  They must not infer RVV semantics from artifact metadata, route ids, ABI
  strings, emitted intrinsic spelling, descriptors, helper names, or tests.
- Runtime/correctness claims for RVV require real `ssh rvv` execution evidence.

## Requirements

- Prove or repair the production path for the selected
  `widening_product_reduce_dequantize_f32` route family:
  selected typed body facts -> RVV resource candidate selection ->
  direct-contraction provider route plan -> `TCRVEmitCLowerableRoute` ->
  Common EmitC -> target artifact bundle -> generated bundle ABI -> `ssh rvv`
  correctness.
- The executable boundary must keep these facts aligned:
  source/product/accumulator/result dtype; SEW/LMUL/EMUL; memory form;
  tail/mask policy; unroll; accumulator count; reduction layout; vsetvl region
  count; peak live vector-group estimate; register budget; runtime AVL/VL;
  runtime ABI order; target capability mirrors; provider route validation;
  header/prototype binding; generated bundle harness semantics.
- Provider-selected resource facts must be consumed before route construction
  or target artifact acceptance. Artifact metadata may mirror those facts only
  after provider validation exists.
- Harden the boundary if any dry-run-only, stale, under-validated, or
  metadata-authorized behavior is found.
- Add or retain focused fail-closed evidence for at least one stale/missing
  executable-boundary fact, preferably stale selected candidate, stale resource
  mirror, wrong dtype/SEW/LMUL/EMUL relation, wrong reduction layout, wrong
  register-budget/capability mirror, stale header/prototype binding, wrong ABI
  value order, or unsupported executable route claim.
- Keep dequant-clamp as bounded reference only. Do not expand this round into a
  broad low-precision matrix.

## Acceptance Criteria

- [x] PRD and Trellis context truthfully describe this bounded module owner.
- [x] Production code evidence shows the
      `widening_product_reduce_dequantize_f32` executable path is provider-owned
      and resource-selected, or source changes repair the precise blocker.
- [x] Positive evidence covers selected boundary/materialized route planning,
      selected low-precision resource candidate, target artifact export,
      generated bundle compile, and real `ssh rvv` correctness if executable
      behavior is claimed.
- [x] Fail-closed evidence covers at least one stale or missing
      executable-boundary fact relevant to the resource-selected route.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Relevant lit/generated-bundle tests for
      `widening_product_reduce_dequantize_f32` pass.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` is exercised for this op in
      dry-run and, when correctness is claimed, non-dry-run `ssh rvv` mode.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-front-door, descriptor, direct-C, helper-name,
      artifact-name, or metadata-as-authority drift.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Task state, journal, archive, and commit are ready for the final coherent
      commit.

## Out Of Scope

- Broad low-precision resource matrices or dtype/LMUL/EMUL clone batches.
- Dequant-clamp executable expansion except as bounded reference evidence.
- MAcc, mask, or unrelated contraction-family rework.
- Performance tuning databases, global autotuning, dashboards, readiness
  state machines, or report-only/index-only work.
- High-level Linalg/Vector/StableHLO frontend work or per-Linalg route
  authority.
- Source-front-door positive routes.
- Common EmitC invention of RVV semantics.
- Moving provider authority into target artifact metadata.
- Runtime performance or llama.cpp parity claims unless same-target correctness
  and timing evidence are produced in this task.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Archived context read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-low-precision-resource-candidate-seed/prd.md`
  and `.trellis/workspace/codex/journal-24.md`.
- Primary source/test paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  and relevant generated-bundle / target artifact tests.

## Current Phase

finish.

## Implementation Completed

- Audited the production executable artifact seam for
  `widening_product_reduce_dequantize_f32` across RVV provider planning,
  direct-contraction resource selection, route description metadata mirrors,
  target artifact validation, target support bundle export, and generated
  bundle evidence tooling.
- Confirmed no production source change was required: current code already
  derives the selected low-precision resource candidate from typed
  contraction-plan and selected target-capability facts, requires that legal
  selected candidate before direct provider route construction, copies it into
  the provider route description, and validates every target artifact mirror
  against provider-selected facts before bundle acceptance.
- Closed the exact executable evidence blocker by proving both explicit and
  pre-realized selected-body `widening_product_reduce_dequantize_f32` generated
  bundles compile and run on `ssh rvv` with source, accumulator, and output-tail
  preservation checks.

## Verification Completed

- [x] `ninja -C build tcrv-opt tcrv-translate
      tianchenrv-rvv-extension-plugin-test
      tianchenrv-target-artifact-export-test`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `build/bin/tianchenrv-target-artifact-export-test`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
      explicit-selected-body-artifact-widening-product-reduce-dequantize-f32`
      from `build/test`
- [x] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
      pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`
      from `build/test`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Explicit selected-body dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --op-kind
      widening_product_reduce_dequantize_f32 --run-id
      stage2-product-dequant-explicit-dry --overwrite`
- [x] Pre-realized selected-body dry-run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
      --pre-realized-selected-body --op-kind
      widening_product_reduce_dequantize_f32 --run-id
      stage2-product-dequant-prerealized-dry --overwrite`
- [x] Explicit selected-body `ssh rvv` correctness:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind
      widening_product_reduce_dequantize_f32 --run-id
      stage2-product-dequant-explicit-ssh --overwrite`
- [x] Pre-realized selected-body `ssh rvv` correctness:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py
      --pre-realized-selected-body --op-kind
      widening_product_reduce_dequantize_f32 --run-id
      stage2-product-dequant-prerealized-ssh --overwrite`
- [x] `git diff --check`
- [x] `git diff --cached --check`
- [x] Bounded added-diff authority scan for `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, `source-front-door`,
      `descriptor`, `direct-C`, `helper-name`, `artifact-name`, and
      `metadata-as-authority` found no added positive route-authority usage.

## Evidence Artifacts

- Explicit dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-explicit-dry/evidence.json`
- Pre-realized dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-prerealized-dry/evidence.json`
- Explicit `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-explicit-ssh/evidence.json`
- Pre-realized `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-prerealized-ssh/evidence.json`
