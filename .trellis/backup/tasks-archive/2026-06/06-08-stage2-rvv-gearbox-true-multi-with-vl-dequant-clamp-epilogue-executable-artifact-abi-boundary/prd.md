# Stage2 RVV Gearbox true multi-with_vl dequant-clamp epilogue executable artifact ABI boundary

## Goal

Make the bounded Stage 2 Gearbox `widening_product_reduce_dequant_clamp_f32`
route executable through the same true multi-`with_vl` handoff boundary that
the adjacent product-reduction dequantize route already proved. Selected and
pre-realized typed `tcrv_rvv` bodies must realize product/reduction in a
producer `with_vl`, forward the i32 reduction through
`tcrv_rvv.gearbox_cross_region_handoff`, consume it in a nested consumer
`with_vl`, then dequantize, clamp with lower/upper compare/select, store, export
provider/target ABI facts, and pass generated-bundle correctness on `ssh rvv`.
If any required handoff, clamp, resource, ABI, target mirror, or runtime fact is
missing or stale, the route must fail closed before an executable claim.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository started clean at `a4714c9d rvv: harden gearbox generated bundle ABI evidence`.
* The previous archived task hardened generated-bundle ABI evidence for the
  non-clamp `widening_product_reduce_dequantize_f32` route and proved explicit
  and pre-realized generated bundles on `ssh rvv`.
* Current dequant-clamp fixtures and dry-run script coverage already check
  compare/select, lower/upper bound roles, ABI order, binding plan, source /
  accumulator / tail preservation, and clamp saturation/non-saturation cases.
* The current spec still describes `widening_product_reduce_dequantize_f32` as
  the established Gearbox cross-region handoff route and notes that
  dequant-clamp keeps a direct reduction-to-dequant epilogue until a separate
  contract exists.
* Current code evidence shows the production seam is real: route planning,
  construction, target validation, and evidence script have begun recognizing
  `tcrv_rvv.gearbox_cross_region_handoff` for
  `widening_product_reduce_dequant_clamp_f32`, but
  `RVVContractionSelectedBodyRealizationOwner.cpp` still materializes the
  handoff only for the non-clamp dequantize route.

## Requirements

* Keep implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck. Python may
  only support generated-bundle tooling and evidence parsing.
* Extend the RVV-owned Gearbox handoff contract to the bounded
  `widening_product_reduce_dequant_clamp_f32` family. Do not use route ids,
  helper names, test names, artifact metadata, descriptors, or Common EmitC as
  compute or route authority.
* The RVV contraction selected-body realization owner must materialize producer
  and consumer `with_vl` scopes, region markers, and
  `gearbox_cross_region_handoff` for both explicit and pre-realized
  dequant-clamp selected bodies before route/provider construction.
* The dequant-clamp consumer scope must preserve the clamp epilogue semantics:
  `dequantize` consumes the handoff result, lower/upper splats use runtime
  bounds, compare/select order is lower-then-upper, and store writes the
  clamped f32 result.
* RVV provider, construction protocol, target artifact validation, generated
  bundle evidence, and tests must agree on the structural typed compute chain:
  `widening_product + standalone_reduce + gearbox_cross_region_handoff +
  dequantize + compare + select`.
* Target artifact/header/object validation must continue to compare provider
  mirrors for runtime ABI order, route operand bindings, resource/scope facts,
  handoff facts, dequantization facts, clamp facts, type/header mapping, and
  provider support before accepting a bundle.
* Runtime correctness claims require non-dry-run generated bundle execution on
  `ssh rvv`; dry-run checks validate only source, header, metadata, and harness
  shape.

## Acceptance Criteria

* [ ] Explicit dequant-clamp selected body realizes to a true multi-`with_vl`
  producer/consumer Gearbox body with handoff before dequantize and clamp.
* [ ] Pre-realized dequant-clamp selected body realizes to the same handoff
  structure before emission-plan and target artifact export.
* [ ] Provider/route/target evidence exposes the handoff/resource/scope facts
  and the clamp compare/select facts as provider-derived mirror fields, not as
  script or artifact authority.
* [ ] Positive generated-bundle evidence passes for explicit and pre-realized
  dequant-clamp routes, preferably both dry-run and non-dry-run `ssh rvv`, over
  representative `n`, patterns, scales, lower/upper bound pairs, and
  saturation/non-saturation cases.
* [ ] At least one focused fail-closed check rejects missing/stale handoff,
  stale producer/consumer scope, stale VL/AVL handoff, stale resource fact,
  stale clamp/bound/compare-select fact, stale ABI/header/object mirror, or
  route-family validation drift.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass.
* [ ] If `scripts/rvv_generated_bundle_abi_e2e.py` changes, its `--self-test`
  passes.
* [ ] Relevant lit/FileCheck RUN lines are replayed if tools are available; if
  not, the direct binary/script checks cover the changed behavior and the
  missing tool is recorded.
* [ ] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, descriptor-driven compute,
  source-front-door authority, or artifact/status authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are reported.

## Out of Scope

* No broad Gearbox matrix, dtype/LMUL clone batch, source-front-door positive
  route, high-level Linalg/Vector/StableHLO frontend, performance tuning
  database, dashboard, or report-only closeout.
* No unrelated MAcc, reduction, memory, segment, TensorExt, IME, offload, or
  scalar fallback expansion.
* No Common EmitC invention of RVV semantics and no descriptor/direct-C/source
  export route.
* No performance or llama.cpp parity claim; this task is executable artifact
  correctness and ABI boundary evidence only.

## Technical Notes

* Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/validation/index.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
* Relevant previous task:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-true-multi-with-vl-executable-artifact-abi-boundary/`.
* Initial production seam candidates:
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the explicit/pre-realized
  dequant-clamp fixtures under `test/Target/RVV` and `test/Scripts`.
