# Stage2 RVV product-reduction executable ABI closure

## Goal

Close executable ABI evidence for one generated RVV low-precision
product-reduction artifact: selected `tcrv.exec` RVV variant -> typed
`tcrv_rvv` signed i8 unit loads -> `tcrv_rvv.widening_product` i8mf4 to
i16mf2 -> `tcrv_rvv.standalone_reduce` signed i16-to-i32 reduce-add -> i32
runtime result boundary -> generated header/object bundle -> external C ABI
consumer -> real `ssh rvv` correctness evidence.

This task improves the executable support level for the production route added
by commit `d706190d`. It must not add new route coverage, benchmark-specific
authority, descriptor/source-front-door computation, or common EmitC route
decisions.

## What I Already Know

- Repository session started with no active Trellis task; this task was created
  from the Hermes Direction Brief.
- `main` is clean at task start and HEAD is `d706190d rvv: add product
  reduction contraction chain`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-low-precision-product-reduction-chain/`
  completed route-supported and target-artifact validation for the typed
  product-plus-reduction chain, but explicitly did not claim runtime
  correctness or performance.
- The route contract is already documented in
  `.trellis/spec/lowering-runtime/emitc-route.md` as
  `RVV Product-Reduction Chain Route Contract`.
- `test/Dialect/RVV/generic-widening-product-reduction-chain-dataflow.mlir`
  proves the typed dataflow shape.
- `test/Target/TargetArtifactExportTest.cpp` has an embedded selected-body
  product-reduction fixture for target artifact validation.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently has no
  `widening_product_reduce_add` op-kind expectation, so it cannot yet dry-run
  or remotely execute this generated artifact through the existing ABI
  evidence path.

## Scope

- Add only the minimal generated-bundle ABI consumer support needed for
  `widening_product_reduce_add`.
- Add or select a representative selected-body target fixture for the product
  reduction chain that uses the existing typed route facts as authority.
- Bind runtime ABI roles exactly as provider facts already specify:
  `lhs`, `rhs`, `acc`, `out`, `n`.
- Generate the artifact bundle through existing `tcrv-opt` /
  `tcrv-translate` flow before any remote execution.
- Compile and run the generated object/header plus external C harness on
  `ssh rvv` before claiming executable correctness.
- Compare runtime output against a host/reference calculation for multiple
  counts and data patterns in the harness.
- If runtime evidence exposes a production route bug, repair the production
  route and rerun focused route/provider/target checks.

## Requirements

- The script support must consume provider-derived artifact/header/runtime
  facts and generated symbols; it must not infer dtype, compute semantics,
  schedule, VL policy, route support, or ABI order from q-like names, artifact
  names, route ids, op-kind strings, or metadata mirrors.
- The correctness oracle may calculate expected scalar results in the external
  harness from concrete runtime input arrays, because that is evidence
  comparison, not compiler route authority.
- Product-reduction evidence must show the typed source/product/result chain:
  i8mf4 sources, i16mf2 product intermediate, i32m1 scalar result boundary,
  and `lhs,rhs,acc,out,n` runtime ABI.
- Dry-run validation must complete before remote execution.
- Remote correctness evidence must use real `ssh rvv`; no runtime correctness
  or performance claim may be made without that run.
- Touched files must not introduce positive legacy `RVVI32M1`,
  `rvv-i32m1`, new `tcrv_rvv.i32_*` route authority, q8/q4/llama benchmark
  authority, descriptor-driven computation, or source-front-door positive
  routes.

## Acceptance Criteria

- [x] A generated-bundle ABI e2e op-kind exists for
  `widening_product_reduce_add`.
- [x] The selected input fixture structurally carries typed signed i8 unit
  loads, an i16 widening product, an i16-to-i32 standalone reduction, i32 result
  store, and `lhs,rhs,acc,out,n` runtime ABI roles.
- [x] Dry-run generated-bundle validation passes for
  `widening_product_reduce_add`.
- [x] The generated header prototype and bundle metadata expose
  `const int8_t *lhs`, `const int8_t *rhs`, `const int32_t *acc`,
  `int32_t *out`, and `size_t n`.
- [x] The generated C harness validates multiple runtime counts and at least
  two data patterns against a reference sum of signed i8 products plus the
  scalar accumulator seed.
- [x] Real `ssh rvv` compile/run passes before executable correctness is
  claimed.
- [x] Focused compiler checks pass for any touched product-reduction fixture or
  provider/target code.
- [x] Script self-test or focused dry-run guards prove the new consumer support
  remains neutral and does not choose route semantics from names.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Bounded old-authority/q-name scan over touched files passes.
- [x] Task status, journal, and final report truthfully describe whether
  production code changed or only neutral harness/evidence support changed.
- [x] One coherent commit records the completed task, or the task remains open
  with the exact next continuation point.

## Completion Evidence

- Added generated-bundle ABI consumer support for
  `widening_product_reduce_add`, using the typed selected-body fixture
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir`.
- Generated and validated the product-reduction artifact locally:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-reduce-dry-run`.
- Executed the generated header/object plus external harness on real `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-reduce-ssh-rvv`.
- `ssh rvv` passed for counts `1,7,16,17,257` and patterns `0,1`, checking
  signed i8 products, i16 product intermediates, i32 accumulator seed, scalar
  output boundary, source preservation, and tail sentinel preservation.
- Production code did change because executable evidence exposed two real route
  bugs: the product-reduction `tcrv_rvv.load` verifier was too early/strict
  for reparsed materialized MLIR, and the provider-derived widening reduction
  intrinsic spelling was not accepted by the RVV C header.
- Harness support also self-repaired its representative data pattern so every
  non-trivial count exercises an i16 product intermediate.

## Definition Of Done

- The product-reduction generated artifact can be dry-run locally and executed
  on `ssh rvv` through the existing generated-bundle ABI evidence path, with
  runtime output matching the harness reference for representative counts and
  patterns.
- No production route behavior is changed unless remote evidence exposes a real
  route bug.
- The worktree is clean after the final commit if the task is complete.

## Out Of Scope

- New q8/q4/llama benchmark-specific routes or names as authority.
- Dequantization, new dtype family expansion, or high-level Linalg/frontend
  lowering.
- Handwritten C demo as the main deliverable; the C harness must consume the
  generated header/object.
- Compatibility wrappers preserving legacy i32m1 route authority.
- Broad smoke matrices, dashboards, report-only expansion, or route-support
  decisions in common EmitC/export.
- Inferring dtype, compute, schedule, or route support from route ids, artifact
  names, test names, op-kind strings, or metadata mirrors.

## Technical Notes

- Specs read before PRD: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-04-stage2-rvv-low-precision-product-reduction-chain/prd.md`
  and task metadata.
- First implementation targets:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir`
  or an equivalent fixture if an existing one is discovered.
- Production C++ route targets are read/repair-only unless evidence exposes a
  route bug:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.
