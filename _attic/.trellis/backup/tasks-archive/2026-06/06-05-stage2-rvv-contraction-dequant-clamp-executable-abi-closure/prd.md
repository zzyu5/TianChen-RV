# Stage2 RVV contraction-dequant-clamp executable ABI closure

## Goal

Close executable support for the bounded Stage 2 RVV
`widening_product_reduce_dequant_clamp_f32` route introduced by commit
`6f89f432`. The task starts from the existing selected pre-realized
`tcrv.exec` RVV fixture, proves RVV plugin-local selected-body realization
before emission, generates the provider-derived target bundle, compiles and
runs an external ABI harness on `ssh rvv`, and compares against a host/reference
oracle for the full signed product/reduction, f32 dequantization, lower/upper
clamp, runtime count, source preservation, and output tail preservation
contract.

## What I Already Know

- The repository started clean on `main`; the latest commit was
  `6f89f432 rvv: compose contraction dequant clamp route`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-contraction-dequant-clamp-composition/`
  completed route-supported compiler composition but explicitly did not claim
  executable correctness.
- `.trellis/spec/index.md` requires the current RVV authority chain:
  `tcrv.exec` envelope -> selected typed `tcrv_rvv` body -> RVV plugin
  realization/provider -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires selected
  pre-realized RVV bodies to pass through the plugin-local selected-body
  realization owner before provider route facts are collected.
- `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  evidence to prove selected-boundary materialization, provider route metadata,
  generated harness invocation, runtime ABI order, and absence of descriptor /
  direct-C / source-export authority.
- The target fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`
  already encodes the full pre-realized composition and checks realized
  `setvl`/`with_vl`/load/load/widening_product/standalone_reduce/dequantize/
  splat/compare/select/store structure.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently supports predecessor
  op-kinds such as `widening_product_reduce_dequantize_f32` and
  `dequant_clamp_f32_epilogue`; it needs bounded neutral support for the
  composed `widening_product_reduce_dequant_clamp_f32` evidence route if dry
  run confirms the op-kind is not yet selectable.

## Requirements

- Use the existing typed selected/pre-realized route as authority. Do not infer
  semantics from route ids, artifact names, ABI strings, metadata mirrors,
  op-kind strings, q-names, exact intrinsic spellings, or legacy i32 helper
  names.
- Positive evidence must use
  `--pre-realized-selected-body --op-kind widening_product_reduce_dequant_clamp_f32`
  and include the selected-boundary materialization pass before emission.
- Dry-run bundle generation must succeed before remote execution.
- Runtime ABI must bind, in provider-validated order:
  `lhs`, `rhs`, `acc`, `scale`, `lower_bound`, `upper_bound`, `out`, and `n`.
- The external harness must compare against a reference calculation over:
  signed i8 lhs/rhs product patterns, i32 accumulator seed variants, at least
  two runtime counts, at least two nonzero f32 scale values, at least two
  ordered lower/upper bound pairs, explicit f32 tolerance, source preservation,
  accumulator/source preservation where applicable, and f32 output tail
  sentinel preservation.
- If executable evidence exposes a verifier/provider/target bug, repair
  production C++ and run the focused plugin/target checks. If no production bug
  appears, keep code changes limited to neutral harness support and evidence.
- Common EmitC/export must remain neutral: it may package provider-built
  payloads but must not choose RVV compute, dtype, scale, clamp bound,
  reduction, schedule, policy, or realization semantics.

## Acceptance Criteria

- [x] `rvv_generated_bundle_abi_e2e.py` can dry-run the composed
      `widening_product_reduce_dequant_clamp_f32` pre-realized selected-body
      fixture and verifies selected-boundary realization before emission.
- [x] The generated bundle validates provider-owned route metadata, runtime
      ABI order, route operand binding plan, header/object paths, and absence
      of descriptor/direct-C/source-export/source-front-door residue.
- [x] A non-dry-run `ssh rvv` compile/run succeeds for multiple runtime counts,
      signed input/product patterns, accumulator variants, at least two
      nonzero scale values, at least two ordered bound pairs, and explicit f32
      tolerance.
- [x] Harness output proves source preservation and f32 output tail sentinel
      preservation.
- [x] Focused fixture RUN lines pass manually with `FileCheck`; `llvm-lit` is
      not installed in this checkout. Predecessor fixture checks were not
      required because no shared production planning code changed.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes
      after harness changes.
- [x] No production C++ changes were made, so
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test` were not required.
- [x] A bounded scan over touched files shows no new `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor,
      q-name, artifact-name, ABI-string, or route-id authority.
- [x] `git diff --check` passes; `git diff --cached --check` is run after
      staging.

## Evidence Recorded

- Dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 7 --runtime-count 257 --run-id codex-product-dequant-clamp-dry --overwrite --dry-run`
  produced `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-product-dequant-clamp-dry`.
- Runtime:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 7 --runtime-count 257 --run-id codex-product-dequant-clamp-ssh --overwrite`
  produced `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-product-dequant-clamp-ssh`
  and returned `rvv_generated_bundle_abi_e2e: success`.
- Runtime evidence covered counts `0,7,257`, signed product patterns `0,1`,
  scales `-0.125,0.375`, bound pairs `[-1.5,2.25]` and `[-8,-0.75]`,
  tolerance `1e-5`, and source/accumulator/tail sentinel preservation.
- Dry-run evidence records `pre_realized_body_consumed: true` and
  `realization_boundary: public selected lowering-boundary materialization
  consumed the pre-realized typed tcrv_rvv body before provider route
  construction`.
- Spec update decision: no `.trellis/spec/` update was needed. This round did
  not introduce a new compiler/API contract; it instantiated the existing RVV
  generated-bundle and `ssh rvv` evidence contracts for the composed route.

## Out Of Scope

- New route coverage beyond `widening_product_reduce_dequant_clamp_f32`.
- q8/q4/llama benchmark route, ProviderSpec/model-name route authority, or
  zero-point expansion.
- Standalone contraction-dequant or dequant-clamp evidence as the main
  deliverable.
- High-level Linalg/Vector/StableHLO frontend work.
- dtype/LMUL clone batches, one-intrinsic wrappers, second realization family,
  compatibility wrappers preserving old i32m1 authority, broad smoke matrices,
  dashboards, or report-only work.

## Technical Notes

- Primary code/evidence file:
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Primary fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`.
- Predecessor fixtures:
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`.
- Production files from the Direction Brief are read/checked if dry-run or
  remote execution exposes a production bug:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
