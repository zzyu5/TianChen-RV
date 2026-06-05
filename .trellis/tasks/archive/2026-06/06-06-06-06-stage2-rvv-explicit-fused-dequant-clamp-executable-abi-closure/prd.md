# Stage2 RVV explicit fused dequant-clamp executable ABI closure

## Goal

Close executable ABI evidence for the explicit selected-body
`widening_product_reduce_dequant_clamp_f32` path. The task starts from the
non-pre-realized typed `tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_body`
fixture, carries it through RVV plugin-local realization, provider-derived
`TCRVEmitCLowerableRoute`, target header/artifact export, generated-bundle
dry-run, and real `ssh rvv` correctness evidence when the remote/toolchain is
reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV explicit fused dequant-clamp executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `5130bebf rvv: add explicit fused dequant clamp realization`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- Commit `5130bebf` already made the explicit selected-body surface and
  plugin-local realization route-supported through target artifact evidence.
- The archived explicit-realization PRD explicitly says it did not claim new
  `ssh rvv` correctness for the explicit path.
- Existing pre-realized generated-bundle evidence already checks the scalar
  oracle, signed lane patterns, scale cases, bound pairs, source preservation,
  accumulator preservation, output tail preservation, and counts `0,1,16,17,257`.
- The immediate gap is the generated-bundle input bridge for the explicit
  selected-body fixture and its real hardware execution evidence.

## Requirements

- Start from
  `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`.
- Produce explicit selected-body generated-bundle dry-run evidence that uses the
  explicit fixture and explicit generated ABI:
  `lhs, rhs, acc, scale, lower_bound, upper_bound, out, n`.
- Execute the generated bundle on real `ssh rvv` if reachable.
- Validate scalar-oracle correctness for counts `0`, `1`, VL-boundary, tail,
  and multi-chunk cases.
- Preserve signed lane patterns, scale cases, lower/upper bound pairs, source
  preservation, accumulator preservation, and output tail sentinel preservation.
- If `scripts/rvv_generated_bundle_abi_e2e.py` only supports the pre-realized
  fused input mode, repair only the minimal production/script bridge needed to
  feed the explicit realized body into the existing generated-bundle ABI path.
- Keep the existing pre-realized fused generated-bundle regression passing if
  shared script or route code is touched.
- Record the exact external blocker if remote hardware/toolchain execution is
  unavailable, and do not claim executable correctness without real `ssh rvv`
  output.

## Acceptance Criteria

- [x] Explicit selected-body generated-bundle dry-run succeeds and evidence JSON
      identifies the explicit input mode/fixture/function rather than the
      pre-realized fixture.
- [x] The dry-run evidence preserves provider-derived fused route facts:
      runtime ABI order, operand binding, route-family plan, provider mirror,
      type mapping, product/reduction/dequant/clamp facts, target header, and
      no descriptor/source-front-door authority.
- [x] Real `ssh rvv` generated-bundle compile/run succeeds, or the exact
      external blocker is recorded without making a runtime correctness claim.
- [x] Runtime evidence, when reachable, covers counts `0,1,16,17,257`,
      patterns `0,1`, scale values `-0.125,0.375`, bound pairs
      `-1.5:2.25,-8:-0.75`, scalar oracle comparison, source preservation,
      accumulator preservation, and output tail preservation.
- [x] Existing explicit target/header fixture remains passing.
- [x] Existing pre-realized fused generated-bundle dry-run remains passing if
      shared script or production code changes.
- [x] Script self-test passes if the script changes.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python changes, if any, are limited to the generated-bundle support harness.
- No new route-family expansion, broad dtype/LMUL/reduction matrix, frontend
  work, performance/autotuning, source-front-door positive route, or common
  EmitC fused-semantic invention is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or an external execution blocker is proven.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-widening-product-reduce-dequant-clamp-f32-explicit-realization/prd.md`.
- Initial files inspected:
  `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run.test`.

## Completion Evidence

- Added explicit selected-body expectation support to
  `scripts/rvv_generated_bundle_abi_e2e.py` for
  `widening_product_reduce_dequant_clamp_f32`, pointing at
  `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`
  and function
  `tcrv_emitc_explicit_wprdc_kernel_explicit_rvv_wprdc`.
- The script now routes this explicit compound body through
  `--tcrv-materialize-selected-lowering-boundaries` before emission planning,
  and evidence records
  `explicit_compound_body_consumed: true`,
  `pre_realized_body_consumed: false`, and
  `rvv-plugin-local-selected-body-realization-owner-registry`.
- Added
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-product-reduce-dequant-clamp-f32-dry-run.test`
  to check explicit fixture/function evidence, provider route mirrors, target
  artifact facts, and harness oracle coverage.
- Explicit generated-bundle dry-run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-product-reduce-dequant-clamp-f32`.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-product-reduce-dequant-clamp-f32-ssh`.
  Remote compile used `riscv64`, `/usr/bin/clang`, Ubuntu clang `18.1.3`.
- Remote PASS marker:
  `PASS op=widening_product_reduce_dequant_clamp_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05`.
- Pre-realized fused generated-bundle dry-run regression passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-product-reduce-dequant-clamp-f32`.
- Checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  focused explicit generated-bundle dry-run;
  focused pre-realized generated-bundle dry-run;
  focused `tcrv-opt` explicit realization/emission-plan command;
  direct `FileCheck` for the new dry-run test's `ROOT`, `WPRDC`, and `HARNESS`
  prefixes;
  `git diff --check`;
  `git diff --cached --check`;
  Trellis context validation;
  bounded old-authority scan over touched files and added diff lines.
- Local `python3 -m lit` is unavailable with `No module named lit`; direct
  `FileCheck` prefix validation was used for the new test file.

## Out Of Scope

- No new route-family expansion beyond
  `widening_product_reduce_dequant_clamp_f32`.
- No high-level Linalg/Vector/StableHLO frontend path.
- No one-op-per-intrinsic wrapping or dtype-prefixed helper growth.
- No source-front-door/source-artifact positive route.
- No compatibility wrapper preserving old i32 authority.
- No broad generated-bundle harness rewrite beyond the explicit-path bridge.
