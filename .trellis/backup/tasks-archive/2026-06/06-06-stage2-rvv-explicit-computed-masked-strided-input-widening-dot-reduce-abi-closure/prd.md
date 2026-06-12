# Stage2 RVV explicit computed-masked strided-input widening-dot reduce executable ABI closure

## Goal

Close executable ABI evidence for the explicit selected-body
`computed_masked_strided_input_widening_dot_reduce_add` path. The task starts
from the existing explicit typed `tcrv_rvv` body with compare-produced mask,
strided signed i16 dot inputs, i32 scalar accumulator/result, runtime
`n/lhs_stride/rhs_stride`, policy/VL facts, and provider-derived contraction
route facts; carries it through `TCRVEmitCLowerableRoute`, target
artifact/generated bundle, and real `ssh rvv` correctness evidence when the
remote/toolchain is reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV explicit computed-masked strided-input widening-dot reduce executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `a6917dad rvv: close explicit fused dequant clamp executable abi`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- The prior archived task closed explicit
  `widening_product_reduce_dequant_clamp_f32` generated-bundle execution on
  `ssh rvv`; it is prior evidence only and not authority for this route.
- Existing target fixtures already expose the explicit and pre-realized
  computed-masked strided-input widening dot-reduce selected bodies.
- Existing dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`
  checks provider/target facts, stale mirror negatives by fixture, and harness
  source-level oracle structure for counts `0,1,16,17,257` and stride pairs
  `2:3,3:2`.
- The immediate gap is proving the explicit generated bundle compiles and runs
  on real `ssh rvv`, or repairing the minimal bounded script/bridge blocker if
  the current generated bundle path cannot execute this route.

## Requirements

- Start from
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.
- Keep authority in the explicit typed `tcrv_rvv` body and RVV provider facts:
  compare mask provenance, strided load address/stride roles, signed
  i16mf2-to-i32m1 widening product/reduction relation, inactive-lane zeroing,
  accumulator seed/result layout, runtime ABI order, policy/VL facts, required
  headers, type mappings, and provider mirror fields.
- Produce and preserve explicit selected-body generated-bundle dry-run evidence
  for ABI order
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`.
- Execute the generated bundle on real `ssh rvv` if reachable.
- Validate scalar-oracle behavior for counts `0`, `1`, VL-boundary, tail, and
  multi-chunk cases, represented by `0,1,16,17,257` or an equivalent focused
  set.
- Validate both existing stride pairs `2:3` and `3:2`, both mask/input
  patterns, signed lanes, accumulator seed/result preservation, source
  preservation, inactive-lane skipped/zeroed behavior, and output tail
  preservation.
- If execution fails because the harness or ABI bridge is incomplete, repair
  only the minimal production/script blocker for this route and keep
  provider-owned semantics intact.
- Keep the existing explicit dry-run regression passing.
- Run script self-test if `scripts/rvv_generated_bundle_abi_e2e.py` changes.
- Record the exact external blocker if remote hardware/toolchain execution is
  unavailable, and do not claim executable correctness without real `ssh rvv`
  output.

## Acceptance Criteria

- [x] Explicit selected-body generated-bundle dry-run succeeds and evidence JSON
      identifies the explicit input mode, explicit fixture, selected variant,
      function prototype, and generated harness for this route.
- [x] The dry-run evidence preserves provider-derived route facts:
      runtime ABI order, route operand binding, contraction route-family plan,
      compare-produced mask provenance, strided source memory facts, inactive
      lane zeroing, widening product/reduction relation, required headers, type
      mapping, provider mirror, and no descriptor/source-front-door authority.
- [x] Real `ssh rvv` generated-bundle compile/run succeeds, or the exact
      external blocker is recorded without making a runtime correctness claim.
- [x] Runtime evidence, when reachable, covers counts `0,1,16,17,257`, stride
      pairs `2:3` and `3:2`, both mask/input patterns, signed lane products,
      accumulator seed/result preservation, source preservation, skipped
      inactive lanes, scalar output, and output tail preservation.
- [x] Existing explicit target/header fixture and stale-mirror negatives remain
      passing.
- [x] Existing explicit generated-bundle dry-run remains passing after any
      script or production change.
- [x] Script self-test passes if the script changes.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python changes, if any, are limited to the generated-bundle support harness
  or execution bridge.
- No new route-family expansion, dequant/clamp work, broad
  reduction/matmul/dtype/LMUL matrix, frontend path, performance/autotuning,
  source-front-door positive route, direct pre-realized route-entry shortcut,
  or common EmitC invention of mask/stride/dot/reduction semantics is
  introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or an external execution blocker is proven.

## Technical Approach

Use the existing dry-run path as the baseline. First confirm the explicit
generated bundle still emits the expected harness and provider facts. Then run
the same route without `--dry-run` so `scripts/rvv_generated_bundle_abi_e2e.py`
generates, transfers, compiles, and executes the bundle on `ssh rvv`. If the
hardware path exposes a compile/runtime mismatch, patch only the route-specific
script or generated bundle bridge needed to align the exported ABI and harness
with the provider-derived facts already validated by the explicit fixture.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-explicit-fused-dequant-clamp-executable-abi-closure/prd.md`.
- Initial files inspected:
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Evidence

- No production compiler or script code change was required. The existing
  explicit generated-bundle path for
  `computed_masked_strided_input_widening_dot_reduce_add` was already executable
  once invoked with the current local build tools.
- Explicit generated-bundle dry-run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-masked-strided-input-widening-dot-reduce-add`.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-masked-strided-input-widening-dot-reduce-add-ssh`.
- Remote compile evidence:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote PASS marker:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2`.
- Remote case output covered both route cases already represented by the
  harness:
  `source_strides=2,3 mask_pattern=0 input_pattern=0` and
  `source_strides=3,2 mask_pattern=1 input_pattern=1`, each for counts
  `0,1,16,17,257`.
- Runtime evidence confirmed scalar oracle labels:
  `compare_masked_strided_signed_horizontal_dot`,
  `seed_added`,
  `inactive_lanes_skipped`,
  `skipped_source_elements_ignored`,
  `scalar_output`,
  `tail_preserved`.
- Provider/target facts preserved in dry-run and `ssh` evidence include:
  runtime ABI order
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`,
  route operand binding plan
  `rvv-route-operand-binding:masked_strided_wdot.v1`,
  contraction route-family plan
  `rvv-contraction-route-family-plan.v1`,
  typed compute op
  `tcrv_rvv.masked_widening_dot_reduce`,
  memory form
  `computed-mask-strided-input-widening-dot-reduce`,
  mask source
  `compare-produced-mask-same-vl-scope`,
  inactive lane zeroing
  `masked-widening-products-zero-inactive-lanes-before-reduction`,
  source memory form `strided-load`,
  strided load intrinsic `__riscv_vlse16_v_i16mf2`,
  masked widening product intrinsic `__riscv_vwmul_vv_i32m1_m`,
  and reduction intrinsic `__riscv_vredsum_vs_i32m1_i32m1`.
- Checks passed:
  explicit generated-bundle dry-run;
  real `ssh rvv` generated-bundle compile/run;
  direct `FileCheck` for the dry-run test's `ROOT`, `MSDOT`, and `HARNESS`
  prefixes;
  focused explicit fixture `PLAN` and `HEADER` FileCheck commands;
  focused stale-mirror negative checks for provider mirror, operand binding,
  runtime ABI order, required headers, type mapping, contraction plan, widening
  relation, reduction store VL, strided load intrinsic, and masked widening
  product intrinsic;
  adjacent pre-realized generated-bundle dry-run regression;
  bounded old-authority scan over this task's added diff lines;
  `git diff --check`;
  `git diff --cached --check`;
  Trellis context validation.
- Local lit `not` helper was not available as a standalone command during
  direct negative-fixture replay. The stale negative checks were rerun with an
  equivalent shell status guard that requires `tcrv-translate` to fail before
  piping diagnostics to `FileCheck`.
- Script self-test was not required because `scripts/rvv_generated_bundle_abi_e2e.py`
  was not changed in this round.

## Spec Update Judgment

No `.trellis/spec/` update is required for this round. The work did not add or
change a command signature, route-provider API, target artifact contract,
validation error matrix, generated-bundle harness contract, or cross-layer
payload shape. It executed and recorded evidence for the already-specified
computed-mask strided-input widening dot-reduce route.

## Out Of Scope

- No new route-family expansion beyond
  `computed_masked_strided_input_widening_dot_reduce_add`.
- No dequant/clamp extension in this task.
- No high-level Linalg/Vector/StableHLO frontend path.
- No per-Linalg route authority or one-op-per-intrinsic wrapping.
- No broad reduction, matmul, dtype, LMUL, or performance matrix.
- No source-front-door/source-artifact positive route.
- No direct pre-realized route-entry shortcut.
- No report/status-only completion.
