# Stage2 RVV explicit computed-masked segment2-update executable ABI closure

## Goal

Close executable ABI evidence for the explicit selected-body
`computed_masked_segment2_update_unit_load` path. The task starts from the
existing selected `tcrv.exec` RVV variant with an explicit typed `tcrv_rvv`
body carrying compare-produced mask, unit-stride field0/field1 payload loads,
add(field0, field1), masked segment2 destination update, source/result
element types, runtime `n`/VL, policy facts, and provider-derived segment2
route facts; carries it through `TCRVEmitCLowerableRoute`, target
artifact/generated bundle, and real `ssh rvv` correctness evidence when the
remote/toolchain is reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV explicit computed-masked segment2-update executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `1debea75 chore(task): archive explicit masked strided dot abi closure`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- The immediately preceding archived task closed explicit
  `computed_masked_strided_input_widening_dot_reduce_add` generated-bundle
  execution on `ssh rvv`; it is prior evidence and command-shape guidance only,
  not authority for this segment2 route.
- Existing target fixtures already expose the explicit and adjacent
  pre-realized computed-masked segment2 update selected bodies:
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-update.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`.
- Existing dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-segment2-update-dry-run.test`
  checks provider/target facts, stale-mirror negatives by fixture, and harness
  source-level oracle structure for counts `0,1,7,16,23,257`.
- The script already has route expectation and harness support for
  `computed_masked_segment2_update_unit_load`; the immediate gap is proving
  generated-bundle execution on real `ssh rvv`, or repairing a minimal bounded
  script/bridge blocker if execution exposes one.

## Requirements

- Start from
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-update.mlir`.
- Keep authority in the explicit typed `tcrv_rvv` body and RVV provider facts:
  compare mask provenance, segment2 grouping/update semantics, field0/field1
  memory roles, source/result element type, runtime ABI order, policy/VL
  facts, required headers, type mappings, segment tuple facts, update
  arithmetic, inactive-lane behavior, and provider mirror fields.
- Produce and preserve explicit selected-body generated-bundle dry-run evidence
  for ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
- Execute the generated bundle on real `ssh rvv` if reachable.
- Validate scalar-oracle behavior for counts `0`, `1`, VL-boundary, tail, and
  multi-chunk cases, represented by `0,1,16,17,257` or an equivalent focused
  set. Existing dry-run coverage may also include `7` and `23`.
- Validate both mask-active and mask-inactive lanes, both segment fields,
  active-lane update `dst.field0 = src0 + src1`, active-lane field1 store,
  inactive-lane preservation of both old destination slots, source
  preservation, and output tail preservation.
- Preserve provider/target facts and stale-mirror negatives for route id,
  provider mirror, operand binding, ABI order, required headers, type mapping,
  mask provenance, inactive-lane contract, field roles, update arithmetic, and
  adjacent pre-realized route behavior if shared code is touched.
- If execution fails because the harness or ABI bridge is incomplete, repair
  only the minimal production/script blocker for this route and keep
  provider-owned semantics intact.
- Run script self-test if `scripts/rvv_generated_bundle_abi_e2e.py` changes.
- Record the exact external blocker if remote hardware/toolchain execution is
  unavailable, and do not claim executable correctness without real `ssh rvv`
  output.

## Acceptance Criteria

- [x] Explicit selected-body generated-bundle dry-run succeeds and evidence JSON
      identifies the explicit input mode, explicit fixture, selected variant,
      function prototype, and generated harness for this route.
- [x] The dry-run evidence preserves provider-derived route facts: runtime ABI
      order, route operand binding, computed-mask memory route-family plan,
      compare-produced mask provenance, segment2 field roles, source/destination
      memory forms, inactive-lane preservation, update arithmetic, required
      headers, type mapping, provider mirror, and no descriptor/source-front-door
      authority.
- [x] Real `ssh rvv` generated-bundle compile/run succeeds, or the exact
      external blocker is recorded without making a runtime correctness claim.
- [x] Runtime evidence, when reachable, covers counts `0,1,16,17,257` or an
      equivalent VL-boundary/tail/multi-chunk set, active and inactive mask
      lanes, segment lane/update cases, source preservation, inactive
      destination preservation, and output tail preservation.
- [x] Existing explicit target/header fixture and stale-mirror negatives remain
      passing.
- [x] Existing explicit generated-bundle dry-run remains passing after any
      script or production change.
- [x] Adjacent pre-realized generated-bundle dry-run regression passes if shared
      segment2/script code is touched.
- [x] Script self-test passes if the script changes.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python changes, if any, are limited to generated-bundle support harness or
  execution bridge tooling.
- No new route-family expansion beyond
  `computed_masked_segment2_update_unit_load`.
- No indexed gather/scatter batch, broad segment2 matrix, dtype/LMUL clone
  batch, high-level Linalg/Vector/StableHLO frontend, performance/autotuning,
  source-front-door positive route, direct pre-realized route-entry shortcut,
  common EmitC invention of segment/update semantics, or report/status-only
  completion is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or an external execution blocker is proven.

## Technical Approach

Use the existing explicit generated-bundle dry-run as the baseline. First
confirm the generated bundle still emits the expected harness and provider
facts. Then run the same route without `--dry-run` so
`scripts/rvv_generated_bundle_abi_e2e.py` generates, transfers, compiles, and
executes the bundle on `ssh rvv`. If the hardware path exposes a compile or
runtime mismatch, patch only the route-specific script or generated bundle
bridge needed to align the exported ABI and harness with the provider-derived
facts already validated by the explicit fixture.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Segment2-specific spec sections read:
  RVV segment2 route-family planning owner boundary,
  RVV segment2 memory statement-plan boundary,
  computed-mask segment2 memory fact surface,
  segment2 target export consumer contract,
  and segment2 memory route validation contract.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-explicit-computed-masked-strided-input-widening-dot-reduce-abi-closure/prd.md`.
- Initial files inspected:
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-update.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-segment2-update-dry-run.test`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Evidence

- No production compiler or script code change was required. The existing
  explicit generated-bundle path for
  `computed_masked_segment2_update_unit_load` was already executable once
  invoked with the current local build tools.
- Local build refresh passed:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Explicit generated-bundle dry-run passed with the runtime evidence count set:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-segment2-update-closure`.
- Direct FileCheck of the existing dry-run test passed for `STDOUT`, `ROOT`,
  `CMSEG`, and `HARNESS` using:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-segment2-update-filecheck`.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-segment2-update-ssh`.
- Remote compile evidence:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote PASS marker:
  `PASS op=computed_masked_segment2_update_unit_load counts=0,1,16,17,257 patterns=0,1`.
- Remote case output covered counts `0,1,16,17,257` for both patterns `0` and
  `1`, with active and inactive mask lanes for multi-lane counts, including
  `n=16`, `n=17`, and `n=257`.
- Runtime evidence confirmed the scalar oracle labels and guards:
  `computed_mask segment2_update`,
  `active_lanes`,
  `inactive_lanes`,
  `inactive_preserved_lanes`,
  `field_distinguishing_lanes`,
  `source_preserved`,
  and `tail_preserved`.
- Provider/target facts preserved in dry-run and `ssh` evidence include:
  runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`,
  route operand binding plan
  `rvv-route-operand-binding:cmseg2_update_unit_load.v1`,
  computed-mask memory route-family plan
  `rvv-computed-mask-memory-route-family-plan.v1`,
  typed compute op `tcrv_rvv.binary`,
  memory form `computed-mask-unit-load-segment2-store`,
  mask source `compare-produced-mask-same-vl-scope`,
  inactive lane contract `masked-store-false-lanes-preserve-output-buffer`,
  segment tuple C type `vint32m1x2_t`,
  segment store intrinsic `__riscv_vsseg2e32_v_i32m1x2_m`,
  tuple create intrinsic `__riscv_vcreate_v_i32m1x2`,
  update arithmetic kind `add`,
  update arithmetic intrinsic `__riscv_vadd_vv_i32m1`,
  field roles `segment-field0-input-buffer` and
  `segment-field1-input-buffer`,
  required headers `stddef.h,stdint.h,riscv_vector.h`,
  and provider mirror
  `provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated`.
- Checks passed:
  explicit generated-bundle dry-run;
  direct FileCheck for `STDOUT`, `ROOT`, `CMSEG`, and `HARNESS`;
  real `ssh rvv` generated-bundle compile/run;
  focused explicit fixture `PLAN` and `HEADER` FileCheck commands;
  focused stale-mirror negative checks for route id, provider mirror, operand
  binding, ABI order, required headers, type mapping, mask source, inactive
  lane contract, field role, and update arithmetic;
  adjacent pre-realized generated-bundle dry-run regression;
  bounded old-authority scan over production/source diff lines;
  `git diff --check`;
  and Trellis context validation.
- Script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed in this round.
- The bounded old-authority production/source diff scan was empty because this
  round did not change compiler, script, fixture, or test production files.

## Spec Update Judgment

No `.trellis/spec/` update is required for this round. The work did not add or
change a command signature, route-provider API, target artifact contract,
validation error matrix, generated-bundle harness contract, or cross-layer
payload shape. It executed and recorded evidence for the already-specified
computed-mask segment2 update route.

## Out Of Scope

- No new route-family expansion beyond
  `computed_masked_segment2_update_unit_load`.
- No indexed gather/scatter batch or broad segment2 route matrix.
- No dtype/LMUL clone batch.
- No high-level Linalg/Vector/StableHLO frontend path.
- No per-Linalg route authority or one-op-per-intrinsic wrapping.
- No performance/autotuning.
- No common EmitC invention of computed-mask, segment2, or update semantics.
- No direct pre-realized route-entry shortcut.
- No source-front-door/source-artifact positive route.
- No report/status-only completion.
