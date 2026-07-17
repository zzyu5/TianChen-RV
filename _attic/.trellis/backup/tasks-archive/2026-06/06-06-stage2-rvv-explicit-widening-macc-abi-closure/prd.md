# Stage2 RVV explicit widening-MAcc executable ABI closure

## Goal

Close executable ABI evidence for the explicit selected-body
`widening_macc_add` route. The task starts from the existing selected
`tcrv.exec` RVV variant with an explicit typed `tcrv_rvv.widening_macc` body
carrying i16 lhs/rhs sources, i32 accumulator seed vector, i32 result vector,
runtime `n`/VL, policy facts, and provider-derived contraction route facts;
carries it through `TCRVEmitCLowerableRoute`, target artifact/generated bundle,
and real `ssh rvv` correctness evidence when the remote/toolchain is reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV explicit widening-MAcc executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `ab3c8243 chore(task): archive explicit indexed gather abi closure`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- Commit `ab3c8243` archived the explicit
  `computed_masked_indexed_gather_load_unit_store` executable ABI closure.
  That prior task proved real `ssh rvv` generated-bundle execution for counts
  `0,1,16,17,257`, patterns `0,1`, active/inactive mask lanes,
  noncontiguous indexed source reads, inactive old-destination passthrough,
  source preservation, and tail preservation.
- Existing explicit target/header fixture
  `test/Target/RVV/explicit-selected-body-artifact-widening-macc-add.mlir`
  already exposes the selected RVV variant, explicit typed
  `tcrv_rvv.widening_macc` body, `lhs,rhs,acc,out,n` runtime ABI order,
  `i16mf2` source typing, `i32m1` accumulator/result typing, widening relation,
  provider mirror, route operand binding summary, target leaf profile,
  required headers, and exported C prototype.
- Existing dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-macc-add-dry-run.test`
  already checks counts `0,1,7,16,23,257`, patterns `0,1`, provider/target
  route facts, stale non-widening-MAcc fact rejection, scalar oracle harness
  structure, signed widening products, source preservation, accumulator
  preservation, and output tail preservation.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has explicit and
  pre-realized expectations for `widening_macc_add`, provider fact constants,
  metadata checks, harness generation, direct pre-realized fail-closed checks,
  dry-run validation, and real `ssh rvv` execution plumbing.

## Requirements

- Start from
  `test/Target/RVV/explicit-selected-body-artifact-widening-macc-add.mlir`.
- Keep authority in the explicit typed `tcrv_rvv` body and RVV provider facts:
  i16 lhs/rhs source vectors, i32 accumulator seed vector, i32 result vector,
  `tcrv_rvv.widening_macc`, runtime ABI order `lhs,rhs,acc,out,n`,
  runtime `n`/VL, tail/mask policy, widening relation
  `signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`, accumulator/result layouts,
  contraction route-family plan, required headers, type mappings, provider
  mirror fields, and target artifact validation.
- Produce and preserve explicit selected-body generated-bundle dry-run evidence
  for ABI order `lhs,rhs,acc,out,n`.
- Execute the generated bundle on real `ssh rvv` if reachable.
- Validate scalar-oracle behavior for counts `0`, `1`, a VL-boundary case,
  a tail case, and a multi-chunk case, represented by `0,1,7,16,23,257` or an
  equivalent focused set.
- Validate both data patterns already represented by the harness, signed
  widening products, positive and negative product cases where available,
  `out[i] = acc[i] + (int32_t)lhs[i] * (int32_t)rhs[i]`, source preservation,
  accumulator preservation, and output tail preservation.
- Preserve provider/target facts and stale-mirror negatives for ABI order,
  operand binding, contraction plan, type mapping, widening-MAcc relation,
  required headers/intrinsics, selected typed compute op, source/result
  SEW/LMUL, accumulator/result layouts, target leaf profile, and provider
  mirror.
- If execution fails because the harness, runtime-count coverage, generated
  bundle, or ABI bridge is incomplete, repair only the minimal production/script
  blocker for this route while preserving RVV-plugin ownership of semantics and
  fail-closed target validation.
- Run script self-test if `scripts/rvv_generated_bundle_abi_e2e.py` changes.
- Record the exact external blocker if remote hardware/toolchain execution is
  unavailable, and do not claim executable correctness without real `ssh rvv`
  output.

## Acceptance Criteria

- [x] Explicit selected-body generated-bundle dry-run succeeds and evidence JSON
      identifies explicit input mode, explicit fixture, selected variant,
      function prototype, provider route facts, and generated harness for this
      route.
- [x] The dry-run evidence preserves provider-derived facts: runtime ABI order,
      route operand binding plan/summary, contraction route-family plan,
      `tcrv_rvv.widening_macc`, source/result SEW/LMUL, accumulator/result
      layout, widening-MAcc relation, required headers, C type mapping, target
      leaf profile, provider mirror, and no descriptor/source-front-door/direct-C
      authority.
- [x] Real `ssh rvv` generated-bundle compile/run succeeds, or the exact
      external blocker is recorded without making a runtime correctness claim.
- [x] Runtime evidence, when reachable, covers counts `0,1,7,16,23,257` or an
      equivalent zero/single/VL-boundary/tail/multi-chunk set, both data
      patterns, signed widening products, source preservation, accumulator
      preservation, and output tail preservation.
- [x] Existing explicit target/header fixture remains passing.
- [x] Adjacent pre-realized `widening_macc_add` generated-bundle dry-run
      regression passes if shared script/provider/target code is touched.
- [x] Script self-test passes if the script changes.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python changes, if any, are limited to generated-bundle support harness or
  execution bridge tooling.
- No route-family expansion beyond `widening_macc_add`.
- No widening-dot/reduction follow-up, dequant/clamp work, masked/indexed/
  segment2 batch, dtype/LMUL clone batch, high-level Linalg/Vector/StableHLO
  frontend, performance/autotuning, source-front-door positive route, direct
  pre-realized route-entry shortcut, common EmitC invention of widening-MAcc
  semantics, or report/status-only completion is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or an external execution blocker is proven.

## Technical Approach

Use the existing explicit generated-bundle dry-run as the baseline. First
confirm the generated bundle still emits the expected provider facts, target
header, and scalar-oracle harness. Then run the same route without `--dry-run`
so `scripts/rvv_generated_bundle_abi_e2e.py` generates, transfers, compiles,
and executes the bundle on `ssh rvv`. If the hardware path exposes a compile or
runtime mismatch, patch only the route-specific script or production bridge
needed to align the exported ABI and harness with the provider-derived facts
already validated by the explicit fixture.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/implementation-stack/index.md`,
  `.trellis/spec/implementation-stack/compiler-stack-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  and `.trellis/spec/guides/index.md`.
- Widening-MAcc-specific spec sections read:
  RVV plugin authority placement and direct contraction route-provider owner
  boundary; EmitC route provider-owned fact surface, MAcc metadata mirror
  contract, and MAcc route validation contract; testing runtime evidence
  contract for generated bundles.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-explicit-computed-masked-indexed-gather-load-abi-closure/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-23.md`, recent sessions 473, 474, and 476.
- Initial files inspected:
  `test/Target/RVV/explicit-selected-body-artifact-widening-macc-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-macc-add-dry-run.test`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `lib/Plugin/RVV/EmitC/`,
  `lib/Target/RVV/`,
  and adjacent widening-MAcc dry-run/fixture search hits.

## Completion Evidence

- No production compiler, script, fixture, or test code change was required.
  The existing explicit selected-body `widening_macc_add` generated-bundle path
  was already executable once invoked with the current local build tools and the
  explicit local `/usr/bin/llvm-readobj-20` path.
- Local build refresh passed:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Explicit generated-bundle dry-run passed for counts `0,1,7,16,23,257`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-macc-add-closure-dry-run`.
- Direct FileCheck of the existing explicit dry-run evidence passed for
  `ROOT`, `WMACC`, and `HARNESS` using the generated dry-run evidence and
  harness.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-macc-add-closure-ssh`.
- Generated bundle path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-macc-add-closure-ssh/widening_macc_add/generated_bundle`.
- Remote compile evidence:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote PASS marker:
  `PASS op=widening_macc_add counts=0,1,7,16,23,257 patterns=0,1`.
- Remote case output covered counts `0,1,7,16,23,257` for both patterns `0`
  and `1`. Runtime evidence included zero-count loop-skip/tail preservation,
  single element cases, boundary/tail cases, and multi-chunk `n=257`.
- Runtime evidence confirmed signed widening products, positive and negative
  product cases, accumulation as
  `out[i] = acc[i] + (int32_t)lhs[i] * (int32_t)rhs[i]`, source preservation,
  accumulator preservation, and output tail preservation.
- Provider/target facts preserved in dry-run and `ssh` evidence include:
  runtime ABI order `lhs,rhs,acc,out,n`,
  route operand binding plan
  `rvv-route-operand-binding:widening_macc_add.v1`,
  route operand binding summary
  `lhs=lhs-input-buffer:lhs:abi|src-load|wmacc-lhs|src-i16mf2|hdr;rhs=rhs-input-buffer:rhs:abi|src-load|wmacc-rhs|src-i16mf2|hdr;acc=accumulator-input-buffer:acc:abi|acc-load|wmacc-acc|acc-i32m1|hdr;out=output-buffer:out:abi|res-store|res-i32m1|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr`,
  typed compute op `tcrv_rvv.widening_macc`,
  memory form `vector-rhs-load`,
  contraction route-family plan `rvv-contraction-route-family-plan.v1`,
  target leaf profile `rvv-v1-i16mf2-i32m1-contraction-leaf-profile.v1`,
  provider mirror
  `provider_supported_mirror:rvv-contraction-family-plan-validated`,
  required headers `stddef.h,stdint.h,riscv_vector.h`,
  C type mapping `vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32`,
  source SEW/LMUL `16/mf2`,
  accumulator SEW/LMUL `32/m1`,
  result SEW/LMUL `32/m1`,
  widening relation `signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`,
  source load intrinsic `__riscv_vle16_v_i16mf2`,
  accumulator load intrinsic `__riscv_vle32_v_i32m1`,
  widening MAcc intrinsic `__riscv_vwmacc_vv_i32m1`,
  and store intrinsic `__riscv_vse32_v_i32m1`.
- Checks passed:
  explicit generated-bundle dry-run with closure counts;
  direct FileCheck for the explicit dry-run test `ROOT`, `WMACC`, and
  `HARNESS`;
  real `ssh rvv` generated-bundle compile/run;
  focused explicit fixture `PLAN` and `HEADER` FileCheck commands;
  adjacent pre-realized generated-bundle dry-run regression;
  direct FileCheck for the pre-realized dry-run test `ROOT`, `WMACC`, and
  `HARNESS`;
  focused pre-realized fixture `REALIZED`, `PLAN`, and `HEADER` FileCheck
  commands;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  bounded old-authority scan over touched files and added diff lines;
  `git diff --check`;
  `git diff --cached --check`;
  and Trellis context validation.
- Script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed in this round.
- The bounded old-authority production/source diff scan was empty because this
  round did not change compiler, script, fixture, or test production files.
  The new PRD contains only negative/out-of-scope mentions of descriptor,
  direct-C, and source-front-door authority.

## Spec Update Judgment

No `.trellis/spec/` update is required for this round. The work did not add or
change a command signature, route-provider API, target artifact contract,
validation error matrix, generated-bundle harness contract, or cross-layer
payload shape. It executed and recorded evidence for the already-specified
explicit `widening_macc_add` route.

## Out Of Scope

- Widening dot-reduce, widening product reduce, dequantization, clamp, masked
  or strided contraction follow-up routes.
- New dtype/LMUL route families or clone batches.
- High-level frontend lowering from Linalg, Vector, or StableHLO.
- Performance/autotuning, dashboards, readiness state machines, or broad smoke
  matrices.
- Direct pre-realized route-entry support or source-front-door positive routes.
- Common EmitC reconstruction of RVV widening-MAcc semantics.
