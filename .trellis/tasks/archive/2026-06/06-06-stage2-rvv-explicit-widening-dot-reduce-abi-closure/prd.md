# Stage2 RVV explicit widening-dot reduce executable ABI closure

## Goal

Close executable ABI evidence for the explicit selected-body
`widening_dot_reduce_add` route. The task starts from the existing selected
`tcrv.exec` RVV variant with an explicit typed
`tcrv_rvv.widening_dot_reduce` body carrying i16 lhs/rhs sources, an i32 scalar
seed from `acc[0]`, an i32 scalar result in `out[0]`, runtime `n`/VL, policy
facts, and provider-derived contraction/reduction route facts; carries it
through `TCRVEmitCLowerableRoute`, target artifact/generated bundle, and real
`ssh rvv` correctness evidence when the remote/toolchain is reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV explicit widening-dot reduce executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `7adefcec chore(task): archive explicit widening macc abi closure`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- Commit `7adefcec` archived the explicit `widening_macc_add` executable ABI
  closure. That prior task proved real `ssh rvv` generated-bundle execution for
  counts `0,1,7,16,23,257`, patterns `0,1`, signed widening products,
  accumulator preservation, source preservation, and tail preservation.
- Existing explicit target/header fixture
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`
  already exposes the selected RVV variant, explicit typed
  `tcrv_rvv.widening_dot_reduce` body, `lhs,rhs,acc,out,n` runtime ABI order,
  `i16mf2` source typing, `i32m1` accumulator/result typing, scalar seed/result
  layouts, widening-dot relation, reduction store VL, provider mirror, route
  operand binding summary, target leaf profile, required headers, and exported
  C prototype.
- Existing dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-dot-reduce-add-dry-run.test`
  already checks counts `0,1,16,17,257`, provider/target route facts, stale
  non-dot fact rejection, scalar oracle harness structure, signed widening
  products, seed-added behavior, scalar out[0] contract, and output tail
  preservation.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has explicit and
  pre-realized expectations for `widening_dot_reduce_add`, provider fact
  constants, metadata checks, harness generation, direct pre-realized
  fail-closed checks, dry-run validation, and real `ssh rvv` execution plumbing.
- Auto-context found one route-specific harness blocker: the
  `widening_dot_reduce_add` generated harness allocates `lhs_before`,
  `rhs_before`, and `acc_before`, but currently does not validate those buffers
  after execution and does not free them. Because the Direction Brief requires
  source preservation, this task must repair that minimal script blocker before
  claiming executable closure.

## Requirements

- Start from
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`
  and the existing explicit generated-bundle dry-run path.
- Keep authority in the explicit typed `tcrv_rvv` body and RVV provider facts:
  i16 lhs/rhs sources, i32 scalar accumulator seed from `acc[0]`, i32 scalar
  output in `out[0]`, `tcrv_rvv.widening_dot_reduce`, runtime ABI order
  `lhs,rhs,acc,out,n`, runtime `n`/VL, tail/mask policy, widening-dot relation
  `signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32`, scalar accumulator and
  result layouts, contraction route-family plan, required headers/type
  mappings, provider mirror fields, and target artifact validation.
- Repair only the route-specific generated harness gap needed to prove source
  and accumulator preservation for `widening_dot_reduce_add`.
- Produce and preserve explicit selected-body generated-bundle dry-run evidence
  for ABI order `lhs,rhs,acc,out,n`.
- Execute the generated bundle on real `ssh rvv` if reachable.
- Validate scalar-oracle behavior for counts `0`, `1`, a VL-boundary case, a
  tail case, and a multi-chunk case, represented by `0,1,16,17,257` or an
  equivalent focused set.
- Validate signed widening products, positive and negative products, seed-added
  behavior distinguishing add-only and mul-only mistakes, scalar output where
  only `out[0]` is written, source preservation, accumulator preservation, and
  output tail preservation.
- Preserve provider/target facts and stale-mirror negatives for ABI order,
  operand binding, contraction plan, type mapping, widening-dot relation,
  reduction store VL, required headers/intrinsics, selected typed compute op,
  source/result SEW/LMUL, scalar accumulator/result layouts, target leaf
  profile, and provider mirror.
- If execution fails because the harness, runtime-count coverage, scalar result
  layout, generated bundle, or ABI bridge is incomplete, repair only the
  minimal production/script blocker for this route while preserving RVV-plugin
  ownership of semantics and fail-closed target validation.
- Run script self-test because `scripts/rvv_generated_bundle_abi_e2e.py`
  changes in this round.
- Record the exact external blocker if remote hardware/toolchain execution is
  unavailable, and do not claim executable correctness without real `ssh rvv`
  output.

## Acceptance Criteria

- [x] Explicit selected-body generated-bundle dry-run succeeds and evidence JSON
      identifies explicit input mode, explicit fixture, selected variant,
      function prototype, provider route facts, scalar oracle harness, and
      generated bundle for this route.
- [x] The dry-run evidence preserves provider-derived facts: runtime ABI order,
      route operand binding plan/summary, contraction route-family plan,
      `tcrv_rvv.widening_dot_reduce`, source/result SEW/LMUL,
      accumulator/result layout, widening-dot relation, reduction store VL,
      required headers, C type mapping, target leaf profile, provider mirror,
      and no descriptor/source-front-door/direct-C authority.
- [x] Real `ssh rvv` generated-bundle compile/run succeeds, or the exact
      external blocker is recorded without making a runtime correctness claim.
- [x] Runtime evidence, when reachable, covers counts `0,1,16,17,257` or an
      equivalent zero/single/VL-boundary/tail/multi-chunk set, signed widening
      products, positive and negative products, seed-added behavior,
      add-only/mul-only distinguishing cases, source preservation, accumulator
      preservation, scalar `out[0]` output, and output tail preservation.
- [x] Existing explicit target/header fixture remains passing.
- [x] Adjacent pre-realized `widening_dot_reduce_add` generated-bundle dry-run
      regression passes because the shared generated-bundle script changes.
- [x] Script self-test passes.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python changes are limited to generated-bundle support harness or execution
  bridge tooling.
- No route-family expansion beyond `widening_dot_reduce_add`.
- No strided-input dot batch, computed-mask dot batch, widening-MAcc follow-up,
  dequant/clamp work, masked/indexed/segment2 batch, dtype/LMUL clone batch,
  high-level Linalg/Vector/StableHLO frontend, performance/autotuning,
  source-front-door positive route, direct pre-realized route-entry shortcut,
  common EmitC invention of widening-dot semantics, or report/status-only
  completion is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or an external execution blocker is proven.

## Technical Approach

Use the existing explicit generated-bundle dry-run as the baseline. First patch
the `widening_dot_reduce_add` harness generation so it actually checks and
reports source/accumulator preservation, and update the explicit dry-run
FileCheck contract to require that signal. Then refresh the local build tools,
run the explicit dry-run, direct FileCheck, focused target/header checks,
script self-test, and adjacent pre-realized regression. Finally run the same
route without `--dry-run` so `scripts/rvv_generated_bundle_abi_e2e.py`
generates, transfers, compiles, and executes the bundle on `ssh rvv`.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  and `.trellis/spec/testing/index.md`.
- Widening-dot-specific spec sections read:
  RVV plugin authority placement, selected-body realization owner registry,
  direct contraction route-provider owner boundary, provider-owned widening
  dot-reduce fact surface, widening dot-reduce route validation contract, and
  testing runtime evidence checklist.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-explicit-widening-macc-abi-closure/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-23.md`, recent sessions 473, 474, 476, and
  477.
- Initial files inspected:
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-dot-reduce-add-dry-run.test`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Evidence

- Local build refresh passed:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- `scripts/rvv_generated_bundle_abi_e2e.py` was repaired for
  `widening_dot_reduce_add` only:
  the generated harness now checks `lhs`, `rhs`, and `acc` snapshots across the
  whole allocation, frees the snapshot buffers on all target return paths,
  runs both data patterns `0,1`, and prints preservation in the success marker.
- Explicit generated-bundle dry-run passed for counts `0,1,16,17,257`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-dot-reduce-add-closure-dry-run`.
- Direct FileCheck of the explicit dry-run evidence passed for `ROOT`, `WDOT`,
  and `HARNESS`; `HARNESS` now checks the mutation diagnostic path,
  pattern-loop call, and `source_preserved accumulator_preserved` success
  marker.
- Focused explicit target fixture passed for both `PLAN` and `HEADER` prefixes:
  `test/Target/RVV/explicit-selected-body-artifact-widening-dot-reduce-add.mlir`.
- Adjacent pre-realized generated-bundle dry-run regression passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-widening-dot-reduce-add-regression-dry-run`.
- Direct FileCheck of the adjacent pre-realized dry-run evidence passed for
  `ROOT`, `WDOT`, and `HARNESS`.
- Script self-test passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-dot-reduce-add-closure-ssh`.
- Generated bundle path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-widening-dot-reduce-add-closure-ssh/widening_dot_reduce_add/generated_bundle`.
- Remote compile evidence:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote PASS marker:
  `PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1`.
- Remote case output covered counts `0,1,16,17,257` for both patterns `0` and
  `1`. Runtime evidence included zero-count loop-skip/scalar seed behavior,
  single element cases, VL-boundary `16`, tail `17`, and multi-chunk `257`.
- Runtime evidence confirmed signed horizontal widening dot products,
  seed-added behavior, add-only and mul-only distinguishing cases,
  scalar output in `out[0]`, source preservation, accumulator preservation, and
  output tail preservation.
- Provider/target facts preserved in dry-run and `ssh` evidence include:
  runtime ABI order `lhs,rhs,acc,out,n`,
  route operand binding plan
  `rvv-route-operand-binding:widening_dot_reduce.v1`,
  route operand binding summary
  `lhs=lhs-input-buffer:lhs:abi|ld|dot-lhs|i16|hdr;rhs=rhs-input-buffer:rhs:abi|ld|dot-rhs|i16|hdr;acc=accumulator-input-buffer:acc:abi|seed|red|i32|hdr;out=output-buffer:out:abi|store|i32|hdr;n=runtime-element-count:n:abi|setvl-avl|loop|hdr`,
  typed compute op `tcrv_rvv.widening_dot_reduce`,
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
  accumulator layout `scalar-i32-seed-lane0-from-accumulator-input`,
  result layout `store-dot-reduction-lane0-to-output-scalar`,
  widening relation `signed-i16mf2xi16mf2-reduce-plus-i32-scalar-to-i32`,
  widening product intrinsic `__riscv_vwmul_vv_i32m1`,
  scalar seed splat intrinsic `__riscv_vmv_v_x_i32m1`,
  reduction intrinsic `__riscv_vredsum_vs_i32m1_i32m1`,
  reduction store VL `1`,
  source load intrinsic `__riscv_vle16_v_i16mf2`,
  setvl intrinsic `__riscv_vsetvl_e32m1`,
  and store intrinsic `__riscv_vse32_v_i32m1`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-target-artifact-export-test` and
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Bounded old-authority scan over touched diff lines was empty for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `descriptor`, `source-front-door`, `source-export`, and `direct-C`.
- `git diff --check` passed.
- Trellis context validation passed with five implement and five check entries.
- `python3 -m lit` was attempted for the explicit and pre-realized script
  tests, but local Python does not have `lit` installed:
  `No module named lit`. Equivalent direct RUN commands and FileCheck prefixes
  were executed instead.

## Self-Repair Performed

- First real `ssh rvv` attempt failed during remote clang compile with:
  `too few arguments to function call, expected 2, have 1` at the generated
  `run_case(counts[index])` call.
- Root cause: the `widening_dot_reduce_add` harness had already grown a
  `run_case(size_t n, int pattern)` signature and pattern-specific input
  initializers, but its `main` still called the single-argument form and did
  not report `patterns=0,1`.
- Repair: changed only the `widening_dot_reduce_add` harness generation to
  iterate `patterns[] = {0, 1}`, pass `pattern` into `run_case`, and print
  `patterns=0,1`; synchronized the explicit and pre-realized dry-run
  `HARNESS` FileCheck contracts.

## Spec Update Judgment

`.trellis/spec/testing/mlir-testing-contract.md` was updated. The new contract
records that generated-bundle harnesses which allocate `*_before` snapshots for
input, accumulator, passthrough, or seed buffers must actually compare those
snapshots after execution, fail with a route-specific mutation diagnostic, free
snapshot buffers on every return path, and expose the preservation success
marker in dry-run `HARNESS` FileCheck. This captures the non-obvious bug fixed
in this round and prevents future evidence-only preservation drift.

## Out Of Scope

- Strided-input dot, computed-mask dot, and computed-mask strided dot follow-up
  routes.
- Widening MAcc, widening product reduce, dequantization, clamp, masked,
  indexed, or segment2 follow-up routes.
- New dtype/LMUL route families or clone batches.
- High-level frontend lowering from Linalg, Vector, or StableHLO.
- Performance/autotuning, dashboards, readiness state machines, or broad smoke
  matrices.
- Direct pre-realized route-entry support or source-front-door positive routes.
- Common EmitC reconstruction of RVV widening-dot semantics.
