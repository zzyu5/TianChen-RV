# Stage2 RVV explicit computed-masked indexed gather-load executable ABI closure

## Goal

Close executable ABI evidence for the explicit selected-body
`computed_masked_indexed_gather_load_unit_store` path. The task starts from
the existing selected `tcrv.exec` RVV variant with an explicit typed
`tcrv_rvv` body carrying compare-produced mask, runtime index vector, masked
indexed source load, old-destination passthrough for inactive lanes,
unit-stride destination store, runtime `n`/VL, policy facts, and
provider-derived indexed-memory route facts; carries it through
`TCRVEmitCLowerableRoute`, target artifact/generated bundle, and real
`ssh rvv` correctness evidence when the remote/toolchain is reachable.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV explicit computed-masked indexed gather-load executable ABI closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `73c92709 chore(task): archive explicit indexed scatter abi closure`.
- No active Trellis task existed at the start of this round, so this task was
  created from the Hermes brief before source edits.
- Commit `73c92709` archived the complementary explicit
  `computed_masked_indexed_scatter_store_unit_load` executable ABI closure.
  That task proved generated-bundle execution on real `ssh rvv` for counts
  `0,1,16,17,257`, patterns `0,1`, active/inactive compare-produced mask
  lanes, noncontiguous indexed destinations, inactive destination
  preservation, source preservation, and tail preservation.
- Existing target fixtures already expose the explicit and adjacent
  pre-realized computed-masked indexed gather selected bodies:
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`.
- Existing dry-run test
  `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-gather-load-dry-run.test`
  checks provider/target facts, explicit input mode, ABI order, mask/index
  provenance, noncontiguous indexed source behavior, source preservation,
  inactive old-destination passthrough preservation, output tail preservation,
  and harness oracle structure. The current test uses counts `7,16,23`; this
  task must exercise `0,1,16,17,257` or equivalent boundary/tail/multi-chunk
  coverage for closure evidence.
- The script already has explicit and pre-realized expectations, harness
  generation, evidence JSON contracts, and self-test guards for
  `computed_masked_indexed_gather_load_unit_store`; the immediate gap is
  proving generated-bundle execution on real `ssh rvv`, or repairing a minimal
  bounded script/bridge blocker if execution exposes one.

## Requirements

- Start from
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir`.
- Keep authority in the explicit typed `tcrv_rvv` body and RVV provider facts:
  compare mask provenance, runtime index source, index EEW/offset, masked
  indexed source load, old-destination passthrough, unit-stride destination
  store, source/result element type, runtime ABI order, policy/VL facts,
  required headers, type mappings, intrinsic leaves, inactive-lane behavior,
  and provider mirror fields.
- Produce and preserve explicit selected-body generated-bundle dry-run evidence
  for ABI order `cmp_lhs,cmp_rhs,src,index,dst,n`.
- Execute the generated bundle on real `ssh rvv` if reachable.
- Validate scalar-oracle behavior for counts `0`, `1`, VL-boundary, tail, and
  multi-chunk cases, represented by `0,1,16,17,257` or an equivalent focused
  set.
- Validate both mask-active and mask-inactive lanes, active
  `dst[i] = src[index[i]]`, inactive `dst[i] = old_dst[i]` passthrough,
  noncontiguous indexed source reads, source preservation, and output tail
  preservation.
- Preserve provider/target facts and stale-mirror negatives for route id,
  provider mirror, operand binding, ABI order, required headers, type mapping,
  mask provenance, index source/EEW/offset, indexed source memory form,
  old-destination passthrough/unit-store behavior, inactive-lane contract,
  masked indexed load leaf, ordinary unit-store leaf, and adjacent
  pre-realized route behavior if shared code is touched.
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
      order, route operand binding, computed-mask indexed memory route-family
      plan, compare-produced mask provenance, runtime index source, masked
      indexed source load, old-destination passthrough, unit-stride destination
      store, inactive-lane preservation, required headers, type mapping,
      provider mirror, and no descriptor/source-front-door authority.
- [x] Real `ssh rvv` generated-bundle compile/run succeeds, or the exact
      external blocker is recorded without making a runtime correctness claim.
- [x] Runtime evidence, when reachable, covers counts `0,1,16,17,257` or an
      equivalent VL-boundary/tail/multi-chunk set, active and inactive mask
      lanes, noncontiguous indexed source cases, source preservation, inactive
      old-destination passthrough preservation, and output tail preservation.
- [x] Existing explicit target/header fixture and stale-mirror negatives remain
      passing.
- [x] Existing explicit generated-bundle dry-run remains passing after any
      script or production change.
- [x] Adjacent pre-realized generated-bundle dry-run regression passes if
      shared indexed-memory/script code is touched.
- [x] Script self-test passes if the script changes.
- [x] Bounded old-authority scan over touched files and added diff lines passes.
- [x] `git diff --check`, `git diff --cached --check`, and Trellis context
      validation pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/TableGen/CMake/lit/FileCheck.
- Python changes, if any, are limited to generated-bundle support harness or
  execution bridge tooling.
- No route-family expansion beyond
  `computed_masked_indexed_gather_load_unit_store`.
- No indexed scatter follow-up, broad indexed gather/scatter matrix, segment2
  batch, dtype/LMUL clone batch, high-level Linalg/Vector/StableHLO frontend,
  performance/autotuning, source-front-door positive route, direct pre-realized
  route-entry shortcut, common EmitC invention of indexed gather semantics, or
  report/status-only completion is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when the
  acceptance criteria are met or an external execution blocker is proven.

## Technical Approach

Use the existing explicit generated-bundle dry-run as the baseline. First
confirm the generated bundle still emits the expected harness and
provider/target facts. Then run the same route without `--dry-run` so
`scripts/rvv_generated_bundle_abi_e2e.py` generates, transfers, compiles, and
executes the bundle on `ssh rvv`. If the hardware path exposes a compile or
runtime mismatch, patch only the route-specific script or generated bundle
bridge needed to align the exported ABI and harness with the provider-derived
facts already validated by the explicit fixture.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Indexed-gather-specific spec sections read:
  RVV computed-mask indexed memory fact surface,
  computed-mask indexed route validation contract,
  base-memory runtime AVL/VL boundary notes,
  and computed-mask memory statement-plan owner boundary.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-explicit-computed-masked-indexed-scatter-store-abi-closure/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-23.md`, Session 475.
- Initial files inspected:
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-gather-load.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-gather-load-dry-run.test`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `lib/Plugin/RVV/EmitC/`,
  and `lib/Target/RVV/` search hits relevant to computed-mask indexed gather
  route facts.

## Completion Evidence

- No production compiler, script, fixture, or test code change was required.
  The existing explicit generated-bundle path for
  `computed_masked_indexed_gather_load_unit_store` was already executable once
  invoked with the current local build tools and the explicit local
  `/usr/bin/llvm-readobj-20` path.
- Local build refresh passed:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Explicit generated-bundle dry-run with closure counts passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-indexed-gather-load-closure-dry-run`.
- Direct FileCheck of the existing explicit dry-run test passed for `STDOUT`,
  `ROOT`, `CMIDX`, and `HARNESS` using:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-indexed-gather-load-filecheck`.
- Adjacent pre-realized generated-bundle dry-run regression passed for
  `STDOUT`, `ROOT`, `CMIDX`, and `HARNESS` using:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/pre-realized-computed-mask-indexed-gather-load-filecheck`.
- Real `ssh rvv` generated-bundle compile/run passed:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-indexed-gather-load-closure-ssh`.
- Generated bundle path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/explicit-computed-mask-indexed-gather-load-closure-ssh/computed-mask-indexed-gather-load/computed_masked_indexed_gather_load_unit_store/generated_bundle`.
- Remote compile evidence:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote PASS marker:
  `PASS op=computed_masked_indexed_gather_load_unit_store counts=0,1,16,17,257 patterns=0,1`.
- Remote case output covered counts `0,1,16,17,257` for both patterns `0` and
  `1`. Multi-lane cases covered active and inactive mask lanes, including
  `n=16`, `n=17`, and `n=257`.
- Runtime evidence confirmed:
  active lanes load `src[index[i]]` into `dst[i]`,
  inactive lanes preserve the old destination passthrough value,
  noncontiguous indexed source reads are exercised,
  source buffers are preserved,
  and destination tail sentinels are preserved.
- Provider/target facts preserved in dry-run and `ssh` evidence include:
  runtime ABI order `cmp_lhs,cmp_rhs,src,index,dst,n`,
  route operand binding plan
  `rvv-route-operand-binding:computed_masked_indexed_gather_load_unit_store.v1`,
  computed-mask indexed memory route-family plan
  `rvv-computed-mask-memory-route-family-plan.v1`,
  typed compute op `tcrv_rvv.masked_indexed_load`,
  memory form `computed-mask-indexed-gather-load-unit-store`,
  mask producer source `vector-compare-rhs-load`,
  mask source `compare-produced-mask-same-vl-scope`,
  inactive lane contract `masked-off-lanes-preserve-old-destination`,
  source memory form `masked-indexed-load`,
  destination memory form `unit-stride-store`,
  indexed memory layout
  `unit-stride-compare-indexed-masked-source-old-destination-runtime-abi`,
  index source `runtime_abi:index`,
  index EEW `32`,
  offset unit `element`,
  indexed data memory form `masked-indexed-load`,
  required headers `stddef.h,stdint.h,riscv_vector.h`,
  C type mapping
  `vl:size_t,compare/source/passthrough:signed-e32m1,index:u32m1,mask:b32,result:masked-indexed-load-store`,
  target leaf profile
  `rvv-v1-e32m1-computed-mask-indexed-gather-load-leaf-profile.v1`,
  and provider mirror
  `provider_supported_mirror:rvv-computed-mask-indexed-gather-load-plan-validated`.
- Checks passed:
  explicit generated-bundle dry-run with closure counts;
  direct FileCheck for the explicit dry-run test `STDOUT`, `ROOT`, `CMIDX`,
  and `HARNESS`;
  real `ssh rvv` generated-bundle compile/run;
  focused explicit fixture `PLAN` and `HEADER` FileCheck commands;
  focused pre-realized fixture `REALIZED`, `PLAN`, and `HEADER` FileCheck
  commands;
  adjacent pre-realized generated-bundle dry-run regression;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  bounded old-authority scan over production/source diff lines;
  `git diff --check`;
  `git diff --cached --check`;
  and Trellis context validation.
- Script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed in this round.
- The bounded old-authority production/source diff scan was empty because this
  round did not change compiler, script, fixture, or test production files.
  The new PRD contains only negative/out-of-scope mentions of descriptor and
  source-front-door authority.

## Spec Update Judgment

No `.trellis/spec/` update is required for this round. The work did not add or
change a command signature, route-provider API, target artifact contract,
validation error matrix, generated-bundle harness contract, or cross-layer
payload shape. It executed and recorded evidence for the already-specified
computed-mask indexed gather route.

## Out Of Scope

- No indexed scatter follow-up.
- No broad indexed gather/scatter route matrix.
- No segment2 batch.
- No dtype/LMUL clone batch.
- No high-level Linalg/Vector/StableHLO frontend path.
- No per-Linalg route authority or one-op-per-intrinsic wrapping.
- No performance/autotuning.
- No source-front-door positive route or direct pre-realized route-entry
  shortcut.
