# Capability-backed RVV i32m1 policy legality

## Goal

Make the existing RVV first-slice i32m1 vector configuration explicit in the
compiler capability model and consume it in the RVV add/sub/mul proposal,
selected lowering-boundary, and emission-readiness path. The bounded slice is
SEW=32, LMUL=m1, tail agnostic, and mask agnostic for the existing finite
`i32-vadd`, `i32-vsub`, and `i32-vmul` RVV microkernel families.

## Why Now

HEAD `1823310` completed descriptor-driven add/sub/mul dispatch route metadata.
The next blocker is not another arithmetic family; it is making RVV
compile-time vector configuration a capability-backed legality input before
future SEW, LMUL, or policy expansion. Current code already carries typed
`tcrv_rvv.policy`, descriptor-local element counts, vlenb/lane capacity facts,
and required march metadata, but SEW/LMUL/tail/mask support is not yet modeled
as stable required capabilities for selected i32m1 RVV paths.

## Requirements

- Reuse existing stable RVV capability ids for RVV presence, hart count,
  vlenb/lane capacity, toolchain, compile/run, march, and mabi facts.
- Define stable first-slice capability ids and symbols for:
  - SEW 32 support;
  - LMUL m1 support;
  - tail agnostic policy support;
  - mask agnostic policy support.
- Keep these ids plugin-local RVV capability facts, not new `tcrv.exec` compute
  ops and not Python compiler data structures.
- Update the C++ RVV capability profile to construct the new first-slice
  capability providers from validated RVV probe facts.
- Update RVV proposal generation so add/sub/mul RVV proposals require RVV
  presence plus the SEW/LMUL/tail/mask first-slice capability ids through the
  existing `VariantProposal` and `TargetCapabilitySet` interfaces.
- Update RVV materialized-variant legality so hand-authored or stale selected
  RVV i32m1 variants must both require and be backed by the first-slice
  config/policy capabilities.
- Update selected lowering-boundary validation and emission-readiness relevant
  to RVV i32m1 paths so missing, unavailable, malformed, or mismatched
  SEW/LMUL/policy capabilities fail before supported source/object/bundle
  emission can be claimed.
- Preserve the existing separation:
  - SEW/LMUL/tail/mask policy are compile-time capability/config requirements;
  - runtime `n`, AVL, VL, and dispatch guard remain runtime IR/ABI/control
    values;
  - descriptor-local `element_count` remains descriptor-local metadata.
- Extend `scripts/rvv_remote_probe.py` and `scripts/rvv_probe_to_mlir.py` only
  enough to emit/replay bounded first-slice config/policy capability facts from
  existing RVV toolchain evidence. These scripts must not decide legality,
  selection, lowering, emission, runtime ABI, or source generation.
- Preserve generated C ABI, route ids, artifact names, self-check markers, and
  add/sub/mul arithmetic semantics.

## Acceptance Criteria

- [x] Stable RVV first-slice capability ids/symbols exist for SEW 32, LMUL m1,
      tail agnostic policy, and mask agnostic policy.
- [x] RVV probe capability construction emits the new first-slice capabilities
      from bounded RVV compile/run evidence.
- [x] Probe-to-MLIR replay emits the same capabilities as MLIR
      `tcrv.exec.capability` providers and self-tests cover them.
- [x] RVV add/sub/mul proposals require RVV presence and the first-slice
      config/policy capabilities.
- [x] Materialized RVV i32m1 variant legality rejects missing, unavailable,
      malformed, or mismatched SEW/LMUL/tail/mask capability support.
- [x] Selected lowering-boundary or emission-readiness validation rejects a
      selected RVV i32m1 path before artifact output when those required
      capabilities are absent or unavailable.
- [x] Positive add/sub/mul RVV and RVV+scalar dispatch paths still
      materialize/export with the required capabilities present.
- [x] Focused lit/C++ tests cover proposal requirements, selected-path
      validation failures, and positive materialization/export preservation.
- [x] `python3 scripts/rvv_remote_probe.py --self-test` passes if the probe
      script changes.
- [x] `python3 scripts/rvv_probe_to_mlir.py --self-test` passes if replay
      changes.
- [x] `git diff --check` passes.
- [x] CMake configure with the repository LLVM/MLIR paths passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes before finish/archive.
- [x] Trellis validation, finish/archive, workspace journal update, and one
      coherent commit are completed if the module is finished.

## Non-goals

- No new arithmetic family beyond `i32-vadd`, `i32-vsub`, and `i32-vmul`.
- No i64/e64, alternative LMULs, masked arithmetic bodies,
  tail-undisturbed codegen, generic RVV lowering, or vector-dialect lowering.
- No performance benchmarking or ratio claims.
- No new runtime correctness claim unless generated C/runtime behavior changes
  and fresh `ssh rvv` evidence is collected.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branches in generic core passes.
- No Python implementation of compiler registry, plugin proposal, legality,
  lowering, emission, route selection, runtime ABI decisions, or source
  generation.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout.

## Technical Notes

- Repo root: `/home/kingdom/phdworks/TianchenRV`
- Starting HEAD: `1823310 feat(rvv): add descriptor-driven dispatch route manifest`
- Starting worktree: clean
- No matching open Trellis task existed; this task was created for the Hermes
  brief.
- Read specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/capability-model/capability-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Prior task read:
  - `.trellis/tasks/archive/2026-05/05-09-descriptor-driven-i32-binary-dispatch-route-manifest/prd.md`
- Current code finding:
  - `RVVCapabilityProfile` already owns stable ids for `rvv`,
    `rvv.hart_count`, `rvv.vlenb_bytes`, `rvv.i32_m1_lane_count`,
    `rvv.toolchain.clang`, `rvv.toolchain.cmake`,
    `rvv.probe.compile_run`, `rvv.toolchain.march`, and
    `rvv.toolchain.mabi`.
  - `RVVExtensionPlugin` already validates RVV presence, hart count,
    vlenb/lane pairing, selected march, typed `tcrv_rvv.policy`, descriptor,
    element count, and explicit microkernel structured body.
  - The missing slice is stable SEW/LMUL/tail/mask capability ids and their
    use as proposal/legality/selected-path requirements.

## Continuation Rule If Unfinished

Keep this task open. Record exactly which layer is complete and which remains:
capability ids, RVV proposal requirements, selected lowering-boundary
validation, emission-readiness validation, probe/probe-to-MLIR emission,
focused tests, full check, archive, or commit. Do not archive or claim
capability-backed RVV policy legality until at least RVV plugin proposals and
selected-path validation consume the new capability requirements.

## Completion Notes

- Added stable RVV first-slice ids/symbols:
  `rvv.i32_m1.sew32` / `@rvv_i32_m1_sew32`,
  `rvv.i32_m1.lmul_m1` / `@rvv_i32_m1_lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic` / `@rvv_i32_m1_tail_agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic` / `@rvv_i32_m1_mask_agnostic`.
- RVV proposal generation now requires `rvv` plus all four first-slice
  config/policy ids. Materialized selected RVV variants must require providers
  satisfying all five ids, either as exact capability symbols or a structured
  relation provider.
- RVV capability profile, probe, and replay surfaces now preserve bounded
  first-slice SEW/LMUL/tail/mask facts while leaving proposal, legality,
  lowering, emission, and ABI decisions in C++/MLIR.
- Selected RVV lowering-boundary validation re-runs plugin legality for
  descriptor/smoke/required-march/capacity selected paths, so missing,
  disabled, or mismatched first-slice capability evidence fails before source,
  object, or bundle support can be reported.
- Generated C/runtime semantics, ABI shape, route ids, artifact names, and
  add/sub/mul arithmetic were not changed. No new `ssh rvv` evidence was
  collected because this round made no new runtime correctness or performance
  claim.
- Final validation passed:
  `git diff --check`;
  `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`;
  `python3 scripts/rvv_remote_probe.py --self-test`;
  `python3 scripts/rvv_probe_to_mlir.py --self-test`;
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
