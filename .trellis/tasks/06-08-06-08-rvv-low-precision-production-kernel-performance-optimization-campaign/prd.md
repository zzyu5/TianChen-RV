# RVV low-precision production-kernel performance optimization campaign

## Goal

Improve the production compiler path for the bounded low-precision RVV product
reduction kernels after the previous campaign made same-target measurement
trustworthy. The campaign owner is the RVV plugin/compiler path from typed
low-precision `tcrv_rvv` selected bodies through Gearbox/resource-aware
selected-body realization, RVV-owned route and statement planning, target
artifact export, and same-target measurement.

This is a macro task and remains active across rounds until the campaign gates
are genuinely complete.

## Direction Brief Source

Hermes selected:

```text
RVV low-precision production-kernel performance optimization campaign
```

The prior archived macro task closed the resource-aware low-precision
production-kernel campaign through same-target Gate 4 measurement. That evidence
showed correctness and trustworthy timing, but also showed the generated RVV
path is materially slower than the named scalar C baseline for the largest
measured shapes. This task must therefore change production RVV compiler owners
or fail-close a precise production blocker; it must not become another
generated-bundle, measurement-dashboard, metadata, or report-only task.

## What I Already Know

- The repository had no `.trellis/.current-task` at session start, so this new
  macro task owns the current Direction Brief.
- The relevant architecture chain is:
  `tcrv.exec` envelope -> selected typed `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact -> `ssh rvv` evidence for runtime,
  correctness, or performance claims.
- The previous campaign archive
  `.trellis/tasks/archive/2026-06/06-08-rvv-production-kernel-capability-campaign/`
  completed Gates 1-4 for capability and measurement, but did not claim
  performance parity or a win.
- Gate 4 evidence under
  `artifacts/tmp/gate4-same-target-measurement/gate4-wpr-dequant-and-clamp-rvv/`
  shows both bounded kernels pass same-target correctness and timing on `ssh
  rvv`.
- For `n=65536`, the raw evidence shows generated
  `widening_product_reduce_dequantize_f32` around `152 us` per iteration versus
  the scalar baseline around `34 us`; generated
  `widening_product_reduce_dequant_clamp_f32` is around `243 us` versus the
  scalar baseline around `54 us`. This is a regression signal, not a parity
  claim.
- The generated C/intrinsic path currently uses signed `i8mf4 -> i16mf2 ->
  i32m1` product-reduction facts and a `u1` direct-contraction resource
  candidate. It emits a loop with `__riscv_vle8_v_i8mf4`,
  `__riscv_vwmul_vv_i16mf2`, `__riscv_vwredsum_vs_i16mf2_i32m1`, and
  scalar carry through `dot_acc_scalar`, followed by a VL=1 dequant or
  dequant-clamp epilogue.
- The likely production bottleneck is in the RVV low-precision resource
  candidate / selected-body realization / direct-contraction statement plan:
  the current `i8mf4/i16mf2/u1` route uses a very small active element count per
  runtime VL chunk and repeats reduction setup for many chunks. This is a
  production compiler owner, not a measurement-script owner.
- `RVVGearboxSchedule.h` already carries resource facts such as
  `unroll_factor`, `accumulator_count`, `vsetvl_region_count`, and
  `peak_live_vector_groups`, but current source treats the bounded product
  reduction path as static `u1`/single-accumulator facts. Changing those mirrors
  alone is not enough unless the production route consumes them into generated
  code or fails closed.

## Campaign Gates

- [x] Gate 1: bottleneck attribution from Gate 4 raw evidence to a named
      production RVV plugin/compiler owner, including generated artifact shape
      and emitted C/intrinsic evidence.
- [x] Gate 2: at least one production-source optimization or fail-closed
      production guard for low-precision widening
      product/reduction/dequant/dequant-clamp realization or route planning.
- [~] Gate 3: runtime correctness for optimized generated artifacts on
      `ssh rvv`, when this task claims executable optimized artifacts. Current
      slice evidence passed for both bounded kernels; the macro gate remains
      open for later production-kernel optimizations.
- [~] Gate 4: same-target measurement against the same named scalar baseline,
      with raw timing evidence and honest improvement/regression reporting.
      Current slice evidence was collected and does not support a performance
      win claim; the macro gate remains open for further optimization and final
      campaign measurement.
- [ ] Gate 5: cleanup of temporary measurement-only scaffolding so production
      path remains plugin-owned and common EmitC/export stays neutral.

## Current Round Milestone

Complete Gate 1 and one bounded Gate 2 slice.

The current slice is:

- Use the previous Gate 4 evidence for
  `widening_product_reduce_dequantize_f32` and
  `widening_product_reduce_dequant_clamp_f32` to attribute the slowdown to a
  named production owner.
- Inspect generated C/intrinsic output, selected-body realization facts,
  resource candidate facts, route planning, direct-contraction statement
  planning, and target validation mirrors.
- Land one coherent production-source change in the RVV owner. A valid Gate 2
  slice may either improve code generation for the bounded kernels or fail-close
  the exact missing production primitive/candidate so the compiler no longer
  treats the slow resource shape as a completed performance-ready route.
- Validate the changed behavior with focused build/tests. If the source change
  produces an executable optimized artifact, collect correctness and
  same-target measurement evidence; if the source change only exposes the
  production blocker, leave Gate 3/Gate 4 open with an exact next slice.

## Requirements

- Preserve plugin ownership: RVV provider/selected-body realization derives
  dtype, SEW, LMUL, policy, memory form, operation kind, resource decisions,
  intrinsic spelling, and statement plans from typed body/config/capability/
  runtime facts.
- Keep Common EmitC/export neutral. Common materialization may carry provider
  payloads, but must not infer RVV semantics, dtype, LMUL, schedule, or
  intrinsic choices.
- Do not add q8/q4 route authority, llama.cpp wrappers, source-front-door
  positive routes, descriptor-driven computation, high-level Linalg/Vector/
  StableHLO frontend work, per-Linalg route authority, or dtype-prefixed helper
  op families.
- Do not count script polish, reports, metadata-only changes, or a measurement
  rerun as the main Gate 2 achievement.
- Do not claim a performance win unless same-target raw evidence supports it.
- Keep this macro task active after the current slice unless all campaign gates
  are complete.

## Acceptance Criteria For This Round

- [x] PRD names the new macro performance-optimization campaign, records the
      Gate 4 evidence-driven bottleneck attribution, and keeps macro gates open.
- [x] Gate 1 records the production owner and evidence: generated C/intrinsic
      shape, resource candidate facts, and same-target timing contrast.
- [x] Production RVV compiler source changes in selected-body realization,
      route planning, statement planning, or validation; no measurement-only
      closeout.
- [x] Focused checks cover the changed production behavior and fail-closed
      boundary.
- [x] Required focused builds/checks are run or an exact blocker is recorded:
      `tcrv-opt`, `tcrv-translate`,
      `tianchenrv-rvv-extension-plugin-test`,
      `tianchenrv-target-artifact-export-test`, relevant low-precision/
      Gearbox/generated-bundle lit tests, `git diff --check`, and
      `git diff --cached --check`.
- [x] If optimized artifacts are produced, collect correctness guard and
      before/after same-target measurement on `ssh rvv`; otherwise record why
      measurement remains pending and name the next production primitive.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor, source-front-door, route-id,
      artifact-name, or q8/q4 authority.
- [x] Commit one coherent slice and leave `.trellis/.current-task` active if
      campaign gates remain open.

## Out Of Scope

- No new adjacent generated-bundle evidence task.
- No q8/q4-named route authority or llama.cpp wrapper owner.
- No broad smoke matrix, measurement dashboard, index, or report-only work.
- No high-level frontend work or per-Linalg route authority.
- No common EmitC invention of RVV semantics.
- No dtype/LMUL clone batch that is not tied to provider-owned typed facts and
  resource validation.
- No performance-win claim without same-target raw evidence.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Archived campaign read:
  `.trellis/tasks/archive/2026-06/06-08-rvv-production-kernel-capability-campaign/prd.md`.
- Evidence read:
  `artifacts/tmp/gate4-same-target-measurement/gate4-wpr-dequant-and-clamp-rvv/evidence.json`,
  per-op `same_target_measurement_evidence.json`,
  per-op `remote_measure_run_stdout.txt`, and per-op
  `materialized_rvv_emitc.cpp`.
- Current likely production owners:
  `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.

## Current Round Result

Gate 1 attribution is complete for this slice. The slow generated path is owned
by RVV direct-contraction route/statement planning and its validation mirrors,
not by the measurement script. The previous generated artifact shape performs
the product/reduction loop with `__riscv_vle8_v_i8mf4`,
`__riscv_vwmul_vv_i16mf2`, `__riscv_vwredsum_vs_i16mf2_i32m1`, and scalar
carry through `dot_acc_scalar`, then spent the VL=1 epilogue on an avoidable
vector i32 splat, vector int-to-float convert, and vector scale before the final
store or clamp/store.

Gate 2 has one bounded production-source optimization. The RVV
direct-contraction statement plan now computes the dequantized scalar result as
`dot_acc_scalar * scale` in the common EmitC expression materializer, then uses
the RVV-owned f32 scalar splat intrinsic for the final VL=1 result vector. Route
family facts, control-policy validation, route planning, and target artifact
validation now agree on that f32 scalar-splat epilogue. Common EmitC remains
neutral: it only materializes provider-supplied scalar product expressions and
does not infer RVV dtype, LMUL, intrinsic, or schedule decisions.

The generated post-loop structure is reduced:

- `widening_product_reduce_dequantize_f32`: old vector seed/convert/scale/store
  epilogue becomes scalar multiply, f32 splat, store.
- `widening_product_reduce_dequant_clamp_f32`: old vector seed/convert/scale
  plus clamp becomes scalar multiply, f32 splat, clamp, store.

`ssh rvv` correctness and same-target measurement were collected under
`artifacts/tmp/gate1-gate2-performance-optimization/scalar-dequant-epilogue-after/`.
All correctness guard lines passed for both bounded kernels. Same-target timing
does not support a performance-win claim. With the current after run at
`n=65536`, best generated timing was about `208.7 us` versus scalar baseline
about `46.9 us` for `widening_product_reduce_dequantize_f32`, and about
`264.5 us` versus scalar baseline about `59.0 us` for
`widening_product_reduce_dequant_clamp_f32`. Because the after run used a fresh
measurement invocation, compare ratios rather than raw absolute time: the
largest-shape generated path remains roughly `4.45x` slower for dequantize and
`4.35x` slower for clamp. The next bottleneck is therefore still the main
loop/reduction resource shape rather than the VL=1 dequant epilogue.

Self-repair note: an intermediate scalar `fmaf` expression produced optimized C
that failed remote linking on `ssh rvv` with an undefined `fmaf` reference in
the current measurement compile path. The slice was repaired to ordinary scalar
multiply before the f32 splat, avoiding a new libm dependency.

## Continuation Point

Continue the open macro task in the RVV production compiler path. The next
slice should target the main low-precision product/reduction loop: resource-aware
selected-body realization and statement planning for repeated VL/control,
accumulator/reduction structure, and live vector group tradeoffs. Do not create
a neighboring generated-bundle evidence task; use same-target measurement only
to validate production-source changes and report regressions honestly.
