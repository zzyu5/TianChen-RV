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
- [x] Gate 5: cleanup of temporary measurement-only scaffolding so production
      path remains plugin-owned and common EmitC/export stays neutral.

## Current Round Milestone

Implement the grouped `u2` product/reduction statement-plan payload, or keep it
fail-closed with a narrower provider-owned reason if repository evidence proves
the payload is still unsafe.

The previous grouped-candidate slice made the next main-loop optimization
boundary explicit and rejected forced grouped `u2` selection because the
statement-plan owner could not yet express a tail-safe grouped main loop. This
round owns that missing payload for
`widening_product_reduce_dequantize_f32`, and should include
`widening_product_reduce_dequant_clamp_f32` if it uses the same production
route-family seam.

The intended executable shape is provider-owned:

- RVV resource selection chooses the grouped `u2` candidate only when the
  typed product/reduction/dequant body, policy, runtime AVL facts, and vector
  register budget all match the supported resource shape.
- The direct-contraction statement-plan owner emits a tail-safe grouped main
  loop plus a scalar-preserving tail loop. The grouped loop must not rely on
  `VL=0` or no-op tail intrinsics and must not perform out-of-bounds second
  slice loads.
- The two-loop payload preserves the existing `dot_acc_vec` accumulator carry,
  final scalar extract, scalar dequant multiply, f32 scalar splat, optional
  clamp, and final store semantics.
- Route-family validation, route description mirrors, target artifact
  validation, generated-bundle parsing, and FileCheck expectations all consume
  grouped `u2` as provider facts; none may infer route support from metadata,
  artifact names, q8/q4 names, or Common EmitC scheduling.
- If grouped `u2` becomes executable, collect `ssh rvv` correctness and raw
  same-target timing against the existing scalar baseline. Report the timing
  honestly without unsupported performance-win claims.
- If a narrower blocker remains, the provider must fail closed before Common
  EmitC or target export with the exact missing payload fact, and no new
  runtime/performance claim is allowed.

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

- [x] PRD/checklist names the grouped `u2` tail-safe statement-plan payload as
      the current unfinished milestone and keeps this macro task active unless
      all campaign gates are genuinely complete.
- [x] RVV low-precision resource selection can choose grouped `u2` for the
      supported product-reduction dequantize/dequant-clamp typed shape, or
      fails closed with a narrower missing payload fact than the previous
      generic pending diagnostic.
- [x] The direct-contraction statement-plan owner emits a provider-owned
      tail-safe grouped main loop and u1 tail loop for grouped `u2`, preserving
      `dot_acc_vec` accumulator/result semantics and avoiding `VL=0` and
      out-of-bounds second-slice loads.
- [x] Route-family validation and route-description validation accept the
      grouped `u2` resource facts only when the statement plan matches the
      supported loop/resource shape, and still reject stale selected-candidate
      or resource mirrors.
- [x] Target artifact validation, generated-bundle ABI parsing, and focused
      FileCheck expectations consume the grouped `u2` selected-candidate facts
      and the two-loop payload shape as mirrors of provider-owned facts.
- [x] Common EmitC remains neutral: any new generic route-container plumbing
      only carries provider-built loops/steps and does not infer RVV dtype,
      intrinsic, tail, schedule, or resource semantics.
- [x] If grouped `u2` emits executable generated code, `ssh rvv` correctness
      and same-target timing are collected for the dequantize path and the
      dequant-clamp path when included. If it remains fail-closed, the exact
      blocker is recorded and runtime/performance checks are not claimed.
- [x] Required focused builds/checks are run or an exact blocker is recorded:
      `tcrv-opt`, `tcrv-translate`,
      `tianchenrv-rvv-extension-plugin-test`,
      `tianchenrv-target-artifact-export-test`, relevant low-precision/
      Gearbox/generated-bundle lit tests, script self-tests where touched,
      `git diff --check`, Trellis task validation, and
      `git diff --cached --check`.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor, source-front-door, route-id,
      artifact-name, or q8/q4 authority.
- [x] Commit one coherent grouped `u2` payload or precise fail-closed slice and
      leave `.trellis/.current-task` active unless final Gate 3/Gate 4 can
      honestly close.

## Current Round Result

This slice implements the grouped `u2` payload instead of keeping it pending.
The RVV resource selector now selects the grouped
`product-reduction-dequantize` and `product-reduction-dequant-clamp`
candidates for the supported typed low-precision body/config/resource shape.
The provider-owned statement plan emits a grouped main loop over
`grouped_tail_start = (n / (full_chunk_vl * 2)) * (full_chunk_vl * 2)` plus a
u1 tail loop, so the second grouped slice is only loaded inside the full grouped
prefix. The second grouped reduction consumes the first grouped reduction result
as its accumulator, preserving the existing `dot_acc_vec` carry/result
semantics.

Executable evidence was collected on `ssh rvv` for runtime counts
`0,1,16,17,257` across dequantize and dequant-clamp, and same-target timing was
collected for counts `257,1024` against the existing scalar C baselines. The raw
timing evidence remains a regression signal, not a performance-win claim:
dequantize recorded 8 summary rows / 16 measurement rows and dequant-clamp
recorded 16 summary rows / 32 measurement rows under
`artifacts/tmp/grouped-u2-same-target/grouped-u2-same-target-measure/`.

The macro task remains active. The next continuation point is final campaign
cleanup/audit: decide whether any stale measurement-only scaffolding or
temporary grouped-u2 evidence expectations should be removed or justified, then
run the final macro-gate closeout only if the production path and evidence
state are still truthful.

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

## Previous Round Result

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

## Previous Gate 5 Round Result

Gate 5 cleanup is complete for this slice. Product-reduction
`widening_product_reduce_dequantize_f32` and
`widening_product_reduce_dequant_clamp_f32` now expose the scalar-dequant
epilogue contract truthfully:

- RVV contraction route-family facts and route descriptions use
  `dequant-splat` C type summaries for product-reduction dequant epilogues.
- Product-reduction dequant route metadata mirrors
  `tcrv_rvv.rhs_broadcast_intrinsic` as the provider-owned final f32 scalar
  splat leaf.
- Product-reduction dequant route metadata no longer mirrors
  `tcrv_rvv.dequantize_convert_intrinsic` or
  `tcrv_rvv.dequantize_scale_intrinsic`; target validation rejects those keys
  when inserted as stale standalone vector-dequant mirrors.
- Standalone `dequantize_i32_to_f32` and standalone dequant-clamp epilogue
  route families keep their existing vector convert/scale contracts.
- Generated-bundle ABI evidence scripts now consume this metadata contract and
  assert the old vector convert/scale keys as forbidden stale mirrors for
  product-reduction dequantization.

This slice changes route metadata ownership and target validation behavior, not
the executable post-loop computation already changed in the previous scalar
dequant epilogue optimization. Same-target `ssh rvv` measurement therefore
remains the previous slice's evidence and was not rerun for this metadata/
validation cleanup.

## Current Main-Loop Vector-Carry Round Result

This slice consumes the bounded low-precision resource/reduction layout in the
production statement plan for
`widening_product_reduce_dequantize_f32` and
`widening_product_reduce_dequant_clamp_f32`.

- RVV resource/route facts now name the reduction layout as
  `vector-i32m1-carry-dot_acc_vec-across-runtime-vl-chunks-final-scalar-extract-f32-store.v1`.
- The direct-contraction statement plan declares one `dot_acc_vec`
  `vint32m1_t` local, seeds it from `acc[0]` before the loop, passes it as the
  current accumulator to each `__riscv_vwredsum_vs_i16mf2_i32m1`, updates it
  across chunks, and extracts one scalar only after the loop.
- Target artifact validation requires the vector-carry local, pre-loop seed
  assignment, vector-carry loop assignment, and post-loop scalar extraction; it
  rejects stale product/reduction scalar-carry loop shapes.
- Generated-bundle ABI evidence parsing and focused FileCheck tests require the
  same vector-carry shape and assert that the loop no longer contains per-chunk
  scalar extraction/resplat.
- Common EmitC remains neutral: it only materializes provider-supplied local
  declaration initializers, call-opaque steps, assignments, loops, and post-loop
  steps.

`ssh rvv` runtime correctness and same-target measurement were collected under
`artifacts/tmp/gate3-gate4-main-loop-vector-carry/vector-carry-main-loop-after/`.
Both bounded kernels passed correctness guards. Same-target timing still does
not support a performance-win claim: generated RVV best-speedup ratios remained
below `1.0` for the measured sizes, roughly `0.56-0.65` for
`widening_product_reduce_dequantize_f32` and `0.51-0.63` for
`widening_product_reduce_dequant_clamp_f32` in the recorded summaries. This
means the slice is a production loop-shape improvement and validation step, not
the final performance closeout.

## Current Grouped-Candidate Fail-Closed Round Result

This slice makes the next resource-aware main-loop optimization boundary
explicit without pretending the grouped loop is executable yet.

- RVV low-precision resource facts now publish candidate-set
  `rvv-low-precision-direct-contraction-resource-candidate-set.v2[i8mf4-i16mf2-i32m1-f32m1:u1-vector-carry,u2-grouped-tail-safe-pending]`.
- `widening_product_reduce_dequantize_f32` and
  `widening_product_reduce_dequant_clamp_f32` each get a grouped `u2`
  resource candidate with `unroll_factor = 2`, `accumulator_count = 2`, an
  explicit larger live-vector budget, and rejection reason
  `unsupported-tail-safe-grouped-product-reduction-statement-plan`.
- The executable selected candidate remains the legal `u1` vector-carry route.
  The grouped `u2` candidate is a named provider-owned candidate, not route
  authority and not an artifact or metadata promise.
- RVV provider validation now rejects stale or forced grouped `u2`
  selected-candidate facts for both dequantize and dequant-clamp before Common
  EmitC or target export can treat the body as executable. The diagnostic names
  the missing tail-safe grouped product-reduction statement-plan payload.
- Self-repair: the first implementation only consumed pass/realized resource
  selected-candidate facts on non-clamp product-dequant bodies. The clamp path
  could therefore ignore a forced grouped `u2` candidate. The derivation now
  consumes pass facts for both product-reduction dequantize and
  dequant-clamp bodies, and the focused negative clamp fixture covers this.
- Common EmitC receives no new RVV scheduling, dtype, intrinsic, tail, or
  resource inference. This slice only changes RVV-owned resource facts,
  provider validation, script mirrors, and focused tests.

Generated executable C is unchanged because grouped `u2` remains fail-closed.
`ssh rvv` correctness and timing were therefore not rerun in this slice; the
macro task remains open and still cannot claim a performance win.

## Continuation Point

Continue the open macro task in the RVV production compiler path. The next
slice should implement the RVV statement-plan payload needed to make the
grouped `u2` candidate tail-safe and executable: either a correct grouped
main/tail loop structure or another provider-owned loop payload that avoids
VL=0/no-op assumptions and out-of-bounds tail loads while preserving typed-body
authority and the explicit live-vector budget. After that production-source
change, reselect the grouped candidate, regenerate artifacts, and collect
same-target `ssh rvv` correctness/timing without making an unsupported win
claim. Do not create a neighboring generated-bundle evidence task.
