# P-B8 verdict — i16 dot-reduce autotuner wins END-TO-END; "若干 kernel" strictly met

## Headline

**The N3 性能灯 is ON end-to-end for the 2nd kernel family (signed i16 widening
dot-reduce), AUTOMATIC and selector-driven.** A plain i16 dot-reduce kernel ->
`--tcrv-rvv-materialize-gearbox-schedules` (stamps the vreg-budget fact) ->
`--tcrv-materialize-selected-lowering-boundaries` (the resource-aware selector
picks the wide i32m8 rung at budget 32 and REALIZES the deferred-wide dot-reduce
typed body) -> `--tcrv-rvv-lower-to-emitc` -> C. The compiler-emitted body is
BYTE-IDENTICAL to the P-B7 measured winner (only the symbol name differs), and a
FRESH ssh-rvv 3-way measurement on the AUTOMATIC output confirms the win at every
n. This crosses the P-B7 "WALL" (selector-driven e2e), the named multi-file gap.

## Does the autotuner now produce the i16 deferred-wide winner END-TO-END? YES.

The selector (NEW `enumerateRVVDotReduceDeferredWideLMULRungs` +
`selectRVVDotReduceDeferredWideMaxLegalLMULRung`, a DISTINCT one-widening cost
model) consumes the pass-stamped vreg-budget fact and, at budget 32, picks the
i16m4 -> i32m8 rung; the realization owner's NEW
`realizeDeferredWideDotReduceBody` then PRODUCES the deferred-wide typed body
(load i16m4 -> vwmul i32m8 -> tcrv_rvv.deferred_accumulate i32m8 -> ONE trailing
standalone_reduce i32m8->i32m1 -> i32 store). The conversion (P-B7) emits it.

## The budget DRIVES narrow-vs-wide for i16 (N3, the prune binds)

The i16 chain is a SINGLE widening (product == i32 accumulator width, the deferred
accumulate is a same-width vadd.vv); peak-live = acc_regs(m8)=8 + reserve=8 = 16.
Crossover (verified live):
- budget >= 16 -> wide i32m8 deferred body (1 tcrv_rvv.deferred_accumulate).
- budget < 16  -> the wide rung is PRUNED -> narrow i16mf2 per-iteration
  tcrv_rvv.widening_dot_reduce body (0 deferred_accumulate).
- no budget stamped (no gearbox pass) -> narrow (the existing REALIZED lit holds).
Lit `pre-realized-selected-body-realize-deferred-wide-dot-reduce-autotuner-e2e.mlir`
exercises BOTH sides (WIDE/EMITC at budget 32, NARROW at sed-injected budget 12).

## The 灯 (real ssh rvv, riscv64/clang18, AUTOMATIC e2e compiler output, I8)

Genuine scalar baseline objdump-verified: 0 vector instructions (compiled
-march=rv64gc). 3-way: warmup + best-of-11x16 clock_gettime(MONOTONIC_RAW).
Numeric correctness EXACT (integer oracle match) at every n BEFORE timing.

| n     | wide vs genuine-scalar | wide vs competent-naive-RVV | wide vs current-emission |
|-------|------------------------|-----------------------------|--------------------------|
| 257   | 4.28x                  | 3.06x                       | 6.64x                    |
| 256   | 5.53x                  | 3.01x                       | 8.06x                    |
| 1024  | 6.86x                  | 3.54x                       | 10.13x                   |
| 4096  | 7.52x                  | 3.80x                       | 11.15x                   |
| 16384 | 5.64x                  | 2.85x                       | 8.33x                    |
| 65536 | 3.97x                  | 2.11x                       | 5.93x                    |

WINS vs genuine scalar (3.97-7.52x) AND vs competent naive RVV (2.11-3.80x) at
EVERY n. Evidence: `dot-reduce-i16/pb8_lamp_3way_ssh_rvv_stdout.txt`. The win
matches P-B7's emitted-body measurement (within noise), as expected from the
proven byte-identity of the AUTOMATIC body to the P-B7 measured body.

## "若干 kernel" — NOW STRICTLY MET (autotuner-driven e2e on 2 families)

| grade | kernels (before P-B8) | kernels (after P-B8) |
|---|---|---|
| autotuner-driven e2e (select->realize->lower->win) | **byte only** | **byte + i16 dot-reduce** |
| emitted-body measured win (selection was the wall) | i16 dot-reduce | (subsumed into e2e) |
| hand-variant ceiling only | u8 | u8 |

The STRICT N3 sense ("the autotuner wins on SEVERAL kernels") is now met: the
autotuner picks + produces + wins on TWO distinct families end-to-end.

## What remains (disclosed follow-up, like byte's P-B6)

The i16 deferred-wide bundle-export (.o/.h via `--tcrv-materialize-emission-plans`)
is FAIL-CLOSED: the route-family description engine does not yet recognize the
deferred-wide dot-reduce typed-compute chain (errors "unsupported generic
tcrv_rvv.widening_product kind 'signed_widening_product'"). The runnable-C e2e 灯
(this evidence) works; the deployable bundle is the cherry, exactly like the byte
path's P-B6 followed P-B5. NOT in scope per the task ("the i16 bundle-export MAY be
left as a follow-up if the runnable-C e2e 灯 is achieved").

## Build / lit

Full clean rebuild GREEN. lit: 601 tests, 598 pass, exactly the 3 pre-existing
documented environmental reds (verified failing at HEAD a5e0b4fe WITHOUT these
changes via stash+rebuild; the new e2e lit + the i16 conversion/dialect + byte +
narrow paths byte-identical). RVV dialect/divergence gtests green. NOT
git-committed (per task).

## Files touched

- include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h — NEW i16 single-widening
  enumerator + selector (RVVDotReduceDeferredWideLMULRung).
- lib/Dialect/RVV/IR/RVVDialect.cpp — dot-reduce body allowlist accepts ONLY the
  vreg-budget resource fact (minimal).
- lib/Plugin/RVV/RVVGearboxSchedules.cpp — stamp the budget fact on the narrow
  i16mf2 dot-reduce pre-realized body.
- lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp — NEW
  createRealizedGenericDeferredAccumulate + realizeDeferredWideDotReduceBody +
  the budget-driven wide branch in the PLAIN dot-reduce realization.
- test/Target/RVV/pre-realized-selected-body-realize-deferred-wide-dot-reduce-autotuner-e2e.mlir
  — NEW e2e lit (WIDE/EMITC/NARROW, the budget-driven crossover).
