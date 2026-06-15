# P-B8 — i16 dot-reduce autotuner-driven e2e (parallel of P-B5)

## Goal
An i16 widening dot-reduce kernel -> gearbox stamps vreg-budget on the dot-reduce
body -> realization owner runs an i16 SINGLE-widening resource-aware selector ->
at budget 32 picks the wide i32m8 rung -> PRODUCES the deferred-wide i16 typed
body (load i16m4 -> vwmul i32m8 -> deferred_accumulate i32m8 -> ONE trailing
standalone_reduce i32m8->i32m1 -> i32 store) -> conversion emits it (already
works, P-B7) -> ssh rvv. End-to-end, automatic, selector-driven, like the byte
path.

## The wall map (verified)
- Conversion (isDeferredWideDotReduceBody / emitDeferredWideDotReduceBody) EXISTS (P-B7).
- Dialect ops (deferred_accumulate + wide verifier branches in product/reduce/store/
  setvl/with_vl) EXIST (P-B7). Strip config must be SEW16/m4 (reduce verifier line 218).
- The gearbox pass (RVVGearboxSchedules.cpp runOnOperation) walks ONLY
  TypedWideningProductReduceDequantize* — it does NOT stamp the dot-reduce body. (gap 1)
- The dot-reduce body op allowlist (isAllowedTypedWideningDotReducePreRealizedBodyAttr,
  RVVDialect.cpp:379) does NOT accept the budget attr. (gap 2 — cheapest wall)
- The realization owner dot-reduce branch (line 2487) always falls to the narrow path. (gap 3)
- The byte enumerator (enumerateRVVLowPrecisionAccumulatorLMULRungs) does TWO widenings;
  the i16 chain is ONE widening (source i16 -> product==acc i32). Need a NEW enumerator. (gap 4)

## Plan (5 edits, byte/narrow/strided/masked BYTE-UNTOUCHED)
1. RVVGearboxSchedule.h: NEW enumerateRVVDotReduceDeferredWideLMULRungs (one-widening
   cost model: source i16{mf2,m1,m2,m4} -> product==acc i32 one step; acc & product
   are the SAME i32 width => peak-live = acc_regs + reserve; budget prune binds) +
   selectRVVDotReduceDeferredWideMaxLegalLMULRung.
2. RVVDialect.cpp: isAllowedTypedWideningDotReducePreRealizedBodyAttr accepts the budget
   attr (MINIMAL — budget only). STALE-AUTH still rejects route_id (forbidden-authority,
   separate check).
3. RVVGearboxSchedules.cpp: a parallel walk stamps the budget fact (only) on
   TypedWideningDotReducePreRealizedBodyOp.
4. RVVContractionSelectedBodyRealizationOwner.cpp: NEW createRealizedGenericDeferredAccumulate
   op-builder + realizeDeferredWideDotReduceBody (own setvl/with_vl at SEW16/m4) + wire
   the budget-driven wide branch into the PLAIN dot-reduce branch (line 2487 only).
5. Tests: lit for budget-drives-narrow-vs-wide (both crossover sides) + e2e ssh rvv.

## N3 crux: the i16 enumerator cost model (one widening)
- source rungs {mf2, m1, m2, m4}; product LMUL = next wider (m1,m2,m4,m8);
  accumulator LMUL == product LMUL (i32, same step — the deferred accumulate is
  same-width vadd.vv). m4 source -> i32m8 acc (the winner); m8 acc has no wider rung.
- peak-live (A=1): the i32 accumulator (== product width, they alias the deferred
  vadd) + reserve (the two i16 source loads + slack). legal iff acc_regs + reserve
  <= budget. At budget 32 the m8 rung (8 + reserve) fits -> wide; a constrained
  budget prunes m8 -> narrower rung -> fall through to narrow i16mf2.
