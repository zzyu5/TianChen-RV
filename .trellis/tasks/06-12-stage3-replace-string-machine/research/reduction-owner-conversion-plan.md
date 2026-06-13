# Stage 3 换心 — retire the reduction owner (6th owner): StandaloneReduction + ComputedMaskAccumulation

HEAD 67077b33. Reduction owner = RVVEmitCReductionAccumulationStatementPlanOwners.cpp (1950 lines).
Owns 4 families:
- Reduction (reduce_add) — CONVERTED (emitReduce, per-chunk-base layout)
- PlainMAcc (macc_add, scalar_broadcast_macc_add) — CONVERTED (emitMAcc)
- ComputedMaskAccumulation (masked_macc: computed-mask + runtime-scalar-cmp) — ALREADY CONVERTED
  (emitMaskedMAcc). Probed all 6 macc fixtures incl m2/runtime-scalar: fail=0.
- StandaloneReduction — NOT CONVERTED (this task's work). Covers:
  - StandaloneReduce{Add,Min,Max}  (tcrv_rvv.standalone_reduce, plain)
  - WideningStandaloneReduceAdd    (tcrv_rvv.standalone_reduce, i16mf2 -> i32m1, vwredsum)
  - ComputedMaskStandaloneReduce{Add,Min,Max}        (tcrv_rvv.masked_standalone_reduce)
  - RuntimeScalarComputedMaskStandaloneReduce{Add,Min,Max} (masked_standalone_reduce + scalar splat)

## Byte-identical oracle structure (from legacy --tcrv-rvv-emitc-to-cpp; saved under reduction-owner-oracles/)

### Plain standalone_reduce (add/min/max), result_layout=store-standalone-reduction-lane0-to-output-scalar
PRE-LOOP (after pre-loop full-chunk setvl, BEFORE the for-loop):
  // standalone_reduce role=compute callee=__riscv_vmv_v_x_<rd>m1
  const <celt> vS = acc[0];           // acc = accumulator_seed ABI (const ptr -> const scalar)
  v<rd>m1 vSeed = __riscv_vmv_v_x_<rd>m1(vS, 1);   // splat seed, VL literal 1, RESULT lmul m1
  // store role=store callee=__riscv_vse<sew>_v_<rd>m1
  __riscv_vse<sew>_v_<rd>m1(out, vSeed, 1);        // store to out BASE (no +i), VL=1
IN-LOOP body:
  load input (vle, source dtype/lmul)
  // standalone_reduce role=compute callee=__riscv_vmv_v_x_<rd>m1
  <celt> vR = out[0];                 // out non-const ptr -> non-const scalar
  v<rd>m1 vRSeed = __riscv_vmv_v_x_<rd>m1(vR, 1);  // VL literal 1, RESULT lmul m1
  // standalone_reduce role=compute callee=__riscv_v<red>_vs_<src><srcLmul>_<rd>m1
  v<rd>m1 vRed = __riscv_v<red>_vs_...(input, vRSeed, bodyVL);
  // store role=store callee=__riscv_vse<sew>_v_<rd>m1
  __riscv_vse<sew>_v_<rd>m1(out, vRed, 1);         // store to out BASE (no +i), VL=1

Note the two standalone_reduce-roled vmv_v_x splats each carry their OWN provenance comment
(role=compute, source_op=tcrv_rvv.standalone_reduce). The store comment is role=store
source_op=tcrv_rvv.store.

### Widening standalone_reduce (signed_widening_reduce_add): src i16mf2 -> result i32m1
Same shape; the in-loop load is vle16_v_i16mf2, the reduce is __riscv_vwredsum_vs_i16mf2_i32m1.
The pre-loop / in-loop seed splats + stores are i32m1 (result type). acc/out are int32_t*.
The lhs ABI buffer is const int16_t*.

### Computed-mask masked_standalone_reduce (add/min/max)
PRE-LOOP: same seed structure but provenance source_op=tcrv_rvv.masked_standalone_reduce.
  const <celt> vS = acc[0]; v<rd>m1 vSeed = vmv_v_x_<rd>m1(vS, 1); vse..(out, vSeed, 1).
IN-LOOP:
  load cmp-lhs; load cmp-rhs; load source;     (all input lmul, e.g. m1/m2)
  compare(cmp-lhs, cmp-rhs) -> mask (vmsle_vv..._b<maskbits>)
  // masked_standalone_reduce role=compute callee=vmv_v_x_<src><srcLmul>
  v<src><srcLmul> neutral = __riscv_vmv_v_x_<src><srcLmul>(<NEUTRAL>, bodyVL);  // INPUT lmul, bodyVL
  // masked_standalone_reduce role=compute callee=vmerge_vvm_<src><srcLmul>
  v<src><srcLmul> masked = __riscv_vmerge_vvm_<src><srcLmul>(neutral, source, mask, bodyVL);
  // masked_standalone_reduce role=compute callee=vmv_v_x_<rd>m1
  <celt> vR = out[0]; v<rd>m1 vRSeed = vmv_v_x_<rd>m1(vR, 1);   // RESULT lmul m1, VL=1
  // masked_standalone_reduce role=compute callee=v<red>_vs_<src><srcLmul>_<rd>m1
  v<rd>m1 vRed = __riscv_v<red>_vs_...(masked, vRSeed, bodyVL);
  // store
  vse..(out, vRed, 1);
NEUTRAL constant by kind+sew (legacy getRVVStandaloneReductionStatementPlanInactiveNeutral):
  add->"0"; min->sew64?"9223372036854775807":"2147483647";
  max->sew64?"(-9223372036854775807-1)":"(-2147483647-1)".

### Runtime-scalar-cmp masked_standalone_reduce (add)
Same as computed-mask but the compare RHS comes from a tcrv_rvv.splat of a runtime scalar ABI
value (int32_t v2) instead of a loaded buffer: emitSplat already exists. Op order in body:
  load source; splat(scalar) -> cmp-rhs; load cmp-rhs-buffer (the compare LHS=source? see oracle);
Actually oracle order: load(lhs=source v1); splat(scalar v2)->v15; load(v3)->v17;
  compare(v14, v15)->v18; masked_standalone_reduce(mask=v18, input=v17, seed=out, vl).
So masked_standalone_reduce input is the v3-loaded buffer, mask from compare(v1-load, scalar-splat).

## Implementation (lib/Conversion/RVV/RVVToEmitC.cpp, code-only)

1. TypeConverter: add i16/mf2 (i16mf2 -> vint16mf2_t). Extend vectorDType (i16->"i16"),
   vectorScalarCType (i16->"int16_t"). Add mf2 fractional lmul to the vector addConversion.
   convertVectorTypeToEmitC and vectorElementWidth already work for i16 once dtype/lmul accepted.
2. Standalone-reduction detection + pre-loop seed emission in VariantToEmitCFunc::matchAndRewrite:
   - detect body contains standalone_reduce or masked_standalone_reduce with
     result_layout == "store-standalone-reduction-lane0-to-output-scalar".
   - emit the pre-loop seed (acc[0] -> splat m1 -> store out base VL=1) between pre-loop setvl
     and the for-loop. The reduction RESULT vector type gives the seed lmul (always m1) + sew.
     The accumulator_seed ABI value gives the acc buffer; the store target = the body's
     tcrv_rvv.store buffer (out).
3. emitStandaloneReduce / emitMaskedStandaloneReduce in-loop handlers (dispatch in the op loop).
   - in-loop seed: out[0] -> splat result-m1 VL=1.
   - reduce: __riscv_v<red>_vs_<src><srcLmul>_<rd>m1 (widening: vwredsum_vs_i16mf2_i32m1).
   - masked: neutral splat (input lmul, bodyVL) + vmerge (input lmul) before the seed+reduce.
4. Standalone store: when the body's store value is a (masked_)standalone_reduce result, store to
   BASE (no +i), VL literal 1. Mirror the existing reduce VL=1 detection at the store dispatch.
5. Guards: malformed bodies (wrong result_layout, unsupported kind, unconvertible type,
   buffer/element mismatch) -> notifyMatchFailure -> legacy fallback (re-guard each negative probe).

## Validation
- code-only rebuild: rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja bin/tcrv-opt bin/tcrv-translate
- cd build/test && lit.py -q .  => exactly 3 environmental reds (widening-DOT-reduce, CONTRACTION owner, hw)
- every standalone fixture converts byte-identical to saved oracle
- reduce_add + macc families still convert (regression)
- if all 4 families convert: DELETE the reduction owner; KEEP route-family/provider-facts/diagnose.
