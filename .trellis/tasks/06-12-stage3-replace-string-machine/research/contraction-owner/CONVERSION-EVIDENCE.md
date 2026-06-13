# Stage 3 换心 — CONTRACTION family conversion (the 7th / final per-family owner)

HEAD ddc33f91. Owner = the DirectContraction statement-plan (in the shared
`RVVEmitCStatementPlanOwners.cpp`) + the route-family/description file
`RVVEmitCContractionRouteFamilyPlanOwners.cpp` (11,745 lines, the biggest).

## Owner op-kind inventory (9 kinds, from isRVVSelectedBodyContractionRouteOperation)

| op-kind | structure | CONVERTED? | HW lamp |
|---|---|---|---|
| WideningProduct (signed i8mf4→i16mf2) | load+load+vwmul+store | YES (byte-identical) | (artifact-only, no harness op-kind) |
| WideningMAccAdd | load×3+vwmacc+store | YES | PASS widening_macc_add |
| WideningDotReduceAdd | load+load+vwmul+vredsum (scalar-carry) | YES | PASS widening_dot_reduce_add |
| StridedInputWideningDotReduceAdd | vlse+vlse+vwmul+vredsum | YES | PASS strided_input_widening_dot_reduce_add |
| ComputedMaskWideningDotReduceAdd | cmp+vwmul_m+vmerge+vredsum | YES | PASS computed_masked_widening_dot_reduce_add |
| ComputedMaskStridedInputWideningDotReduceAdd | vlse+cmp+vwmul_m+vmerge+vredsum | YES (byte-identical) | (covered by strided + masked lamps) |
| WideningProductReduceAdd | load+load+vwmul+vwredsum (widening reduce) | YES | PASS widening_product_reduce_add |
| WideningProductReduceDequantizeF32 | **Gearbox multi-region, mutable acc** | **NO — BLOCKS** | n/a (Gearbox) |
| WideningProductReduceDequantClampF32 | **Gearbox multi-region + packed-i4** | **NO — BLOCKS** | n/a (Gearbox) |

Plus the unsigned `u8` (ui8/ui16/ui32 → vwmulu/vwredsumu) variants of WideningProduct /
WideningProductReduceAdd: verifier-legal, **metadata-only lit (no body-C RUN line)**, currently
fall back to the owner. Converter DECLINES them (the `kind == "signed_widening_product"` guard),
so the owner emits `vwmulu` — no mislower (adversarial-probe confirmed).

## What was implemented (lib/Conversion/RVV/RVVToEmitC.cpp, +424 lines, code-only)

1. TypeConverter rung: i8/mf4 → vint8mf4_t (the fractional-LMUL low-precision multiplicand load);
   vectorDType/vectorScalarCType extended for i8 (i8 / int8_t).
2. emitWideningProduct: vwmul_vv_<resultD><resultL> (intrinsic from the RESULT type).
3. emitWideningMAcc: vwmacc_vv_<resultD><resultL>, accumulator-first C call order.
4. emitWideningDotReduce: vwmul + running-seed + vredsum, reusing the scalar-carry standalone
   reduction machinery (pre-loop out[0]=acc[0], lane-0 store base VL=1).
5. emitMaskedWideningDotReduce: zero bg (running vl) + vwmul_vv_..._m (masked) + vmerge + reduce.
6. isStandaloneReductionOp / emitStandaloneReductionPreLoopSeed extended to recognize the
   WideningDotReduce / MaskedWideningDotReduce ops (shared scalar-carry result layout).
7. emitStandaloneReduce input guard relaxed to accept a widening_product producer (the
   widening-product-reduce chain: load→load→vwmul→vwredsum).
8. orderedOps reorder: computed-mask dot-reduce moves the compare to immediately follow its two
   compare-input loads (BEFORE the dot-input loads) to match the legacy mask-early emit order.

## Verification

- Code-only rebuild: `rm -f build/bin/tcrv-opt build/bin/tcrv-translate && ninja bin/tcrv-opt
  bin/tcrv-translate`. tcrv-opt timestamp > libTianChenRVConversionRVV.a.
- 7 signed families: CONVERTED by the conversion (NOT owner-fallback) AND byte-identical to the
  saved legacy `--tcrv-rvv-emitc-to-cpp` oracle (oracle-*.c).
- 4 new structural lit tests under test/Conversion/RVV/ (widening-product, widening-macc,
  widening-dot-reduce, computed-masked-widening-dot-reduce) — all PASS.
- Full lit `cd build/test && lit.py .`: 573 passed, EXACTLY the 3 environmental reds (verified
  identical at the clean HEAD baseline: self-test + the 2 computed-masked-strided-input dry-runs).
  No non-environmental new red.
- Adversarial: a verifier-legal UNSIGNED widening-product is DECLINED (no vwmul mislower → owner
  emits vwmulu); malformed kinds/non-m1 results refuse.
- ssh rvv (riscv64, no --dry-run): PASS op=widening_dot_reduce_add / widening_macc_add /
  strided_input_widening_dot_reduce_add / computed_masked_widening_dot_reduce_add /
  widening_product_reduce_add — all `ssh_evidence: true`, `status: success` (hw/ evidence.json).

## Owner deletion — BLOCKED (STOP before deletion, per task instruction)

The owner statement-plan CANNOT be deleted: it remains load-bearing for
1. the 2 Gearbox dequant families (WideningProductReduceDequantizeF32 /
   WideningProductReduceDequantClampF32) — multi-region mutable-accumulator bodies the per-op SSA
   conversion model cannot represent (emitc.assign / cross-region handoff). Their body C is
   lit-tested by the 2 packed-i4 `--tcrv-rvv-emitc-to-cpp` fixtures (still owner-emitted, PASS).
2. the unsigned-u8 vwmulu/vwredsumu variants (verifier-legal, metadata-only lit; left on the
   strangler-fig fallback — the WideningProductOp ODS already documents unsigned as a fail-closed
   surface "until production support is implemented").

The 11,745-line `RVVEmitCContractionRouteFamilyPlanOwners.cpp` is the route-family / description /
provider-facts / diagnoseMissing layer (0 statement-plan steps — grep TCRVEmitCCallOpaqueStep=0),
which the established precedent KEEPS regardless. The contraction BODY emission lives in the shared
`RVVEmitCStatementPlanOwners.cpp` DirectContraction path, which is now dead for the 7 converted
families but still live (and required) for the dequant/packed-i4/unsigned ones.

NET: 7 of 9 contraction families (+all 5 harness op-kinds with a hardware lamp) converted through
the real DialectConversion, byte-identical, hardware-validated. Owner NOT deleted — 2 Gearbox
dequant rungs genuinely block it (not forced, per "do NOT force a deletion that breaks fixtures").
