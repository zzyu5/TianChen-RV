# Stage 3 换心 — Gearbox dequant + unsigned-u8 conversion: architecture analysis

HEAD 81184017. Goal: retire `RVVEmitCContractionRouteFamilyPlanOwners.cpp` (11,745 lines)
by converting the LAST 2 blocking families + unsigned-u8 through the real
RVV->emitc DialectConversion (`lib/Conversion/RVV/RVVToEmitC.cpp`).

## The byte-identical oracle (regenerated from the live legacy materializer)

REGEN-*.c in this dir are the legacy `--tcrv-rvv-emitc-to-cpp` emission for each
dequant fixture (the byte-identical target). Pipeline:
`tcrv-opt FX --tcrv-rvv-materialize-gearbox-schedules --tcrv-materialize-selected-lowering-boundaries
 --tcrv-materialize-emission-plans | tcrv-translate --tcrv-rvv-emitc-to-cpp`.

### Four distinct C structures the dequant family emits

1. **dequantize-f32 (unpacked, GROUPED u2)** REGEN-dequantize-f32.c — 91 lines, THREE regions:
   - PRE-LOOP: `vint32m1_t dot_acc_vec = __riscv_vmv_v_x_i32m1(0,1);` (mutable acc VARIABLE),
     `size_t grouped_tail_start = 0;` (mutable VARIABLE), full-chunk setvl,
     seed `dot_acc_vec = vmv_v_x_i32m1(acc[0],1)`, then arithmetic
     `grouped_tail_start = (n/(vl*2))*(vl*2)`.
   - MAIN LOOP `for (i=0; i<grouped_tail_start; i += vl*2)`: two unrolled product-reduce
     slices (load/load/vwmul/vwredsum on i, then on i+vl), each `dot_acc_vec = reduced`.
   - TAIL LOOP `for (i=grouped_tail_start; i<n; i += vl)`: one product-reduce slice,
     `dot_acc_vec = reduced`.
   - EPILOGUE: scalar extract `vmv_x_s_i32m1_i32(dot_acc_vec)`, `(float)`, `*scale`,
     `vfmv_v_f_f32m1(splat,1)`, `vse32_v_f32m1(out, splat, 1)`.

2. **dequant-clamp-f32 (unpacked, GROUPED u2)** REGEN-dequant-clamp-f32.c — 103 lines:
   identical body to #1 PLUS 2 extra ABI params (clamp_min, clamp_max) and a clamp
   epilogue: vfmv splat(min) + vmflt(splat,min) + vmerge (max with min) + vfmv splat(max)
   + vmflt + vmerge — i.e. min-then-max clamp via compare/select.

3. **dequantize-f32 packed-i4 (SINGLE region)** REGEN-dequantize-f32-packed-i4.c — 53 lines:
   single `for (i=0;i<n;i+=vl)` loop, NIBBLE UNPACK:
   `vsll_vx_i8mf4(lhs,4)`,`vsll_vx_i8mf4(rhs,4)`,`vwmul_vv_i16mf2(lo,lo)`,`vsra_vx_i16mf2(prod,8)`,
   `vsra_vx_i8mf4(lhs,4)`,`vsra_vx_i8mf4(rhs,4)`,`vwmacc_vv_i16mf2(scaled,hiL,hiR)`,
   `vwredsum_vs(dot_acc_vec)` -> `dot_acc_vec = reduced`. Mutable `dot_acc_vec` variable.
   Epilogue: scalar extract + `(float)` + `*scale` + SCALAR store `out[0] = v;` (vs splat-store).

4. **dequant-clamp packed-i4** REGEN-dequant-clamp-f32-packed-i4.c — 57 lines: #3 + scalar clamp.

Also the standalone **dequantize-i32-to-f32** (REGEN-dequantize-i32-to-f32.c, 43 lines:
single loop, vle32 i32 load, vfcvt_f_x_v + vfmul_vf dequant, vse32 f32 store) and
**dequant-clamp-f32-epilogue** (REGEN, 38 lines) — these are the Dequantization-owner
consumers, already routed by their own owner (NOT in the contraction owner's BLOCKING set).

## Why the per-op SSA model (current RVVToEmitC) cannot represent #1/#2/#3 as-is

The current `VariantToEmitCFunc::matchAndRewrite` builds exactly ONE `emitc.for` from
`scope.getBody().front()` op order, with per-op SSA into a valueMap. The dequant grouped
families need:
- A mutable `emitc.variable` (`dot_acc_vec`) declared+seeded BEFORE the loop and reassigned
  (`emitc.assign`) inside — the accumulator carries ACROSS loop iterations (SSA can't).
- THREE control regions (grouped main + scalar tail + scalar epilogue) for #1/#2, where
  the typed REALIZED IR contains only ONE iteration's ops + region markers + a metadata
  `realized_unroll_factor=2`/`vsetvl_region_count=3`/`reduction_layout=...dot_acc_vec...`.
  The grouped/tail UNROLL is SYNTHESIZED FROM METADATA by the legacy builder
  (`buildRVVSelectedBodyProductReductionDequantizationStatementPlan`,
  RVVEmitCStatementPlanOwners.cpp:945+), NOT present in the typed body.

THE KEY ARCHITECTURAL FACT: the typed REALIZED body is
`[load,load,widening_product,standalone_reduce,gearbox_cross_region_handoff, (nested with_vl: dequantize,store)]`
— one slice. The grouped-by-2 main loop, the tail loop, and the `dot_acc_vec`/`grouped_tail_start`
mutable variables are NOT in the IR; they are reconstructed by the materializer from the
`low_precision_resource.*` metadata. Per I4/I5, that metadata is a MIRROR — but here the legacy
materializer is READING the unroll factor / region count FROM metadata to BUILD compute structure.

## Decision

emitc HAS emitc.variable/emitc.assign/emitc.for, so the mutable accumulator + multi-loop
CAN be built. The conversion must:
- detect the dequant grouped/packed-i4 body (gearbox handoff + dequantize present),
- emit `dot_acc_vec` as emitc.variable, seed+assign,
- for GROUPED: synthesize the grouped main loop (step vl*2, 2 slices) + grouped_tail_start
  variable + tail loop, reading unroll=2 / region facts from the typed scope's resource attrs,
- for PACKED-I4: single loop with the vsll/vsra/vwmul/vsra/vwmacc nibble unpack,
- lower gearbox_cross_region_handoff to the accumulator value pass-through (the handoff value
  IS dot_acc_vec carried across regions; in C it is the variable),
- emit the dequant epilogue (scalar extract + (float) + *scale + splat-store OR scalar store),
- clamp epilogue for the *clamp* variants.
- unsigned-u8: vwmulu/vwredsumu/vle8_v_u8mf4/vmv_v_x_u32m1 keyed on source signedness.

GUARD: every negative/adversarial probe must still refuse (no mislower).

## DEFINITIVE FINDING (empirically confirmed) — Gearbox dequant CANNOT convert byte-identically yet

After implementing unsigned-u8 (DONE, byte-identical) I dumped the REALIZED typed IR for BOTH
the unpacked-grouped and the packed-i4 dequant bodies (commands in this dir). The result is the
load-bearing architectural fact:

**The typed REALIZED body is the SAME op sequence for packed-i4 and unpacked-grouped:**
`[load, load, widening_product, standalone_reduce, gearbox_cross_region_handoff,
  (nested with_vl: dequantize, store)]` — ONE slice, identical ops, identical types.

The DRAMATICALLY different emitted C — packed-i4 = single loop with a 7-op nibble unpack
(vsll/vsll/vwmul/vsra/vsra/vsra/vwmacc); unpacked = grouped-by-2 unrolled MAIN loop + scalar
TAIL loop + `dot_acc_vec` mutable variable — is SYNTHESIZED ENTIRELY FROM METADATA by the legacy
materializer. The only IR difference between the two is attribute strings on the SAME ops:
- `operand_form`: "packed-i4-nibbles" vs "unpacked-byte-elements"
- `unpack_intent`: "sign-extend-i4-nibbles-before-widening-product" vs "none-direct-widening-product"
- `realized_unroll_factor`: 1 vs 2
- `realized_vsetvl_region_count`: 2 vs 3

There are NO typed RVV ops for the nibble unpack (no def Sll/Sra/Shift/Nibble/Unpack in RVVOps.td)
and NO typed op for the grouped/unroll/tail loop structure. The legacy
`buildRVVSelectedBodyProductReductionDequantizationStatementPlan`
(RVVEmitCStatementPlanOwners.cpp:945+) READS `operand_form`/`realized_unroll_factor`/`unpack_intent`
metadata strings and emits the corresponding C-statement plan.

### Why this BLOCKS the conversion (and is an I5-grade finding, not a coding gap)

Per I5: executable dtype/config/OPERATION must be STRUCTURAL in the typed body, NOT inferred from
metadata strings. The legacy dequant materializer VIOLATES I5: the *compute structure itself* (which
SIMD ops run, how many times the loop unrolls, how nibbles are unpacked) is reconstructed from
`low_precision_resource.*` metadata, which I4 declares is only a MIRROR.

To make the DialectConversion emit BYTE-IDENTICAL C, it would have to read those SAME metadata
strings and synthesize the SAME grouped/tail/nibble structure — i.e. REPLICATE the I5 violation
inside the conversion. That defeats Stage 3's entire purpose (replace the string machine with real
structural IR), and produces a conversion that is, structurally, still a string machine.

The correct fix is IR-LEVEL FIRST: the gearbox realization
(`--tcrv-rvv-materialize-gearbox-schedules`) must be changed to emit the unpack/unroll/region
structure as ACTUAL typed ops (a typed nibble-unpack op; the grouped/tail loop expressed as real
multi-region typed control structure carrying the mutable accumulator). THEN the conversion can walk
real ops and emit emitc.variable/emitc.assign/emitc.for from structure, not from metadata strings.
That realization change is a separate, larger module (it touches RVVContractionSelectedBodyRealizationOwner.cpp
+ new ODS ops + the gearbox schedule pass) and is out of scope for a per-family conversion task.

### Honest outcome (matches the task's explicit "valid STOP" clause)

- unsigned-u8 widening product + widening product-reduce: CONVERTED byte-identical (vwmulu/vwredsumu),
  lit-tested, hardware-validated.
- Gearbox dequant (WideningProductReduceDequantizeF32 + WideningProductReduceDequantClampF32, both the
  unpacked-grouped and packed-i4 candidates): BLOCKED — "Gearbox needs IR-level handoff modeling first".
  The mutable accumulator (emitc.variable/assign) is buildable; the BLOCKER is that the
  unpack/unroll/region COMPUTE STRUCTURE lives only in metadata, so a byte-identical conversion would
  re-introduce the I5 violation. This is the genuine architectural limit, not a forced give-up.
- Contraction owner RVVEmitCContractionRouteFamilyPlanOwners.cpp: NOT deleted — still load-bearing for
  the 2 Gearbox dequant families (their body C is still owner-emitted; the 2 packed-i4 fixtures PASS).
