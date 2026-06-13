# Stage 3 换心 — WIDENING-CONVERSION family conversion (+ dual-cmp confirmation)

HEAD 312515f9 (code-only edit on top). Additive coverage: convert the remaining
TRACTABLE families that still fell back to the legacy materialization path. No
owner deleted (strangler-fig additive — owners stay live for the gearbox/dead
families).

## Families converted

| family | op-kind | status at HEAD | this change |
|---|---|---|---|
| widening_convert i16/mf2 -> i32/m1 | `widen_i16_to_i32` | FELL BACK (no handler) | NEW `emitWideningConvert` + dispatch + guard |
| widening_convert i32/m1 -> i64/m2 | `widen_i32_to_i64` | FELL BACK (no handler) | same handler (RESULT-typed callee) |
| runtime-scalar dual cmp + mask_and + select | `runtime_scalar_dual_cmp_mask_and_select` | ALREADY CONVERTED | structural test only (no new emit code) |

dual-cmp was already fully converting at HEAD: every op in its body
(load/splat/compare/mask_and/select/store) already had a handler, so the body
lowered with zero new code. The task listed it as a fallback candidate from a
stale survey; the HEAD measurement (matpass output == conversion output, only
`emitc.func` survives, byte-identical) shows it is a free win. Added a structural
lit test to lock the two-compare-composition coverage and ran the hardware lamp
to confirm the converted path runs on real RISC-V.

## NOT converted (correctly out of scope — reported, not forced)

- **dequantize-i32-to-f32** (standalone): the legacy/selected oracle is **Gearbox
  unroll-by-2** (`v6 = v5 * 2`, two unrolled bodies per loop iteration, `v += v6`).
  The per-op SSA `with_vl` conversion emits a single-iteration loop, so converting
  it would (a) fail byte-identity against the only oracle that exists and (b)
  silently discard a real N3 Gearbox unroll schedule. The body will not even
  materialize without the pass-produced `tcrv_rvv.gearbox.candidate_set` fact. The
  existing `emitDequantize` guard already refuses a standalone dequantize (no
  select) on purpose (RVVToEmitC.cpp emitDequantize scope guard). Stays on the
  validated gearbox path; converting would discard the N3 unroll-by-2.
- **broadcast-mul / broadcast-sub**: dead Stage1 `tcrv_rvv.i32_load` op that
  fail-closes before conversion (per task: dead Stage1 code, not a conversion gap).
- **widening-product-reduce-dequant*** (5 fixtures): the Gearbox dequant —
  multi-region mutable-accumulator bodies the per-op SSA model cannot represent
  (separate effort, skip).

## Patterns added (lib/Conversion/RVV/RVVToEmitC.cpp, code-only)

1. `riscvIntrinsicName` new mnemonic `vwcvt_x_x_v` ->
   `__riscv_vwcvt_x_x_v_<dtype><lmul>` suffix from the WIDENED RESULT type.
2. `emitWideningConvert`: one `__riscv_vwcvt_x_x_v_<resultD><resultL>` call.
   - callee dtype/lmul from the RESULT vector (i32m1 / i64m2);
   - load uses the SOURCE type (vle16 i16mf2 / vle32 i32m1); setvl + store use the
     DEST config (e32m1 / e64m2);
   - GUARD: accepts only the two bounded SIGNED kinds the op verifier permits
     (`sign_extend_widen_vf2` i16mf2->i32m1, `widen_i32_to_i64` i32m1->i64m2);
     refuses any other kind or any (source,result) pairing outside the signed
     grid (unsigned would need `vwcvtu`, which this does NOT emit) -> falls back.
3. Dispatch case for `tcrv_rvv.widening_convert` in the with_vl body loop.

## Byte-identity (authoritative oracle = legacy materializer C, extern "C" =
export contract `emitExternC=true`)

Both widen families: `--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`
diff vs the legacy materializer C (captured from a stashed pre-change build) ==
BYTE-IDENTICAL, including the `extern "C"` qualifier the export path applies.
dual-cmp: byte-identical (already converted, no change).

Key intrinsics (from the result type):
- widen_i16_to_i32: `__riscv_vwcvt_x_x_v_i32m1`
- widen_i32_to_i64: `__riscv_vwcvt_x_x_v_i64m2`

## Survey — legacy-path fallback count dropped 11 -> 8

Accurate survey (realize boundary + gearbox schedules, then probe whether the
conversion reduces the module to emitc-only):

- HEAD: 11 fallbacks.
- After: 8 fallbacks (the 3 widen fixtures now convert).
- Remaining 8 = 2 broadcast (dead Stage1 i32_load) + 1 dequantize-i32-to-f32
  (gearbox unroll-2) + 5 widening-product-reduce-dequant* (gearbox dequant).
  Exactly the Gearbox dequant + dead broadcast Stage1, as expected.

(Note: the task prose said "12 / 166". My measured HEAD baseline is 11; the
difference is the already-converted dual-cmp variants the stale survey still
counted. The verifiable delta is the 3-fixture widen drop.)

## Verification

- Code-only rebuild: `rm -f build/bin/tcrv-opt build/bin/tcrv-translate &&
  cd build && ninja bin/tcrv-opt bin/tcrv-translate`. tcrv-opt timestamp >
  libTianChenRVConversionRVV.a (verified).
- Full lit (`cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py
  .`): 582 tests, EXACTLY the 3 environmental
  reds (self-test + the 2 computed-masked-strided-input dry-runs). No new red.
  (4 new structural tests added: +4 total over the pre-change 578.)
- 4 new structural lit tests under test/Conversion/RVV/ — all PASS:
  widening-convert-i16-to-i32, widening-convert-i32-to-i64,
  runtime-scalar-dual-cmp-mask-and-select, widening-convert-negative.
- Adversarial guard probe: a widening_convert with a malformed kind
  (`bogus_widen_kind`, `widen_i64_to_i128`) REFUSES — 0 `vwcvt` mislower,
  0 `emitc.func`. Mechanism: the op VERIFIER fail-closes the malformed kind
  (`isSupportedGenericWideningConvertKind`) before conversion runs; the
  emitWideningConvert kind/grid guard is defense-in-depth with no reachable
  adversarial case (the verifier ties kind<->grid exactly as the guard does).
  The probe pins the regression lesson "malformed => no vwcvt, no emitc.func";
  it is NOT evidence the conversion guard itself is independently load-bearing.
  Negative lit test pins this.

## Conversion is the ACTIVE path at BOTH seams (not just the materialize seam)

The hardware lamp drives the export seam (`--tcrv-materialize-emission-plans |
tcrv-translate --tcrv-export-target-artifact-bundle`), NOT the
materialize-lowerable-routes seam used for the byte-identity matpass diff. Both
seams gate the conversion identically: TargetArtifactExport.cpp:2132-2154 tries
`conversion::rvv::convertRVVModuleToEmitC` FIRST on a clone and, on full
conversion, returns the converted emitc module (line 2152) — the legacy string
route is built ONLY in the fallback (line 2157, "reachable ONLY here"). Since
widen now fully converts, the export seam takes the converted module and never
builds the legacy route. Therefore the widen hardware PASS attributes
correctness to the conversion END-TO-END (widen stops using the legacy path at
both the materialize seam and the export/bundle seam).

## ssh rvv (riscv64, NO --dry-run) — all PASS, ssh_evidence=true, status=success

- `PASS op=widen_i32_to_i64 counts=1,7,16,17,257` (sign_extension_checked
  wide_magnitude_checked tail_preserved two_input_patterns_checked)
- `PASS op=widen_i16_to_i32 counts=1,7,16,17,257` (--pre-realized-selected-body;
  sign_extension_checked two_input_patterns_checked tail_preserved)
- `PASS op=runtime_scalar_dual_cmp_mask_and_select counts=1,7,16,17,257
  rhs_scalar_a_values=-37,91 rhs_scalar_b_values=-37,91` (confirms the converted
  path on hardware)

Evidence under this dir: widen_i32/ widen_i16/ dual_cmp/ each with evidence.json
(status=success, ssh_evidence=true) + remote stdout + generated bundle.
