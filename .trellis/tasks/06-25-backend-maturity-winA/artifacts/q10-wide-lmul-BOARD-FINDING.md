# q1_0 wide-LMUL block-dot emit — BOARD FINDING (2026-06-26)

## Headline
**q1_0 block-dot vs ggml·rvv: 0.033× LOSS → 1.38× WIN, byte-exact, on the real rvv board (VLEN128).**
The wide-LMUL emitter-maturity fix is landed AND wired into the gearbox (auto-stamped by VLEN).

## Check-2 verdict: WITHIN-BLOCK CLEAN (not cross-block / not repack)
- q1_0 super-block = 128 elements = 4 q8_0 sub-blocks of 32 elements each (`block_q1_0 = {fp16 d; uint8 qs[16]}`, stride 18; bit `8b+i` -> lane `8b+i`).
- The fix collapses the prior **4 narrow 8-lane reductions per sub-block** into **ONE 32-lane reduction per sub-block**. The 32-lane reduce covers EXACTLY one q8 sub-block (32 contiguous q8 bytes + the 4 contiguous weight bit-bytes) and never touches a second block. `vl` stays 32. This is pure within-block lowering (a Win-A backend knob), NOT block-as-lane / repack / front-end layout.

## The fix = ggml's shipped `_vl128` lane structure
Prior emit (the 0.033× form): per 8-lane group — kmask `{1,2,..,128}` table load, `vmv_v_x_u8` broadcast, `vand`, `vmsne` -> mask, `vwcvt` widen to i16, i16-domain `vneg`/`vmerge`, then `vwredsum i16m2->i32m1`. 4 groups × 4 sub-blocks = 16 narrow reductions/super-block, 8× more work.

New emit (one 32-lane sub-block body, ggml-identical):
- `vlm_v_b{ratio}(qsbits, 32)` — the 4 packed bit-bytes load DIRECTLY as the i8 sign mask (the bits ARE the mask). No kmask table, no vmv/vand/vmsne.
- `vle8` the 32 q8 quants; **i8-domain** `vneg` + `vmerge(neg, q, mask)` -> signed q8.
- ONE `vwredsum_vs_i8{anchor}_i16m1` per sub-block (widens i8->i16 during the reduce; no separate vwcvt).

## Gearbox auto-select (VLEN-driven, not hard-coded)
The 32-element sub-block straddles m1's i8 VLMAX boundary between VLEN128/256 (exactly like q8_0), so the anchor MOVES with VLEN:
- **VLEN128 -> integer_core_lmul = "m2"** (e8m1 VLMAX 16 < 32; vbool4_t / `vlm_v_b4`).
- **VLEN256 -> integer_core_lmul = "m1"** (e8m1 VLMAX 32; vbool8_t / `vlm_v_b8`); m1 ties m2 on the capability-blind cost and the lighter footprint breaks the tie.
- Selected by the UNIFIED schedule autotuner (`--tcrv-rvv-materialize-schedule`, auto-discovery via `TunableScheduleOpInterface`; NO per-q1_0 pass). The legality threshold is `getRVVStripVLMAXElements(anchor, 8, minimum_vlen) >= 32` — the SAME single-source-of-truth formula the op verifier recomputes legality from. Stamped: `integer_core_lmul` + the SEMANTIC `minimum_vlen` (128/256).
- The emitter/verifier DEFAULT anchor is **m2** (the VLEN-universal-safe floor: e8m2 VLMAX 32 spans the sub-block at every VLEN), so an attr-less op verifies + lowers correctly and the gearbox is free to refine m2->m1 at VLEN>=256. The explicit aggressive `m1@minimum_vlen=128` is REJECTED fail-closed (I7) — the silent-wrong VLEN guard.

## Board gate (rvv, VLEN128, g++ 14.2 -O3 -march=rv64gcv, taskset -c 0)
```
AGREEMENT max_rel_norm=0.000e+00 nonzero=1 nonfinite=0
RESULT ours 10808.2
RESULT ggml(real,vl128) 14915.0
RATIO ggml/ours 1.380
```
- **byte-exact** (max_rel 0.000e+00, 8 seed × 6 size).
- **1.38× WIN** vs ggml's shipped `ggml_vec_dot_q1_0_q8_0_vl128` (was 0.033× = ~30× LOSS).

## byte-exact note (a CONSCIOUS tradeoff, recorded)
The new emit negates in the i8 domain (matching ggml `_vl128` exactly), so q1_0's -128 boundary is now ggml-IDENTICAL instead of the old ggml-superset (the old i16-widen path was "more correct" at -128). The real q8_0 quant domain is [-127,127] (-128 never occurs), so the gate is unaffected; this trades the unreachable -128 robustness for byte-exact-by-construction (same ops as ggml). Not a gate failure.

## VLEN256 = BOARD-PENDING (k1)
rvv is VLEN128 only. VLEN256 byte-exact needs the k1 board (not run this session). The two-VLEN LOGIC is verified both ways via the divergence lit (the VLEN256 emit is byte-DIFFERENT from VLEN128 = non-NULL: `vlm_v_b8`/`vsetvl_e8m1`/`vle8_v_i8m1`/`vwredsum_vs_i8m1_i16m1` vs the m2 spellings).

## Files
- Emitter: `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (`emitQ1_0Q8_0BlockDot`, ~line 7700-8090).
- Verifier + interface impl: `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (`GgmlBlockDotQ10Q80Op::verify` ~4423, `getScheduleKernelKey`/`isSchedulePinned` ~920).
- ODS: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (q1_0 op: +`TunableScheduleOpInterface`, +`minimum_vlen` attr, updated description).
- Descriptor: `lib/Plugin/RVV/RVVScheduleDescriptorRegistry.cpp` (custom single-knob `"q1_0"` descriptor + `enumerateRVVQ10ShapeCandidates`).
- Lits: `test/Conversion/RVV/rvv-to-emitc-q1-0-q8-0-block-dot.mlir`, `test/Conversion/RVV/rvv-q1-0-q8-0-block-dot-autotuner-divergence.mlir` (new), `test/Dialect/RVV/q1-0-q8-0-block-dot-dataflow.mlir`.
- Gate harness: `rvv-fill-q41-q10/{q10_harness.cpp, q10_ggml_ref.cpp, q10_ours_m2.cpp}`.

## Regression
208/208 RVV lit pass; shape-selection + low-precision-lmul gtests pass; the 3 bricks (q8_0 + FP4 + Track B selection/lit) unchanged.
