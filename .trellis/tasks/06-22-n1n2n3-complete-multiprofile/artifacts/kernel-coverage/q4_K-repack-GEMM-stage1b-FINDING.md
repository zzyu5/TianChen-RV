# q4_K x q8_K 16x1-REPACKED PREFILL GEMM emitter — stage-1b — FINDING + rvv-oracle handoff

**Date:** 2026-06-24
**Scope:** the q4_K repack **GEMM** (prefill, M>>1) emitter `tcrv_rvv.repack_gemm_q4_K_q8_K`
(`GgmlRepackGemmQ4KQ8KOp`, op+verifier on main `a6fbc7cf`). The dominant-quant PREFILL e2e-win path:
the GEVM's 6-bit unpack + nibble decode amortized ONCE per 16-weight group and REUSED across the 4
interleaved activation columns (the GEVM single-column path loses; the GEMM amortization is the point).

## STATUS — structurally complete, **NUMERICALLY UNVERIFIED**

- **1b-i (mechanical, DELIVERED):** scaffold (q4_1 GEMM renamed) inserted, wired (dispatch entry +
  `isRepackGemmQ4KQ8KBody` recognizer + 2 decls), `*Min*→*Dmin*` / `*Sum*→*Bsums*` getter rename.
  Clean rebuild (RVVOps.cpp.inc + 3 .o + tcrv-opt) — **COMPILES + LINKS** (exit 0, tcrv-opt 55MB).
- **1b-iii (the real work, DONE — code shape):** scaffold body REPLACED with the q4_K graft — the
  GEVM's oracle-verified 8-sub-block 6-bit scale/min lane-wise unpack + dual d/dmin fold, wrapped in
  the q4_1 GEMM's nr→nc→strip→M-column shell. Weight-side unpack + nibble decode SHARED outside the
  column loop; per-(column,strip) `sumi`/`bsums`/`sumf` accumulators. Clean rebuild — COMPILES + LINKS.
- **Lit (DONE):** `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir` — CHECK + NOWALL PASS.
  Asserts the LOWERED STRUCTURE (8-sub-block 6-bit unpack vsll/vor/vzext, per-column fp32 act-d,
  per-column bsums-min fold, multi-column vfmacc/vfnmsac, NO vredsum). **The lit does NOT prove
  numerics** — it cannot catch a wrong qs/bsums column-interleave.
- **NOT done (deferred to rvv):** byte-exact numeric oracle. See handoff below.

## Structural self-check (emitted-IR counts — the multi-column amortization is real)

`tcrv-opt … --tcrv-rvv-lower-to-emitc` on the lit (VLEN128 / half_lanes=8 / numHalves=2 / 4 act cols).
NOTE: the conversion framework emits each op TWICE (benign legalize-then-finalize; the known-good q4_K
GEVM and q4_1 GEMM baselines show the same 2× route-marker doubling). Counts below are the RAW emitted
totals; divide by 2 for the per-(col,strip) logical shape.

| emitted intrinsic | raw count | logical (÷2) | expected shape |
|---|---:|---:|---|
| `*(const float *)` act-d reads | 4 | — | 4 cols × fp32 d (NOT fp16) |
| `*(const _Float16 *)` act reads | **0** | 0 | the q4_1-ism is fully killed |
| `*(const int8_t *)` act-qs reads | 2048 | 1024 | 256 elem × 4 cols (full q8_Kx4 qs) |
| `*(const int16_t *)` bsums reads | 128 | 64 | 8 sub × 2 pair × 4 cols |
| `vsll_vx_u8mf2` (6-bit unpack) | 32 | 16 | q4_1 GEMM had **0** → 8-sub-block extension landed |
| `vwmacc_vx_i16m1` (nibble dot) | 8192 | 4096 | shared-decode × 4 cols inner dot |
| `vwmacc_vv_i32m2` (scale fold) | 512 | 256 | per-sub-block 6-bit-scale-weighted promote |
| `vwmacc_vx_i32m2` (min fold) | 256 | 128 | per-(col,strip,sub) bsums-min |
| `vfmacc_vv_f32m2` (main fold) | 16 | 8 | 4 cols × 2 strips |
| `vfnmsac_vv_f32m2` (min subtract) | 16 | 8 | 4 cols × 2 strips |
| `vse32_v_f32m2` (store) | 16 | 8 | 4 cols × 2 strips |
| `redsum` | **0** | 0 | lane-wise wall-free (NOWALL) |

## rvv-oracle HANDOFF — the EXACT indices the byte-exact oracle must check

**Reference (now in-tree):** `ggml_gemm_q4_K_16x1_q8_K_generic`
(`/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/repack.cpp:2442`) consuming `block_q4_Kx16`
(weight) + `block_q8_Kx4` (activation, produced by `ggml_quantize_mat_q8_K_4x1`, `repack.cpp:90`).
Repack the SAME 16 original `block_q4_K` weights via `make_block_q4_Kx16` (`repack.cpp:2913`,
`blck_size_interleave==1`) and 4 activation rows via `ggml_quantize_mat_q8_K_4x1`; compare our emitted
kernel's `nr×nc` output to the generic ref (or to an independent scalar dequant-matmul like the GEVM
oracle did). **DO NOT re-derive the interleave — emit these literal byte forms and check THESE indices:**

Per weight-column lane `j` (0..15, lanes = strip h ∈ {0,1} × half 8), per activation column `m` (0..3),
global sub-block `gsub` (0..7), super-half `J = gsub/4`, local sub `sb = gsub%4`, position `i` (0..31):

- **act-d:** `d_y_m = *(const float*)(al + m*4)` — fp32, 4 deltas at byte 0/4/8/12 (NOT fp16).
- **act-qs (interleaved, element e of column m at qs[e*4+m], base 16):**
  - low nibble: `al + 16 + gsub*128 + i*4 + m`
  - high nibble: `al + 16 + gsub*128 + 128 + i*4 + m`
  (in emitter coords: `aLoBase = 16 + J*512 + sbLo*128`, `aHiBase = 16 + J*512 + sbHi*128`, `+ i*4 + m`.)
- **act-bsums (group16-major / column-minor, index = group16*4 + m, base 1040, int16):**
  pair for sub-block `gsub`, column `m`:
  `*(const int16_t*)(al + 1040 + (gsub*8 + m)*2)` + `*(const int16_t*)(al + 1040 + (gsub*8 + m + 4)*2)`
  (gsub spans groups 2*gsub and 2*gsub+1; group g16 col m = bsums[g16*4 + m]).
- **weight nibble:** `bl + 256 + gsub*256 + i*16 + j`  → low = `&0xF`, high = `>>4`
  (emitter: `qsPairBase = 256 + J*1024 + pair*512`, `+ i*16 + h*half`).
- **weight 6-bit scale/min** (SAME as GEVM, byte-identical weight ABI): lane-wise from `scales` region
  at +64; low byte `64 + J*64 + sb*16 + h*half`, high byte `64 + 128 + sb*16 + h*half`; ggml unpack
  (repack.cpp:2480-2490): `scales = lo&0x0F | hi-bits; mins = lo>>4 | hi-bits`.
- **store:** `s[(y*4 + m)*bs + x*16 + j]`.
- **math:** `out[m][col=x*16+j] = Σ_l ( d_x[j]·d_y_m·Σ_gsub scale_gsub[j]·sumi_gsub − dmin_x[j]·d_y_m·Σ_gsub min_gsub[j]·bsum_pair )`.

**The single load-bearing fact no build/lit catches = the act qs/bsums column-interleave above.** If the
rvv oracle finds garbage, check those byte formulas FIRST (esp. `gsub*8+m / gsub*8+m+4` bsums pair and
`i*4+m` qs stride) before re-deriving.

## Files
- `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` — `emitRepackGemmQ4KQ8K` (the graft).
- `lib/Conversion/RVV/RVVToEmitC.cpp` — dispatch entry + `isRepackGemmQ4KQ8KBody` recognizer.
- `lib/Conversion/RVV/RVVToEmitCInternal.h` — 2 decls.
- `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir` — structural lit (CHECK+NOWALL PASS).
