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

---

# q4_K x q8_K 16x1-REPACKED PREFILL GEMM — numeric oracle (M=4 rows, mf2/VLEN128) — FINDING

**Date:** 2026-06-24. **Provenance:** `ssh rvv` (Sophgo SG2044, RVV1.0, **VLEN128**, 64 cores),
clang18 (`-O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`), `taskset -c 0`.
This is the GATING correctness step for the q4_K repack **GEMM** emitter
`tcrv_rvv.repack_gemm_q4_K_q8_K` (committed on main `02c41f4d`, structurally complete but
numerically UNVERIFIED). Methodology per `N3-METHODOLOGY.md`: this is a CORRECTNESS check (norm).

## VERDICT — PASS (correct). The q4_K GEMM emitter is now NUMERICALLY VALIDATED.

The compiler-emitted `tcrv_emitc_..._ggml_repack_gemm_q4_K_q8_K` matches an INDEPENDENT scalar
q4_K dequant-MATMUL reference (M=4 activation rows × nc weight columns) within fp32 rounding across
**all 8 shapes**. **WORST_NORM = 7.05e-7**, bar = norm < 1e-4. This upgrades the GEMM from
"structurally complete, numerically UNVERIFIED" to "**correct**." No kernel code modified; the kernel
is the frozen on-main emit (`tcrv-opt … --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`).

| shape (nc × n) | max_abs_err | rms(ref) | **norm** | rel | verdict |
|---|---|---|---|---|---|
| 16 × 256    | 1.526e-05 | 6.419e+01 | **2.377e-07** | 3.87e-07 | PASS |
| 16 × 4096   | 1.221e-04 | 3.049e+02 | **4.004e-07** | 2.58e-06 | PASS |
| 16 × 11008  | 4.272e-04 | 6.750e+02 | **6.329e-07** | 1.33e-05 | PASS |
| 32 × 256    | 1.526e-05 | 1.098e+02 | **1.389e-07** | 1.72e-07 | PASS |
| 32 × 4096   | 1.221e-04 | 2.696e+02 | **4.527e-07** | 4.41e-06 | PASS |
| 256 × 256   | 3.052e-05 | 7.859e+01 | **3.883e-07** | 1.16e-05 | PASS |
| 256 × 4096  | 1.831e-04 | 3.794e+02 | **4.826e-07** | 2.84e-04 | PASS |
| 256 × 11008 | 3.662e-04 | 5.198e+02 | **7.045e-07** | 6.37e-05 | PASS |

Norm sits at ~1e-7 (same regime as the q4_K GEVM oracle ~7e-7) — the integer dot is exact; the only
fp deltas are the fp16 d/dmin scale strips + the f32 block-fold vs the reference's double accumulation.
`out` zero-init per trial → an unwritten element reads 0 vs ref ~O(60–680) and would spike the norm;
norm ~1e-7 confirms the FULL M×nc output is correct, not spot-checked.

## Oracle framing (independent reference; the GEMM-NEW axis = the 4-row interleave)

Reference = INDEPENDENT scalar dequant-matmul from the **ORIGINAL pre-repack `block_q4_K`** weights,
dotted PER-ROW with the per-row `block_q8_K` int8 quants (`−dmin·m` baked into each dequantized weight
float, so the min/bsums correction is tested implicitly + exactly). The kernel reads the **INTERLEAVED
`block_q8_Kx4`** activation (stride 1168); the ref reads the 4 per-row `block_q8_K` **directly and never
de-interleaves** → a kernel de-interleave bug cannot agree. The decisive new axis vs the GEVM oracle is
the **4-ROW INTERLEAVE**: we build 4 DISTINCT per-row `block_q8_K` (distinct d + DC/qs per row) and
interleave into `block_q8_Kx4` via ggml's VERBATIM index math (`ggml_quantize_mat_q8_K_4x1`,
repack.cpp:90: `m=j%4, e=j/4; qs_x4[j]=row[m].qs[e]; index=((j>>6)<<2)+(j&3); bsums_x4[index]+=…`).
The byte ABI is exactly the FINDING handoff: act-qs `al+16+gsub*128+i*4+m`, act-bsums
`al+1040+(gsub*8+m)*2 / +(gsub*8+m+4)*2`, act-d `al+m*4` (fp32), store `s[(y*4+m)*bs + x*16 + j]`.

## Why the PASS is a TRUSTWORTHY gate — THREE negative controls, all spike hard (shape 256×4096)

Each control perturbs the REFERENCE; the resulting norm-vs-kernel SPIKE proves the corresponding
kernel structure is genuinely exercised (the correct test would catch that class of kernel bug):

- **bsums diagnostic:** q8_Kx4 `mean|bsum| = 183.1, max|bsum| = 714` (nonzero → min term active).
- **NOMIN** (zero the min term in ref): norm **1.71e-1** → min/bsums path exercised at **354,886×** the
  true-ref norm. A min-term sign/index bug spikes at ~1e-1, far above the 1e-4 bar.
- **PERM** (rotate per-sub-block 6-bit scale +1 in ref): norm **1.93e0** → distinct-sub-block-scale
  pairing exercised at **4,005,994×**. A sub-block-offset ("every 8th element") bug spikes at ~2e0.
- **ROWROT** (ref dots activation row `(m+1)%4`) — **the GEMM-new control for the 4-row interleave**:
  norm **3.66e0** → the 4-row mapping is exercised at **7,578,336×**. A kernel that mis-mapped/swapped
  the interleaved activation rows would match this perturbed ref, not the true one; it doesn't.

So the ~1e-7 PASS is not "small because the test is weak" — the min term, the sub-block-scale teeth,
AND the load-bearing 4-row activation interleave (the one thing no build/lit catches) are all driven at
≥10⁵× margin, and the kernel still matches to fp-rounding.

## Oracle artifacts
- `q4_K-emit/oracle_gemm_q4K.cpp` — harness (independent scalar dequant-matmul ref, `make_block_q4_Kx16`
  + `make_block_q8_Kx4` verbatim interleave, 4-distinct-row builder, `--controls` mode w/ ROWROT).
- `q4_K-emit/k_gemm_q4K.cpp` — the compiler-emitted GEMM kernel under test (frozen on-main emit).
- On rvv `~/q4k-oracle-agent/`: same sources + `oracle_gemm_q4k` binary + `q4_K-gemm-oracle.log`.

---

# q4_K repack PREFILL GEMM — Win-B·micro (the decisive amortization test, vs ggml's REAL RVV kernel)

**Date:** 2026-06-24. **Provenance:** `ssh rvv` (Sophgo SG2044, RVV1.0, **VLEN128**, 64 cores),
clang18 (`-O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`), `taskset -c 0`, best-of-reps min,
3 runs. Per N3-METHODOLOGY: **Win-B baseline = ggml's OWN shipped RVV kernel, NOT scalar/naive.**

## THE BASELINE (methodology-correct — ggml's real VLEN128 prefill fallback)

At VLEN128 ggml does **NOT** route q4_K through its repack (`repack.cpp:4619` `case 128: break // TODO`
→ nullptr; the `q4_K_16x1_q8_K` repack is VLEN256-only). So prefill falls back to the plain per-row
block-dot `ggml_vec_dot_q4_K_q8_K`, which on this part is the hand-written RVV inline-asm
**`ggml_vec_dot_q4_K_q8_K_vl128`** (`quants.c:1770`). For a prefill GEMM (M=4 rows) ggml has no GEMM,
so it calls `_vl128` **M×nc times** — once per (row, output column) — re-streaming + re-decoding the
weights M=4× (lifted **verbatim**, reads the **ORIGINAL pre-repack `block_q4_K`** weights + per-row
`block_q8_K`, the same q4_K/q8_K values our GEMM consumes in the interleaved layout). Both sides produce
the **full M×nc output**; the agreement norm over the entire M×nc vector confirms a fair same-output
comparison (NOT two layouts of one bug agreeing — ours reads `block_q4_Kx16`+`block_q8_Kx4`, ggml reads
the originals + per-row q8_K).

## VERDICT — LOSS at all 4 shapes (reported honestly). The amortization does NOT win.

**The HYPOTHESIS — ours decodes each weight group's 6-bit unpack + nibbles ONCE and reuses it across
the 4 activation rows (ggml re-decodes M=4×), so the GEMM wins where the GEVM lost — DOES NOT
materialize on VLEN128.** ggml's `_vl128` is ~1.1–1.7× faster. ratio = ggml/ours (>1 ⇒ ours faster).

| shape (nc × n) | ours ns (min/3) | ggml `_vl128`×Mnc ns (min/3) | **ratio (ggml/ours)** | agreement norm | verdict |
|---|---:|---:|---:|---:|---|
| 16 × 4096   |   58138.3 |   51846.2 | **0.885–0.892** | 3.00e-07 | LOSS (ggml ~1.13× faster) |
| 16 × 11008  |  164340.6 |  143534.6 | **0.870–0.873** | 2.99e-07 | LOSS (ggml ~1.15× faster) |
| 256 × 4096  |  987729.9 |  813432.0 | **0.815–0.829** | 4.28e-07 | LOSS (ggml ~1.22× faster) |
| 256 × 11008 | 3501102.5 | 2239889.5 | **0.588–0.640** | 5.90e-07 | LOSS (ggml ~1.6× faster) |

**Ratio is the headline** — common-mode-cancelled, stable across 3 runs (the three smaller-ratio shapes
spread <2%; the 256×11008 ratio swings 0.588–0.640 as the shared 64-core box loads, but stays a clear
LOSS). Absolute ns vary with box load (e.g. 16×4096 ours swung 58k→98k across runs); the table reports
**min-across-3-runs** absolutes, ratio is the stable invariant. Agreement norm 3e-7…5.9e-7 = same
fp32-fold-order residual as the oracle — genuinely fair same-output comparison.

## What the number says (honest read — it is a NARROWER loss than the GEVM, not a win)

The amortization is **real but insufficient on VLEN128**. The GEMM ratio (0.59–0.89) is materially
BETTER than the q4_K GEVM Win-B loss (0.47–0.66, same hardware): the M=4 weight-decode + weight-traffic
amortization does close part of the gap — most at the small shapes (0.87–0.89 at nc=16) and least at
the large compute-bound shape (0.59 at 256×11008, where ggml's VLEN128-native hand-tuned asm pulls
furthest ahead). But it never crosses 1.0. Root cause is unchanged from the GEVM Win-B
(`q4_K-repack-oracle-FINDING.md`): our block-as-lane form is **VLEN256-shaped** (16 e32 column
accumulators want one vector register, which needs VLEN≥256); on VLEN128 RVV1.0 leaves
`integer_core_lmul` unset → the fractional **mf2 / half_lanes=8 2×8-lane-strip** path, which loses to
ggml's VLEN128-native `_vl128` (split hi/low nibble streams, m2 widening redsum, lane-extract scale
folds). This is an **N3 Gearbox capability/resource-aware-tune SHAPE-MISMATCH gap** (the repack-GEMM
*shape* is mismatched to VLEN128), **NOT a kernel-correctness problem** — the oracle PASS
(WORST_NORM 7.05e-7) stands and the Win-B agreement norm re-confirms correctness against ggml's real
kernel. Contrast q4_0, whose GEMM **won 6.36×**: q4_0's simpler block (no 6-bit super-block scale/min
unpack, no per-sub-block fold) maps cleanly to the VLEN128 strip; the q4_K super-block structure does
not. **Consistent with the campaign-central finding (memory): compute-bound kernel microbench wins
don't always materialize, and here the amortization that should help is throttled by the VLEN128 shape
mismatch.** Reported as a KERNEL MICROBENCH (not an e2e claim).

## Win-B·micro artifacts
- `q4_K-emit/winb_gemm_q4K.cpp` — Win-B harness: oracle data construction verbatim (distinct weights +
  4 distinct rows + `make_block_q4_Kx16`/`make_block_q8_Kx4` interleave), `ggml_vec_dot_q4_K_q8_K_vl128`
  lifted verbatim from `quants.c:1770` + an M×nc same-output GEMM loop over the originals, best-of-reps
  min timing of the full M×nc output for both sides.
- On rvv `~/q4k-oracle-agent/`: `winb_gemm_q4k` binary + `q4_K-gemm-winB-micro.log` (3-run canonical log).
- Built: `clang++-18 -O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast -c {winb_gemm_q4K,k_gemm_q4K}.cpp`
  then link → `winb_gemm_q4k`. (`restrict` in the lifted C body → `__restrict__` for C++; semantics identical.)
