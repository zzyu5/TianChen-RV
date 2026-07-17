# Block-dot single-kernel coverage — OUR emitted kernel vs ggml's shipped RVV kernel

**Goal**: close the "single-kernel test for ALL kernels" gap. Several block-dot
kernels we EMIT had never been (a) verified correct or (b) compared to ggml's own
shipped RVV kernel. This file banks BOTH per kernel, checkpointed as each finishes.

**Provenance**: `ssh rvv` (Sophgo SG2044, RVV1.0, **VLEN=128**, zvfh). Compiler:
clang-18 (`-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`), both TUs identical
flags, single binary. taskset-pinned (`taskset -c 2`). Best-of-`reps` min,
`timing_n` super-blocks/call.

**Method (kernel-isolated microbench)**: OUR block-dot is emitted via
`build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<q>-block-dot.mlir
--tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (run on the dev host).
The ggml baseline is the **shipped RVV kernel** for THIS host's VLEN — the
`_vl128` variant (or the `vlenb==16` branch of the unified `#if __riscv_v` body for
q5_0/q5_1) — lifted verbatim from
`/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c`, expressed
against raw byte offsets + a 4-arg ABI adapter `kern_ggml(size_t,float*,const
uint8_t*,const uint8_t*)`. Both fed identical random super-block bytes. Agreement is
the **relative norm** `max|ours-ggml|/|ggml|` over 8 seeds x 6 sizes.

**Honest label**: these block-dots are the SAME algorithm as ggml's vec_dot (our
emit vs ggml's emit). This is a **maturity + micro-perf** comparison, NOT a Win-B
algorithm change. ratio = ggml/ours; ratio>1 ⇒ OURS faster, ratio<1 ⇒ ours slower
(a loss, reported honestly).

## Results

| kernel | ggml variant | correctness (rel-norm) | ours ns/call | ggml ns/call | ratio (ggml/ours) | verdict |
|--------|--------------|-----------------------:|-------------:|-------------:|------------------:|---------|
| q5_0 | unified `__riscv_v` (vlenb==16) | 0.000e+00 (bit-exact) | 5040.0 | 1303.8 | 0.259 | LOSS (ggml 3.87x faster) |
| q5_1 | unified `__riscv_v` (vlenb==16) | 0.000e+00 (bit-exact) | 4448.0 | 1305.7 | 0.294 | LOSS (ggml 3.41x faster) |
| iq4_nl | `_vl128` | 0.000e+00 (bit-exact) | 760.7 | 1006.6 | 1.323 | WIN (ours 1.32x faster) |
| tq2_0 | `_vl128` | 0.000e+00 (bit-exact) | 3197.4 | 1925.3 | 0.602 | LOSS (ggml 1.66x faster) |
| tq1_0 | `_vl128` | 1.525e-05 (fp-fold-order; 0.0 matched-assoc) | 6705.5 | 2957.8 | 0.441 | LOSS (ggml 2.27x faster) |

## Per-kernel detail

### q5_0 (`q5_0_q8_0_block_dot` vs `ggml_vec_dot_q5_0_q8_0`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes (32..2048 elems),
  nonzero=1, nonfinite=0. EXACT match — integer decode + single per-block fp fold
  `(d_x*d_y)*sumi` is identical to ggml's order, so no rounding divergence at all.
  OUR q5_0 emit (qh 5th-bit offset-binary decode, nibble unpack, vwredsum) is a
  verified-correct drop-in.
- PERF: ours 5040.0 ns / ggml 1303.8 ns, ratio 0.259 → **LOSS, ggml ~3.87x faster**.
  ggml's vl128 path does one fused i16m4 widening + single vwredsum over the whole
  32-lane block; ours emits a narrower strip shape — correct but not LMUL-tuned
  (the N3/Gearbox gap).
- timing_n=2048 (64 super-blocks), iters=4000, reps=200, taskset -c 2.

### q5_1 (`q5_1_q8_1_block_dot` vs `ggml_vec_dot_q5_1_q8_1`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1,
  nonfinite=0. EXACT. q5_1 decode (5th-bit `vor` injection — no inversion, unlike
  q5_0's `vmnand`+`vsub`) plus the dual-term fold `(d_x*d_y)*sumi + x.m*y.s` matches
  ggml bit-for-bit. Verified-correct drop-in.
- PERF: ours 4448.0 ns / ggml 1305.7 ns, ratio 0.294 → **LOSS, ggml ~3.41x faster**.
  Same shape story as q5_0.
- timing_n=2048, iters=4000, reps=200, taskset -c 2.

### iq4_nl (`iq4_nl_q8_0_block_dot` vs `ggml_vec_dot_iq4_nl_q8_0`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes (EVEN nb only — see
  caveat), nonzero=1, nonfinite=0. EXACT. OUR FP4-codebook gather decode
  (`tcrv_iq4_nl_kvalues` table identical to ggml's `kvalues_iq4nl`, nibble unpack +
  `vrgather` + `vwredsum`, fold `(d_x*d_y)*acc`) is a verified-correct drop-in.
- PERF: ours 760.7 ns / ggml 1006.6 ns, ratio 1.323 → **WIN, ours ~1.32x faster**.
  This is the predicted gather-heavy win: ggml's `_vl128` does full i8m2 32-lane
  `vrgather` (2-blocks/iter); OURS emits the vl256-style split lo/hi i8m1 16-lane
  gathers (one block/iter), which map better onto VLEN128's narrower register file
  — a cheaper gather shape wins here, same as q2_K did on K1.
- CAVEAT (honest): ggml's `_vl128` loop is `ib+1 < nb` (2 blocks/iter) and SILENTLY
  DROPS the last block when nb is ODD. OURS loops all nb one-at-a-time. The harness
  uses EVEN-nb sizes exclusively so both cover identical blocks; the 0.0 agreement
  is therefore apples-to-apples. (ggml's odd-nb drop is a ggml-side limitation, not
  an emitter concern of ours.)
- timing_n=2048 (nb=64, even), iters=4000, reps=200, taskset -c 2.

### tq2_0 (`tq2_0_q8_k_block_dot` vs `ggml_vec_dot_tq2_0_q8_K`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes (256..8192 elems),
  nonzero=1, nonfinite=0. EXACT. OUR ternary 2-bit decode (4 bit-field unpacks
  `vand 0x03` + `vsub 1` per byte, `vwmacc` against q8, integer `vwredsum`, then the
  scalar fold `sumi*(y.d_float32 * fp16(x.d))`) matches ggml bit-for-bit. q8_K's
  activation `d` is float32 (NOT the fp16 weight d) — both kernels read it as
  float32@0; bsums are present but unused by tq2_0. Verified-correct drop-in.
- PERF: ours 3197.4 ns / ggml 1925.3 ns, ratio 0.602 → **LOSS, ggml ~1.66x faster**.
  ggml's `_vl128` accumulates at i16m4 (32-lane) over 2 sub-blocks/iter; OURS emits
  16-lane m1 over 16 sub-blocks — correct but a narrower, more-iteration shape (the
  N3/Gearbox LMUL-tune gap).
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### tq1_0 (`tq1_0_q8_k_block_dot` vs `ggml_vec_dot_tq1_0_q8_K`) — DONE
- CORRECTNESS: rel-norm = **1.525e-05** vs *verbatim* ggml (finite, nonzero), and
  **0.000e+00** against a fp-fold-association-matched ggml_ref (diagnostic). The
  integer base-3 ternary decode — incl. the 3 sub-loops (i16m4 32-lane, i16m2
  16-lane, and the `qh` base-3 third loop with `pow[16]` broadcast) — is BIT-EXACT.
  PROOF: ggml's tq1_0 (quants.c:6068) folds `(sumi·y.d)·x.d` while OURS (and ggml's
  OWN tq2_0 at :6375) fold `sumi·(y.d·x.d)` — a single fp32 multiply reassociation.
  Matching ggml_ref's association to ours collapses the norm to exactly 0.0,
  isolating the 1.5e-5 to fp reassociation amplified by cross-block sign
  cancellation in `sumf` (max over nb=32) — NOT an integer-decode bug. (Also
  confirmed ours reads y.d as float32 + x.d as fp16 correctly, so no d-field
  misread.) Verified-correct drop-in; same fp-fold-order story as the K-quant
  FINDING, surfaced here only because ggml's two ternary kernels disagree on assoc.
- PERF: ours 6705.5 ns / ggml 2957.8 ns, ratio 0.441 → **LOSS, ggml ~2.27x faster**.
  ggml's `_vl128` runs the bulk at i16m4 (32-lane) with the base-3 digits unrolled;
  OURS emits narrower m1/m2 sub-loops — correct but not LMUL/shape-tuned (N3 gap).
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

## IQ-quant coverage (gather-heavy hypothesis test)

**Hypothesis under test**: IQ-quants are gather-heavy (codebook lookups), and iq4_nl
WON 1.32x on that gather axis (our split 16-lane `vrgather` beat ggml's 32-lane) — so
iq2/iq3 may also WIN. **Method identical** to the 5 above: OUR emit (`tcrv-opt … |
mlir-translate-20`) vs ggml's SHIPPED `_vl128` kernel (each has `_vl128`/`_vl256`
variants; VLEN128 host → `_vl128` is the shipped path) lifted verbatim from
`arch/riscv/quants.c` with grids from `ggml-common.h` (`GGML_COMMON_IMPL_C`) +
`keven_signs_q2xs[1024]` lifted from quants.c:4235 (NOT in ggml-common.h). All q8_K
(292B: d@0 float32, qs[256]@4, bsums[16]@260); WB per-struct. Sizes multiples of
QK_K=256; loop is one-super-block-at-a-time → no tail-drop caveat.

**KEY FINDING — hypothesis does NOT transfer.** iq4_nl won via *register* `vrgather`
(split lo/hi 16-lane). But our iq2/iq3 emits use **scalar** table indexing
(`tcrv_iqXXX_ksigns[idx]` + grid byte-view, 0 `vrgather`/`vluxei`), whereas ggml's
`_vl128` uses hardware `vluxei16` **indexed vector loads** from grid64/signs64. The win
*mechanism* (register vrgather) is absent from our IQ grid-codebook emits, so
"gather-heavy ⇒ win" tests a primitive that isn't present — and ours LOSES.

| kernel | ggml variant | correctness (rel-norm) | ours ns/call | ggml ns/call | ratio (ggml/ours) | verdict |
|--------|--------------|-----------------------:|-------------:|-------------:|------------------:|---------|
| iq2_xxs | `_vl128` | 0.000e+00 (bit-exact) | 27552.1 | 4026.8 | 0.146 | LOSS (ggml 6.85x faster) |
| iq3_xxs | `_vl128` | 0.000e+00 (bit-exact) | 123930.7 | 5511.0 | 0.044 | LOSS (ggml 22.5x faster) |
| iq2_xs | `_vl128` | 0.000e+00 (bit-exact) | 29247.4 | 3324.1 | 0.114 | LOSS (ggml 8.80x faster) |
| iq2_s | `_vl128` | 0.000e+00 (bit-exact) | 38496.0 | 7455.6 | 0.194 | LOSS (ggml 5.15x faster) |
| iq3_s | `_vl128` | 0.000e+00 (bit-exact) | 46982.4 | 7548.7 | 0.161 | LOSS (ggml 6.22x faster) |
| iq1_s | `_vl128` | 0.000e+00 (bit-exact) | 24840.7 | 3412.3 | 0.137 | LOSS (ggml 7.28x faster) |
| iq1_m | `_vl128` | 0.000e+00 (bit-exact) | 58930.5 | 6151.5 | 0.104 | LOSS (ggml 9.58x faster) |

### iq2_xxs (`iq2_xxs_q8_k_block_dot` vs `ggml_vec_dot_iq2_xxs_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes (256..8192 elems),
  nonzero=1, nonfinite=0. EXACT. OUR 2-bit grid-codebook decode (grid byte-view +
  scalar `tcrv_iq2xxs_ksigns[128]` sign-table + `tcrv_iq2xxs_kmask[8]`, `0.125f` final
  scale, fold `sum*(fp16(x.d)*y.d_f32)`) matches ggml bit-for-bit. Verified-correct drop-in.
- PERF: ours 27552.1 ns / ggml 4026.8 ns, ratio 0.146 → **LOSS, ggml ~6.85x faster.**
  GATHER-PRIMITIVE MISMATCH (the crux): ggml's `_vl128` gathers the 64-bit grid AND
  sign entries with hardware `__riscv_vluxei16_v_u64m2` indexed vector loads (4-at-a-time
  per sub-block); OURS emits **0** `vluxei`/`vrgather` and does the grid+sign lookup with
  SCALAR per-element `ksigns[idx]` indexing — a serial scalar gather. So this is NOT the
  iq4_nl register-vrgather shape; the hypothesized win primitive is simply not present.
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### iq3_xxs (`iq3_xxs_q8_k_block_dot` vs `ggml_vec_dot_iq3_xxs_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1, nonfinite=0.
  EXACT. OUR 3-bit grid-codebook decode (u32 grid byte-view + scalar `tcrv_iq3xxs_ksigns`
  sign-table, metadata at qs+64, `0.25f` final scale) matches ggml bit-for-bit.
- PERF: ours 123930.7 ns / ggml 5511.0 ns, ratio 0.044 → **LOSS, ggml ~22.5x faster.**
  WORST of the IQ set, same gather-primitive mismatch as iq2_xxs but amplified: ggml's
  `_vl128` does `__riscv_vluxei16_v_u32m4` grid gather + `vrgather_vv` aux-expand +
  `vluxei16` sign gather (16+8 lanes/iter, m4/m8 LMUL); OURS emits **0** vluxei/vrgather
  and serializes the grid+sign lookup scalar per-element. Confirms the hypothesis fails
  hardest where the grid is largest/most-gathered.
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### iq2_xs (`iq2_xs_q8_k_block_dot` vs `ggml_vec_dot_iq2_xs_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1, nonfinite=0.
  EXACT. OUR decode (iq2xs_grid[512] byte-view + scalar ksigns, per-u16 index split to
  grid `&511` + sign `>>9`, `scales[8]` 4-bit pair, `0.125f` final) matches ggml.
- PERF: ours 29247.4 / ggml 3324.1, ratio 0.114 → **LOSS, ggml ~8.80x faster.** Same
  gather-primitive mismatch: ggml `_vl128` uses `__riscv_vluxei16_v_u64m4` grid+sign
  gather (8 lanes/iter, m4/m8); OURS emits **0** vluxei/vrgather (scalar tables).
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### iq2_s (`iq2_s_q8_k_block_dot` vs `ggml_vec_dot_iq2_s_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1, nonfinite=0.
  EXACT. OUR decode (iq2s_grid[1024], 10-bit index from qs-low + qh-high `0x1800` OR,
  EMBEDDED signs at qs+32 broadcast per-bit, qh@66/scales@74, `0.125f`) matches ggml.
- PERF: ours 38496.0 / ggml 7455.6, ratio 0.194 → **LOSS, ggml ~5.15x faster.** ggml
  `_vl128` uses `vluxei16` grid gather + `vrgather_vv` sign-broadcast (m2 LMUL); OURS
  emits **0** vluxei/vrgather (scalar). Best ratio of the IQ losses (still a loss) —
  iq2_s has the most non-gather vector work (sign-mask path), diluting the scalar penalty.
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### iq3_s (`iq3_s_q8_k_block_dot` vs `ggml_vec_dot_iq3_s_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1, nonfinite=0.
  EXACT. OUR decode (iq3s_grid[512] u32, 11-bit index `qs<<2 | qh-bit<<10`, EMBEDDED
  signs@74 broadcast-per-bit, qh@66/scales@106, plain `sumf` final — no 0.125/0.25)
  matches ggml bit-for-bit. NOTE: iq3_s d-fold is `fp16(x.d)` then `* y.d` (combined),
  ours reads x.d fp16 + y.d f32 correctly.
- PERF: ours 46982.4 / ggml 7548.7, ratio 0.161 → **LOSS, ggml ~6.22x faster.** ggml
  `_vl128` uses `vluxei16` grid gather + `vrgather_vv` sign-broadcast + `vwmulsu` (m2/m4);
  OURS emits **0** vluxei/vrgather (scalar tables).
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### iq1_s (`iq1_s_q8_k_block_dot` vs `ggml_vec_dot_iq1_s_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1, nonfinite=0.
  EXACT. OUR decode (iq1s_grid, qh@34 u16 → ls/delta + per-sub-block grid index, USES
  y.bsums delta term, fold `fp16(x.d)*y.d*(sumi + IQ1S_DELTA*sumi1)`, IQ1S_DELTA=0.125)
  matches ggml bit-for-bit. NOTE: iq1_s is the one IQ kernel that consumes y.bsums; the
  harness builds bsums consistent with q8 so both sides agree.
- PERF: ours 24840.7 / ggml 3412.3, ratio 0.137 → **LOSS, ggml ~7.28x faster.** ggml
  `_vl128` uses `vluxei16` i64m4 grid gather (8/iter) + `vrgather_vv` qh-index; OURS
  emits **0** vluxei/vrgather (scalar tables).
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

### iq1_m (`iq1_m_q8_k_block_dot` vs `ggml_vec_dot_iq1_m_q8_K_vl128`) — DONE
- CORRECTNESS: **rel-norm = 0.000e+00** over 8 seeds x 6 sizes, nonzero=1, nonfinite=0.
  EXACT. OUR decode (iq1s_grid, NO d field — scale fp16 RECONSTRUCTED from 4 u16 in
  scales[8]@48 via `iq1m_scale_t`, delta-merge sign, `vwmacc` accumulate, fold
  `y.d*fp16(scale)*(sumi1 + IQ1M_DELTA*sumi2)`, IQ1M_DELTA=0.125) matches ggml. Harness
  clamps sc[3] top nibble so the reconstructed scale stays finite (else random→NaN/Inf).
  ggml_ref needs BOTH `GGML_COMMON_DECL_C` (for `ggml_half`/`iq1m_scale_t` types) AND
  `GGML_COMMON_IMPL_C` (for iq1s_grid) — DECL alone or IMPL alone won't compile.
- PERF: ours 58930.5 / ggml 6151.5, ratio 0.104 → **LOSS, ggml ~9.58x faster.** ggml
  `_vl128` uses `vluxei16` u64m4 grid gather; OURS emits **0** vluxei/vrgather (scalar).
- timing_n=8192 (nb=32), iters=2000, reps=200, taskset -c 2.

## IQ SUMMARY — gather-heavy WIN hypothesis REJECTED for iq2/iq3 (and iq1)
- COVERAGE: all 7 IQ kernels (iq2_xxs, iq3_xxs, iq2_xs, iq2_s, iq3_s, iq1_s, iq1_m) EMIT
  cleanly (no emitter coverage gap) AND are numerically verified vs ggml's shipped
  `_vl128` RVV kernel on real VLEN128 hardware. **No emitter bug found in any.**
- CORRECTNESS (the maturity gate): **ALL 7 hit rel-norm 0.000e+00** (bit-exact) — every
  integer grid-codebook decode, embedded/keven sign path, bsums delta (iq1_s),
  reconstructed-scale fold (iq1_m), and the per-kernel fp fold matches ggml exactly.
- PERF: **0 wins / 7 losses.** Ratios 0.044–0.194 (ggml 5.15x–22.5x faster). Sorted
  best→worst for us: iq2_s 0.194, iq3_s 0.161, iq2_xxs 0.146, iq1_s 0.137, iq2_xs 0.114,
  iq1_m 0.104, iq3_xxs 0.044.
- **WHY THE HYPOTHESIS FAILS (the crux finding):** iq4_nl WON because its gather is a
  *register* `vrgather` over a tiny 16-entry FP4 codebook, where our split lo/hi 16-lane
  shape beat ggml's 32-lane. But iq2/iq3/iq1 gather from LARGE 64-bit grids
  (iq2xxs/iq2xs/iq2s/iq3xxs/iq3s/iq1s grids, 256–1024 entries) — ggml's `_vl128` does this
  with hardware `__riscv_vluxei16` **indexed vector memory loads** (+ `vrgather_vv` for
  sign broadcast). OUR emitter lowers these to **0 `vluxei`/`vrgather`** — it does the
  grid+sign lookups with SCALAR per-element table indexing (`tcrv_iqXXX_ksigns[idx]` +
  grid byte-view). So "gather-heavy ⇒ WIN" tested a primitive (register vrgather over a
  small codebook) that is simply ABSENT from our large-grid IQ emits; the actual primitive
  there is a serial scalar gather, which loses badly (worse the larger/more-gathered the
  grid → iq3_xxs worst at 22.5x). The iq4_nl win does NOT generalize to iq2/iq3/iq1.
- N3/GEARBOX IMPLICATION: the gap is a capability/lowering one — to be competitive on the
  IQ grid-codebooks the emitter must lower the grid/sign lookup to `vluxei16` indexed
  vector loads (and `vrgather` sign broadcast), not scalar table indexing. This is a
  concrete, named emitter target, not a tuning knob.

## Status — 5 base DONE; IQ ALL 7 DONE
- q5_0: DONE. q5_1: DONE. iq4_nl: DONE. tq2_0: DONE. tq1_0: DONE.
- IQ: iq2_xxs DONE. iq3_xxs DONE. iq2_xs DONE. iq2_s DONE. iq3_s DONE. iq1_s DONE. iq1_m DONE.

## Summary (all 5 block-dot kernels covered)
- COVERAGE: all 5 target kernels EMIT cleanly (no emitter coverage gap) AND are now
  numerically verified vs ggml's shipped RVV kernel on real VLEN128 hardware.
- CORRECTNESS (the maturity gate): EVERY kernel's integer core is bit-exact to
  ggml. 4 of 5 hit rel-norm 0.000e+00 outright (q5_0, q5_1, iq4_nl, tq2_0). tq1_0
  is 1.5e-5 vs verbatim ggml but 0.0 under matched fp-fold association — a proven
  fp32-reassociation, not a decode bug. NO emitter bug found in any kernel.
- PERF (honest, mixed; same-algorithm our-emit vs ggml-emit — a maturity comparison,
  NOT a Win-B algorithm change):
  - WIN: iq4_nl (1.323x — ours faster). Gather-heavy FP4 codebook: ours' split
    lo/hi i8m1 16-lane `vrgather` beats ggml's i8m2 32-lane gather on VLEN128.
  - LOSS: q5_0 (0.259x), q5_1 (0.294x), tq2_0 (0.602x), tq1_0 (0.441x). ggml's
    `_vl128` paths use wider LMUL (m2/m4) fused shapes; OUR emit is the correct
    decode at a narrower single LMUL shape — exactly the N3/Gearbox capability/
    resource-aware LMUL-tune gap, mirroring the K-quant FINDING (where only the
    gather-heavy q2_K won and the rest lost to hand-tuned `_vl256`).
- Pattern: OUR emitter is a verified-correct drop-in for all five; it wins where a
  cheaper/narrower vector shape suits the gather (iq4_nl, q2_K) and loses where
  ggml's hand-tuned wide-LMUL fold dominates (q5_*, tq*, most K-quants).

## Harness provenance / reproduce
- Bench sources: `artifacts/kernel-coverage/blockdot-bench/<quant>/{ours.cpp,
  ggml_ref.cpp,harness.cpp}` (+ `tq1_0/ggml_ref_matched.cpp` diagnostic).
- ours.cpp = `build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<q>-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (dev host).
- ggml_ref.cpp = the `_vl128` body (or `vlenb==16` branch of the unified
  `#if __riscv_v` body for q5_0/q5_1) lifted verbatim from quants.c, re-expressed
  against raw byte offsets + a 4-arg ABI adapter `kern_ggml`.
- On rvv: `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast harness.cpp
  ours.cpp ggml_ref.cpp -o bench && taskset -c 2 ./bench`. (binaries live in
  `~/blockdot-cov/<quant>/` on rvv.)
- Agreement over 8 seeds x 6 sizes; timing best-of-200-min, iters 2000-4000.
- iq4_nl caveat: ggml's `_vl128` is 2-blocks/iter (`ib+1<nb`) and drops the tail on
  ODD nb; harness uses EVEN-nb sizes so the comparison covers identical blocks.
- EMITTABILITY: all 5 target kernels emit cleanly (tcrv-opt + mlir-translate-20),
  NO emitter-level coverage gap. ours.cpp line counts: q5_0=124, q5_1=126,
  iq4_nl=80, tq2_0=498, tq1_0=692.
