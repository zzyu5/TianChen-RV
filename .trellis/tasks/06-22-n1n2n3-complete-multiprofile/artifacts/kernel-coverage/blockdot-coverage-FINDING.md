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

## Status — ALL 5 DONE
- q5_0: DONE. q5_1: DONE. iq4_nl: DONE. tq2_0: DONE. tq1_0: DONE.

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
