# k1 (SpacemiT X60, VLEN256) micro-bench fill — OUR emit vs ggml's SHIPPED `_vl256`

**Goal**: fill the self-check-table cells measured only on rvv (VLEN128), never on k1
(VLEN256): FP4 codebook (iq4_nl/iq4_xs/mxfp4), the 7 IQ-quants, and ternary
(tq2_0/tq1_0). Measure-only; no lib/ changes.

## Provenance / board
- `ssh k1`, SpacemiT X60, `vlenb=32 → VLEN=256` (probed with a 1-line binary, not
  inferred from the model name): `VLMAX e8m1=32, e8mf2=16`.
- clang-18 (Bianbu 18.1.8), `-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`, both TUs
  one binary, **`taskset -c 0-3`** (IME/harts 0-3). Build + bench on `/data` NVMe
  (110G free; never wrote `/`). **One bench at a time, serial** — never concurrent on
  the shared board.
- timing = best-of-`reps` min, same harnesses as the rvv coverage (8 seeds × 6 sizes
  agreement, then `timing_n` super-blocks/call).

## The methodology fix that makes this a FAIR fight (load-bearing)
The archived rvv `ggml_ref.cpp` files contain ggml's **`_vl128`** body. On k1 ggml's
runtime dispatch (`switch(__riscv_vlenb()*8){ case 128: vl128; case 256: vl256; }`)
selects **`_vl256`** for EVERY one of these quants (confirmed by grep on all 12). So the
correct, *shipped* k1 baseline is `_vl256`, NOT `_vl128`. Reusing the `_vl128` body on
k1 would run VLEN128-shaped code on a VLEN256 machine (artificially slow ggml) and
manufacture fake wins. **Every ggml_ref here was re-lifted to the `_vl256` body**
(verbatim intrinsics from `arch/riscv/quants.c`, reframed onto the same raw byte
offsets the `_vl128` refs used). OUR emitted kernel (`ours.cpp` via `tcrv-opt
--tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`) is identical regardless
of board.

**OUR emit is VLEN128-shaped by construction.** The FP4 codebook gather is hard-anchored
at m1 with `vsetvl_e8m1(16)` and is documented in the emitter as "inherently Zvl128b-gated"
(an N1 capability fact) — there is NO VLEN256 knob; the emit does not reshape for VLEN256.
At VLEN256 `vsetvl_e8m1(16)` caps active vl at 16 of VLMAX=32 → **half the m1 register
used** (sub-VLMAX). ggml's `_vl256` FP4 path uses `mf2` (VLMAX=16 = FULL register) +
2-blocks/iter packing. So every FP4/ternary k1 row is **our VLEN128-shaped emit vs ggml's
VLEN256-native emit** — a shape + maturity gap (a named gearbox target), not a clean
maturity number. Honest label, per the VLEN-SHAPE-MATCH discipline.

ratio = ggml_ns / ours_ns. >1 ⇒ OURS faster (WIN); <1 ⇒ ours slower (LOSS); ≈1 ⇒ TIE.

## Results — FP4 codebook + ternary (vs ggml `_vl256`)

| kernel | byte-exact vs `_vl256` | ours ns | ggml `_vl256` ns | ratio ggml/ours | verdict (k1) | rvv was |
|--------|------------------------|--------:|-----------------:|----------------:|--------------|---------|
| iq4_nl | **0.000e+00** | 2045.2 | 1718.1 | **0.840** | LOSS (ggml 1.19x) | WIN 1.32x |
| iq4_xs | int bit-exact (0.0 matched-assoc; 1.16e-5 fp-reassoc vs verbatim) | 8435.5 | 8775.9 | **1.040** (stable ×3) | **TIE / parity** | WIN 1.28x |
| mxfp4  | **0.000e+00** | 2077.1 | 1658.4 | **0.798** | LOSS (ggml 1.25x) | WIN 1.22x |
| tq2_0  | **0.000e+00** | 8770.4 | 3167.5 | **0.361** | LOSS (ggml 2.77x) | LOSS 1.66x |
| tq1_0  | 1.525e-05 (fp-reassoc; int bit-exact) | 13380.6 | 4861.4 | **0.363** | LOSS (ggml 2.75x) | LOSS 2.27x |

## Results — IQ-quants (vs ggml `_vl256`)

| kernel | byte-exact vs `_vl256` | ours ns | ggml `_vl256` ns | ratio ggml/ours | verdict (k1) | rvv was (OLD emit) |
|--------|------------------------|--------:|-----------------:|----------------:|--------------|--------------------|
| iq1_s   | **0.000e+00** | 16498.1 | 5966.4  | **0.362** | LOSS (ggml 2.77x) | LOSS 7.28x (0-vluxei) |
| iq1_m   | **0.000e+00** | 47601.7 | 11324.9 | **0.238** | LOSS (ggml 4.20x) | LOSS 9.58x (0-vluxei) |
| iq2_xxs | **0.000e+00** | 14755.2 | 9710.7  | **0.658** | LOSS (ggml 1.52x) | LOSS 6.85x (0-vluxei) |
| iq2_xs  | **0.000e+00** | 17599.5 | 9196.4  | **0.523** | LOSS (ggml 1.91x) | LOSS 8.80x (0-vluxei) |
| iq2_s   | **0.000e+00** | 17734.2 | 11701.4 | **0.660** | LOSS (ggml 1.52x) | LOSS 5.15x (0-vluxei) |
| iq3_xxs | **0.000e+00** | 56142.1 | 11047.6 | **0.197** | LOSS (ggml 5.08x) | LOSS 22.5x (0-vluxei) |
| iq3_s   | **0.000e+00** | 47020.6 | 9881.6  | **0.210** | LOSS (ggml 4.76x) | LOSS 6.22x (0-vluxei) |

All 7 IQ: byte-exact 0.000e+00, all LOSS. k1 ratios 0.197–0.660 (ggml `_vl256` 1.5x–5.1x
faster). Best→worst for us: iq2_s 0.660, iq2_xxs 0.658, iq2_xs 0.523, iq1_s 0.362, iq1_m
0.238, iq3_s 0.210, iq3_xxs 0.197. The xxs/3-bit grids (largest, most-gathered) lose
hardest; iq2 small-grid kernels are the narrowest. iq2_s/iq3_s embedded-sign lifts gated
clean (byte-exact), confirming the file-static sign-mask helper arrays were lifted correctly.

### KEY honesty flag on the IQ rows (NOT a clean board delta)
The archived rvv IQ column was measured on an OLDER emit that lowered the grid/sign
lookup to **0 `vluxei`/0 `vrgather`** (scalar table indexing) — that's why rvv showed
5–22x losses. **The current host emit has matured: it now emits real
`__riscv_vluxei16_v_i64m2` indexed vector gathers** (verified by grep on today's
`ours.cpp`: iq2_xxs 16 real vluxei calls, iq1_s 8, etc.). So the jump "iq2_xxs rvv 0.146
→ k1 0.658" is **dominated by emitter maturation (scalar-gather → vluxei16), NOT a pure
VLEN128→VLEN256 effect.** The k1 IQ rows reflect the current vluxei16 emit vs ggml
`_vl256`; they are honest k1 numbers but are NOT directly comparable to the archived rvv
IQ rows (different emit). A clean pure-VLEN delta would need an rvv re-run on the current
emit (separate, out of scope here). The IQ ~1.5x gap is a *different mechanism* from the
FP4 sub-VLMAX story: ggml `_vl256` runs its vluxei16 grid gather + dot at m1-32-lane
while ours gathers at i64m2 — a gather-LMUL/packing gap, not the FP4 vsetvl(16) waste.

## Per-kernel detail

### iq4_nl — LOSS 0.840 (flipped from rvv WIN)
ggml `_vl256` is `mf2` split lo/hi 16-lane gather, 2-blocks/iter — **the SAME gather
shape OURS used to WIN with on VLEN128, now native + better-packed on VLEN256**. Our emit
stays at `vsetvl_e8m1(16)` (sub-VLMAX on VLEN256). The 1.32x rvv WIN was a VLEN128 artifact
(our split-16 beat ggml's _vl128 i8m2-32); at VLEN256 ggml adopts our shape and wins on
packing. Byte-exact 0.0.

### iq4_xs — TIE 1.040 (parity; integer bit-exact)
ggml `_vl256` uses ONE wide `i8m4` 128-nibble gather + a `vrgatherei16` u64m4 reorder +
4-way `vget`/`vwredsum`. Integer decode bit-exact: 0.000e+00 vs a matched-association ref,
1.163e-05 vs verbatim ggml (pure fp32 reassoc — ggml folds the per-sub-block scale into
the integer accumulator then ONE fp mul/super-block; ours distributes the fold). Same
fp-reassoc story as rvv. Ratio a STABLE 1.040 across 3 runs → genuine TIE, ours marginally
faster. The only non-LOSS; rechecked per task discipline (任何 >1× 要复核).

### mxfp4 — LOSS 0.798 (flipped from rvv WIN)
Structurally identical to iq4_nl `_vl256` (mf2 split 16-lane, 2-blocks/iter) — only the
E8M0 scale decode differs. Byte-exact 0.0 (E8M0→fp32-half decode matches across both
denormal x<2 and normalized branches). Same shape story as iq4_nl.

### tq2_0 — LOSS 0.361 (wider than rvv's 1.66x)
ggml `_vl256` uses `e8m1` (32 lanes = FULL m1 at VLEN256) + `vwmacc_i16m2`, 32 weights/iter.
OUR emit is 16-lane m1 over more iterations — sub-VLMAX. The VLEN256 m1-32 shape pulls
FURTHER ahead than at VLEN128 (1.66x→2.77x). Byte-exact 0.0.

### tq1_0 — LOSS 0.363 (1.5e-5 fp-reassoc, int bit-exact)
ggml `_vl256` is 3 loops (m1-32 i16m2, mf2-16 i16m1, qh base-3 i16m1 w/ pow[16]).
rel-norm 1.525e-05 vs verbatim (IDENTICAL to rvv) — ggml folds (sumi·y.d)·x.d, ours folds
sumi·(y.d·x.d): pure fp32 reassoc, integer decode bit-exact (matched-assoc ref = 0.0 on
rvv; same construction here). Wider loss than rvv (2.27x→2.75x), same shape mechanism.

### IQ-quants (all 7) — LOSS 0.197–0.660 (vluxei16 emit; see IQ flag above)
Both sides now use real `vluxei16` indexed grid/sign gathers (this is the maturity case
the memory note flagged: "采用 ggml 自己的指令(vluxei16)预期是 parity/maturity 不是 beat-ggml").
The remaining gap is gather LMUL + packing: ggml `_vl256` gathers grid64/signs64 at
`vluxei16_v_u64m1`/`u64m2`/`u32m2` tuned to VLEN256's 32-lane m1, with the dot at
m1-32-lane (iq2_xxs/iq2_s) or wider; OUR emit gathers at `vluxei16_v_i64m2` with more
per-super-block dispatch overhead. All 7 byte-exact 0.000e+00 (grid decode, keven_signs64
OR embedded-sign broadcast, 0.125f/0.25f/plain final scale all match). NO emitter bug.
- iq2_s 0.660 / iq2_xxs 0.658: smallest gap — small grids, most non-gather vector work.
- iq2_xs 0.523, iq1_s 0.362, iq1_m 0.238 (scale reconstructed from 4×u16, no d field).
- iq3_s 0.210 / iq3_xxs 0.197: worst — the 3-bit grids are largest/most-gathered, and
  ggml `_vl256` widens its grid gather there while our per-super-block overhead dominates.
- The k1 IQ ratios are FAR narrower than the archived rvv IQ ratios (e.g. iq3_xxs 22.5x→5.1x)
  — but that improvement is the scalar→vluxei16 emitter maturation, NOT a VLEN256 effect
  (the rvv numbers were the old 0-vluxei emit). Not directly board-comparable. See IQ flag.

## Reproduce
- ggml_ref.cpp (`_vl256` lifts) + ours.cpp banked under `bench/<q>/`; `keven_signs_q2xs.inc`
  (lifted from quants.c:4235-4267) + `ggml-common.h` (GGML_COMMON_IMPL_C) for the IQ refs.
- ours.cpp = `build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<q>-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (dev host).
- k1: `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast [-I<inc>] harness.cpp
  ours.cpp ggml_ref.cpp -lm -o bench && taskset -c 0-3 ./bench`. Binaries on k1
  `/data/k1micro/<q>/`.

## Synthesis — k1 (VLEN256) micro coverage COMPLETE (12 kernels, all byte-exact)
- **12/12 byte-exact** vs ggml's shipped `_vl256` (iq4_xs + tq1_0 are integer-bit-exact
  with a small fp-reassoc vs verbatim; the other 10 are 0.000e+00). NO emitter bug found.
- **PERF: 0 wins, 1 TIE (iq4_xs 1.040, stable ×3), 11 LOSSES.** Ratios 0.197–1.040.
- **The rvv FP4 "wins" do NOT survive on k1 — as expected.** iq4_nl (rvv 1.32x→k1 0.84),
  mxfp4 (1.22→0.80), iq4_xs (1.28→1.04 TIE). On VLEN256 ggml's `_vl256` adopts the SAME
  mf2 split-16-lane gather shape OURS won with on VLEN128, plus 2-blocks/iter packing,
  while our emit stays VLEN128-shaped (`vsetvl_e8m1(16)` = sub-VLMAX, half the m1 register
  at VLEN256). The rvv FP4 wins were VLEN128 shape artifacts, not portable advantages.
- **This is the maturity result the framing predicted**: adopting ggml's own instructions
  (mf2 gather, vluxei16) gives PARITY-or-LOSS, not beat-ggml. The single >1× (iq4_xs 1.04)
  is a noise-level TIE, rechecked ×3 per task discipline.
- **Two named gearbox/emitter targets** (consistent with the existing FINDINGs):
  1. **VLEN-aware FP4/ternary lowering** — emit `mf2`/`e8m1`-at-32 shapes for VLEN256
     instead of the hard-anchored `vsetvl_e8m1(16)`. The codebook class is currently
     Zvl128b-gated by construction; a VLEN256 reshape would close the FP4 sub-VLMAX gap.
  2. **IQ gather LMUL/packing tune** — both sides emit vluxei16 now; the residual 1.5–5x
     is gather-LMUL + per-super-block dispatch overhead, a tuning gap not a primitive gap.

## Status — k1 micro fill DONE (all 12)
- FP4 + ternary: DONE (iq4_nl 0.84, iq4_xs 1.04 TIE, mxfp4 0.80, tq2_0 0.36, tq1_0 0.36).
- IQ ALL 7 DONE (iq1_s 0.36, iq1_m 0.24, iq2_xxs 0.66, iq2_xs 0.52, iq2_s 0.66,
  iq3_xxs 0.20, iq3_s 0.21) — all byte-exact, all LOSS.
