# Win-A brick #2 FP4 codebook VLEN256 mf2 auto-select — k1 BOARD SEAL (commit e65edf76)

The last gate for "FP4 codebook VLEN256 auto-select mf2 wide form": the gearbox now
defaults to selecting the codebook integer-core LMUL anchor from the REAL VLEN fact.
VLEN128(rvv)→**m1** (already rvv-validated); VLEN256(k1)→**mf2/factor-1/elided**
(the ggml `_vl256` shape: mf2 VLMAX=16 = a FULL mf2 register holds exactly the 16
codebook entries; the gather indexes all 16). This is the load-bearing seal of a
brand-new k1 emit: byte-exact vs ggml `_vl256` + parity-or-better micro, recovering
the prior VLEN128-shape LOSS.

MEASURE-ONLY. No lib/ changed, no rebuild, no git commit. Existing `build/bin/tcrv-opt`.

## Gearbox auto-select confirmed (host, the attr-less divergence test)
The attr-less `iq4_nl`/`mxfp4` block-dot op carries NO shape knobs; the compiler computes
them from `--tcrv-rvv-materialize-schedule=march=<M>`:
- `march=rv64gcv`         → `integer_core_lmul="m1"`,  `minimum_vlen=128`, `multi_block_factor=1`
- `march=rv64gcv_zvl256b` → `integer_core_lmul="mf2"`, `minimum_vlen=256`, `multi_block_factor=1`, `strip_elision="elided"`
(BOTH iq4_nl + mxfp4.) The 256 threshold emerges from strip-VLMAX arithmetic: mf2 VLMAX
8<16 PRUNED @128, 16 reaches the 16-entry table-index range @256. factor pinned to 1
(this brick = the anchor-flip only, factor-4 unroll is its own brick).

**Sanity: the auto-select VLEN256 emit is byte-identical to the hand-shaped `…-vl256.mlir`**
form (the validated wide form), modulo the factor-1 single-step loop vs that test file's
factor-2 packing — the gearbox correctly emits factor-1 as this brick pins. mf2 anchor +
all intrinsic spellings identical.

## Board / live VLEN
`ssh k1`, SpacemiT X60, probed live: `vlenb=32 → VLEN=256`, `e8m1 VLMAX=32, e8mf2 VLMAX=16`.
clang-18 `-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`, **`taskset -c 0-3`**, build+run
on `/data/fp4mf2/` (never wrote `/`), serial (one bench at a time on the shared board).
Reference = ggml's REAL shipped `_vl256` body (`vle8_v_i8mf2` split-16 gather, 2-blocks/iter),
the same re-lifted `ggml_ref.cpp` k1-micro-fill used (so the mf2-vs-ggml and mf2-vs-m1
comparisons are apples-to-apples with the prior 0.840/0.798 LOSS).
ratio = ggml_ns / ours_ns. >1 ⇒ OURS faster.

## objdump confirmation — mf2 wide form is real on silicon (per kernel)
llvm-objdump-18 of the k1-compiled `ours_mf2.o`:
- **iq4_nl**: codebook core emits `vsetvli zero, zero, e8, mf2` (+ `e16, m1` widened product,
  one step wider than mf2). **NO `e8, m1` and NO `e16, m2`** (the VLEN128 m1 form is ABSENT).
- **mxfp4**: `vsetivli zero, 0x10, e8, mf2` + `vsetvli zero, zero, e8, mf2`. **NO `e8, m1`,
  NO `e16, m2`.**
The old m1 `ours_m1.o` (for the delta) confirms the contrast: `vsetivli …,0x10,e8,m1` +
`vsetvli …,e8,m1` (the sub-VLMAX `vsetvl_e8m1(16)` = 16 of 32 lanes = half register @256).

## Results (3-way, k1 VLEN256; mf2 rechecked ×3 stable)

| kernel | byte-exact vs `_vl256` | mf2 ours ns | ggml `_vl256` ns | **mf2 vs ggml** | old-m1 ours ns | m1 vs ggml (was) | **mf2 vs old-m1** |
|--------|------------------------|------------:|-----------------:|----------------:|---------------:|-----------------:|------------------:|
| iq4_nl | **0.000e+00** | 1597.9 | 1717.9 | **1.075 (parity, marginally above; ×3 stable)** | 2044.6 | 0.840 (matches prior) | **1.28x faster** |
| mxfp4  | **0.000e+00** | 1638.6 | 1658.0 | **1.012 (parity; ×2 stable)** | 2077.7 | 0.798 (matches prior) | **1.27x faster** |

- **Byte-exact: both 0.000e+00** (FULL bit-exact, not just integer — the m1 vwredsum/scale
  fold is unchanged by the anchor flip; negative control nonzero=1, nonfinite=0). **NO gather
  bug** — the mf2 gather at VLMAX=16 correctly indexes all 16 codebook entries (the specific
  risk the task flagged). This is the load-bearing result and it PASSES.
- **Parity LOSS recovered (ceiling is parity, not win).** The old VLEN128-shaped m1 emit
  reproduced EXACTLY the prior k1-micro-fill LOSS (iq4_nl 0.840, mxfp4 0.798) — confirming
  the baseline is identical. The mf2 anchor-flip pulls iq4_nl 0.840→**1.075** and mxfp4
  0.798→**1.012**. Both now AT parity (iq4_nl marginally above, mxfp4 on the nose),
  clearing the ≥0.95 target. Per the framing (adopting ggml's own mf2 gather → parity,
  not beat-ggml), iq4_nl's 1.075 is reported as parity, marginally above — NOT a structural
  win claim.
- **mf2 vs old-m1: ~1.27–1.28x** — the anchor-flip's direct value. m1 at VLEN256 wastes half
  the register (`vsetvl_e8m1(16)` caps active vl at 16 of VLMAX=32); mf2 VLMAX=16 = a FULL
  register. Closing that sub-VLMAX waste is the whole gain.

## Honest framing (per VLEN-SHAPE-MATCH discipline; ceiling = parity)
This brick pins `multi_block_factor=1`; ggml `_vl256` is mf2 **+ 2-blocks/iter**. The mf2
anchor-flip alone fixed the dominant loss source (the half-register sub-VLMAX waste), which
is why the bulk of the 0.80/0.84 LOSS came back and both kernels land AT parity (iq4_nl
1.075, mxfp4 1.012). **The headline is: the LOSS is gone** — we adopted ggml's own mf2
gather shape and reached parity, exactly the capability-driven-lowering claim ("采用 ggml
自己指令→parity 非 beat-ggml").

On the iq4_nl 1.075 marginally-above-parity: it is x3 stable (not noise), so it is a small
real effect, but it does NOT support a structural-win claim and is NOT reported as one. The
naive packing argument (our factor-1 has LESS packing than ggml's factor-2 → predicts a
loss) would put us below parity, not above — so the residual is not explained by packing
alone. The plausible mechanism on the **in-order SpacemiT X60**: 2-block packing buys
latency-hiding on out-of-order cores but pays little on in-order issue, where our simpler
single-block loop pipelines about as well with less register pressure. That is a board-
specific micro-architectural nuance, not a portable advantage — reported as parity. (Note
this is unlike the rvv VLEN128 1.32x iq4_nl "win" that did NOT survive to k1; here we are
AT parity on real VLEN256 silicon, which is the honest ceiling.)

## Reproduce
- mf2 emit (shipping artifact, gearbox auto-select):
  `build/bin/tcrv-opt test/Conversion/RVV/rvv-<q>-q8-0-block-dot-autotuner-divergence.mlir
   --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc
   | mlir-translate-20 --mlir-to-cpp` (dev host). q ∈ {iq4-nl, mxfp4}.
- m1 emit (old VLEN128 form, for the delta) = k1-micro-fill `bench/<q>/ours.cpp`.
- ggml `_vl256` ref + harness = k1-micro-fill `bench/<q>/{ggml_ref.cpp,harness.cpp}`
  (harness times ours AND ggml interleaved best-of-200 min within one process; same OURS
  symbol for mf2 + m1 → swap the ours file, normalize each to its OWN in-run ggml).
- k1: `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast harness.cpp ggml_ref.cpp
   ours_{mf2,m1}.cpp -lm -o bench && taskset -c 0-3 ./bench`. Board temp `/data/fp4mf2/` cleaned.

## Verdict — k1 VLEN256 mf2 auto-select SEALED (LOSS recovered to parity)
Both kernels byte-exact (0.000e+00, no gather bug) + the prior FP4 LOSS recovered to
PARITY (iq4_nl 0.840→1.075 marginally above, mxfp4 0.798→1.012 on the nose) + mf2
~1.27–1.28x over the old VLEN128 m1. The defensible seal: byte-exact (no gather bug) +
LOSS-recovered-to-parity + mf2 1.27x over old m1 (our-vs-our, both byte-exact). The
gearbox's VLEN→codebook-anchor flip is a correctness-safe, capability-driven Win-A
lowering, now sealed on real VLEN256 silicon. The prior k1-micro-fill FP4 LOSSES
(iq4_nl 0.84 / mxfp4 0.80) were VLEN128-shape artifacts of the old hard-anchored emit
(half-register sub-VLMAX waste) — the gearbox closes them. Ceiling is parity, not win;
no beat-ggml claim. No lib/ change; no git commit; board cleaned.
