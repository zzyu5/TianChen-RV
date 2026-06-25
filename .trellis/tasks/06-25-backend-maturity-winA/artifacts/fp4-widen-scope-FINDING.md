# FP4 codebook "widen to full register at VLEN256" — scope determination

**Question**: is closing the k1 (VLEN256) FP4 LOSS by filling the vector register a
**pure lowering** (in-scope backend Win-A: same bytes, no data movement, just LMUL/AVL
+ schedule) or a **block reconstruction** (block-as-lane = repack = front-end, out of
scope)? Read-only; no lib/ changes.

## The premise correction that decides everything (read this first)

The task framing says "补到 VLEN256 全宽(32 lane)". **That is NOT what ggml's shipped
`_vl256` does, and it is NOT what closing the gap requires.** The packed FP4 weight byte
holds **two nibbles**: low nibble → activation lane `i` (q8[0..15]), high nibble →
activation lane `16+i` (q8[16..31]). The dot is
`vwmul(w_low, q8[0..15])` then `vwmacc(w_high, q8[16..31])` — see
`emitOffsetBinaryProductFromDecodedValue` (RVVToEmitC.cpp:2563) and `y1 = y0 + 16` at the
load sites (e.g. iq4_xs `q8HighPtr = q8Base + 16`, line 687). **The nibble-pairing means
our emit (and ggml's `_vl256`) *choose* a 16-lane shape**: the packed qs is 16 bytes, so
splitting lo/hi nibbles gives two 16-lane halves naturally. So `halfBlock = qk/2 = 16`
(lines 83, 1049, 566) is the chosen lane count, not a VLEN128-VLMAX accident; the `16` in
`vsetvl_e8m1(16)` is that half-block count (the codebook-gather VLMAX>=16 requirement
coincides, lines 57, 3827-3838), not a VLEN cap.

**The real scope discriminator is in-memory weight rearrangement, NOT lane count.** A flat
**32-lane single-block** dot is *also* pure lowering: in-register `vcreate(lo16 || hi16)`
→ 32-lane weight, gather at m1 (VLMAX=32 ≥ 16 ✓), one contiguous 32-byte `vle8_i8m1` of
q8[0..31] (contiguous within the block), `vwmul`→i16m2, one `vwredsum` — no memory moved.
(iq4_xs's `_vl256` below proves the >16-lane in-register technique.) Repack/front-end is
only crossed by *in-memory* rearrangement (de-interleaving `d` from `qs`, or block-as-lane
GEMM packing). That is exactly why a 32-lane load **across two adjacent blocks** WOULD need
repack — stride-18 AoS puts each block's `d` between the qs arrays, so qs of block N and
block N+1 are not contiguous — but a 32-lane load **within one block** does not.

**ggml `_vl256` for iq4_nl (quants.c:5552-5586) and mxfp4 (6543-6577) fill the register
WITHOUT a 32-lane block op**: it keeps the 16-lane nibble-pair shape but switches LMUL
`m1 → mf2` (at VLEN256, mf2 VLMAX = 16 = a FULL register; our `m1` at vl=16 uses half of
a 32-lane m1 register = the sub-VLMAX waste the k1 FINDING measured) and unrolls **2
independent blocks per iteration** (ib+0, ib+1 — each a self-contained 16-lane dot, no
cross-block lane mixing). That is the entire `_vl256` advantage: LMUL spelling + unroll.

## Per-kernel verdict

### iq4_nl — **PURE LOWERING** (no block reconstruction)
ggml `_vl256` is our exact `emitStripReduce` (split lo/hi nibble → gather → vwmul/vwmacc →
vwredsum), twice, at mf2 instead of m1. All 32 elements of a block come from that ONE
block's contiguous qs (16 bytes) + q8 (32 bytes); the 2nd "block" is the next independent
block, already handled by the existing `multiBlockFactor` unroll (line 64, code 453-491).
No cross-block data is ever pulled into one op. **How to change**: (a) set `coreLmul = "mf2"`
+ the widened product `wideLmul m1` (currently HARDCODED `"m2"` at line 68 — must become
mf2→m1, not m1→m2) + reduce `vwredsum_vs_i16m1_i32m1`; (b) `multiBlockFactor = 2`
(already implemented, byte-exact, folds stay ascending); (c) gate it on VLEN256. **Risk /
verifier**: the codebook op verifier pins `integer_core_lmul == "m1"` fail-closed
(RVVDialectWideningOps.cpp:3831-3838) with the rationale "mf4→VLMAX=4 at **VLEN=128**" — it
has no VLEN256 notion. mf2 at VLEN256 gives VLMAX=16 (gather valid, register full), but at
VLEN128 mf2 VLMAX=8 < 16 (gather breaks). So the *only* new machinery is a **VLEN256-gated
verifier relaxation** to admit mf2 under that capability fact — still pure lowering, no
repack. **Fold-back: NOT touched** — reduction stays a single 16-lane vwredsum, just at
mf2/m1 widths; no exposure to the historical VLEN256 fold-back bug.

### mxfp4 — **PURE LOWERING** (identical to iq4_nl)
`_vl256` (quants.c:6543-6577) is byte-identical in shape to iq4_nl's — only the E8M0 scale
decode differs (not in the vector core). Same nibble-pairing, same mf2 + 2-block unroll,
same contiguous single-block data. Same change recipe (mf2 coreLmul + wideLmul-m1 +
multiBlockFactor=2 + VLEN256 verifier relax), same NO fold-back change.

### iq4_xs — **PURE LOWERING, but a DIFFERENT recipe (and fold-back IS live)**
ggml `_vl256` (quants.c:5687-5749) does NOT use the nibble-pair/mf2 shape. It loads **64
packed qs bytes = 128 weights = 4 sub-blocks at once** via `vle8_v_u8m2(iq4, 64)`
(`iq4 += 64`), `vrgatherei16` **reorders within-register** into natural 0..127 element
order, a **flat** `vwmul_vv_i16m8(iq4b, q8b, 128)` against `vle8_v_i8m4(q8, 128)`, then a
**4-way `vget` + `vwredsum`** back into the 4 per-sub-block scales. **Crucially all 64 qs
bytes + 128 q8 bytes are contiguous within the SAME super-block** (`x[ibl].qs`,
`y[ibl].qs`) — the reorder is a lane shuffle of already-loaded data, NOT a memory gather
across super-blocks. **No data is pulled from a neighbouring block → no repack → still
pure lowering.** But it is a *wider reshape* (m2/m4/m8) with a **4-way fold-back**, which
our per-sub-block `emitSubBlockSumi` (16-lane m1 vwredsum, line 653) does not have. So
unlike iq4_nl/mxfp4, **adopting iq4_xs's `_vl256` does change the reduction structure →
fold-back IS live here** and must be re-validated at VLEN256 (the historical silent-bug
surface). Lower-risk alternative that stays in our current shape: keep per-sub-block m1
but widen the sub-block dot to mf2 like iq4_nl — smaller win, no fold-back change.
(Consistent with the k1 FINDING: iq4_xs already measured 1.04 TIE — see honesty note.)

## Honesty note (so this doesn't oversell as a Win-A "beat-ggml" brick)
All three are **in-scope pure lowering** — answer is YES. But the change *adopts ggml's
own mf2/m4 shapes*, so the ceiling is **parity, not win** (the k1 FINDING already shows
iq4_xs 1.04 TIE and predicts iq4_nl/mxfp4 closing toward ~1.0 once mf2-shaped). This is a
legitimate **emitter-maturity / VLEN-aware-lowering brick** (the named gearbox target #1
in k1-micro-fill-FINDING.md), not an independent novelty number. It removes the
VLEN256 sub-VLMAX LOSS; it does not manufacture a VLEN256 win.

## Net (8 lines)
- **iq4_nl = PURE LOWERING.** Discriminator is in-memory rearrangement, not lane count:
  ggml/our emit *choose* 16-lane mf2, but a 32-lane flat within-block dot is equally
  lowering — all data is single-block-contiguous. (Only a 32-lane load *across* two AoS
  blocks would need repack: stride-18 `d` sits between the qs arrays.) Recommended change
  = match ggml `_vl256`: coreLmul=mf2, wideLmul m1 (hardcoded "m2" line 68),
  multiBlockFactor=2 (exists), + VLEN256-gated verifier relax (op pins m1 @
  RVVDialectWideningOps.cpp:3831). Reduction UNCHANGED → no fold-back.
- **mxfp4 = PURE LOWERING.** Byte-identical shape to iq4_nl (only E8M0 scale differs).
  Same recipe, same no-fold-back.
- **iq4_xs = PURE LOWERING but different recipe.** `_vl256` = m2 64-byte load (4
  sub-blocks, same-super-block contiguous) + within-register vrgatherei16 reorder + flat
  m8 vwmul + 4-way vget/vwredsum. No cross-block memory → not repack. BUT reduction
  changes → **fold-back IS live**, must re-validate @ VLEN256.
- **None require block reconstruction / block-as-lane / repack.** All bytes stay where
  they are; only LMUL spelling, unroll, and (iq4_xs) the reduce shape move.
- **Ceiling = parity, not win** (adopts ggml's shapes; iq4_xs already 1.04 TIE). Valid
  maturity brick, not a beat-ggml brick.
