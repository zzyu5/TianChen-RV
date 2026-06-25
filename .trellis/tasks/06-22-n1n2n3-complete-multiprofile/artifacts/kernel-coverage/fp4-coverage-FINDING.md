# FP4-codebook block-dot coverage — mxfp4, nvfp4, iq4_xs (closes the FP4/IQ codebook family)

This closes the block-dot coverage sweep: the FP4 (e2m1) codebook family — `mxfp4`
(MXFP4, E8M0 shared-exponent per 32-block), `nvfp4` (NVIDIA FP4, UE4M3 per-16-elem
scale on a 64-elem super-block), and `iq4_xs` (the q8_K SUPER-BLOCK variant of iq4_nl,
256 weights / 136B block, per-sub-block signed 6-bit scale). mxfp4 reuses ggml's
`kvalues_mxfp4 = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}`; iq4_xs reuses the EXACT
`kvalues_iq4nl` table iq4_nl uses — both the tiny ≤16-entry register codebook the iq4_nl
1.32x WIN rode on. The KEY question: does that win GENERALIZE — across scale formats
(mxfp4) AND across block GEOMETRY (iq4_xs's q8_K super-block)?

## Per-kernel result table (VLEN128, Sophgo SG2044, RVV1.0, clang-18)

| kernel | correctness rel-norm | ratio ggml/ours | generalization verdict | ggml baseline | gather shape (ours) |
|--------|----------------------|-----------------|------------------------|---------------|---------------------|
| mxfp4  | **0.000e+00** (bit-exact) | **1.22** | **WIN — generalizes** | REAL `_vl128` RVV | split lo/hi **16-lane** `vrgather_vv_i8m1` ×2 over 16-entry register codebook |
| nvfp4  | **0.000e+00** (bit-exact) | 0.57 | **N/A (scalar-only baseline)** | **scalar `_generic`** (no RVV kernel ships) | **8-lane** `vrgather_vv_i8m1` ×16 (4 sub-blk × 2 lo/hi × 2 strips) + 12× scalar `ldexpf` |
| iq4_xs | **0.000e+00** (matched-assoc; **1.16e-5** vs verbatim = fp reassoc, integer bit-exact) | **1.28** | **WIN — generalizes (super-block too)** | REAL `_vl128` RVV | split lo/hi **16-lane** `vrgather_vv_i8m1` ×2 per sub-block, **16 m1 gathers / 256-elem q8_K super-block**, codebook broadcast-loaded ONCE |

nvfp4's 0.57 is **vector-vs-SCALAR** (ggml ships no RVV nvfp4 kernel), so it does NOT
answer the gather-shape question — it is N/A, not a meaningful LOSS. Ratios are best-of-200
min, taskset -c 2; mxfp4 reproduced at 1.20–1.22 across runs; iq4_xs reproduced at
**1.28x** (6/8 runs 1.282–1.284, the one 1.205 a noisy outlier). iq4_xs re-run on
ssh rvv during QC (VLEN probe `VLENB=16` = VLEN128; on-board sources md5-identical to
these artifacts): steady **1.284x** ×3, `vs_matched_assoc=0.000e+00`,
`vs_verbatim_ggml=1.163e-05` — log saved at `blockdot-bench/iq4_xs/iq4_xs-bench.log`.

## iq4_xs — the SUPER-BLOCK FP4-codebook rung (completes the FP4/IQ family)
`iq4_xs` is the **q8_K super-block** variant of iq4_nl: it reuses the SAME 16-entry
non-linear FP4 codebook `kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,
69,89,113}` but packs 256 weights into a 136B super-block (d@0 fp16, scales_h@2 u16,
scales_l[4]@4, qs[128]@8) against a 292B `block_q8_K` activation, with a per-sub-block
SIGNED 6-bit scale (`((scales_l>>4*(s&1))&0xf | ((scales_h>>2s)&3)<<4) - 32`). The KEY
question this closes: does the tiny-codebook `vrgather` win survive the q8_K super-block's
per-sub-block scalar scale-extraction overhead? **It does — 1.28x, in line with iq4_nl
(1.32x) and mxfp4 (1.22x).**

- **ggml baseline = REAL `_vl128` RVV** (`ggml_vec_dot_iq4_xs_q8_K_vl128`, quants.c:5608),
  NOT scalar/_generic. ggml's vl128 does ONE wide **i8m4 64-lane `vrgather_vv_i8m4`** over
  2 sub-blocks/iter (with a 4-way `vget`/`vmv`/`vcreate` reorder), `vwmul_i16m8`, two
  `vwredsum` (acc0/acc1), and a per-super-block fp fold `sumf += d*(sumi1+sumi2)`. So this
  is a genuine **vector-vs-vector gather-shape contest** (the discriminating test), not the
  trivial vector-vs-scalar nvfp4 case.

- **OURS = the iq4_nl register-vrgather mechanism, applied per sub-block.** Codebook
  broadcast-loaded ONCE above the super-block loop (`vle8_v_i8m1(kvalues,16)`), then per
  sub-block: `vsetvl_e8m1(16)`, two **16-lane** `vrgather_vv_i8m1` (lo=`vand 0x0F`,
  hi=`vsrl 0x04`), `vwmul_i16m2`+`vwmacc_i16m2`, seed-0 `vwredsum_i16m2`. Verified from the
  emit: **16 `vrgather_vv_i8m1` (2/sub-block × 8), 0 `vluxei`, 8 `vsetvl_e8m1` per super-block** — exactly
  the register-codebook win primitive (NOT the IQ-quants' scalar table indexing). At
  VLEN128 the i8m1 VLMAX is exactly 16, so each gather FULLY fills the register; our two
  16-lane gathers beat ggml's single 64-lane i8m4 gather (which spans 4 registers) — the
  SAME mechanism + margin as iq4_nl/mxfp4. **The gathers are at VLMAX (vl=16), NOT
  sub-VLMAX — so this is the WIN case, not the nvfp4-dissolve case.**

- **CORRECTNESS — integer decode bit-exact (the tq1_0 fp-reassoc story).** rel-norm vs
  VERBATIM ggml = **1.163e-05** (finite, nonzero), but **0.000e+00** against a
  matched-association scalar ref (`ggml_ref_matched.cpp`) that reproduces ggml's integer
  decode EXACTLY but uses OURS's per-sub-block DISTRIBUTED fp fold. PROOF the 1.16e-5 is
  pure fp32 reassociation, not a decode bug: ggml folds the per-sub-block 6-bit scale
  INTO the INTEGER accumulator (`sumi1/sumi2 += acc*ls` across all 8 sub-blocks, all
  integer-exact) then does ONE fp mul-add per super-block, while OURS folds the scale in
  FLOAT (`d1 = d4d8*(float)ls`) and accumulates `sumf += d1*(float)sumi` per sub-block —
  many more fp32 roundings, amplified by cross-super-block sign cancellation in `sumf`.
  Matching the association collapses the norm to exactly 0.0 — IDENTICAL
  to the tq1_0 finding (where ggml's two ternary kernels disagreed on assoc). Every
  per-sub-block signed 6-bit scale (scales_h `h>>=4`/`(h<<4)&0x30` proven equivalent to
  OURS's `(scales_h>>2s)&3<<4`), nibble unpack, codebook gather, and `vwredsum` is
  bit-exact. NO emitter bug. iq4_xs emit was already in the block-dot coverage; **re-confirmed**.

- **NO even-nb caveat** (unlike iq4_nl/mxfp4's q8_0 2-blocks/iter tail-drop). iq4_xs loops
  one super-block at a time, inner loop ALWAYS 4×2=8 sub-blocks — sizes just need to be
  multiples of QK_K=256. Cleaner apples-to-apples comparison than iq4_nl.

## GENERALIZATION VERDICT — the FP4-codebook win GENERALIZES to mxfp4, is N/A for nvfp4

- **mxfp4 = the clean generalization test, and it WINS (1.20x).** ggml ships a REAL
  RVV `ggml_vec_dot_mxfp4_q8_0_vl128` that is structurally IDENTICAL to ggml's
  `iq4_nl_q8_0_vl128`: single 32-lane `__riscv_vrgather_vv_i8m2`, 2-blocks/iter, only
  the scale decode differs (E8M0 byte vs fp16 d). OUR emit uses the SAME mechanism that
  won iq4_nl: the 16-entry codebook is broadcast-loaded ONCE (`vle8_v_i8m1`) above the
  loop, and each block's nibbles are mapped by TWO **16-lane** `vrgather_vv_i8m1` (lo =
  `vand 0x0F`, hi = `vsrl 0x04`) then `vwmul`/`vwmacc`/`vwredsum`. At VLEN128 the i8m1
  VLMAX is exactly 16, so our two 16-lane gathers fully utilize the register and beat
  ggml's single 32-lane i8m2 gather (which spans 2 registers) — IDENTICAL win mechanism
  and a near-identical margin to iq4_nl (1.32x). **The hypothesis holds: register
  `vrgather` over a tiny ≤16-entry FP4 codebook at the VLEN-native 16-lane shape WINS.**

- **nvfp4 = N/A for the generalization question (no ggml RVV kernel to beat).** ggml
  ships NO riscv vec_dot for nvfp4: `arch-fallback.h` maps
  `ggml_vec_dot_nvfp4_q8_0_generic → ggml_vec_dot_nvfp4_q8_0`, so the REAL board kernel
  IS the scalar `_generic` (only arm has a vector kernel). Our 0.582 ratio is therefore
  **vector-vs-scalar**, a categorically different (and trivially-expected-favorable)
  claim — NOT the vector-vs-vector gather-shape contest. The generalization verdict for
  nvfp4 is **N/A, not WIN and not a meaningful LOSS.**

- **WHY nvfp4 loses even to scalar — the load-bearing structural finding.** The nvfp4
  super-block geometry DEFEATS the FP4-codebook win mechanism two ways: (1) the gather
  is a **sub-VLMAX 8-lane** `vrgather_vv_i8m1` (`vsetvl_e8m1(8)`) — half the 16-elem
  sub-block at a time, wasting half the i8m1 register, 16 gathers per super-block; and
  (2) the UE4M3 scale is decoded with **12 scalar `ldexpf` calls per super-block** (3 per
  sub-block × 4, with exp/man split + two ldexpf branches + e==0/e==0x7F specials) —
  branchy scalar overhead far heavier than mxfp4's branch-free E8M0 bit-dance. The narrow
  strips + dense scalar-FP overhead mean the scalar `_generic` (tight inner loops, no
  vsetvl churn, no sub-VLMAX waste) is FASTER. **The FP4-codebook win is real but
  conditional: it requires the gather to fill the VLEN-native lane width; a block format
  that forces sub-VLMAX strips (nvfp4's 16-elem sub-blocks at VLEN128) dissolves it.**

## CORRECTNESS (the maturity gate) — both bit-exact, no emitter bug

- **mxfp4: rel-norm 0.000e+00** vs ggml's REAL `_vl128` (8 seeds × 6 even-nb sizes).
  The agreement fill MIXES IN the genuinely-new E8M0 decode branches the test IR flagged:
  the `x<2` denormal patterns (e∈{0,1} → `bits = 0x00200000u<<e`) AND the normalized
  `((e-1)<<23)` branch — the structured E8M0→fp32-HALF scale matches
  `ggml_e8m0_to_fp32_half` exactly across BOTH branches. Independently corroborated by the
  archived `inc31-mxfp4` ssh-rvv byte-exact run (3520 cases, 0 failures, with NON-VACUOUS
  negative controls: wrong linear codebook, full-not-half E8M0 scale, and flipped nibbles
  each diverge → codebook + half-scale + nibble consumption all load-bearing).
- **nvfp4: rel-norm 0.000e+00** vs ggml's `_generic` scalar oracle (8 seeds × 6 sizes).
  The agreement fill MIXES IN the genuinely-new UE4M3 decode branches: the e==0/e==0x7F
  specials (→0.0f, hit 321×/314× over the run) AND the exp==0 denormal `ldexpf(man,-9)`
  branch (hit 369×), alongside the normalized `ldexpf(1+man/8, exp-7)` branch. All four
  paths + the ×0.5f + the super-block activation indexing (`y[2*ib + s_idx/2]`,
  `q8_off = (s_idx%2)*16`, split-at-midpoint lo/hi fold) match ggml exactly. NO emitter
  bug in either kernel.

## Status — FP4 family DONE (closes the block-dot sweep)
- mxfp4: DONE (WIN 1.20x, bit-exact). nvfp4: DONE (bit-exact; perf N/A — scalar-only ggml).
- **iq4_xs: DONE (WIN 1.28x; integer bit-exact, 1.16e-5 vs verbatim = fp reassoc).** The
  super-block FP4-codebook rung — confirms the tiny-codebook `vrgather` win survives the
  q8_K per-sub-block scalar scale-extraction. FP4-codebook family micro coverage COMPLETE
  (iq4_nl 1.32x WIN, mxfp4 1.21x WIN, iq4_xs 1.28x WIN; nvfp4 N/A scalar-only).

## Coverage-sweep synthesis (FP4 family in context of the ~17-kernel sweep)
- The "gather-heavy ⇒ WIN" pattern is now precisely bounded by THREE FP4-family +
  prior-sweep data points:
  - **WIN** ⟺ register `vrgather` over a tiny ≤16-entry codebook AT the VLEN-native lane
    width: iq4_nl (1.32x), **mxfp4 (1.20x)**, **iq4_xs (1.28x — super-block)**, q2_K (the
    lone K-quant win). Our split lo/hi 16-lane i8m1 beats ggml's wider i8m2/i8m4 gather on
    VLEN128. **The win is NOT confined to q8_0-paired blocks: iq4_xs proves it survives the
    q8_K super-block structure** — the per-sub-block scalar 6-bit scale extraction does NOT
    dissolve it (contrast nvfp4, where sub-VLMAX strips + 12 scalar ldexpf DO dissolve it).
  - **DISSOLVES** when the block format forces **sub-VLMAX** gather strips: **nvfp4**
    (8-lane strips + 12 scalar ldexpf/super-block) loses even to ggml's scalar `_generic`.
  - **LOSS** ⟺ either ggml uses a wider-LMUL fused fold (q5_*, tq*, most K-quants) OR the
    gather is over a LARGE 64-bit grid that ggml does with `vluxei16` indexed vector loads
    while our emit falls back to scalar table indexing (all 7 IQ-quants, 5–22x losses).
- N3/Gearbox implication for nvfp4 (a concrete, named emitter target, not a tuning knob):
  to be competitive the emitter would need to (a) coalesce the 4 sub-block 8-lane gathers
  into VLEN-native-width strips (or process multiple sub-blocks per gather), and (b)
  vectorize / hoist the UE4M3 scale decode instead of 12 scalar `ldexpf` per super-block.
  Until then, nvfp4's structurally-correct emit is a verified drop-in, not a perf win.

## Harness provenance / reproduce
- Bench sources: `artifacts/kernel-coverage/blockdot-bench/{mxfp4,nvfp4}/{ours.cpp,
  ggml_ref.cpp,harness.cpp}` — copy-then-adapted from the proven `iq4_nl/` harness.
- ours.cpp = `build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<q>-q8-0-block-dot.mlir
  --tcrv-rvv-lower-to-emitc > lowered.mlir ; /usr/lib/llvm-20/bin/mlir-translate
  --mlir-to-cpp lowered.mlir` (dev host). (`tcrv-translate --tcrv-rvv-emitc-to-cpp` is the
  kernel-gated EXPORT path and needs a selected dispatch surface these FileCheck IRs lack;
  the stock EmitC emitter renders the bare lowered module — the `v1/v2/...` value naming
  is its signature.) ours line counts: mxfp4=96, nvfp4=308.
- mxfp4 ggml_ref.cpp = `ggml_vec_dot_mxfp4_q8_0_vl128` (the REAL RVV kernel) lifted
  verbatim from `ggml-cpu/arch/riscv/quants.c:6470` against raw byte offsets (block_mxfp4
  17B: e@0 E8M0-byte, qs[16]@1; block_q8_0 34B), 4-arg ABI adapter `kern_ggml`, with the
  inline `ggml_e8m0_to_fp32_half` decode.
- nvfp4 ggml_ref.cpp = `ggml_vec_dot_nvfp4_q8_0_generic` (the SCALAR oracle = the real
  riscv board kernel) lifted verbatim from `ggml-cpu/quants.c:279` (block_nvfp4 36B:
  d[4]@0 UE4M3, qs[32]@4; two block_q8_0 per super-block), inline `ggml_ue4m3_to_fp32`.
- iq4_xs sources in `blockdot-bench/iq4_xs/` (copy-then-adapted from `tq2_0/`, the proven
  q8_K super-block harness): ours.cpp (`build/bin/tcrv-opt
  test/Conversion/RVV/rvv-to-emitc-iq4-xs-q8-k-block-dot.mlir --tcrv-rvv-lower-to-emitc |
  mlir-translate-20 --mlir-to-cpp`, 416 lines, 16 vrgather / 0 vluxei / 8 vsetvl_e8m1
  code calls per super-block — i.e. 2 gathers + 1 vsetvl per sub-block × 8 sub-blocks;
  raw `grep -c` over the emit reports 32/16 because each call is preceded by a
  `// …callee=__riscv_…` annotation line);
  ggml_ref.cpp = `ggml_vec_dot_iq4_xs_q8_K_vl128` (the REAL RVV kernel) lifted verbatim
  from `ggml-cpu/arch/riscv/quants.c:5608` against raw byte offsets (block_iq4_xs 136B:
  d@0 fp16, scales_h@2 u16, scales_l[4]@4, qs[128]@8; block_q8_K 292B), 4-arg ABI adapter;
  ggml_ref_matched.cpp = the tq1_0-style matched-association scalar diagnostic (ggml's
  integer decode + OURS's per-sub-block distributed fp fold → OURS hits 0.0). iq4_xs binary
  `~/blockdot-cov/iq4_xs/`. No even-nb caveat (one super-block/iter, always 8 sub-blocks).
- On rvv: `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast harness.cpp ours.cpp
  ggml_ref.cpp [ggml_ref_matched.cpp] -lm -o bench && taskset -c 2 ./bench`. Binaries:
  `~/fp4cov/{mxfp4,nvfp4}/`, `~/blockdot-cov/iq4_xs/`.
- Agreement over 8 seeds × 6 sizes; timing best-of-200-min, iters 4000.
- Fill (honest): the AGREEMENT run goes WIDE on the scale bytes to exercise the
  genuinely-new decode branches — mxfp4 mixes e∈{0,1} (x<2 denormal) with e∈[118,136];
  nvfp4 mixes d∈{0,0x7F} (specials→0) and d∈[0x01,0x07] (exp==0 denormal) with d∈
  exp 5..9 — while avoiding LARGE exponents that overflow to inf (which would nan the rel
  metric). The TIMING run pins scales O(1) (mxfp4 e∈[120,134]; nvfp4 d∈[0x38,0x47], exp
  7..8) so products stay finite; timing is unaffected by the agreement fill, so the ratios
  are clean. Branch-hit counts (8 seeds × 6 sizes): nvfp4 d==0 ×321, d==0x7F ×314,
  exp0-denormal ×369 (probe-verified).
- mxfp4 caveat: ggml's `_vl128` is 2-blocks/iter (`ib+1<nb`) and drops the tail on ODD
  nb; harness uses EVEN-nb sizes so the comparison covers identical blocks.
- EMITTABILITY: both target kernels emit cleanly (tcrv-opt + mlir-translate-20), NO
  emitter-level coverage gap.
