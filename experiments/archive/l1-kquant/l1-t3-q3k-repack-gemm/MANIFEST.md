# cell MANIFEST — l1-t3-q3k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (T3-tile fmt3 — q3_K repack GEMM tiling-lever transfer test on
  rvv/VLEN128: does the q4_K S6 register-cliff lever, adapted no-min for q3_K — decoded signed-6bit-scale
  int16 STACK PANEL via vse16/vle16, reload deferred to first use, NO min-fold panel since q3_K has no
  min — drive q3_K's fully-unrolled dual-plane (3-bit qs low-2 + subtractive hmask high-1, 16 sub-blocks)
  body to the ≤32-vreg cliff, byte-exactly? ★weight-bound prediction: q3_K SHARES the q6_K
  `kquant_single_scale_no_min` fold (parent reuses the q6_K `KQuantDecodeFacts` WHOLE), so it should be
  weight-reconstruction-bound like q6_K — NOT eat the cliff like q4_K/q2_K. Sibling of the q6_K NULL cell
  `l1-t3-q6k-repack-gemm`, the q2_K WIN cell `l1-t3-q2k-repack-gemm`, and the q4_K WIN cell
  `l1-tile-s6-q4k-repack-gemm`. Logged ALONE.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ RESULT = **NULL / weight-bound like q6_K — the register-cliff lever does NOT transfer to q3_K.** (1)
  objdump at the measured clang-17.0.6 -O2 board form (campaign-canonical fixture
  `rvv-to-emitc-repack-gemm-q3-K-q8-K.mlir` + vs*r.v/vl*r.v seal, NOT driver-export): spill UNTILED
  **978 → TILED 894** (−8.6%, only the ~84 staged scale strips shaved; NOT →~0 — q2_K hit 7, q4_K hit 3),
  reload 1095→1016, maxVreg **v31 → v31** (the ≤32-vreg cliff was NEVER reached, no spare register) at
  BOTH -O2/-O3 (983→910 at -O3); vwmacc **2176 UNCHANGED** = byte-exact multiset; textB 75686→72464. The
  ★key discriminator: UNTILED spill **978** is q6_K-MAGNITUDE (~900, even above q6_K's 913), an ORDER OF
  MAGNITUDE above q4_K golden's 84 — the ~894 residual is the DUAL-PLANE 3-bit qs|hmask weight
  reconstruction (2048 vand + 2048 vsrl + 1024 vsll/vor/vsub → the i16 weight-partials feeding the 256
  vwmacc_vv), NOT the decode-scale strips the panel stages. (2) byte-exact GREEN both ways: silicon verify
  UNTILED & TILED both INT_mismatch_total=0, NORM worst_ulp=760 worst_norm=4.973e-07 (identical);
  IDENTITY-DUMP o_untiled.bin == o_tiled.bin (nr64 32768 / nr16 8192 floats, 0 mismatch). (3) A/B
  tiled/untiled paired cold (nr=64 N=12 / nr=16 N=10): **1.080× / 1.084× = a modest +8.0%/+8.4% GAIN**
  (ours cv 0.04–0.21%, non-overlapping, SHAPE-ROBUST) — the clean panel shaves the ~8% scale-spill
  overhead, but this is an ORDER OF MAGNITUDE short of the q2_K register-cliff win (+565%) and does NOT
  clear the spill→~0 ∧ ≤32-cliff bar, so it is a NULL-on-the-lever, not a HOLDS. (4) vs-opponent block-dot
  (clean anchor nr=64, opp cv 1.26% across 24 samples): UNTILED parity **0.176×** (reproduces the
  board-harvest2 q3_K prefill 0.176× LOSS → basis validated) → TILED **0.191×** (nudged by the +8%, still
  a ~5.2× LOSS; nr=16 0.173×→0.190×). ggml_vec_dot_q3_K_q8_K block-dot is ~5.2–5.8× faster and tiling does
  NOT rescue it (contrast q2_K, SAME lever, 0.21→1.41 flip). The pre-registered gate (spill→~0 ∧ maxVreg≤32
  ∧ MAC/cycle up ∧ wall T-N) is NOT met (spill 978→894 not ~0; v31 no cliff) ⇒ **tiling_holds = FALSE**.
  This CONFIRMS on silicon the parent constructor's "weight-bound like q6_K, ship PLAIN" q3_K ruling (and
  quantifies the +8% overhead-shave the plain form leaves on the table — far short of a rescue).
  **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single core; opponent = single-thread block-dot
  proxy; our correctness = construction oracle + the byte-exact verify + identity gates here.
- **role**: L1-maturity T3-tile A/B evidence. Isolates the q3_K tiling increment by measuring UNTILED
  (working-tree HEAD 1b367c1f front-door PLAIN export of the campaign-canonical region fixture via
  read-only `build/bin/tcrv-opt` + mlir-translate-20, md5 813e5851, == the retired-direct-emit /
  board-harvest2 0.176× baseline) vs TILED (a deterministic /tmp S6 stack-panel transform of that plain
  export — `tools/e2e-harness/board/s6_tile_scales.py` stages all 32 decoded signed-6bit-scale vint16m1
  vectors through a function-scope int16 stack panel, vse16 after decode + vle16 reload deferred to first
  vwmacc_vv use, the q3_K analogue of the q6_K S6 no-min emitter, md5 31bdf294) as separate board binaries
  sharing the same extern-C symbol, against the opponent's REAL linked block-dot, with a silicon verify + a
  full-output IDENTITY cmp proving the timed TILED is byte-exact to UNTILED/oracle. Harness/driver live
  under `tools/e2e-harness/board/` (`kquant_gemm_tile_q3k_t3_ab.sh` + `kquant_gemm_paired_q3k_driver.c` +
  `s6_tile_scales.py` + reused `kquant_repack_verify_q3K.c`). Data/evidence only; the exported/tiled .c are
  NOT stored (deterministically regenerable — see the objdump seal + the transform script +
  `experiments/active/kquant-l1-q6q2q3-repack/EXPORT_RECIPE.md`). Main tree + build/ UNTOUCHED (no rebuild —
  the SHIPPED emitter is already PLAIN; the TILED variant lived only in /tmp scratch); nothing committed.
- **T8 UNTOUCHED**: this cell does not read or write the T8 win/loss gap ledger (`experiments/active/
  result-tables`); it is a standalone L1 tiling datapoint.

## durable files

- `tile_q3k_t3_findings.md`          — writeup: byte-exact gates (verify + identity), spill 978→894 (cliff
  NOT reached, v31, weight-bound), A/B +8.0%/+8.4% (nr64+nr16, overhead-shave not cliff), vs-opponent
  0.176→0.191, NULL/weight-bound verdict, the q4_K→q3_K (vs q6_K-shared-fold) transfer-boundary diagnosis,
  loadavg honesty, [NG-4] discipline.
- `tile_q3k_t3_ab_nr64.txt`          — raw board run, cold N=12, nr=64 (clean opponent anchor): preflight
  + objdump seal + hot-loop spill classification + silicon verify (both) + identity cmp + 12 A/B rounds + loadavg.
- `tile_q3k_t3_ab_nr16.txt`          — raw board A/B, cold N=10, nr=16 (shape-robust ours-side +8.4%).
- `tile_q3k_t3_ab.csv`               — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, spill, reload, maxVreg, vwmacc, textB).
- `objdump_tile_q3k_t3_seal.objdump` — UNTILED→TILED vsetvli/spill/reload/maxVreg/vwmacc/text seal at clang-17 -O2 AND -O3 + hot-loop spill classification + the q4_K→q3_K (q6_K-shared no-min fold) transfer-boundary mechanism.

## disposition (post-measurement ruling) — q3_K SHIPS PLAIN (already; confirmed on silicon)
Per **裁决一 [更简单者胜]** on this NULL: q3_K already ships the **plain, UNTILED** GEMM body
(`emitRepackKQuantGemmBodyQ3K` in `RVVToEmitCBlockQuantLinear.cpp`, byte-exact to the retired direct
emitter `emitRepackGemmQ3KQ8K` — the parent constructor shipped PLAIN on an analytical "weight-bound like
q6_K" assessment, NO tiled emitter was ever written). This cell's **TILED** binary was a /tmp scratch S6
transform of the plain export used ONLY to *test* whether a tile *would* have helped — the source tree was
NEVER modified, so there is nothing to revert; the shipped emitter stays PLAIN. The silicon result CONFIRMS
the ruling: the S6 lever does not reach the register cliff (spill 978→894 not ~0, v31 no cliff), and the
+8.0%/+8.4% overhead-shave it does buy is an order of magnitude short of a rescue and does not flip
vs-opponent (0.176×→0.191×) — so plain is the correct end state (adding the panel to the emitter would add
complexity for a weight-bound-limited +8%). Construction is INTACT: `lowerToRepackGem{v,m}KQuant` q3_K
no-min branch + core `decode_model "q3_K"` + schema six-state `constructed` (C_construct 39) + whitelist 11
+ direct-emitter deletion all UNCHANGED (**[F-EMIT] GREEN**). Parent owns the commit (this cell
git-add/commit-free).

## note — data-only cell
Data/evidence only. The load-bearing result is a **NULL (q3_K S6-scheme does NOT reach the register
cliff)**: the q4_K S6 register-cliff lever does not transfer to q3_K — staging the decode-scale strips to a
stack panel leaves q3_K's ~900-spill dual-plane fully-unrolled body largely untouched (spill 978→894,
maxVreg v31, no ≤32 cliff), buying only a modest +8.0%/+8.4% overhead-shave, vs-opponent stays a ~5.3× LOSS
(0.176→0.191) — all **byte-exactly** (verify INT_mismatch=0 both + identity cmp 0 mismatch; vwmacc 2176
unchanged). [NG-4] — L1 kernel datapoint, NOT a beat. Nothing here was committed / git add-ed; main tree +
build/ untouched (no rebuild — UNTILED is the working-tree read-only PLAIN export; TILED is a /tmp scratch
S6 transform). Logged as the q3_K T3 negative datapoint — with the q6_K T3 NULL and the q2_K/q4_K T3 WINs
it completes the tiling-lever transfer-boundary map (the `kquant_dmin_bsums_min` min-fold family
q4_K/q5_K/q2_K WINS vs the `kquant_single_scale_no_min` no-min family q6_K/q3_K NULL), ALONE.
