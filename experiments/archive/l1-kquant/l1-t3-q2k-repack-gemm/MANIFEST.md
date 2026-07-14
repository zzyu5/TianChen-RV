# cell MANIFEST — l1-t3-q2k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (T3-tile fmt2 — q2_K repack GEMM tiling-lever transfer test on
  rvv/VLEN128: does the q4_K S6 register-cliff lever transfer to q2_K, which SHARES the q4_K min fold
  `kquant_dmin_bsums_min` (dual d/dmin + bsums-min, 16 sub-blocks)? ★register-cliff prediction: q2_K
  should eat the cliff (like q4_K), NOT be weight-bound (unlike q6_K). Sibling of the q4_K WIN cell
  `l1-tile-s6-q4k-repack-gemm` and the q6_K NULL cell `l1-t3-q6k-repack-gemm`. Logged ALONE.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ RESULT = **HOLDS / decisive WIN — the lever DOES transfer to q2_K (mirror image of q6_K).** (1)
  objdump at the measured clang-17.0.6 -O2 board form (campaign-canonical fixture
  `rvv-to-emitc-repack-gemm-q2-K-q8-K.mlir` + vs*r.v seal, NOT driver-export): spill UNTILED **619 →
  TILED 7** (−98.9%, collapsed to ~0), reload 747→12, maxVreg **v31 → v30** (the ≤32-vreg register cliff
  was REACHED — same v31→v30 step q4_K S6 hit) at BOTH -O2/-O3 (603→7 at -O3); vwmacc **2304 UNCHANGED**
  = byte-exact multiset; textB HALVED 51014→26968. q2_K's ~600-spill full-unroll body is q6_K-magnitude
  in COUNT but q4_K-shaped in KIND — the pressure is the decode-4bit-scale/min strips + the idle
  i32 MIN (bsums) accumulator, EXACTLY what the S6 int16/i32 stack-panels stage — so staging them
  collapses the whole spill set (the 7 residual vs*r.v are prologue/epilogue-only, benign). (2) byte-exact
  GREEN both ways: silicon verify UNTILED & TILED both INT_mismatch_total=0, NORM worst_ulp=3402
  worst_norm=6.618e-07 (identical — the higher ULP vs q6_K's 220 is the DUAL fp16 d/dmin fold on the
  adversarial-NORM path only, INT byte-exact); IDENTITY-DUMP o_untiled.bin == o_tiled.bin (nr64 32768 /
  nr16 8192 floats, 0 mismatch). (3) A/B tiled/untiled paired cold (nr=64 N=12 / nr=16 N=10): **6.653× /
  5.648× — a +565%/+465% WIN** (ours cv 0.12–1.40%, per-round A/B min 5.525 > 1 → non-overlapping,
  SHAPE-ROBUST). (4) vs-opponent block-dot (clean anchor nr=64, opp cv 1.14% across 24 samples): UNTILED
  parity **0.212×** (reproduces the board-harvest2 q2_K prefill ~0.20× LOSS → basis validated) → TILED
  **1.413× WIN** (our tiled repack GEMM 1.41× FASTER than ggml_vec_dot_q2_K_q8_K block-dot). The
  pre-registered gate (spill→~0 ∧ MAC/cycle up ∧ wall passes T-N) PASSES on all three ⇒ q2_K S6-scheme
  HOLDS. **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single core; opponent = single-thread
  block-dot proxy; our correctness = construction oracle + the byte-exact verify + identity gates here.
- **role**: L1-maturity T3-tile A/B evidence. Isolates the q2_K tiling increment by measuring UNTILED
  (HEAD c3cf7301 full-unroll direct-emit, == the 924dc31f cached export / the kquant-l1-q6q2q3-repack
  cell's q2_K GEMM, md5 6c9322f6; provenance-clean — no q2_K emitter change 924dc31f..c3cf7301) vs TILED
  (working-tree S6-scheme dmin+bsums-min stack-panel emitter delta on `RVVToEmitCBlockQuantLinear.cpp`
  `emitRepackKQuantGemmBodyQ2K`, md5 9c5bac28, regenerated via read-only `build/bin/tcrv-opt` on the
  campaign-canonical front-door region fixture) as separate board binaries sharing the same extern-C
  symbol, against the opponent's REAL linked block-dot, with a silicon verify + a full-output IDENTITY cmp
  proving the timed TILED is byte-exact to UNTILED/oracle. Harness/driver live under
  `tools/e2e-harness/board/` (`kquant_gemm_tile_q2k_t3_ab.sh` + `kquant_gemm_paired_q2k_driver.c` +
  reused `kquant_repack_verify_q2K.c`). Data/evidence only; the exported .c are NOT stored
  (deterministically regenerable — see the objdump seal + `experiments/active/kquant-l1-q6q2q3-repack/
  EXPORT_RECIPE.md`; the TILED tile is the working-tree emitter delta, uncommitted). Main tree + build/
  UNTOUCHED (no rebuild); nothing committed.

## durable files

- `tile_q2k_t3_findings.md`          — writeup: byte-exact gates (verify + identity), spill 619→7 (cliff
  REACHED, v30), A/B 6.65×/5.65× (nr64+nr16), vs-opponent 0.21→1.41 flip, HOLDS/WIN verdict, the
  q4_K→q2_K vs q4_K→q6_K transfer-boundary diagnosis, loadavg honesty, [NG-4] discipline.
- `tile_q2k_t3_ab_nr64.txt`          — raw board run, cold N=12, nr=64 (clean opponent anchor): preflight
  + objdump seal + hot-loop spill classification + silicon verify (both) + identity cmp + 12 A/B rounds + loadavg.
- `tile_q2k_t3_ab_nr16.txt`          — raw board A/B, cold N=10, nr=16 (shape-robust ours-side 5.65×; opponent noise-dominated).
- `tile_q2k_t3_ab.csv`               — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, vsetvli, spill, reload, maxVreg, vwmacc, textB).
- `objdump_tile_q2k_t3_seal.objdump` — UNTILED→TILED vsetvli/spill/reload/maxVreg/vwmacc/text seal at clang-17 -O2 AND -O3 + hot-loop spill classification + the q4_K→q2_K (vs q6_K) transfer-boundary mechanism.

## disposition (post-measurement ruling) — q2_K SHIPS TILED (S6)
Per **裁决一 [更简单者胜]** applied to a decisive WIN: the S6 tile is BYTE-EXACT to the untiled body
(verify INT_mismatch=0 both + identity 0 mismatch + vwmacc 2304 unchanged) AND 5.6–6.7× faster AND flips
vs-opponent 0.21×→1.41×, so the tiled emitter is the shippable form — no revert (unlike q6_K, which shipped
construction-only). The working-tree `emitRepackKQuantGemmBodyQ2K` S6 delta is the correct end state. The
GEVM sibling is plain (never tiled; its untiled-vs-tiled code body is byte-identical modulo the front-door
provenance relabel + one removed result-less zero-seed). Construction is INTACT: `lowerToRepackGem{v,m}KQuant`
q2_K plain branch + core decode_model "q2_K" + schema six-state `constructed` (C_construct 37→38) + whitelist
13→12 + direct-emitter deletion. Parent will commit (this cell git-add/commit-free).

## note — data-only cell
Data/evidence only. The load-bearing result is a **decisive HOLDS/WIN (q2_K S6-scheme transfers the q4_K
register-cliff lever)**: staging the decode-scale/min strips + the idle MIN accumulator to stack panels
collapses q2_K's ~600-spill full-unroll body (spill 619→7, maxVreg v30, ≤32 cliff reached, text halved),
throughput rises 6.65×/5.65×, vs-opponent flips 0.21× LOSS → 1.41× WIN — all **byte-exactly** (verify
INT_mismatch=0 both + identity cmp 0 mismatch; vwmacc 2304 unchanged). [NG-4] — L1 kernel datapoint, NOT a
beat. Nothing here was committed / git add-ed; main tree + build/ untouched (no rebuild — UNTILED is the
cached HEAD export; TILED is the working-tree emitter export via read-only `tcrv-opt`). Logged as the q2_K
T3 POSITIVE datapoint — with the q6_K T3 NULL it delineates the tiling-lever transfer boundary (the
`kquant_dmin_bsums_min` min-fold family q4_K/q5_K/q2_K vs the dual-plane-weight q6_K), ALONE.
