# cell MANIFEST — l1-t3-q6k-repack-gemm

- **campaign**: repack / [KQUANT-L1] (T3-tile — q6_K repack GEMM tiling-lever transfer test on
  rvv/VLEN128: does the q4_K S6 register-cliff lever, adapted no-min for q6_K — decoded signed-scale
  int16 STACK PANEL via vse16/vle16 + on-demand d f16 widen deferred to the end-of-block fold, NO
  min-fold panel since q6_K has no min — drive q6_K's fully-unrolled dual-plane (6-bit ql+qh, 16
  sub-blocks) body to the ≤32-vreg cliff, byte-exactly, and does throughput follow? Sibling of the
  q4_K WIN cell `l1-tile-s6-q4k-repack-gemm`. Logged ALONE.)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz, gov=performance)**.
  ★ RESULT = **NULL / small-regression — the lever does NOT transfer to q6_K.** (1) objdump at the
  measured clang-17.0.6 -O2 board form (campaign-canonical fixture + vs*r.v/vl*r.v seal): spill
  UNTILED **913 → TILED 949** (ROSE, not →~0), reload 953→957, maxVreg **v31 → v31** (the ≤32-vreg
  cliff was NEVER reached, no spare register) at BOTH -O2/-O3; vwmacc **2304 UNCHANGED** = byte-exact
  multiset. q6_K's fully-unrolled body carries ~900 whole-register spills (vs q4_K golden 84 / S1 23) —
  dominated by the DUAL-PLANE 6-bit weight reconstruction (ql@1312 + qh@288), NOT the decode-scale
  strips the panel stages, so staging the strips is a drop in the bucket + adds panel traffic ⇒ spill
  nudges UP. (2) byte-exact GREEN both ways: silicon verify UNTILED & TILED both INT_mismatch_total=0,
  NORM worst_ulp=220 worst_norm=4.579e-07 (identical); IDENTITY-DUMP o_untiled.bin == o_tiled.bin
  (32768 floats, 0 mismatch on the small-finite fp16-scale fold path). (3) A/B tiled/untiled paired cold
  (nr=64 N=12 / nr=16 N=10): **0.985 / 0.977 = a consistent −1.5%/−2.3% REGRESSION** (both sides cv
  0.15–0.35%, max A/B 0.992 < 1 → non-overlapping, SHAPE-ROBUST). (4) vs-opponent block-dot (clean
  anchor nr=64, opp cv 0.575% across 24 samples): UNTILED parity **0.184×** (reproduces the
  board-harvest2 q6_K prefill 0.18× LOSS → basis validated) → TILED **0.181×** (slightly WORSE); the
  opponent ggml_vec_dot_q6_K_q8_K block-dot is ~5.4–5.5× faster and tiling does not rescue it. The
  pre-registered gate (spill→~0 ∧ MAC/cycle up ∧ wall passes T-N) FAILS on all three ⇒ q6_K S6-scheme
  does NOT hold. The construction is sound + byte-exact; it just buys no register cliff on this format.
  This maps the q4_K S6 lever's transfer boundary (C3′ pattern-library: q4_K/q5_K-shaped, not
  q6_K-shaped) and **refutes on silicon** any "pending-hardware spill→0/≤32/win" projection for q6_K.
  **[NG-4] NOT a beat**: eight [PERF-1] gates unwalked; single core; opponent = single-thread block-dot
  proxy; our correctness = construction oracle (924dc31f) + the byte-exact verify + identity gates here.
- **role**: L1-maturity T3-tile A/B evidence. Isolates the q6_K tiling increment by measuring UNTILED
  (HEAD 924dc31f full-unroll direct-emit, == cached export / the kquant-l1-q6q2q3-repack cell's q6_K
  GEMM, md5 0f14791e) vs TILED (working-tree S6-scheme no-min stack-panel emitter delta on
  `RVVToEmitCBlockQuantLinear.cpp` `emitRepackKQuantGemmBodyQ6K`, md5 28cbeebd, regenerated via
  read-only `build/bin/tcrv-opt`) as separate board binaries sharing the same extern-C symbol, against
  the opponent's REAL linked block-dot, with a silicon verify + a full-output IDENTITY cmp proving the
  timed TILED is byte-exact to UNTILED/oracle. Harness/driver live under `tools/e2e-harness/board/`
  (`kquant_gemm_tile_q6k_t3_ab.sh` + `kquant_gemm_paired_q6k_driver.c` + reused `kquant_repack_verify_q6K.c`).
  Data/evidence only; the exported .c are NOT stored (deterministically regenerable — see the objdump
  seal + `experiments/active/kquant-l1-q6q2q3-repack/EXPORT_RECIPE.md`; the TILED tile is the
  working-tree emitter delta, uncommitted). Main tree + build/ UNTOUCHED (no rebuild); nothing committed.

## durable files

- `tile_q6k_t3_findings.md`          — writeup: byte-exact gates (verify + identity), spill 913→949
  (cliff NOT reached, v31), A/B −1.5%/−2.3% (nr64+nr16), vs-opponent 0.184→0.181, NULL verdict, the
  q4_K→q6_K transfer-boundary diagnosis, loadavg honesty, [NG-4] discipline.
- `tile_q6k_t3_ab_nr64.txt`          — raw board run, cold N=12, nr=64 (clean opponent anchor): preflight
  + objdump seal + hot-loop spill classification + silicon verify (both) + identity cmp + 12 A/B rounds + loadavg.
- `tile_q6k_t3_ab_nr16.txt`          — raw board A/B, cold N=10, nr=16 (shape-robust ours-side −2.3%; opponent noise-dominated).
- `tile_q6k_t3_ab.csv`               — parsed summary (nr, variant, ours/opp GMAC/s, cv, parity, A/B, vsetvli, spill, reload, maxVreg, vwmacc, textB).
- `objdump_tile_q6k_t3_seal.objdump` — UNTILED→TILED vsetvli/spill/reload/maxVreg/vwmacc/text seal at clang-17 -O2 AND -O3 + hot-loop spill classification + the q4_K→q6_K transfer-boundary mechanism.

## disposition (post-measurement ruling) — q6_K SHIPS CONSTRUCTION-ONLY
Per **裁决一 [更简单者胜]** on this NULL: the q6_K GEMM S6 tile is **REVERTED** — `emitRepackKQuantGemmBodyQ6K`
in `RVVToEmitCBlockQuantLinear.cpp` now emits the **plain, UNTILED** body, **byte-exact to the retired
direct emitter `emitRepackGemmQ6KQ8K`** (the ~913-spill board baseline). Concretely the four S6 deltas
were undone: the decoded-signed-scale int16 stack-panel (vse16/vle16) is gone, `dF32` is widened at the
block top again (not on-demand at the fold), and the per-block `scaleVal` SmallVector is restored (no
`scaleRe` panel reload). The GEVM sibling was **already plain** (never tiled). A source-level diff of the
reverted `emitRepackKQuantGemmBodyQ6K` vs HEAD's `emitRepackGemmQ6KQ8K` shows the ONLY differences are the
front-door construction (parametrized signature, `notifyMatchFailure(loc,…)`, result-less seed removal) —
the KERNEL body is byte-identical. So **q6_K = front-door CONSTRUCTION-ONLY** (no output tiling on this
format). Construction is INTACT: `lowerToRepackGem{v,m}KQuant` q6_K no-min branch + schema six-state
`constructed` + whitelist 13 + direct-emitter deletion all UNCHANGED (**[F-EMIT] GREEN**, C_construct 37).
Zero regression: forced-clean rebuild EXIT=0, RVV lit **517/517** (Conversion 206 + Dialect/Target 311),
all four q6_K conversion tests GREEN. NOTE: the working-tree emitter, which produced this cell's **TILED**
binary during the A/B above, has since been reverted — it now re-emits the **UNTILED** form, so the
historical A/B remains a faithful record of the tiled-vs-untiled comparison that motivated shipping plain.
Parent will commit (this cell git-add/commit-free).

## note — data-only cell
Data/evidence only. The load-bearing result is a **NULL (q6_K S6-scheme does NOT hold)**: the q4_K S6
register-cliff lever does not transfer to q6_K — staging the decode-scale strips to a stack panel leaves
q6_K's ~900-spill dual-plane fully-unrolled body untouched (spill 913→949, maxVreg v31, no ≤32 cliff),
throughput regresses ~1.5–2.3%, vs-opponent stays a ~5.4× LOSS (0.184→0.181) — all **byte-exactly**
(verify INT_mismatch=0 both + identity cmp 0 mismatch; vwmacc 2304 unchanged). [NG-4] — L1 kernel
datapoint, NOT a beat. Nothing here was committed / git add-ed; main tree + build/ untouched (no rebuild —
UNTILED is the cached HEAD export; TILED is the working-tree emitter export via read-only `tcrv-opt`).
Logged as the q6_K T3 negative datapoint (tiling-lever transfer boundary), ALONE.
